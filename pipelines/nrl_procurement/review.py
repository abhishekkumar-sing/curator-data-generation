"""Prepare and validate immutable human-review and frozen-evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonl_io import write_jsonl_rows

REVIEW_DIMENSIONS = (
    "factual_correctness",
    "answer_completeness",
    "source_attribution",
    "temporal_correctness",
    "qualification_preservation",
    "multi_source_necessity",
    "rationale_faithfulness",
    "question_naturalness",
    "training_usefulness",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_sample(
    rows: list[dict[str, Any]],
    count: int,
    seed: str,
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('record_id', '')}".encode()
        ).hexdigest(),
    )[:count]


def prepare_review(
    files_dir: Path,
    output: Path,
    *,
    accepted_count: int = 100,
    rejected_count: int = 25,
    seed: str = "nrl-human-review-v1",
) -> dict[str, int]:
    """Write a reproducible review template without inventing reviewer labels."""
    accepted = _read_jsonl(files_dir / "canonical.jsonl")
    rejected = [
        *(_read_jsonl(files_dir / "qa_rejected.jsonl")),
        *(_read_jsonl(files_dir / "cross_rejected.jsonl")),
        *(_read_jsonl(files_dir / "path_answers_rejected.jsonl")),
    ]
    rows = []
    for disposition, candidates, count in (
        ("accepted", accepted, accepted_count),
        ("rejected", rejected, rejected_count),
    ):
        for row in _stable_sample(candidates, count, f"{seed}:{disposition}"):
            record_id = str(
                row.get("record_id")
                or row.get("parent_request_id")
                or row.get("question_id")
            )
            rows.append(
                {
                    "review_id": "review-"
                    + hashlib.sha256(
                        f"{files_dir.name}:{disposition}:{record_id}".encode()
                    ).hexdigest()[:24],
                    "record_id": record_id,
                    "pipeline_disposition": disposition,
                    "record_sha256": hashlib.sha256(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "record": row,
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "dimensions": {name: None for name in REVIEW_DIMENSIONS},
                    "overall_accept": None,
                    "notes": "",
                }
            )
    write_jsonl_rows(output, rows)
    return {
        "accepted_available": len(accepted),
        "accepted_sampled": sum(
            row["pipeline_disposition"] == "accepted" for row in rows
        ),
        "rejected_available": len(rejected),
        "rejected_sampled": sum(
            row["pipeline_disposition"] == "rejected" for row in rows
        ),
    }


def validate_reviews(path: Path) -> dict[str, Any]:
    """Verify completeness and immutable record hashes for supplied labels."""
    rows = _read_jsonl(path)
    issues = []
    reviewed_accepted = 0
    reviewed_rejected = 0
    for row in rows:
        expected_hash = hashlib.sha256(
            json.dumps(
                row.get("record", {}),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if row.get("record_sha256") != expected_hash:
            issues.append(f"{row.get('review_id')}:record_hash_mismatch")
        dimensions = row.get("dimensions", {})
        if (
            not row.get("reviewer_id")
            or not row.get("reviewed_at")
            or row.get("overall_accept") is None
            or set(dimensions) != set(REVIEW_DIMENSIONS)
            or any(value not in {True, False} for value in dimensions.values())
        ):
            issues.append(f"{row.get('review_id')}:incomplete_review")
            continue
        disposition = row.get("pipeline_disposition")
        reviewed_accepted += disposition == "accepted"
        reviewed_rejected += disposition == "rejected"
    return {
        "passed": not issues and reviewed_accepted >= 100,
        "issues": sorted(set(issues)),
        "reviewed_accepted": reviewed_accepted,
        "reviewed_rejected": reviewed_rejected,
        "minimum_accepted_required": 100,
        "frozen_evaluation_complete": False,
        "frozen_evaluation_note": (
            "Reviewed generated training records are not an independent gold set."
        ),
    }


def main() -> None:
    """Run review-template preparation or validation."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("files_dir", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--accepted-count", type=int, default=100)
    prepare.add_argument("--rejected-count", type=int, default=25)
    prepare.add_argument("--seed", default="nrl-human-review-v1")
    validate = subparsers.add_parser("validate")
    validate.add_argument("reviews", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_review(
            args.files_dir,
            args.output,
            accepted_count=args.accepted_count,
            rejected_count=args.rejected_count,
            seed=args.seed,
        )
    else:
        result = validate_reviews(args.reviews)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
