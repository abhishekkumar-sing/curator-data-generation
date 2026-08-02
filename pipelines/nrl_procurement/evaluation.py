"""Frozen external evaluation registry and deterministic regression reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jsonl_io import write_jsonl_rows
from settings import CONFIG, PROJECT_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def validate_manual_folds(
    manuals: list[dict[str, Any]],
    folds: dict[str, str],
) -> dict[str, str]:
    """Require every registered manual to occur in exactly one valid fold."""
    known = {str(manual["manual_id"]) for manual in manuals}
    normalized = {str(key): str(value) for key, value in folds.items()}
    missing = sorted(known - set(normalized))
    unknown = sorted(set(normalized) - known)
    invalid = sorted(
        manual_id
        for manual_id, fold in normalized.items()
        if fold not in {"train", "validation", "test"}
    )
    issues = []
    if missing:
        issues.append(f"missing manuals: {missing}")
    if unknown:
        issues.append(f"unknown manuals: {unknown}")
    if invalid:
        issues.append(f"invalid folds: {invalid}")
    if issues:
        raise ValueError("Invalid splits.manual_folds: " + "; ".join(issues))
    for manual in manuals:
        source = str(manual["manual_id"])
        for target in manual.get("amends", []):
            target = str(target)
            if target in normalized and normalized[source] != normalized[target]:
                raise ValueError(
                    "Amendment-connected manuals must share one fold: "
                    f"{source}={normalized[source]}, {target}={normalized[target]}"
                )
    return normalized


def load_frozen_evaluation(
    config: dict[str, Any],
    *,
    required: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a hash-pinned, human-reviewed external set without mutation."""
    frozen = (config.get("evaluation", {}) or {}).get("frozen_external", {}) or {}
    relative = str(frozen.get("path", "")).strip()
    expected_hash = str(frozen.get("sha256") or "").strip().casefold()
    if not relative or not expected_hash:
        if required:
            raise SystemExit(
                "Full run blocked: configure evaluation.frozen_external.path and "
                "its SHA-256 after independent human review"
            )
        return [], {"configured": False, "verified": False}
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Frozen evaluation path must remain inside the project") from exc
    if not path.is_file():
        raise SystemExit(f"Frozen external evaluation file not found: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise SystemExit(
            "Frozen external evaluation hash mismatch; review and deliberately "
            "update the configured pin"
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("record_id", "")) for row in rows]
    if not rows or "" in ids or len(ids) != len(set(ids)):
        raise ValueError("Frozen evaluation requires non-empty unique record_id values")
    not_reviewed = [
        record_id
        for record_id, row in zip(ids, rows, strict=True)
        if row.get("human_review", {}).get("status") != "approved"
    ]
    if not_reviewed:
        raise ValueError(
            "Frozen evaluation contains records without approved human review: "
            + ", ".join(not_reviewed)
        )
    return rows, {
        "configured": True,
        "verified": True,
        "path": relative,
        "sha256": actual_hash,
        "records": len(rows),
    }


def frozen_overlap_issues(
    generated: list[dict[str, Any]], frozen: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Detect identity or normalized-question overlap with external gold data."""
    frozen_ids = {str(row["record_id"]) for row in frozen}
    frozen_questions = {_normalized(str(row.get("question", ""))) for row in frozen}
    return {
        "record_ids": sorted(
            str(row["record_id"])
            for row in generated
            if str(row["record_id"]) in frozen_ids
        ),
        "normalized_questions": sorted(
            str(row["record_id"])
            for row in generated
            if _normalized(str(row.get("question", ""))) in frozen_questions
        ),
    }


def regression_report(
    frozen: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score answerability, normalized exact answers, and citation-chunk recall."""
    predicted = {str(row.get("record_id", "")): row for row in predictions}
    details = []
    for reference in frozen:
        record_id = str(reference["record_id"])
        candidate = predicted.get(record_id, {})
        expected_answerable = bool(reference.get("answerable", True))
        predicted_answerable = candidate.get("answerable")
        answerability_correct = predicted_answerable is expected_answerable
        answer_exact = _normalized(str(candidate.get("answer", ""))) == _normalized(
            str(reference.get("answer", ""))
        )
        expected_chunks = set(reference.get("source_chunk_ids", []))
        predicted_chunks = set(candidate.get("source_chunk_ids", []))
        citation_recall = (
            len(expected_chunks & predicted_chunks) / len(expected_chunks)
            if expected_chunks
            else 1.0
        )
        details.append(
            {
                "record_id": record_id,
                "prediction_present": bool(candidate),
                "answerability_correct": answerability_correct,
                "answer_exact": answer_exact,
                "citation_chunk_recall": round(citation_recall, 6),
            }
        )
    total = len(details)
    metrics = {
        "records": total,
        "prediction_coverage": round(
            sum(item["prediction_present"] for item in details) / total, 6
        ),
        "answerability_accuracy": round(
            sum(item["answerability_correct"] for item in details) / total, 6
        ),
        "normalized_exact_answer_accuracy": round(
            sum(item["answer_exact"] for item in details) / total, 6
        ),
        "mean_citation_chunk_recall": round(
            sum(item["citation_chunk_recall"] for item in details) / total, 6
        ),
    }
    baseline_metrics = (baseline or {}).get("metrics", {})
    deltas = {
        key: round(value - float(baseline_metrics[key]), 6)
        for key, value in metrics.items()
        if key != "records" and key in baseline_metrics
    }
    return {"metrics": metrics, "baseline_deltas": deltas, "records": details}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> None:
    """Write a deterministic regression report for frozen external records."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    frozen, registry = load_frozen_evaluation(CONFIG, required=True)
    baseline_path = (
        PROJECT_ROOT / str(CONFIG["evaluation"].get("baseline_path", ""))
    ).resolve()
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else None
    )
    report = {
        "frozen_registry": registry,
        **regression_report(frozen, _read_jsonl(args.predictions), baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)


__all__ = [
    "frozen_overlap_issues",
    "load_frozen_evaluation",
    "regression_report",
    "validate_manual_folds",
    "write_jsonl_rows",
]
