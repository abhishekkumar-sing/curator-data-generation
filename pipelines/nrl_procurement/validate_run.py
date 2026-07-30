"""Fail-closed validation report for a completed procurement pipeline run."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonl_io import write_jsonl_rows

REQUIRED_EXPORTS = {
    "qa": "qa_sft.jsonl",
    "qa_cot": "qa_cot_sft.jsonl",
    "cross_document_qa": "cross_document_qa_sft.jsonl",
    "cross_document_qa_cot": "cross_document_qa_cot_sft.jsonl",
}
FAILURE_PATTERNS = {
    "invalid_enum": re.compile(r"literal_error|input should be .+ or ", re.I | re.S),
    "missing_field": re.compile(r"field required|type=missing", re.I),
    "absent_tool_call": re.compile(r"tool call.+(?:received 0|not found)", re.I),
    "malformed_array": re.compile(r"valid array|list_type|input_type=(?:str|list)", re.I),
    "timeout_or_connection": re.compile(
        r"timeout|connection refused|connecterror", re.I
    ),
}


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _failure_distribution(
    working_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Count failures for the latest manifest attempt, not historical caches."""
    counts: Counter[str] = Counter()
    if not working_dir.is_dir():
        return {}
    roots: list[Path] = []
    stage_events = (manifest or {}).get("resume", {}).get("stage_events", {})
    for stage, event in stage_events.items():
        if event.get("status") not in {"executed", "resumed_partial_cache"}:
            continue
        fingerprint = event.get("producer", {}).get("stage_fingerprint")
        if fingerprint:
            root = working_dir / stage / fingerprint
            if root.is_dir():
                roots.append(root)
    paths = (
        [path for root in roots for path in root.rglob("failed_requests.jsonl")]
        if roots
        else list(working_dir.rglob("failed_requests.jsonl"))
    )
    for path in paths:
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if not line.strip():
                continue
            matched = [
                name
                for name, pattern in FAILURE_PATTERNS.items()
                if pattern.search(line)
            ]
            counts[matched[0] if matched else "other"] += 1
    return dict(sorted(counts.items()))


def validate_run(
    files_dir: Path,
    working_dir: Path | None = None,
    reviews: Path | None = None,
) -> dict[str, Any]:
    """Reconcile terminal lineage, exports, audits, failures, and human gates."""
    issues = []
    manifest_path = files_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    if not manifest:
        issues.append("missing_terminal_manifest")
    elif manifest.get("status") != "complete":
        issues.append(f"manifest_status:{manifest.get('status', 'missing')}")
    terminal = manifest.get("terminal_request_completeness", {})
    if not terminal.get("complete", False):
        issues.append("incomplete_request_lineage")

    export_counts = {
        task: _jsonl_count(files_dir / filename)
        for task, filename in REQUIRED_EXPORTS.items()
    }
    for task, count in export_counts.items():
        if count < 1:
            issues.append(f"empty_required_export:{task}")
    canonical_count = _jsonl_count(files_dir / "canonical.jsonl")
    if canonical_count != sum(export_counts.values()):
        issues.append("canonical_and_task_export_counts_do_not_reconcile")

    leakage_path = files_dir / "leakage_audit.json"
    leakage = (
        json.loads(leakage_path.read_text(encoding="utf-8"))
        if leakage_path.is_file()
        else {}
    )
    if not leakage.get("passed", False):
        issues.append("leakage_audit_missing_or_failed")

    review_counts = {"reviewed_accepted": 0, "reviewed_rejected": 0}
    if reviews and reviews.is_file():
        from review import validate_reviews

        review_result = validate_reviews(reviews)
        review_counts = {
            key: review_result[key]
            for key in ("reviewed_accepted", "reviewed_rejected")
        }
        if not review_result["passed"]:
            issues.append("human_review_incomplete_or_invalid")
    else:
        issues.append("human_review_not_supplied")

    failures = (
        _failure_distribution(working_dir, manifest)
        if working_dir
        else {}
    )
    report = {
        "passed": not issues,
        "issues": sorted(set(issues)),
        "run_id": manifest.get("run_id"),
        "manifest_status": manifest.get("status"),
        "terminal_request_completeness": terminal,
        "export_counts": export_counts,
        "canonical_count": canonical_count,
        "leakage_audit_passed": leakage.get("passed", False),
        "post_retry_model_failure_distribution": failures,
        "human_review": {
            **review_counts,
            "minimum_accepted_required": 100,
        },
    }
    return report


def main() -> None:
    """Validate one completed run and return a release-oriented exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("files_dir", type=Path)
    parser.add_argument("--working-dir", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_run(args.files_dir, args.working_dir, args.reviews)
    if args.output:
        write_jsonl_rows(args.output, [report])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
