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


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _classify_failed_request_line(line: str) -> str:
    """Classify one `failed_requests.jsonl` row.

    Curator now tags every failed row with a structured ``error_category``
    (timeout/truncation/schema_validation/rate_limit/other/unknown) derived
    from the actual terminal exception, and that is always preferred. The
    regex heuristic below only remains as a fallback for `failed_requests.jsonl`
    files captured before that field existed: those older rows are the raw
    outgoing request payload (prompt text) with no error information in them
    at all, so the regex match was frequently just matching prompt wording and
    should not be trusted over a real structured category when one exists.
    """
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        row = None
    if isinstance(row, dict) and row.get("error_category"):
        return str(row["error_category"])
    matched = [name for name, pattern in FAILURE_PATTERNS.items() if pattern.search(line)]
    return matched[0] if matched else "other"


def _failure_distribution(
    working_dir: Path,
    manifest: dict[str, Any] | None = None,
    *,
    stage_names: set[str] | None = None,
) -> dict[str, int]:
    """Count failures for the latest manifest attempt, not historical caches.

    `stage_names`, when given, restricts counting to stages in that set
    (matched against `manifest.resume.stage_events` keys) instead of every
    `failed_requests.jsonl` under `working_dir`. Callers that need a failure
    count scoped to exactly the stages a particular denominator covers (e.g.
    `_schema_validity_rate`) pass this so the numerator and denominator stay
    stage-consistent; the unscoped, whole-run distribution used elsewhere in
    this report is unaffected (default `stage_names=None` preserves the
    original full-`working_dir` behavior, including its full-rglob fallback
    when no `stage_events` are recorded at all).
    """
    counts: Counter[str] = Counter()
    if not working_dir.is_dir():
        return {}
    roots: list[Path] = []
    stage_events = (manifest or {}).get("resume", {}).get("stage_events", {})
    for stage, event in stage_events.items():
        if stage_names is not None and stage not in stage_names:
            continue
        if event.get("status") not in {"executed", "resumed_partial_cache"}:
            continue
        fingerprint = event.get("producer", {}).get("stage_fingerprint")
        if fingerprint:
            root = working_dir / stage / fingerprint
            if root.is_dir():
                roots.append(root)
    if roots:
        paths = [path for root in roots for path in root.rglob("failed_requests.jsonl")]
    elif stage_names is None:
        paths = list(working_dir.rglob("failed_requests.jsonl"))
    else:
        # A stage-scoped caller must never silently widen to every stage in
        # `working_dir` just because none of its requested stages produced a
        # locatable root (e.g. those stages were skipped, or `stage_events`
        # is missing/stale) -- that would mix in failures from stages the
        # matching denominator does not cover.
        paths = []
    for path in paths:
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if not line.strip():
                continue
            counts[_classify_failed_request_line(line)] += 1
    return dict(sorted(counts.items()))


# LLM stages that already expose an unambiguous, non-duplicated
# `request_coverage()` denominator in today's manifest (see
# `_schema_validity_denominator`'s docstring for why the others don't).
# Keep in sync with the stage names `generate.py` passes as the first
# argument to `_execute_llm_stage` for these specific stages.
SCHEMA_VALIDITY_STAGE_NAMES = {
    "qa_blueprints",
    "generation",
    "judge",
    "cross_generation",
    "cross_judge",
}
_SCHEMA_VALIDITY_SCOPE_NOTE = (
    "Covers only qa_blueprints/generation/judge/cross_generation/cross_judge "
    "(the stages that already report a request_coverage() expected-request "
    "denominator in the manifest today; qa_blueprints shares generation's "
    "denominator, same planned cohort, and is not double-counted). "
    "propositions, temporal_alignment_judge, path_questions/path_answers/"
    "path_ablation_trials/path_ablation_judge, drafting_generation/"
    "drafting_judge, and unanswerable_generation/answerability_judge are not "
    "yet covered -- see audit-remediation.md task T31b-ii."
)


def _schema_validity_denominator(manifest: dict[str, Any]) -> dict[str, int]:
    """Non-overlapping expected-request denominators for schema-validity scope.

    `request_coverage()` tracks one planned-request cohort's survival through
    several checkpoints of the same stage family: within
    `request_coverage.single_document`, `blueprinted`, `generated`, and
    `accepted` are all computed from the exact same planned-request list and
    therefore always report an identical `expected_requests` (only
    `materialized_requests`/`missing_request_ids` differ across them) -- the
    same is true of `generated`/`accepted` within `request_coverage.cross_document`.
    Summing those checkpoints as though they were independent per-call
    denominators would multiply-count the same requests. `judged` coverage is
    genuinely distinct (a smaller, judge-eligible subset). This returns the
    four denominators that are non-overlapping in today's manifest;
    `qa_blueprints` is represented by (not added on top of) the generation
    cohort it shares a denominator with.
    """
    coverage = manifest.get("request_coverage", {})
    single = coverage.get("single_document", {})
    cross = coverage.get("cross_document", {})
    return {
        "single_document_generation": int(single.get("generated", {}).get("expected_requests", 0)),
        "single_document_judge": int(single.get("judged", {}).get("expected_requests", 0)),
        "cross_document_generation": int(cross.get("generated", {}).get("expected_requests", 0)),
        "cross_document_judge": int(cross.get("judged", {}).get("expected_requests", 0)),
    }


def _schema_validity_rate(
    working_dir: Path | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Report the share of scoped expected requests that did not fail schema validation.

    Deliberately scoped, not pipeline-wide -- see `_SCHEMA_VALIDITY_SCOPE_NOTE`.
    `computed` is `False` (rate left as `None`, not fabricated as 1.0) when
    either no `--working-dir` was supplied or the manifest has no
    `resume.stage_events` lineage to scope failure counts against, since a
    silently-full or silently-empty stage-events lookup would make the rate
    meaningless rather than merely incomplete.
    """
    denominators = _schema_validity_denominator(manifest)
    expected_total = sum(denominators.values())
    stage_events = manifest.get("resume", {}).get("stage_events", {})
    if working_dir is None:
        return {
            "computed": False,
            "reason": "no --working-dir supplied",
            "expected_requests_by_stage": denominators,
            "expected_requests_total": expected_total,
            "schema_validation_failures": None,
            "schema_validity_rate": None,
            "scope": _SCHEMA_VALIDITY_SCOPE_NOTE,
        }
    if not stage_events:
        return {
            "computed": False,
            "reason": "manifest has no resume.stage_events lineage to scope failures by stage",
            "expected_requests_by_stage": denominators,
            "expected_requests_total": expected_total,
            "schema_validation_failures": None,
            "schema_validity_rate": None,
            "scope": _SCHEMA_VALIDITY_SCOPE_NOTE,
        }
    schema_failures = _failure_distribution(
        working_dir, manifest, stage_names=SCHEMA_VALIDITY_STAGE_NAMES
    ).get("schema_validation", 0)
    return {
        "computed": True,
        "expected_requests_by_stage": denominators,
        "expected_requests_total": expected_total,
        "schema_validation_failures": schema_failures,
        "schema_validity_rate": (
            round(1 - (schema_failures / expected_total), 4) if expected_total else None
        ),
        "scope": _SCHEMA_VALIDITY_SCOPE_NOTE,
    }


def _question_opener_diversity_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Report post-cap and pre-cap opener concentration side by side.

    `manifest.statistics.question_opener_diversity` (post-cap) is computed on
    the final exported pool and, by construction, always looks healthy once
    `enforce_question_opener_diversity` has already removed the
    overrepresented records — it cannot show how concentrated generation
    actually was before that cap ran. `manifest.question_opener_diversity_pre_cap`
    (added for audit task T9) is computed on the raw generated pool before
    dedup or the cap, so a regression toward near-total template homogeneity
    is visible here even though the post-cap pool would never show it.
    Both default to an empty/zero shape when a manifest predates the field
    that produces them (older runs, or this run failed before that stage)
    rather than raising, since this is a descriptive report field, not a gate.
    """
    post_cap = manifest.get("statistics", {}).get("question_opener_diversity", {})
    pre_cap = manifest.get("question_opener_diversity_pre_cap", {})
    return {
        "post_cap": post_cap,
        "pre_cap": pre_cap,
    }


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

    export_rows = {
        task: _jsonl_rows(files_dir / filename)
        for task, filename in REQUIRED_EXPORTS.items()
    }
    export_counts = {task: len(rows) for task, rows in export_rows.items()}
    required_tasks = set(
        manifest.get("required_task_type_counts", REQUIRED_EXPORTS).keys()
    )
    for task in required_tasks:
        count = export_counts.get(task, 0)
        if count < 1:
            issues.append(f"empty_required_export:{task}")
    evaluation_rows = _jsonl_rows(files_dir / "eval.jsonl")
    training_rows = [row for rows in export_rows.values() for row in rows]
    if any(row.get("split") != "train" for row in training_rows):
        issues.append("non_train_record_in_training_export")
    if any(row.get("split") == "train" for row in evaluation_rows):
        issues.append("train_record_in_eval_export")
    training_ids = {str(row.get("record_id", "")) for row in training_rows}
    evaluation_ids = {str(row.get("record_id", "")) for row in evaluation_rows}
    if training_ids & evaluation_ids:
        issues.append("training_eval_record_id_overlap")
    canonical_count = _jsonl_count(files_dir / "canonical.jsonl")
    if canonical_count != len(training_rows) + len(evaluation_rows):
        issues.append("canonical_and_task_export_counts_do_not_reconcile")

    quality_acceptance = manifest.get("quality_acceptance", {})
    if not quality_acceptance.get("portfolio_quality_complete", False):
        issues.append("portfolio_quality_incomplete")
    stage_quality_evidence = manifest.get("stage_quality_evidence", {})
    for stage in ("cross_document", "drafting"):
        evidence = stage_quality_evidence.get(stage, {})
        if evidence.get("required", True) and not evidence.get("passed", False):
            issues.append(f"stage_quality_evidence_incomplete:{stage}")

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
        "eval_count": len(evaluation_rows),
        "training_eval_record_id_overlap": len(training_ids & evaluation_ids),
        "canonical_count": canonical_count,
        "stage_quality_evidence": stage_quality_evidence,
        "leakage_audit_passed": leakage.get("passed", False),
        "post_retry_model_failure_distribution": failures,
        "schema_validity": _schema_validity_rate(working_dir, manifest),
        "question_opener_diversity": _question_opener_diversity_summary(manifest),
        "human_review": {
            **review_counts,
            "minimum_accepted_required": 100,
        },
    }
    return report


def main(argv: list[str] | None = None) -> None:
    """Validate one completed run and return a release-oriented exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("files_dir", type=Path)
    parser.add_argument("--working-dir", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_run(args.files_dir, args.working_dir, args.reviews)
    if args.output:
        write_jsonl_rows(args.output, [report])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
