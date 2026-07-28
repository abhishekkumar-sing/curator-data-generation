"""Generate, verify, judge, split, and export grounded procurement data."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset

from settings import CONFIG, PROJECT_ROOT, require_private_endpoint, require_setting
from corpus import load_corpus
from cross_document import build_bundles
from cross_stage import CrossDocumentGenerator, CrossDocumentJudge, cross_judge_rows
from export import assign_splits, export_records
from schemas import CandidateBatch, JudgeBatch
from validation import deduplicate, validate_record

# settings enforces local-only mode before Curator is imported.
from bespokelabs import curator

PATHS = CONFIG["paths"]
GENERATION = CONFIG["models"]["generation"]
QUALITY = CONFIG.get("quality", {})
SPLITS = CONFIG.get("splits", {})


def _model_settings(profile: dict[str, Any]) -> tuple[str, str, str]:
    model = require_setting(profile["served_model_env"])
    base_url = (
        require_private_endpoint(profile["base_url_env"])
        if profile.get("private_endpoint_only", True)
        else require_setting(profile["base_url_env"])
    )
    api_key = require_setting(profile["api_key_env"])
    return model, base_url, api_key


def _llm_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
    model, base_url, api_key = _model_settings(profile)
    return {
        "model_name": f"hosted_vllm/{model}",
        "backend": "litellm",
        "generation_params": profile["generation_params"],
        "backend_params": {
            "base_url": base_url,
            "api_key": api_key,
            "request_timeout": profile["request_timeout"],
            "max_concurrent_requests": profile["max_concurrent_requests"],
            "require_all_responses": False,
        },
    }


class ProcurementGenerator(curator.LLM):
    """Generate explicit QA or concise, auditable QA-with-rationale records."""

    response_format = CandidateBatch

    def prompt(self, row: dict) -> str:
        """Render a grounded single-document generation request."""
        return f"""Generate up to {QUALITY.get("examples_per_chunk", 3)} diverse training records.

Allowed question types: direct_fact, definition, authority, threshold,
conditional_rule, exception, procedure, scenario, multi_section, temporal,
unanswerable.

Rules:
- The question must stand alone and identify the relevant organization or manual.
- Use only the passage. Do not convert Government guidance into NRL policy.
- Preserve dates, thresholds, modality, conditions, exceptions, and amendments.
- For answerable records, every evidence quote must be copied exactly from the passage.
- Use task_type qa for direct questions and provide no reasoning_steps.
- Use qa_cot only for genuinely multi-step scenario, temporal, conditional, exception,
  procedure, or multi_section questions. Provide 2-4 short, auditable reasoning steps.
- Reasoning steps are an evidence-based teaching rationale, not hidden/private model thoughts.
- Each reasoning step must cite one or more exact evidence_quotes.
- Include some unanswerable questions only when the passage lacks the answer. Their exact
  answer must be: "Not answerable from the provided sources."

Source authority:
manual_id: {row["manual_id"]}
title: {row["title"]}
issuer: {row["issuing_organization"]}
policy_scope: {row["policy_scope"]}
revision_date: {row["revision_date"]}
as_of_date: {row["as_of_date"]}
page: {row["page"]}
section: {row["section"]}

Passage:
{row["passage"]}
"""

    def parse(self, row: dict, response: CandidateBatch) -> list[dict]:
        """Verify drafts and attach stable source provenance."""
        records = []
        for candidate in response.examples:
            draft = candidate.model_dump()
            reasons = validate_record(draft, row["passage"])
            if reasons:
                continue
            evidence = []
            for item in draft["evidence"]:
                quote = item["quote"]
                start = row["passage"].find(quote)
                evidence.append(
                    {
                        "quote": quote,
                        "chunk_id": row["chunk_id"],
                        "page": row["page"],
                        "section": row["section"],
                        "start_char": start,
                        "end_char": start + len(quote),
                    }
                )
            identity = json.dumps(
                [row["chunk_id"], draft["task_type"], draft["question"]],
                ensure_ascii=False,
            )
            record_id = "nrlqa-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            records.append(
                {
                    "record_id": record_id,
                    **draft,
                    "evidence": evidence,
                    "manual_id": row["manual_id"],
                    "manual_title": row["title"],
                    "issuing_organization": row["issuing_organization"],
                    "policy_scope": row["policy_scope"],
                    "revision_date": row["revision_date"],
                    "as_of_date": row["as_of_date"],
                    "source_file": row["source_file"],
                    "source_sha256": row["source_sha256"],
                    "source_chunk_ids": [row["chunk_id"]],
                    "_source_passage": row["passage"],
                    "generation_model": self.model_name,
                    "deterministic_checks": {"passed": True, "issues": []},
                }
            )
        return records


class ProcurementJudge(curator.LLM):
    """Apply a separate rubric after deterministic validation."""

    response_format = JudgeBatch

    def prompt(self, row: dict) -> str:
        """Render the deterministic-survivor quality review batch."""
        return f"""Judge every record strictly against its supplied evidence.

Check factual support, answer relevance, preservation of conditions/exceptions,
correct issuing authority, and validity/necessity of each rationale step. A QA
record with no rationale is reasoning_valid=true. An unanswerable record is
supported only if the evidence truly does not answer it. Scores 4-5 are accepted.
Return one judgment for every record_id and do not rewrite records.

Records:
{json.dumps([item["review"] for item in row["judge_items"]], ensure_ascii=False)}
"""

    def parse(self, row: dict, response: JudgeBatch) -> list[dict]:
        """Attach judge decisions and enforce the configured threshold."""
        original = {item["record_id"]: item["record"] for item in row["judge_items"]}
        results = []
        for judgment in response.judgments:
            record = original.get(judgment.record_id)
            if record is None:
                continue
            decision = judgment.decision.model_dump()
            record["judge"] = {
                **decision,
                "model": self.model_name,
                "accepted": all(
                    decision[field]
                    for field in (
                        "supported",
                        "relevant",
                        "preserves_qualifications",
                        "authority_correct",
                        "reasoning_valid",
                    )
                )
                and decision["score"] >= int(QUALITY.get("minimum_judge_score", 4)),
            }
            results.append(record)
        return results


def _judge_rows(records: list[dict[str, Any]], batch_size: int) -> Dataset:
    rows = []
    for start in range(0, len(records), batch_size):
        items = []
        for record in records[start : start + batch_size]:
            compact = {
                "record_id": record["record_id"],
                "question": record["question"],
                "answer": record["answer"],
                "answerable": record["answerable"],
                "task_type": record["task_type"],
                "reasoning_steps": record["reasoning_steps"],
                "issuer": record["issuing_organization"],
                "policy_scope": record["policy_scope"],
                "as_of_date": record["as_of_date"],
                "evidence": record["evidence"],
                "source_passage": record["_source_passage"],
            }
            items.append({"record_id": record["record_id"], "record": record, "review": compact})
        rows.append({"judge_items": items})
    return Dataset.from_list(rows)


def main() -> None:
    """Run single- and cross-document generation through verified exports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / PATHS["source_dir"])
    parser.add_argument("--ocr-dir", type=Path, default=PROJECT_ROOT / PATHS["ocr_dir"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / PATHS["output_dir"])
    parser.add_argument("--limit", type=int, help="Limit corpus chunks for a pilot")
    parser.add_argument(
        "--cross-document-limit",
        type=int,
        help="Limit cross-document source bundles (defaults to --limit for pilots)",
    )
    parser.add_argument("--skip-cross-document", action="store_true")
    parser.add_argument("--skip-judge", action="store_true", help="Development only")
    args = parser.parse_args()

    all_rows, manuals = load_corpus(args.source_dir.resolve(), args.ocr_dir.resolve())
    rows = all_rows
    if args.limit is not None:
        rows = rows[: args.limit]
    os.environ["HOSTED_VLLM_API_KEY"] = _model_settings(GENERATION)[2]
    generated = ProcurementGenerator(**_llm_kwargs(GENERATION))(
        Dataset.from_list(rows), working_dir=str(args.output_dir / ".cache" / "generation")
    ).dataset.to_list()
    generated, duplicates = deduplicate(
        generated, float(QUALITY.get("dedupe_threshold", 94))
    )
    if not generated:
        raise SystemExit("No records passed deterministic validation")

    if args.skip_judge:
        if not QUALITY.get("allow_unjudged_exports", False):
            raise SystemExit(
                "--skip-judge is disabled by config; set quality.allow_unjudged_exports=true "
                "only for development"
            )
        accepted = generated
    else:
        judge_profile = CONFIG["models"].get("judge", GENERATION)
        os.environ["HOSTED_VLLM_API_KEY"] = _model_settings(judge_profile)[2]
        judged = ProcurementJudge(**_llm_kwargs(judge_profile))(
            _judge_rows(generated, int(QUALITY.get("judge_batch_size", 8))),
            working_dir=str(args.output_dir / ".cache" / "judge"),
        ).dataset.to_list()
        accepted = [row for row in judged if row["judge"]["accepted"]]

    cross_accepted: list[dict[str, Any]] = []
    cross_duplicates = 0
    cross_config = CONFIG.get("cross_document", {})
    if cross_config.get("enabled", False) and not args.skip_cross_document:
        bundles = build_bundles(all_rows, cross_config)
        cross_limit = (
            args.cross_document_limit
            if args.cross_document_limit is not None
            else args.limit
        )
        if cross_limit is not None:
            bundles = bundles[:cross_limit]
        if bundles:
            os.environ["HOSTED_VLLM_API_KEY"] = _model_settings(GENERATION)[2]
            cross_generated = CrossDocumentGenerator(**_llm_kwargs(GENERATION))(
                Dataset.from_list(bundles),
                working_dir=str(args.output_dir / ".cache" / "cross_generation"),
            ).dataset.to_list()
            cross_generated, cross_duplicates = deduplicate(
                cross_generated, float(QUALITY.get("dedupe_threshold", 94))
            )
            if args.skip_judge:
                cross_accepted = cross_generated
            elif cross_generated:
                judge_profile = CONFIG["models"].get("judge", GENERATION)
                os.environ["HOSTED_VLLM_API_KEY"] = _model_settings(judge_profile)[2]
                cross_judged = CrossDocumentJudge(**_llm_kwargs(judge_profile))(
                    Dataset.from_list(
                        cross_judge_rows(
                            cross_generated, int(QUALITY.get("judge_batch_size", 8))
                        )
                    ),
                    working_dir=str(args.output_dir / ".cache" / "cross_judge"),
                ).dataset.to_list()
                cross_accepted = [
                    row for row in cross_judged if row["judge"]["accepted"]
                ]

    accepted.extend(cross_accepted)
    if not accepted:
        raise SystemExit("No records passed the quality judge")
    for record in accepted:
        record.pop("_source_passage", None)
    train_fraction = float(SPLITS.get("train", 0.8))
    validation_fraction = float(SPLITS.get("validation", 0.1))
    test_fraction = float(SPLITS.get("test", 0.1))
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise SystemExit("splits.train + splits.validation + splits.test must equal 1")
    assign_splits(
        accepted,
        manuals,
        train_fraction,
        validation_fraction,
        str(SPLITS.get("seed", "nrl-procurement-v1")),
    )
    stats = export_records(accepted, manuals, args.output_dir.resolve())
    print(
        f"Exported {stats['records']} accepted records to {args.output_dir.resolve()} "
        f"({duplicates + cross_duplicates} near-duplicates removed; "
        f"{len(cross_accepted)} cross-document records)"
    )


if __name__ == "__main__":
    main()
