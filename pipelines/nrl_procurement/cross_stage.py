"""Curator generation and source-ablation judging for cross-document QA."""

# ruff: noqa: I001

from __future__ import annotations

import hashlib
import json
from typing import Any

from cross_document import evidence_location
from schemas import CrossCandidateBatch, CrossJudgeBatch
from settings import CONFIG
from validation import validate_cross_record

from bespokelabs import curator

QUALITY = CONFIG.get("quality", {})


def _render_sources(row: dict[str, Any]) -> str:
    rendered = []
    for document in row["source_documents"]:
        rendered.append(
            f"""SOURCE ID: {document["source_id"]}
MANUAL ID: {document["manual_id"]}
TITLE: {document["title"]}
ISSUER: {document["issuing_organization"]}
POLICY SCOPE: {document["policy_scope"]}
REVISION: {document["revision_date"]}
AS OF: {document["as_of_date"]}
PAGE: {document["page"]}
SECTION: {document["section"]}

{document["passage"]}"""
        )
    return "\n\n--- NEXT SOURCE ---\n\n".join(rendered)


class CrossDocumentGenerator(curator.LLM):
    """Generate records whose complete answer requires both supplied documents."""

    response_format = CrossCandidateBatch

    def prompt(self, row: dict) -> str:
        """Render two explicitly related source passages."""
        count = CONFIG.get("cross_document", {}).get("examples_per_bundle", 2)
        return f"""Generate up to {count} genuinely cross-document procurement records.

RELATIONSHIP: {row["relationship_type"]}
PAIR: {row["pair_id"]}

Requirements:
- A complete answer must require BOTH source_a and source_b. If either can answer the
  entire question alone, do not generate that record.
- Produce a mix of cross_document_qa and cross_document_qa_cot when supported.
- QA records have no reasoning_steps. QA-with-CoT records contain 2-4 concise,
  evidence-backed teaching steps, not private hidden thoughts.
- Every material claim must cite exact source-specific quotations.
- Collectively, answerable claims must use both sources.
- A rationale must form a connected synthesis: extract from each source, then compare,
  combine, resolve authority/time, apply a condition, calculate, or conclude.
- Questions must be standalone and name the manuals, authorities, domains, or dates needed
  to disambiguate them.
- Preserve thresholds, modality, conditions, exceptions, revision dates, and policy scope.
- Similar language does not prove adoption, equivalence, precedence, or supersession.
- Missing text does not prove absence, deletion, or inapplicability.
- For same_authority_temporal, identify both dated states and make no unsupported claim
  about what is currently in force.
- Use cross-document unanswerable only for a missing required link or unsupported premise;
  the exact answer must be "Not answerable from the provided sources."

Candidate alignment terms are only retrieval hints, never facts:
{json.dumps(row["shared_terms"], ensure_ascii=False)}

{_render_sources(row)}
"""

    def parse(self, row: dict, response: CrossCandidateBatch) -> list[dict]:
        """Attach stable identity and source-specific evidence provenance."""
        results = []
        for candidate in response.examples:
            draft = candidate.model_dump()
            reasons = validate_cross_record(draft, row["source_documents"])
            if reasons:
                continue
            claims, flat_evidence = [], {}
            for index, claim in enumerate(draft.pop("claims"), 1):
                claim_id = f"claim-{index}"
                evidence = []
                for item in claim["evidence"]:
                    located = evidence_location(
                        row["source_documents"], item["source_id"], item["quote"]
                    )
                    if located is not None:
                        evidence.append(located)
                        flat_evidence[
                            (located["source_id"], located["chunk_id"], located["quote"])
                        ] = located
                claims.append(
                    {"claim_id": claim_id, "statement": claim["statement"], "evidence": evidence}
                )
            steps = []
            for index, step in enumerate(draft["reasoning_steps"], 1):
                evidence = [
                    located
                    for item in step["evidence"]
                    if (
                        located := evidence_location(
                            row["source_documents"], item["source_id"], item["quote"]
                        )
                    )
                    is not None
                ]
                for located in evidence:
                    flat_evidence[
                        (located["source_id"], located["chunk_id"], located["quote"])
                    ] = located
                steps.append(
                    {
                        "step": index,
                        "operation": step["operation"],
                        "statement": step["statement"],
                        "evidence": evidence,
                    }
                )
            draft["reasoning_steps"] = steps
            identity = f"{row['source_bundle_id']}:{draft['task_type']}:{draft['question']}"
            record_id = "nrlxd-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            results.append(
                {
                    "record_id": record_id,
                    **draft,
                    "claims": claims,
                    "evidence": list(flat_evidence.values()),
                    "relationship_type": row["relationship_type"],
                    "source_bundle_id": row["source_bundle_id"],
                    "pair_id": row["pair_id"],
                    "hop_count": 2,
                    "required_source_ids": ["source_a", "source_b"],
                    "source_documents": row["source_documents"],
                    "source_chunk_ids": [
                        document["chunk_id"] for document in row["source_documents"]
                    ],
                    "generation_model": self.model_name,
                    "deterministic_checks": {"passed": True, "issues": []},
                }
            )
        return results


class CrossDocumentJudge(curator.LLM):
    """Judge support and test necessity by removing each required source."""

    response_format = CrossJudgeBatch

    def prompt(self, row: dict) -> str:
        """Request full-context and counterfactual source-ablation judgments."""
        return f"""Evaluate each cross-document record using three contexts:
1. Both sources together: the answer and every claim must be fully supported.
2. Remove source_a: the complete answer must become unsupported or materially incomplete.
3. Remove source_b: the complete answer must become unsupported or materially incomplete.

For an answerable record, unsupported_without_source_ids must contain BOTH source_a and
source_b. This is the counterfactual source-ablation test. Also check relevance,
qualifications, authority, temporal scope, relationship correctness, and whether the
rationale is a connected evidence path. QA records without rationale have
reasoning_valid=true and connected_reasoning=true when their claims still synthesize both
sources. Scores 4-5 are accepted. Do not rewrite records.

Records:
{json.dumps([item["review"] for item in row["judge_items"]], ensure_ascii=False)}
"""

    def parse(self, row: dict, response: CrossJudgeBatch) -> list[dict]:
        """Accept only records that require every declared source."""
        originals = {item["record_id"]: item["record"] for item in row["judge_items"]}
        results = []
        for judgment in response.judgments:
            record = originals.get(judgment.record_id)
            if record is None:
                continue
            decision = judgment.decision.model_dump()
            ablation_passed = (
                set(decision["unsupported_without_source_ids"]) == {"source_a", "source_b"}
                if record["answerable"]
                else True
            )
            required = (
                "supported",
                "relevant",
                "preserves_qualifications",
                "authority_correct",
                "reasoning_valid",
                "full_context_supported",
                "connected_reasoning",
                "relationship_correct",
            )
            record["judge"] = {
                **decision,
                "source_ablation_passed": ablation_passed,
                "model": self.model_name,
                "accepted": all(decision[field] for field in required)
                and ablation_passed
                and decision["score"] >= int(QUALITY.get("minimum_judge_score", 4)),
            }
            results.append(record)
        return results


def cross_judge_rows(records: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    """Pack cross-document records into bounded judge requests."""
    rows = []
    for start in range(0, len(records), batch_size):
        items = []
        for record in records[start : start + batch_size]:
            review = {
                key: record[key]
                for key in (
                    "record_id",
                    "task_type",
                    "relationship_type",
                    "question",
                    "answer",
                    "answerable",
                    "claims",
                    "reasoning_steps",
                    "required_source_ids",
                    "source_documents",
                )
            }
            items.append({"record_id": record["record_id"], "record": record, "review": review})
        rows.append({"judge_items": items})
    return rows
