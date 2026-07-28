"""Deterministic quality gates for synthetic procurement records."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from rapidfuzz.fuzz import token_set_ratio

NUMBER = re.compile(r"(?<!\w)(?:₹|Rs\.?\s*)?\d[\d,.]*(?:\s*%|\s+\w+)?", re.IGNORECASE)
QUALIFIERS = {"must", "shall", "may", "not", "except", "unless", "only", "subject to"}


def validate_record(record: dict[str, Any], passage: str) -> list[str]:
    """Return machine-checkable rejection reasons."""
    reasons: list[str] = []
    evidence = record.get("evidence", [])
    quotes = [item["quote"].strip() for item in evidence]
    if record["answerable"] and not quotes:
        reasons.append("answerable_without_evidence")
    if not record["answerable"] and record["answer"].strip().lower() not in {
        "not answerable from the provided sources.",
        "the provided sources do not contain enough information to answer.",
    }:
        reasons.append("unsafe_unanswerable_answer")
    for quote in quotes:
        if quote not in passage:
            reasons.append("non_verbatim_evidence")
    support = " ".join(quotes).lower()
    for number in NUMBER.findall(record["answer"]):
        if number.strip().lower() not in support:
            reasons.append(f"unsupported_number:{number.strip()}")
    answer = record["answer"].lower()
    for qualifier in QUALIFIERS:
        if qualifier in support and qualifier not in answer and len(quotes) == 1:
            reasons.append(f"dropped_qualifier:{qualifier}")
    steps = record.get("reasoning_steps", [])
    if record["task_type"] == "qa_cot" and len(steps) < 2:
        reasons.append("cot_requires_multiple_steps")
    if record["task_type"] == "qa" and steps:
        reasons.append("qa_must_not_include_reasoning_steps")
    for step in steps:
        if any(quote not in passage for quote in step.get("evidence_quotes", [])):
            reasons.append("reasoning_uses_non_verbatim_evidence")
    return sorted(set(reasons))


def deduplicate(
    records: Iterable[dict[str, Any]], threshold: float = 94.0
) -> tuple[list[dict[str, Any]], int]:
    """Remove exact and near-duplicate questions deterministically."""
    accepted: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        question = " ".join(record["question"].lower().split())
        if any(
            token_set_ratio(question, existing["_normalized_question"]) >= threshold
            for existing in accepted
        ):
            removed += 1
            continue
        record["_normalized_question"] = question
        accepted.append(record)
    for record in accepted:
        record.pop("_normalized_question", None)
    return accepted, removed


def validate_cross_record(record: dict[str, Any], documents: list[dict[str, Any]]) -> list[str]:
    """Check source-specific evidence and connected two-document structure."""
    reasons: list[str] = []
    known = {document["source_id"]: document["passage"] for document in documents}
    used_claim_sources: set[str] = set()
    used_reasoning_sources: set[str] = set()
    if set(known) != {"source_a", "source_b"}:
        reasons.append("invalid_source_bundle")
    for claim in record.get("claims", []):
        if not claim.get("evidence"):
            reasons.append("claim_without_evidence")
        for evidence in claim.get("evidence", []):
            source_id, quote = evidence["source_id"], evidence["quote"]
            if source_id not in known or quote not in known.get(source_id, ""):
                reasons.append("misattributed_or_non_verbatim_evidence")
            used_claim_sources.add(source_id)
    claim_support = " ".join(
        evidence["quote"]
        for claim in record.get("claims", [])
        for evidence in claim.get("evidence", [])
    ).lower()
    for number in NUMBER.findall(record["answer"]):
        if number.strip().lower() not in claim_support:
            reasons.append(f"unsupported_number:{number.strip()}")
    for step in record.get("reasoning_steps", []):
        for evidence in step.get("evidence", []):
            source_id, quote = evidence["source_id"], evidence["quote"]
            if source_id not in known or quote not in known.get(source_id, ""):
                reasons.append("misattributed_or_non_verbatim_reasoning_evidence")
            used_reasoning_sources.add(source_id)
    if record["answerable"] and used_claim_sources != {"source_a", "source_b"}:
        reasons.append("claims_do_not_require_both_sources")
    is_cot = record["task_type"] == "cross_document_qa_cot"
    if is_cot and (len(record.get("reasoning_steps", [])) < 2 or used_reasoning_sources != {"source_a", "source_b"}):
        reasons.append("cot_is_not_connected_to_both_sources")
    if not is_cot and record.get("reasoning_steps"):
        reasons.append("qa_must_not_include_reasoning_steps")
    return sorted(set(reasons))
