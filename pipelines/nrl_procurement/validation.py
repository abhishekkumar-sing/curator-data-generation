"""Deterministic quality gates for synthetic procurement records."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from rapidfuzz.fuzz import token_set_ratio

# Keep the numeric core separate from prose. The previous ``\s+\w+`` suffix
# swallowed arbitrary words (for example, ``2019 Manual``), turning ordinary
# document-version metadata into a fabricated "unit".
NUMBER = re.compile(
    r"""(?<!\w)
    (?P<prefix>₹|Rs\.?\s*)?
    (?P<value>\d(?:[\d,.]*\d)?(?:/\d(?:[\d,.]*\d)?)?)
    (?:\s*\([^)]{1,24}\))?
    (?P<unit>
        \s*(?:%|per\s+cent|percent|percentage|
        days?|weeks?|months?|years?|hours?|minutes?|
        crores?|lakhs?|millions?|billions?)
    )?
    (?!\w)
    """,
    re.IGNORECASE | re.VERBOSE,
)
QUALIFIERS = {"must", "shall", "may", "not", "except", "unless", "only", "subject to"}
DANGLING_FINAL_WORD = re.compile(
    r"\b(?:a|an|and|at|but|by|for|from|if|in|of|on|or|over|than|that|the|"
    r"to|under|when|which|while|whose|with)\s*$",
    re.IGNORECASE,
)


def _canonical_unit(unit: str) -> str:
    normalized = " ".join(unit.lower().split())
    if normalized in {"%", "per cent", "percent", "percentage"}:
        return "%"
    return normalized.removesuffix("s")


def _quantities(text: str) -> list[tuple[str, str, str]]:
    """Return display text, normalized numeric value, and typed unit."""
    quantities = []
    for match in NUMBER.finditer(text):
        value = match.group("value").replace(",", "").lower()
        unit = _canonical_unit(match.group("unit") or "")
        quantities.append((match.group(0).strip(), value, unit))
    return quantities


def _unsupported_quantities(answer: str, support_text: str) -> list[str]:
    supported = {(value, unit) for _, value, unit in _quantities(support_text)}
    unsupported = []
    for display, value, unit in _quantities(answer):
        exact = (value, unit) in supported
        parent_section = (
            not unit
            and value.isdigit()
            and any(
                not support_unit and support_value.startswith(f"{value}.")
                for support_value, support_unit in supported
            )
        )
        if not exact and not parent_section:
            unsupported.append(display)
    return unsupported


def _is_incomplete_evidence_fragment(quote: str) -> bool:
    """Detect only high-confidence dangling prose without rejecting headings."""
    text = quote.strip()
    return bool(text and DANGLING_FINAL_WORD.search(text))


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
        if _is_incomplete_evidence_fragment(quote):
            reasons.append("incomplete_evidence_fragment")
    support = " ".join(quotes)
    for number in _unsupported_quantities(record["answer"], support):
        reasons.append(f"unsupported_number:{number}")
    support_lower = support.lower()
    answer = record["answer"].lower()
    for qualifier in QUALIFIERS:
        if qualifier in support_lower and qualifier not in answer and len(quotes) == 1:
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
            if _is_incomplete_evidence_fragment(quote):
                reasons.append("incomplete_evidence_fragment")
            used_claim_sources.add(source_id)
    claim_support = " ".join(
        evidence["quote"]
        for claim in record.get("claims", [])
        for evidence in claim.get("evidence", [])
    )
    # Manual identity and version dates are valid support for attribution in the
    # answer even when they are not repeated inside the quoted policy sentence.
    metadata_support = " ".join(
        str(document.get(field, ""))
        for document in documents
        for field in (
            "manual_id",
            "title",
            "revision_date",
            "as_of_date",
            "page",
            "section",
        )
    )
    for number in _unsupported_quantities(
        record["answer"], f"{claim_support}\n{metadata_support}"
    ):
        reasons.append(f"unsupported_number:{number}")
    for step in record.get("reasoning_steps", []):
        for evidence in step.get("evidence", []):
            source_id, quote = evidence["source_id"], evidence["quote"]
            if source_id not in known or quote not in known.get(source_id, ""):
                reasons.append("misattributed_or_non_verbatim_reasoning_evidence")
            if _is_incomplete_evidence_fragment(quote):
                reasons.append("incomplete_reasoning_evidence_fragment")
            used_reasoning_sources.add(source_id)
    if record["answerable"] and used_claim_sources != {"source_a", "source_b"}:
        reasons.append("claims_do_not_require_both_sources")
    is_cot = record["task_type"] == "cross_document_qa_cot"
    if is_cot and (len(record.get("reasoning_steps", [])) < 2 or used_reasoning_sources != {"source_a", "source_b"}):
        reasons.append("cot_is_not_connected_to_both_sources")
    if not is_cot and record.get("reasoning_steps"):
        reasons.append("qa_must_not_include_reasoning_steps")
    return sorted(set(reasons))
