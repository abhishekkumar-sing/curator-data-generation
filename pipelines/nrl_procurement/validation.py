"""Deterministic quality gates for synthetic procurement records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from rapidfuzz.fuzz import token_set_ratio


def judge_batch_identity_issues(
    expected_ids: list[str],
    returned_ids: list[str],
) -> list[str]:
    """Report exact one-to-one identity failures in a batched judge response."""
    expected_counts = Counter(map(str, expected_ids))
    returned_counts = Counter(map(str, returned_ids))
    issues: list[str] = []
    duplicate_expected = sorted(record_id for record_id, count in expected_counts.items() if count > 1)
    duplicate_returned = sorted(record_id for record_id, count in returned_counts.items() if count > 1)
    missing = sorted(expected_counts.keys() - returned_counts.keys())
    unexpected = sorted(returned_counts.keys() - expected_counts.keys())
    if duplicate_expected:
        issues.append(f"duplicate_expected_record_ids:{','.join(duplicate_expected)}")
    if duplicate_returned:
        issues.append(f"duplicate_judge_record_ids:{','.join(duplicate_returned)}")
    if missing:
        issues.append(f"missing_judge_record_ids:{','.join(missing)}")
    if unexpected:
        issues.append(f"unexpected_judge_record_ids:{','.join(unexpected)}")
    if len(returned_ids) != len(expected_ids):
        issues.append(f"judge_cardinality_mismatch:expected={len(expected_ids)},returned={len(returned_ids)}")
    return issues


def quarantine_invalid_judge_batch(
    judge_items: list[dict[str, Any]],
    returned_ids: list[str],
    model_name: str,
) -> list[dict[str, Any]] | None:
    """Return one rejected original per expected ID when batch identity is invalid."""
    expected_ids = [str(item["record_id"]) for item in judge_items]
    issues = judge_batch_identity_issues(expected_ids, returned_ids)
    if not issues:
        return None
    return [
        {
            **item["record"],
            "judge": {
                "accepted": False,
                "batch_integrity_passed": False,
                "model": model_name,
                "issues": issues,
            },
        }
        for item in judge_items
    ]

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
QUALIFIER_EQUIVALENTS = {
    "must": (r"\bmust\b", r"\bshall\b", r"\brequired\b", r"\bmandatory\b"),
    "shall": (
        r"\bshall\b",
        r"\bmust\b",
        r"\brequired\b",
        r"\bmandatory\b",
        r"\b(?:is|are)\s+put\b",
    ),
    "may": (r"\bmay\b", r"\bcan\b", r"\bpermitted\b", r"\ballowed\b", r"\bentitled\b"),
    "not": (r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bneither\b", r"\bnor\b", r"\bwithout\b", r"\bcannot\b"),
    "except": (r"\bexcept\b", r"\bexcluding\b", r"\bother\s+than\b"),
    "unless": (r"\bunless\b", r"\bif\s+not\b", r"\bexcept\s+when\b"),
    "only": (r"\bonly\b", r"\bsolely\b", r"\bexclusively\b", r"\blimited\s+to\b"),
    "subject to": (r"\bsubject\s+to\b", r"\bprovided\s+that\b", r"\bconditional\s+on\b"),
}
DANGLING_FINAL_WORD = re.compile(
    r"\b(?:a|an|and|at|but|by|for|from|if|in|of|on|or|over|than|that|the|" r"to|under|when|which|while|whose|with)\s*$",
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
            not unit and value.isdigit() and any(not support_unit and support_value.startswith(f"{value}.") for support_value, support_unit in supported)
        )
        if not exact and not parent_section:
            unsupported.append(display)
    return unsupported


def _is_incomplete_evidence_fragment(quote: str) -> bool:
    """Detect only high-confidence dangling prose without rejecting headings."""
    text = quote.strip()
    return bool(text and DANGLING_FINAL_WORD.search(text))


_QUOTE_MARK_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))


def _normalized_text(value: str) -> str:
    """Normalize only whitespace; preserve every lexical and punctuation token."""
    return " ".join(str(value or "").split())


def _unwrap_balanced_quote(value: str) -> str:
    text = str(value or "").strip()
    for opening, closing in _QUOTE_MARK_PAIRS:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            return text[len(opening) : -len(closing)].strip()
    return text


def judge_quotes_are_grounded(
    judge_quotes: list[str],
    source_text: str,
    evidence_quotes: list[str],
) -> bool:
    """Verify judge witnesses without weakening the primary evidence contract.

    A witness may be one source substring or a lossless concatenation of
    consecutive, already-verified evidence items. Persisted evidence remains
    unchanged and must have passed the stricter source-specific validator.
    """
    if not judge_quotes:
        return False
    normalized_source = _normalized_text(source_text)
    normalized_evidence = [_normalized_text(quote) for quote in evidence_quotes if quote]
    allowed_concatenations: set[str] = set()
    for start in range(len(normalized_evidence)):
        for end in range(start + 2, len(normalized_evidence) + 1):
            allowed_concatenations.add(" ".join(normalized_evidence[start:end]))
    for quote in judge_quotes:
        normalized_quote = _normalized_text(_unwrap_balanced_quote(quote))
        if not normalized_quote:
            return False
        if normalized_quote not in normalized_source and normalized_quote not in allowed_concatenations:
            return False
    return True


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


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
    answer = record["answer"]
    for qualifier, patterns in QUALIFIER_EQUIVALENTS.items():
        if _has_pattern(support, (patterns[0],)) and not _has_pattern(answer, patterns) and len(quotes) == 1:
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


def deduplicate(records: Iterable[dict[str, Any]], threshold: float = 94.0) -> tuple[list[dict[str, Any]], int]:
    """Remove exact and near-duplicate questions deterministically."""
    accepted: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        question = " ".join(record["question"].lower().split())
        if any(token_set_ratio(question, existing["_normalized_question"]) >= threshold for existing in accepted):
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
    claim_support = " ".join(evidence["quote"] for claim in record.get("claims", []) for evidence in claim.get("evidence", []))
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
    for number in _unsupported_quantities(record["answer"], f"{claim_support}\n{metadata_support}"):
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
