"""Deterministic quality gates for synthetic procurement records."""

from __future__ import annotations

import math
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
DEONTIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "prohibition": (
        r"\b(?:shall|must|may)\s+not\b",
        r"\b(?:is|are)\s+prohibited\b",
        r"\bprohibit(?:s|ed|ing)?\b",
        r"\bno\b[^.\n]{0,80}\bshall\b",
    ),
    "obligation": (
        r"\bshall\b",
        r"\bmust\b",
        r"\brequired\s+to\b",
        r"\bmandatory\b",
    ),
    "recommendation": (r"\bshould\b", r"\brecommended\b", r"\badvisable\b"),
    "permission": (
        r"\bmay\b",
        r"\bpermitted\b",
        r"\ballowed\b",
        r"\bentitled\b",
    ),
}
ABSENCE_CLAIM = re.compile(
    r"\b(?:"
    r"(?:is|are|was|were)\s+(?:not\s+present|absent)|"
    r"(?:does|do|did)\s+not\s+"
    r"(?:contain|include|mention|provide|state|require|mandate|specify)|"
    r"(?:not|never)\s+(?:mentioned|provided|stated)|"
    r"no\s+(?:such\s+)?provision|"
    r"lack(?:s|ed|ing)?\s+(?:a|the|such)\s+provision"
    r")\b",
    re.IGNORECASE,
)
DANGLING_FINAL_WORD = re.compile(
    r"\b(?:a|an|and|at|but|by|for|from|if|in|of|on|or|over|than|that|the|" r"to|under|when|which|while|whose|with)\s*$",
    re.IGNORECASE,
)
ANSWER_DANGLING_FINAL_WORD = re.compile(
    r"\b(?:a|an|and|are|at|but|by|for|from|has|have|if|in|is|of|on|or|over|" r"than|that|the|to|under|was|were|when|which|while|whose|with)\s*$",
    re.IGNORECASE,
)
TRUNCATED_TERMINAL = re.compile(r"(?:[,;:]|\.{3}|…)\s*$")
BRACKET_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9&/-]{1,15}\b")
INSTRUCTIONAL_EMBELLISHMENTS = (
    "case study",
    "check for understanding",
    "role-play",
    "role play",
    "lecture point",
    "let's dive",
    "feel free to share",
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


def _quantity_is_supported(value: str, unit: str, support_text: str) -> bool:
    supported = {(support_value, support_unit) for _, support_value, support_unit in _quantities(support_text)}
    if (value, unit) in supported:
        return True
    return not unit and value.isdigit() and any(not support_unit and support_value.startswith(f"{value}.") for support_value, support_unit in supported)


def _unsupported_quantities(answer: str, support_text: str) -> list[str]:
    return [display for display, value, unit in _quantities(answer) if not _quantity_is_supported(value, unit, support_text)]


def _claim_statement_wildcard_pattern(statement: str) -> str:
    """Build a regex matching `statement` with every number wildcarded.

    Tolerating a substituted numeric span lets a misattributed answer number
    still resolve to the claim whose subject/entity wording it appears under,
    instead of being matched by exact text (which a wrong number can never
    satisfy) or left unscoped.
    """
    normalized = _normalized_text(statement).rstrip(".")
    pieces: list[str] = []
    last = 0
    for match in NUMBER.finditer(normalized):
        pieces.append(re.escape(normalized[last : match.start()]))
        pieces.append(r"\S+(?:\s+\S+){0,2}")
        last = match.end()
    pieces.append(re.escape(normalized[last:]))
    return "".join(pieces)


def _claim_answer_spans(answer: str, claims: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """Locate each claim's own assertion inside the (normalized) answer text."""
    spans: list[tuple[int, int, str]] = []
    for claim in claims:
        statement = str(claim.get("statement", "")).strip()
        if not statement:
            continue
        pattern = _claim_statement_wildcard_pattern(statement)
        if not pattern:
            continue
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            evidence = " ".join(str(item.get("quote", "")) for item in claim.get("evidence", []))
            spans.append((match.start(), match.end(), evidence))
    return spans


def _unsupported_answer_quantities(
    answer: str,
    claims: list[dict[str, Any]],
    fallback_support: str,
) -> list[str]:
    """Scope each answer-level number to the claim whose wording contains it.

    A blind join across every claim's evidence lets a number that is correct
    for a *different* claim's subject pass unnoticed when the answer
    misattributes it. Resolving a claim-specific span first closes that gap;
    a number outside every span (or inside more than one, i.e. ambiguous)
    falls back to the exact prior union-of-all-evidence behavior, so this is
    never stricter than before on a case we cannot confidently attribute.
    """
    normalized_answer = _normalized_text(answer)
    spans = _claim_answer_spans(normalized_answer, claims)
    unsupported: list[str] = []
    for match in NUMBER.finditer(normalized_answer):
        display = match.group(0).strip()
        value = match.group("value").replace(",", "").lower()
        unit = _canonical_unit(match.group("unit") or "")
        containing = [evidence for start, end, evidence in spans if start <= match.start() < end]
        support = containing[0] if len(containing) == 1 else fallback_support
        if not _quantity_is_supported(value, unit, support):
            unsupported.append(display)
    return unsupported


def _is_incomplete_evidence_fragment(quote: str) -> bool:
    """Detect only high-confidence dangling prose without rejecting headings."""
    text = quote.strip()
    return bool(text and DANGLING_FINAL_WORD.search(text))


def answer_completeness_issues(answer: str) -> list[str]:
    """Detect high-confidence surface evidence that an answer was truncated."""
    text = str(answer or "").strip()
    if not text:
        return ["empty_answer"]
    issues: list[str] = []
    if ANSWER_DANGLING_FINAL_WORD.search(text):
        issues.append("incomplete_answer_dangling_word")
    if TRUNCATED_TERMINAL.search(text):
        issues.append("incomplete_answer_terminal_fragment")
    if any(text.count(opening) != text.count(closing) for opening, closing in BRACKET_PAIRS):
        issues.append("incomplete_answer_unbalanced_brackets")
    return sorted(set(issues))


def answer_format_issues(
    answer: str,
    support_text: str,
    answer_format: str,
    bounds: dict[str, list[int] | tuple[int, int]],
) -> list[str]:
    """Validate concise, source-grounded response presentation.

    Formats are assigned deterministically from the planned question type;
    they are not another model-selected label. The checks stay deliberately
    high precision so normal prose variation is not mistaken for a defect.
    """
    text = str(answer or "").strip()
    issues: list[str] = []
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", text))
    configured = bounds.get(answer_format)
    if configured and len(configured) == 2:
        minimum, maximum = map(int, configured)
        if word_count < minimum:
            issues.append(f"answer_too_short_for_format:{answer_format}:{word_count}")
        if word_count > maximum:
            issues.append(f"answer_too_long_for_format:{answer_format}:{word_count}")
    normalized_answer = text.casefold()
    normalized_support = str(support_text or "").casefold()
    for phrase in INSTRUCTIONAL_EMBELLISHMENTS:
        if phrase in normalized_answer and phrase not in normalized_support:
            issues.append(f"unsupported_instructional_embellishment:{phrase.replace(' ', '_')}")
    if answer_format == "ordered_steps" and not re.search(
        r"(?im)(?:^|\n)\s*(?:step\s+\d+|\d+[.)]|[-*])\s+",
        text,
    ):
        issues.append("ordered_steps_format_missing_structure")
    if answer_format == "audit_check" and not re.search(
        r"(?i)\b(?:verify|confirm|check|evidence|compliance|compliant)\b",
        text,
    ):
        issues.append("audit_check_format_missing_verification_action")
    return sorted(set(issues))


def _unsupported_acronyms(answer: str, support_text: str) -> list[str]:
    supported = set(ACRONYM.findall(str(support_text or "")))
    ignored = {"QA", "SOP"} if not support_text else {"QA"}
    return sorted(acronym for acronym in set(ACRONYM.findall(str(answer or ""))) if acronym not in supported and acronym not in ignored)


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


def recover_grounded_judge_quotes(
    judge_quotes: list[str],
    *,
    answer_found_in_source: bool,
    supported: bool,
    source_text: str,
    evidence_quotes: list[str],
    maximum: int = 3,
) -> tuple[list[str], bool]:
    """Recover omitted judge witnesses from already-validated exact evidence.

    This is intentionally unavailable when the independent judge did not find
    or support the answer. It repairs only an empty witness list and retains
    exact, source-grounded evidence; it never repairs a substantive verdict.
    """
    if judge_quotes or not answer_found_in_source or not supported:
        return judge_quotes, False
    recovered: list[str] = []
    for quote in evidence_quotes:
        if quote and quote not in recovered and judge_quotes_are_grounded([quote], source_text, evidence_quotes):
            recovered.append(quote)
        if len(recovered) >= maximum:
            break
    return recovered, bool(recovered)


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _statement_modalities(text: str) -> set[str]:
    """Return the deontic-modality categories a statement asserts.

    Shared by `semantic_support_issues` (answer-vs-support) and
    `cross_claim_contradiction_issues` (claim-vs-claim / step-vs-answer) so
    both use one definition of what counts as an "obligation" vs. a
    "prohibition", etc.
    """
    modalities = {category for category, patterns in DEONTIC_PATTERNS.items() if _has_pattern(text, patterns)}
    # Prohibitions contain obligation/permission auxiliaries lexically but are
    # semantically their own category.
    if "prohibition" in modalities:
        modalities -= {"obligation", "permission"}
    return modalities


_DEONTIC_MARKER = re.compile(
    r"\b(?:shall|must|may)\s+not\b|\bis\s+prohibited\b|\bare\s+prohibited\b|"
    r"\bprohibit(?:s|ed|ing)?\b|\bshall\b|\bmust\b|\bmay\b|\brequired\s+to\b|"
    r"\bmandatory\b|\bshould\b|\brecommended\b|\badvisable\b|\bpermitted\b|"
    r"\ballowed\b|\bentitled\b",
    re.IGNORECASE,
)


def _core_subject_tokens(statement: str) -> str:
    """Strip deontic markers, returning only the subject/action the statement is about.

    Used to tell "same subject, opposite modality" (a real contradiction)
    apart from "different subject, coincidentally opposite modality" (two
    unrelated rules that happen to use `must` and `must not`).
    """
    stripped = _DEONTIC_MARKER.sub(" ", statement)
    return " ".join(re.findall(r"[a-z0-9]+", stripped.casefold()))


def _is_opposite_modality(left: set[str], right: set[str]) -> bool:
    """Only prohibition-vs-obligation/permission is treated as a true opposite.

    Other pairs (e.g. obligation vs. recommendation) are a *strength*
    difference, not a contradiction, and are already covered separately by
    `semantic_support_issues`'s weakened/strengthened-modality checks.
    """
    return ("prohibition" in left and bool({"obligation", "permission"} & right)) or (
        "prohibition" in right and bool({"obligation", "permission"} & left)
    )


def cross_claim_contradiction_issues(
    statements: list[tuple[str, str]],
    *,
    subject_overlap_threshold: float = 85.0,
    minimum_subject_words: int = 3,
) -> list[str]:
    """Flag statement pairs asserting opposite modalities about the same subject.

    `statements` is a list of `(label, text)` pairs (e.g. `("claim:0", ...)`,
    `("step:1", ...)`, `("answer", ...)`); every distinct pair is compared.

    Deliberately conservative (Finding V2 asks for this, but false positives
    here reject good data): a pair is only flagged when (a) the modalities
    are a true opposite per `_is_opposite_modality` (prohibition vs.
    obligation/permission -- the one pairing the coarse auxiliary-verb-based
    `DEONTIC_PATTERNS` categories can identify without conflating a strength
    difference for a contradiction), and (b) the statements are near-
    identical once their deontic markers are stripped out -- i.e. the same
    core subject/action, not just two rules that happen to share a topic or
    a few incidental words. `minimum_subject_words` additionally guards
    against flagging near-empty core text (e.g. two one-word statements)
    where a high fuzzy-overlap ratio is not a meaningful signal.
    """
    issues: list[str] = []
    annotated = []
    for label, text in statements:
        statement = str(text or "").strip()
        if not statement:
            continue
        modalities = _statement_modalities(statement)
        if not modalities:
            continue
        subject = _core_subject_tokens(statement)
        if len(subject.split()) < minimum_subject_words:
            continue
        annotated.append((label, modalities, subject))
    for left in range(len(annotated)):
        left_label, left_modalities, left_subject = annotated[left]
        for right in range(left + 1, len(annotated)):
            right_label, right_modalities, right_subject = annotated[right]
            if not _is_opposite_modality(left_modalities, right_modalities):
                continue
            if token_set_ratio(left_subject, right_subject) < subject_overlap_threshold:
                continue
            issues.append(f"cross_claim_contradiction:{left_label}:{right_label}")
    return sorted(set(issues))


def semantic_support_issues(answer: str, support_text: str) -> list[str]:
    """Detect high-confidence absence and legal-modality support failures."""
    issues: list[str] = []
    if ABSENCE_CLAIM.search(answer) and not ABSENCE_CLAIM.search(support_text):
        issues.append("unsupported_absence_claim")

    answer_modalities = _statement_modalities(answer)
    support_modalities = _statement_modalities(support_text)

    if "obligation" in answer_modalities and "obligation" not in support_modalities:
        if "permission" in support_modalities:
            issues.append("strengthened_modality:permission_to_obligation")
        elif "recommendation" in support_modalities:
            issues.append("strengthened_modality:recommendation_to_obligation")
        elif not support_modalities:
            issues.append("introduced_modality:obligation")
    if "obligation" in support_modalities and "obligation" not in answer_modalities:
        if "permission" in answer_modalities:
            issues.append("weakened_modality:obligation_to_permission")
        elif "recommendation" in answer_modalities:
            issues.append("weakened_modality:obligation_to_recommendation")
    if "prohibition" in answer_modalities and "prohibition" not in support_modalities:
        issues.append("introduced_modality:prohibition")
    return sorted(set(issues))


def validate_record(record: dict[str, Any], passage: str) -> list[str]:
    """Return machine-checkable rejection reasons."""
    reasons: list[str] = []
    evidence = record.get("evidence", [])
    quotes = [item["quote"].strip() for item in evidence]
    claims = record.get("claims", [])
    reasons.extend(answer_completeness_issues(record.get("answer", "")))
    if record["answerable"] and not quotes:
        reasons.append("answerable_without_evidence")
    if record["answerable"] and not claims:
        reasons.append("answerable_without_claims")
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
    claim_quotes: list[str] = []
    for claim in claims:
        if not str(claim.get("statement", "")).strip():
            reasons.append("claim_without_statement")
        claim_evidence = [item["quote"].strip() for item in claim.get("evidence", [])]
        if not claim_evidence:
            reasons.append("claim_without_evidence")
            continue
        claim_quotes.extend(claim_evidence)
        claim_support = " ".join(claim_evidence)
        reasons.extend(
            f"claim_{issue}"
            for issue in semantic_support_issues(
                str(claim.get("statement", "")),
                claim_support,
            )
        )
        for number in _unsupported_quantities(
            str(claim.get("statement", "")),
            claim_support,
        ):
            reasons.append(f"claim_unsupported_number:{number}")
        for quote in claim_evidence:
            if quote not in passage:
                reasons.append("claim_uses_non_verbatim_evidence")
            if _is_incomplete_evidence_fragment(quote):
                reasons.append("claim_uses_incomplete_evidence_fragment")
    if sorted(set(claim_quotes)) != sorted(set(quotes)):
        reasons.append("claim_evidence_mismatch")
    # Claim-vs-claim only (Finding V2's other example, reasoning-step-vs-
    # terminal-answer, was investigated and deliberately not wired in here:
    # real-data validation found it produces real false positives on this
    # pipeline's own accepted output, because CoT steps are *designed* to be
    # near-restatements that combine into the final answer, and often use
    # investigative "Identify X"/"Contrast X vs Y" framing rather than
    # first-order factual assertions -- see T11's research notes).
    reasons.extend(
        cross_claim_contradiction_issues(
            [(f"claim:{index}", str(claim.get("statement", ""))) for index, claim in enumerate(claims)]
        )
    )
    support = " ".join(quotes)
    reasons.extend(semantic_support_issues(record["answer"], support))
    for number in _unsupported_answer_quantities(record["answer"], claims, support):
        reasons.append(f"unsupported_number:{number}")
    for acronym in _unsupported_acronyms(
        record["answer"],
        f"{passage}\n{record.get('question', '')}",
    ):
        reasons.append(f"unsupported_acronym:{acronym}")
    answer = record["answer"]
    for qualifier, patterns in QUALIFIER_EQUIVALENTS.items():
        if _has_pattern(support, (patterns[0],)) and not _has_pattern(answer, patterns) and len(quotes) == 1:
            reasons.append(f"dropped_qualifier:{qualifier}")
    steps = record.get("reasoning_steps", [])
    if record["task_type"] == "qa_cot" and len(steps) < 2:
        reasons.append("cot_requires_multiple_steps")
    if record["task_type"] == "qa" and steps:
        reasons.append("qa_must_not_include_reasoning_steps")
    if len(steps) > 1:
        normalized_steps = {" ".join(str(step.get("statement", "")).lower().split()) for step in steps}
        if len(normalized_steps) != len(steps):
            reasons.append("cot_repeats_reasoning_step")
        evidence_sets = {tuple(sorted(str(quote).strip() for quote in step.get("evidence_quotes", []))) for step in steps}
        if len(evidence_sets) == 1:
            reasons.append("cot_reuses_identical_evidence_for_all_steps")
    for step in steps:
        if step.get("operation") not in {
            "lookup",
            "compare",
            "apply_condition",
            "resolve_authority",
            "resolve_time",
            "combine",
            "calculate",
            "conclude",
        }:
            reasons.append("cot_step_missing_or_invalid_operation")
        if not step.get("evidence_quotes"):
            reasons.append("cot_step_has_no_grounded_input")
        if any(quote not in passage for quote in step.get("evidence_quotes", [])):
            reasons.append("reasoning_uses_non_verbatim_evidence")
    return sorted(set(reasons))


def deduplicate(
    records: Iterable[dict[str, Any]],
    threshold: float = 94.0,
    preserve_within_group: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Remove near duplicates, optionally preserving deliberate sibling trials."""
    accepted: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        question = " ".join(record["question"].lower().split())
        group = str(record.get(preserve_within_group, "")) if preserve_within_group else ""
        if any(
            not (preserve_within_group and group and group == str(existing.get(preserve_within_group, "")))
            and token_set_ratio(question, existing["_normalized_question"]) >= threshold
            for existing in accepted
        ):
            removed += 1
            continue
        record["_normalized_question"] = question
        accepted.append(record)
    for record in accepted:
        record.pop("_normalized_question", None)
    return accepted, removed


def question_opener_key(question: str, words: int = 4) -> str:
    """Normalize a question's opening n-gram into a template fingerprint."""
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    return " ".join(tokens[:words])


SOURCE_FRAMING_PREFIX = re.compile(
    r"^\s*(?:according\s+to|as\s+per|in\s+accordance\s+with|" r"under\s+(?:the\s+)?(?:manual|policy|rules?|guidelines?))\b",
    re.IGNORECASE,
)


def question_style_issues(question: str, persona: str) -> list[str]:
    """Reject source-reading templates and cosmetic persona preambles."""
    issues: list[str] = []
    value = " ".join(str(question).split())
    if SOURCE_FRAMING_PREFIX.search(value):
        issues.append("templated_source_attribution_opener")
    normalized_persona = " ".join(str(persona).replace("_", " ").split())
    if normalized_persona and normalized_persona != "general user":
        role_prefix = re.compile(
            rf"^\s*as\s+(?:a|an|the)\s+{re.escape(normalized_persona)}\b",
            re.IGNORECASE,
        )
        if role_prefix.search(value):
            issues.append("cosmetic_persona_preamble")
    return issues


def remove_cosmetic_persona_prefix(
    question: str,
    persona: str,
) -> tuple[str, bool]:
    """Remove only an exact ``As a <persona>,`` wrapper, preserving content."""
    normalized_persona = " ".join(str(persona).replace("_", " ").split())
    if not normalized_persona or normalized_persona == "general user":
        return question, False
    prefix = re.compile(
        rf"^\s*as\s+(?:a|an|the)\s+{re.escape(normalized_persona)}\s*,\s*",
        re.IGNORECASE,
    )
    stripped, count = prefix.subn("", question, count=1)
    if not count or not stripped.strip():
        return question, False
    stripped = stripped.strip()
    return stripped[:1].upper() + stripped[1:], True


REASONING_OPERATIONS = {
    "lookup",
    "compare",
    "apply_condition",
    "resolve_authority",
    "resolve_time",
    "combine",
    "calculate",
    "conclude",
}


def canonical_reasoning_operation(operation: str) -> str | None:
    """Map unambiguous natural-language operation labels to the fixed vocabulary."""
    normalized = "_".join(re.findall(r"[a-z0-9]+", str(operation).casefold()))
    if normalized in REASONING_OPERATIONS:
        return normalized
    if "authorit" in normalized:
        return "resolve_authority"
    if any(token in normalized for token in ("time", "date", "revision")):
        return "resolve_time"
    if normalized.startswith(("compare", "contrast")):
        return "compare"
    if normalized.startswith(("apply", "check_condition", "confirm_scope")):
        return "apply_condition"
    if normalized.startswith(("combine", "connect", "synthesi")):
        return "combine"
    if normalized.startswith(("calculate", "compute")):
        return "calculate"
    if normalized.startswith(("conclude", "confirm", "verify")):
        return "conclude"
    if normalized.startswith(("identify", "determine", "find", "trace", "extract")):
        return "lookup"
    return None


def realign_whitespace_verbatim_quote(quote: str, passage: str) -> str | None:
    """Recover an exact source span when only whitespace formatting changed.

    Markdown line wrapping and list layout are presentation differences.
    Punctuation, spelling, casing, and token changes remain unrecoverable and
    therefore fail closed.
    """
    if quote in passage:
        return quote
    tokens = str(quote).split()
    if not tokens:
        return None
    match = re.search(r"\s+".join(re.escape(token) for token in tokens), passage)
    return match.group(0) if match else None


def is_extractive_answer(record: dict[str, Any], minimum_words: int = 4) -> bool:
    """Return whether an answer is a substantial verbatim evidence span.

    Short names, values, and labels are excluded because paraphrasing them would
    usually reduce precision rather than add useful supervision. Whitespace and
    case are normalized, but lexical content is not changed.
    """
    if not record.get("answerable", True):
        return False
    answer_words = re.findall(r"[a-z0-9]+", str(record.get("answer", "")).casefold())
    if len(answer_words) < minimum_words:
        return False
    answer = " ".join(answer_words)
    return any(answer in " ".join(re.findall(r"[a-z0-9]+", str(item.get("quote", "")).casefold())) for item in record.get("evidence", []))


def _cap_binary_share(
    records: list[dict[str, Any]],
    selected: list[bool],
    max_share: float,
) -> tuple[list[dict[str, Any]], int]:
    """Keep the earliest selected rows while enforcing their final-pool share."""
    if not 0 < max_share < 1:
        raise ValueError("max_share must be between 0 and 1")
    selected_count = sum(selected)
    other_count = len(records) - selected_count
    if not selected_count:
        return records, 0
    # k / (other_count + k) <= max_share
    allowed = math.floor((max_share * other_count) / (1.0 - max_share))
    selected_seen = 0
    kept: list[dict[str, Any]] = []
    for record, is_selected in zip(records, selected, strict=True):
        if is_selected:
            selected_seen += 1
            if selected_seen > allowed:
                continue
        kept.append(record)
    return kept, len(records) - len(kept)


def enforce_extractive_answer_diversity(
    records: Iterable[dict[str, Any]],
    max_share: float = 0.35,
    minimum_words: int = 4,
) -> tuple[list[dict[str, Any]], int]:
    """Cap substantial span-copied answers in the resulting record pool."""
    records = list(records)
    return _cap_binary_share(
        records,
        [is_extractive_answer(record, minimum_words) for record in records],
        max_share,
    )


def enforce_question_opener_diversity(
    records: Iterable[dict[str, Any]],
    max_share: float = 0.08,
) -> tuple[list[dict[str, Any]], int]:
    """Cap how much of the accepted pool may share one opening template.

    Generation requests are independent and stateless, so no single call can
    see that many other requests already produced the same opening
    construction; this is a corpus-level property and must be enforced by
    inspecting the accumulated output, not by asking one isolated call to
    self-diversify. Same shape and processing order as `deduplicate` above,
    which already performs this class of growing-pool filter for near-
    identical full questions; this applies it to shared opening templates.
    """
    records = list(records)
    if not records:
        return [], 0
    if not 0 < max_share < 1:
        raise ValueError("max_share must be between 0 and 1")
    keys = [question_opener_key(str(record.get("question", ""))) for record in records]
    active = [True] * len(records)
    counts = Counter(key for key in keys if key)
    indexes: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        if key:
            indexes.setdefault(key, []).append(index)
    total = len(records)
    while counts and total:
        key, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
        # A single occurrence is diversity, not a repeated template. Very small
        # pools may therefore have an unavoidable top share above the target.
        if count <= 1 or count / total <= max_share:
            break
        index = indexes[key].pop()
        active[index] = False
        counts[key] -= 1
        total -= 1
    kept = [record for record, include in zip(records, active, strict=True) if include]
    return kept, len(records) - len(kept)


def enforce_category_diversity(
    records: Iterable[dict[str, Any]],
    field: str,
    max_share: float,
) -> tuple[list[dict[str, Any]], int]:
    """Cap a categorical portfolio field while preserving small-pool coverage."""
    records = list(records)
    if not records:
        return [], 0
    if not 0 < max_share < 1:
        raise ValueError("max_share must be between 0 and 1")
    keys = [str(record.get(field, "")).strip() for record in records]
    active = [True] * len(records)
    counts = Counter(key for key in keys if key)
    indexes: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        if key:
            indexes.setdefault(key, []).append(index)
    total = len(records)
    while counts and total:
        key, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
        if count <= 1 or count / total <= max_share:
            break
        index = indexes[key].pop()
        active[index] = False
        counts[key] -= 1
        total -= 1
    kept = [record for record, include in zip(records, active, strict=True) if include]
    return kept, len(records) - len(kept)


def validate_cross_record(record: dict[str, Any], documents: list[dict[str, Any]]) -> list[str]:
    """Check source-specific evidence and connected two-document structure."""
    reasons: list[str] = []
    known = {document["source_id"]: document["passage"] for document in documents}
    used_claim_sources: set[str] = set()
    used_reasoning_sources: set[str] = set()
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
    if set(known) != {"source_a", "source_b"}:
        reasons.append("invalid_source_bundle")
    reasons.extend(answer_completeness_issues(record.get("answer", "")))
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
        claim_support = " ".join(evidence.get("quote", "") for evidence in claim.get("evidence", []))
        reasons.extend(
            f"claim_{issue}"
            for issue in semantic_support_issues(
                str(claim.get("statement", "")),
                claim_support,
            )
        )
        for number in _unsupported_quantities(
            str(claim.get("statement", "")),
            f"{claim_support}\n{metadata_support}",
        ):
            reasons.append(f"claim_unsupported_number:{number}")
    claim_support = " ".join(evidence["quote"] for claim in record.get("claims", []) for evidence in claim.get("evidence", []))
    reasons.extend(semantic_support_issues(record["answer"], claim_support))
    # Manual identity and version dates are valid support for attribution in the
    # answer even when they are not repeated inside the quoted policy sentence.
    for number in _unsupported_answer_quantities(
        record["answer"],
        record.get("claims", []),
        f"{claim_support}\n{metadata_support}",
    ):
        reasons.append(f"unsupported_number:{number}")
    for step in record.get("reasoning_steps", []):
        if step.get("operation") not in {
            "lookup",
            "compare",
            "apply_condition",
            "resolve_authority",
            "resolve_time",
            "combine",
            "calculate",
            "conclude",
        }:
            reasons.append("cot_step_missing_or_invalid_operation")
        if not step.get("evidence"):
            reasons.append("cot_step_has_no_grounded_input")
        step_support = " ".join(evidence.get("quote", "") for evidence in step.get("evidence", []))
        reasons.extend(
            f"reasoning_{issue}"
            for issue in semantic_support_issues(
                str(step.get("statement", "")),
                step_support,
            )
        )
        for evidence in step.get("evidence", []):
            source_id, quote = evidence["source_id"], evidence["quote"]
            if source_id not in known or quote not in known.get(source_id, ""):
                reasons.append("misattributed_or_non_verbatim_reasoning_evidence")
            if _is_incomplete_evidence_fragment(quote):
                reasons.append("incomplete_reasoning_evidence_fragment")
            used_reasoning_sources.add(source_id)
    steps = record.get("reasoning_steps", [])
    if len(steps) > 1:
        normalized_steps = {" ".join(str(step.get("statement", "")).lower().split()) for step in steps}
        if len(normalized_steps) != len(steps):
            reasons.append("cot_repeats_reasoning_step")
        evidence_sets = {
            tuple(
                sorted(
                    (
                        str(evidence.get("source_id", "")),
                        str(evidence.get("quote", "")).strip(),
                    )
                    for evidence in step.get("evidence", [])
                )
            )
            for step in steps
        }
        if len(evidence_sets) == 1:
            reasons.append("cot_reuses_identical_evidence_for_all_steps")
    if record["answerable"] and used_claim_sources != {"source_a", "source_b"}:
        reasons.append("claims_do_not_require_both_sources")
    is_cot = record["task_type"] == "cross_document_qa_cot"
    if is_cot and (len(record.get("reasoning_steps", [])) < 2 or used_reasoning_sources != {"source_a", "source_b"}):
        reasons.append("cot_is_not_connected_to_both_sources")
    if is_cot:
        synthesis_steps = [
            step
            for step in steps
            if step.get("operation")
            in {
                "compare",
                "apply_condition",
                "resolve_authority",
                "resolve_time",
                "combine",
                "calculate",
                "conclude",
            }
        ]
        if not any({str(evidence.get("source_id", "")) for evidence in step.get("evidence", [])} == {"source_a", "source_b"} for step in synthesis_steps):
            reasons.append("cot_missing_two_source_synthesis_step")
    if not is_cot and record.get("reasoning_steps"):
        reasons.append("qa_must_not_include_reasoning_steps")
    return sorted(set(reasons))
