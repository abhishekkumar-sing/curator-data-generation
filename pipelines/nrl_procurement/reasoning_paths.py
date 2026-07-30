"""Construct and validate proposition-grounded reasoning paths before QA."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Literal

PathType = Literal[
    "comparison",
    "bridge",
    "temporal_transition",
    "complementary_procedure",
    "exception_condition_interaction",
    "cross_domain_comparison",
]

PATH_SCHEMA_VERSION = "1"
WORD = re.compile(r"[a-z][a-z0-9_-]{2,}", re.IGNORECASE)
GENERIC_TERMS = {
    "and",
    "bid",
    "buyer",
    "contract",
    "contractor",
    "document",
    "goods",
    "manual",
    "may",
    "must",
    "nrl",
    "procurement",
    "rule",
    "shall",
    "source",
    "the",
    "under",
    "with",
}
UNSAFE_RELATION_CLAIMS = re.compile(
    r"\b(?:adopted|equivalent|governs|in\s+force|supersed(?:e|es|ed)|" r"currently\s+applicable|no\s+longer\s+applies)\b",
    re.IGNORECASE,
)


def _terms(value: str) -> set[str]:
    return {term.casefold() for term in WORD.findall(value or "") if term.casefold() not in GENERIC_TERMS}


def _semantic_text(proposition: dict[str, Any]) -> str:
    fields = [
        proposition["subject"],
        proposition["action"],
        proposition["object"],
        *proposition.get("conditions", []),
        *proposition.get("exceptions", []),
        proposition.get("threshold", {}).get("value", ""),
        proposition.get("threshold", {}).get("unit", ""),
        proposition.get("temporal_scope", ""),
    ]
    return " ".join(str(value) for value in fields if value)


def _render_proposition(proposition: dict[str, Any]) -> str:
    core = " ".join(
        (
            proposition["subject"],
            proposition["action"],
            proposition["object"],
        )
    ).strip()
    additions = []
    if proposition.get("conditions"):
        additions.append("conditions: " + "; ".join(proposition["conditions"]))
    if proposition.get("exceptions"):
        additions.append("exceptions: " + "; ".join(proposition["exceptions"]))
    threshold = proposition.get("threshold", {})
    threshold_text = " ".join(
        value
        for value in (
            str(threshold.get("value", "")).strip(),
            str(threshold.get("unit", "")).strip(),
        )
        if value
    )
    if threshold_text:
        additions.append("threshold: " + threshold_text)
    if proposition.get("temporal_scope"):
        additions.append("temporal scope: " + proposition["temporal_scope"])
    return core + (f" ({'; '.join(additions)})" if additions else "")


def _source_id(proposition: dict[str, Any]) -> str:
    authority = proposition["authority"]
    evidence = proposition["evidence"]
    return f"{authority['manual_id']}:{evidence['chunk_id']}"


def _family(manual_id: str) -> str:
    for family in ("goods", "works", "services", "consultancy"):
        if family in manual_id:
            return family
    return manual_id


def _bridge_entities(
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[str]:
    generic = {
        "bidder",
        "buyer",
        "consultant",
        "contractor",
        "department",
        "entity",
        "government",
        "manual",
        "materials",
        "officer",
        "payment",
        "procurement",
        "project",
        "provider",
        "service",
        "supplier",
        "tender",
    }
    forward = _terms(left["object"]) & (_terms(right["subject"]) | _terms(right["object"]))
    reverse = _terms(right["object"]) & (_terms(left["subject"]) | _terms(left["object"]))
    return sorted((forward | reverse) - generic)


def _compatible_signature(
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[str]:
    subject = _terms(left["subject"]) & _terms(right["subject"])
    action = _terms(left["action"]) & _terms(right["action"])
    objects = _terms(left["object"]) & _terms(right["object"])
    if not subject or not (action or objects):
        return []
    return sorted(subject | action | objects)


def _path_type(
    configured_relationship: str,
    left: dict[str, Any],
    right: dict[str, Any],
    signature: list[str],
    bridges: list[str],
) -> PathType | None:
    if signature and (left.get("conditions") or right.get("conditions") or left.get("exceptions") or right.get("exceptions")):
        return "exception_condition_interaction"
    if configured_relationship == "same_authority_temporal" and signature:
        return "temporal_transition"
    if configured_relationship == "complementary_procedure" and bridges:
        return "complementary_procedure"
    if bridges and not signature:
        return "bridge"
    if configured_relationship == "company_cross_domain" and signature:
        return "cross_domain_comparison"
    if configured_relationship == "government_company_comparison" and signature:
        return "comparison"
    return None


def _operations(path_type: PathType) -> list[str]:
    if path_type == "bridge":
        return ["lookup", "bridge", "combine", "conclude"]
    if path_type == "complementary_procedure":
        return ["lookup", "apply_prerequisite", "combine", "conclude"]
    if path_type == "exception_condition_interaction":
        return ["lookup", "apply_condition", "compare", "conclude"]
    if path_type == "temporal_transition":
        return ["lookup", "resolve_time", "compare", "conclude"]
    if path_type == "cross_domain_comparison":
        return ["lookup", "resolve_domain", "compare", "conclude"]
    return ["lookup", "resolve_authority", "compare", "conclude"]


def _output_statement(
    path_type: PathType,
    left: dict[str, Any],
    right: dict[str, Any],
) -> str:
    left_authority = left["authority"]
    right_authority = right["authority"]
    left_label = f"{left_authority['manual_title']} " f"(as of {left.get('temporal_scope') or left_authority['as_of_date']})"
    right_label = f"{right_authority['manual_title']} " f"(as of {right.get('temporal_scope') or right_authority['as_of_date']})"
    relationship = {
        "comparison": "The two attributed source states can be compared",
        "bridge": "The two attributed source propositions form a bridge",
        "temporal_transition": "The two attributed dated states can be compared",
        "complementary_procedure": ("The two attributed source propositions describe complementary procedure"),
        "exception_condition_interaction": ("The attributed condition or exception modifies the compared rule"),
        "cross_domain_comparison": ("The two attributed source states can be compared across procurement domains"),
    }[path_type]
    return f"{relationship}: under {left_label}, {_render_proposition(left)}; " f"under {right_label}, {_render_proposition(right)}."


def _path_identity(
    path_type: PathType,
    left_id: str,
    right_id: str,
) -> tuple[str, str]:
    canonical_inputs = sorted((left_id, right_id))
    payload = json.dumps(
        [PATH_SCHEMA_VERSION, path_type, *canonical_inputs],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return "path-" + digest[:24], "claim-" + digest[24:48]


def _operation_steps(
    path_id: str,
    operations: list[str],
    input_ids: list[str],
    source_ids: list[str],
    output_claim_id: str,
) -> list[dict[str, Any]]:
    steps = [
        {
            "operation": "lookup",
            "input_source_ids": [source_id],
            "input_claim_ids": [],
            "output_claim_id": proposition_id,
        }
        for source_id, proposition_id in zip(source_ids, input_ids, strict=True)
    ]
    current_inputs = list(input_ids)
    derived_operations = operations[1:]
    for index, operation in enumerate(derived_operations):
        is_terminal = index == len(derived_operations) - 1
        step_output = output_claim_id if is_terminal else f"intermediate-{path_id.removeprefix('path-')}-{index + 1}"
        steps.append(
            {
                "operation": operation,
                "input_source_ids": [],
                "input_claim_ids": current_inputs,
                "output_claim_id": step_output,
            }
        )
        current_inputs = [step_output]
    return steps


def validate_reasoning_path(
    path: dict[str, Any],
    propositions: dict[str, dict[str, Any]],
    configured_pair: dict[str, str] | None = None,
) -> list[str]:
    """Return structural, source, relationship, and ablation failures."""
    issues: list[str] = []
    input_ids = path.get("input_claim_ids", [])
    if len(input_ids) != 2 or len(set(input_ids)) != 2:
        issues.append("requires_two_distinct_input_propositions")
        return issues
    if any(proposition_id not in propositions for proposition_id in input_ids):
        issues.append("unknown_input_proposition")
        return issues
    left, right = (propositions[proposition_id] for proposition_id in input_ids)
    required_sources = path.get("required_source_ids", [])
    expected_sources = [_source_id(left), _source_id(right)]
    if required_sources != expected_sources or len(set(required_sources)) != 2:
        issues.append("invalid_required_sources")
    if any(not proposition.get("deterministic_checks", {}).get("passed", False) for proposition in (left, right)):
        issues.append("ungrounded_input_proposition")

    left_manual = left["authority"]["manual_id"]
    right_manual = right["authority"]["manual_id"]
    if configured_pair is not None and {
        left_manual,
        right_manual,
    } != {
        configured_pair["left_manual"],
        configured_pair["right_manual"],
    }:
        issues.append("input_manuals_do_not_match_configured_pair")

    path_type = path.get("relationship_type")
    if path_type not in {
        "comparison",
        "bridge",
        "temporal_transition",
        "complementary_procedure",
        "exception_condition_interaction",
        "cross_domain_comparison",
    }:
        issues.append("invalid_relationship_type")
        return sorted(set(issues))
    signature = _compatible_signature(left, right)
    bridges = _bridge_entities(left, right)
    declared_anchors = path.get("connection_anchors", [])
    if any(anchor not in signature + bridges for anchor in declared_anchors):
        issues.append("unsupported_connection_anchor")
    if (
        path_type
        in {
            "comparison",
            "temporal_transition",
            "exception_condition_interaction",
            "cross_domain_comparison",
        }
        and not signature
    ):
        issues.append("incompatible_proposition_signatures")
    if path_type in {"bridge", "complementary_procedure"} and not bridges:
        issues.append("missing_explicit_bridge")
    if path_type == "temporal_transition":
        if left["authority"]["issuing_organization"] != right["authority"]["issuing_organization"]:
            issues.append("temporal_authority_mismatch")
        if _family(left_manual) != _family(right_manual):
            issues.append("temporal_manual_family_mismatch")
        if left["authority"]["as_of_date"] == right["authority"]["as_of_date"]:
            issues.append("temporal_states_have_same_date")
    if path_type == "cross_domain_comparison" and _family(left_manual) == _family(right_manual):
        issues.append("cross_domain_path_uses_one_domain")
    if path_type == "exception_condition_interaction" and not any(
        proposition.get("conditions") or proposition.get("exceptions") for proposition in (left, right)
    ):
        issues.append("condition_exception_path_has_no_qualification")

    output_claim = path.get("output_claim", {})
    if (
        not output_claim.get("claim_id")
        or not output_claim.get("statement")
        or output_claim.get("derived_from") != input_ids
        or path.get("output_claim_id") != output_claim.get("claim_id")
    ):
        issues.append("invalid_output_claim")
    if UNSAFE_RELATION_CLAIMS.search(str(output_claim.get("statement", ""))):
        issues.append("unsupported_legal_relationship_claim")
    if path.get("operations") != _operations(path_type):
        issues.append("invalid_operations")
    expected_steps = _operation_steps(
        path["path_id"],
        _operations(path_type),
        input_ids,
        expected_sources,
        str(path.get("output_claim_id", "")),
    )
    if path.get("operation_steps") != expected_steps:
        issues.append("invalid_operation_graph")

    ablations = path.get("structural_ablation", {})
    for proposition_id in input_ids:
        result = ablations.get(f"without:{proposition_id}", {})
        if result.get("complete") is not False or result.get("missing_inputs") != [proposition_id]:
            issues.append("invalid_structural_ablation")
    return sorted(set(issues))


def build_reasoning_paths(
    propositions: list[dict[str, Any]],
    cross_config: dict[str, Any],
    maximum_per_pair: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct high-precision connected paths from accepted propositions."""
    if maximum_per_pair < 1:
        raise ValueError("maximum_per_pair must be positive")
    accepted = {
        proposition["proposition_id"]: proposition
        for proposition in propositions
        if proposition.get("proposition_id") and proposition.get("deterministic_checks", {}).get("passed", False)
    }
    by_manual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposition in accepted.values():
        by_manual[proposition["authority"]["manual_id"]].append(proposition)

    paths: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for pair in cross_config.get("pairs", []):
        candidates = []
        for left in by_manual.get(pair["left_manual"], []):
            for right in by_manual.get(pair["right_manual"], []):
                signature = _compatible_signature(left, right)
                bridges = _bridge_entities(left, right)
                path_type = _path_type(
                    pair["relationship_type"],
                    left,
                    right,
                    signature,
                    bridges,
                )
                if path_type is None:
                    continue
                anchors = signature or bridges
                score = len(anchors)
                candidates.append(
                    (
                        -score,
                        left["proposition_id"],
                        right["proposition_id"],
                        path_type,
                        anchors,
                        left,
                        right,
                    )
                )
        for (
            _,
            left_id,
            right_id,
            path_type,
            anchors,
            left,
            right,
        ) in sorted(candidates)[:maximum_per_pair]:
            path_id, output_claim_id = _path_identity(path_type, left_id, right_id)
            input_ids = [left_id, right_id]
            source_ids = [_source_id(left), _source_id(right)]
            operations = _operations(path_type)
            path = {
                "path_id": path_id,
                "relationship_type": path_type,
                "configured_pair_id": pair["pair_id"],
                "configured_relationship_type": pair["relationship_type"],
                "required_source_ids": source_ids,
                "input_claim_ids": input_ids,
                "operations": operations,
                "operation_steps": _operation_steps(
                    path_id,
                    operations,
                    input_ids,
                    source_ids,
                    output_claim_id,
                ),
                "connection_anchors": anchors,
                "output_claim_id": output_claim_id,
                "output_claim": {
                    "claim_id": output_claim_id,
                    "statement": _output_statement(path_type, left, right),
                    "derived_from": input_ids,
                },
                "structural_ablation": {
                    f"without:{proposition_id}": {
                        "complete": False,
                        "missing_inputs": [proposition_id],
                    }
                    for proposition_id in input_ids
                },
                "schema_version": PATH_SCHEMA_VERSION,
            }
            issues = validate_reasoning_path(path, accepted, pair)
            path["deterministic_checks"] = {
                "passed": not issues,
                "issues": issues,
            }
            (paths if not issues else rejected).append(path)
    return paths, rejected
