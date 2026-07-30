"""Content-addressed claim/evidence graphs and release leakage audits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any


def _stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}-" + hashlib.sha256(encoded.encode()).hexdigest()[:24]


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def build_reasoning_graph(record: dict[str, Any]) -> dict[str, Any]:
    """Derive one replayable graph without changing the canonical answer."""
    evidence_items: dict[str, dict[str, Any]] = {}
    claims: list[dict[str, Any]] = []
    source_claim_ids: list[str] = []
    for claim in record.get("claims", []):
        evidence_ids = []
        for evidence in claim.get("evidence", []):
            payload = {
                key: evidence.get(key, "")
                for key in (
                    "source_id",
                    "proposition_id",
                    "manual_id",
                    "chunk_id",
                    "quote",
                )
            }
            evidence_id = _stable_id("evidence", payload)
            evidence_items[evidence_id] = {"evidence_id": evidence_id, **payload}
            evidence_ids.append(evidence_id)
        claim_id = _stable_id(
            "claim",
            {"statement": claim.get("statement", ""), "evidence_ids": sorted(evidence_ids)},
        )
        claims.append(
            {
                "claim_id": claim_id,
                "statement": claim.get("statement", ""),
                "evidence_refs": sorted(evidence_ids),
                "kind": "source",
            }
        )
        source_claim_ids.append(claim_id)

    steps = []
    previous_outputs: list[str] = []
    raw_steps = record.get("reasoning_steps", [])
    if raw_steps:
        for index, raw in enumerate(raw_steps):
            evidence_refs = []
            raw_evidence = raw.get("evidence", [])
            if not raw_evidence:
                raw_evidence = [
                    {"quote": quote} for quote in raw.get("evidence_quotes", [])
                ]
            for evidence in raw_evidence:
                matches = [
                    evidence_id
                    for evidence_id, item in evidence_items.items()
                    if item["quote"] == evidence.get("quote", "")
                    and (
                        not evidence.get("source_id")
                        or item["source_id"] == evidence.get("source_id")
                    )
                ]
                evidence_refs.extend(matches)
            grounded_inputs = [
                claim["claim_id"]
                for claim in claims
                if set(claim["evidence_refs"]) & set(evidence_refs)
            ]
            input_claim_ids = sorted(set(previous_outputs + grounded_inputs))
            operation = str(raw.get("operation") or ("lookup" if index == 0 else "combine"))
            output_claim_id = _stable_id(
                "claim",
                {
                    "record_id": record.get("record_id", ""),
                    "step": index,
                    "operation": operation,
                    "statement": raw.get("statement", ""),
                    "inputs": input_claim_ids,
                },
            )
            claims.append(
                {
                    "claim_id": output_claim_id,
                    "statement": raw.get("statement", ""),
                    "evidence_refs": sorted(set(evidence_refs)),
                    "kind": "intermediate",
                }
            )
            steps.append(
                {
                    "step_id": _stable_id("step", [record.get("record_id", ""), index]),
                    "operation": operation,
                    "input_claim_ids": input_claim_ids,
                    "output_claim_id": output_claim_id,
                    "evidence_refs": sorted(set(evidence_refs)),
                    "statement": raw.get("statement", ""),
                }
            )
            previous_outputs = [output_claim_id]

    terminal_inputs = previous_outputs or source_claim_ids
    terminal_claim_id = _stable_id(
        "claim",
        {
            "record_id": record.get("record_id", ""),
            "answer": record.get("answer", ""),
            "inputs": terminal_inputs,
        },
    )
    claims.append(
        {
            "claim_id": terminal_claim_id,
            "statement": record.get("answer", ""),
            "evidence_refs": sorted(evidence_items),
            "kind": "terminal",
        }
    )
    steps.append(
        {
            "step_id": _stable_id("step", [record.get("record_id", ""), "terminal"]),
            "operation": "conclude",
            "input_claim_ids": terminal_inputs,
            "output_claim_id": terminal_claim_id,
            "evidence_refs": sorted(evidence_items),
            "statement": record.get("answer", ""),
        }
    )
    graph = {
        "graph_id": _stable_id("graph", record.get("record_id", "")),
        "evidence": sorted(evidence_items.values(), key=lambda item: item["evidence_id"]),
        "claims": claims,
        "steps": steps,
        "source_claim_ids": source_claim_ids,
        "terminal_claim_ids": [terminal_claim_id],
    }
    graph["validation"] = validate_reasoning_graph(graph)
    return graph


def validate_reasoning_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Verify references, single producers, connectivity, acyclicity, and terminals."""
    issues: list[str] = []
    claim_ids = [str(item.get("claim_id", "")) for item in graph.get("claims", [])]
    evidence_ids = {str(item.get("evidence_id", "")) for item in graph.get("evidence", [])}
    if not claim_ids or len(claim_ids) != len(set(claim_ids)) or "" in claim_ids:
        issues.append("invalid_or_duplicate_claim_ids")
    producers: dict[str, str] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    used_inputs: set[str] = set()
    for step in graph.get("steps", []):
        output = str(step.get("output_claim_id", ""))
        inputs = [str(value) for value in step.get("input_claim_ids", [])]
        refs = set(step.get("evidence_refs", []))
        if not step.get("operation"):
            issues.append("step_missing_operation")
        if output not in claim_ids or any(value not in claim_ids for value in inputs):
            issues.append("step_references_unknown_claim")
        if not refs.issubset(evidence_ids):
            issues.append("step_references_unknown_evidence")
        if output in producers:
            issues.append("claim_has_multiple_producers")
        producers[output] = str(step.get("step_id", ""))
        for value in inputs:
            adjacency[value].add(output)
            used_inputs.add(value)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            issues.append("reasoning_graph_cycle")
            return
        if node in visited:
            return
        visiting.add(node)
        for target in adjacency[node]:
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for claim_id in claim_ids:
        visit(claim_id)
    terminals = set(graph.get("terminal_claim_ids", []))
    if not terminals or not terminals.issubset(claim_ids):
        issues.append("invalid_terminal_claims")
    if any(adjacency[terminal] for terminal in terminals):
        issues.append("terminal_claim_has_dependents")
    if not set(graph.get("source_claim_ids", [])).issubset(used_inputs):
        issues.append("unused_source_claim")
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)
    ancestors = set(terminals)
    frontier = list(terminals)
    while frontier:
        for parent in reverse[frontier.pop()]:
            if parent not in ancestors:
                ancestors.add(parent)
                frontier.append(parent)
    if set(claim_ids) - ancestors:
        issues.append("disconnected_claims")
    return {"passed": not issues, "issues": sorted(set(issues))}


def leakage_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report cross-split collisions at every release dependency level."""
    fields: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set)
        for name in (
            "source_hash",
            "manual",
            "section",
            "chunk",
            "path_family",
            "canonical_record",
            "normalized_question",
        )
    }
    for row in records:
        split = str(row.get("split", "unassigned"))
        fields["canonical_record"][str(row.get("record_id", ""))].add(split)
        fields["normalized_question"][_normalized_question(str(row.get("question", "")))].add(split)
        fields["path_family"][str(row.get("path_id") or row.get("source_bundle_id") or row.get("record_id", ""))].add(split)
        for document in row.get("source_documents", []):
            fields["source_hash"][str(document.get("source_sha256", ""))].add(split)
            fields["manual"][str(document.get("manual_id", ""))].add(split)
            fields["section"][f"{document.get('manual_id', '')}:{document.get('section', '')}"].add(split)
        for chunk_id in row.get("source_chunk_ids", []):
            fields["chunk"][str(chunk_id)].add(split)
    collisions = {
        level: [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(values.items())
            if value and len(splits) > 1
        ]
        for level, values in fields.items()
    }
    return {
        "passed": not any(collisions.values()),
        "collisions": collisions,
        "unique_values": {
            level: sum(bool(value) for value in values) for level, values in fields.items()
        },
    }
