"""Build bounded, provenance-preserving windows from immutable corpus chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

WINDOW_SCHEMA_VERSION = "1"
GENERIC_HEADINGS = {
    "",
    "chapter",
    "general",
    "introduction",
    "note",
    "notes",
    "section",
}
SPACE = re.compile(r"\s+")


def _section_key(row: dict[str, Any]) -> tuple[str, ...]:
    path = row.get("section_path") or []
    normalized = tuple(SPACE.sub(" ", str(value).casefold()).strip() for value in path)
    if not normalized or normalized[-1] in GENERIC_HEADINGS:
        return ()
    return normalized


def _window_id(rows: list[dict[str, Any]]) -> str:
    identity = {
        "schema_version": WINDOW_SCHEMA_VERSION,
        "manual_id": rows[0]["manual_id"],
        "source_sha256": rows[0]["source_sha256"],
        "chunk_ids": [row["chunk_id"] for row in rows],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return "window-" + digest[:24]


def _materialize(
    rows: list[dict[str, Any]],
    maximum_input_tokens: int,
    reserved_prompt_tokens: int,
    chars_per_token: float,
) -> dict[str, Any]:
    text = "\n\n".join(row["generation_passage"] for row in rows)
    estimated_source_tokens = int(len(text) / chars_per_token) + 1
    return {
        "window_id": _window_id(rows),
        "manual_id": rows[0]["manual_id"],
        "manual_title": rows[0]["title"],
        "issuing_organization": rows[0]["issuing_organization"],
        "policy_scope": rows[0]["policy_scope"],
        "revision_date": rows[0]["revision_date"],
        "as_of_date": rows[0]["as_of_date"],
        "source_sha256": rows[0]["source_sha256"],
        "section_path": rows[0].get("section_path") or [],
        "boundary_confidence": "explicit_markdown_heading" if _section_key(rows[0]) else "physical_adjacency",
        "chunk_ids": [row["chunk_id"] for row in rows],
        "pages": list(dict.fromkeys(row["page"] for row in rows)),
        "chunks": [
            {
                "chunk_id": row["chunk_id"],
                "page": row["page"],
                "document_order": row["document_order"],
                "section": row["section"],
                "passage": row["passage"],
                "generation_passage": row["generation_passage"],
            }
            for row in rows
        ],
        "generation_passage": text,
        "token_budget": {
            "method": "conservative_character_estimate",
            "chars_per_token": chars_per_token,
            "estimated_source_tokens": estimated_source_tokens,
            "reserved_prompt_tokens": reserved_prompt_tokens,
            "maximum_input_tokens": maximum_input_tokens,
            "passed": estimated_source_tokens + reserved_prompt_tokens <= maximum_input_tokens,
        },
        "support_edges": [],
        "schema_version": WINDOW_SCHEMA_VERSION,
    }


def build_source_windows(
    chunks: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group adjacent same-section chunks, splitting at configured budgets."""
    maximum_chunks = int(config.get("max_chunks", 4))
    maximum_input_tokens = int(config.get("max_input_tokens", 8192))
    reserved_prompt_tokens = int(config.get("reserved_prompt_tokens", 3072))
    chars_per_token = float(config.get("conservative_chars_per_token", 2.5))
    if maximum_chunks < 1 or chars_per_token <= 0:
        raise ValueError("Invalid source-window bounds")
    available = maximum_input_tokens - reserved_prompt_tokens
    if available < 1:
        raise ValueError("Source-window prompt reservation exhausts context")

    by_manual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        by_manual[row["manual_id"]].append(row)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for manual_rows in by_manual.values():
        ordered = sorted(manual_rows, key=lambda row: row["document_order"])
        current: list[dict[str, Any]] = []
        current_key: tuple[str, ...] = ()
        for row in ordered:
            key = _section_key(row)
            candidate = [*current, row]
            estimate = int(len("\n\n".join(item["generation_passage"] for item in candidate)) / chars_per_token) + 1
            boundary = bool(current and key and current_key and key != current_key)
            if current and (boundary or len(candidate) > maximum_chunks or estimate > available):
                accepted.append(
                    _materialize(
                        current,
                        maximum_input_tokens,
                        reserved_prompt_tokens,
                        chars_per_token,
                    )
                )
                current = []
            current.append(row)
            current_key = key
            single = _materialize(
                current,
                maximum_input_tokens,
                reserved_prompt_tokens,
                chars_per_token,
            )
            if not single["token_budget"]["passed"]:
                rejected.append(
                    {
                        **single,
                        "rejection_reasons": ["source_chunk_exceeds_token_budget"],
                    }
                )
                current = []
                current_key = ()
        if current:
            accepted.append(
                _materialize(
                    current,
                    maximum_input_tokens,
                    reserved_prompt_tokens,
                    chars_per_token,
                )
            )
    return accepted, rejected
