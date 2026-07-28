"""Deterministic candidate construction for cross-document generation."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

WORD = re.compile(r"[a-z][a-z0-9_-]{2,}", re.IGNORECASE)
STOPWORDS = {
    "and", "are", "for", "from", "has", "have", "into", "manual", "may", "must",
    "not", "shall", "that", "the", "their", "this", "under", "with", "will",
    "chapter", "section", "page", "procurement",
}
RELATIONSHIPS = {
    "same_authority_temporal",
    "government_company_comparison",
    "company_cross_domain",
    "complementary_procedure",
}


def _terms(value: str) -> set[str]:
    return {word.lower() for word in WORD.findall(value) if word.lower() not in STOPWORDS}


def _similarity(
    left: dict[str, Any],
    right: dict[str, Any],
    term_cache: dict[str, tuple[set[str], set[str]]],
) -> tuple[float, list[str]]:
    left_terms, section_left = term_cache[left["chunk_id"]]
    right_terms, section_right = term_cache[right["chunk_id"]]
    shared = sorted(left_terms & right_terms)
    if not left_terms or not right_terms:
        return 0.0, shared
    content_overlap = len(shared) / len(left_terms | right_terms)
    section_overlap = (
        len(section_left & section_right) / len(section_left | section_right)
        if section_left and section_right
        else 0.0
    )
    return 100.0 * (0.65 * content_overlap + 0.35 * section_overlap), shared


def validate_pairs(config: dict[str, Any], known_manuals: set[str]) -> list[dict[str, str]]:
    """Validate configured manual relationships before candidate construction."""
    pairs = config.get("pairs", [])
    seen: set[str] = set()
    normalized = []
    for raw in pairs:
        pair = {key: str(raw[key]) for key in ("pair_id", "left_manual", "right_manual", "relationship_type")}
        if pair["pair_id"] in seen:
            raise ValueError(f"Duplicate cross-document pair_id: {pair['pair_id']}")
        if pair["left_manual"] == pair["right_manual"]:
            raise ValueError(f"Cross-document pair uses one manual twice: {pair['pair_id']}")
        missing = {pair["left_manual"], pair["right_manual"]} - known_manuals
        if missing:
            raise ValueError(f"Cross-document pair {pair['pair_id']} has unknown manuals: {sorted(missing)}")
        if pair["relationship_type"] not in RELATIONSHIPS:
            raise ValueError(f"Unsupported cross-document relationship: {pair['relationship_type']}")
        seen.add(pair["pair_id"])
        normalized.append(pair)
    return normalized


def _source(row: dict[str, Any], source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "manual_id": row["manual_id"],
        "title": row["title"],
        "issuing_organization": row["issuing_organization"],
        "policy_scope": row["policy_scope"],
        "revision_date": row["revision_date"],
        "as_of_date": row["as_of_date"],
        "source_file": row["source_file"],
        "source_sha256": row["source_sha256"],
        "chunk_id": row["chunk_id"],
        "page": row["page"],
        "section": row["section"],
        "passage": row["passage"],
    }


def build_bundles(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Propose bounded two-source bundles; similarity never asserts a legal relationship."""
    by_manual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_manual[row["manual_id"]].append(row)
    term_cache = {
        row["chunk_id"]: (
            _terms(f"{row.get('section', '')} {row['passage']}"),
            _terms(str(row.get("section", ""))),
        )
        for row in rows
    }
    pairs = validate_pairs(config, set(by_manual))
    minimum = float(config.get("minimum_similarity", 18))
    minimum_shared = int(config.get("minimum_shared_terms", 3))
    maximum = int(config.get("max_bundles_per_pair", 100))
    bundles = []
    for pair in pairs:
        candidates = []
        for left in by_manual[pair["left_manual"]]:
            for right in by_manual[pair["right_manual"]]:
                score, shared = _similarity(left, right, term_cache)
                if score >= minimum and len(shared) >= minimum_shared:
                    candidates.append((score, left["chunk_id"], right["chunk_id"], shared, left, right))
        for score, _, _, shared, left, right in sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[:maximum]:
            identity = f"{pair['pair_id']}:{left['chunk_id']}:{right['chunk_id']}"
            bundles.append(
                {
                    "source_bundle_id": "xdb-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                    **pair,
                    "alignment_score": round(score, 3),
                    "shared_terms": shared[:30],
                    "source_documents": [_source(left, "source_a"), _source(right, "source_b")],
                }
            )
    return bundles


def evidence_location(
    source_documents: list[dict[str, Any]], source_id: str, quote: str
) -> dict[str, Any] | None:
    """Resolve an exact quotation to its declared source and offsets."""
    document = next((item for item in source_documents if item["source_id"] == source_id), None)
    if document is None:
        return None
    start = document["passage"].find(quote)
    if start < 0:
        return None
    return {
        "source_id": source_id,
        "manual_id": document["manual_id"],
        "chunk_id": document["chunk_id"],
        "page": document["page"],
        "section": document["section"],
        "quote": quote,
        "start_char": start,
        "end_char": start + len(quote),
    }
