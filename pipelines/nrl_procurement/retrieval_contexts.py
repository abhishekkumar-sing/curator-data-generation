"""Build deterministic RAG evaluation contexts without golden leakage."""

from __future__ import annotations

import hashlib
import re
from typing import Any

WORD = re.compile(r"[a-z][a-z0-9_-]{2,}", re.I)
STOP = {"and", "for", "from", "that", "the", "this", "with", "shall", "may"}


def _terms(row: dict[str, Any]) -> set[str]:
    value = f"{row.get('section', '')} {row.get('generation_passage', '')}"
    return {word.casefold() for word in WORD.findall(value) if word.casefold() not in STOP}


def _family(manual_id: str) -> str:
    for family in ("goods", "works", "services", "consultancy"):
        if family in manual_id:
            return family
    return "other"


def _best_distractor(
    record: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    predicate,
    seed: str,
) -> str | None:
    golden = set(record.get("source_chunk_ids", []))
    question_terms = {
        word.casefold()
        for word in WORD.findall(str(record.get("question", "")))
        if word.casefold() not in STOP
    }
    candidates = [
        row for row in rows if row["chunk_id"] not in golden and predicate(row)
    ]
    if not candidates:
        return None
    chosen = max(
        candidates,
        key=lambda row: (
            len(question_terms & _terms(row)),
            hashlib.sha256(
                f"{seed}:{record['record_id']}:{row['chunk_id']}".encode()
            ).hexdigest(),
        ),
    )
    return str(chosen["chunk_id"])


def build_retrieval_contexts(
    records: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    seed: str,
) -> list[dict[str, Any]]:
    """Attach oracle and controlled counterfactual chunk-ID contexts."""
    by_chunk = {str(row["chunk_id"]): row for row in corpus_rows}
    results: list[dict[str, Any]] = []
    for record in records:
        golden = list(dict.fromkeys(record.get("source_chunk_ids", [])))
        golden_rows = [by_chunk[item] for item in golden if item in by_chunk]
        manual_ids = {str(row["manual_id"]) for row in golden_rows}
        categories = {str(row.get("source_category")) for row in golden_rows}
        families = {_family(manual_id) for manual_id in manual_ids}
        contexts = [{"kind": "oracle", "chunk_ids": golden}]
        contexts.extend(
            {
                "kind": "missing_source",
                "removed_chunk_id": removed,
                "chunk_ids": [item for item in golden if item != removed],
            }
            for removed in golden
        )
        wrong_edition = _best_distractor(
            record,
            corpus_rows,
            predicate=lambda row, fs=families, mids=manual_ids: (
                _family(str(row["manual_id"])) in fs
                and str(row["manual_id"]) not in mids
            ),
            seed=seed + ":wrong-edition",
        )
        wrong_authority = _best_distractor(
            record,
            corpus_rows,
            predicate=lambda row, cats=categories: str(
                row.get("source_category")
            )
            not in cats,
            seed=seed + ":wrong-authority",
        )
        hard_topical = _best_distractor(
            record,
            corpus_rows,
            predicate=lambda row: True,
            seed=seed + ":hard-topical",
        )
        for kind, chunk_id in (
            ("wrong_edition", wrong_edition),
            ("wrong_authority", wrong_authority),
            ("hard_topical_distractor", hard_topical),
        ):
            if chunk_id:
                contexts.append({"kind": kind, "chunk_ids": [chunk_id]})
        if any(set(context["chunk_ids"]) & set(golden) for context in contexts if context["kind"] not in {"oracle", "missing_source"}):
            raise ValueError("Golden chunks leaked into retrieval distractors")
        result = {
            "record_id": record["record_id"],
            "golden_chunk_ids": golden,
            "contexts": contexts,
        }
        record["retrieval_evaluation"] = result
        results.append(result)
    return results
