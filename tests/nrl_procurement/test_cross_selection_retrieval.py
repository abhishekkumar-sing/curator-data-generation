"""Atomic cross bindings, best-of-N, and retrieval context regressions."""

import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from cross_stage import (  # noqa: E402
    cross_binding_issues,
    select_best_cross_candidates,
)
from retrieval_contexts import build_retrieval_contexts  # noqa: E402


def _cross(record_id: str, score: int) -> dict:
    evidence_a = {
        "source_id": "source_a",
        "citation_id": "citation-a",
        "quote": "Source A rule",
    }
    evidence_b = {
        "source_id": "source_b",
        "citation_id": "citation-b",
        "quote": "Source B rule",
    }
    return {
        "record_id": record_id,
        "parent_request_id": "parent-1",
        "question": f"How do both rules apply in {record_id}?",
        "required_source_ids": ["source_a", "source_b"],
        "claims": [
            {
                "claim_id": "claim-1",
                "statement": "Combined rule",
                "evidence": [evidence_a, evidence_b],
            }
        ],
        "claim_source_bindings": [
            {
                "claim_id": "claim-1",
                "source_ids": ["source_a", "source_b"],
                "citation_ids": ["citation-a", "citation-b"],
            }
        ],
        "citations": [
            {"citation_id": "citation-a"},
            {"citation_id": "citation-b"},
        ],
        "judge": {
            "accepted": True,
            "score": score,
            "preserves_qualifications": True,
        },
    }


def test_atomic_cross_bindings_are_bidirectional() -> None:
    row = _cross("one", 5)
    assert cross_binding_issues(row) == []
    row["claim_source_bindings"][0]["citation_ids"] = ["missing"]
    assert "cross_binding_citation_mismatch:claim-1" in cross_binding_issues(row)


def test_best_of_n_keeps_highest_quality_sibling() -> None:
    winner = _cross("winner", 5)
    loser = _cross("loser", 4)
    selected, rejected = select_best_cross_candidates([loser, winner])
    assert [row["record_id"] for row in selected] == ["winner"]
    assert [row["record_id"] for row in rejected] == ["loser"]
    assert rejected[0]["best_of_n"]["winner_record_id"] == "winner"


def test_retrieval_distractors_never_contain_golden_chunks() -> None:
    corpus = [
        {
            "chunk_id": "gold",
            "manual_id": "goods_2024",
            "source_category": "government_manual",
            "section": "Bid security",
            "generation_passage": "Bid security and tender requirements.",
        },
        {
            "chunk_id": "wrong-edition",
            "manual_id": "goods_2017",
            "source_category": "government_manual",
            "section": "Bid security",
            "generation_passage": "Earlier bid security requirements.",
        },
        {
            "chunk_id": "wrong-authority",
            "manual_id": "nrl_goods_rev1",
            "source_category": "company_manual",
            "section": "Bid security",
            "generation_passage": "Company bid security requirements.",
        },
    ]
    records = [
        {
            "record_id": "record-1",
            "question": "What are the bid security requirements?",
            "source_chunk_ids": ["gold"],
        }
    ]
    result = build_retrieval_contexts(records, corpus, "seed")[0]
    distractors = [
        context
        for context in result["contexts"]
        if context["kind"] not in {"oracle", "missing_source"}
    ]
    assert distractors
    assert all("gold" not in context["chunk_ids"] for context in distractors)
    assert records[0]["retrieval_evaluation"] == result
