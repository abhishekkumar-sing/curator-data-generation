"""Adversarial unanswerable construction and promotion regressions."""

import sys
from pathlib import Path
from types import SimpleNamespace

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from schemas import AnswerabilityDecision, UnanswerableQuestionDraft  # noqa: E402
from unanswerable import (  # noqa: E402
    ABSTENTION,
    AdversarialUnanswerableGenerator,
    IndependentAnswerabilityJudge,
    answer_type,
    build_unanswerable_inputs,
)


def _record(record_id: str, chunk_id: str, answer: str, quote: str) -> dict:
    return {
        "record_id": record_id,
        "task_type": "qa",
        "task": "tendering",
        "persona": "procurement_officer",
        "question_type": "threshold",
        "question": f"What threshold applies to procurement case {record_id}?",
        "answer": answer,
        "answerable": True,
        "evidence": [
            {
                "quote": quote,
                "chunk_id": chunk_id,
                "page": 2,
                "section": "Threshold",
            }
        ],
        "manual_id": "goods_2024",
        "manual_title": "Goods Manual",
        "issuing_organization": "Government of India",
        "policy_scope": "government_reference",
        "revision_date": "2024",
        "as_of_date": "2024",
        "source_file": "goods.md",
        "source_sha256": "a" * 64,
        "source_chunk_ids": [chunk_id],
        "parent_request_id": f"parent-{record_id}",
        "_source_passage": quote + " Additional policy context.",
    }


def test_same_type_distractor_is_distinct_and_deterministic() -> None:
    records = [
        _record("one", "chunk-one", "Rs 5 lakh", "The limit is Rs 5 lakh."),
        _record("two", "chunk-two", "Rs 10 lakh", "The limit is Rs 10 lakh."),
    ]
    assert answer_type(records[0]) == "number_or_threshold"
    first = build_unanswerable_inputs(records, 0.5, "seed")
    second = build_unanswerable_inputs(records, 0.5, "seed")
    assert first == second
    assert len(first) == 2
    assert first[0]["distractor"]["chunk_id"] not in first[0]["seed_record"][
        "source_chunk_ids"
    ]


def test_generator_materializes_abstention_with_lineage() -> None:
    source = _record(
        "one", "chunk-one", "Rs 5 lakh", "The limit is Rs 5 lakh."
    )
    distractor = _record(
        "two", "chunk-two", "Rs 10 lakh", "The limit is Rs 10 lakh."
    )
    row = build_unanswerable_inputs([source, distractor], 0.5, "seed")[0]
    result = AdversarialUnanswerableGenerator.parse(
        SimpleNamespace(model_name="generator"),
        row,
        UnanswerableQuestionDraft(
            question="What limit applies when the procurement is made on Mars?",
            missing_premise="procurement made on Mars",
        ),
    )
    assert result["answer"] == ABSTENTION
    assert result["answerable"] is False
    assert result["claims"] == []
    assert result["deterministic_checks"]["passed"]
    assert result["unanswerable_construction"]["distractor"]["quote"]


def test_independent_judge_fails_closed() -> None:
    row = {"record_id": "negative-1"}
    accepted = IndependentAnswerabilityJudge.parse(
        SimpleNamespace(model_name="judge"),
        row,
        AnswerabilityDecision(
            record_id="negative-1",
            full_context_answerable=False,
            altered_premise_absent=True,
            distractor_is_same_type=True,
            abstention_is_appropriate=True,
            score=5,
        ),
    )
    assert accepted["answerability_judge"]["accepted"]

    rejected = IndependentAnswerabilityJudge.parse(
        SimpleNamespace(model_name="judge"),
        row,
        AnswerabilityDecision(
            record_id="negative-1",
            full_context_answerable=True,
            altered_premise_absent=False,
            distractor_is_same_type=True,
            abstention_is_appropriate=False,
            score=5,
        ),
    )
    assert not rejected["answerability_judge"]["accepted"]
