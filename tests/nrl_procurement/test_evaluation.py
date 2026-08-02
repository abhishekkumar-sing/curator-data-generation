"""Explicit-fold and frozen external evaluation regressions."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

import evaluation as evaluation_module  # noqa: E402
from evaluation import (  # noqa: E402
    frozen_overlap_issues,
    load_frozen_evaluation,
    regression_report,
    validate_manual_folds,
)
from export import assign_splits  # noqa: E402


def test_manual_folds_are_complete_and_keep_amendments_together() -> None:
    manuals = [
        {"manual_id": "base"},
        {"manual_id": "amendment", "amends": ["base"]},
    ]
    assert validate_manual_folds(
        manuals, {"base": "test", "amendment": "test"}
    ) == {"base": "test", "amendment": "test"}
    with pytest.raises(ValueError, match="must share one fold"):
        validate_manual_folds(
            manuals, {"base": "train", "amendment": "test"}
        )


def test_explicit_folds_reject_cross_fold_record() -> None:
    record = {
        "record_id": "cross",
        "manual_id": "left",
        "source_documents": [
            {"manual_id": "left"},
            {"manual_id": "right"},
        ],
    }
    with pytest.raises(ValueError, match="spans explicit manual folds"):
        assign_splits(
            [record],
            [{"manual_id": "left"}, {"manual_id": "right"}],
            0.8,
            0.1,
            "seed",
            manual_folds={"left": "train", "right": "test"},
        )


def test_frozen_file_requires_hash_and_approved_review(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(evaluation_module, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "frozen.jsonl"
    row = {
        "record_id": "gold-1",
        "question": "What is required?",
        "answer": "Approval is required.",
        "human_review": {"status": "approved", "reviewer_id": "reviewer-1"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "evaluation": {
            "frozen_external": {"path": "frozen.jsonl", "sha256": digest}
        }
    }
    rows, registry = load_frozen_evaluation(config, required=True)
    assert rows == [row]
    assert registry["verified"]

    row["human_review"]["status"] = "pending"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config["evaluation"]["frozen_external"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="without approved human review"):
        load_frozen_evaluation(config, required=True)


def test_overlap_and_regression_metrics() -> None:
    frozen = [
        {
            "record_id": "gold-1",
            "question": "What approval is required?",
            "answer": "Board approval.",
            "answerable": True,
            "source_chunk_ids": ["chunk-1"],
        }
    ]
    assert frozen_overlap_issues(
        [{"record_id": "generated", "question": "What approval is required?"}],
        frozen,
    )["normalized_questions"] == ["generated"]
    report = regression_report(
        frozen,
        [
            {
                "record_id": "gold-1",
                "answer": "Board approval",
                "answerable": True,
                "source_chunk_ids": ["chunk-1"],
            }
        ],
        {"metrics": {"answerability_accuracy": 0.5}},
    )
    assert report["metrics"]["answerability_accuracy"] == 1.0
    assert report["metrics"]["normalized_exact_answer_accuracy"] == 1.0
    assert report["baseline_deltas"]["answerability_accuracy"] == 0.5
