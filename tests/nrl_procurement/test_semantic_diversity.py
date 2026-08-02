"""Tests for cached semantic-diversity analysis and calibrated selection."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from semantic_diversity import (  # noqa: E402
    EmbeddingClient,
    EmbeddingSettings,
    calibration_candidates,
    calibration_report,
    embed_records,
    load_embedding_settings,
    run_semantic_diversity,
    semantic_select,
)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [
            [1.0, 0.0] if "other" not in text.casefold() else [0.0, 1.0]
            for text in texts
        ]


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }


class _FakeSession:
    def post(self, *args, **kwargs) -> _FakeResponse:
        assert kwargs["json"]["input_type"] == "query"
        assert kwargs["json"]["truncate"] == "NONE"
        assert kwargs["headers"]["Authorization"].startswith("Bearer ")
        return _FakeResponse()


def _settings(**overrides) -> EmbeddingSettings:
    values = {
        "endpoint": "https://example.invalid/v1/embeddings",
        "model": "embedding-model",
        "api_key": "secret",
        "dimensions": 2,
        "batch_size": 2,
        "top_k": 1,
    }
    values.update(overrides)
    return EmbeddingSettings(**values)


def test_embedding_client_preserves_response_index_order() -> None:
    vectors = EmbeddingClient(_settings(), session=_FakeSession()).embed(
        ["first", "second"]
    )
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_record_embedding_cache_is_id_and_text_keyed(tmp_path: Path) -> None:
    records = [
        {"record_id": "one", "question": "Same question"},
        {"record_id": "two", "question": "Other question"},
    ]
    client = _FakeClient()
    first, first_stats = embed_records(records, _settings(), tmp_path, client)
    second, second_stats = embed_records(records, _settings(), tmp_path, client)
    assert set(first) == set(second) == {"one", "two"}
    assert first_stats == {"records": 2, "cache_hits": 0, "api_embeddings": 2}
    assert second_stats == {"records": 2, "cache_hits": 2, "api_embeddings": 0}
    assert len(client.calls) == 1
    cache_text = next(tmp_path.rglob("questions.jsonl")).read_text(
        encoding="utf-8"
    )
    assert "secret" not in cache_text


def test_calibration_candidates_and_quality_selection() -> None:
    records = [
        {
            "record_id": "low",
            "question": "What record is retained?",
            "question_type": "direct_fact",
            "manual_id": "manual",
            "judge": {"score": 4, "preserves_qualifications": True},
            "claims": [{"evidence": [{"quote": "rule"}]}],
            "evidence": [{"quote": "rule"}],
        },
        {
            "record_id": "high",
            "question": "Which record must be retained?",
            "question_type": "direct_fact",
            "manual_id": "manual",
            "judge": {"score": 5, "preserves_qualifications": True},
            "claims": [{"evidence": [{"quote": "rule"}]}],
            "evidence": [{"quote": "rule"}],
        },
        {
            "record_id": "other",
            "question": "What threshold applies?",
            "question_type": "threshold",
            "manual_id": "manual",
            "judge": {"score": 5, "preserves_qualifications": True},
            "claims": [{"evidence": [{"quote": "threshold"}]}],
            "evidence": [{"quote": "threshold"}],
        },
    ]
    vectors = {
        "low": [1.0, 0.0],
        "high": [0.999, 0.001],
        "other": [0.0, 1.0],
    }
    candidates = calibration_candidates(records, vectors, top_k=1)
    assert candidates[0]["human_label"] is None
    assert {candidates[0]["left_record_id"], candidates[0]["right_record_id"]} == {
        "low",
        "high",
    }
    kept, removed, stats = semantic_select(records, vectors, threshold=0.99)
    assert {row["record_id"] for row in kept} == {"high", "other"}
    assert [row["record_id"] for row in removed] == ["low"]
    assert stats["records_removed"] == 1


def test_selection_requires_calibrated_threshold(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    config = {
        "embeddings": {
            "enabled": True,
            "selection_enabled": True,
            "dimensions": 2,
        }
    }
    try:
        load_embedding_settings(config)
    except RuntimeError as exc:
        assert "human-calibrated" in str(exc)
    else:
        raise AssertionError("selection unexpectedly accepted a missing threshold")


def test_embedding_endpoint_rejects_url_credentials(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv(
        "EMBEDDING_BASE_URL",
        "https://example.invalid/embeddings?api_key=leak",
    )
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    try:
        load_embedding_settings({"embeddings": {"enabled": True}})
    except RuntimeError as exc:
        assert "query parameters" in str(exc)
    else:
        raise AssertionError("embedding endpoint unexpectedly accepted a URL secret")


def test_calibration_report_requires_enough_both_class_labels() -> None:
    rows = [
        {"cosine_similarity": 0.99, "human_label": "duplicate"},
        {"cosine_similarity": 0.98, "human_label": "duplicate"},
        {"cosine_similarity": 0.90, "human_label": "related"},
        {"cosine_similarity": 0.20, "human_label": "distinct"},
    ]
    report = calibration_report(
        rows,
        minimum_precision=0.95,
        minimum_labeled_pairs=4,
        minimum_class_pairs=2,
    )
    assert report["sufficient_for_recommendation"] is True
    assert report["recommended"]["threshold"] == 0.98
    assert report["recommended"]["precision"] == 1.0


def test_calibration_report_does_not_recommend_from_tiny_sample() -> None:
    report = calibration_report(
        [{"cosine_similarity": 0.99, "human_label": "duplicate"}]
    )
    assert report["recommended"] is None
    assert report["recommendation_reason"] == "insufficient_labeled_pairs"


def test_run_emits_probe_and_calibration_without_deleting(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    records = [
        {"record_id": "one", "question": "Same question"},
        {"record_id": "two", "question": "Other question"},
    ]
    config = {
        "embeddings": {
            "enabled": True,
            "dimensions": 2,
            "batch_size": 2,
            "calibration_neighbors_per_record": 1,
            "selection_enabled": False,
        }
    }
    kept, removed, candidates, stats = run_semantic_diversity(
        records,
        config,
        tmp_path,
        _FakeClient(),
    )
    assert kept == records
    assert removed == []
    assert candidates
    assert stats["probe"] == {"passed": True, "returned_dimensions": 2}
    assert stats["selection"]["enabled"] is False
    assert "api_key" not in str(stats).casefold()
