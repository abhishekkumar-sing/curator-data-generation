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
    verified_equivalence_select,
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
    monkeypatch.setenv("EMBEDDING_CREDENTIAL_ROTATED", "1")
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


def test_verified_equivalence_mode_needs_no_cosine_cutoff(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    monkeypatch.setenv("EMBEDDING_CREDENTIAL_ROTATED", "1")
    settings = load_embedding_settings(
        {
            "embeddings": {
                "enabled": True,
                "selection_enabled": True,
                "selection_mode": "verified_equivalence",
                "dimensions": 2,
            }
        }
    )
    assert settings is not None
    assert settings.selection_mode == "verified_equivalence"
    assert settings.similarity_threshold is None


def test_verified_equivalence_removes_only_same_grounded_target(
    tmp_path: Path, monkeypatch
) -> None:
    base = {
        "task_type": "qa",
        "task": "compliance_and_audit",
        "persona": "auditor",
        "question_type": "compliance_check",
        "answer_format": "audit_check",
        "answerable": True,
        "reasoning_operation": "lookup",
        "difficulty": "basic",
        "material_focus": "evidence_requirement",
        "answer": "Retain the approved procurement record.",
        "evidence": [{"chunk_id": "chunk-1", "quote": "Retain the approved procurement record."}],
        "source_chunk_ids": ["chunk-1"],
        "claims": [{"evidence": [{"quote": "Retain the approved procurement record."}]}],
    }
    records = [
        {
            **base,
            "record_id": "weak",
            "question": "Which approved procurement record must be retained?",
            "judge": {"score": 4, "preserves_qualifications": True},
        },
        {
            **base,
            "record_id": "strong",
            "question": "What approved procurement record must the buyer keep?",
            "judge": {"score": 5, "preserves_qualifications": True},
        },
        {
            **base,
            "record_id": "distinct",
            "question": "When does the retention exception apply?",
            "material_focus": "exception",
            "judge": {"score": 5, "preserves_qualifications": True},
        },
    ]
    vectors = {
        "weak": [1.0, 0.0],
        "strong": [0.99, 0.01],
        "distinct": [0.98, 0.02],
    }
    kept, removed, stats = verified_equivalence_select(records, vectors)
    assert {row["record_id"] for row in kept} == {"strong", "distinct"}
    assert [row["record_id"] for row in removed] == ["weak"]
    assert removed[0]["semantic_selection"]["reason"] == "verified_grounded_equivalence"
    assert stats["records_removed"] == 1
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    monkeypatch.setenv("EMBEDDING_CREDENTIAL_ROTATED", "1")
    run_kept, run_removed, _candidates, run_stats = run_semantic_diversity(
        records,
        {
            "embeddings": {
                "enabled": True,
                "selection_enabled": True,
                "selection_mode": "verified_equivalence",
                "dimensions": 2,
                "batch_size": 3,
                "calibration_neighbors_per_record": 1,
            }
        },
        tmp_path,
        _FakeClient(),
    )
    assert {row["record_id"] for row in run_kept} == {"strong", "distinct"}
    assert [row["record_id"] for row in run_removed] == ["weak"]
    assert run_stats["selection"]["selection_mode"] == "verified_equivalence"


def test_embedding_endpoint_rejects_url_credentials(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv(
        "EMBEDDING_BASE_URL",
        "https://example.invalid/embeddings?api_key=leak",
    )
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    monkeypatch.setenv("EMBEDDING_CREDENTIAL_ROTATED", "1")
    try:
        load_embedding_settings({"embeddings": {"enabled": True}})
    except RuntimeError as exc:
        assert "query parameters" in str(exc)
    else:
        raise AssertionError("embedding endpoint unexpectedly accepted a URL secret")


def test_embedding_requires_credential_rotation_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    monkeypatch.delenv("EMBEDDING_CREDENTIAL_ROTATED", raising=False)
    try:
        load_embedding_settings({"embeddings": {"enabled": True}})
    except RuntimeError as exc:
        assert "confirmed rotated" in str(exc)
        assert "EMBEDDING_CREDENTIAL_ROTATED" in str(exc)
    else:
        raise AssertionError(
            "embedding settings unexpectedly loaded without rotation confirmation"
        )


def test_embedding_credential_rotation_confirmation_accepts_various_true_values(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    monkeypatch.setenv("EMBEDDING_CREDENTIAL_ROTATED", "true")
    settings = load_embedding_settings({"embeddings": {"enabled": True, "dimensions": 2}})
    assert settings is not None


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
    monkeypatch.setenv("EMBEDDING_CREDENTIAL_ROTATED", "1")
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
