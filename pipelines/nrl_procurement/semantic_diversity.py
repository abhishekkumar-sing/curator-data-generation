"""Cached embeddings, calibration pairs, and semantic duplicate selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from jsonl_io import write_jsonl_rows
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CALIBRATION_LABELS = frozenset({"duplicate", "related", "distinct"})


@dataclass(frozen=True)
class EmbeddingSettings:
    """Non-secret NVIDIA/OpenAI-compatible embedding configuration."""

    endpoint: str
    model: str
    api_key: str
    input_type: str = "query"
    dimensions: int = 1024
    batch_size: int = 64
    timeout_seconds: float = 120.0
    truncate: str = "NONE"
    top_k: int = 5
    selection_enabled: bool = False
    selection_mode: str = "calibrated_threshold"
    similarity_threshold: float | None = None

    @property
    def fingerprint(self) -> str:
        """Identify vector semantics without credentials."""
        payload = {
            "endpoint": self.endpoint,
            "model": self.model,
            "input_type": self.input_type,
            "dimensions": self.dimensions,
            "truncate": self.truncate,
            "contract": 1,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:20]

    def manifest(self) -> dict[str, Any]:
        """Return safe configuration for run artifacts."""
        return {
            "enabled": True,
            "endpoint": self.endpoint,
            "model": self.model,
            "input_type": self.input_type,
            "dimensions": self.dimensions,
            "truncate": self.truncate,
            "cache_fingerprint": self.fingerprint,
            "selection_enabled": self.selection_enabled,
            "selection_mode": self.selection_mode,
            "similarity_threshold": self.similarity_threshold,
        }


def load_embedding_settings(config: dict[str, Any]) -> EmbeddingSettings | None:
    """Resolve an optional embedding profile from config and environment."""
    section = config.get("embeddings", {})
    if not section.get("enabled", False):
        return None
    api_key_env = str(section.get("api_key_env", "EMBEDDING_API_KEY"))
    endpoint_env = str(section.get("base_url_env", "EMBEDDING_BASE_URL"))
    model_env = str(section.get("model_env", "EMBEDDING_MODEL"))
    missing = [
        name
        for name in (api_key_env, endpoint_env, model_env)
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Embedding generation is enabled but required environment settings "
            f"are missing: {', '.join(missing)}"
        )
    endpoint = os.environ[endpoint_env].strip()
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"{endpoint_env} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise RuntimeError(f"{endpoint_env} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise RuntimeError(
            f"{endpoint_env} must not contain query parameters or a fragment"
        )
    if parsed.scheme != "https" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise RuntimeError("Public embedding endpoints must use HTTPS")
    threshold = section.get("similarity_threshold")
    if threshold is not None:
        threshold = float(threshold)
        if not 0.0 < threshold <= 1.0:
            raise RuntimeError("embeddings.similarity_threshold must be in (0, 1]")
    selection_enabled = bool(section.get("selection_enabled", False))
    selection_mode = str(
        section.get("selection_mode", "calibrated_threshold")
    ).strip()
    if selection_mode not in {
        "calibrated_threshold",
        "verified_equivalence",
        "hybrid_equivalence",
    }:
        raise RuntimeError(
            "embeddings.selection_mode must be calibrated_threshold, "
            "verified_equivalence, or hybrid_equivalence"
        )
    if (
        selection_enabled
        and selection_mode == "calibrated_threshold"
        and threshold is None
    ):
        raise RuntimeError(
            "Semantic selection requires a human-calibrated "
            "embeddings.similarity_threshold"
        )
    if (
        selection_enabled
        and selection_mode == "hybrid_equivalence"
        and threshold is None
    ):
        # hybrid_equivalence still requires an explicit operator-supplied
        # cosine floor via the same `similarity_threshold` field, even though
        # it is a secondary gate on top of the structural signature (not the
        # sole decision boundary calibrated_threshold uses it for) — see
        # `hybrid_equivalence_select`'s docstring. No silent default is
        # chosen in code, matching calibrated_threshold's existing policy.
        raise RuntimeError(
            "hybrid_equivalence selection requires an explicit "
            "embeddings.similarity_threshold cosine floor"
        )
    input_type = str(section.get("input_type", "query")).lower()
    if input_type not in {"query", "passage"}:
        raise RuntimeError("embeddings.input_type must be query or passage")
    truncate = str(section.get("truncate", "NONE")).upper()
    if truncate not in {"NONE", "START", "END"}:
        raise RuntimeError("embeddings.truncate must be NONE, START, or END")
    dimensions = int(section.get("dimensions", 1024))
    if dimensions <= 0:
        raise RuntimeError("embeddings.dimensions must be positive")
    timeout_seconds = float(section.get("timeout_seconds", 120))
    if timeout_seconds <= 0:
        raise RuntimeError("embeddings.timeout_seconds must be positive")
    return EmbeddingSettings(
        endpoint=endpoint,
        model=os.environ[model_env].strip(),
        api_key=os.environ[api_key_env].strip(),
        input_type=input_type,
        dimensions=dimensions,
        batch_size=max(1, int(section.get("batch_size", 64))),
        timeout_seconds=timeout_seconds,
        truncate=truncate,
        top_k=max(1, int(section.get("calibration_neighbors_per_record", 5))),
        selection_enabled=selection_enabled,
        selection_mode=selection_mode,
        similarity_threshold=threshold,
    )


class EmbeddingClient:
    """Small authenticated client for the NVIDIA embedding contract."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the client with bounded retries for transient failures."""
        self.settings = settings
        self.session = session or requests.Session()
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            status=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"POST"}),
        )
        if session is None:
            self.session.mount("https://", HTTPAdapter(max_retries=retry))
            self.session.mount("http://", HTTPAdapter(max_retries=retry))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one ordered batch and validate cardinality and dimensions."""
        if not texts:
            return []
        response = self.session.post(
            self.settings.endpoint,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.model,
                "input": texts,
                "input_type": self.settings.input_type,
                "encoding_format": "float",
                "truncate": self.settings.truncate,
                "dimensions": self.settings.dimensions,
            },
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("Embedding response is missing a data list")
        by_index: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                raise RuntimeError("Embedding response contains an invalid index")
            vector = item.get("embedding")
            if not isinstance(vector, list) or len(vector) != self.settings.dimensions:
                raise RuntimeError(
                    "Embedding response vector has an unexpected dimension"
                )
            numeric = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in numeric):
                raise RuntimeError("Embedding response contains a non-finite value")
            by_index[item["index"]] = numeric
        if set(by_index) != set(range(len(texts))):
            raise RuntimeError("Embedding response cardinality/index mismatch")
        return [by_index[index] for index in range(len(texts))]


def probe_embedding_endpoint(settings: EmbeddingSettings) -> dict[str, Any]:
    """Run one non-sensitive capability probe without persisting its vector."""
    vectors = EmbeddingClient(settings).embed(
        ["Procurement embedding endpoint capability probe."]
    )
    return {
        **settings.manifest(),
        "passed": len(vectors) == 1,
        "returned_dimensions": len(vectors[0]),
    }


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_cache(path: Path) -> dict[tuple[str, str], list[float]]:
    cached: dict[tuple[str, str], list[float]] = {}
    if not path.is_file():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cached[(str(row["record_id"]), str(row["text_sha256"]))] = [
            float(value) for value in row["embedding"]
        ]
    return cached


def _write_cache(
    path: Path,
    cached: dict[tuple[str, str], list[float]],
) -> None:
    rows = [
        {
            "record_id": record_id,
            "text_sha256": text_sha256,
            "embedding": vector,
        }
        for (record_id, text_sha256), vector in sorted(cached.items())
    ]
    temporary = path.with_suffix(".jsonl.tmp")
    write_jsonl_rows(temporary, rows)
    temporary.replace(path)


def embed_records(
    records: list[dict[str, Any]],
    settings: EmbeddingSettings,
    cache_root: Path,
    client: EmbeddingClient | None = None,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Embed record questions with a persistent ID-and-text keyed cache."""
    cache_path = cache_root / settings.fingerprint / "questions.jsonl"
    cached = _read_cache(cache_path)
    result: dict[str, list[float]] = {}
    pending: list[tuple[str, str, str]] = []
    for record in records:
        record_id = str(record["record_id"])
        question = str(record.get("question", "")).strip()
        text_hash = _text_sha256(question)
        vector = cached.get((record_id, text_hash))
        if vector is None or len(vector) != settings.dimensions:
            pending.append((record_id, text_hash, question))
        else:
            result[record_id] = vector
    api = client or EmbeddingClient(settings)
    for start in range(0, len(pending), settings.batch_size):
        batch = pending[start : start + settings.batch_size]
        vectors = api.embed([question for _, _, question in batch])
        for (record_id, text_hash, _), vector in zip(batch, vectors, strict=True):
            cached[(record_id, text_hash)] = vector
            result[record_id] = vector
        _write_cache(cache_path, cached)
    return result, {
        "records": len(records),
        "cache_hits": len(records) - len(pending),
        "api_embeddings": len(pending),
    }


def _normalized_matrix(
    records: list[dict[str, Any]],
    vectors: dict[str, list[float]],
) -> np.ndarray:
    matrix = np.asarray(
        [vectors[str(record["record_id"])] for record in records],
        dtype=np.float32,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Embedding response contains a zero vector")
    return matrix / norms


def calibration_candidates(
    records: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    top_k: int,
    block_size: int = 256,
) -> list[dict[str, Any]]:
    """Return nearest question pairs with an intentionally blank human label."""
    if len(records) < 2:
        return []
    matrix = _normalized_matrix(records, vectors)
    pairs: dict[tuple[int, int], float] = {}
    width = min(top_k, len(records) - 1)
    for start in range(0, len(records), block_size):
        stop = min(start + block_size, len(records))
        similarities = matrix[start:stop] @ matrix.T
        for local_index, row in enumerate(similarities):
            index = start + local_index
            row[index] = -np.inf
            nearest = np.argpartition(row, -width)[-width:]
            for other in nearest:
                left, right = sorted((index, int(other)))
                pairs[(left, right)] = max(
                    pairs.get((left, right), -1.0), float(row[other])
                )
    ordered = sorted(
        pairs.items(),
        key=lambda item: (-item[1], str(records[item[0][0]]["record_id"]), str(records[item[0][1]]["record_id"])),
    )
    return [
        {
            "left_record_id": records[left]["record_id"],
            "right_record_id": records[right]["record_id"],
            "left_question": records[left].get("question", ""),
            "right_question": records[right].get("question", ""),
            "left_question_type": records[left].get("question_type", ""),
            "right_question_type": records[right].get("question_type", ""),
            "left_manual_id": records[left].get("manual_id", ""),
            "right_manual_id": records[right].get("manual_id", ""),
            "cosine_similarity": round(similarity, 6),
            "human_label": None,
            "review_notes": "",
        }
        for (left, right), similarity in ordered
    ]


def calibration_report(
    rows: list[dict[str, Any]],
    *,
    minimum_precision: float = 0.95,
    minimum_labeled_pairs: int = 50,
    minimum_class_pairs: int = 10,
) -> dict[str, Any]:
    """Evaluate reviewed pairs and recommend a conservative dev-set threshold."""
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    labeled: list[tuple[float, bool]] = []
    invalid_labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        raw_label = row.get("human_label")
        if raw_label is None or not str(raw_label).strip():
            continue
        label = str(raw_label).strip().casefold()
        if label not in CALIBRATION_LABELS:
            invalid_labels.append({"row": index, "label": raw_label})
            continue
        score = float(row["cosine_similarity"])
        if not -1.0 <= score <= 1.0 or not math.isfinite(score):
            raise ValueError(f"calibration row {index} has an invalid cosine score")
        labeled.append((score, label == "duplicate"))
    if invalid_labels:
        allowed = ", ".join(sorted(CALIBRATION_LABELS))
        raise ValueError(
            f"invalid human_label values; expected {allowed}: {invalid_labels[:10]}"
        )
    positives = sum(is_duplicate for _, is_duplicate in labeled)
    negatives = len(labeled) - positives
    operating_points: list[dict[str, Any]] = []
    for threshold in sorted({score for score, _ in labeled}, reverse=True):
        true_positive = sum(
            score >= threshold and is_duplicate for score, is_duplicate in labeled
        )
        false_positive = sum(
            score >= threshold and not is_duplicate for score, is_duplicate in labeled
        )
        false_negative = positives - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / positives if positives else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        operating_points.append(
            {
                "threshold": round(threshold, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
            }
        )
    sufficient = (
        len(labeled) >= minimum_labeled_pairs
        and positives >= minimum_class_pairs
        and negatives >= minimum_class_pairs
    )
    eligible = [
        point
        for point in operating_points
        if point["true_positives"] > 0
        and point["precision"] >= minimum_precision
    ]
    recommended = None
    reason = "insufficient_labeled_pairs"
    if sufficient and eligible:
        recommended = max(
            eligible,
            key=lambda point: (
                point["recall"],
                point["precision"],
                point["threshold"],
            ),
        )
        reason = "highest_recall_at_or_above_minimum_precision"
    elif sufficient:
        reason = "no_threshold_meets_minimum_precision"
    return {
        "label_contract": {
            "positive": "duplicate",
            "negative": ["related", "distinct"],
        },
        "labeled_pairs": len(labeled),
        "duplicate_pairs": positives,
        "nonduplicate_pairs": negatives,
        "minimum_precision": minimum_precision,
        "minimum_labeled_pairs": minimum_labeled_pairs,
        "minimum_class_pairs": minimum_class_pairs,
        "sufficient_for_recommendation": sufficient,
        "recommended": recommended,
        "recommendation_reason": reason,
        "operating_points": operating_points,
        "warning": (
            "This is an in-sample development calibration. Validate the selected "
            "threshold on a separate reviewed holdout before enabling deletion."
        ),
    }


def _semantic_components(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    parent = list(range(len(matrix)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    block_size = 256
    for start in range(0, len(matrix), block_size):
        stop = min(start + block_size, len(matrix))
        similarities = matrix[start:stop] @ matrix.T
        for local_index in range(stop - start):
            index = start + local_index
            matches = np.flatnonzero(similarities[local_index, index + 1 :] >= threshold)
            for offset in matches:
                union(index, index + 1 + int(offset))
    grouped: dict[int, list[int]] = {}
    for index in range(len(matrix)):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _quality_key(record: dict[str, Any]) -> tuple[int, int, int, int, str]:
    judge = record.get("judge", {})
    grounded_claims = sum(
        bool(claim.get("evidence")) for claim in record.get("claims", [])
    )
    return (
        int(judge.get("score", 0)),
        int(bool(judge.get("preserves_qualifications", False))),
        grounded_claims,
        len(record.get("evidence", [])),
        # `min(..., key=...)` uses this final value for a stable tie break.
        str(record["record_id"]),
    )


def semantic_select(
    records: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep the strongest accepted record in each calibrated semantic cluster."""
    if len(records) < 2:
        return records, [], {"clusters": len(records), "multi_record_clusters": 0}
    matrix = _normalized_matrix(records, vectors)
    components = _semantic_components(matrix, threshold)
    kept_indexes: set[int] = set()
    removed: list[dict[str, Any]] = []
    for component in components:
        # Descending quality dimensions with ascending record ID tie-breaking.
        ranked = sorted(
            component,
            key=lambda index: (
                -_quality_key(records[index])[0],
                -_quality_key(records[index])[1],
                -_quality_key(records[index])[2],
                -_quality_key(records[index])[3],
                _quality_key(records[index])[4],
            ),
        )
        winner = ranked[0]
        kept_indexes.add(winner)
        for index in ranked[1:]:
            removed.append(
                {
                    **records[index],
                    "semantic_selection": {
                        "accepted": False,
                        "reason": "semantic_duplicate_cluster",
                        "representative_record_id": records[winner]["record_id"],
                        "threshold": threshold,
                    },
                }
            )
    kept = [record for index, record in enumerate(records) if index in kept_indexes]
    return kept, removed, {
        "clusters": len(components),
        "multi_record_clusters": sum(len(component) > 1 for component in components),
        "records_removed": len(removed),
    }


def _normalized_semantic_text(value: Any) -> str:
    """Normalize generated target text for a strict equivalence signature."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _grounded_equivalence_signature(
    record: dict[str, Any],
    *,
    require_answer_match: bool = True,
) -> tuple[Any, ...]:
    """Identify the same grounded training target without using question wording.

    `require_answer_match=True` (the `verified_equivalence` mode) additionally
    requires byte-identical normalized answer text, which is the exact
    boundary the audit found too narrow to catch realistic paraphrase
    duplicates: two independent generations of the same fact almost never
    produce identical answer wording, even when every structural field below
    already matches. `require_answer_match=False` (the `hybrid_equivalence`
    mode) drops that one field from the signature so structurally-identical
    records (same task/persona/intent/coverage) can still group together;
    callers of that mode must additionally clear a cosine-similarity floor
    before deleting anything (see `hybrid_equivalence_select`), so dropping
    the answer-text field alone never authorizes deletion by itself.
    """
    evidence = tuple(
        sorted(
            (
                str(item.get("chunk_id", item.get("source_id", ""))),
                " ".join(str(item.get("quote", "")).split()),
            )
            for item in record.get("evidence", [])
        )
    )
    sources = tuple(sorted(str(item) for item in record.get("source_chunk_ids", [])))
    required = (
        record.get("record_id"),
        record.get("task_type"),
        record.get("task"),
        record.get("persona"),
        record.get("question_type"),
        record.get("answer"),
        evidence,
        sources,
    )
    if not all(required):
        # Defensive fail-closed behavior for malformed or legacy rows: an
        # incomplete signature can never authorize deletion.
        return ("ineligible_for_verified_equivalence", str(record.get("record_id", "")))
    return (
        str(record.get("task_type", "")),
        str(record.get("task", "")),
        str(record.get("persona", "")),
        str(record.get("question_type", "")),
        str(record.get("answer_format", "")),
        bool(record.get("answerable", True)),
        str(record.get("reasoning_operation", "")),
        str(record.get("difficulty", "")),
        str(record.get("material_focus", "")),
        _normalized_semantic_text(record.get("answer", "")) if require_answer_match else None,
        evidence,
        sources,
    )


def _cosine_similarity(
    vectors: dict[str, list[float]],
    left_record_id: str,
    right_record_id: str,
) -> float:
    left = np.asarray(vectors[left_record_id], dtype=float)
    right = np.asarray(vectors[right_record_id], dtype=float)
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if not denominator:
        return 0.0
    return float(np.dot(left, right) / denominator)


def verified_equivalence_select(
    records: list[dict[str, Any]],
    vectors: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Delete only paraphrases of the same grounded training target.

    Dense similarity remains an auditable diagnostic. It is not trusted as the
    deletion boundary: records must share every source, evidence, answer, and
    coverage invariant in `_grounded_equivalence_signature`.
    """
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(_grounded_equivalence_signature(record), []).append(index)
    kept_indexes: set[int] = set()
    removed: list[dict[str, Any]] = []
    multi_record_groups = 0
    for component in grouped.values():
        if len(component) > 1:
            multi_record_groups += 1
        ranked = sorted(
            component,
            key=lambda index: (
                -_quality_key(records[index])[0],
                -_quality_key(records[index])[1],
                -_quality_key(records[index])[2],
                -_quality_key(records[index])[3],
                _quality_key(records[index])[4],
            ),
        )
        winner = ranked[0]
        kept_indexes.add(winner)
        winner_id = str(records[winner]["record_id"])
        for index in ranked[1:]:
            similarity = _cosine_similarity(vectors, winner_id, str(records[index]["record_id"]))
            removed.append(
                {
                    **records[index],
                    "semantic_selection": {
                        "accepted": False,
                        "reason": "verified_grounded_equivalence",
                        "representative_record_id": records[winner]["record_id"],
                        "cosine_similarity": round(similarity, 6),
                        "verification": (
                            "same_task_persona_intent_answer_evidence_operation_"
                            "difficulty_material_focus_and_sources"
                        ),
                    },
                }
            )
    kept = [record for index, record in enumerate(records) if index in kept_indexes]
    return kept, removed, {
        "selection_mode": "verified_equivalence",
        "equivalence_groups": len(grouped),
        "multi_record_groups": multi_record_groups,
        "records_removed": len(removed),
    }


def hybrid_equivalence_select(
    records: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    similarity_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Delete grounded-equivalent paraphrases without requiring verbatim answer text.

    This closes the realistic gap `verified_equivalence` cannot: two
    independent generations of the same fact almost never share byte-identical
    answer wording, so `verified_equivalence`'s all-fields-including-answer
    signature essentially never fires on real paraphrase duplicates (see
    audit T10 / Finding S1).

    The structural fields below (task/persona/intent/coverage — everything
    `verified_equivalence` requires *except* the literal answer text) must
    still match exactly; that boundary is unchanged and unconditionally safe.
    A record is only deleted when it *also* clears an uncalibrated-but-
    conservative cosine-similarity floor on its question embedding against
    the retained representative — so a structural match alone (e.g. two
    genuinely distinct sub-questions that happen to cite the same evidence
    passage) is never sufficient by itself to authorize deletion. This
    mirrors the standard "cosine-threshold semantic dedup" pattern used by
    e.g. MinishLab/semhash and the SemDeDup paper (Abbas et al.), except more
    conservatively: here the cosine floor is a secondary gate on top of an
    already-strict structural match, not the sole decision boundary, because
    no human-labeled calibration set exists yet for this deployment (see
    `TASKS.md`'s open "hand-label a calibration set" item) and the floor
    value is therefore a defensible-but-uncalibrated default, not a measured
    operating point.

    Structural matches that fall short of the floor are kept, not deleted —
    they remain visible to human review through the existing nearest-
    neighbor `semantic_calibration.jsonl` pairs (`calibration_candidates`),
    the same "route uncertain cases to review, never auto-delete on an
    uncalibrated signal alone" pattern this module already uses elsewhere.
    """
    if not 0.0 < similarity_floor <= 1.0:
        raise ValueError("similarity_floor must be in (0, 1]")
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(
            _grounded_equivalence_signature(record, require_answer_match=False),
            [],
        ).append(index)
    kept_indexes: set[int] = set(range(len(records)))
    removed: list[dict[str, Any]] = []
    multi_record_groups = 0
    below_floor_pairs = 0
    for component in grouped.values():
        if len(component) > 1:
            multi_record_groups += 1
        ranked = sorted(
            component,
            key=lambda index: (
                -_quality_key(records[index])[0],
                -_quality_key(records[index])[1],
                -_quality_key(records[index])[2],
                -_quality_key(records[index])[3],
                _quality_key(records[index])[4],
            ),
        )
        winner = ranked[0]
        winner_id = str(records[winner]["record_id"])
        for index in ranked[1:]:
            similarity = _cosine_similarity(vectors, winner_id, str(records[index]["record_id"]))
            if similarity < similarity_floor:
                below_floor_pairs += 1
                continue
            kept_indexes.discard(index)
            removed.append(
                {
                    **records[index],
                    "semantic_selection": {
                        "accepted": False,
                        "reason": "hybrid_grounded_equivalence",
                        "representative_record_id": records[winner]["record_id"],
                        "cosine_similarity": round(similarity, 6),
                        "similarity_floor": similarity_floor,
                        "verification": (
                            "same_task_persona_intent_evidence_operation_"
                            "difficulty_material_focus_and_sources"
                            "_plus_question_similarity_floor"
                        ),
                    },
                }
            )
    kept = [record for index, record in enumerate(records) if index in kept_indexes]
    return kept, removed, {
        "selection_mode": "hybrid_equivalence",
        "equivalence_groups": len(grouped),
        "multi_record_groups": multi_record_groups,
        "records_removed": len(removed),
        "structural_matches_below_similarity_floor": below_floor_pairs,
        "similarity_floor": similarity_floor,
    }


def run_semantic_diversity(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    cache_root: Path,
    client: EmbeddingClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Embed accepted questions, emit calibration pairs, and optionally select."""
    settings = load_embedding_settings(config)
    if settings is None:
        return records, [], [], {"enabled": False}
    api = client or EmbeddingClient(settings)
    probe_vectors = api.embed(
        ["Procurement embedding endpoint capability probe."]
    )
    probe = {
        "passed": len(probe_vectors) == 1,
        "returned_dimensions": len(probe_vectors[0]),
    }
    vectors, cache_stats = embed_records(records, settings, cache_root, api)
    candidates = calibration_candidates(records, vectors, settings.top_k)
    kept = records
    removed: list[dict[str, Any]] = []
    selection_stats: dict[str, Any] = {
        "enabled": False,
        "reason": "selection_not_enabled",
    }
    if settings.selection_enabled:
        if settings.selection_mode == "verified_equivalence":
            kept, removed, selection_stats = verified_equivalence_select(
                records, vectors
            )
        elif settings.selection_mode == "hybrid_equivalence":
            assert settings.similarity_threshold is not None
            kept, removed, selection_stats = hybrid_equivalence_select(
                records,
                vectors,
                settings.similarity_threshold,
            )
        else:
            assert settings.similarity_threshold is not None
            kept, removed, selection_stats = semantic_select(
                records,
                vectors,
                settings.similarity_threshold,
            )
        selection_stats["enabled"] = True
    return kept, removed, candidates, {
        **settings.manifest(),
        "probe": probe,
        "cache": cache_stats,
        "calibration_pairs": len(candidates),
        "selection": selection_stats,
    }
