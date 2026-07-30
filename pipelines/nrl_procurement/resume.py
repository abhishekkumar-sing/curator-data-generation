"""Fingerprint-aware logical stage checkpoints for resumable pipeline runs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from jsonl_io import write_jsonl_rows

RESUME_SCHEMA_VERSION = "nrl-resume-v1"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pipeline_source_fingerprint(pipeline_dir: Path) -> str:
    """Hash relevant Python sources so helper-validator changes invalidate work."""
    files = sorted(pipeline_dir.glob("*.py"))
    payload = [
        {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]
    return _canonical_hash(payload)


def semantic_model_identity(profile: dict[str, Any]) -> dict[str, Any]:
    """Return secret-free model semantics, separating deployment from transport."""
    model = os.environ.get(str(profile["served_model_env"]), "").strip()
    base_url = os.environ.get(str(profile["base_url_env"]), "").strip()
    identity_env = str(profile.get("deployment_identity_env", "")).strip()
    deployment_identity = os.environ.get(identity_env, "").strip() if identity_env else ""
    semantic = (
        {"deployment_identity": deployment_identity, "model": model}
        if deployment_identity
        else {"model": model, "base_url": base_url}
    )
    return {
        "profile": profile.get("profile_name"),
        "semantic": semantic,
        "deployment_identity_env": identity_env or None,
        "credential_env": profile.get("api_key_env"),
        "structured_output_mode": profile.get("structured_output_mode", "auto"),
        "generation_params": profile.get("generation_params", {}),
        "request_timeout": profile.get("request_timeout"),
        "max_retries": profile.get("max_retries"),
        "max_concurrent_requests": profile.get("max_concurrent_requests"),
        "max_requests_per_minute": profile.get("max_requests_per_minute"),
        "max_tokens_per_minute": profile.get("max_tokens_per_minute"),
    }


def _checkpoint_input(value: Any) -> Any:
    """Remove invocation-local cache labels from logical scientific inputs."""
    if isinstance(value, list):
        return [_checkpoint_input(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _checkpoint_input(item)
            for key, item in value.items()
            if key not in {"proposition_cache_fingerprint", "prompt_budget"}
        }
    return value


class ResumeManager:
    """Manage one logical run across repeated, fingerprinted attempts."""

    def __init__(
        self,
        *,
        run_id: str,
        output_root: Path,
        cache_root: Path,
        config: dict[str, Any],
        pipeline_dir: Path,
        generation_profile: dict[str, Any],
        judge_profile: dict[str, Any],
        refresh_stages: set[str] | None = None,
    ) -> None:
        """Capture immutable inputs needed to manage repeated attempts."""
        self.run_id = run_id
        self.run_root = output_root / run_id
        self.files_dir = self.run_root / "files"
        self.cache_root = cache_root
        self.config_hash = _canonical_hash(config)
        self.source_hash = pipeline_source_fingerprint(pipeline_dir)
        self.model_identities = {
            "generation": semantic_model_identity(generation_profile),
            "judge": semantic_model_identity(judge_profile),
        }
        self.refresh_stages = refresh_stages or set()
        self.stage_events: dict[str, dict[str, Any]] = {}
        self.state_path = self.run_root / "run_state.json"
        self.attempt_id = ""

    def start(self) -> dict[str, Any]:
        """Start or resume one attempt and preserve a prior terminal manifest."""
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.is_file()
            else {
                "schema_version": RESUME_SCHEMA_VERSION,
                "run_id": self.run_id,
                "attempts": [],
            }
        )
        if state.get("run_id") != self.run_id:
            raise RuntimeError("Run-state identity does not match requested run ID")
        manifest = self.files_dir / "manifest.json"
        if manifest.is_file():
            history = self.run_root / "resume_history"
            history.mkdir(parents=True, exist_ok=True)
            prior = json.loads(manifest.read_text(encoding="utf-8"))
            sequence = len(state.get("attempts", []))
            _atomic_json(history / f"manifest-{sequence:04d}.json", prior)
        now = datetime.now(timezone.utc).isoformat()
        self.attempt_id = f"attempt-{len(state.get('attempts', [])):04d}"
        state.setdefault("attempts", []).append(
            {
                "attempt_id": self.attempt_id,
                "started_at": now,
                "status": "running",
                "generation_identity": self.model_identities["generation"],
                "judge_identity": self.model_identities["judge"],
            }
        )
        state["current_attempt_id"] = self.attempt_id
        state["status"] = "running"
        _atomic_json(self.state_path, state)
        return {
            "run_id": self.run_id,
            "status": "running",
            "attempt_id": self.attempt_id,
            "resume": {
                "schema_version": RESUME_SCHEMA_VERSION,
                "attempt_number": len(state["attempts"]),
                "previous_attempts": len(state["attempts"]) - 1,
            },
        }

    def _contract_hash(self, stage: str) -> str:
        return _canonical_hash(
            {
                "schema_version": RESUME_SCHEMA_VERSION,
                "stage": stage,
                "pipeline_source_sha256": self.source_hash,
                "config_sha256": self.config_hash,
            }
        )

    def _stage_fingerprint(self, stage: str, role: str) -> str:
        return _canonical_hash(
            {
                "schema_version": RESUME_SCHEMA_VERSION,
                "stage": stage,
                "role": role,
                "model": self.model_identities[role],
                "contract_sha256": self._contract_hash(stage),
            }
        )

    def execute_llm_stage(
        self,
        *,
        stage: str,
        role: str,
        llm: Any,
        inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reuse a completed logical checkpoint or execute one fingerprinted stage."""
        logical_input_hash = _canonical_hash(_checkpoint_input(inputs))
        contract_hash = self._contract_hash(stage)
        checkpoint_key = _canonical_hash(
            {
                "stage": stage,
                "input_sha256": logical_input_hash,
                "contract_sha256": contract_hash,
            }
        )
        checkpoint_dir = self.run_root / "checkpoints" / stage / checkpoint_key
        data_path = checkpoint_dir / "records.jsonl"
        metadata_path = checkpoint_dir / "metadata.json"
        if (
            stage not in self.refresh_stages
            and data_path.is_file()
            and metadata_path.is_file()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("status") == "complete"
                and metadata.get("input_sha256") == logical_input_hash
                and metadata.get("contract_sha256") == contract_hash
            ):
                rows = _read_jsonl(data_path)
                self.stage_events[stage] = {
                    "status": "reused_checkpoint",
                    "checkpoint_key": checkpoint_key,
                    "producer": metadata.get("producer"),
                    "records": len(rows),
                }
                return rows

        stage_fingerprint = self._stage_fingerprint(stage, role)
        working_dir = self.cache_root / self.run_id / stage / stage_fingerprint
        cache_existed = working_dir.is_dir() and any(working_dir.iterdir())
        working_dir.mkdir(parents=True, exist_ok=True)
        rows = llm(
            Dataset.from_list(inputs),
            working_dir=str(working_dir),
        ).dataset.to_list()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if data_path.is_file() or metadata_path.is_file():
            history = checkpoint_dir / "history" / self.attempt_id
            history.mkdir(parents=True, exist_ok=True)
            if data_path.is_file():
                data_path.replace(history / data_path.name)
            if metadata_path.is_file():
                metadata_path.replace(history / metadata_path.name)
        temporary_data = checkpoint_dir / "records.jsonl.tmp"
        write_jsonl_rows(temporary_data, rows)
        temporary_data.replace(data_path)
        metadata = {
            "schema_version": RESUME_SCHEMA_VERSION,
            "status": "complete",
            "stage": stage,
            "input_sha256": logical_input_hash,
            "contract_sha256": contract_hash,
            "output_sha256": _canonical_hash(rows),
            "records": len(rows),
            "producer": {
                "attempt_id": self.attempt_id,
                "role": role,
                "model_identity": self.model_identities[role],
                "stage_fingerprint": stage_fingerprint,
            },
        }
        _atomic_json(metadata_path, metadata)
        self.stage_events[stage] = {
            "status": "resumed_partial_cache" if cache_existed else "executed",
            "checkpoint_key": checkpoint_key,
            "producer": metadata["producer"],
            "records": len(rows),
        }
        return rows

    def summary(self) -> dict[str, Any]:
        """Return secret-free attempt and stage provenance for the manifest."""
        return {
            "schema_version": RESUME_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "stage_events": self.stage_events,
        }

    def finish(self, status: str) -> None:
        """Atomically mark the current attempt terminal in run state."""
        if not self.state_path.is_file():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        for attempt in state.get("attempts", []):
            if attempt.get("attempt_id") == self.attempt_id:
                attempt["status"] = status
                attempt["ended_at"] = datetime.now(timezone.utc).isoformat()
                attempt["stages"] = self.stage_events
                break
        state["status"] = status
        _atomic_json(self.state_path, state)
