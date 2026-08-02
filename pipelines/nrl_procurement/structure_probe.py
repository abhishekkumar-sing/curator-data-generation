"""Fingerprint and verify structured-output capability for configured roles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from datasets import Dataset
from pydantic import BaseModel, Field

# Importing settings applies the pipeline's local-only Curator controls before
# Curator itself is imported.
from settings import CONFIG  # noqa: F401

from bespokelabs import curator

PROBE_CONTRACT_VERSION = "nrl-structured-output-probe-v1"


class StructureProbeItem(BaseModel):
    """One nested probe item exercising enum and list decoding."""

    label: Literal["alpha", "beta"]
    values: list[int] = Field(min_length=2, max_length=2)


class StructureProbeEnvelope(BaseModel):
    """Nested response contract representative of pipeline containers."""

    status: Literal["probe_ok"]
    items: list[StructureProbeItem] = Field(min_length=2, max_length=2)


class EndpointStructureProbe(curator.LLM):
    """Exercise the configured production structured-output transport."""

    response_format = StructureProbeEnvelope

    def prompt(self, row: dict[str, Any]) -> str:
        """Request exact sentinels so schema validity alone is insufficient."""
        return """Return the structured capability probe exactly as follows:
- status is probe_ok
- items contains exactly two objects in this order
- first item: label alpha and integer values [1, 2]
- second item: label beta and integer values [3, 4]
Do not add, stringify, rename, reorder, or omit any container or value.
"""

    def parse(
        self,
        row: dict[str, Any],
        response: StructureProbeEnvelope,
    ) -> dict[str, Any]:
        """Return only safe semantic checks, never raw provider output."""
        payload = response.model_dump()
        checks = {
            "status": payload["status"] == "probe_ok",
            "labels": [item["label"] for item in payload["items"]]
            == ["alpha", "beta"],
            "nested_integer_lists": [item["values"] for item in payload["items"]]
            == [[1, 2], [3, 4]],
        }
        return {"passed": all(checks.values()), "checks": checks}


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def probe_identity(role: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Return the exact secret-free deployment/transport probe identity."""
    resolved: dict[str, str] = {}
    for field in ("served_model_env", "base_url_env"):
        variable = str(profile[field])
        value = os.environ.get(variable, "").strip()
        if not value:
            raise RuntimeError(
                f"Cannot resolve {role} structure probe; set {variable}"
            )
        resolved[field] = value
    parsed = urlparse(resolved["base_url_env"])
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            f"Cannot persist {role} probe identity: base URL must be a "
            "credential-free absolute HTTP(S) URL"
        )
    generation_params = profile.get("generation_params", {})
    sensitive_keys = {
        str(key)
        for key in _walk_keys(generation_params)
        if any(
            marker in str(key).casefold()
            for marker in ("api_key", "authorization", "credential", "secret")
        )
    }
    if sensitive_keys:
        raise RuntimeError(
            "Structured-output generation parameters must not contain secret "
            f"fields: {', '.join(sorted(sensitive_keys))}"
        )
    deployment_env = str(profile.get("deployment_identity_env", "")).strip()
    deployment_identity = (
        os.environ.get(deployment_env, "").strip() if deployment_env else ""
    )
    return {
        "contract_version": PROBE_CONTRACT_VERSION,
        "role": role,
        "profile": profile.get("profile_name"),
        "served_model": resolved["served_model_env"],
        "base_url": resolved["base_url_env"],
        "deployment_identity": deployment_identity or None,
        "structured_output_mode": profile.get("structured_output_mode", "auto"),
        "dereference_tool_schema": bool(
            profile.get("dereference_tool_schema", False)
        ),
        "generation_params": generation_params,
        "transport_versions": {
            "curator": _package_version("bespokelabs-curator"),
            "instructor": _package_version("instructor"),
            "litellm": _package_version("litellm"),
        },
    }


def _walk_keys(value: Any) -> list[str]:
    """Return nested mapping keys for conservative secret-field rejection."""
    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
        return keys
    if isinstance(value, list):
        return [key for item in value for key in _walk_keys(item)]
    return []


def probe_fingerprint(role: str, profile: dict[str, Any]) -> str:
    """Hash every non-secret input that can change probe semantics."""
    encoded = json.dumps(
        probe_identity(role, profile),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def probe_result_path(
    cache_root: Path,
    role: str,
    profile: dict[str, Any],
) -> Path:
    """Locate the atomic result for the exact role/deployment fingerprint."""
    fingerprint = probe_fingerprint(role, profile)
    return cache_root / "structure_probes" / f"{role}-{fingerprint}.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_probe_result(
    cache_root: Path,
    role: str,
    profile: dict[str, Any],
    *,
    passed: bool,
    checks: dict[str, bool] | None = None,
    failure_class: str | None = None,
    latency_seconds: float | None = None,
) -> dict[str, Any]:
    """Persist a bounded, credential-free probe result atomically."""
    identity = probe_identity(role, profile)
    fingerprint = probe_fingerprint(role, profile)
    result = {
        "contract_version": PROBE_CONTRACT_VERSION,
        "fingerprint": fingerprint,
        "role": role,
        "passed": bool(passed),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": latency_seconds,
        "checks": checks or {},
        "failure_class": failure_class,
        "identity": identity,
    }
    _atomic_json(probe_result_path(cache_root, role, profile), result)
    return result


def run_structure_probe(
    cache_root: Path,
    role: str,
    profile: dict[str, Any],
    llm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Execute one uncached live probe through the production Curator mode."""
    started = time.monotonic()
    try:
        exact_kwargs = deepcopy(llm_kwargs)
        exact_kwargs["backend_params"] = {
            **exact_kwargs.get("backend_params", {}),
            "require_all_responses": True,
        }
        probe = EndpointStructureProbe(**exact_kwargs)
        request_root = cache_root / "structure_probe_requests"
        request_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"{role}-",
            dir=request_root,
        ) as working_dir:
            rows = probe(
                Dataset.from_list([{"role": role}]),
                working_dir=working_dir,
            ).dataset.to_list()
        if len(rows) != 1:
            return record_probe_result(
                cache_root,
                role,
                profile,
                passed=False,
                failure_class="missing_parsed_probe_row",
                latency_seconds=round(time.monotonic() - started, 3),
            )
        checks = {
            str(key): bool(value)
            for key, value in rows[0].get("checks", {}).items()
        }
        return record_probe_result(
            cache_root,
            role,
            profile,
            passed=bool(rows[0].get("passed", False)) and all(checks.values()),
            checks=checks,
            failure_class=(
                None
                if bool(rows[0].get("passed", False)) and all(checks.values())
                else "sentinel_mismatch"
            ),
            latency_seconds=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        return record_probe_result(
            cache_root,
            role,
            profile,
            passed=False,
            failure_class=type(exc).__name__,
            latency_seconds=round(time.monotonic() - started, 3),
        )


def require_successful_structure_probe(
    cache_root: Path,
    role: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the exact deployment fingerprint passed its probe."""
    expected = probe_fingerprint(role, profile)
    path = probe_result_path(cache_root, role, profile)
    if not path.is_file():
        raise SystemExit(
            f"Missing structured-output probe for {role} profile "
            f"{profile.get('profile_name')!r}. Run "
            f"probe_structure.py before a full generation."
        )
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"Invalid structured-output probe result: {path}") from exc
    if (
        result.get("contract_version") != PROBE_CONTRACT_VERSION
        or result.get("fingerprint") != expected
        or result.get("role") != role
        or result.get("passed") is not True
    ):
        raise SystemExit(
            f"Structured-output probe has not passed for the current {role} "
            f"deployment/mode. Re-run the endpoint probe before full generation."
        )
    return result
