"""Tests for secret-free structured-output capability probe gating."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

import generate as generation_pipeline  # noqa: E402
from structure_probe import (  # noqa: E402
    EndpointStructureProbe,
    StructureProbeEnvelope,
    StructureProbeItem,
    probe_fingerprint,
    probe_identity,
    record_probe_result,
    require_successful_structure_probe,
)


def _profile(**overrides) -> dict:
    profile = {
        "profile_name": "test-profile",
        "served_model_env": "TEST_PROBE_MODEL",
        "base_url_env": "TEST_PROBE_BASE_URL",
        "api_key_env": "TEST_PROBE_API_KEY",
        "deployment_identity_env": "TEST_PROBE_DEPLOYMENT",
        "structured_output_mode": "tools_auto",
        "dereference_tool_schema": True,
        "generation_params": {"temperature": 1.0, "max_tokens": 256},
    }
    profile.update(overrides)
    return profile


def _environment(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROBE_MODEL", "served/model")
    monkeypatch.setenv("TEST_PROBE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("TEST_PROBE_API_KEY", "do-not-persist")
    monkeypatch.setenv("TEST_PROBE_DEPLOYMENT", "deployment-v1")


def test_probe_parse_checks_exact_nested_sentinels() -> None:
    response = StructureProbeEnvelope(
        status="probe_ok",
        items=[
            StructureProbeItem(label="alpha", values=[1, 2]),
            StructureProbeItem(label="beta", values=[3, 4]),
        ],
    )
    probe = object.__new__(EndpointStructureProbe)
    parsed = EndpointStructureProbe.parse(probe, {}, response)
    assert parsed == {
        "passed": True,
        "checks": {
            "status": True,
            "labels": True,
            "nested_integer_lists": True,
        },
    }


def test_probe_result_is_secret_free_and_satisfies_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _environment(monkeypatch)
    profile = _profile()
    result = record_probe_result(
        tmp_path,
        "generation",
        profile,
        passed=True,
        checks={"status": True, "labels": True, "nested_integer_lists": True},
        latency_seconds=1.25,
    )
    accepted = require_successful_structure_probe(
        tmp_path,
        "generation",
        profile,
    )
    assert accepted == result
    serialized = json.dumps(result)
    assert "do-not-persist" not in serialized
    assert "api_key" not in serialized.casefold()


def test_probe_fingerprint_invalidates_transport_changes(monkeypatch) -> None:
    _environment(monkeypatch)
    initial = probe_fingerprint("generation", _profile())
    changed_mode = probe_fingerprint(
        "generation",
        _profile(structured_output_mode="json_schema"),
    )
    monkeypatch.setenv("TEST_PROBE_BASE_URL", "http://127.0.0.1:9000/v1")
    changed_endpoint = probe_fingerprint("generation", _profile())
    assert len({initial, changed_mode, changed_endpoint}) == 3


def test_failed_probe_does_not_satisfy_gate(tmp_path: Path, monkeypatch) -> None:
    _environment(monkeypatch)
    profile = _profile()
    record_probe_result(
        tmp_path,
        "judge",
        profile,
        passed=False,
        failure_class="ValidationError",
    )
    try:
        require_successful_structure_probe(tmp_path, "judge", profile)
    except SystemExit as exc:
        assert "has not passed" in str(exc)
    else:
        raise AssertionError("failed probe unexpectedly satisfied the gate")


def test_probe_identity_rejects_url_or_parameter_secrets(monkeypatch) -> None:
    _environment(monkeypatch)
    monkeypatch.setenv(
        "TEST_PROBE_BASE_URL",
        "http://127.0.0.1:8000/v1?api_key=leak",
    )
    try:
        probe_identity("generation", _profile())
    except RuntimeError as exc:
        assert "credential-free" in str(exc)
    else:
        raise AssertionError("probe identity accepted a URL credential")

    monkeypatch.setenv("TEST_PROBE_BASE_URL", "http://127.0.0.1:8000/v1")
    try:
        probe_identity(
            "generation",
            _profile(generation_params={"extra_body": {"api_key": "leak"}}),
        )
    except RuntimeError as exc:
        assert "secret fields" in str(exc)
    else:
        raise AssertionError("probe identity accepted a secret generation field")


def test_full_run_requires_both_roles_but_bounded_pilot_does_not(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        generation_pipeline,
        "require_successful_structure_probe",
        lambda cache_root, role, profile: calls.append(role),
    )
    generation_pipeline._require_structure_probes_for_run(
        Namespace(limit=5, skip_judge=False)
    )
    assert calls == []
    generation_pipeline._require_structure_probes_for_run(
        Namespace(limit=None, skip_judge=False)
    )
    assert calls == ["generation", "judge"]

    calls.clear()
    generation_pipeline._require_structure_probes_for_run(
        Namespace(limit=None, skip_judge=True)
    )
    assert calls == ["generation"]
