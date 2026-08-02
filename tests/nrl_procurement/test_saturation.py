"""Regression tests for bounded, failure-aware saturation control."""

import json
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from saturation import (  # noqa: E402
    SaturationController,
    SaturationPolicy,
    saturation_policy,
)


def test_policy_is_bounded_and_override_enables_passes() -> None:
    policy = saturation_policy(
        {"saturation": {"enabled": False, "max_passes": 1}},
        max_passes_override=3,
    )
    assert policy.enabled
    assert policy.max_passes == 3
    with pytest.raises(ValueError, match="at least 1"):
        SaturationPolicy(max_passes=0)


def test_disabled_configuration_remains_single_pass_without_override() -> None:
    disabled = saturation_policy(
        {"saturation": {"enabled": False, "max_passes": 9}}
    )
    assert disabled.enabled is False
    assert disabled.max_passes == 1
    configured = saturation_policy(
        {"saturation": {"enabled": True, "max_passes": 3}}
    )
    assert configured.enabled is True
    assert configured.max_passes == 3


def test_only_complete_valid_passes_advance_patience(tmp_path: Path) -> None:
    controller = SaturationController(
        SaturationPolicy(
            enabled=True,
            max_passes=4,
            minimum_novelty_gain=0.2,
            patience=2,
        ),
        "qa",
        tmp_path / "state.json",
    )
    incomplete = controller.observe(
        pass_index=1,
        planned=10,
        successful=9,
        valid=9,
        accepted_novel=0,
    )
    assert not incomplete["eligible_saturation_observation"]
    assert controller.state["low_gain_streak"] == 0

    controller.observe(
        pass_index=2,
        planned=10,
        successful=10,
        valid=10,
        accepted_novel=1,
    )
    controller.observe(
        pass_index=3,
        planned=10,
        successful=10,
        valid=10,
        accepted_novel=0,
    )
    assert controller.state["converged"]
    assert controller.state["stop_reason"] == "marginal_novelty_saturated"


def test_checkpoint_resume_and_fingerprint_rejection(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    policy = SaturationPolicy(enabled=True, max_passes=2)
    first = SaturationController(policy, "qa", state)
    first.observe(
        pass_index=1,
        planned=2,
        successful=2,
        valid=2,
        accepted_novel=2,
    )
    resumed = SaturationController(policy, "qa", state)
    assert resumed.next_pass == 2
    assert json.loads(state.read_text(encoding="utf-8"))["next_pass"] == 2
    with pytest.raises(ValueError, match="does not match"):
        SaturationController(policy, "cross_document", state)


def test_hard_limit_does_not_claim_convergence() -> None:
    controller = SaturationController(
        SaturationPolicy(enabled=True, max_passes=1, patience=2),
        "qa",
    )
    controller.observe(
        pass_index=1,
        planned=2,
        successful=1,
        valid=1,
        accepted_novel=1,
    )
    assert not controller.should_continue
    assert not controller.state["converged"]
    assert controller.state["stop_reason"] == "hard_pass_limit"
