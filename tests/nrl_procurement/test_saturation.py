"""Regression tests for per-parent, failure-aware saturation control."""

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


def test_zero_means_unlimited_and_negative_is_invalid() -> None:
    policy = saturation_policy(
        {"saturation": {"enabled": False, "max_passes": 1}},
        max_passes_override=0,
    )
    assert policy.enabled
    assert policy.unlimited
    assert policy.max_passes == 0
    with pytest.raises(ValueError, match="zero or greater"):
        SaturationPolicy(max_passes=-1)


def test_disabled_configuration_remains_single_pass_without_override() -> None:
    disabled = saturation_policy({"saturation": {"enabled": False, "max_passes": 9}})
    assert disabled.enabled is False
    assert disabled.max_passes == 1
    configured = saturation_policy(
        {
            "saturation": {
                "enabled": True,
                "max_passes": 3,
                "empty_passes_required": 3,
            }
        }
    )
    assert configured.enabled is True
    assert configured.max_passes == 3
    assert configured.empty_passes_required == 3


def test_parents_saturate_independently_and_novelty_resets_streak() -> None:
    controller = SaturationController(
        SaturationPolicy(enabled=True, max_passes=0, empty_passes_required=2),
        "cross_document",
    )
    controller.register_parents({"a", "b"})
    controller.observe_parents(
        pass_index=1,
        outcomes={"a": "novel", "b": "empty"},
        novel_record_ids={"a": ["record-a-1"]},
    )
    assert controller.active_parent_ids == ["a", "b"]
    controller.observe_parents(
        pass_index=2,
        outcomes={"a": "empty", "b": "empty"},
    )
    assert controller.active_parent_ids == ["a"]
    assert controller.should_continue
    controller.observe_parents(pass_index=3, outcomes={"a": "empty"})
    assert not controller.should_continue
    assert controller.state["converged"]
    assert controller.state["stop_reason"] == "per_parent_zero_novelty"


def test_failures_are_quarantined_not_saturation_evidence() -> None:
    controller = SaturationController(
        SaturationPolicy(enabled=True, max_passes=0),
        "cross_document",
    )
    controller.register_parents({"failed", "overflow"})
    observation = controller.observe_parents(
        pass_index=1,
        outcomes={
            "failed": "generation_failed",
            "overflow": "prompt_overflow",
        },
    )
    assert observation["eligible_saturation_observations"] == 0
    assert not controller.state["converged"]
    assert controller.state["incomplete"]
    assert controller.state["stop_reason"] == "incomplete_quarantined_parents"
    assert controller.quarantined_parent_ids == ["failed", "overflow"]


def test_transient_failure_reactivates_on_resume_but_overflow_does_not(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.json"
    policy = SaturationPolicy(enabled=True, max_passes=0)
    first = SaturationController(policy, "cross_document", state)
    first.register_parents({"failed", "overflow"})
    first.observe_parents(
        pass_index=1,
        outcomes={
            "failed": "validation_failed",
            "overflow": "prompt_overflow",
        },
    )

    resumed = SaturationController(policy, "cross_document", state)
    assert resumed.active_parent_ids == ["failed"]
    assert resumed.quarantined_parent_ids == ["overflow"]
    assert resumed.should_continue
    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["reactivations"][-1]["parent_ids"] == ["failed"]


def test_checkpoint_replay_and_population_fingerprint(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    policy = SaturationPolicy(enabled=True, max_passes=3)
    first = SaturationController(policy, "cross_document", state)
    first.register_parents({"a", "b"})
    outcomes = {"a": "novel", "b": "empty"}
    first.observe_parents(
        pass_index=1,
        outcomes=outcomes,
        novel_record_ids={"a": ["record-a-1"]},
    )

    resumed = SaturationController(policy, "cross_document", state)
    resumed.register_parents({"a", "b"})
    assert resumed.next_pass == 2
    assert resumed.parent_ids_for_pass(1) == ["a", "b"]
    assert resumed.outcomes_for_pass(1) == outcomes
    assert resumed.novel_record_ids_for_pass(1) == {"a": ["record-a-1"]}
    with pytest.raises(ValueError, match="population"):
        resumed.register_parents({"a", "different"})
    with pytest.raises(ValueError, match="does not match"):
        SaturationController(policy, "qa", state)


def test_hard_limit_is_explicitly_incomplete() -> None:
    controller = SaturationController(
        SaturationPolicy(enabled=True, max_passes=1, empty_passes_required=2),
        "cross_document",
    )
    controller.register_parents({"a"})
    controller.observe_parents(
        pass_index=1,
        outcomes={"a": "novel"},
        novel_record_ids={"a": ["record-a-1"]},
    )
    assert not controller.should_continue
    assert not controller.state["converged"]
    assert controller.state["incomplete"]
    assert controller.state["stop_reason"] == "hard_pass_limit"


def test_disabled_policy_executes_exactly_one_pass() -> None:
    controller = SaturationController(SaturationPolicy(), "cross_document")
    controller.register_parents({"a", "b"})
    controller.observe_parents(
        pass_index=1,
        outcomes={"a": "novel", "b": "empty"},
        novel_record_ids={"a": ["record-a-1"]},
    )
    assert controller.state["converged"]
    assert controller.state["stop_reason"] == "single_pass"
    assert not controller.should_continue
