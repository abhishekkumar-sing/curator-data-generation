"""Bounded, checkpointed marginal-novelty saturation control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SaturationPolicy:
    """Validated stopping policy for an iterative generation family."""

    enabled: bool = False
    max_passes: int = 1
    minimum_novelty_gain: float = 0.05
    patience: int = 2

    def __post_init__(self) -> None:
        """Reject unbounded or nonsensical stopping settings."""
        if self.max_passes < 1:
            raise ValueError("saturation.max_passes must be at least 1")
        if not 0.0 <= self.minimum_novelty_gain <= 1.0:
            raise ValueError(
                "saturation.minimum_novelty_gain must be between 0 and 1"
            )
        if self.patience < 1:
            raise ValueError("saturation.patience must be at least 1")


def saturation_policy(
    config: dict[str, Any],
    *,
    max_passes_override: int | None = None,
) -> SaturationPolicy:
    """Resolve and validate the configured saturation policy."""
    raw = config.get("saturation", {}) or {}
    configured_enabled = bool(raw.get("enabled", False))
    explicitly_requested = max_passes_override is not None
    maximum = int(
        max_passes_override
        if explicitly_requested
        else raw.get("max_passes", 1) if configured_enabled else 1
    )
    return SaturationPolicy(
        enabled=(configured_enabled or explicitly_requested) and maximum > 1,
        max_passes=maximum,
        minimum_novelty_gain=float(raw.get("minimum_novelty_gain", 0.05)),
        patience=int(raw.get("patience", 2)),
    )


def policy_fingerprint(policy: SaturationPolicy, family: str) -> str:
    """Hash the policy and stage family used by a checkpoint."""
    payload = {"contract_version": 1, "family": family, **asdict(policy)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class SaturationController:
    """Track valid marginal novelty without treating failures as saturation.

    A pass is eligible to advance low-gain patience only when every planned
    parent reaches a successful, schema-valid terminal response. Invalid or
    missing responses remain explicit incomplete observations. The hard pass
    limit always terminates the controller, but an incomplete final state is
    never reported as converged.
    """

    def __init__(
        self,
        policy: SaturationPolicy,
        family: str,
        state_path: Path | None = None,
    ) -> None:
        """Load a matching checkpoint or initialize an empty controller."""
        self.policy = policy
        self.family = family
        self.state_path = state_path
        self.fingerprint = policy_fingerprint(policy, family)
        self.state = self._load()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "contract_version": 1,
            "family": self.family,
            "fingerprint": self.fingerprint,
            "next_pass": 1,
            "low_gain_streak": 0,
            "converged": False,
            "stop_reason": None,
            "observations": [],
        }

    def _load(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.is_file():
            return self._initial_state()
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError(
                "Saturation checkpoint does not match the configured policy/family"
            )
        return payload

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    @property
    def next_pass(self) -> int:
        """Return the next one-based pass index."""
        return int(self.state["next_pass"])

    @property
    def should_continue(self) -> bool:
        """Whether another pass is authorized by the bounded policy."""
        return (
            not self.state["converged"]
            and self.next_pass <= self.policy.max_passes
        )

    def observe(
        self,
        *,
        pass_index: int,
        planned: int,
        successful: int,
        valid: int,
        accepted_novel: int,
    ) -> dict[str, Any]:
        """Persist one terminal pass observation and update stop state."""
        if pass_index != self.next_pass:
            raise ValueError(
                f"Expected saturation pass {self.next_pass}, received {pass_index}"
            )
        values = (planned, successful, valid, accepted_novel)
        if any(value < 0 for value in values):
            raise ValueError("Saturation observation counts cannot be negative")
        if not accepted_novel <= valid <= successful <= planned:
            raise ValueError(
                "Saturation counts must satisfy accepted_novel <= valid <= "
                "successful <= planned"
            )
        eligible = planned > 0 and successful == planned and valid == successful
        gain = accepted_novel / valid if valid else 0.0
        low_gain = eligible and gain < self.policy.minimum_novelty_gain
        if low_gain:
            self.state["low_gain_streak"] += 1
        elif eligible:
            self.state["low_gain_streak"] = 0
        observation = {
            "pass_index": pass_index,
            "planned": planned,
            "successful": successful,
            "valid": valid,
            "accepted_novel": accepted_novel,
            "marginal_novelty_gain": round(gain, 6),
            "eligible_saturation_observation": eligible,
            "low_gain": low_gain,
            "failures": planned - successful,
            "invalid": successful - valid,
        }
        self.state["observations"].append(observation)
        self.state["next_pass"] = pass_index + 1
        if not self.policy.enabled:
            self.state["converged"] = eligible
            self.state["stop_reason"] = "single_pass"
        elif self.state["low_gain_streak"] >= self.policy.patience:
            self.state["converged"] = True
            self.state["stop_reason"] = "marginal_novelty_saturated"
        elif self.state["next_pass"] > self.policy.max_passes:
            self.state["stop_reason"] = "hard_pass_limit"
        self._save()
        return observation

    def summary(self) -> dict[str, Any]:
        """Return a serialization-safe controller summary."""
        return {**self.state, "policy": asdict(self.policy)}
