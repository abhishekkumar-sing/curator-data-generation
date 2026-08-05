"""Per-parent, checkpointed novelty saturation control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ParentOutcome = Literal[
    "novel",
    "empty",
    "generation_failed",
    "validation_failed",
    "prompt_overflow",
]
_FAILURE_OUTCOMES = {
    "generation_failed",
    "validation_failed",
    "prompt_overflow",
}
_REACTIVATABLE_FAILURES = {"generation_failed", "validation_failed"}


@dataclass(frozen=True)
class SaturationPolicy:
    """Validated stopping policy for an iterative generation family.

    ``max_passes=0`` means that there is no numeric pass cap. Such a run still
    has a mandatory convergence rule: every parent must have the configured
    number of consecutive successful passes that yield no accepted novelty.
    """

    enabled: bool = False
    max_passes: int = 1
    empty_passes_required: int = 2

    def __post_init__(self) -> None:
        """Reject negative limits and empty convergence windows."""
        if self.max_passes < 0:
            raise ValueError("saturation.max_passes must be zero or greater")
        if self.empty_passes_required < 1:
            raise ValueError("saturation.empty_passes_required must be at least 1")

    @property
    def unlimited(self) -> bool:
        """Whether convergence, rather than a number, terminates the run."""
        return self.enabled and self.max_passes == 0


def saturation_policy(
    config: dict[str, Any],
    *,
    max_passes_override: int | None = None,
) -> SaturationPolicy:
    """Resolve CLI/config precedence and validate saturation settings."""
    raw = config.get("saturation", {}) or {}
    configured_enabled = bool(raw.get("enabled", False))
    explicitly_requested = max_passes_override is not None
    maximum = int(max_passes_override if explicitly_requested else raw.get("max_passes", 1) if configured_enabled else 1)
    # Keep the previous ``patience`` key as a configuration compatibility
    # alias, but expose the precise per-parent meaning in the policy/manifest.
    empty_passes = int(raw.get("empty_passes_required", raw.get("patience", 2)))
    return SaturationPolicy(
        enabled=(configured_enabled or explicitly_requested) and (maximum == 0 or maximum > 1),
        max_passes=maximum,
        empty_passes_required=empty_passes,
    )


def policy_fingerprint(policy: SaturationPolicy, family: str) -> str:
    """Hash the policy and stage family used by a checkpoint."""
    payload = {"contract_version": 2, "family": family, **asdict(policy)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class SaturationController:
    """Track consecutive zero-novelty evidence independently per parent.

    Only a successful, schema-valid terminal pass may advance a parent's empty
    streak. Missing, malformed, and prompt-overflow outcomes are quarantined,
    leave the run explicitly incomplete, and never count as saturation.
    Transient endpoint/schema failures are reactivated on a later invocation;
    deterministic prompt overflow remains quarantined until inputs/configuration
    change (which should use a new compatible checkpoint).
    """

    def __init__(
        self,
        policy: SaturationPolicy,
        family: str,
        state_path: Path | None = None,
    ) -> None:
        """Load a compatible checkpoint and reactivate transient failures."""
        self.policy = policy
        self.family = family
        self.state_path = state_path
        self.fingerprint = policy_fingerprint(policy, family)
        self.state, loaded = self._load()
        self._reactivated_parent_ids: list[str] = []
        if loaded:
            self._reactivate_transient_failures()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "contract_version": 2,
            "family": self.family,
            "fingerprint": self.fingerprint,
            "next_pass": 1,
            "converged": False,
            "incomplete": False,
            "stop_reason": None,
            "parents": {},
            "observations": [],
            "reactivations": [],
        }

    def _load(self) -> tuple[dict[str, Any], bool]:
        if self.state_path is None or not self.state_path.is_file():
            return self._initial_state(), False
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError("Saturation checkpoint does not match the configured policy/family")
        return payload, True

    def _reactivate_transient_failures(self) -> None:
        reactivated = []
        for parent_id, parent in self.state["parents"].items():
            if parent.get("status") == "quarantined" and parent.get("failure_reason") in _REACTIVATABLE_FAILURES:
                parent["status"] = "active"
                parent["failure_reason"] = None
                reactivated.append(parent_id)
        if reactivated:
            self.state["converged"] = False
            self.state["incomplete"] = False
            self.state["stop_reason"] = None
            self.state["reactivations"].append(
                {
                    "before_pass": self.next_pass,
                    "parent_ids": sorted(reactivated),
                }
            )
            self._reactivated_parent_ids = sorted(reactivated)
            self._save()

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

    def register_parents(self, parent_ids: list[str] | set[str]) -> None:
        """Bind the checkpoint to one exact, stable planning population."""
        normalized = sorted({str(parent_id) for parent_id in parent_ids})
        existing = sorted(self.state["parents"])
        if existing and existing != normalized:
            raise ValueError("Saturation checkpoint parent population does not match the " "current planning population")
        if not existing:
            self.state["parents"] = {
                parent_id: {
                    "status": "active",
                    "empty_passes": 0,
                    "failure_reason": None,
                    "last_pass": None,
                }
                for parent_id in normalized
            }
            if not normalized:
                self.state["converged"] = True
                self.state["stop_reason"] = "planning_space_exhausted"
            self._save()

    @property
    def next_pass(self) -> int:
        """Return the next one-based pass that has not been observed."""
        return int(self.state["next_pass"])

    @property
    def active_parent_ids(self) -> list[str]:
        """Return stable parent IDs eligible for the next pass."""
        return sorted(parent_id for parent_id, parent in self.state["parents"].items() if parent.get("status") == "active")

    @property
    def should_continue(self) -> bool:
        """Whether active work remains within the optional numeric limit."""
        within_limit = self.policy.max_passes == 0 or self.next_pass <= self.policy.max_passes
        return bool(not self.state["converged"] and self.active_parent_ids and within_limit)

    def parent_ids_for_pass(self, pass_index: int) -> list[str]:
        """Return the exact stable parents recorded for checkpoint replay."""
        for observation in self.state["observations"]:
            if int(observation["pass_index"]) == pass_index:
                return list(observation["active_parent_ids"])
        raise ValueError(f"No saturation observation exists for pass {pass_index}")

    def outcomes_for_pass(self, pass_index: int) -> dict[str, str]:
        """Return recorded outcomes for deterministic replay verification."""
        for observation in self.state["observations"]:
            if int(observation["pass_index"]) == pass_index:
                return dict(observation["outcomes"])
        raise ValueError(f"No saturation observation exists for pass {pass_index}")

    def novel_record_ids_for_pass(self, pass_index: int) -> dict[str, list[str]]:
        """Return accepted novel record IDs recorded for checkpoint replay."""
        for observation in self.state["observations"]:
            if int(observation["pass_index"]) == pass_index:
                return {
                    parent_id: list(record_ids)
                    for parent_id, record_ids in observation.get(
                        "novel_record_ids", {}
                    ).items()
                }
        raise ValueError(f"No saturation observation exists for pass {pass_index}")

    def observe_parents(
        self,
        *,
        pass_index: int,
        outcomes: dict[str, ParentOutcome],
        novel_record_ids: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Persist one pass and update every active parent's terminal state."""
        if pass_index != self.next_pass:
            raise ValueError(f"Expected saturation pass {self.next_pass}, received {pass_index}")
        active = self.active_parent_ids
        if sorted(outcomes) != active:
            raise ValueError("Saturation outcomes must cover exactly the active parent IDs")
        invalid = sorted(set(outcomes.values()) - {"novel", "empty", *_FAILURE_OUTCOMES})
        if invalid:
            raise ValueError(f"Unknown saturation outcomes: {', '.join(invalid)}")
        normalized_novel_ids = {
            str(parent_id): sorted({str(record_id) for record_id in record_ids})
            for parent_id, record_ids in (novel_record_ids or {}).items()
        }
        expected_novel_parents = {
            parent_id for parent_id, outcome in outcomes.items() if outcome == "novel"
        }
        if set(normalized_novel_ids) != expected_novel_parents or any(
            not record_ids for record_ids in normalized_novel_ids.values()
        ):
            raise ValueError(
                "Novel record IDs must be non-empty and cover exactly parents "
                "with a novel outcome"
            )

        counts = {name: 0 for name in ("novel", "empty", *_FAILURE_OUTCOMES)}
        completed_ids = []
        quarantined_ids = []
        for parent_id in active:
            outcome = outcomes[parent_id]
            counts[outcome] += 1
            parent = self.state["parents"][parent_id]
            parent["last_pass"] = pass_index
            if outcome == "novel":
                parent["empty_passes"] = 0
            elif outcome == "empty":
                parent["empty_passes"] += 1
                if not self.policy.enabled or parent["empty_passes"] >= self.policy.empty_passes_required:
                    parent["status"] = "completed"
                    completed_ids.append(parent_id)
            else:
                parent["status"] = "quarantined"
                parent["failure_reason"] = outcome
                quarantined_ids.append(parent_id)

        if not self.policy.enabled:
            for parent_id in active:
                parent = self.state["parents"][parent_id]
                if parent["status"] == "active":
                    parent["status"] = "completed"
                    completed_ids.append(parent_id)

        observation = {
            "pass_index": pass_index,
            "active_parent_ids": active,
            "outcomes": dict(sorted(outcomes.items())),
            "novel_record_ids": dict(sorted(normalized_novel_ids.items())),
            "counts": counts,
            "completed_parent_ids": sorted(completed_ids),
            "quarantined_parent_ids": sorted(quarantined_ids),
            "eligible_saturation_observations": counts["novel"] + counts["empty"],
        }
        self.state["observations"].append(observation)
        self.state["next_pass"] = pass_index + 1

        remaining = self.active_parent_ids
        quarantined = self.quarantined_parent_ids
        if not remaining:
            self.state["incomplete"] = bool(quarantined)
            self.state["converged"] = not quarantined
            self.state["stop_reason"] = (
                "incomplete_quarantined_parents" if quarantined else ("single_pass" if not self.policy.enabled else "per_parent_zero_novelty")
            )
        elif self.policy.max_passes and self.next_pass > self.policy.max_passes:
            self.state["incomplete"] = True
            self.state["converged"] = False
            self.state["stop_reason"] = "hard_pass_limit"
        else:
            self.state["incomplete"] = False
            self.state["converged"] = False
            self.state["stop_reason"] = None
        self._save()
        return observation

    @property
    def quarantined_parent_ids(self) -> list[str]:
        """Return parents whose failures cannot count toward convergence."""
        return sorted(parent_id for parent_id, parent in self.state["parents"].items() if parent.get("status") == "quarantined")

    def summary(self) -> dict[str, Any]:
        """Return the audit-ready state, policy, and parent status counts."""
        statuses = {"active": 0, "completed": 0, "quarantined": 0}
        for parent in self.state["parents"].values():
            statuses[parent["status"]] += 1
        return {
            **self.state,
            "policy": asdict(self.policy),
            "unlimited": self.policy.unlimited,
            "parent_status_counts": statuses,
        }
