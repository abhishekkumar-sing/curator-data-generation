"""Fail-closed temporal alignment, change, export, and curriculum utilities."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from jsonl_io import write_jsonl_rows
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bespokelabs import curator

TEMPORAL_SCHEMA_VERSION = "1"
WORD = re.compile(r"[a-z][a-z0-9_-]{2,}", re.IGNORECASE)
NUMBER = re.compile(r"(?<!\w)(?:₹|rs\.?\s*)?\d[\d,.]*(?:\s*%|\s+[a-z]+)?", re.IGNORECASE)
CURRENTNESS = re.compile(
    r"\b(?:current(?:ly)?|active|in force|supersed(?:e|es|ed)|no longer applies)\b",
    re.IGNORECASE,
)
GENERIC = {
    "and",
    "authority",
    "bid",
    "contract",
    "document",
    "goods",
    "manual",
    "procurement",
    "rule",
    "shall",
    "source",
    "the",
    "under",
}


class TemporalPair(BaseModel):
    """One explicitly configured, same-authority historical/target pair."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str = Field(min_length=1)
    historical_manual_id: str = Field(min_length=1)
    target_manual_id: str = Field(min_length=1)
    lineage_basis: Literal["documented_amendment", "documented_supersession", "publication_series"]
    max_targets_per_historical: int = Field(default=3, ge=1, le=10)
    minimum_shared_terms: int = Field(default=2, ge=1, le=20)

    @model_validator(mode="after")
    def distinct_manuals(self) -> "TemporalPair":
        """Reject self-pairs because they cannot encode a transition."""
        if self.historical_manual_id == self.target_manual_id:
            raise ValueError("temporal pair manuals must differ")
        return self


class ScheduleWeights(BaseModel):
    """Weights at one trainer-controlled curriculum anchor."""

    model_config = ConfigDict(extra="forbid")

    step_fraction: float = Field(ge=0.0, le=1.0)
    historical_context: float = Field(ge=0.0, le=1.0)
    temporal_transition: float = Field(ge=0.0, le=1.0)
    target_context: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ScheduleWeights":
        """Require a complete probability distribution at every anchor."""
        total = self.historical_context + self.temporal_transition + self.target_context
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"schedule weights must sum to 1.0, got {total}")
        return self


class TemporalConfig(BaseModel):
    """Strict temporal configuration; secrets are neither accepted nor hashed."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    verification_cutoff: date
    discover_pairs_from_manifest: bool = True
    pairs: list[TemporalPair]
    schedule: list[ScheduleWeights]
    holdout_rule_family_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    split_seed: str = Field(default="nrl-temporal-v1", min_length=1)

    @model_validator(mode="after")
    def validate_collections(self) -> "TemporalConfig":
        """Validate unique pairs and a complete, ordered schedule domain."""
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("temporal pair_id values must be unique")
        anchors = [anchor.step_fraction for anchor in self.schedule]
        if not anchors or anchors != sorted(set(anchors)):
            raise ValueError("schedule anchors must be non-empty, unique, and increasing")
        if anchors[0] != 0.0 or anchors[-1] != 1.0:
            raise ValueError("schedule must include step_fraction anchors 0.0 and 1.0")
        return self


class TemporalJudgeVerdict(BaseModel):
    """Independent same-provision and temporal-change judgment."""

    model_config = ConfigDict(extra="forbid")

    same_rule_family: bool
    same_subject: bool
    material_change: bool
    dates_ordered: bool
    authority_isolated: bool
    evidence_sufficient: bool
    accepted: bool
    rationale: str = Field(min_length=1)


def load_temporal_config(raw: dict[str, Any]) -> TemporalConfig:
    """Parse temporal configuration with strict types and no implicit extras."""
    return TemporalConfig.model_validate(raw)


def temporal_config_fingerprint(config: TemporalConfig) -> str:
    """Return a stable, secret-free fingerprint of the validated configuration."""
    payload = config.model_dump(mode="json", exclude={"enabled"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_manifest_pairs(
    config: TemporalConfig,
    manuals: list[dict[str, Any]],
) -> TemporalConfig:
    """Add only explicitly registered temporal predecessor edges."""
    discovered: list[dict[str, Any]] = []
    known_ids = {str(manual["manual_id"]) for manual in manuals}
    for manual in manuals:
        target = str(manual["manual_id"])
        for edge in manual.get("temporal_predecessors", []):
            historical = str(edge.get("manual_id", ""))
            if not historical or historical not in known_ids:
                raise ValueError(f"{target} temporal predecessor {historical!r} is not registered")
            discovered.append(
                {
                    "pair_id": str(edge.get("pair_id", f"{historical}_to_{target}")),
                    "historical_manual_id": historical,
                    "target_manual_id": target,
                    "lineage_basis": edge["lineage_basis"],
                    "max_targets_per_historical": edge.get("max_targets_per_historical", 3),
                    "minimum_shared_terms": edge.get("minimum_shared_terms", 2),
                }
            )
    configured = [pair.model_dump(mode="json") for pair in config.pairs]
    combined = configured + discovered if config.discover_pairs_from_manifest else configured
    unique = {item["pair_id"]: item for item in combined}
    if len(unique) != len(combined):
        raise ValueError("configured and manifest temporal pair_id values overlap")
    payload = config.model_dump(mode="json")
    payload["pairs"] = list(unique.values())
    return TemporalConfig.model_validate(payload)


def _terms(value: str) -> set[str]:
    return {term.casefold() for term in WORD.findall(value or "") if term.casefold() not in GENERIC}


def _semantic_terms(proposition: dict[str, Any]) -> set[str]:
    return _terms(
        " ".join(
            [
                proposition.get("subject", ""),
                proposition.get("action", ""),
                proposition.get("object", ""),
                *proposition.get("conditions", []),
                *proposition.get("exceptions", []),
            ]
        )
    )


def _parse_source_date(value: str) -> date:
    """Parse manifest date precision conservatively for ordering only."""
    value = str(value).strip()
    for pattern in ("%d.%m.%Y", "%B %Y", "%Y"):
        try:
            parsed = datetime.strptime(value, pattern)
            return parsed.date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported source date: {value!r}")


def _authority_group(proposition: dict[str, Any]) -> tuple[str, str]:
    authority = proposition["authority"]
    return (
        str(authority.get("issuing_organization", "")).casefold(),
        str(authority.get("policy_scope", "")).casefold(),
    )


def _alignment_id(pair_id: str, historical_id: str, target_ids: list[str]) -> str:
    payload = [TEMPORAL_SCHEMA_VERSION, pair_id, historical_id, *sorted(target_ids)]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return "tal-" + digest[:24]


def build_temporal_alignments(
    propositions: list[dict[str, Any]],
    config: TemporalConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build bounded one-to-many candidates; reject unsafe pairs with reasons."""
    accepted_props = [row for row in propositions if row.get("proposition_id") and row.get("deterministic_checks", {}).get("passed", False)]
    by_manual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_props:
        by_manual[row["authority"]["manual_id"]].append(row)

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for pair in config.pairs:
        historical_rows = by_manual.get(pair.historical_manual_id, [])
        target_rows = by_manual.get(pair.target_manual_id, [])
        for historical in historical_rows:
            scored: list[tuple[int, dict[str, Any], list[str]]] = []
            for target in target_rows:
                issues: list[str] = []
                if _authority_group(historical) != _authority_group(target):
                    issues.append("authority_or_policy_scope_mismatch")
                historical_date = _parse_source_date(historical["authority"]["as_of_date"])
                target_date = _parse_source_date(target["authority"]["as_of_date"])
                if historical_date >= target_date:
                    issues.append("reversed_or_identical_dates")
                shared = sorted(_semantic_terms(historical) & _semantic_terms(target))
                subject_shared = _terms(historical["subject"]) & _terms(target["subject"])
                if len(shared) < pair.minimum_shared_terms or not subject_shared:
                    issues.append("unrelated_subject_or_insufficient_signature")
                if historical["evidence"]["source_sha256"] == target["evidence"]["source_sha256"]:
                    issues.append("same_source_document")
                if issues:
                    rejected.append(
                        {
                            "pair_id": pair.pair_id,
                            "historical_proposition_id": historical["proposition_id"],
                            "target_proposition_id": target["proposition_id"],
                            "issues": sorted(set(issues)),
                        }
                    )
                    continue
                scored.append((len(shared), target, shared))

            chosen = sorted(
                scored,
                key=lambda item: (-item[0], item[1]["proposition_id"]),
            )[: pair.max_targets_per_historical]
            if not chosen:
                continue
            targets = [item[1] for item in chosen]
            target_ids = [row["proposition_id"] for row in targets]
            candidates.append(
                {
                    "alignment_id": _alignment_id(
                        pair.pair_id,
                        historical["proposition_id"],
                        target_ids,
                    ),
                    "pair_id": pair.pair_id,
                    "lineage_basis": pair.lineage_basis,
                    "historical_proposition_id": historical["proposition_id"],
                    "target_proposition_ids": target_ids,
                    "historical_manual_id": pair.historical_manual_id,
                    "target_manual_id": pair.target_manual_id,
                    "historical_as_of": historical["authority"]["as_of_date"],
                    "target_as_of": targets[0]["authority"]["as_of_date"],
                    "shared_terms": sorted(set().union(*(set(item[2]) for item in chosen))),
                    "schema_version": TEMPORAL_SCHEMA_VERSION,
                    "judge": {"status": "not_run", "accepted": False},
                }
            )
    return candidates, rejected


def _proposition_state(proposition: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposition_id": proposition["proposition_id"],
        "manual_id": proposition["authority"]["manual_id"],
        "manual_title": proposition["authority"]["manual_title"],
        "issuing_organization": proposition["authority"]["issuing_organization"],
        "policy_scope": proposition["authority"]["policy_scope"],
        "as_of_date": proposition["authority"]["as_of_date"],
        "subject": proposition["subject"],
        "action": proposition["action"],
        "object": proposition["object"],
        "modality": proposition["modality"],
        "polarity": proposition["polarity"],
        "conditions": proposition["conditions"],
        "exceptions": proposition["exceptions"],
        "threshold": proposition["threshold"],
        "evidence": proposition["evidence"],
    }


def classify_change(
    historical: dict[str, Any],
    targets: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Classify only differences directly observable in grounded fields."""
    issues: list[str] = []
    if not targets:
        return "unresolved", ["missing_target_state"]
    historical_text = " ".join([historical["subject"], historical["action"], historical["object"]]).casefold()
    target_texts = [" ".join([row["subject"], row["action"], row["object"]]).casefold() for row in targets]
    if (
        len(targets) == 1
        and historical_text == target_texts[0]
        and historical["evidence"]["quote"].strip().casefold() == targets[0]["evidence"]["quote"].strip().casefold()
    ):
        same_structured = all(historical.get(field) == targets[0].get(field) for field in ("modality", "polarity", "conditions", "exceptions", "threshold"))
        if same_structured:
            return "identical_state", ["identical_states_are_not_transitions"]
    if historical["modality"] != targets[0]["modality"]:
        return "modality_change", issues
    if historical["polarity"] != targets[0]["polarity"]:
        return "polarity_change", issues
    historical_numbers = NUMBER.findall(historical["evidence"]["quote"])
    target_numbers = [number for target in targets for number in NUMBER.findall(target["evidence"]["quote"])]
    if historical_numbers != target_numbers:
        return "numeric_or_threshold_change", issues
    if historical.get("conditions") != targets[0].get("conditions"):
        return "condition_change", issues
    if historical.get("exceptions") != targets[0].get("exceptions"):
        return "exception_change", issues
    if len(targets) > 1:
        return "one_to_many_restructure", issues
    return "wording_or_scope_change", issues


def extract_temporal_changes(
    alignments: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize exact, source-grounded state changes and rejection audit."""
    by_id = {row["proposition_id"]: row for row in propositions if row.get("proposition_id")}
    changes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for alignment in alignments:
        historical = by_id.get(alignment["historical_proposition_id"])
        targets = [by_id[target_id] for target_id in alignment["target_proposition_ids"] if target_id in by_id]
        issues = []
        if historical is None:
            issues.append("missing_historical_proposition")
        if len(targets) != len(alignment["target_proposition_ids"]):
            issues.append("missing_target_proposition")
        if issues:
            rejected.append({**alignment, "issues": issues})
            continue
        assert historical is not None
        change_type, classification_issues = classify_change(historical, targets)
        issues.extend(classification_issues)
        change_id = (
            "tchg-"
            + hashlib.sha256(
                json.dumps(
                    [
                        TEMPORAL_SCHEMA_VERSION,
                        alignment["alignment_id"],
                        change_type,
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()[:24]
        )
        change = {
            "change_id": change_id,
            "change_lineage_id": f"lineage-{alignment['pair_id']}-{historical['proposition_id']}",
            "alignment_id": alignment["alignment_id"],
            "pair_id": alignment["pair_id"],
            "lineage_basis": alignment["lineage_basis"],
            "change_type": change_type,
            "historical_state": _proposition_state(historical),
            "target_states": [_proposition_state(target) for target in targets],
            "deterministic_checks": {"passed": not issues, "issues": issues},
            "judge": alignment.get("judge", {"status": "not_run", "accepted": False}),
            "schema_version": TEMPORAL_SCHEMA_VERSION,
        }
        if issues:
            rejected.append(change)
        else:
            changes.append(change)
    return changes, rejected


def temporal_judge_issues(
    change: dict[str, Any],
    verdict: dict[str, Any],
) -> list[str]:
    """Validate an independent temporal verdict and prohibit unsafe assertions."""
    issues: list[str] = []
    required = {
        "same_rule_family",
        "same_subject",
        "material_change",
        "dates_ordered",
        "authority_isolated",
        "evidence_sufficient",
        "accepted",
    }
    if missing := sorted(required - set(verdict)):
        issues.append("missing_judge_fields:" + ",".join(missing))
        return issues
    if not all(
        verdict[field]
        for field in (
            "same_rule_family",
            "same_subject",
            "material_change",
            "dates_ordered",
            "authority_isolated",
            "evidence_sufficient",
        )
    ):
        issues.append("judge_rejected_temporal_binding")
    if bool(verdict["accepted"]) != (not issues):
        issues.append("inconsistent_judge_acceptance")
    if CURRENTNESS.search(str(verdict.get("rationale", ""))) and change["lineage_basis"] not in {
        "documented_amendment",
        "documented_supersession",
    }:
        issues.append("unsupported_currentness_or_supersession")
    return issues


def build_temporal_judge_inputs(
    alignments: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach exact grounded states for independent alignment judgment."""
    by_id = {row["proposition_id"]: row for row in propositions if row.get("proposition_id")}
    inputs = []
    for alignment in alignments:
        historical = by_id.get(alignment["historical_proposition_id"])
        targets = [by_id[target_id] for target_id in alignment["target_proposition_ids"] if target_id in by_id]
        if historical is None or len(targets) != len(alignment["target_proposition_ids"]):
            continue
        inputs.append(
            {
                "alignment": alignment,
                "historical_state": _proposition_state(historical),
                "target_states": [_proposition_state(target) for target in targets],
            }
        )
    return inputs


class TemporalAlignmentJudge(curator.LLM):
    """Independently reject unsafe or unrelated temporal alignments."""

    response_format = TemporalJudgeVerdict

    def prompt(self, row: dict[str, Any]) -> str:
        """Render an evidence-only, authority-aware temporal judgment."""
        payload = {
            "alignment": row["alignment"],
            "historical_state": row["historical_state"],
            "target_states": row["target_states"],
        }
        return f"""TASK
Judge whether the historical proposition and all target propositions represent
dated states of the same procurement rule and a material, source-supported
change. This is a strict quality gate, not a writing task.

REJECT WHEN ANY APPLY
- identical states or merely cosmetic restatement;
- unrelated subjects or rule families;
- historical date is not earlier than target date;
- Government and NRL authority/policy scopes are mixed;
- exact evidence does not support the structured fields;
- a changed number, modality, condition, exception, or scope is missed;
- currentness, legal supersession, adoption, or causal reason is inferred
  without an explicit documented lineage basis.

OUTPUT
Return exactly the enforced TemporalJudgeVerdict. `accepted` is true only when
all six boolean quality dimensions are true. Keep rationale factual and do not
introduce currentness or supersession language for publication_series pairs.

UNTRUSTED GROUNDED CANDIDATE
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""

    def parse(
        self,
        row: dict[str, Any],
        response: TemporalJudgeVerdict,
    ) -> dict[str, Any]:
        """Attach a verdict after validating internal consistency."""
        verdict = response.model_dump()
        issues = temporal_judge_issues(row["alignment"], verdict)
        accepted = bool(verdict["accepted"]) and not issues
        return {
            **row["alignment"],
            "judge": {
                **verdict,
                "accepted": accepted,
                "status": "accepted" if accepted else "rejected",
                "issues": issues,
            },
        }


def _rule_family(change: dict[str, Any]) -> str:
    state = change["historical_state"]
    signature = sorted(_terms(state["subject"]) | _terms(state["action"]))
    return "rule-" + hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()[:16]


def assign_temporal_splits(
    changes: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: str,
) -> None:
    """Keep lineages together and hold out complete semantic rule families."""
    families = sorted({_rule_family(change) for change in changes})
    ranked = sorted(
        families,
        key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(),
    )
    holdout_count = int(round(len(ranked) * holdout_fraction))
    if holdout_fraction and ranked:
        holdout_count = max(1, holdout_count)
    holdout = set(ranked[:holdout_count])
    for change in changes:
        family = _rule_family(change)
        change["rule_family_id"] = family
        change["split"] = "evaluation" if family in holdout else "train"


def _record_id(change_id: str, family: str, cot: bool) -> str:
    digest = hashlib.sha256(f"{change_id}:{family}:{int(cot)}".encode()).hexdigest()
    return "temporal-" + digest[:24]


def build_temporal_records(changes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Create dated QA and QA-CoT records without unsupported currentness."""
    exports = {
        "historical_qa": [],
        "historical_qa_cot": [],
        "transition_qa": [],
        "transition_qa_cot": [],
        "target_qa": [],
        "target_qa_cot": [],
    }
    for change in changes:
        historical = change["historical_state"]
        targets = change["target_states"]
        target = targets[0]
        historical_scope = f"{historical['manual_title']} (as of {historical['as_of_date']}, " f"{historical['issuing_organization']})"
        target_scope = f"{target['manual_title']} (as of {target['as_of_date']}, " f"{target['issuing_organization']})"
        historical_answer = historical["evidence"]["quote"]
        target_answer = "\n".join(state["evidence"]["quote"] for state in targets)
        transition_answer = (
            f"Historical state — {historical_scope}: {historical_answer}\n"
            f"Target state — {target_scope}: {target_answer}\n"
            f"Observed change type: {change['change_type']}."
        )
        common = {
            "change_id": change["change_id"],
            "change_lineage_id": change["change_lineage_id"],
            "rule_family_id": change["rule_family_id"],
            "split": change["split"],
            "lineage_basis": change["lineage_basis"],
            "historical_as_of": historical["as_of_date"],
            "target_as_of": target["as_of_date"],
            "authority_scope": historical["policy_scope"],
            "schema_version": TEMPORAL_SCHEMA_VERSION,
        }
        definitions = [
            (
                "historical",
                f"What did {historical_scope} state about {historical['subject']}?",
                historical_answer,
                [historical["evidence"]],
                [
                    "lookup: locate the explicitly dated historical proposition",
                    "attribute: retain its issuing authority and policy scope",
                    "answer: reproduce only its exact supported state",
                ],
            ),
            (
                "transition",
                f"How does the rule concerning {historical['subject']} differ between " f"{historical_scope} and {target_scope}?",
                transition_answer,
                [historical["evidence"], *(state["evidence"] for state in targets)],
                [
                    "lookup: locate the explicitly dated historical state",
                    "lookup: locate the explicitly dated target state",
                    "compare: inspect grounded modality, number, condition, exception, and wording fields",
                    "conclude: describe only the observed change and documented lineage basis",
                ],
            ),
            (
                "target",
                f"What does {target_scope} state about {target['subject']}?",
                target_answer,
                [state["evidence"] for state in targets],
                [
                    "lookup: locate the explicitly dated target proposition",
                    "attribute: retain its issuing authority and policy scope",
                    "answer: reproduce only its exact supported state",
                ],
            ),
        ]
        for family, question, answer, evidence, steps in definitions:
            for cot in (False, True):
                exports[f"{family}_qa_cot" if cot else f"{family}_qa"].append(
                    {
                        **common,
                        "record_id": _record_id(change["change_id"], family, cot),
                        "phase": f"{family}_context" if family != "transition" else "temporal_transition",
                        "task_type": "qa_cot" if cot else "qa",
                        "question": question,
                        "answer": answer,
                        "reasoning_steps": steps if cot else [],
                        "evidence": evidence,
                    }
                )
    return exports


def temporal_record_issues(record: dict[str, Any]) -> list[str]:
    """Reject missing time/authority labels and leaked currentness claims."""
    issues: list[str] = []
    visible = f"{record.get('question', '')} {record.get('answer', '')}"
    if str(record.get("historical_as_of", "")) not in visible and record["phase"] != "target_context":
        issues.append("missing_historical_temporal_label")
    if str(record.get("target_as_of", "")) not in visible and record["phase"] != "historical_context":
        issues.append("missing_target_temporal_label")
    if not str(record.get("authority_scope", "")).strip():
        issues.append("missing_authority_scope")
    if CURRENTNESS.search(visible) and record["lineage_basis"] not in {
        "documented_amendment",
        "documented_supersession",
    }:
        issues.append("unsupported_currentness_or_supersession")
    return issues


def write_temporal_artifacts(
    output_dir: Path,
    propositions: list[dict[str, Any]],
    raw_config: dict[str, Any],
    manuals: list[dict[str, Any]],
    *,
    run_id: str,
    judged_alignments: list[dict[str, Any]] | None = None,
    alignment_rejected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and atomically write all temporal audits, exports, and curriculum."""
    config = resolve_manifest_pairs(load_temporal_config(raw_config), manuals)
    candidates, deterministic_rejected = build_temporal_alignments(propositions, config)
    judged_alignments = judged_alignments or []
    rejected_judgments = [row for row in judged_alignments if not row.get("judge", {}).get("accepted", False)]
    accepted_alignments = [row for row in judged_alignments if row.get("judge", {}).get("accepted", False)]
    judged_ids = {row["alignment_id"] for row in judged_alignments}
    missing_judgments = [
        {
            **row,
            "judge": {
                "status": "missing",
                "accepted": False,
                "issues": ["missing_temporal_judge_response"],
            },
        }
        for row in candidates
        if row["alignment_id"] not in judged_ids
    ]
    all_alignment_rejected = [
        *deterministic_rejected,
        *(alignment_rejected or []),
        *rejected_judgments,
        *missing_judgments,
    ]
    changes, change_rejected = extract_temporal_changes(accepted_alignments, propositions)
    assign_temporal_splits(
        changes,
        holdout_fraction=config.holdout_rule_family_fraction,
        seed=config.split_seed,
    )
    exports = build_temporal_records(changes)
    validation_rejected = []
    for name, rows in exports.items():
        accepted = []
        for row in rows:
            issues = temporal_record_issues(row)
            if issues:
                validation_rejected.append({"record_id": row["record_id"], "export": name, "issues": issues})
            else:
                accepted.append(row)
        exports[name] = accepted

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_rows(output_dir / "temporal_alignment_candidates.jsonl", candidates)
    write_jsonl_rows(
        output_dir / "temporal_alignment_rejected.jsonl",
        all_alignment_rejected,
    )
    write_jsonl_rows(output_dir / "temporal_changes.jsonl", changes)
    write_jsonl_rows(
        output_dir / "temporal_changes_rejected.jsonl",
        [*change_rejected, *validation_rejected],
    )
    filenames = {
        "historical_qa": "temporal_historical_qa.jsonl",
        "historical_qa_cot": "temporal_historical_qa_cot.jsonl",
        "transition_qa": "temporal_transition_qa.jsonl",
        "transition_qa_cot": "temporal_transition_qa_cot.jsonl",
        "target_qa": "temporal_target_qa.jsonl",
        "target_qa_cot": "temporal_target_qa_cot.jsonl",
    }
    for name, filename in filenames.items():
        write_jsonl_rows(output_dir / filename, exports[name])

    curriculum = {
        "run_id": run_id,
        "schema_version": TEMPORAL_SCHEMA_VERSION,
        "verification_cutoff": config.verification_cutoff.isoformat(),
        "config_fingerprint": temporal_config_fingerprint(config),
        "schedule": [row.model_dump(mode="json") for row in config.schedule],
        "schedule_status": "experiment_configuration_not_validated_benefit",
        "training_implemented_by_curator": False,
        "record_ids_by_phase": {name: [row["record_id"] for row in rows] for name, rows in exports.items()},
    }
    manifest_path = output_dir / "temporal_curriculum_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(curriculum, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return {
        "enabled": config.enabled,
        "config_fingerprint": curriculum["config_fingerprint"],
        "verification_cutoff": curriculum["verification_cutoff"],
        "alignment_candidates": len(candidates),
        "alignment_rejected": len(all_alignment_rejected),
        "changes": len(changes),
        "changes_rejected": len(change_rejected),
        "record_counts": {name: len(rows) for name, rows in exports.items()},
        "validation_rejected": len(validation_rejected),
        "judge_status": ("complete" if len(judged_alignments) == len(candidates) else "incomplete"),
        "missing_judge_responses": len(missing_judgments),
        "schedule_status": curriculum["schedule_status"],
    }
