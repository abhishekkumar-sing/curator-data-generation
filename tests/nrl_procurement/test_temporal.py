"""Focused tests for fail-closed temporal dataset construction."""

import json
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from temporal import (  # noqa: E402
    assign_temporal_splits,
    build_temporal_alignments,
    build_temporal_records,
    ensure_temporal_pair_rows,
    extract_temporal_changes,
    load_temporal_config,
    resolve_manifest_pairs,
    temporal_config_fingerprint,
    temporal_judge_issues,
    temporal_record_issues,
    write_temporal_artifacts,
)


def _proposition(
    proposition_id: str,
    manual_id: str,
    as_of_date: str,
    quote: str,
    *,
    modality: str = "mandatory",
    issuer: str = "Department of Expenditure",
) -> dict:
    return {
        "proposition_id": proposition_id,
        "subject": "The procuring entity",
        "action": "shall publish",
        "object": "the tender notice",
        "modality": modality,
        "polarity": "positive",
        "conditions": [],
        "exceptions": [],
        "threshold": {"value": "", "unit": ""},
        "temporal_scope": "",
        "authority": {
            "manual_id": manual_id,
            "manual_title": f"Manual {manual_id}",
            "issuing_organization": issuer,
            "policy_scope": "government_reference_guidance",
            "revision_date": as_of_date,
            "as_of_date": as_of_date,
        },
        "evidence": {
            "source_file": f"{manual_id}.md",
            "source_sha256": f"sha-{manual_id}",
            "chunk_id": f"chunk-{manual_id}",
            "page": 1,
            "section": "Tender notice",
            "quote": quote,
            "start_char": 0,
            "end_char": len(quote),
        },
        "deterministic_checks": {"passed": True, "issues": []},
    }


def _config() -> dict:
    return {
        "enabled": True,
        "verification_cutoff": "2026-07-30",
        "discover_pairs_from_manifest": True,
        "pairs": [],
        "schedule": [
            {
                "step_fraction": 0.0,
                "historical_context": 0.4,
                "temporal_transition": 0.3,
                "target_context": 0.3,
            },
            {
                "step_fraction": 1.0,
                "historical_context": 0.2,
                "temporal_transition": 0.3,
                "target_context": 0.5,
            },
        ],
        "holdout_rule_family_fraction": 0.2,
        "split_seed": "test",
    }


def _manuals() -> list[dict]:
    return [
        {"manual_id": "goods_2017"},
        {
            "manual_id": "goods_2024",
            "temporal_predecessors": [
                {
                    "manual_id": "goods_2017",
                    "pair_id": "goods_temporal",
                    "lineage_basis": "publication_series",
                }
            ],
        },
    ]


def test_manifest_driven_pairs_and_secret_free_fingerprint():
    config = resolve_manifest_pairs(load_temporal_config(_config()), _manuals())
    assert [pair.pair_id for pair in config.pairs] == ["goods_temporal"]
    fingerprint = temporal_config_fingerprint(config)
    assert len(fingerprint) == 64
    assert "token" not in json.dumps(config.model_dump(mode="json"))


def test_temporal_pilot_selection_reserves_matched_manual_pairs():
    config = resolve_manifest_pairs(load_temporal_config(_config()), _manuals())

    def row(manual_id: str, chunk_id: str, text: str) -> dict:
        return {
            "manual_id": manual_id,
            "chunk_id": chunk_id,
            "generation_passage": text * 8,
            "section": "Bid security requirements",
            "content_class": "policy",
        }

    corpus = [
        row("goods_2017", "old-1", "Bid security shall be furnished. "),
        row("goods_2024", "new-1", "Bid security must be furnished. "),
        row("other", "other-1", "Unrelated contract administration. "),
    ]
    selected = ensure_temporal_pair_rows(
        [corpus[-1]],
        corpus,
        config,
        limit=3,
        seed="test",
        pairs_per_edge=1,
    )
    assert {"old-1", "new-1"}.issubset(
        {item["chunk_id"] for item in selected}
    )


def test_schedule_is_strictly_validated():
    raw = _config()
    raw["schedule"][0]["historical_context"] = 0.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_temporal_config(raw)
    raw = _config()
    raw["api_key"] = "secret"
    with pytest.raises(ValueError, match="extra"):
        load_temporal_config(raw)


def test_bounded_alignment_change_and_six_exports():
    historical = _proposition(
        "prop-old",
        "goods_2017",
        "2017",
        "The procuring entity shall publish the tender notice within 21 days.",
    )
    target = _proposition(
        "prop-new",
        "goods_2024",
        "2024",
        "The procuring entity shall publish the tender notice within 30 days.",
    )
    config = resolve_manifest_pairs(load_temporal_config(_config()), _manuals())
    candidates, rejected = build_temporal_alignments([historical, target], config)
    assert len(candidates) == 1
    assert candidates[0]["target_proposition_ids"] == ["prop-new"]
    assert rejected == []
    changes, change_rejected = extract_temporal_changes(candidates, [historical, target])
    assert change_rejected == []
    assert changes[0]["change_type"] == "numeric_or_threshold_change"
    assign_temporal_splits(changes, holdout_fraction=0.2, seed="test")
    exports = build_temporal_records(changes)
    assert set(exports) == {
        "historical_qa",
        "historical_qa_cot",
        "transition_qa",
        "transition_qa_cot",
        "target_qa",
        "target_qa_cot",
    }
    assert all(len(rows) == 1 for rows in exports.values())
    assert all(not temporal_record_issues(row) for rows in exports.values() for row in rows)


def test_rejects_reversed_dates_authority_leakage_and_identical_states():
    historical = _proposition("prop-old", "goods_2017", "2024", "The procuring entity shall publish.")
    target = _proposition(
        "prop-new",
        "goods_2024",
        "2017",
        "The procuring entity shall publish.",
        issuer="Numaligarh Refinery Limited",
    )
    config = resolve_manifest_pairs(load_temporal_config(_config()), _manuals())
    candidates, rejected = build_temporal_alignments([historical, target], config)
    assert candidates == []
    issues = set(rejected[0]["issues"])
    assert "reversed_or_identical_dates" in issues
    assert "authority_or_policy_scope_mismatch" in issues


def test_judge_contract_rejects_unsupported_currentness():
    change = {"lineage_basis": "publication_series"}
    verdict = {
        "same_rule_family": True,
        "same_subject": True,
        "material_change": True,
        "dates_ordered": True,
        "authority_isolated": True,
        "evidence_sufficient": True,
        "accepted": True,
        "rationale": "The target is currently in force and supersedes the old rule.",
    }
    assert "unsupported_currentness_or_supersession" in temporal_judge_issues(change, verdict)


def test_writes_audits_exports_and_curriculum(tmp_path):
    propositions = [
        _proposition(
            "prop-old",
            "goods_2017",
            "2017",
            "The procuring entity shall publish the tender notice within 21 days.",
        ),
        _proposition(
            "prop-new",
            "goods_2024",
            "2024",
            "The procuring entity shall publish the tender notice within 30 days.",
        ),
    ]
    stats = write_temporal_artifacts(
        tmp_path,
        propositions,
        _config(),
        _manuals(),
        run_id="test-run",
        judged_alignments=[
            {
                **build_temporal_alignments(
                    propositions,
                    resolve_manifest_pairs(load_temporal_config(_config()), _manuals()),
                )[0][0],
                "judge": {"status": "accepted", "accepted": True},
            }
        ],
    )
    assert stats["changes"] == 1
    assert (tmp_path / "temporal_alignment_candidates.jsonl").is_file()
    assert (tmp_path / "temporal_transition_qa_cot.jsonl").is_file()
    manifest = json.loads((tmp_path / "temporal_curriculum_manifest.json").read_text())
    assert manifest["training_implemented_by_curator"] is False
    assert manifest["schedule_status"] == ("experiment_configuration_not_validated_benefit")
