"""Focused tests for the local procurement pipeline."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ValidationError

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))
REPO_ROOT = Path(__file__).resolve().parents[2]

import generate as generation_pipeline  # noqa: E402
import resume as resume_module  # noqa: E402
from corpus import (  # noqa: E402
    _content_class,
    _document_family,
    corpus_quality_report,
    generation_text,
    load_corpus,
    representative_rows,
    source_quality_issues,
)
from cross_document import build_bundles  # noqa: E402
from cross_stage import (  # noqa: E402
    CrossDocumentGenerator,
    CrossDocumentJudge,
    CrossSourceAblationAnswerGenerator,
    CrossSourceAblationJudge,
    SingularCrossDocumentJudge,
    adjudicate_cross_ablation_trials,
    apply_cross_ablation_gate,
    build_cross_ablation_judge_inputs,
    build_cross_ablation_trial_inputs,
    cross_ablation_trial_validation_issues,
)
from drafting import (  # noqa: E402
    TenderDraftingGenerator,
    TenderDraftingJudge,
    _stable_block_union,
    build_drafting_inputs,
    compact_drafting,
    drafting_citation_integrity_issues,
    drafting_validation_issues,
    normalize_drafting_response,
    read_drafting_seeds,
    reconcile_drafting_support,
)
from export import (  # noqa: E402
    answer_length_statistics,
    answer_style_diversity,
    assert_unique_record_ids,
    assign_drafting_splits,
    assign_splits,
    batch_efficiency_stats,
    categorical_diversity,
    export_records,
    question_answer_relevance_diagnostics,
    question_opener_diversity,
)
from generate import (  # noqa: E402
    QUESTION_OPENER_EXAMPLES,
    ProcurementBlueprintGenerator,
    ProcurementGenerator,
    ProcurementJudge,
    SingularProcurementJudge,
    _budget_judge_rows,
    _judge_prompt_budget,
    _singular_judge_batch_size,
    build_generation_validation_rescue_inputs,
    eligible_question_types,
    expand_single_generation_candidates,
    judge_eligible_planned,
    materialize_blueprint_rejection,
    materialize_terminal_failures,
    plan_cross_document_requests,
    plan_question_styles,
    plan_question_types,
    plan_single_document_requests,
    request_coverage,
    select_best_single_candidates,
)
from judge_calibration import calibrate_judge, load_judge_calibration  # noqa: E402
from path_qa import (  # noqa: E402
    SourceAblationAnswerGenerator,
    SourceAblationJudge,
    ablation_trial_validation_issues,
    adjudicate_ablation_trials,
    answer_validation_issues,
    build_ablation_judge_inputs,
    build_ablation_trial_inputs,
    build_missing_hop_contrasts,
    false_premise_quarantine,
    question_validation_issues,
)
from prompt_budget import (  # noqa: E402
    configured_context_window,
    measure_rendered_request,
    vllm_tokenize_chat,
)
from propositions import (  # noqa: E402
    PropositionExtractor,
    materialize_empty_extraction,
    materialize_proposition,
    proposition_cache_fingerprint,
    proposition_validation_issues,
    read_cached_propositions,
    write_proposition_cache,
)
from provenance import build_reasoning_graph, leakage_audit  # noqa: E402
from reasoning_paths import build_reasoning_paths, validate_reasoning_path  # noqa: E402
from resume import ResumeManager  # noqa: E402
from review import REVIEW_DIMENSIONS, prepare_review, validate_reviews  # noqa: E402
from schemas import (  # noqa: E402
    AblationJudgeDecision,
    CrossAblationTrialDraft,
    CrossCandidateBatch,
    CrossJudgeBatch,
    CrossJudgedCandidate,
    DraftingBlock,
    DraftingFieldClaim,
    DraftingResult,
    GroundedCandidateDraft,
    JudgeBatch,
    JudgedCandidate,
    JudgeDecision,
    PropositionBatch,
    QABlueprintDraft,
    collect_structural_repairs,
)
from source_windows import (  # noqa: E402
    build_source_windows,
    resolve_component_references,
)
from validate_run import _failure_distribution, _schema_validity_rate, validate_run  # noqa: E402
from validation import (  # noqa: E402
    SOURCE_FRAMING_PREFIX,
    answer_format_issues,
    canonical_reasoning_operation,
    cross_claim_contradiction_issues,
    deduplicate,
    enforce_category_diversity,
    enforce_extractive_answer_diversity,
    enforce_question_opener_diversity,
    is_extractive_answer,
    judge_batch_identity_issues,
    judge_quotes_are_grounded,
    question_style_issues,
    realign_whitespace_verbatim_quote,
    recover_grounded_judge_quotes,
    remove_cosmetic_persona_prefix,
    semantic_support_issues,
    validate_cross_record,
    validate_record,
)


def _judge_decision(**overrides):
    decision = {
        "supported": True,
        "relevant": True,
        "preserves_qualifications": True,
        "authority_correct": True,
        "reasoning_valid": True,
        "question_natural": True,
        "persona_relevant": True,
        "recommended_task": "general_reference",
        "recommended_persona": "general_user",
        "answer_found_in_source": True,
        "answer_quotes": ["Supported text."],
        "score": 5,
        "issues": [],
    }
    decision.update(overrides)
    return decision


def test_judge_batch_identity_reports_duplicate_missing_and_unexpected_ids() -> None:
    issues = judge_batch_identity_issues(
        ["record-a", "record-b"],
        ["record-a", "record-a", "record-c"],
    )
    assert "duplicate_judge_record_ids:record-a" in issues
    assert "missing_judge_record_ids:record-b" in issues
    assert "unexpected_judge_record_ids:record-c" in issues
    assert "judge_cardinality_mismatch:expected=2,returned=3" in issues


def test_singular_judge_contracts_are_direct_objects_and_batch_size_fails_closed() -> None:
    assert "judgments" not in SingularProcurementJudge.response_format.model_json_schema()["properties"]
    assert "judgments" not in SingularCrossDocumentJudge.response_format.model_json_schema()["properties"]
    assert _singular_judge_batch_size() == 1
    original = generation_pipeline.QUALITY["judge_batch_size"]
    generation_pipeline.QUALITY["judge_batch_size"] = 2
    try:
        try:
            _singular_judge_batch_size()
        except SystemExit as exc:
            assert "must be 1" in str(exc)
        else:
            raise AssertionError("unsupported judge batching must fail closed")
    finally:
        generation_pipeline.QUALITY["judge_batch_size"] = original


def test_singular_judge_wrong_id_reuses_fail_closed_quarantine() -> None:
    row = {
        "judge_items": [
            {
                "record_id": "expected-id",
                "record": {"record_id": "expected-id"},
            }
        ]
    }
    response = JudgedCandidate.model_validate(
        {
            "record_id": "wrong-id",
            "decision": _judge_decision(),
        }
    )
    judged = SingularProcurementJudge.parse(
        SimpleNamespace(model_name="judge-model"),
        row,
        response,
    )
    assert judged[0]["judge"]["accepted"] is False
    assert "missing_judge_record_ids:expected-id" in judged[0]["judge"]["issues"]

    cross_response = CrossJudgedCandidate.model_validate(
        {
            "record_id": "wrong-id",
            "decision": _judge_decision(
                full_context_supported=True,
                unsupported_without_source_ids=["source_a", "source_b"],
                connected_reasoning=True,
                relationship_correct=True,
            ),
        }
    )
    cross_judged = SingularCrossDocumentJudge.parse(
        SimpleNamespace(model_name="judge-model"),
        row,
        cross_response,
    )
    assert cross_judged[0]["judge"]["accepted"] is False


def test_cross_judge_quarantines_entire_batch_when_model_duplicates_an_id() -> None:
    records = [
        {"record_id": "record-a"},
        {"record_id": "record-b"},
    ]
    row = {"judge_items": [{"record_id": record["record_id"], "record": record} for record in records]}
    response = CrossJudgeBatch.model_validate(
        {
            "judgments": [
                {
                    "record_id": "record-a",
                    "decision": _judge_decision(
                        full_context_supported=True,
                        unsupported_without_source_ids=["source_a", "source_b"],
                        connected_reasoning=True,
                        relationship_correct=True,
                    ),
                },
                {
                    "record_id": "record-a",
                    "decision": _judge_decision(
                        full_context_supported=True,
                        unsupported_without_source_ids=["source_a", "source_b"],
                        connected_reasoning=True,
                        relationship_correct=True,
                    ),
                },
            ]
        }
    )
    judged = CrossDocumentJudge.parse(
        SimpleNamespace(model_name="judge-model"),
        row,
        response,
    )
    assert [record["record_id"] for record in judged] == ["record-a", "record-b"]
    assert all(record["judge"]["accepted"] is False for record in judged)
    assert all(record["judge"]["batch_integrity_passed"] is False for record in judged)
    assert all("duplicate_judge_record_ids:record-a" in record["judge"]["issues"] for record in judged)
    assert all("missing_judge_record_ids:record-b" in record["judge"]["issues"] for record in judged)


def test_export_identity_gate_rejects_duplicate_and_missing_stable_ids() -> None:
    try:
        assert_unique_record_ids(
            [{"record_id": "record-a"}, {"record_id": "record-a"}],
            dataset_name="test records",
        )
    except ValueError as exc:
        assert "duplicate record_id values" in str(exc)
    else:
        raise AssertionError("duplicate stable IDs must fail closed")

    try:
        assert_unique_record_ids([{"record_id": ""}], dataset_name="test records")
    except ValueError as exc:
        assert "missing record_id" in str(exc)
    else:
        raise AssertionError("missing stable IDs must fail closed")


def test_ablation_trials_are_three_blind_context_variants() -> None:
    propositions = [
        {
            "proposition_id": "prop-a",
            "evidence": {"quote": "Source A exact evidence."},
        },
        {
            "proposition_id": "prop-b",
            "evidence": {"quote": "Source B exact evidence."},
        },
    ]
    answer = {
        "record_id": "answer-1",
        "question_id": "question-1",
        "path_id": "path-1",
        "question": "What follows from the two procurement rules?",
        "task_type": "cross_document_qa",
        "claims": [{"statement": "Canonical claim", "evidence": []}],
        "propositions": propositions,
    }
    trials = build_ablation_trial_inputs([answer])
    assert [trial["variant"] for trial in trials] == [
        "full",
        "source_a_only",
        "source_b_only",
    ]
    assert [trial["visible_proposition_ids"] for trial in trials] == [
        ["prop-a", "prop-b"],
        ["prop-a"],
        ["prop-b"],
    ]
    prompts = [SourceAblationAnswerGenerator.prompt(SimpleNamespace(), trial) for trial in trials]
    assert all("Canonical claim" not in prompt for prompt in prompts)
    assert all("source_a_only" not in prompt and "source_b_only" not in prompt for prompt in prompts)


def _cross_source_document(source_id: str, passage: str) -> dict:
    return {
        "source_id": source_id,
        "manual_id": f"manual-{source_id}",
        "title": f"Title {source_id}",
        "issuing_organization": "NRL",
        "policy_scope": "goods",
        "revision_date": "2022-01-01",
        "as_of_date": "2022-01-01",
        "page": "1",
        "section": "Scope",
        "chunk_id": f"chunk-{source_id}",
        "passage": passage,
    }


def test_cross_ablation_answer_generator_renders_only_visible_source() -> None:
    source_a = _cross_source_document("source_a", "Source A exact evidence text.")
    source_b = _cross_source_document("source_b", "Source B exact evidence text.")
    trial = {
        "trial_id": "cross-ablation-1",
        "variant": "source_a_only",
        "record_id": "record-1",
        "question": "How do source A and source B interact?",
        "visible_source_documents": [source_a],
        "visible_source_ids": ["source_a"],
        "withheld_source_ids": ["source_b"],
    }
    prompt = CrossSourceAblationAnswerGenerator.prompt(SimpleNamespace(), trial)
    assert source_a["passage"] in prompt
    assert source_b["passage"] not in prompt
    assert "source_a_only" not in prompt and "source_b_only" not in prompt

    draft_citing_withheld_source = {
        "answerable": True,
        "answer": "A supported answer.",
        "claims": [
            {
                "statement": "A supported material claim.",
                "evidence": [{"source_id": "source_b", "quote": source_b["passage"]}],
            }
        ],
        "limitation_reason": "",
    }
    issues = cross_ablation_trial_validation_issues(draft_citing_withheld_source, trial)
    assert "trial_uses_non_visible_source" in issues

    parsed = CrossSourceAblationAnswerGenerator.parse(
        SimpleNamespace(model_name="test-model"),
        trial,
        CrossAblationTrialDraft.model_validate(draft_citing_withheld_source),
    )
    assert parsed[0]["deterministic_checks"]["passed"] is False
    assert "trial_uses_non_visible_source" in parsed[0]["deterministic_checks"]["issues"]


def test_cross_ablation_trials_are_three_blind_source_variants() -> None:
    source_a = _cross_source_document("source_a", "Source A exact evidence text.")
    source_b = _cross_source_document("source_b", "Source B exact evidence text.")
    candidate = {
        "record_id": "record-1",
        "question": "What follows from source A and source B together?",
        "task_type": "cross_document_qa",
        "required_source_ids": ["source_a", "source_b"],
        "source_documents": [source_a, source_b],
        "claims": [{"statement": "Canonical claim", "evidence": []}],
    }
    trials = build_cross_ablation_trial_inputs([candidate])
    assert [trial["variant"] for trial in trials] == [
        "full",
        "source_a_only",
        "source_b_only",
    ]
    assert [trial["visible_source_ids"] for trial in trials] == [
        ["source_a", "source_b"],
        ["source_a"],
        ["source_b"],
    ]
    prompts = [CrossSourceAblationAnswerGenerator.prompt(SimpleNamespace(), trial) for trial in trials]
    assert all("Canonical claim" not in prompt for prompt in prompts)
    assert all("source_a_only" not in prompt and "source_b_only" not in prompt for prompt in prompts)


def test_build_cross_ablation_trial_inputs_skips_candidates_missing_a_required_source() -> None:
    source_a = _cross_source_document("source_a", "Source A exact evidence text.")
    single_source_candidate = {
        "record_id": "record-single",
        "question": "What follows from source A alone?",
        "task_type": "cross_document_qa",
        "required_source_ids": ["source_a", "source_b"],
        "source_documents": [source_a],
        "claims": [],
    }
    assert build_cross_ablation_trial_inputs([single_source_candidate]) == []


def test_real_cross_ablation_adjudication_requires_full_claim_coverage() -> None:
    candidate = {
        "record_id": "record-1",
        "claims": [
            {"evidence": [{"source_id": "source_a"}]},
            {"evidence": [{"source_id": "source_b"}]},
        ],
    }

    def trial(variant: str, answerable: bool, source_ids: list[str]) -> dict:
        return {
            "record_id": "record-1",
            "variant": variant,
            "trial_output": {
                "answerable": answerable,
                "claims": ([{"evidence": [{"source_id": source_id} for source_id in source_ids]}] if source_ids else []),
            },
            "deterministic_checks": {"passed": True},
        }

    valid = [
        trial("full", True, ["source_a", "source_b"]),
        trial("source_a_only", False, []),
        trial("source_b_only", True, ["source_b"]),
    ]
    assert adjudicate_cross_ablation_trials([candidate], valid)[0]["passed"] is True

    same_source_leak = [*valid]
    # source_a_only wrongly reproduces the full answer, including a claim that
    # actually requires source_b (the "trivially answerable from source_a
    # alone" same-source leak this stage exists to catch).
    same_source_leak[1] = trial("source_a_only", True, ["source_a", "source_b"])
    result = adjudicate_cross_ablation_trials([candidate], same_source_leak)[0]
    assert result["passed"] is False
    assert "source_a_only_fully_covers_answer" in result["issues"]


def test_cross_ablation_gate_rejects_a_same_source_leak_the_imagined_judge_would_pass() -> None:
    # T14e regression: before this empirical gate existed, CrossDocumentJudge's
    # single imagined ablation judgment (unsupported_without_source_ids) was the
    # only signal deciding whether a cross-document candidate needed both
    # sources. A same-source leak -- where source_a alone can trivially answer
    # the question even though the imagined judge reported both sources
    # necessary -- went undetected and would have been exported. It must now be
    # caught by the real 3-trial empirical gate.
    candidate = {
        "record_id": "record-leak",
        "answerable": True,
        "claims": [
            {"evidence": [{"source_id": "source_a"}]},
            {"evidence": [{"source_id": "source_b"}]},
        ],
        # The old imagined ablation judgment: both sources reported necessary --
        # exactly what the pre-T14e gate alone would have accepted.
        "judge": {
            "unsupported_without_source_ids": ["source_a", "source_b"],
            "source_ablation_passed": True,
            "accepted": True,
        },
    }
    assert set(candidate["judge"]["unsupported_without_source_ids"]) == {"source_a", "source_b"}
    assert candidate["judge"]["source_ablation_passed"] is True

    def trial(variant: str, answerable: bool, source_ids: list[str]) -> dict:
        return {
            "record_id": "record-leak",
            "variant": variant,
            "trial_output": {
                "answerable": answerable,
                "claims": ([{"evidence": [{"source_id": source_id} for source_id in source_ids]}] if source_ids else []),
            },
            "deterministic_checks": {"passed": True},
        }

    # The actual source_a_only trial fully reproduces the answer using both
    # claims' evidence -- a same-source leak the imagined judge never tested.
    leaking_trials = [
        trial("full", True, ["source_a", "source_b"]),
        trial("source_a_only", True, ["source_a", "source_b"]),
        trial("source_b_only", False, []),
    ]
    adjudications = adjudicate_cross_ablation_trials([candidate], leaking_trials)
    assert adjudications[0]["passed"] is False
    assert "source_a_only_fully_covers_answer" in adjudications[0]["issues"]

    # Even though an independent judge reviewing only the actual trial bundle
    # accepted it, the failed deterministic adjudication alone must still
    # reject the candidate -- the gate requires both to pass.
    judged = [{"record_id": "record-leak", "judge": {"accepted": True, "score": 5}}]
    kept, rejected = apply_cross_ablation_gate([candidate], adjudications, judged)
    assert kept == []
    assert len(rejected) == 1
    assert rejected[0]["record_id"] == "record-leak"
    assert rejected[0]["empirical_ablation"]["passed"] is False
    assert "source_a_only_fully_covers_answer" in rejected[0]["empirical_ablation"]["issues"]


def test_cross_ablation_gate_keeps_a_genuinely_two_source_dependent_candidate() -> None:
    candidate = {
        "record_id": "record-valid",
        "answerable": True,
        "claims": [
            {"evidence": [{"source_id": "source_a"}]},
            {"evidence": [{"source_id": "source_b"}]},
        ],
    }

    def trial(variant: str, answerable: bool, source_ids: list[str]) -> dict:
        return {
            "record_id": "record-valid",
            "variant": variant,
            "trial_output": {
                "answerable": answerable,
                "claims": ([{"evidence": [{"source_id": source_id} for source_id in source_ids]}] if source_ids else []),
            },
            "deterministic_checks": {"passed": True},
        }

    valid_trials = [
        trial("full", True, ["source_a", "source_b"]),
        trial("source_a_only", False, []),
        trial("source_b_only", False, []),
    ]
    adjudications = adjudicate_cross_ablation_trials([candidate], valid_trials)
    assert adjudications[0]["passed"] is True
    judged = [{"record_id": "record-valid", "judge": {"accepted": True, "score": 5}}]
    kept, rejected = apply_cross_ablation_gate([candidate], adjudications, judged)
    assert rejected == []
    assert len(kept) == 1
    assert kept[0]["record_id"] == "record-valid"
    assert kept[0]["empirical_ablation"]["passed"] is True


def test_cross_ablation_gate_rejects_missing_judge_response() -> None:
    # A candidate whose deterministic adjudication passed but never received an
    # independent judge response (e.g. exhausted retries) must still be
    # rejected, not silently accepted on deterministic evidence alone.
    candidate = {"record_id": "record-missing", "answerable": True, "claims": []}
    adjudications = [{"record_id": "record-missing", "passed": True, "issues": []}]
    kept, rejected = apply_cross_ablation_gate([candidate], adjudications, [])
    assert kept == []
    assert rejected[0]["empirical_ablation"]["issues"] == ["missing_ablation_judge_response"]


def test_cross_ablation_judge_reviews_only_complete_actual_trial_bundles() -> None:
    candidate = {
        "record_id": "record-a",
        "question": "How do A and B apply?",
        "answer": "A and B apply.",
        "source_documents": [
            {"source_id": "source_a"},
            {"source_id": "source_b"},
        ],
        "claims": [
            {
                "statement": "A",
                "evidence": [{"source_id": "source_a", "quote": "A policy text."}],
            },
            {
                "statement": "B",
                "evidence": [{"source_id": "source_b", "quote": "B policy text."}],
            },
        ],
    }
    trials = [
        {
            "record_id": "record-a",
            "variant": variant,
            "trial_output": {"answerable": variant == "full", "claims": []},
        }
        for variant in ("full", "source_a_only", "source_b_only")
    ]
    inputs = build_cross_ablation_judge_inputs(
        [candidate],
        trials,
        [{"record_id": "record-a", "passed": True}],
    )
    assert len(inputs) == 1
    assert set(inputs[0]["actual_trials"]) == {
        "full",
        "source_a_only",
        "source_b_only",
    }
    prompt = object.__new__(CrossSourceAblationJudge).prompt(inputs[0])
    assert "ACTUAL OUTPUTS" in prompt


def test_cross_ablation_judge_rejects_when_source_a_alone_is_declared_complete() -> None:
    row = {
        "record_id": "record-a",
        "candidate": {
            "question": "How do A and B apply?",
            "answer": "A and B apply.",
            "claims": [],
            "source_documents": [],
        },
        "actual_trials": {},
    }
    decision = AblationJudgeDecision(
        record_id="record-a",
        full_context_supported=True,
        # source_a alone still fully answers the question: source_b was never
        # actually necessary, so this must be rejected even though every
        # other boolean below claims a valid, necessary experiment.
        source_a_only_incomplete=False,
        source_b_only_incomplete=True,
        comparison_valid=True,
        score=5,
        issues=[],
    )
    parsed = CrossSourceAblationJudge.parse(
        SimpleNamespace(model_name="judge-model"),
        row,
        decision,
    )
    assert parsed[0]["judge"]["accepted"] is False


def test_ablation_trial_validation_rejects_withheld_or_inexact_evidence() -> None:
    row = {
        "visible_propositions": [
            {
                "proposition_id": "prop-a",
                "evidence": {"quote": "Source A exact evidence."},
            }
        ]
    }
    valid = {
        "answerable": True,
        "answer": "A supported answer.",
        "claims": [
            {
                "statement": "A supported material claim.",
                "evidence": [
                    {
                        "proposition_id": "prop-a",
                        "quote": "Source A exact evidence.",
                    }
                ],
            }
        ],
        "limitation_reason": "",
    }
    assert ablation_trial_validation_issues(valid, row) == []

    withheld = json.loads(json.dumps(valid))
    withheld["claims"][0]["evidence"][0]["proposition_id"] = "prop-b"
    assert "trial_uses_non_visible_proposition" in ablation_trial_validation_issues(
        withheld,
        row,
    )

    inexact = json.loads(json.dumps(valid))
    inexact["claims"][0]["evidence"][0]["quote"] = "Paraphrased evidence."
    assert "non_exact_trial_evidence" in ablation_trial_validation_issues(
        inexact,
        row,
    )

    invalid_abstention = {
        "answerable": False,
        "answer": "",
        "claims": [],
        "limitation_reason": "",
    }
    assert "abstaining_trial_missing_limitation" in ablation_trial_validation_issues(
        invalid_abstention,
        row,
    )


def test_real_ablation_adjudication_requires_full_claim_coverage() -> None:
    answer = {
        "record_id": "record-1",
        "claims": [
            {"evidence": [{"proposition_id": "p1"}]},
            {"evidence": [{"proposition_id": "p2"}]},
        ],
    }

    def trial(variant: str, answerable: bool, proposition_ids: list[str]) -> dict:
        return {
            "record_id": "record-1",
            "variant": variant,
            "trial_output": {
                "answerable": answerable,
                "claims": ([{"evidence": [{"proposition_id": proposition_id} for proposition_id in proposition_ids]}] if proposition_ids else []),
            },
            "deterministic_checks": {"passed": True},
        }

    valid = [
        trial("full", True, ["p1", "p2"]),
        trial("source_a_only", False, []),
        trial("source_b_only", True, ["p2"]),
    ]
    assert adjudicate_ablation_trials([answer], valid)[0]["passed"] is True

    incomplete = [*valid]
    incomplete[0] = trial("full", True, ["p1"])
    result = adjudicate_ablation_trials([answer], incomplete)[0]
    assert result["passed"] is False
    assert "full_context_missing_required_claim_coverage" in result["issues"]


def test_cot_rejects_repeated_steps_with_identical_evidence() -> None:
    passage = "The buyer shall publish the notice and retain the record."
    record = {
        "task_type": "qa_cot",
        "answerable": True,
        "answer": "The buyer shall publish the notice and retain the record.",
        "claims": [
            {
                "statement": "The buyer shall publish the notice and retain the record.",
                "evidence": [{"quote": passage}],
            }
        ],
        "evidence": [{"quote": passage}],
        "reasoning_steps": [
            {
                "operation": "lookup",
                "statement": "Apply the stated rule.",
                "evidence_quotes": [passage],
            },
            {
                "operation": "lookup",
                "statement": "Apply the stated rule.",
                "evidence_quotes": [passage],
            },
        ],
    }
    issues = validate_record(record, passage)
    assert "cot_repeats_reasoning_step" in issues
    assert "cot_reuses_identical_evidence_for_all_steps" in issues


def test_grounded_blueprint_is_singular_and_auditable() -> None:
    row = {
        "planned_request_id": "single-request",
        "planned_task_type": "qa",
        "planned_question_type": "direct_fact",
        "planned_answer_format": "concise_direct",
        "planned_answerable": True,
        "passage": "The buyer shall retain the procurement record.",
    }
    result = ProcurementBlueprintGenerator.parse(
        SimpleNamespace(model_name="generator"),
        row,
        QABlueprintDraft(
            task="compliance_and_audit",
            persona="auditor",
            persona_need=("Check whether the procurement record was retained for audit."),
            instruction_goal="Determine which procurement record must be retained.",
            must_cover=["The buyer must retain the procurement record."],
            evidence=[{"quote": row["passage"]}],
        ),
    )
    assert result["blueprint_checks"] == {"passed": True, "issues": []}
    assert result["blueprint_id"].startswith("qabp-")
    assert result["task"] == "compliance_and_audit"

    swapped = ProcurementBlueprintGenerator.parse(
        SimpleNamespace(model_name="generator"),
        row,
        QABlueprintDraft(
            task="auditor",
            persona="compliance_and_audit",
            persona_need=("Check whether the procurement record was retained for audit."),
            instruction_goal="Determine which procurement record must be retained.",
            must_cover=["The buyer must retain the procurement record."],
            evidence=[{"quote": row["passage"]}],
        ),
    )
    assert swapped["task"] == "compliance_and_audit"
    assert swapped["persona"] == "auditor"
    assert swapped["blueprint_repairs"] == ["swapped_task_and_persona"]


def test_blueprint_realigns_whitespace_only_to_exact_source_span() -> None:
    passage = "The exceptions are:\n\n- first condition;\n- second condition."
    flattened = "The exceptions are: - first condition; - second condition."
    assert realign_whitespace_verbatim_quote(flattened, passage) == passage
    assert (
        realign_whitespace_verbatim_quote(
            "The exceptions are: - changed condition.", passage
        )
        is None
    )
    row = {
        "planned_request_id": "single-whitespace",
        "planned_task_type": "qa",
        "planned_question_type": "exception",
        "planned_answer_format": "rule_and_exception",
        "planned_answerable": True,
        "passage": passage,
    }
    result = ProcurementBlueprintGenerator.parse(
        SimpleNamespace(model_name="generator"),
        row,
        QABlueprintDraft(
            task="compliance_and_audit",
            persona="auditor",
            persona_need="Check which listed exception applies to the review.",
            instruction_goal="Identify both listed exception conditions.",
            must_cover=["Both conditions must be identified."],
            evidence=[{"quote": flattened}],
        ),
    )
    assert result["blueprint_checks"]["passed"] is True
    assert result["blueprint_evidence"] == [{"quote": passage}]
    assert result["blueprint_repairs"] == [
        "realigned_blueprint_evidence_whitespace:0"
    ]


def test_generation_validation_rescue_is_bounded_per_wholly_invalid_blueprint() -> None:
    inputs = [
        {
            "blueprint_id": "bp-invalid",
            "candidate_request_id": "bp-invalid-candidate-01",
            "candidate_index": 1,
        },
        {
            "blueprint_id": "bp-invalid",
            "candidate_request_id": "bp-invalid-candidate-02",
            "candidate_index": 2,
        },
        {
            "blueprint_id": "bp-valid",
            "candidate_request_id": "bp-valid-candidate-01",
            "candidate_index": 1,
        },
    ]
    audit = [
        {
            "record_id": "invalid-1",
            "blueprint_id": "bp-invalid",
            "candidate_request_id": "bp-invalid-candidate-01",
            "question": "Question one?",
            "answer": "Unsupported answer.",
            "claims": [],
            "reasoning_steps": [],
            "deterministic_checks": {
                "passed": False,
                "issues": ["unsupported_number:10"],
            },
        },
        {
            "record_id": "invalid-2",
            "blueprint_id": "bp-invalid",
            "candidate_request_id": "bp-invalid-candidate-02",
            "deterministic_checks": {
                "passed": False,
                "issues": ["incomplete_evidence_fragment"],
            },
        },
        {
            "record_id": "valid",
            "blueprint_id": "bp-valid",
            "candidate_request_id": "bp-valid-candidate-01",
            "deterministic_checks": {"passed": True, "issues": []},
        },
    ]
    rescue = build_generation_validation_rescue_inputs(inputs, audit)
    assert len(rescue) == 1
    assert rescue[0]["candidate_request_id"].endswith("-validation-rescue")
    assert rescue[0]["validation_rescue_of"] == "invalid-1"
    assert rescue[0]["validation_rescue_issues"] == ["unsupported_number:10"]


def test_manifest_metadata_and_stable_chunk(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manual.md").write_text(
        "<!-- Page 1 -->\n# Rule\n" + "The buyer shall retain this record. " * 10,
        encoding="utf-8",
    )
    (source / "manuals.yaml").write_text(
        """manuals:
  - manual_id: test_manual
    title: Test Manual
    file: manual.md
    release_date: "2026"
    revision_date: "2026"
    as_of_date: "2026"
    start_page: 1
    exclude_pages: []
""",
        encoding="utf-8",
    )
    first, manuals = load_corpus(source, tmp_path / "ocr")
    second, _ = load_corpus(source, tmp_path / "ocr")
    assert first[0]["chunk_id"] == second[0]["chunk_id"]
    assert first[0]["issuing_organization"] == "Government of India"
    assert manuals[0]["policy_scope"] == "government_reference"
    assert first[0]["document_order"] == 1
    assert first[0]["section_path"] == ["Rule"]


def _window_chunk(
    chunk_id: str,
    order: int,
    page: int,
    section_path: list[str],
    passage: str = "A grounded procurement provision.",
) -> dict:
    return {
        "manual_id": "manual",
        "title": "Manual",
        "issuing_organization": "Government of India",
        "policy_scope": "government_reference",
        "revision_date": "2026",
        "as_of_date": "2026",
        "source_sha256": "a" * 64,
        "chunk_id": chunk_id,
        "page": page,
        "document_order": order,
        "section": section_path[-1] if section_path else None,
        "section_path": section_path,
        "passage": passage,
        "generation_passage": passage,
    }


def test_source_windows_preserve_adjacency_boundaries_and_provenance() -> None:
    chunks = [
        _window_chunk("c1", 1, 1, ["Evaluation"]),
        _window_chunk("c2", 2, 2, ["Evaluation"]),
        _window_chunk("c3", 3, 2, ["Award"]),
        _window_chunk("c4", 4, 3, ["General"]),
    ]
    accepted, rejected = build_source_windows(
        chunks,
        {
            "max_chunks": 3,
            "max_input_tokens": 1000,
            "reserved_prompt_tokens": 100,
            "conservative_chars_per_token": 2.5,
        },
    )
    assert rejected == []
    assert [row["chunk_ids"] for row in accepted] == [
        ["c1", "c2"],
        ["c3", "c4"],
    ]
    assert accepted[0]["pages"] == [1, 2]
    assert accepted[0]["boundary_confidence"] == "explicit_markdown_heading"
    assert accepted[0]["token_budget"]["passed"] is True
    assert accepted[0]["chunks"][0]["passage"] == chunks[0]["passage"]


def test_source_windows_split_by_bound_and_reject_oversize_chunk() -> None:
    chunks = [
        _window_chunk("c1", 1, 1, ["Evaluation"], "a" * 50),
        _window_chunk("c2", 2, 1, ["Evaluation"], "b" * 50),
        _window_chunk("c3", 3, 1, ["Evaluation"], "c" * 500),
    ]
    accepted, rejected = build_source_windows(
        chunks,
        {
            "max_chunks": 1,
            "max_input_tokens": 100,
            "reserved_prompt_tokens": 20,
            "conservative_chars_per_token": 2.5,
        },
    )
    assert [row["chunk_ids"] for row in accepted] == [["c1"], ["c2"]]
    assert rejected[0]["chunk_ids"] == ["c3"]
    assert rejected[0]["rejection_reasons"] == ["source_chunk_exceeds_token_budget"]


def test_component_references_resolve_only_unique_same_manual_targets() -> None:
    chunks = [
        _window_chunk(
            "anchor",
            1,
            1,
            ["Evaluation"],
            "5.6.8 The committee shall record its recommendation.",
        ),
        _window_chunk(
            "source",
            2,
            2,
            ["Award"],
            "Apply para 5.6.8 and para 9.9.9 before award.",
        ),
    ]
    audit = resolve_component_references(chunks)
    assert audit["source"][0]["status"] == "resolved"
    assert audit["source"][0]["target_chunk_ids"] == ["anchor"]
    assert audit["source"][1]["status"] == "missing"
    windows, _ = build_source_windows(
        chunks,
        {
            "max_chunks": 4,
            "max_input_tokens": 1000,
            "reserved_prompt_tokens": 100,
            "conservative_chars_per_token": 2.5,
        },
    )
    source_window = next(row for row in windows if "source" in row["chunk_ids"])
    assert source_window["support_edges"][0]["target_chunk_ids"] == ["anchor"]
    assert len(source_window["reference_audit"]) == 2


def test_component_references_do_not_resolve_ambiguous_targets() -> None:
    chunks = [
        _window_chunk("a1", 1, 1, ["A"], "5.6.8 First copy."),
        _window_chunk("a2", 2, 2, ["B"], "5.6.8 Duplicate copy."),
        _window_chunk("source", 3, 3, ["C"], "See clause 5.6.8."),
    ]
    audit = resolve_component_references(chunks)
    assert audit["source"][0]["status"] == "ambiguous"
    assert audit["source"][0]["target_chunk_ids"] == []


class _FakeTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(12))

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(range(5))


def test_rendered_prompt_budget_uses_template_and_schema_tokens() -> None:
    result = measure_rendered_request(
        [{"role": "user", "content": "Question"}],
        {"type": "object"},
        context_window=100,
        reserved_completion_tokens=50,
        safety_margin_tokens=10,
        conservative_chars_per_token=2.5,
        tokenizer=_FakeTokenizer(),
        tokenizer_identity="local-tokenizer",
        tokenizer_revision="rev",
        require_exact=True,
    )
    assert result["method"] == "tokenizer_chat_template"
    assert result["prompt_tokens"] == 17
    assert result["passed"] is True
    assert result["chat_template_sha256"]


def test_prompt_budget_fallback_is_labeled_and_exact_mode_fails() -> None:
    fallback = measure_rendered_request(
        [{"role": "user", "content": "x" * 100}],
        {"type": "object"},
        context_window=20,
        reserved_completion_tokens=5,
        safety_margin_tokens=2,
        conservative_chars_per_token=2,
    )
    assert fallback["method"] == "conservative_character_estimate"
    assert fallback["passed"] is False
    try:
        measure_rendered_request(
            [{"role": "user", "content": "Question"}],
            {},
            context_window=100,
            reserved_completion_tokens=10,
            safety_margin_tokens=5,
            conservative_chars_per_token=2.5,
            require_exact=True,
        )
    except ValueError as exc:
        assert "local tokenizer" in str(exc)
    else:
        raise AssertionError("exact prompt counting must fail without tokenizer")


def test_endpoint_prompt_budget_uses_smaller_server_context() -> None:
    result = measure_rendered_request(
        [{"role": "user", "content": "Question"}],
        {"type": "object", "description": "not rendered"},
        context_window=16_384,
        reserved_completion_tokens=100,
        safety_margin_tokens=20,
        conservative_chars_per_token=2.5,
        include_response_schema=False,
        exact_prompt_tokens=900,
        server_context_window=1_000,
    )

    assert result["method"] == "vllm_tokenize_endpoint"
    assert result["prompt_tokens"] == 900
    assert result["configured_context_window"] == 16_384
    assert result["server_context_window"] == 1_000
    assert result["context_window"] == 1_000
    assert result["passed"] is False


def test_vllm_tokenize_chat_uses_root_route_and_template_inputs(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"count": 42, "max_model_len": 8192, "tokens": []}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("prompt_budget.urlopen", fake_urlopen)
    result = vllm_tokenize_chat(
        [{"role": "user", "content": "Question"}],
        model="/models/judge",
        base_url="http://127.0.0.1:8000/v1",
        api_key="secret",
        chat_template_kwargs={"enable_thinking": False},
        tools=[{"type": "function", "function": {"name": "Result"}}],
    )

    assert result == {"count": 42, "max_model_len": 8192}
    assert captured["url"] == "http://127.0.0.1:8000/tokenize"
    assert captured["payload"]["model"] == "/models/judge"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["tools"][0]["function"]["name"] == "Result"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 30.0


def test_model_context_window_is_explicit_and_profile_local() -> None:
    nemotron = generation_pipeline.CONFIG["model_profiles"]["nemotron"]
    source_windows = generation_pipeline.CONFIG["source_windows"]

    assert configured_context_window(nemotron) == 131072
    assert nemotron["structured_output_mode"] == "tools_auto"
    assert generation_pipeline.CONFIG["model_profiles"]["glm"]["structured_output_mode"] == "json_schema"
    assert configured_context_window(generation_pipeline.CONFIG["model_profiles"]["glm"]) == 32768
    assert generation_pipeline.CONFIG["model_profiles"]["gemma"]["structured_output_mode"] == "json_schema"
    assert source_windows["max_input_tokens"] == 8192
    for invalid in ({}, {"context_window": 0}, {"context_window": True}):
        try:
            configured_context_window(invalid)
        except ValueError as exc:
            assert "positive context_window" in str(exc)
        else:
            raise AssertionError("missing or invalid model context must fail closed")


def test_manifest_marks_source_windows_as_currently_unconsumed() -> None:
    """Regression test for the dead source_windows stage (audit Track B, T6).

    build_source_windows() computes bounded multi-chunk windows, but no
    generation stage currently reads them -- the manifest must say so
    explicitly rather than silently looking like an active QC gate.
    """

    def manifest(source_window_stats: dict | None) -> dict:
        return generation_pipeline._final_manifest(
            run_id="pilot-t6",
            status="complete",
            stats={},
            manuals=[],
            corpus_report={},
            selected_rows=[],
            single_coverage={},
            cross_coverage={},
            drafting_stats={"enabled": False},
            duplicates=0,
            source_window_stats=source_window_stats,
        )

    disabled = manifest(None)
    assert disabled["source_windows"] == {"enabled": False}

    enabled = manifest(
        {
            "enabled": True,
            "accepted": 3,
            "rejected": 1,
            "schema_version": "1",
            "consumed_by": [],
        }
    )
    assert enabled["source_windows"]["enabled"] is True
    assert enabled["source_windows"]["consumed_by"] == []


def test_judge_prompt_budget_reserves_output_and_quarantines_overflow() -> None:
    class Response:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "properties": {"accepted": {"type": "boolean"}}}

    judge = SimpleNamespace(
        response_format=Response,
        prompt=lambda row: row["review_text"],
    )
    profile = {
        "context_window": 100,
        "generation_params": {"max_tokens": 40},
    }
    row = {
        "review_text": "x" * 1000,
        "judge_items": [
            {
                "record_id": "record-1",
                "record": {"record_id": "record-1"},
            }
        ],
    }

    budget = _judge_prompt_budget(judge, row, profile)
    accepted, rejected = _budget_judge_rows(judge, [row], profile)

    assert budget["reserved_completion_tokens"] == 40
    assert budget["passed"] is False
    assert accepted == []
    assert rejected[0]["judge"]["issues"] == ["judge_prompt_exceeds_context_window"]
    assert rejected[0]["judge_prompt_budget"] == budget


def test_judge_prompt_budget_matches_selected_transport(monkeypatch) -> None:
    class Response(BaseModel):
        accepted: bool

    judge = SimpleNamespace(
        response_format=Response,
        prompt=lambda row: row["review_text"],
    )
    captured = {}

    def fake_tokenize(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return {"count": 123, "max_model_len": 8_000}

    monkeypatch.setattr(generation_pipeline, "vllm_tokenize_chat", fake_tokenize)
    monkeypatch.setattr(
        generation_pipeline,
        "_model_settings",
        lambda _profile: ("served-model", "http://127.0.0.1:8000/v1", "key"),
    )
    profile = {
        "context_window": 10_000,
        "structured_output_mode": "tools_auto",
        "generation_params": {
            "max_tokens": 200,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    }
    budget = _judge_prompt_budget(
        judge,
        {"review_text": "review", "judge_items": []},
        profile,
    )

    assert budget["method"] == "vllm_tokenize_endpoint"
    assert budget["prompt_tokens"] == 123
    assert budget["context_window"] == 8_000
    assert budget["structured_output_mode"] == "tools_auto"
    assert budget["measurement_error"] is None
    assert captured["messages"][0]["role"] == "system"
    assert captured["tools"][0]["function"]["name"] == "Response"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_validation_rejects_unsupported_number() -> None:
    record = {
        "task_type": "qa",
        "question": "How long must the buyer retain the record?",
        "answer": "The buyer must retain it for 10 years.",
        "answerable": True,
        "evidence": [{"quote": "The buyer shall retain it for 5 years."}],
        "reasoning_steps": [],
    }
    assert "unsupported_number:10 years" in validate_record(record, "The buyer shall retain it for 5 years.")


def test_validation_rejects_number_misattributed_to_the_wrong_claim() -> None:
    """A number correct for one entity but misattributed to another must not pass.

    Joining every claim's evidence before checking the answer would let this
    slip through, because "5 days" genuinely appears in Entity B's evidence —
    just not in Entity A's, which is the claim the answer actually attaches it
    to. See audit T7 / Finding V1.
    """
    passage = "Entity A shall respond within 3 days. Entity B shall decide within 5 days."
    record = {
        "task_type": "qa",
        "question": "How long do Entity A and Entity B have?",
        "answer": "Entity A requires 5 days to respond, and Entity B requires 5 days to decide.",
        "answerable": True,
        "evidence": [
            {"quote": "Entity A shall respond within 3 days."},
            {"quote": "Entity B shall decide within 5 days."},
        ],
        "claims": [
            {
                "statement": "Entity A requires 3 days to respond.",
                "evidence": [{"quote": "Entity A shall respond within 3 days."}],
            },
            {
                "statement": "Entity B requires 5 days to decide.",
                "evidence": [{"quote": "Entity B shall decide within 5 days."}],
            },
        ],
        "reasoning_steps": [],
    }
    reasons = validate_record(record, passage)
    assert "unsupported_number:5 days" in reasons


def test_validation_still_accepts_correctly_attributed_multi_claim_numbers() -> None:
    """Same shape as the misattribution test, but every number is correct.

    Guards against the scoped check becoming stricter than the prior
    union-of-all-evidence behavior on legitimate multi-claim answers.
    """
    passage = "Entity A shall respond within 3 days. Entity B shall decide within 5 days."
    record = {
        "task_type": "qa",
        "question": "How long do Entity A and Entity B have?",
        "answer": "Entity A requires 3 days to respond, and Entity B requires 5 days to decide.",
        "answerable": True,
        "evidence": [
            {"quote": "Entity A shall respond within 3 days."},
            {"quote": "Entity B shall decide within 5 days."},
        ],
        "claims": [
            {
                "statement": "Entity A requires 3 days to respond.",
                "evidence": [{"quote": "Entity A shall respond within 3 days."}],
            },
            {
                "statement": "Entity B requires 5 days to decide.",
                "evidence": [{"quote": "Entity B shall decide within 5 days."}],
            },
        ],
        "reasoning_steps": [],
    }
    reasons = validate_record(record, passage)
    assert not any(reason.startswith("unsupported_number") for reason in reasons)


def _proposition_source_row() -> dict:
    passage = "If delivery is delayed, the buyer shall recover liquidated damages " "at 0.5% per week, except where force majeure applies."
    return {
        "manual_id": "nrl_goods_rev1",
        "title": "NRL Manual for Procurement of Goods, Rev1",
        "issuing_organization": "Numaligarh Refinery Limited",
        "policy_scope": "company_policy",
        "revision_date": "16.03.2023",
        "as_of_date": "16.03.2023",
        "source_file": "manual.md",
        "source_sha256": "a" * 64,
        "chunk_id": "chunk-1",
        "page": 7,
        "section": "Liquidated damages",
        "passage": passage,
    }


def _proposition_draft() -> dict:
    return {
        "subject": "the buyer",
        "action": "shall recover",
        "object": "liquidated damages",
        "modality": "mandatory",
        "polarity": "positive",
        "conditions": ["If delivery is delayed"],
        "exceptions": ["except where force majeure applies"],
        "threshold_value": "0.5%",
        "threshold_unit": "per week",
        "temporal_scope": "",
        "evidence_quote": ("If delivery is delayed, the buyer shall recover liquidated damages " "at 0.5% per week, except where force majeure applies."),
    }


def test_proposition_materialization_preserves_authority_and_offsets() -> None:
    row = _proposition_source_row()
    draft = _proposition_draft()
    assert proposition_validation_issues(draft, row) == []
    result = materialize_proposition(draft, row, "fingerprint")
    assert result["proposition_id"].startswith("prop-")
    assert result["authority"]["issuing_organization"] == "Numaligarh Refinery Limited"
    assert result["authority"]["policy_scope"] == "company_policy"
    assert result["authority"]["revision_date"] == "16.03.2023"
    assert result["authority"]["as_of_date"] == "16.03.2023"
    assert result["evidence"]["quote"] == row["passage"]
    assert result["evidence"]["start_char"] == 0
    assert result["evidence"]["end_char"] == len(row["passage"])
    assert result["deterministic_checks"]["passed"] is True


def test_proposition_allows_grounded_clause_without_separate_object() -> None:
    passage = "Relevant format for BG should be provided in the tender document."
    row = {
        **_proposition_source_row(),
        "passage": passage,
    }
    draft = {
        "subject": "Relevant format for BG",
        "action": "should be provided in the tender document",
        "object": "",
        "modality": "recommended",
        "polarity": "positive",
        "conditions": [],
        "exceptions": [],
        "threshold_value": "",
        "threshold_unit": "",
        "temporal_scope": "",
        "evidence_quote": passage,
    }

    parsed = PropositionBatch(propositions=[draft])
    assert parsed.propositions[0].object == ""
    assert proposition_validation_issues(draft, row) == []
    result = materialize_proposition(draft, row, "fingerprint")
    assert result["object"] == ""
    assert result["deterministic_checks"]["passed"] is True


def test_proposition_offsets_resolve_against_original_source_chunk() -> None:
    row = _proposition_source_row()
    row["source_passage"] = f"![page image](image.png)\n\n{row['passage']}"
    result = materialize_proposition(_proposition_draft(), row, "fingerprint")
    assert result["deterministic_checks"]["passed"] is True
    assert result["evidence"]["start_char"] == len("![page image](image.png)\n\n")
    assert row["source_passage"][result["evidence"]["start_char"] : result["evidence"]["end_char"]] == result["evidence"]["quote"]


def test_proposition_validation_rejects_semantic_and_location_drift() -> None:
    row = _proposition_source_row()
    unsupported = {
        **_proposition_draft(),
        "action": "may recover",
        "modality": "permitted",
    }
    issues = proposition_validation_issues(unsupported, row)
    assert "non_verbatim_action" in issues
    assert "unsupported_modality" in issues

    duplicate_row = {**row, "passage": f"{row['passage']} {row['passage']}"}
    assert "ambiguous_evidence_occurrence" in proposition_validation_issues(
        _proposition_draft(),
        duplicate_row,
    )


def test_proposition_cache_fingerprint_and_round_trip(tmp_path: Path) -> None:
    row = _proposition_source_row()
    model = {
        "profile": "glm",
        "model": "local-model",
        "base_url": "http://private/v1",
        "generation_params": {"temperature": 1.0},
    }
    first = proposition_cache_fingerprint(row, model)
    assert first == proposition_cache_fingerprint(dict(row), dict(model))
    assert first != proposition_cache_fingerprint(
        {**row, "passage": row["passage"] + " Changed."},
        model,
    )
    assert first != proposition_cache_fingerprint(
        row,
        {**model, "model": "another-model"},
    )

    record = materialize_proposition(_proposition_draft(), row, first)
    write_proposition_cache(tmp_path, [record])
    cached, hits = read_cached_propositions(tmp_path, {first, "missing"})
    assert hits == {first}
    assert cached == [record]

    empty_fingerprint = "e" * 64
    empty = {
        "proposition_id": "",
        "cache_fingerprint": empty_fingerprint,
        "empty_extraction": True,
        "source_chunk_id": "chunk-empty",
        "schema_version": "1",
        "deterministic_checks": {"passed": True, "issues": []},
    }
    write_proposition_cache(tmp_path, [empty])
    cached_empty, empty_hits = read_cached_propositions(
        tmp_path,
        {empty_fingerprint},
    )
    assert empty_hits == {empty_fingerprint}
    assert cached_empty == [empty]


def test_proposition_and_empty_extraction_share_arrow_schema(tmp_path: Path) -> None:
    from datasets import Dataset
    from datasets.arrow_writer import ArrowWriter

    row = _proposition_source_row()
    fingerprint = "f" * 64
    proposition = materialize_proposition(_proposition_draft(), row, fingerprint)
    empty = materialize_empty_extraction(row, fingerprint)

    assert proposition.keys() == empty.keys()
    assert proposition["empty_extraction"] is False
    assert empty["empty_extraction"] is True

    arrow_path = tmp_path / "mixed-propositions.arrow"
    with ArrowWriter(path=str(arrow_path)) as writer:
        writer.write(empty)
        writer.write(proposition)
        writer.finalize()
    dataset = Dataset.from_file(str(arrow_path))

    assert len(dataset) == 2
    assert dataset["empty_extraction"] == [True, False]


def test_proposition_extractor_uses_full_schema_for_empty_batch() -> None:
    row = {
        **_proposition_source_row(),
        "proposition_cache_fingerprint": "f" * 64,
    }
    extractor = object.__new__(PropositionExtractor)
    parsed = extractor.parse(row, PropositionBatch(propositions=[]))

    assert parsed == [
        materialize_empty_extraction(
            row,
            row["proposition_cache_fingerprint"],
        )
    ]


def _path_proposition(
    proposition_id: str,
    manual_id: str,
    title: str,
    issuer: str,
    as_of_date: str,
    *,
    subject: str = "procurement committee",
    action: str = "evaluates",
    object_: str = "technical bid",
    conditions: list[str] | None = None,
    exceptions: list[str] | None = None,
    chunk_id: str | None = None,
) -> dict:
    return {
        "proposition_id": proposition_id,
        "subject": subject,
        "authority": {
            "manual_id": manual_id,
            "manual_title": title,
            "issuing_organization": issuer,
            "policy_scope": "government_reference",
            "revision_date": as_of_date,
            "as_of_date": as_of_date,
        },
        "action": action,
        "object": object_,
        "modality": "stated",
        "polarity": "positive",
        "conditions": conditions or [],
        "exceptions": exceptions or [],
        "threshold": {"value": "", "unit": ""},
        "temporal_scope": "",
        "evidence": {
            "source_file": f"{manual_id}.md",
            "source_sha256": "a" * 64,
            "chunk_id": chunk_id or f"chunk-{proposition_id}",
            "page": 1,
            "section": "Evaluation",
            "quote": "Source evidence.",
            "start_char": 0,
            "end_char": 16,
        },
        "schema_version": "2",
        "cache_fingerprint": "b" * 64,
        "deterministic_checks": {"passed": True, "issues": []},
    }


def test_reasoning_paths_are_connected_stable_and_source_distinct() -> None:
    left = _path_proposition(
        "prop-left",
        "goods_2017",
        "Goods 2017",
        "Government of India",
        "2017",
    )
    right = _path_proposition(
        "prop-right",
        "goods_2024",
        "Goods 2024",
        "Government of India",
        "2024",
        conditions=["when technical evaluation is complete"],
    )
    config = {
        "pairs": [
            {
                "pair_id": "goods-temporal",
                "left_manual": "goods_2017",
                "right_manual": "goods_2024",
                "relationship_type": "same_authority_temporal",
            }
        ]
    }

    first, rejected = build_reasoning_paths([right, left], config, 5)
    second, _ = build_reasoning_paths([left, right], config, 5)

    assert rejected == []
    assert first == second
    assert len(first) == 1
    path = first[0]
    assert path["relationship_type"] == "exception_condition_interaction"
    assert path["input_claim_ids"] == ["prop-left", "prop-right"]
    assert len(set(path["required_source_ids"])) == 2
    assert path["output_claim_id"] == path["output_claim"]["claim_id"]
    assert path["operation_steps"][0]["output_claim_id"] == "prop-left"
    assert path["operation_steps"][1]["output_claim_id"] == "prop-right"
    assert path["operation_steps"][-1]["output_claim_id"] == path["output_claim_id"]
    assert path["deterministic_checks"]["passed"] is True
    assert all(result["complete"] is False for result in path["declared_requirement"].values())


def test_reasoning_paths_reject_unrelated_and_unsafe_relationship_claims() -> None:
    left = _path_proposition(
        "prop-left",
        "goods_2017",
        "Goods 2017",
        "Government of India",
        "2017",
    )
    unrelated = _path_proposition(
        "prop-other",
        "goods_2024",
        "Goods 2024",
        "Government of India",
        "2024",
        subject="contract manager",
        action="closes",
        object_="purchase order",
    )
    pair = {
        "pair_id": "goods-temporal",
        "left_manual": "goods_2017",
        "right_manual": "goods_2024",
        "relationship_type": "same_authority_temporal",
    }
    accepted, rejected = build_reasoning_paths(
        [left, unrelated],
        {"pairs": [pair]},
        5,
    )
    assert accepted == []
    assert rejected == []

    right = _path_proposition(
        "prop-right",
        "goods_2024",
        "Goods 2024",
        "Government of India",
        "2024",
    )
    accepted, _ = build_reasoning_paths([left, right], {"pairs": [pair]}, 5)
    path = accepted[0]
    path["output_claim"]["statement"] = "Goods 2024 supersedes Goods 2017."
    issues = validate_reasoning_path(
        path,
        {"prop-left": left, "prop-right": right},
        pair,
    )
    assert "unsupported_legal_relationship_claim" in issues

    path["operation_steps"][-1]["input_claim_ids"] = ["unknown"]
    issues = validate_reasoning_path(
        path,
        {"prop-left": left, "prop-right": right},
        pair,
    )
    assert "invalid_operation_graph" in issues


def test_reasoning_bridge_requires_an_exact_non_generic_entity() -> None:
    left = _path_proposition(
        "prop-left",
        "manual_a",
        "Manual A",
        "Government of India",
        "2024",
        object_="technical evaluation report",
    )
    right = _path_proposition(
        "prop-right",
        "manual_b",
        "Manual B",
        "Government of India",
        "2024",
        subject="technical evaluation report",
        action="supports",
        object_="award recommendation",
    )
    pair = {
        "pair_id": "procedure",
        "left_manual": "manual_a",
        "right_manual": "manual_b",
        "relationship_type": "complementary_procedure",
    }
    accepted, rejected = build_reasoning_paths(
        [left, right],
        {"pairs": [pair]},
        5,
    )
    assert rejected == []
    assert accepted[0]["relationship_type"] == "complementary_procedure"
    assert "technical" in accepted[0]["connection_anchors"]
    assert "evaluation" in accepted[0]["connection_anchors"]


def _verified_path_question_row() -> dict:
    left = _path_proposition(
        "prop-left",
        "goods_2017",
        "Goods Manual 2017",
        "Government of India",
        "2017",
    )
    right = _path_proposition(
        "prop-right",
        "goods_2024",
        "Goods Manual 2024",
        "Government of India",
        "2024",
    )
    pair = {
        "pair_id": "goods-temporal",
        "left_manual": "goods_2017",
        "right_manual": "goods_2024",
        "relationship_type": "same_authority_temporal",
    }
    paths, _ = build_reasoning_paths([left, right], {"pairs": [pair]}, 1)
    return {"path": paths[0], "propositions": [left, right]}


def test_path_question_requires_standalone_dates_and_rejects_answer_leakage() -> None:
    row = _verified_path_question_row()
    valid = {
        "task": "currentness",
        "persona": "procurement_officer",
        "question_type": "temporal",
        "difficulty": "advanced",
        "question": (
            "How does the procurement committee's technical-bid evaluation "
            "differ between the Goods Manual 2017 state as of 2017 and the "
            "Goods Manual 2024 state as of 2024?"
        ),
    }
    assert question_validation_issues(valid, row) == []
    missing_date = {
        **valid,
        "question": "How do the two Goods Manual rules differ?",
    }
    assert "missing_standalone_date" in question_validation_issues(
        missing_date,
        row,
    )
    leaked = {
        **valid,
        "question": row["path"]["output_claim"]["statement"] + "?",
    }
    assert "output_claim_leaked_into_question" in question_validation_issues(
        leaked,
        row,
    )


def test_path_answer_requires_exact_evidence_from_every_input() -> None:
    row = _verified_path_question_row()
    draft = {
        "answer": "The two dated source states describe the evaluation rule.",
        "claims": [
            {
                "statement": "The 2017 state contains the first rule.",
                "evidence": [
                    {
                        "proposition_id": "prop-left",
                        "quote": "Source evidence.",
                    }
                ],
            },
            {
                "statement": "The 2024 state contains the second rule.",
                "evidence": [
                    {
                        "proposition_id": "prop-right",
                        "quote": "Source evidence.",
                    }
                ],
            },
        ],
        "rationale_steps": [],
    }
    assert answer_validation_issues(draft, row) == []
    draft["claims"][1]["evidence"][0]["quote"] = "Changed evidence."
    issues = answer_validation_issues(draft, row)
    assert "non_exact_answer_evidence" in issues
    assert "answer_does_not_use_every_path_input" in issues


def test_missing_hop_and_false_premise_lineage_are_separate() -> None:
    row = _verified_path_question_row()
    question = {
        **row,
        "question_id": "pathq-1",
        "path_id": row["path"]["path_id"],
        "question": "How do the two dated rules differ?",
    }
    contrasts = build_missing_hop_contrasts([question])
    assert len(contrasts) == 2
    assert {item["withheld_proposition_id"] for item in contrasts} == {"prop-left", "prop-right"}
    assert all(item["answer"] == "Not answerable from the provided sources." for item in contrasts)
    quarantine = false_premise_quarantine([question])
    assert quarantine[0]["status"] == "quarantined"
    assert quarantine[0]["reason"] == "contradiction_verifier_not_implemented"


def test_validation_uses_qualifier_tokens_and_modality_equivalence() -> None:
    passage = "You are requested to note that the contract shall be cancelled. " "The bidder shall be put on the holiday list."
    consequence = {
        "task_type": "qa",
        "question": "What is the consequence?",
        "answer": "The contract shall be cancelled.",
        "answerable": True,
        "evidence": [{"quote": ("You are requested to note that the contract shall be cancelled.")}],
        "reasoning_steps": [],
    }
    assert "dropped_qualifier:not" not in validate_record(consequence, passage)

    categorical = {
        **consequence,
        "question": "What happens to the bidder?",
        "answer": "The bidder is put on the holiday list.",
        "evidence": [{"quote": "The bidder shall be put on the holiday list."}],
    }
    assert "dropped_qualifier:shall" not in validate_record(categorical, passage)

    weakened = {**categorical, "answer": "The bidder may be put on the holiday list."}
    assert "dropped_qualifier:shall" in validate_record(weakened, passage)


def test_semantic_support_rejects_absence_and_deontic_drift() -> None:
    assert semantic_support_issues(
        "The buyer shall deduct liquidated damages.",
        "The buyer may recover liquidated damages.",
    ) == ["strengthened_modality:permission_to_obligation"]
    assert semantic_support_issues(
        "The envelopes must be sealed separately.",
        "The envelopes should be sealed separately.",
    ) == ["strengthened_modality:recommendation_to_obligation"]
    assert semantic_support_issues(
        "The provision was not present in the 2019 Manual.",
        "The contractor is liable to pay liquidated damages.",
    ) == ["unsupported_absence_claim"]
    assert (
        semantic_support_issues(
            "There is no provision for consortium registration.",
            "There is no provision for registration of Consortium.",
        )
        == []
    )
    assert (
        semantic_support_issues(
            "The supplier must deliver the goods.",
            "The supplier shall deliver the goods.",
        )
        == []
    )


def test_cross_claim_contradiction_flags_opposite_modality_same_subject() -> None:
    """T11: two claims about the same core subject asserting opposite modalities."""
    assert cross_claim_contradiction_issues(
        [
            ("claim:0", "The evaluation committee must include an external member."),
            ("claim:1", "The evaluation committee must not include an external member."),
        ]
    ) == ["cross_claim_contradiction:claim:0:claim:1"]


def test_cross_claim_contradiction_ignores_strength_differences() -> None:
    """Obligation vs. permission/recommendation is a strength gap, not a
    contradiction -- already covered by `semantic_support_issues` elsewhere.
    """
    assert (
        cross_claim_contradiction_issues(
            [
                ("claim:0", "The vendor must submit the bank guarantee."),
                ("claim:1", "The vendor may submit the bank guarantee."),
            ]
        )
        == []
    )


def test_cross_claim_contradiction_ignores_different_subjects() -> None:
    """Opposite modalities about genuinely different subjects must not fire."""
    assert (
        cross_claim_contradiction_issues(
            [
                ("claim:0", "The vendor must submit the bank guarantee."),
                ("claim:1", "The buyer must not accept a late tender."),
            ]
        )
        == []
    )


def test_cross_claim_contradiction_ignores_short_core_subjects() -> None:
    """Near-empty core text after stripping deontic markers is not a
    meaningful "same subject" signal (guarded by `minimum_subject_words`)."""
    assert (
        cross_claim_contradiction_issues(
            [
                ("claim:0", "This must apply."),
                ("claim:1", "This must not apply."),
            ]
        )
        == []
    )


def test_validate_record_rejects_contradictory_claims() -> None:
    """End-to-end: `validate_record` rejects a record whose two claims
    assert opposite modalities about the same subject (constructed per T11's
    audit reference, Finding V2)."""
    passage = (
        "The evaluation committee must include an external member. "
        "The evaluation committee must not include an external member "
        "for procurements below the threshold."
    )
    record = {
        "task_type": "qa",
        "question": "Must the evaluation committee include an external member?",
        "answer": "It depends on the procurement value.",
        "answerable": True,
        "evidence": [
            {"quote": "The evaluation committee must include an external member."},
        ],
        "claims": [
            {
                "statement": "The evaluation committee must include an external member.",
                "evidence": [{"quote": "The evaluation committee must include an external member."}],
            },
            {
                "statement": ("The evaluation committee must not include an external member for procurements below the threshold."),
                "evidence": [
                    {
                        "quote": (
                            "The evaluation committee must not include an external member "
                            "for procurements below the threshold."
                        )
                    }
                ],
            },
        ],
        "reasoning_steps": [],
    }
    reasons = validate_record(record, passage)
    assert any(reason.startswith("cross_claim_contradiction:") for reason in reasons)


def test_validate_record_does_not_flag_unrelated_multi_claim_records() -> None:
    """Two claims that both use `must`/`must not` but describe genuinely
    different subjects must not be rejected -- the required "same core
    subject after stripping deontic markers" gate must hold end-to-end
    through `validate_record`, not just in the unit-level helper."""
    passage = (
        "The evaluation committee must include a technical specialist. "
        "Bids received after the deadline must not be considered."
    )
    record = {
        "task_type": "qa",
        "question": "What are the committee-composition and late-bid rules?",
        "answer": ("The evaluation committee must include a technical specialist, and bids received after the deadline must not be considered."),
        "answerable": True,
        "evidence": [
            {"quote": "The evaluation committee must include a technical specialist."},
            {"quote": "Bids received after the deadline must not be considered."},
        ],
        "claims": [
            {
                "statement": "The evaluation committee must include a technical specialist.",
                "evidence": [{"quote": "The evaluation committee must include a technical specialist."}],
            },
            {
                "statement": "Bids received after the deadline must not be considered.",
                "evidence": [{"quote": "Bids received after the deadline must not be considered."}],
            },
        ],
        "reasoning_steps": [],
    }
    reasons = validate_record(record, passage)
    assert not any(reason.startswith("cross_claim_contradiction:") for reason in reasons)


def test_drafting_validation_applies_modality_support_gate() -> None:
    source = "The buyer may recover liquidated damages."
    row = {
        "manual_passages": [source],
        "combined_source_text": f"Tender mode: Limited.\n{source}",
        "tender_context": ["Tender mode: Limited."],
        "instruction": "Draft the liquidated damages clause.",
        "require_block_attribution": True,
    }
    result = DraftingResult.model_validate(
        {
            "document_blocks": [
                {
                    "block_type": "heading",
                    "text": "Liquidated Damages",
                    "instruction_evidence_quotes": ["liquidated damages clause"],
                },
                {
                    "block_type": "paragraph",
                    "text": "The buyer shall deduct liquidated damages.",
                    "manual_evidence_quotes": [source],
                },
                {
                    "block_type": "field",
                    "text": "Tender mode: Limited.",
                    "tender_facts_used": ["Tender mode: Limited."],
                },
            ],
            "manual_evidence_quotes": [source],
            "tender_facts_used": ["Tender mode: Limited."],
        }
    )

    issues = drafting_validation_issues(row, result)
    assert "strengthened_modality:permission_to_obligation" in issues
    assert "block_1:strengthened_modality:permission_to_obligation" in issues


def test_judge_witness_accepts_only_lossless_grounded_forms() -> None:
    evidence = [
        "Hospitality must never be solicited, directly or indirectly.",
        "Gifts must never be solicited, directly or indirectly.",
        "Cash gift cheques may not be accepted regardless of the amount.",
    ]
    source = f"{evidence[0]}\nIntervening policy text.\n{evidence[1]} " f"{evidence[2]}"
    combined = f"{evidence[1]} {evidence[2]}"
    assert judge_quotes_are_grounded([combined], source, evidence)
    assert judge_quotes_are_grounded(
        ["“Hospitality must never be solicited, directly or indirectly.”"],
        source,
        evidence,
    )
    assert not judge_quotes_are_grounded(
        [f"{evidence[2]} {evidence[1]}"],
        source,
        evidence,
    )
    assert not judge_quotes_are_grounded(
        ["Cash gift cheques may be accepted regardless of the amount."],
        source,
        evidence,
    )


def test_empty_judge_quotes_recover_only_from_supported_exact_evidence() -> None:
    source = "Before. Exact supported sentence. After."
    quotes, recovered = recover_grounded_judge_quotes(
        [],
        answer_found_in_source=True,
        supported=True,
        source_text=source,
        evidence_quotes=["Exact supported sentence."],
    )
    assert quotes == ["Exact supported sentence."]
    assert recovered is True
    assert recover_grounded_judge_quotes(
        [],
        answer_found_in_source=True,
        supported=False,
        source_text=source,
        evidence_quotes=["Exact supported sentence."],
    ) == ([], False)


def test_cross_validation_accepts_typed_quantities_and_metadata_dates() -> None:
    documents = [
        {
            "source_id": "source_a",
            "manual_id": "works_2019",
            "title": "Manual for Procurement of Works, 2019",
            "revision_date": "2019",
            "as_of_date": "2019",
            "page": 1,
            "section": "Liquidated damages",
            "passage": ("Damages shall not exceed 10 (ten) per cent of the Contract Price."),
        },
        {
            "source_id": "source_b",
            "manual_id": "works_2025",
            "title": "Manual for Procurement of Works, Second Edition, 2025",
            "revision_date": "2025",
            "as_of_date": "2025",
            "page": 2,
            "section": "Liquidated damages",
            "passage": ("Damages shall not exceed 10 (ten) per cent of the Contract Price. " "Milestone LD shall be refunded without interest."),
        },
    ]
    record = {
        "task_type": "cross_document_qa",
        "question": "What changed between the 2019 and 2025 manuals?",
        "answer": ("The 2019 and 2025 manuals cap damages at 10 percent, while the " "2025 manual additionally refunds milestone LD without interest."),
        "answerable": True,
        "claims": [
            {
                "statement": "The 2019 and 2025 editions both cap damages.",
                "evidence": [
                    {
                        "source_id": "source_a",
                        "quote": documents[0]["passage"],
                    },
                    {
                        "source_id": "source_b",
                        "quote": documents[1]["passage"],
                    },
                ],
            }
        ],
        "reasoning_steps": [],
    }

    assert validate_cross_record(record, documents) == []


def test_cross_validation_rejects_number_misattributed_to_the_wrong_claim() -> None:
    """The same right-value/wrong-entity gap as T7, in the cross-document validator."""
    documents = [
        {
            "source_id": "source_a",
            "manual_id": "goods_2019",
            "title": "Manual for Procurement of Goods, 2019",
            "revision_date": "2019",
            "as_of_date": "2019",
            "page": 1,
            "section": "Response",
            "passage": "Entity A shall respond within 3 days.",
        },
        {
            "source_id": "source_b",
            "manual_id": "goods_2025",
            "title": "Manual for Procurement of Goods, 2025",
            "revision_date": "2025",
            "as_of_date": "2025",
            "page": 2,
            "section": "Decision",
            "passage": "Entity B shall decide within 5 days.",
        },
    ]
    record = {
        "task_type": "cross_document_qa",
        "question": "How long do Entity A and Entity B have?",
        "answer": "Entity A requires 5 days to respond, and Entity B requires 5 days to decide.",
        "answerable": True,
        "claims": [
            {
                "statement": "Entity A requires 3 days to respond.",
                "evidence": [{"source_id": "source_a", "quote": documents[0]["passage"]}],
            },
            {
                "statement": "Entity B requires 5 days to decide.",
                "evidence": [{"source_id": "source_b", "quote": documents[1]["passage"]}],
            },
        ],
        "reasoning_steps": [],
    }
    assert "unsupported_number:5 days" in validate_cross_record(record, documents)


def test_quantity_validation_does_not_swallow_following_prose() -> None:
    record = {
        "task_type": "qa",
        "question": "Which edition applies?",
        "answer": "The 2025 Manual applies.",
        "answerable": True,
        "evidence": [{"quote": "The 2025 edition applies."}],
        "claims": [
            {
                "statement": "The 2025 Manual applies.",
                "evidence": [{"quote": "The 2025 edition applies."}],
            }
        ],
        "reasoning_steps": [],
    }

    assert validate_record(record, "The 2025 edition applies.") == []


def test_unsupported_standard_procedure_absence_claim_is_rejected() -> None:
    support = "A GTE should be issued only after no Indian manufacturer is found."
    record = {
        "task_type": "qa",
        "question": "How does GTE differ from standard procurement?",
        "answer": "Standard procurement does not mandate this precondition.",
        "answerable": True,
        "evidence": [{"quote": support}],
        "claims": [
            {
                "statement": "Standard procurement does not mandate this precondition.",
                "evidence": [{"quote": support}],
            }
        ],
        "reasoning_steps": [],
    }
    issues = validate_record(record, support)
    assert "unsupported_absence_claim" in issues
    assert "claim_unsupported_absence_claim" in issues


def test_validation_rejects_truncated_answers_but_allows_concise_facts() -> None:
    support = "Bid security is five percent of the estimated value."
    base = {
        "task_type": "qa",
        "question": "What is the bid-security threshold?",
        "answer": "Five percent.",
        "answerable": True,
        "claims": [
            {
                "statement": "The threshold is five percent.",
                "evidence": [{"quote": support}],
            }
        ],
        "evidence": [{"quote": support}],
        "reasoning_steps": [],
    }
    assert validate_record(base, support) == []

    truncated = {**base, "answer": "The bid-security threshold is"}
    assert "incomplete_answer_dangling_word" in validate_record(truncated, support)

    punctuation_fragment = {**base, "answer": "The following threshold:"}
    assert "incomplete_answer_terminal_fragment" in validate_record(
        punctuation_fragment,
        support,
    )


def test_validation_requires_atomic_claim_evidence_without_cross_claim_leakage() -> None:
    permissive = "NRL may cancel the order."
    mandatory = "The supplier shall replace rejected goods."
    record = {
        "task_type": "qa",
        "question": "What remedies apply?",
        "answer": ("NRL may cancel the order, and the supplier shall replace rejected " "goods."),
        "answerable": True,
        "claims": [
            {
                "statement": "NRL shall cancel the order.",
                "evidence": [{"quote": permissive}],
            },
            {
                "statement": "The supplier shall replace rejected goods.",
                "evidence": [{"quote": mandatory}],
            },
        ],
        "evidence": [{"quote": permissive}, {"quote": mandatory}],
        "reasoning_steps": [],
    }
    issues = validate_record(record, f"{permissive}\n{mandatory}")
    assert "claim_strengthened_modality:permission_to_obligation" in issues

    record["claims"][0]["statement"] = "NRL may cancel the order."
    record["evidence"].append({"quote": "An unused supporting quotation."})
    issues = validate_record(
        record,
        f"{permissive}\n{mandatory}\nAn unused supporting quotation.",
    )
    assert "claim_evidence_mismatch" in issues


def test_cross_validation_checks_each_claim_against_its_own_evidence() -> None:
    documents = [
        {"source_id": "source_a", "passage": "The buyer may cancel the bid."},
        {
            "source_id": "source_b",
            "passage": "The supplier shall replace rejected goods.",
        },
    ]
    record = {
        "task_type": "cross_document_qa",
        "question": "What remedies do the two sources provide?",
        "answer": ("The buyer may cancel the bid, and the supplier shall replace " "rejected goods."),
        "answerable": True,
        "claims": [
            {
                "statement": "The buyer shall cancel the bid.",
                "evidence": [
                    {
                        "source_id": "source_a",
                        "quote": "The buyer may cancel the bid.",
                    }
                ],
            },
            {
                "statement": "The supplier shall replace rejected goods.",
                "evidence": [
                    {
                        "source_id": "source_b",
                        "quote": "The supplier shall replace rejected goods.",
                    }
                ],
            },
        ],
        "reasoning_steps": [],
    }
    assert "claim_strengthened_modality:permission_to_obligation" in validate_cross_record(record, documents)


def test_validation_rejects_dangling_evidence_without_requiring_punctuation() -> None:
    truncated = "Before releasing the PBG, ensure that there is nothing outstanding from the"
    record = {
        "task_type": "qa",
        "question": "What must be checked before releasing the PBG?",
        "answer": "Ensure that nothing is outstanding from the contractor.",
        "answerable": True,
        "evidence": [{"quote": truncated}],
        "reasoning_steps": [],
    }
    assert "incomplete_evidence_fragment" in validate_record(record, truncated)

    heading_evidence = "Defects Liability Certificate"
    record["answer"] = "A Defects Liability Certificate is required."
    record["evidence"] = [{"quote": heading_evidence}]
    assert "incomplete_evidence_fragment" not in validate_record(record, heading_evidence)


def test_dedup_and_amendment_connected_split() -> None:
    records = [
        {"record_id": "a", "manual_id": "base", "question": "What is the threshold?"},
        {"record_id": "b", "manual_id": "amendment", "question": "Who approves this?"},
        {"record_id": "c", "manual_id": "other", "question": "What is the threshold?"},
    ]
    unique, removed = deduplicate(records)
    assert removed == 1
    siblings = [
        {"record_id": "s1", "blueprint_id": "bp", "question": "What is the threshold?"},
        {"record_id": "s2", "blueprint_id": "bp", "question": "What is the threshold?"},
    ]
    preserved, sibling_removed = deduplicate(siblings, preserve_within_group="blueprint_id")
    assert len(preserved) == 2
    assert sibling_removed == 0
    manuals = [
        {"manual_id": "base"},
        {"manual_id": "amendment", "amends": ["base"]},
        {"manual_id": "other"},
    ]
    assign_splits(records[:2], manuals, 0.8, 0.1, "test")
    assert records[0]["split"] == records[1]["split"]


def test_enforce_question_opener_diversity_caps_dominant_template() -> None:
    records = [{"record_id": f"template-{i}", "question": f"According to the manual, what is fact {i}?"} for i in range(17)]
    records += [{"record_id": f"varied-{i}", "question": f"Under Section {i}, who approves this step?"} for i in range(3)]
    kept, removed = enforce_question_opener_diversity(records, max_share=0.15)
    # The quota applies to the resulting pool, not the original 20 rows. With
    # only three alternatives, one repeated-template row is the irreducible
    # deterministic minimum (1/4 > 15% is unavoidable in such a tiny pool).
    assert removed == 16
    kept_templates = sum(1 for r in kept if r["record_id"].startswith("template-"))
    assert kept_templates == 1
    assert sum(1 for r in kept if r["record_id"].startswith("varied-")) == 3
    assert len(kept) + removed == len(records)


def test_enforce_question_opener_diversity_leaves_diverse_pool_untouched() -> None:
    records = [{"record_id": f"r{i}", "question": f"Question number {i} is phrased differently each time?"} for i in range(10)]
    kept, removed = enforce_question_opener_diversity(records, max_share=0.15)
    assert removed == 0
    assert kept == records


def test_question_opener_diversity_reports_top_share() -> None:
    records = [{"question": "According to the manual, what applies?"} for _ in range(3)]
    records.append({"question": "Who approves this exception?"})
    report = question_opener_diversity(records)
    assert report["unique_openers"] == 2
    assert report["top_opener"] == "according to the manual"
    assert report["top_opener_count"] == 3
    assert report["top_opener_share"] == 0.75
    assert question_opener_diversity([]) == {
        "unique_openers": 0,
        "top_opener": "",
        "top_opener_count": 0,
        "top_opener_share": 0.0,
    }


def test_batch_efficiency_stats_reports_ratio_and_removal_rate() -> None:
    report = batch_efficiency_stats(
        200,
        150,
        {"near_duplicates": 30, "question_opener_overrepresented": 20},
    )
    assert report["generated_records"] == 200
    assert report["accepted_records"] == 150
    assert report["generation_to_acceptance_ratio"] == 0.75
    assert report["total_removed"] == 50
    assert report["removal_rate"] == 0.25
    assert report["removed_by_reason"] == {
        "near_duplicates": 30,
        "question_opener_overrepresented": 20,
    }


def test_batch_efficiency_stats_handles_zero_generated_records() -> None:
    report = batch_efficiency_stats(0, 0, {})
    assert report["generation_to_acceptance_ratio"] == 0.0
    assert report["removal_rate"] == 0.0


def _manifest_kwargs(**overrides) -> dict:
    base = {
        "run_id": "run-1",
        "status": "complete",
        "stats": {"records": 0},
        "manuals": [],
        "corpus_report": {},
        "selected_rows": [],
        "single_coverage": {},
        "cross_coverage": {},
        "drafting_stats": {},
        "duplicates": 0,
    }
    base.update(overrides)
    return base


def test_manifest_reports_pre_cap_opener_concentration_and_waste_ratio() -> None:
    """T9: the post-cap `question_opener_diversity` stat is healthy by
    construction, so the manifest must separately show how concentrated the
    raw generated pool was before the cap ran, and how much it discarded.
    """
    pre_cap_report = question_opener_diversity(
        [{"question": "According to the manual, what applies?"} for _ in range(9)] + [{"question": "Who approves this exception?"}]
    )
    manifest = generation_pipeline._final_manifest(
        **_manifest_kwargs(
            opener_overrepresented=6,
            single_generated_pre_cap_count=10,
            question_opener_diversity_pre_cap=pre_cap_report,
        )
    )
    reported = manifest["question_opener_diversity_pre_cap"]
    assert reported["top_opener"] == "according to the manual"
    assert reported["top_opener_share"] == 0.9
    assert reported["pool_size"] == 10
    assert reported["cap_waste_ratio"] == 0.6


def test_manifest_pre_cap_opener_field_defaults_safely_when_unset() -> None:
    manifest = generation_pipeline._final_manifest(**_manifest_kwargs())
    reported = manifest["question_opener_diversity_pre_cap"]
    assert reported["pool_size"] == 0
    assert reported["cap_waste_ratio"] == 0.0
    assert reported["unique_openers"] == 0


def test_extractive_answer_diversity_caps_final_pool_share() -> None:
    copied = [
        {
            "record_id": f"copied-{index}",
            "task_type": "qa",
            "answer": f"The complete copied answer number {index} applies",
            "evidence": [{"quote": f"The complete copied answer number {index} applies in this case."}],
        }
        for index in range(8)
    ]
    synthesized = [
        {
            "record_id": f"synthesized-{index}",
            "task_type": "qa",
            "answer": f"Use the applicable combined procedure {index}.",
            "evidence": [{"quote": "First verify eligibility, and then obtain approval."}],
        }
        for index in range(4)
    ]
    kept, removed = enforce_extractive_answer_diversity(
        [*copied, *synthesized],
        max_share=0.35,
    )
    report = answer_style_diversity(kept)
    assert removed == 6
    assert report["extractive_answers"] == 2
    assert report["extractive_answer_share"] <= 0.35


def test_question_answer_relevance_diagnostics_flags_off_topic_answer() -> None:
    """T12: an answer with zero lexical/topical connection to its question
    must be reported in `flagged_sample`, and reduce the aggregate stats."""
    records = [
        {
            "record_id": "on-topic",
            "task_type": "qa",
            "answerable": True,
            "question": "What is the tender validity period for goods procurement?",
            "answer": "The tender validity period for goods procurement is 90 days.",
            "evidence": [{"quote": "The tender validity period for goods procurement is 90 days."}],
        },
        {
            "record_id": "off-topic",
            "task_type": "qa",
            "answerable": True,
            "question": "What is the tender validity period for goods procurement?",
            "answer": "Consortium members must jointly and severally execute the contract agreement.",
            "evidence": [{"quote": "Consortium members must jointly and severally execute the contract agreement."}],
        },
    ]
    report = question_answer_relevance_diagnostics(records, near_zero_overlap_ratio=0.05)
    assert report["records_evaluated"] == 2
    assert report["flagged_near_zero_overlap"] == 1
    assert [item["record_id"] for item in report["flagged_sample"]] == ["off-topic"]


def test_question_answer_relevance_diagnostics_ignores_unanswerable_and_empty_pool() -> None:
    assert question_answer_relevance_diagnostics([]) == {
        "records_evaluated": 0,
        "near_zero_overlap_ratio_threshold": 0.05,
        "flagged_near_zero_overlap": 0,
        "flagged_share": 0.0,
        "mean_overlap_ratio": 0.0,
        "median_overlap_ratio": 0.0,
        "flagged_sample": [],
    }
    unanswerable = [
        {
            "record_id": "unanswerable-1",
            "task_type": "qa",
            "answerable": False,
            "question": "What is the tender validity period for goods procurement?",
            "answer": "Not answerable from the provided sources.",
            "evidence": [],
        }
    ]
    report = question_answer_relevance_diagnostics(unanswerable)
    assert report["records_evaluated"] == 0


def test_short_precise_answers_are_not_classified_as_span_copying() -> None:
    assert not is_extractive_answer(
        {
            "answer": "Rs 10 crore",
            "evidence": [{"quote": "The threshold is Rs 10 crore."}],
        }
    )


def test_connected_split_targets_records_without_leaking_components() -> None:
    sizes = [26, 16, 6, 3, 3, 3, 2, 1]
    manuals = [{"manual_id": f"manual-{index}"} for index in range(len(sizes))]
    records = [
        {
            "record_id": f"record-{component}-{index}",
            "manual_id": f"manual-{component}",
        }
        for component, size in enumerate(sizes)
        for index in range(size)
    ]

    assign_splits(records, manuals, 0.8, 0.1, "pilot-011")

    split_by_manual = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        split_by_manual.setdefault(record["manual_id"], record["split"])
        assert split_by_manual[record["manual_id"]] == record["split"]
        counts[record["split"]] += 1
    assert all(counts[split] > 0 for split in counts)
    assert counts["train"] >= counts["validation"]
    assert counts["train"] >= counts["test"]


def test_assign_drafting_splits_matches_the_manual_fold_qa_records_use() -> None:
    manuals = [{"manual_id": "m-train"}, {"manual_id": "m-test"}]
    manual_folds = {"m-train": "train", "m-test": "test"}
    chunk_manuals = {
        "chunk-train-1": {"manual_id": "m-train", "source_sha256": "sha-train", "section": "s1"},
        "chunk-test-1": {"manual_id": "m-test", "source_sha256": "sha-test", "section": "s2"},
    }
    drafting_records = [
        {"id": "draft-train", "instruction": "Draft A", "manual_chunk_ids": ["chunk-train-1"]},
        {"id": "draft-test", "instruction": "Draft B", "manual_chunk_ids": ["chunk-test-1"]},
    ]
    assign_drafting_splits(
        drafting_records,
        manuals,
        chunk_manuals,
        0.8,
        0.1,
        "seed",
        manual_folds=manual_folds,
    )
    by_id = {row["id"]: row["split"] for row in drafting_records}
    assert by_id["draft-train"] == "train"
    assert by_id["draft-test"] == "test"

    # A QA record citing the same eval-fold manual must land in the identical split
    # a drafting record referencing it does — the T13a invariant.
    qa_record = {"record_id": "qa-1", "manual_id": "m-test"}
    assign_splits([qa_record], manuals, 0.8, 0.1, "seed", manual_folds=manual_folds)
    assert qa_record["split"] == by_id["draft-test"]


def test_drafting_records_are_now_covered_by_leakage_audit_against_qa_records() -> None:
    # Regression for the T13/T13b bypass: before this fix, drafting_accepted was
    # written straight to drafting.jsonl and never entered leakage_audit at all, so
    # a drafting record built from an eval-fold chunk that collides with a train-fold
    # QA record's source would have gone undetected. It must now be caught.
    manuals = [{"manual_id": "m-a"}, {"manual_id": "m-b"}]
    manual_folds = {"m-a": "train", "m-b": "test"}
    qa_record = {
        "record_id": "qa-1",
        "split": "train",
        "question": "What is the threshold?",
        "manual_id": "m-a",
        "source_sha256": "shared-sha",
        "source_chunk_ids": ["shared-chunk"],
        "citations": [{"section": "s"}],
    }
    # A corpus/config inconsistency (the exact class of bug this gate exists to
    # catch) puts the same source hash under a chunk resolved to the test-fold
    # manual for the drafting seed.
    chunk_manuals = {
        "shared-chunk": {"manual_id": "m-b", "source_sha256": "shared-sha", "section": "s"},
    }
    drafting_records = [
        {"id": "draft-leak", "instruction": "Draft something", "manual_chunk_ids": ["shared-chunk"]},
    ]
    assign_drafting_splits(
        drafting_records,
        manuals,
        chunk_manuals,
        0.8,
        0.1,
        "seed",
        manual_folds=manual_folds,
    )
    assert drafting_records[0]["split"] == "test"
    audit = leakage_audit([qa_record, *drafting_records])
    assert not audit["passed"]
    assert audit["collisions"]["source_hash"] == [
        {"value": "shared-sha", "splits": ["test", "train"]}
    ]


def _cross_row(manual_id: str, chunk_id: str, passage: str) -> dict:
    return {
        "manual_id": manual_id,
        "title": manual_id,
        "issuing_organization": "NRL",
        "policy_scope": "test",
        "revision_date": "2026",
        "as_of_date": "2026",
        "source_file": f"{manual_id}.md",
        "source_sha256": f"sha-{manual_id}",
        "chunk_id": chunk_id,
        "page": 1,
        "section": "Bid security requirements",
        "passage": passage,
    }


def test_cross_document_bundle_and_source_attribution() -> None:
    left_quote = "The bidder shall submit bid security with every tender."
    right_quote = "Bid security shall be submitted through the procurement portal."
    rows = [
        _cross_row("left_manual", "left-1", left_quote),
        _cross_row("right_manual", "right-1", right_quote),
    ]
    config = {
        "minimum_similarity": 1,
        "minimum_shared_terms": 2,
        "max_bundles_per_pair": 2,
        "pairs": [
            {
                "pair_id": "pair",
                "left_manual": "left_manual",
                "right_manual": "right_manual",
                "relationship_type": "complementary_procedure",
            }
        ],
    }
    bundle = build_bundles(rows, config)[0]
    record = {
        "task_type": "cross_document_qa_cot",
        "question": "How do both manuals collectively require bid-security submission?",
        "answer": "Bid security accompanies the tender and is submitted through the portal.",
        "answerable": True,
        "claims": [
            {
                "statement": "It accompanies the tender.",
                "evidence": [{"source_id": "source_a", "quote": left_quote}],
            },
            {
                "statement": "It is submitted through the portal.",
                "evidence": [{"source_id": "source_b", "quote": right_quote}],
            },
        ],
        "reasoning_steps": [
            {
                "operation": "lookup",
                "statement": "Identify the tender requirement.",
                "evidence": [{"source_id": "source_a", "quote": left_quote}],
            },
            {
                "operation": "combine",
                "statement": "Combine it with the submission method.",
                "evidence": [
                    {"source_id": "source_a", "quote": left_quote},
                    {"source_id": "source_b", "quote": right_quote},
                ],
            },
        ],
    }
    assert validate_cross_record(record, bundle["source_documents"]) == []
    record["claims"][0]["evidence"][0]["source_id"] = "source_b"
    assert "misattributed_or_non_verbatim_evidence" in validate_cross_record(record, bundle["source_documents"])


def test_deterministic_rejections_are_materialized_for_audit() -> None:
    passage = "The buyer shall retain the record for 5 years."
    single_row = {
        **_cross_row("manual", "chunk-1", passage),
        "planned_request_id": "single-request",
        "planned_task_type": "qa",
        "planned_question_type": "threshold",
        "planned_answer_format": "concise_direct",
        "planned_answerable": True,
        "blueprint_id": "qabp-test",
        "task": "compliance_and_audit",
        "persona": "auditor",
        "instruction_goal": "Determine the required retention period.",
        "must_cover": ["The record is retained for the stated period."],
        "blueprint_evidence": [{"quote": passage}],
    }
    single_response = GroundedCandidateDraft.model_validate(
        {
            "question": "How long must the record be retained?",
            "answer": "The record must be retained for 10 years.",
            "claims": [
                {
                    "statement": "The record must be retained for 10 years.",
                    "evidence": [{"quote": passage}],
                }
            ],
            "reasoning_steps": [],
        }
    )
    single = ProcurementGenerator.parse(
        SimpleNamespace(model_name="generator"),
        single_row,
        single_response,
    )
    assert len(single) == 1
    assert single[0]["deterministic_checks"]["passed"] is False
    assert "unsupported_number:10 years" in single[0]["deterministic_checks"]["issues"]

    cross_row = {
        "source_bundle_id": "bundle",
        "pair_id": "pair",
        "relationship_type": "government_company_comparison",
        "source_documents": [
            {"source_id": "source_a", **_cross_row("a", "a-1", passage)},
            {"source_id": "source_b", **_cross_row("b", "b-1", passage)},
        ],
        "planned_request_id": "cross-request",
        "planned_task_type": "cross_document_qa",
        "planned_answerable": True,
    }
    cross_response = CrossCandidateBatch.model_validate(
        {
            "examples": [
                {
                    "task_type": "cross_document_qa_cot",
                    "task": "compliance_and_audit",
                    "persona": "auditor",
                    "question_type": "comparison",
                    "question": "How do the two manuals state the retention rule?",
                    "answer": "Both require retention for 5 years.",
                    "answerable": True,
                    "claims": [
                        {
                            "statement": "Both require retention.",
                            "evidence": [
                                {"source_id": "source_a", "quote": passage},
                                {"source_id": "source_b", "quote": passage},
                            ],
                        }
                    ],
                    "reasoning_steps": [
                        {
                            "operation": "lookup",
                            "statement": "Read source A.",
                            "evidence": [{"source_id": "source_a", "quote": passage}],
                        },
                        {
                            "operation": "combine",
                            "statement": "Combine with source B.",
                            "evidence": [
                                {"source_id": "source_a", "quote": passage},
                                {"source_id": "source_b", "quote": passage},
                            ],
                        },
                    ],
                }
            ]
        }
    )
    cross = CrossDocumentGenerator.parse(
        SimpleNamespace(model_name="generator"),
        cross_row,
        cross_response,
    )
    assert len(cross) == 1
    assert cross[0]["deterministic_checks"] == {"passed": True, "issues": []}
    assert cross[0]["task_type"] == "cross_document_qa"
    assert cross[0]["reasoning_steps"] == []
    assert cross[0]["structural_repairs"] == [
        "injected_planned_task_type:cross_document_qa_cot->cross_document_qa",
        "removed_reasoning_steps_for_cross_document_qa",
    ]


def test_qa_evidence_offsets_resolve_against_original_source_chunk() -> None:
    quote = "The buyer shall retain the record for 5 years."
    row = {
        **_cross_row("manual", "chunk-1", quote),
        "source_passage": f"![page image](image.png)\n\n{quote}",
        "planned_request_id": "single-request",
        "planned_task_type": "qa",
        "planned_question_type": "threshold",
        "planned_answer_format": "concise_direct",
        "planned_answerable": True,
        "blueprint_id": "qabp-test",
        "task": "compliance_and_audit",
        "persona": "auditor",
        "instruction_goal": "Determine the required retention period.",
        "must_cover": ["The record is retained for five years."],
        "blueprint_evidence": [{"quote": quote}],
    }
    response = GroundedCandidateDraft.model_validate(
        {
            "question": "How long must the record be retained?",
            "answer": "The record must be retained for 5 years.",
            "claims": [
                {
                    "statement": "The record must be retained for 5 years.",
                    "evidence": [{"quote": quote}],
                }
            ],
            "reasoning_steps": [],
        }
    )
    result = ProcurementGenerator.parse(
        SimpleNamespace(model_name="generator"),
        row,
        response,
    )
    assert len(result) == 1
    record = result[0]
    assert record["deterministic_checks"]["passed"] is True
    evidence = record["evidence"][0]
    assert evidence["start_char"] == len("![page image](image.png)\n\n")
    assert row["source_passage"][evidence["start_char"] : evidence["end_char"]] == quote
    assert record["citations"][0]["start_char"] == evidence["start_char"]


def test_cot_must_execute_its_planned_reasoning_operation() -> None:
    quote = "The bidder may proceed only if the authority approves the exception."
    row = {
        **_cross_row("manual", "chunk-1", quote),
        "planned_request_id": "single-request",
        "planned_task_type": "qa_cot",
        "planned_question_type": "exception",
        "planned_question_style": "exception_check",
        "planned_answer_format": "rule_and_exception",
        "planned_answerable": True,
        "planned_reasoning_operation": "apply_condition",
        "planned_difficulty": "intermediate",
        "planned_material_focus": "exception",
        "planned_coverage_cell_id": "qacell-test",
        "blueprint_id": "qabp-test",
        "candidate_request_id": "qabp-test-candidate-01",
        "task": "compliance_and_audit",
        "persona": "auditor",
        "persona_need": "Decide whether the stated exception permits proceeding.",
        "instruction_goal": "Apply the approval condition to the exception.",
        "must_cover": ["Approval is required."],
        "blueprint_evidence": [{"quote": quote}],
    }
    response = GroundedCandidateDraft.model_validate(
        {
            "question": "When may the bidder proceed under the exception?",
            "answer": "The bidder may proceed only after authority approval.",
            "claims": [
                {
                    "statement": "Authority approval is required to proceed.",
                    "evidence": [{"quote": quote}],
                }
            ],
            "reasoning_steps": [
                {
                    "operation": "lookup",
                    "statement": "Identify the exception rule.",
                    "evidence_quotes": [quote],
                },
                {
                    "operation": "conclude",
                    "statement": "State that approval is required.",
                    "evidence_quotes": [quote],
                },
            ],
        }
    )
    result = ProcurementGenerator.parse(SimpleNamespace(model_name="generator"), row, response)[0]
    assert "planned_reasoning_operation_missing" in result["deterministic_checks"]["issues"]


def test_qa_evidence_rejects_unresolvable_citation_offset() -> None:
    quote = "The buyer shall retain the record for 5 years."
    row = {
        **_cross_row("manual", "chunk-1", quote),
        "source_passage": "This chunk text no longer contains the quoted sentence at all.",
        "planned_request_id": "single-request",
        "planned_task_type": "qa",
        "planned_question_type": "threshold",
        "planned_answer_format": "concise_direct",
        "planned_answerable": True,
        "blueprint_id": "qabp-test",
        "task": "compliance_and_audit",
        "persona": "auditor",
        "instruction_goal": "Determine the required retention period.",
        "must_cover": ["The record is retained for five years."],
        "blueprint_evidence": [{"quote": quote}],
    }
    response = GroundedCandidateDraft.model_validate(
        {
            "question": "How long must the record be retained?",
            "answer": "The record must be retained for 5 years.",
            "claims": [
                {
                    "statement": "The record must be retained for 5 years.",
                    "evidence": [{"quote": quote}],
                }
            ],
            "reasoning_steps": [],
        }
    )
    result = ProcurementGenerator.parse(
        SimpleNamespace(model_name="generator"),
        row,
        response,
    )
    assert len(result) == 1
    record = result[0]
    assert record["deterministic_checks"]["passed"] is False
    assert "citation_offset_unresolvable" in record["deterministic_checks"]["issues"]
    assert record["evidence"][0]["start_char"] == -1
    assert record["evidence"][0]["end_char"] == -1


def test_drafting_seed_resolution_validation_and_compact_output(tmp_path: Path) -> None:
    seed_path = tmp_path / "drafting.jsonl"
    seed_path.write_text(
        '{"id":"draft-1","tender_id":"tender-1","task":"drafting",'
        '"instruction":"Draft the delayed-delivery clause.",'
        '"tender_context":["Tender mode: Limited."],'
        '"manual_chunk_ids":["chunk-1"]}\n',
        encoding="utf-8",
    )
    corpus = [
        {
            "chunk_id": "chunk-1",
            "page": 4,
            "section": "Liquidated damages",
            "passage": "LD is 0.5% per week and capped at 5% of delayed goods.",
        }
    ]
    inputs = build_drafting_inputs(read_drafting_seeds(seed_path), corpus)
    result = DraftingResult(
        document_blocks=[
            DraftingBlock(
                text="Delayed Delivery & Liquidated Damages",
                instruction_evidence_quotes=["Draft the delayed-delivery clause."],
            ),
            DraftingBlock(
                text="LD is 0.5% per week and capped at 5% of delayed goods.",
                manual_evidence_quotes=["LD is 0.5% per week and capped at 5% of delayed goods."],
            ),
            DraftingBlock(
                text="Tender mode: Limited.",
                tender_facts_used=["Tender mode: Limited."],
            ),
        ],
        manual_evidence_quotes=["LD is 0.5% per week and capped at 5% of delayed goods."],
        tender_facts_used=["Tender mode: Limited."],
        field_claims=[
            DraftingFieldClaim(
                block_index=1,
                field_name="liquidated damages rule",
                value="LD is 0.5% per week and capped at 5% of delayed goods.",
                manual_evidence_quotes=["LD is 0.5% per week and capped at 5% of delayed goods."],
            ),
            DraftingFieldClaim(
                block_index=2,
                field_name="tender mode",
                value="Limited",
                tender_facts_used=["Tender mode: Limited."],
            ),
        ],
    )
    assert drafting_validation_issues(inputs[0], result) == []
    compact = compact_drafting(
        {
            **inputs[0],
            "citations": ["chunk-1", "tender-1"],
            "context": [*inputs[0]["tender_context"], *result.manual_evidence_quotes],
            "response": "\n\n".join(block.text for block in result.document_blocks),
            "field_claims": [claim.model_dump() for claim in result.field_claims],
            "citation_details": [
                {
                    "citation_id": "chunk-1",
                    "source_type": "manual",
                    "page": 4,
                    "section": "Liquidated damages",
                    "chunk_id": "chunk-1",
                    "quote": "LD is 0.5% per week and capped at 5% of delayed goods.",
                },
                {
                    "citation_id": "tender-1:fact:typed",
                    "source_type": "tender_seed",
                    "tender_id": "tender-1",
                    "fact_index": 0,
                    "fact": "Tender mode: Limited.",
                },
            ],
        }
    )
    assert list(compact) == [
        "id",
        "tender_id",
        "task",
        "instruction",
        "context",
        "response",
        "field_claims",
        "citation_details",
        "citations",
    ]
    assert compact["citations"] == ["chunk-1", "tender-1"]
    assert inputs[0]["candidate_citation_ids"][0] == "chunk-1"
    assert inputs[0]["candidate_citation_ids"][1].startswith("tender-1:fact:")


def test_drafting_field_claims_and_tender_citations_are_atomic(tmp_path: Path) -> None:
    seed_path = tmp_path / "drafting.jsonl"
    seed_path.write_text(
        '{"id":"draft-1","tender_id":"tender-1","task":"drafting",'
        '"instruction":"Draft the delayed-delivery clause.",'
        '"tender_context":["Tender mode: Limited."],'
        '"manual_chunk_ids":["chunk-1"]}\n',
        encoding="utf-8",
    )
    inputs = build_drafting_inputs(
        read_drafting_seeds(seed_path),
        [
            {
                "chunk_id": "chunk-1",
                "manual_id": "manual-1",
                "title": "Manual",
                "source_file": "manual.md",
                "page": 4,
                "section": "Damages",
                "passage": "The cap is 5%.",
            }
        ],
    )
    result = DraftingResult(
        document_blocks=[
            DraftingBlock(
                text="Delayed Delivery",
                instruction_evidence_quotes=["delayed-delivery"],
            ),
            DraftingBlock(
                text="The cap is 5%.",
                manual_evidence_quotes=["The cap is 5%."],
            ),
            DraftingBlock(
                text="Tender mode: Limited.",
                tender_facts_used=["Tender mode: Limited."],
            ),
        ],
        manual_evidence_quotes=["The cap is 5%."],
        tender_facts_used=["Tender mode: Limited."],
        field_claims=[
            DraftingFieldClaim(
                block_index=1,
                field_name="cap",
                value="5%",
                manual_evidence_quotes=["The cap is 5%."],
            ),
            DraftingFieldClaim(
                block_index=2,
                field_name="tender mode",
                value="Limited",
                tender_facts_used=["Tender mode: Limited."],
            ),
        ],
    )
    parsed = TenderDraftingGenerator.parse(
        SimpleNamespace(model_name="generator"),
        inputs[0],
        result,
    )[0]
    assert parsed["deterministic_checks"] == {"passed": True, "issues": []}
    tender_details = [detail for detail in parsed["citation_details"] if detail["source_type"] == "tender_seed"]
    assert tender_details[0]["fact"] == "Tender mode: Limited."
    assert tender_details[0]["citation_id"].startswith("tender-1:fact:")
    assert parsed["field_claims"][1]["block_index"] == 2

    invalid = result.model_copy(
        update={
            "field_claims": [
                DraftingFieldClaim(
                    block_index=1,
                    field_name="cap",
                    value="10%",
                    manual_evidence_quotes=["The cap is 5%."],
                )
            ]
        }
    )
    issues = drafting_validation_issues(inputs[0], invalid)
    assert "field_claim_value_not_in_block:0" in issues
    assert "material_block_without_field_claim:2" in issues
    assert "unclaimed_tender_fact" in issues


def test_drafting_support_reconciliation_is_exact_and_audited() -> None:
    manual = "## Damages\n\nThe total damages shall not exceed 5%."
    tender_fact = "Tender mode: Limited."
    instruction = "Draft the damages clause with the maximum cap."
    result = DraftingResult(
        document_blocks=[
            DraftingBlock(
                text=tender_fact,
                manual_evidence_quotes=[tender_fact],
                tender_facts_used=[tender_fact],
            ),
            DraftingBlock(
                text="The total damages shall not exceed 5%.",
                manual_evidence_quotes=["## Damages\nThe total damages shall not exceed 5%."],
                instruction_evidence_quotes=["with the ... cap"],
            ),
        ],
        manual_evidence_quotes=["The total damages shall not exceed 5%."],
        tender_facts_used=[tender_fact],
        field_claims=[
            DraftingFieldClaim(
                block_index=0,
                field_name="tender_mode",
                value="Limited",
                manual_evidence_quotes=[tender_fact],
                tender_facts_used=[tender_fact],
            ),
            DraftingFieldClaim(
                block_index=1,
                field_name="cap",
                value="5%",
                manual_evidence_quotes=["The total damages shall not exceed 5%."],
                instruction_evidence_quotes=["with the ... cap"],
            ),
        ],
    )
    reconciled, repairs = reconcile_drafting_support(
        {
            "manual_passages": [manual],
            "tender_context": [tender_fact],
            "instruction": instruction,
        },
        result,
    )
    assert reconciled.document_blocks[0].manual_evidence_quotes == []
    assert reconciled.document_blocks[0].tender_facts_used == [tender_fact]
    assert reconciled.document_blocks[1].manual_evidence_quotes == [
        manual,
        "The total damages shall not exceed 5%.",
    ]
    assert reconciled.document_blocks[1].instruction_evidence_quotes == []
    assert "dropped_invalid_manual_support:document_blocks[0]" in repairs
    assert "resolved_manual_support_whitespace:document_blocks[1]" in repairs
    assert "promoted_field_claim_support:document_blocks[1]" in repairs


def test_drafting_rejects_unknown_chunks_and_unsupported_values(tmp_path: Path) -> None:
    seed_path = tmp_path / "drafting.jsonl"
    seed_path.write_text(
        '{"id":"draft-1","tender_id":"tender-1","task":"drafting",'
        '"instruction":"Draft the delayed-delivery clause.",'
        '"tender_context":["Tender mode: Limited."],'
        '"manual_chunk_ids":["missing"]}\n',
        encoding="utf-8",
    )
    seeds = read_drafting_seeds(seed_path)
    try:
        build_drafting_inputs(seeds, [])
    except ValueError as exc:
        assert "unknown chunk" in str(exc)
    else:
        raise AssertionError("unknown drafting chunk should fail")

    row = {
        "manual_passages": ["The cap is 5%."],
        "combined_source_text": "The cap is 5%.",
        "tender_context": ["Tender mode: Limited."],
    }
    result = DraftingResult(
        document_blocks=[
            DraftingBlock(text="Delayed Delivery"),
            DraftingBlock(text="The cap is 10%. Contact invented@example.com."),
        ],
        manual_evidence_quotes=["The cap is 5%."],
        tender_facts_used=["Tender mode: Limited."],
    )
    issues = drafting_validation_issues(row, result)
    assert "unsupported_number:10%" in issues
    assert "unsupported_email:invented@example.com" in issues


def test_registered_drafting_seeds_cover_multiple_categories_and_document_types() -> None:
    """Regression test for the single-instance drafting seed pool.

    The registered seed file must resolve against the real corpus and span
    more than one procurement category (goods/works/services) and more than
    one tender instance, so `drafting.minimum_accepted_records: 1` cannot be
    satisfied entirely from one narrow scenario.
    """
    seed_path = REPO_ROOT / "data" / "seeds" / "drafting_requests.jsonl"
    seeds = read_drafting_seeds(seed_path)
    assert len(seeds) >= 5

    rows, _manuals = load_corpus(
        REPO_ROOT / "data" / "source", REPO_ROOT / "data" / "interim" / "ocr"
    )
    # build_drafting_inputs raises if any seed references an unknown/stale chunk.
    inputs = build_drafting_inputs(seeds, rows)
    chunks_by_id = {str(row["chunk_id"]): row for row in rows}

    categories: set[str] = set()
    for seed in seeds:
        manual_ids = {
            str(chunks_by_id[chunk_id]["manual_id"]) for chunk_id in seed.manual_chunk_ids
        }
        categories.update(_document_family(manual_id) for manual_id in manual_ids)
    assert categories >= {"goods", "works", "services"}, categories

    tender_ids = {seed.tender_id for seed in seeds}
    assert len(tender_ids) >= 3, "seeds should not all reference one tender instance"

    # Document-type diversity: distinct id prefixes stand in for distinct
    # drafted-document types (NIT header vs. clause vs. applicability note).
    id_prefixes = {seed.id.split("-", 2)[1] for seed in seeds}
    assert len(id_prefixes) >= 3, id_prefixes
    assert len(inputs) == len(seeds)


def test_drafting_citation_integrity_is_bidirectional_and_allows_repeated_details() -> None:
    details = [
        {
            "citation_id": "chunk-used",
            "source_type": "manual",
            "quote": "First supporting quotation.",
        },
        {
            "citation_id": "chunk-used",
            "source_type": "manual",
            "quote": "Second supporting quotation.",
        },
        {
            "citation_id": "tender-1:fact:typed",
            "source_type": "tender_seed",
            "tender_id": "tender-1",
            "fact_index": 0,
            "fact": "Tender mode: Limited.",
        },
    ]
    assert (
        drafting_citation_integrity_issues(
            ["chunk-used", "tender-1:fact:typed"],
            details,
            tender_id="tender-1",
            evidence_quote_count=2,
        )
        == []
    )
    issues = drafting_citation_integrity_issues(
        ["chunk-used", "chunk-unused"],
        details[:-1],
        tender_id="tender-1",
        evidence_quote_count=3,
    )
    assert "dangling_drafting_citations:chunk-unused" in issues
    assert "unresolved_drafting_evidence:expected=3,resolved=2" in issues
    assert "invalid_tender_seed_provenance:expected=1,resolved=0" in issues


def test_single_document_prompts_preserve_specification_contract() -> None:
    row = {
        "manual_id": "manual",
        "title": "Procurement Manual",
        "issuing_organization": "NRL",
        "policy_scope": "company_policy",
        "revision_date": "2026",
        "as_of_date": "2026-07-28",
        "page": 3,
        "section": "Bid security",
        "passage": "The bidder shall submit bid security.",
        "planned_task_type": "qa_cot",
        "planned_question_type": "compliance_check",
        "planned_answer_format": "audit_check",
        "planned_answerable": True,
        "task": "security_and_guarantees",
        "persona": "technical_evaluator",
        "instruction_goal": "Verify whether the bidder submitted bid security.",
        "must_cover": ["The bidder shall submit bid security."],
        "blueprint_evidence": [{"quote": "The bidder shall submit bid security."}],
    }
    prompt = ProcurementGenerator.prompt(None, row)
    for required in (
        "TASK",
        "PLANNED CONTRACT",
        "fixed procurement task is security_and_guarantees",
        "fixed persona is technical_evaluator",
        "SOURCE POLICY",
        "CONSTRAINTS",
        "OUTPUT CONTRACT",
        "FIXED GROUNDED BLUEPRINT",
        "FINAL CHECK",
        "qa_cot",
        "two to four",
        "private hidden chain-of-thought",
        "evidence_quotes",
        "Government",
        "guidance as NRL policy",
        "---BEGIN UNTRUSTED SOURCE PASSAGE---",
        "---END UNTRUSTED SOURCE PASSAGE---",
    ):
        assert required in prompt
    # T8: zero-shot instructions alone did not reliably keep generation off
    # the "According to.../As a <persona>..." openers (audit Repetition &
    # Diversity Analysis); varied few-shot opener examples must be present so
    # the boundary is established by demonstration, not just prohibition.
    assert "Vary the opening construction" in prompt
    for example in QUESTION_OPENER_EXAMPLES:
        assert example in prompt
        assert not SOURCE_FRAMING_PREFIX.search(example)

    blueprint_prompt = ProcurementBlueprintGenerator.prompt(None, row)
    assert "Do not write the final question or answer" in blueprint_prompt
    assert "question_type: compliance_check" in blueprint_prompt
    assert "Avoid page-number" in blueprint_prompt

    review = {
        "judge_items": [
            {
                "review": {
                    "record_id": "record-1",
                    "_source_passage": row["passage"],
                }
            }
        ]
    }
    judge_prompt = ProcurementJudge.prompt(None, review)
    for required in (
        "exactly one judgment for each record_id",
        "supported=true",
        "preserves_qualifications=true",
        "authority_correct=true",
        "reasoning_valid=true",
        "Scores 4-5",
        "---BEGIN UNTRUSTED REVIEW BATCH---",
        "---END UNTRUSTED REVIEW BATCH---",
    ):
        assert required in judge_prompt


def test_cross_document_prompts_preserve_specification_contract() -> None:
    row = {
        "relationship_type": "complementary_procedure",
        "pair_id": "pair",
        "shared_terms": ["bid", "security"],
        "source_documents": [
            {
                "source_id": "source_a",
                **_cross_row(
                    "manual-a",
                    "chunk-a",
                    "The bidder shall submit bid security.",
                ),
            },
            {
                "source_id": "source_b",
                **_cross_row(
                    "manual-b",
                    "chunk-b",
                    "Bid security is submitted through the portal.",
                ),
            },
        ],
        "planned_task_type": "cross_document_qa_cot",
        "planned_answerable": True,
    }
    prompt = CrossDocumentGenerator.prompt(None, row)
    for required in (
        "source_a and source_b",
        "alignment terms are untrusted data",
        "PLANNED CONTRACT",
        "underlying procurement work",
        "persona",
        "cross_document_qa_cot",
        "two to four",
        "private hidden chain-of-thought",
        "exact, correctly attributed source-specific evidence",
        "Not answerable from the provided sources.",
        "same_authority_temporal",
        "failure of the complete answer under either-source ablation",
        "---BEGIN UNTRUSTED SOURCES---",
        "---END UNTRUSTED SOURCES---",
    ):
        assert required in prompt

    review = {"judge_items": [{"review": {"record_id": "record-1"}}]}
    judge_prompt = CrossDocumentJudge.prompt(None, review)
    for required in (
        "Full context",
        "Without source_a",
        "Without source_b",
        "unsupported_without_source_ids",
        "reasoning_valid=true",
        "connected_reasoning=true",
        "relationship_correct=true",
        "Scores 4-5",
        "---BEGIN UNTRUSTED REVIEW BATCH---",
        "---END UNTRUSTED REVIEW BATCH---",
    ):
        assert required in judge_prompt


def test_drafting_prompts_preserve_specification_contract() -> None:
    row = {
        "tender_id": "tender-1",
        "task": "drafting",
        "instruction": "Draft the bid-security clause.",
        "tender_context": ["Tender mode: Limited."],
        "manual_context": "The bidder shall submit bid security.",
    }
    prompt = TenderDraftingGenerator.prompt(None, row)
    for required in (
        "complete, ready-to-use",
        "not a title alone, outline, summary",
        "document_blocks",
        "organization identity",
        "bidding structure",
        "[NOT PROVIDED]",
        "state the conflict instead of blending",
        "Do not include a citation list",
        "manual_evidence_quotes",
        "complete verbatim item",
        "---BEGIN UNTRUSTED TENDER FACTS---",
        "---END UNTRUSTED MANUAL CONTEXT---",
        "FINAL CHECK",
    ):
        assert required in prompt

    judge_row = {
        "instruction": row["instruction"],
        "_combined_source_text": "Tender mode: Limited.\nManual rule.",
        "response": "Completed draft.",
    }
    judge_prompt = TenderDraftingJudge.prompt(None, judge_row)
    for required in (
        "supported=true",
        "follows_instruction=true",
        "preserves_policy_qualifications=true",
        "resolves_source_conflicts_safely=true",
        "Scores 4-5",
        "Do not rewrite the draft",
        "---BEGIN UNTRUSTED INSTRUCTION---",
        "---BEGIN UNTRUSTED SOURCES---",
        "---BEGIN UNTRUSTED DRAFT---",
        "FINAL CHECK",
    ):
        assert required in judge_prompt


def test_generation_text_and_representative_pilot_selection() -> None:
    raw = "A useful procurement rule applies to every tender. " * 6 + "\n\n![Page image](page.png)\n\n<table><tr><td>5%</td></tr></table>"
    assert "![Page image]" not in generation_text(raw)
    assert "<table>" in generation_text(raw)
    rows = []
    for index, (manual, category) in enumerate(
        (
            ("goods_2022", "government_manual"),
            ("works_2022", "government_manual"),
            ("nrl_goods_rev1", "nrl_manual"),
            ("nrl_works_rev1", "nrl_manual"),
        ),
        1,
    ):
        rows.append(
            {
                "chunk_id": f"chunk-{index}",
                "manual_id": manual,
                "source_category": category,
                "page": index * 10,
                "start_page": 1,
                "section": "Policy",
                "passage": raw,
                "generation_passage": generation_text(raw),
                "content_class": "policy",
            }
        )
    selected = representative_rows(rows, 3, "seed")
    assert len({row["manual_id"] for row in selected}) == 3
    assert {row["source_category"] for row in selected} == {
        "government_manual",
        "nrl_manual",
    }
    report = corpus_quality_report(rows, [])
    assert report["chunks_with_image_markdown"] == 4
    assert report["chunks_with_html_tables"] == 4


def test_source_quality_preflight_rejects_non_answer_bearing_structures() -> None:
    toc = "\n".join(f"{index}. Procurement topic ........ {index + 10}" for index in range(1, 8))
    toc_row = {
        "generation_passage": toc,
        "content_class": "policy",
    }
    assert "table_of_contents_only" in source_quality_issues(toc_row)

    html_toc = {
        "generation_passage": (
            "MANUAL FOR PROCUREMENT\nTable of Contents\n<table>"
            + "".join(f"<tr><td>{index}. Rule</td><td>{index + 20}</td></tr>" for index in range(8))
            + "</table>"
        ),
        "content_class": "table",
    }
    assert "table_of_contents_only" in source_quality_issues(html_toc)

    glossary = "\n".join(
        (
            "ABC - Alpha Buying Committee",
            "DEF - Department Evaluation Form",
            "GHI - General Handling Instruction",
            "JKL - Joint Knowledge List",
            "MNO - Manual Notice Office",
            "PQR - Procurement Quality Review",
        )
    )
    glossary_row = {
        "generation_passage": glossary,
        "content_class": "policy",
    }
    assert "abbreviation_glossary_only" in source_quality_issues(glossary_row)

    policy = {
        "generation_passage": (
            "The tendering officer shall record the evaluation before award. "
            "The approving authority may request clarification when the record "
            "is incomplete, but no bidder may alter its quoted price. " * 3
        ),
        "content_class": "policy",
    }
    assert source_quality_issues(policy) == []


def test_late_introduction_is_policy_not_front_matter() -> None:
    row = {
        "generation_passage": "The contractor shall submit the work programme.",
        "section": "7.4.1 Introduction to contract monitoring",
        "page": 185,
        "start_page": 1,
    }
    assert _content_class(row) == "policy"


def test_short_manual_is_not_treated_as_all_front_matter() -> None:
    # A one-page Office Memorandum whose entire loaded content fits inside
    # the position-based front-matter window (start_page..start_page+7) has
    # no front matter to skip -- unlike a 260+ page manual, applying the
    # position rule here would zero out the whole manual.
    row = {
        "generation_passage": "It has been decided to partially amend para 5.6.8 as under.",
        "section": None,
        "page": 1,
        "start_page": 1,
        "manual_page_count": 1,
    }
    assert _content_class(row) == "policy"


def test_position_rule_still_applies_when_manual_extends_past_the_window() -> None:
    # A manual long enough that page 1 really is inside real front matter
    # (e.g. a cover page) keeps the existing behavior.
    row = {
        "generation_passage": "MANUAL FOR PROCUREMENT OF GOODS 2017",
        "section": None,
        "page": 1,
        "start_page": 1,
        "manual_page_count": 262,
    }
    assert _content_class(row) == "front_matter"


def test_position_rule_applies_when_manual_page_count_is_unknown() -> None:
    # Callers that don't supply manual_page_count (e.g. ad hoc row
    # construction) keep the pre-fix position-only behavior.
    row = {
        "generation_passage": "Some early-page text.",
        "section": None,
        "page": 2,
        "start_page": 1,
    }
    assert _content_class(row) == "front_matter"


def test_short_exact_evidence_is_schema_valid() -> None:
    draft = QABlueprintDraft(
        task="general_reference",
        persona="general_user",
        persona_need="Identify the named party before acting on the rule.",
        instruction_goal="Identify the party named in the source table.",
        must_cover=["The named party is Seller."],
        evidence=[{"quote": "Seller"}],
    )
    assert draft.evidence[0].quote == "Seller"


def test_question_intent_planning_is_feasible_balanced_and_deterministic(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        generation_pipeline.QUALITY,
        "question_type_weights",
        {
            "direct_fact": 0.2,
            "threshold": 0.4,
            "exception": 0.4,
        },
    )
    rows = [
        {
            "chunk_id": "threshold",
            "generation_passage": "The security shall be 5% and remain valid for 30 days.",
        },
        {
            "chunk_id": "exception",
            "generation_passage": "The bidder shall submit the form unless the stated exception applies.",
        },
        {
            "chunk_id": "plain",
            "generation_passage": "The procurement record contains the approved title.",
        },
    ]
    assert eligible_question_types(rows[0]) >= {"threshold", "direct_fact"}
    first = plan_question_types(rows, "seed")
    second = plan_question_types(rows, "seed")
    assert first == second
    assert first["threshold"] in eligible_question_types(rows[0])
    assert first["exception"] in eligible_question_types(rows[1])
    assert first["plain"] == "direct_fact"


def test_question_style_planning_is_compatible_balanced_and_deterministic() -> None:
    question_types = {f"row-{index}": question_type for index, question_type in enumerate(["direct_fact", "procedure", "exception", "comparison"] * 4)}
    first = plan_question_styles(question_types, "style-seed")
    assert first == plan_question_styles(question_types, "style-seed")
    for chunk_id, style in first.items():
        question_type = question_types[chunk_id]
        assert style in generation_pipeline.QUESTION_TYPE_STYLES[question_type]
    counts = {style: list(first.values()).count(style) for style in set(first.values())}
    assert max(counts.values()) <= 4


def test_coverage_planner_never_assigns_cot_to_direct_fact(monkeypatch) -> None:
    monkeypatch.setitem(generation_pipeline.QUALITY, "question_type_weights", {"direct_fact": 1.0})
    monkeypatch.setitem(generation_pipeline.QUALITY, "qa_cot_fraction", 1.0)
    planned = plan_single_document_requests(
        [
            {
                "chunk_id": "complex-looking-fact",
                "generation_passage": ("If approved, the buyer shall retain the record unless the " "authority directs otherwise."),
            }
        ],
        "coverage-seed",
    )[0]
    assert planned["planned_task_type"] == "qa"
    assert planned["planned_reasoning_operation"] == "lookup"
    assert planned["planned_difficulty"] == "basic"
    assert planned["planned_coverage_cell_id"].startswith("qacell-")


def test_difficult_cot_cells_expand_and_grounded_quality_wins(monkeypatch) -> None:
    monkeypatch.setitem(
        generation_pipeline.QUALITY,
        "single_document_best_of_n",
        {"basic": 1, "intermediate": 2, "advanced": 3},
    )
    blueprint = {
        "blueprint_id": "blueprint-1",
        "planned_task_type": "qa_cot",
        "planned_reasoning_operation": "apply_condition",
        "planned_difficulty": "advanced",
    }
    expanded = expand_single_generation_candidates([blueprint])
    assert len(expanded) == 3
    assert len({row["candidate_request_id"] for row in expanded}) == 3

    weak = {
        "record_id": "weak",
        "blueprint_id": "blueprint-1",
        "answer": "A longer but weaker answer.",
        "claims": [{"evidence": [{"quote": "rule"}]}],
        "evidence": [{"quote": "rule"}],
        "judge": {"score": 4, "preserves_qualifications": True},
    }
    strong = {
        **weak,
        "record_id": "strong",
        "answer": "Qualified answer.",
        "judge": {"score": 5, "preserves_qualifications": True},
    }
    selected, rejected = select_best_single_candidates([weak, strong])
    assert [row["record_id"] for row in selected] == ["strong"]
    assert rejected[0]["best_of_n"]["reason"] == "weaker_grounded_sibling"


def test_question_style_gate_rejects_source_and_cosmetic_persona_templates() -> None:
    assert question_style_issues("According to the manual, who approves the tender?", "general_user") == ["templated_source_attribution_opener"]
    assert question_style_issues("As an auditor, what record should I inspect?", "auditor") == ["cosmetic_persona_preamble"]
    assert question_style_issues("Under what circumstances may the authority reject the bid?", "auditor") == []


def test_cosmetic_persona_prefix_and_operation_aliases_are_narrowly_repaired() -> None:
    question, repaired = remove_cosmetic_persona_prefix(
        "As a procurement officer, what record must be retained?",
        "procurement_officer",
    )
    assert repaired is True
    assert question == "What record must be retained?"
    unchanged, repaired = remove_cosmetic_persona_prefix(
        "As an auditor reviewing the 2025 Manual, what must be checked?",
        "auditor",
    )
    assert repaired is False
    assert unchanged.startswith("As an auditor reviewing")
    assert canonical_reasoning_operation("Identify acceptance requirement") == "lookup"
    assert canonical_reasoning_operation("Connect to manual revision date") == "resolve_time"
    assert canonical_reasoning_operation("synthesize_rule") == "combine"
    assert canonical_reasoning_operation("invent a scenario") is None


def test_role_profile_preserves_profile_defaults_but_role_limits_win() -> None:
    glm = generation_pipeline._role_profile("generation", "glm")
    assert glm["request_timeout"] == 1200
    assert glm["max_retries"] == 1
    assert glm["max_concurrent_requests"] == 45

    resolved = generation_pipeline._role_profile("judge", "gemma")
    assert resolved["generation_params"]["max_tokens"] == 2048
    assert resolved["generation_params"]["temperature"] == 1.0
    assert resolved["generation_params"]["top_p"] == 0.95
    assert resolved["generation_params"]["top_k"] == 64
    assert resolved["generation_params"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    ministral = generation_pipeline._role_profile("judge", "ministral")
    assert ministral["generation_params"]["temperature"] == 0.05
    assert ministral["generation_params"]["max_tokens"] == 2048
    assert ministral["generation_params"]["extra_body"] == {}
    assert ministral["max_concurrent_requests"] == 16
    assert ministral["served_model_env"] == "MINISTRAL_MODEL"
    assert configured_context_window(ministral) == 65536

    gemma_judge = generation_pipeline._role_profile("judge", "gemma_structured")
    assert generation_pipeline.CONFIG["models"]["judge"]["default_profile"] == ("gemma_structured")
    assert gemma_judge["profile_name"] == "gemma_structured"
    assert gemma_judge["generation_params"]["max_tokens"] == 2048
    assert gemma_judge["generation_params"]["temperature"] == 1.0
    assert gemma_judge["generation_params"]["top_p"] == 0.95
    assert gemma_judge["generation_params"]["top_k"] == 64
    assert gemma_judge["generation_params"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert gemma_judge["max_concurrent_requests"] == 45


def test_output_rescue_raises_the_recovery_completion_budget_and_timeout() -> None:
    generation = generation_pipeline._role_profile("generation", "glm")
    generation_rescue = generation_pipeline._output_rescue_profile(generation)
    assert generation["generation_params"]["max_tokens"] == 5000
    assert generation_rescue is not None
    assert generation_rescue["generation_params"]["max_tokens"] == 12000
    assert generation_rescue["max_concurrent_requests"] == 45
    # The rescue's request_timeout must differ from and exceed the primary
    # profile's, not be silently inherited unchanged (T18).
    assert generation_rescue["request_timeout"] != generation["request_timeout"]
    assert generation_rescue["request_timeout"] > generation["request_timeout"]
    assert generation_rescue["request_timeout"] == 3600

    judge = generation_pipeline._role_profile("judge", "gemma_thinking")
    judge_rescue = generation_pipeline._output_rescue_profile(judge)
    assert judge["generation_params"]["max_tokens"] == 2048
    assert judge_rescue is not None
    assert judge_rescue["generation_params"]["max_tokens"] == 4096
    assert judge_rescue["max_concurrent_requests"] == 45
    assert judge_rescue["generation_params"]["temperature"] == 1.0
    assert judge_rescue["generation_params"]["top_p"] == 0.95
    assert judge_rescue["generation_params"]["top_k"] == 64
    assert judge_rescue["request_timeout"] > judge["request_timeout"]
    assert generation_pipeline._rescue_input({"record_id": "one"}, 4096)["_output_rescue_max_tokens"] == 4096


def test_output_rescue_timeout_override_is_opt_in_and_never_lowers_timeout() -> None:
    # No output_rescue_request_timeout configured: behavior is unchanged,
    # request_timeout is inherited from the primary profile as before.
    profile_without_override = {
        "request_timeout": 1800,
        "max_concurrent_requests": 128,
        "context_window": 32768,
        "output_rescue_max_tokens": 4096,
        "generation_params": {"max_tokens": 2048},
    }
    rescue = generation_pipeline._output_rescue_profile(profile_without_override)
    assert rescue is not None
    assert rescue["request_timeout"] == 1800

    # A configured override below the primary timeout never lowers it.
    profile_with_low_override = {
        **profile_without_override,
        "output_rescue_request_timeout": 900,
    }
    rescue_low = generation_pipeline._output_rescue_profile(profile_with_low_override)
    assert rescue_low is not None
    assert rescue_low["request_timeout"] == 1800

    # A configured override above the primary timeout raises it.
    profile_with_high_override = {
        **profile_without_override,
        "output_rescue_request_timeout": 3600,
    }
    rescue_high = generation_pipeline._output_rescue_profile(profile_with_high_override)
    assert rescue_high is not None
    assert rescue_high["request_timeout"] == 3600


def test_output_rescue_retries_only_missing_rows_in_separate_checkpoint(
    monkeypatch,
) -> None:
    calls: list[tuple[str, list[dict]]] = []

    class RescueLLM:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(
        generation_pipeline,
        "_llm_kwargs",
        lambda _profile: {},
    )
    monkeypatch.setattr(
        generation_pipeline,
        "_rendered_prompt_budget",
        lambda _llm, _row, _profile: {"passed": True},
    )

    def fake_execute(stage, role, _llm, inputs):
        calls.append((stage, inputs))
        assert role == "generation"
        return [{"id": inputs[0]["id"], "value": "rescued"}]

    monkeypatch.setattr(
        generation_pipeline,
        "_execute_llm_stage",
        fake_execute,
    )
    profile = {
        "context_window": 32768,
        "output_rescue_max_tokens": 4096,
        "generation_params": {"max_tokens": 2048},
    }
    rows, rescued, rejected = generation_pipeline._rescue_missing_generation_rows(
        stage="example",
        llm_type=RescueLLM,
        profile=profile,
        inputs=[{"id": "one"}, {"id": "two"}],
        outputs=[{"id": "one", "value": "primary"}],
        input_id=lambda row: row["id"],
        output_id=lambda row: row.get("id"),
    )
    assert {row["id"] for row in rows} == {"one", "two"}
    assert rescued == 1
    assert rejected == []
    assert calls[0][0] == "example_output_rescue"
    assert calls[0][1][0]["_output_rescue_max_tokens"] == 4096


def test_judge_output_rescue_dispatches_only_missing_decision(monkeypatch) -> None:
    calls: list[tuple[str, list[dict]]] = []

    class RescueJudge:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(generation_pipeline, "_llm_kwargs", lambda _profile: {})
    monkeypatch.setattr(
        generation_pipeline,
        "_budget_judge_rows",
        lambda _judge, rows, _profile: (rows, []),
    )

    def fake_execute(stage, role, _llm, inputs):
        calls.append((stage, inputs))
        assert role == "judge"
        return [{"record_id": inputs[0]["judge_items"][0]["record_id"]}]

    monkeypatch.setattr(
        generation_pipeline,
        "_execute_llm_stage",
        fake_execute,
    )
    profile = {
        "context_window": 65536,
        "output_rescue_max_tokens": 4096,
        "generation_params": {"max_tokens": 2048},
    }
    inputs = [
        {"judge_items": [{"record_id": "one"}]},
        {"judge_items": [{"record_id": "two"}]},
    ]
    rows, rescued, rejected = generation_pipeline._rescue_missing_judge_rows(
        stage="cross_judge_pass_001",
        llm_type=RescueJudge,
        profile=profile,
        inputs=inputs,
        outputs=[{"record_id": "one"}],
    )
    assert {row["record_id"] for row in rows} == {"one", "two"}
    assert rescued == 1
    assert rejected == []
    assert calls[0][0] == "cross_judge_pass_001_output_rescue"
    assert calls[0][1][0]["judge_items"][0]["record_id"] == "two"


def test_answerability_output_rescue_accepts_direct_record_inputs(
    monkeypatch,
) -> None:
    calls: list[tuple[str, list[dict]]] = []

    class RescueJudge:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(generation_pipeline, "_llm_kwargs", lambda _profile: {})
    monkeypatch.setattr(
        generation_pipeline,
        "_rendered_prompt_budget",
        lambda _judge, _row, _profile: {"passed": True},
    )

    def fake_execute(stage, role, _llm, inputs):
        calls.append((stage, inputs))
        assert role == "judge"
        return [{"record_id": inputs[0]["record_id"]}]

    monkeypatch.setattr(generation_pipeline, "_execute_llm_stage", fake_execute)
    profile = {
        "context_window": 32768,
        "output_rescue_max_tokens": 4096,
        "generation_params": {"max_tokens": 2048},
    }
    rows, rescued, rejected = generation_pipeline._rescue_missing_judge_rows(
        stage="answerability_judge",
        llm_type=RescueJudge,
        profile=profile,
        inputs=[{"record_id": "one"}, {"record_id": "two"}],
        outputs=[{"record_id": "one"}],
    )

    assert {row["record_id"] for row in rows} == {"one", "two"}
    assert rescued == 1
    assert rejected == []
    assert calls[0][0] == "answerability_judge_output_rescue"
    assert calls[0][1][0]["record_id"] == "two"


def test_thinking_generation_profile_preserves_template_and_sampling() -> None:
    resolved = generation_pipeline._role_profile("generation", "gemma_thinking")
    params = resolved["generation_params"]

    assert resolved["served_model_env"] == "MODEL"
    assert resolved["base_url_env"] == "LLM_BASE_URL"
    assert resolved["api_key_env"] == "LLM_API_KEY"
    assert resolved["max_concurrent_requests"] == 45
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["top_k"] == 64
    assert params["max_tokens"] == 5000
    assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_stringified_json_list_recovery_is_narrow_and_audited() -> None:
    candidate = GroundedCandidateDraft.model_validate(
        {
            "question": "What record must the buyer retain?",
            "answer": "The procurement record.",
            "claims": ('[{"statement":"Retain the record.",' '"evidence":[{"quote":"procurement record"}]}]'),
            "reasoning_steps": "[]",
        }
    )
    assert candidate.model_dump()["claims"][0]["statement"] == "Retain the record."
    assert collect_structural_repairs(candidate) == [
        "stringified_json_list:claims",
        "stringified_json_list:reasoning_steps",
    ]
    assert "_structural_repairs" not in GroundedCandidateDraft.model_json_schema()["properties"]
    try:
        GroundedCandidateDraft.model_validate(
            {
                "question": "What record must the buyer retain?",
                "answer": "The procurement record.",
                "claims": "claims appear here",
                "reasoning_steps": [],
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("non-JSON list prose must not be repaired")


def test_blueprint_scalar_and_overlong_lists_are_bounded_and_audited() -> None:
    blueprint = QABlueprintDraft.model_validate(
        {
            "task": "general_reference",
            "persona": "general_user",
            "persona_need": "Check the record before taking procurement action.",
            "instruction_goal": "Identify the complete source-supported requirement.",
            "must_cover": "The buyer retains the procurement record.",
            "evidence": [{"quote": f"complete evidence quote {index}"} for index in range(5)],
        }
    )
    assert blueprint.must_cover == ["The buyer retains the procurement record."]
    assert len(blueprint.evidence) == 4
    assert collect_structural_repairs(blueprint) == [
        "scalar_string_to_list:must_cover",
        "list_clipped:evidence:5>4",
    ]


def test_unknown_model_labels_parse_for_audited_fail_closed_rejection() -> None:
    decision = JudgeDecision.model_validate(
        _judge_decision(
            recommended_task="delivery_of_sold_material",
            recommended_persona="procuring_entity",
        )
    )
    assert decision.recommended_task == "delivery_of_sold_material"
    cross = CrossCandidateBatch.model_validate(
        {
            "examples": [
                {
                    "task_type": "wrong_shape",
                    "task": "payment_procedures",
                    "persona": "procuring_entity",
                    "question_type": "invented_type",
                    "question": "Which source-supported action should be taken?",
                    "answer": "Use both sources.",
                    "claims": [
                        {
                            "statement": "A material claim.",
                            "evidence": [
                                {
                                    "source_id": "source_a",
                                    "quote": "A complete evidence quotation.",
                                }
                            ],
                        }
                    ],
                    "reasoning_steps": [
                        {
                            "operation": "sequence",
                            "statement": "An unsupported operation label.",
                            "evidence": [
                                {
                                    "source_id": "source_a",
                                    "quote": "A complete evidence quotation.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )
    assert cross.examples[0].task == "payment_procedures"
    assert cross.examples[0].reasoning_steps[0].operation == "sequence"


def test_answer_format_and_category_portfolio_checks() -> None:
    bounds = {
        "concise_direct": [3, 12],
        "ordered_steps": [3, 30],
    }
    assert (
        answer_format_issues(
            "The authority approves the tender.",
            "The authority approves the tender.",
            "concise_direct",
            bounds,
        )
        == []
    )
    issues = answer_format_issues(
        "Lecture point: imagine a bidder using 99 percent.",
        "The bidder shall submit its tender.",
        "ordered_steps",
        bounds,
    )
    assert "unsupported_instructional_embellishment:lecture_point" in issues
    assert "ordered_steps_format_missing_structure" in issues

    records = [{"question_type": "direct_fact", "record_id": f"direct-{index}"} for index in range(8)] + [
        {"question_type": "threshold", "record_id": "threshold"},
        {"question_type": "exception", "record_id": "exception"},
    ]
    kept, removed = enforce_category_diversity(
        records,
        "question_type",
        0.5,
    )
    report = categorical_diversity(kept, "question_type")
    assert removed == 6
    assert report["top_share"] == 0.5

    lengths = answer_length_statistics(
        [
            {"answer": "one two three", "answer_format": "concise_direct"},
            {"answer": "one two three four five", "answer_format": "concise_direct"},
        ]
    )
    assert lengths["overall"]["median"] in {3, 5}
    assert lengths["overall"]["maximum"] == 5


def test_explicit_task_planning_and_request_coverage(monkeypatch) -> None:
    monkeypatch.setitem(generation_pipeline.QUALITY, "qa_cot_fraction", 0.4)
    rows = [
        {
            "chunk_id": f"chunk-{index}",
            "generation_passage": ("The buyer shall act if the stated condition applies. " f"Rule {index}. " * 8),
        }
        for index in range(5)
    ]
    planned = plan_single_document_requests(rows, "seed")
    assert sum(row["planned_task_type"] == "qa_cot" for row in planned) == 2
    assert all(row["planned_answerable"] for row in planned)
    materialized = [
        {"parent_request_id": planned[0]["planned_request_id"]},
        {"parent_request_id": planned[1]["planned_request_id"]},
    ]
    coverage = request_coverage(planned, materialized)
    assert coverage["expected_requests"] == 5
    assert coverage["materialized_requests"] == 2
    assert len(coverage["missing_request_ids"]) == 3

    bundles = [{"source_bundle_id": f"bundle-{index}"} for index in range(4)]
    cross = plan_cross_document_requests(bundles, "seed")
    assert {row["planned_task_type"] for row in cross} == {
        "cross_document_qa",
        "cross_document_qa_cot",
    }


def test_judge_coverage_excludes_deterministic_and_budget_rejections() -> None:
    planned = [{"planned_request_id": f"request-{index}", "planned_task_type": "qa"} for index in range(4)]
    # request-0: deterministically rejected, never reached the judge.
    # request-1: passed determinism/dedup but was prompt-budget rejected.
    # request-2: passed determinism/dedup and reached the judge stage.
    # request-3: passed determinism/dedup but the judge stage never returned
    #            a terminal decision (a genuinely missing judge response).
    generated = [
        {"parent_request_id": "request-1", "record_id": "record-1"},
        {"parent_request_id": "request-2", "record_id": "record-2"},
        {"parent_request_id": "request-3", "record_id": "record-3"},
    ]
    prompt_rejected = [{"judge_items": [{"record_id": "record-1"}]}]
    eligible = judge_eligible_planned(planned, generated, prompt_rejected)
    assert {row["planned_request_id"] for row in eligible} == {"request-2", "request-3"}

    judged = [{"parent_request_id": "request-2", "record_id": "record-2"}]
    coverage = request_coverage(eligible, judged)
    assert coverage["missing_request_ids"] == ["request-3"]


def test_post_retry_omissions_become_terminal_audit_rows() -> None:
    planned = [
        {"planned_request_id": "request-a", "planned_task_type": "qa"},
        {"planned_request_id": "request-b", "planned_task_type": "qa_cot"},
    ]
    rows = [{"parent_request_id": "request-a", "record_id": "record-a"}]
    terminal = materialize_terminal_failures(
        planned,
        rows,
        planned_id=lambda row: row["planned_request_id"],
        record_id=lambda row: row.get("parent_request_id"),
        stage="generation",
        base_fields=lambda row: {
            "parent_request_id": row["planned_request_id"],
            "task_type": row["planned_task_type"],
        },
    )
    assert request_coverage(planned, terminal)["missing_request_ids"] == []
    failure = next(row for row in terminal if row.get("parent_request_id") == "request-b")
    assert failure["terminal_state"] == "model_failure_after_retries"


def test_failed_blueprint_retains_answerability_in_generation_audit() -> None:
    failure = materialize_blueprint_rejection(
        {
            "planned_request_id": "request-a",
            "planned_task_type": "qa",
            "planned_question_type": "direct_fact",
            "planned_question_style": "plain_query",
            "planned_answer_format": "concise_direct",
            "planned_answerable": True,
            "manual_id": "manual-a",
            "terminal_state": "model_failure_after_retries",
        }
    )
    assert failure["answerable"] is True
    assert failure["terminal_state"] == "blueprint_rejected_or_failed"
    assert failure["deterministic_checks"]["issues"] == ["model_failure_after_retries"]


def test_judge_rejects_false_abstention_and_taxonomy_acquiescence() -> None:
    source = "Retail and Wholesale traders may register for Priority Sector Lending only."
    record = {
        "record_id": "record-1",
        "task": "preference_policy_application",
        "persona": "bidder",
        "answerable": False,
        "_source_passage": source,
    }
    row = {"judge_items": [{"record_id": "record-1", "record": record}]}
    response = JudgeBatch.model_validate(
        {
            "judgments": [
                {
                    "record_id": "record-1",
                    "decision": {
                        "supported": True,
                        "relevant": True,
                        "preserves_qualifications": True,
                        "authority_correct": True,
                        "reasoning_valid": True,
                        "question_natural": True,
                        "persona_relevant": True,
                        "recommended_task": "nit_filling",
                        "recommended_persona": "bidder",
                        "answer_found_in_source": True,
                        "answer_quotes": [source],
                        "score": 5,
                        "issues": [],
                    },
                }
            ]
        }
    )
    judged = ProcurementJudge.parse(
        SimpleNamespace(model_name="judge-model"),
        row,
        response,
    )[0]["judge"]
    assert judged["task_correct"] is False
    assert judged["answerability_correct"] is False
    assert judged["accepted"] is False


def test_judge_accepts_multiple_independent_verbatim_answer_spans() -> None:
    first = "Cash or gift cheques may not be accepted."
    second = "Particular care is required for firms in current tenders."
    source = f"{first} Other guidance appears here. {second}"
    record = {
        "record_id": "record-1",
        "task": "ethics_and_risk_management",
        "persona": "procurement_officer",
        "answerable": True,
        "_source_passage": source,
    }
    row = {"judge_items": [{"record_id": "record-1", "record": record}]}
    response = JudgeBatch.model_validate(
        {
            "judgments": [
                {
                    "record_id": "record-1",
                    "decision": {
                        "supported": True,
                        "relevant": True,
                        "preserves_qualifications": True,
                        "authority_correct": True,
                        "reasoning_valid": True,
                        "question_natural": True,
                        "persona_relevant": True,
                        "recommended_task": "ethics_and_risk_management",
                        "recommended_persona": "procurement_officer",
                        "answer_found_in_source": True,
                        "answer_quotes": [first, second],
                        "score": 5,
                        "issues": [],
                    },
                }
            ]
        }
    )

    judged = ProcurementJudge.parse(
        SimpleNamespace(model_name="judge-model"),
        row,
        response,
    )[0]["judge"]

    assert judged["answerability_correct"] is True
    assert judged["accepted"] is True


def test_drafting_surface_normalization_and_semantic_validation() -> None:
    normalized, repairs = normalize_drafting_response("Heading<br>1. Scope\r\nBody")
    assert normalized == "Heading\n1. Scope\nBody"
    assert repairs == ["html_breaks_to_newlines"]
    row = {
        "manual_passages": ["The cap is 5%."],
        "combined_source_text": ("Tender ID: NRL-GOODS-CRANE-1009379-V2.\n" "Organization: NUMALIGARH REFINERY LIMITED.\nThe cap is 5%."),
        "tender_context": ["Tender ID: NRL-GOODS-CRANE-1009379-V2."],
    }
    valid = DraftingResult(
        document_blocks=[
            DraftingBlock(text="1. Scope"),
            DraftingBlock(text="Tender ID: NRL-GOODS-CRANE-1009379-V2."),
            DraftingBlock(text="Organization: NUMALIGARH REFINERY LIMITED."),
            DraftingBlock(text="The cap is 5%."),
        ],
        manual_evidence_quotes=["The cap is 5%."],
        tender_facts_used=["Tender ID: NRL-GOODS-CRANE-1009379-V2."],
    )
    assert drafting_validation_issues(row, valid) == []
    invalid = valid.model_copy(
        update={
            "document_blocks": [
                DraftingBlock(text="<b>Scope</b>"),
                DraftingBlock(text="Organization: Invented Division."),
                DraftingBlock(text="The cap is 10%."),
            ]
        }
    )
    issues = drafting_validation_issues(row, invalid)
    assert "draft_contains_html_markup" in issues
    assert "unsupported_authority:Invented Division" in issues
    assert "unsupported_number:10%" in issues


def test_drafting_aggregate_is_derived_from_block_first_use() -> None:
    blocks = [
        DraftingBlock(
            text="First",
            tender_facts_used=["fact-b", "fact-a"],
            manual_evidence_quotes=["quote-b"],
        ),
        DraftingBlock(
            text="Second",
            tender_facts_used=["fact-a", "fact-c"],
            manual_evidence_quotes=["quote-a", "quote-b"],
        ),
    ]

    assert _stable_block_union(blocks, "tender_facts_used") == [
        "fact-b",
        "fact-a",
        "fact-c",
    ]
    assert _stable_block_union(blocks, "manual_evidence_quotes") == [
        "quote-b",
        "quote-a",
    ]


def test_exports_keep_qa_and_rationale_task_files_disjoint(tmp_path: Path) -> None:
    records = []
    for index, task_type in enumerate(("qa", "qa_cot", "cross_document_qa", "cross_document_qa_cot")):
        cross = task_type.startswith("cross_document_")
        records.append(
            {
                "record_id": f"record-{index}",
                "split": "train",
                "manual_id": "manual",
                "source_documents": ([{"manual_id": "manual"}, {"manual_id": "other"}] if cross else []),
                "source_chunk_ids": [f"chunk-{index}"],
                "citations": [
                    {
                        "citation_id": f"chunk-{index}",
                        "manual_id": "manual",
                        "chunk_id": f"chunk-{index}",
                        "page": 1,
                        "section": "Policy",
                        "quote": "The stated action is required.",
                    }
                ],
                "task_type": task_type,
                "task": "general_reference",
                "persona": "general_user",
                "question": "What is required?",
                "answer": "The stated action is required.",
                "answerable": True,
                "reasoning_steps": (
                    [
                        {
                            "operation": "lookup",
                            "statement": "Apply the stated rule.",
                            "evidence_quotes": ["The stated action is required."],
                        }
                    ]
                    if task_type.endswith("_cot")
                    else []
                ),
                "claims": [
                    {
                        "statement": "The stated action is required.",
                        "evidence": [
                            {
                                "manual_id": "manual",
                                "chunk_id": f"chunk-{index}",
                                "quote": "The stated action is required.",
                            }
                        ],
                    }
                ],
                "evidence": [
                    {
                        "manual_id": "manual",
                        "chunk_id": f"chunk-{index}",
                        "quote": "The stated action is required.",
                    }
                ],
                "question_type": "direct_fact",
                **(
                    {
                        "relationship_type": "comparison",
                        "source_bundle_id": f"bundle-{index}",
                    }
                    if cross
                    else {}
                ),
            }
        )
    stats = export_records(
        records,
        [{"manual_id": "manual"}, {"manual_id": "other"}],
        tmp_path,
        "test-run",
    )
    assert stats["records"] == 4
    expected = {
        "qa_sft.jsonl": "record-0",
        "qa_cot_sft.jsonl": "record-1",
        "cross_document_qa_sft.jsonl": "record-2",
        "cross_document_qa_cot_sft.jsonl": "record-3",
    }
    for filename, record_id in expected.items():
        text = (tmp_path / filename).read_text(encoding="utf-8")
        assert record_id in text
        assert text.count("\n") == 1
        assert list(json.loads(text))[-1] == "citations"
    for line in (tmp_path / "canonical.jsonl").read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["reasoning_graph"]["validation"]["passed"]


def _split_record(record_id: str, split: str, manual_id: str) -> dict:
    quote = f"The stated action is required for {manual_id}."
    return {
        "record_id": record_id,
        "split": split,
        "manual_id": manual_id,
        "source_chunk_ids": [f"chunk-{record_id}"],
        "citations": [
            {
                "citation_id": f"chunk-{record_id}",
                "manual_id": manual_id,
                "chunk_id": f"chunk-{record_id}",
                "page": 1,
                "section": "Policy",
                "quote": quote,
            }
        ],
        "task_type": "qa",
        "task": "general_reference",
        "persona": "general_user",
        "question": f"What is required under {manual_id}?",
        "answer": quote,
        "answerable": True,
        "reasoning_steps": [],
        "claims": [
            {
                "statement": quote,
                "evidence": [
                    {
                        "manual_id": manual_id,
                        "chunk_id": f"chunk-{record_id}",
                        "quote": quote,
                    }
                ],
            }
        ],
        "evidence": [
            {
                "manual_id": manual_id,
                "chunk_id": f"chunk-{record_id}",
                "quote": quote,
            }
        ],
        "question_type": "direct_fact",
    }


def test_export_never_mixes_splits_between_sft_and_eval_files(tmp_path: Path) -> None:
    records = [
        _split_record("train-1", "train", "manual-a"),
        _split_record("train-2", "train", "manual-b"),
        _split_record("val-1", "validation", "manual-c"),
        _split_record("test-1", "test", "manual-d"),
    ]
    export_records(
        records,
        [{"manual_id": m} for m in ("manual-a", "manual-b", "manual-c", "manual-d")],
        tmp_path,
        "test-run",
    )

    qa_sft_ids = {json.loads(line)["record_id"] for line in (tmp_path / "qa_sft.jsonl").read_text(encoding="utf-8").splitlines()}
    rag_ids = {json.loads(line)["record_id"] for line in (tmp_path / "rag.jsonl").read_text(encoding="utf-8").splitlines()}
    eval_ids = {json.loads(line)["record_id"] for line in (tmp_path / "eval.jsonl").read_text(encoding="utf-8").splitlines()}

    assert qa_sft_ids == {"train-1", "train-2"}
    assert rag_ids == {"train-1", "train-2"}
    assert eval_ids == {"val-1", "test-1"}
    # The defining invariant: no record_id is ready-to-train and ready-to-eval at once.
    assert not (qa_sft_ids & eval_ids)
    assert not (rag_ids & eval_ids)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    stats = manifest["statistics"]
    assert stats["qa_sft_records"] == 2
    assert stats["rag_records"] == 2
    assert stats["eval_records"] == 2
    assert stats["records"] == 4


def test_reasoning_graph_is_stable_connected_and_terminal() -> None:
    record = {
        "record_id": "record-graph",
        "answer": "Rule A and Rule B apply together.",
        "claims": [
            {
                "statement": "Rule A applies.",
                "evidence": [{"source_id": "source_a", "quote": "Rule A applies."}],
            },
            {
                "statement": "Rule B applies.",
                "evidence": [{"source_id": "source_b", "quote": "Rule B applies."}],
            },
        ],
        "reasoning_steps": [
            {
                "operation": "combine",
                "statement": "Combine both rules.",
                "evidence": [
                    {"source_id": "source_a", "quote": "Rule A applies."},
                    {"source_id": "source_b", "quote": "Rule B applies."},
                ],
            }
        ],
    }
    first = build_reasoning_graph(record)
    second = build_reasoning_graph(record)
    assert first == second
    assert first["validation"] == {"passed": True, "issues": []}
    assert len(first["terminal_claim_ids"]) == 1


def _exportable_record(record_id: str, *, unused_claim: bool) -> dict:
    claims = [
        {
            "statement": "Rule A applies.",
            "evidence": [{"source_id": "source_a", "quote": "Rule A applies."}],
        }
    ]
    if unused_claim:
        claims.append(
            {
                "statement": "Rule B also applies.",
                "evidence": [{"source_id": "source_a", "quote": "Rule B also applies."}],
            }
        )
    return {
        "record_id": record_id,
        "split": "train",
        "manual_id": "manual",
        "source_chunk_ids": ["chunk-1"],
        "citations": [
            {
                "citation_id": "chunk-1",
                "manual_id": "manual",
                "chunk_id": "chunk-1",
                "page": 1,
                "section": "Policy",
                "quote": "Rule A applies.",
            }
        ],
        "task_type": "qa_cot",
        "task": "general_reference",
        "persona": "general_user",
        "question": "What rule applies?",
        "answer": "Rule A applies.",
        "answerable": True,
        "claims": claims,
        "evidence": [{"manual_id": "manual", "chunk_id": "chunk-1", "quote": "Rule A applies."}],
        "question_type": "direct_fact",
        "reasoning_steps": [
            {
                "operation": "lookup",
                "statement": "Apply rule A.",
                "evidence_quotes": ["Rule A applies."],
            }
        ],
    }


def test_export_drops_only_graph_invalid_records_instead_of_aborting(tmp_path: Path) -> None:
    records = [
        _exportable_record("record-good", unused_claim=False),
        _exportable_record("record-bad", unused_claim=True),
    ]
    stats = export_records(records, [{"manual_id": "manual"}], tmp_path, "test-run")
    assert stats["records"] == 1
    assert stats["reasoning_graphs_rejected"] == 1
    # export_records mutates the passed-in list to match what was exported.
    assert [row["record_id"] for row in records] == ["record-good"]
    canonical_ids = [json.loads(line)["record_id"] for line in (tmp_path / "canonical.jsonl").read_text(encoding="utf-8").splitlines()]
    assert canonical_ids == ["record-good"]
    rejected = [json.loads(line) for line in (tmp_path / "reasoning_graph_rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejected == [{"record_id": "record-bad", "issues": ["disconnected_claims", "unused_source_claim"]}]


def test_leakage_audit_detects_question_and_chunk_across_splits() -> None:
    rows = [
        {
            "record_id": "one",
            "split": "train",
            "question": "What rule applies?",
            "source_chunk_ids": ["chunk-a"],
        },
        {
            "record_id": "two",
            "split": "test",
            "question": "What rule applies!",
            "source_chunk_ids": ["chunk-a"],
        },
    ]
    audit = leakage_audit(rows)
    assert not audit["passed"]
    assert audit["collisions"]["chunk"]
    assert audit["collisions"]["normalized_question"]


def test_leakage_audit_covers_single_document_manual_and_section() -> None:
    rows = [
        {
            "record_id": "one",
            "split": "train",
            "question": "What does the Goods manual say?",
            "manual_id": "goods-2021",
            "source_sha256": "sha-goods-2021",
            "source_chunk_ids": ["chunk-a"],
            "citations": [{"section": "Supply order rules"}],
        },
        {
            "record_id": "two",
            "split": "validation",
            "question": "What does the earlier Goods edition say?",
            "manual_id": "goods-2021",
            "source_sha256": "sha-goods-2021",
            "source_chunk_ids": ["chunk-b"],
            "citations": [{"section": "Supply order rules"}],
        },
        {
            "record_id": "three",
            "split": "train",
            "question": "cross-document unaffected",
            "source_documents": [
                {"manual_id": "left", "section": "x", "source_sha256": "left-sha"},
                {"manual_id": "right", "section": "y", "source_sha256": "right-sha"},
            ],
            "source_chunk_ids": ["left-1", "right-1"],
        },
    ]
    audit = leakage_audit(rows)
    assert not audit["passed"]
    assert audit["collisions"]["manual"] == [{"value": "goods-2021", "splits": ["train", "validation"]}]
    assert audit["collisions"]["section"] == [{"value": "goods-2021:Supply order rules", "splits": ["train", "validation"]}]
    assert audit["collisions"]["source_hash"] == [{"value": "sha-goods-2021", "splits": ["train", "validation"]}]
    assert audit["unique_values"]["manual"] == 3


def test_ablation_judge_reviews_only_complete_actual_trial_bundles() -> None:
    answer = {
        "record_id": "record-a",
        "question": "How do A and B apply?",
        "answer": "A and B apply.",
        "propositions": [
            {"proposition_id": "prop-a"},
            {"proposition_id": "prop-b"},
        ],
        "claims": [
            {
                "statement": "A",
                "evidence": [{"proposition_id": "prop-a", "quote": "A policy text."}],
            },
            {
                "statement": "B",
                "evidence": [{"proposition_id": "prop-b", "quote": "B policy text."}],
            },
        ],
    }
    trials = [
        {
            "record_id": "record-a",
            "variant": variant,
            "trial_output": {"answerable": variant == "full", "claims": []},
        }
        for variant in ("full", "source_a_only", "source_b_only")
    ]
    inputs = build_ablation_judge_inputs(
        [answer],
        trials,
        [{"record_id": "record-a", "passed": True}],
    )
    assert len(inputs) == 1
    assert set(inputs[0]["actual_trials"]) == {
        "full",
        "source_a_only",
        "source_b_only",
    }
    prompt = object.__new__(SourceAblationJudge).prompt(inputs[0])
    assert "ACTUAL OUTPUTS" in prompt


def _minimal_final_manifest_kwargs() -> dict:
    return {
        "run_id": "run-1",
        "status": "complete",
        "stats": {"records": 1},
        "manuals": [],
        "corpus_report": {},
        "selected_rows": [],
        "single_coverage": {},
        "cross_coverage": {},
        "drafting_stats": {},
        "duplicates": 0,
    }


def test_release_ready_requires_both_status_complete_and_human_review() -> None:
    incomplete_review = {"complete": False}
    complete_review = {"complete": True}
    assert generation_pipeline._release_ready("complete", complete_review) is True
    assert generation_pipeline._release_ready("complete", incomplete_review) is False
    assert generation_pipeline._release_ready("partial", complete_review) is False
    assert generation_pipeline._release_ready("partial", incomplete_review) is False
    assert generation_pipeline._release_ready("complete", {}) is False


def test_final_manifest_human_review_defaults_to_honest_placeholder() -> None:
    manifest = generation_pipeline._final_manifest(**_minimal_final_manifest_kwargs())
    assert manifest["human_review"] == {
        "required_accepted_records": 100,
        "required_rejected_records": 25,
        "reviewed_accepted_records": 0,
        "reviewed_rejected_records": 0,
        "complete": False,
        "note": "Human labels are external release evidence and are never inferred.",
    }


def test_final_manifest_reflects_real_review_data_when_supplied() -> None:
    manifest = generation_pipeline._final_manifest(
        **_minimal_final_manifest_kwargs(),
        human_review={
            "review_file": "reviews.jsonl",
            "required_accepted_records": 100,
            "required_rejected_records": 25,
            "reviewed_accepted_records": 100,
            "reviewed_rejected_records": 25,
            "complete": True,
            "issues": [],
            "note": "Human labels are external release evidence and are never inferred.",
        },
    )
    assert manifest["human_review"]["reviewed_accepted_records"] == 100
    assert manifest["human_review"]["reviewed_rejected_records"] == 25
    assert manifest["human_review"]["complete"] is True


def test_human_review_template_is_reproducible_and_never_self_certifies(
    tmp_path: Path,
) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "canonical.jsonl").write_text(
        json.dumps({"record_id": "record-a", "question": "Question?"}) + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    prepare_review(files_dir, first, accepted_count=1, rejected_count=0)
    prepare_review(files_dir, second, accepted_count=1, rejected_count=0)
    assert first.read_bytes() == second.read_bytes()
    result = validate_reviews(first)
    assert not result["passed"]
    assert result["frozen_evaluation_complete"] is False
    row = json.loads(first.read_text(encoding="utf-8"))
    assert set(row["dimensions"]) == set(REVIEW_DIMENSIONS)
    assert row["overall_accept"] is None


def _complete_review_row(record_id: str, disposition: str, accept: bool) -> dict:
    record = {"record_id": record_id, "question": "Question?"}
    return {
        "review_id": f"review-{record_id}",
        "record_id": record_id,
        "pipeline_disposition": disposition,
        "record_sha256": hashlib.sha256(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "record": record,
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-08-08T00:00:00Z",
        "dimensions": dict.fromkeys(REVIEW_DIMENSIONS, accept),
        "overall_accept": accept,
        "notes": "",
    }


def test_validate_reviews_enforces_the_rejected_sample_minimum(
    tmp_path: Path,
) -> None:
    reviews = tmp_path / "reviews.jsonl"
    rows = [
        _complete_review_row(f"accepted-{index}", "accepted", True)
        for index in range(100)
    ]
    reviews.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = validate_reviews(reviews)
    assert result["reviewed_accepted"] == 100
    assert result["reviewed_rejected"] == 0
    assert result["minimum_rejected_required"] == 25
    assert not result["passed"]

    rows.extend(
        _complete_review_row(f"rejected-{index}", "rejected", False)
        for index in range(25)
    )
    reviews.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = validate_reviews(reviews)
    assert result["reviewed_rejected"] == 25
    assert result["passed"]


def test_judge_threshold_is_selected_on_development_and_verified_on_holdout(
    tmp_path: Path,
) -> None:
    def review_row(record_id: str, score: int, accepted: bool) -> dict:
        judge = {
            feature: True
            for feature in (
                "supported",
                "relevant",
                "preserves_qualifications",
                "authority_correct",
                "reasoning_valid",
                "question_natural",
                "persona_relevant",
                "task_correct",
                "persona_correct",
                "answerability_correct",
            )
        }
        record = {"record_id": record_id, "judge": {**judge, "score": score}}
        record_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {
            "review_id": f"review-{record_id}",
            "record_id": record_id,
            "pipeline_disposition": "accepted" if score >= 4 else "rejected",
            "record_sha256": record_hash,
            "record": record,
            "reviewer_id": "reviewer-1",
            "reviewed_at": "2026-08-02T00:00:00Z",
            "dimensions": {name: accepted for name in REVIEW_DIMENSIONS},
            "overall_accept": accepted,
            "notes": "",
        }

    development = tmp_path / "development.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    development.write_text(
        "\n".join(
            json.dumps(review_row(f"dev-{index}", score, accepted)) for index, (score, accepted) in enumerate([(5, True), (4, True), (3, False), (2, False)])
        )
        + "\n",
        encoding="utf-8",
    )
    holdout.write_text(
        "\n".join(
            json.dumps(review_row(f"hold-{index}", score, accepted)) for index, (score, accepted) in enumerate([(5, True), (4, True), (3, False), (1, False)])
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = calibrate_judge(
        development,
        holdout,
        minimum_precision=1.0,
        minimum_records=4,
    )
    assert artifact["recommended_threshold"] == 4
    assert artifact["passed"] is True
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    loaded = load_judge_calibration(
        {
            "judge_calibration": {
                "path": str(artifact_path),
                "sha256": artifact_hash,
                "minimum_holdout_precision": 1.0,
                "minimum_records_per_split": 4,
            }
        },
        required=True,
    )
    assert loaded["recommended_threshold"] == 4
    assert loaded["holdout_metrics"]["precision"] == 1.0


def test_release_validation_requires_all_four_exports_and_human_review(
    tmp_path: Path,
) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "status": "complete",
                "terminal_request_completeness": {"complete": True},
                "required_task_type_counts": {
                    "qa": 1,
                    "qa_cot": 1,
                    "cross_document_qa": 1,
                    "cross_document_qa_cot": 1,
                },
                "quality_acceptance": {"portfolio_quality_complete": True},
                "stage_quality_evidence": {
                    "cross_document": {"required": True, "passed": True},
                    "drafting": {"required": True, "passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (files_dir / "leakage_audit.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    for index, filename in enumerate(
        (
            "qa_sft.jsonl",
            "qa_cot_sft.jsonl",
            "cross_document_qa_sft.jsonl",
            "cross_document_qa_cot_sft.jsonl",
        )
    ):
        (files_dir / filename).write_text(
            json.dumps({"record_id": f"train-{index}", "split": "train"}) + "\n",
            encoding="utf-8",
        )
    (files_dir / "eval.jsonl").write_text("", encoding="utf-8")
    (files_dir / "canonical.jsonl").write_text("{}\n" * 4, encoding="utf-8")
    report = validate_run(files_dir)
    assert not report["passed"]
    assert report["export_counts"] == {
        "qa": 1,
        "qa_cot": 1,
        "cross_document_qa": 1,
        "cross_document_qa_cot": 1,
    }
    assert report["issues"] == ["human_review_not_supplied"]
    manifest = json.loads((files_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["stage_quality_evidence"]["drafting"]["passed"] = False
    (files_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    failed_stage = validate_run(files_dir)
    assert "stage_quality_evidence_incomplete:drafting" in failed_stage["issues"]


def test_release_validation_surfaces_pre_cap_and_post_cap_opener_diversity(
    tmp_path: Path,
) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "status": "complete",
                "terminal_request_completeness": {"complete": True},
                "required_task_type_counts": {"qa": 1},
                "quality_acceptance": {"portfolio_quality_complete": True},
                "statistics": {
                    "question_opener_diversity": {
                        "unique_openers": 6,
                        "top_opener": "what is",
                        "top_opener_count": 12,
                        "top_opener_share": 0.2,
                    },
                },
                "question_opener_diversity_pre_cap": {
                    "unique_openers": 2,
                    "top_opener": "according to",
                    "top_opener_count": 480,
                    "top_opener_share": 0.8,
                    "pool_size": 600,
                    "cap_waste_ratio": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    (files_dir / "leakage_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    row = json.dumps({"record_id": "r1", "split": "train"}) + "\n"
    (files_dir / "qa_sft.jsonl").write_text(row, encoding="utf-8")
    (files_dir / "eval.jsonl").write_text("", encoding="utf-8")
    (files_dir / "canonical.jsonl").write_text("{}\n", encoding="utf-8")
    report = validate_run(files_dir)
    assert report["question_opener_diversity"]["post_cap"]["top_opener_share"] == 0.2
    assert report["question_opener_diversity"]["pre_cap"]["top_opener_share"] == 0.8
    assert report["question_opener_diversity"]["pre_cap"]["cap_waste_ratio"] == 0.5


def test_release_validation_defaults_opener_diversity_when_manifest_predates_it(
    tmp_path: Path,
) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "status": "complete",
                "terminal_request_completeness": {"complete": True},
                "required_task_type_counts": {"qa": 1},
                "quality_acceptance": {"portfolio_quality_complete": True},
            }
        ),
        encoding="utf-8",
    )
    (files_dir / "leakage_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    row = json.dumps({"record_id": "r1", "split": "train"}) + "\n"
    (files_dir / "qa_sft.jsonl").write_text(row, encoding="utf-8")
    (files_dir / "eval.jsonl").write_text("", encoding="utf-8")
    (files_dir / "canonical.jsonl").write_text("{}\n", encoding="utf-8")
    report = validate_run(files_dir)
    assert report["question_opener_diversity"] == {"post_cap": {}, "pre_cap": {}}


def test_release_validation_detects_train_eval_overlap(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "status": "complete",
                "terminal_request_completeness": {"complete": True},
                "required_task_type_counts": {"qa": 1},
                "quality_acceptance": {"portfolio_quality_complete": True},
            }
        ),
        encoding="utf-8",
    )
    (files_dir / "leakage_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    row = json.dumps({"record_id": "shared", "split": "train"}) + "\n"
    (files_dir / "qa_sft.jsonl").write_text(row, encoding="utf-8")
    (files_dir / "eval.jsonl").write_text(row, encoding="utf-8")
    (files_dir / "canonical.jsonl").write_text("{}\n" * 2, encoding="utf-8")
    report = validate_run(files_dir)
    assert "train_record_in_eval_export" in report["issues"]
    assert "training_eval_record_id_overlap" in report["issues"]


def test_failure_distribution_prefers_the_structured_error_category(
    tmp_path: Path,
) -> None:
    """T19: real failure causes, not a regex over prompt text that never matches."""
    stage_dir = tmp_path / "generation" / "fingerprint-a"
    stage_dir.mkdir(parents=True)
    rows = [
        {"model": "m", "messages": [], "error_category": "timeout"},
        {"model": "m", "messages": [], "error_category": "timeout"},
        {"model": "m", "messages": [], "error_category": "truncation"},
        # No error_category at all (older run captured before this field
        # existed): falls back to the legacy regex heuristic rather than
        # silently disappearing from the distribution.
        {"model": "m", "messages": ["describe the procurement policy"]},
    ]
    (stage_dir / "failed_requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    distribution = _failure_distribution(tmp_path, manifest=None)

    assert distribution == {"other": 1, "timeout": 2, "truncation": 1}


def _schema_validity_manifest() -> dict:
    return {
        "resume": {
            "stage_events": {
                "generation": {
                    "status": "executed",
                    "producer": {"stage_fingerprint": "fp-generation"},
                },
                "judge": {
                    "status": "executed",
                    "producer": {"stage_fingerprint": "fp-judge"},
                },
                # Not in SCHEMA_VALIDITY_STAGE_NAMES: its own schema failures
                # must never leak into the scoped rate below, since the
                # denominator does not include propositions' expected requests.
                "propositions": {
                    "status": "executed",
                    "producer": {"stage_fingerprint": "fp-propositions"},
                },
            },
        },
        "request_coverage": {
            "single_document": {
                "blueprinted": {"expected_requests": 10},
                "generated": {"expected_requests": 10},
                "judged": {"expected_requests": 8},
                "accepted": {"expected_requests": 10},
            },
            "cross_document": {
                "generated": {"expected_requests": 0},
                "judged": {"expected_requests": 0},
                "accepted": {"expected_requests": 0},
            },
        },
    }


def test_schema_validity_rate_scopes_failures_to_covered_stages(
    tmp_path: Path,
) -> None:
    manifest = _schema_validity_manifest()
    generation_dir = tmp_path / "generation" / "fp-generation"
    generation_dir.mkdir(parents=True)
    (generation_dir / "failed_requests.jsonl").write_text(
        "\n".join(
            json.dumps({"model": "m", "messages": [], "error_category": category})
            for category in ("schema_validation", "schema_validation", "timeout")
        )
        + "\n",
        encoding="utf-8",
    )
    judge_dir = tmp_path / "judge" / "fp-judge"
    judge_dir.mkdir(parents=True)
    (judge_dir / "failed_requests.jsonl").write_text(
        json.dumps({"model": "m", "messages": [], "error_category": "schema_validation"}) + "\n",
        encoding="utf-8",
    )
    # This stage's schema-validation failures are real, but out of the
    # scoped denominator's coverage and must not be counted.
    propositions_dir = tmp_path / "propositions" / "fp-propositions"
    propositions_dir.mkdir(parents=True)
    (propositions_dir / "failed_requests.jsonl").write_text(
        json.dumps({"model": "m", "messages": [], "error_category": "schema_validation"}) + "\n",
        encoding="utf-8",
    )

    result = _schema_validity_rate(tmp_path, manifest)

    assert result["computed"] is True
    assert result["expected_requests_by_stage"] == {
        "single_document_generation": 10,
        "single_document_judge": 8,
        "cross_document_generation": 0,
        "cross_document_judge": 0,
    }
    assert result["expected_requests_total"] == 18
    # 2 (generation) + 1 (judge) == 3; propositions' 1 is excluded.
    assert result["schema_validation_failures"] == 3
    assert result["schema_validity_rate"] == round(1 - 3 / 18, 4)


def test_schema_validity_rate_not_computed_without_working_dir() -> None:
    result = _schema_validity_rate(None, _schema_validity_manifest())
    assert result["computed"] is False
    assert result["schema_validity_rate"] is None
    assert "working-dir" in result["reason"]


def test_schema_validity_rate_not_computed_without_stage_events(
    tmp_path: Path,
) -> None:
    manifest = _schema_validity_manifest()
    manifest["resume"]["stage_events"] = {}
    # Even if a failed_requests.jsonl happens to exist somewhere in
    # working_dir, without stage_events there is no way to scope it to the
    # covered stages, so the rate must not be fabricated from an unscoped scan.
    stray_dir = tmp_path / "generation" / "fp-generation"
    stray_dir.mkdir(parents=True)
    (stray_dir / "failed_requests.jsonl").write_text(
        json.dumps({"model": "m", "messages": [], "error_category": "schema_validation"}) + "\n",
        encoding="utf-8",
    )

    result = _schema_validity_rate(tmp_path, manifest)

    assert result["computed"] is False
    assert result["schema_validity_rate"] is None
    assert "stage_events" in result["reason"]


def test_schema_validity_rate_surfaced_in_validate_run_report_without_working_dir(
    tmp_path: Path,
) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    manifest = {
        "run_id": "run",
        "status": "complete",
        "terminal_request_completeness": {"complete": True},
        "required_task_type_counts": {"qa": 1},
        "quality_acceptance": {"portfolio_quality_complete": True},
        **_schema_validity_manifest(),
    }
    (files_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (files_dir / "leakage_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    row = json.dumps({"record_id": "r1", "split": "train"}) + "\n"
    (files_dir / "qa_sft.jsonl").write_text(row, encoding="utf-8")
    (files_dir / "eval.jsonl").write_text("", encoding="utf-8")
    (files_dir / "canonical.jsonl").write_text("{}\n", encoding="utf-8")

    report = validate_run(files_dir)

    assert report["schema_validity"]["computed"] is False
    assert report["schema_validity"]["reason"] == "no --working-dir supplied"


def test_schema_validity_rate_surfaced_in_validate_run_report_with_working_dir(
    tmp_path: Path,
) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    working_dir = tmp_path / "working"
    manifest = {
        "run_id": "run",
        "status": "complete",
        "terminal_request_completeness": {"complete": True},
        "required_task_type_counts": {"qa": 1},
        "quality_acceptance": {"portfolio_quality_complete": True},
        **_schema_validity_manifest(),
    }
    (files_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (files_dir / "leakage_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    row = json.dumps({"record_id": "r1", "split": "train"}) + "\n"
    (files_dir / "qa_sft.jsonl").write_text(row, encoding="utf-8")
    (files_dir / "eval.jsonl").write_text("", encoding="utf-8")
    (files_dir / "canonical.jsonl").write_text("{}\n", encoding="utf-8")
    generation_dir = working_dir / "generation" / "fp-generation"
    generation_dir.mkdir(parents=True)
    (generation_dir / "failed_requests.jsonl").write_text(
        json.dumps({"model": "m", "messages": [], "error_category": "schema_validation"}) + "\n",
        encoding="utf-8",
    )

    report = validate_run(files_dir, working_dir)

    assert report["schema_validity"]["computed"] is True
    assert report["schema_validity"]["expected_requests_total"] == 18
    assert report["schema_validity"]["schema_validation_failures"] == 1
    assert report["schema_validity"]["schema_validity_rate"] == round(1 - 1 / 18, 4)


def test_run_layout_and_curator_cache_are_project_local(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generation_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generation_pipeline, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(generation_pipeline, "CACHE_ROOT", tmp_path / ".curator_working")
    fixed_time = datetime(2026, 7, 28, 15, 30, 12, 123456, tzinfo=timezone.utc)

    run_id, files_dir = generation_pipeline._run_layout(None, fixed_time)

    assert run_id == "run-20260728T153012-123456Z"
    assert files_dir == tmp_path / "outputs" / run_id / "files"
    assert files_dir.is_dir()


def test_run_layout_rejects_unsafe_or_existing_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generation_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generation_pipeline, "OUTPUT_ROOT", tmp_path / "outputs")

    for unsafe in ("../escape", "/absolute", "has spaces"):
        try:
            generation_pipeline._run_layout(unsafe)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"unsafe run ID should fail: {unsafe}")

    _, files_dir = generation_pipeline._run_layout("pilot-001")
    (files_dir / "existing.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        generation_pipeline._run_layout("pilot-001")
    except SystemExit as exc:
        assert "without a recognized manifest/run_state" in str(exc)
    else:
        raise AssertionError("non-empty run output should not be overwritten")

    (files_dir / "manifest.json").write_text(
        '{"run_id":"pilot-001","status":"failed"}\n',
        encoding="utf-8",
    )
    resumed_run_id, resumed_files = generation_pipeline._run_layout("pilot-001")
    assert resumed_run_id == "pilot-001"
    assert resumed_files == files_dir


def _resume_profile(prefix: str, profile_name: str) -> dict:
    return {
        "profile_name": profile_name,
        "served_model_env": f"{prefix}_MODEL",
        "base_url_env": f"{prefix}_BASE_URL",
        "api_key_env": f"{prefix}_API_KEY",
        "deployment_identity_env": f"{prefix}_DEPLOYMENT_ID",
        "structured_output_mode": "tools_auto",
        "generation_params": {"temperature": 1.0},
        "request_timeout": 10,
        "max_retries": 1,
        "max_concurrent_requests": 2,
        "max_requests_per_minute": 100,
        "max_tokens_per_minute": 1000,
    }


class _PromptStageLLMV1:
    """A stage LLM stub with a real, inspectable `prompt()`/`parse()` pair.

    Unlike `_FakeStageLLM` (a bare callable with no `prompt`/`parse` at all,
    used by the transport/model-identity resume tests), this fixture exists
    specifically to exercise T22's prompt/parse source hashing.
    """

    def __init__(self, rows: list[dict], calls: list[str], fail: bool = False):
        self.rows = rows
        self.calls = calls
        self.fail = fail

    def prompt(self, row: dict) -> str:
        return f"Answer using opener phrasing 'According to the manual': {row}"

    def parse(self, row: dict, response) -> dict:
        return {**row, "response": response}

    def __call__(self, dataset, working_dir: str):
        self.calls.append(working_dir)
        if self.fail:
            raise AssertionError("completed checkpoint should have been reused")
        return SimpleNamespace(
            dataset=SimpleNamespace(to_list=lambda: self.rows),
        )


class _PromptStageLLMV2(_PromptStageLLMV1):
    """Same class shape as `_PromptStageLLMV1`, but with edited prompt wording."""

    def prompt(self, row: dict) -> str:
        return f"Ask this as a specific persona voice, not a generic opener: {row}"


class _FakeStageLLM:
    def __init__(self, rows: list[dict], calls: list[str], fail: bool = False):
        self.rows = rows
        self.calls = calls
        self.fail = fail

    def __call__(self, dataset, working_dir: str):
        self.calls.append(working_dir)
        if self.fail:
            raise AssertionError("completed checkpoint should have been reused")
        return SimpleNamespace(
            dataset=SimpleNamespace(to_list=lambda: self.rows),
        )


def _resume_manager(
    tmp_path: Path,
    monkeypatch,
    *,
    generation_model: str,
    generation_url: str,
    generation_deployment: str,
    judge_model: str = "judge-a",
    generation_timeout: int = 10,
    config: dict | None = None,
    refresh_stages: set[str] | None = None,
) -> ResumeManager:
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir(exist_ok=True)
    (pipeline_dir / "stage.py").write_text("VERSION = 1\n", encoding="utf-8")
    for name, value in {
        "GEN_MODEL": generation_model,
        "GEN_BASE_URL": generation_url,
        "GEN_API_KEY": "secret-generation-key",
        "GEN_DEPLOYMENT_ID": generation_deployment,
        "JDG_MODEL": judge_model,
        "JDG_BASE_URL": "http://10.0.0.2:8000/v1",
        "JDG_API_KEY": "secret-judge-key",
        "JDG_DEPLOYMENT_ID": "judge-deployment-v1",
    }.items():
        monkeypatch.setenv(name, value)
    return ResumeManager(
        run_id="same-run",
        output_root=tmp_path / "outputs",
        cache_root=tmp_path / ".curator_working",
        config=config or {"quality": {"minimum": 4}},
        pipeline_dir=pipeline_dir,
        generation_profile={
            **_resume_profile("GEN", "generation"),
            "request_timeout": generation_timeout,
        },
        judge_profile=_resume_profile("JDG", "judge"),
        refresh_stages=refresh_stages,
    )


def test_completed_stage_survives_generation_and_judge_model_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    first.start()
    calls: list[str] = []
    expected = [{"record_id": "kept", "generation_model": "model-a"}]
    assert (
        first.execute_llm_stage(
            stage="generation",
            role="generation",
            llm=_FakeStageLLM(expected, calls),
            inputs=[{"planned_request_id": "one"}],
        )
        == expected
    )
    judged = [{"record_id": "kept", "judge_model": "judge-a"}]
    assert (
        first.execute_llm_stage(
            stage="judge",
            role="judge",
            llm=_FakeStageLLM(judged, calls),
            inputs=expected,
        )
        == judged
    )
    first.finish("partial")

    second = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="renamed-model-b",
        generation_url="http://10.0.0.9:9000/v1",
        generation_deployment="deployment-b",
        judge_model="judge-b",
    )
    second.start()
    reused = second.execute_llm_stage(
        stage="generation",
        role="generation",
        llm=_FakeStageLLM([], calls, fail=True),
        inputs=[{"planned_request_id": "one"}],
    )
    assert reused == expected
    reused_judgment = second.execute_llm_stage(
        stage="judge",
        role="judge",
        llm=_FakeStageLLM([], calls, fail=True),
        inputs=expected,
    )
    assert reused_judgment == judged
    assert len(calls) == 2
    assert second.summary()["stage_events"]["generation"]["status"] == ("reused_checkpoint")
    assert second.summary()["stage_events"]["judge"]["status"] == ("reused_checkpoint")


def test_transport_only_change_reuses_partial_cache_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://127.0.0.1:3011/v1",
        generation_deployment="same-deployment",
    )
    first.start()
    old_fingerprint = first._stage_fingerprint("generation", "generation")
    second = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://127.0.0.1:9999/v1",
        generation_deployment="same-deployment",
        generation_timeout=600,
    )
    second.start()
    assert second._stage_fingerprint("generation", "generation") == old_fingerprint


def test_transport_tuning_does_not_change_scientific_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://127.0.0.1:3011/v1",
        generation_deployment="same-deployment",
        config={
            "quality": {"minimum": 4},
            "models": {"generation": {"request_timeout": 1800}},
        },
    )
    second = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://127.0.0.1:3011/v1",
        generation_deployment="same-deployment",
        generation_timeout=600,
        config={
            "quality": {"minimum": 4},
            "models": {
                "generation": {
                    "request_timeout": 600,
                    "max_retries": 1,
                    "max_concurrent_requests": 64,
                    "output_rescue_max_tokens": 12000,
                    "output_rescue_max_concurrent_requests": 16,
                }
            },
        },
    )
    assert first._contract_hash("generation") == second._contract_hash("generation")
    assert first._stage_fingerprint("generation", "generation") == second._stage_fingerprint("generation", "generation")


def test_saturation_replay_can_reuse_integrity_checked_historical_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    first.start()
    calls: list[str] = []
    expected = [{"record_id": "historical-record"}]
    assert (
        first.execute_llm_stage(
            stage="cross_generation_pass_001",
            role="generation",
            llm=_FakeStageLLM(expected, calls),
            inputs=[{"planned_request_id": "one"}],
        )
        == expected
    )
    first.finish("partial")

    metadata_path = next((tmp_path / "outputs" / "same-run" / "checkpoints" / "cross_generation_pass_001").glob("*/metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["contract_sha256"] = "legacy-transport-inclusive-contract"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    replay = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    replay.start()
    reused = replay.execute_llm_stage(
        stage="cross_generation_pass_001",
        role="generation",
        llm=_FakeStageLLM([], calls, fail=True),
        inputs=[{"planned_request_id": "one"}],
        prefer_historical_checkpoint=True,
    )
    assert reused == expected
    assert replay.summary()["stage_events"]["cross_generation_pass_001"]["compatibility"] == "saturation_replay_candidate_historical_artifact"


def test_completed_stage_survives_pipeline_source_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    first.start()
    expected = [{"record_id": "immutable-result"}]
    calls: list[str] = []
    assert (
        first.execute_llm_stage(
            stage="generation",
            role="generation",
            llm=_FakeStageLLM(expected, calls),
            inputs=[{"planned_request_id": "one"}],
        )
        == expected
    )
    first.finish("partial")

    pipeline_source = tmp_path / "pipeline" / "stage.py"
    pipeline_source.write_text("VERSION = 2\n", encoding="utf-8")
    second = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    second.start()
    assert (
        second.execute_llm_stage(
            stage="generation",
            role="generation",
            llm=_FakeStageLLM([], calls, fail=True),
            inputs=[{"planned_request_id": "one"}],
        )
        == expected
    )
    assert len(calls) == 1
    assert second.summary()["stage_events"]["generation"]["compatibility"] == "current_contract"


def test_v1_completed_checkpoint_is_invalidated_after_resume_contract_upgrade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    manager.start()
    inputs = [{"planned_request_id": "one"}]
    input_hash = resume_module._canonical_hash(resume_module._checkpoint_input(inputs))
    legacy_dir = tmp_path / "outputs" / "same-run" / "checkpoints" / "generation" / "legacy-v1-key"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "records.jsonl").write_text(
        '{"record_id":"legacy"}\n',
        encoding="utf-8",
    )
    (legacy_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "nrl-resume-v1",
                "status": "complete",
                "stage": "generation",
                "input_sha256": input_hash,
                "contract_sha256": "old-source-dependent-contract",
                "producer": {"pipeline_source_sha256": "old-source"},
            }
        ),
        encoding="utf-8",
    )

    llm = _FakeStageLLM([{"record_id": "fresh"}], [])
    result = manager.execute_llm_stage(
        stage="generation",
        role="generation",
        llm=llm,
        inputs=inputs,
    )
    assert result == [{"record_id": "fresh"}]
    assert len(llm.calls) == 1
    assert manager.summary()["stage_events"]["generation"]["status"] == "executed"


def test_refresh_stage_preserves_checkpoint_history_and_redacts_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    first.start()
    first.execute_llm_stage(
        stage="generation",
        role="generation",
        llm=_FakeStageLLM([{"value": "old"}], []),
        inputs=[{"id": "one"}],
    )
    first.finish("partial")
    second = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-b",
        generation_url="http://10.0.0.8:8000/v1",
        generation_deployment="deployment-b",
        refresh_stages={"generation"},
    )
    second.start()
    assert second.execute_llm_stage(
        stage="generation",
        role="generation",
        llm=_FakeStageLLM([{"value": "new"}], []),
        inputs=[{"id": "one"}],
    ) == [{"value": "new"}]
    checkpoint_root = tmp_path / "outputs" / "same-run" / "checkpoints"
    assert list(checkpoint_root.rglob("history/*/records.jsonl"))
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "outputs" / "same-run").rglob("*.json"))
    assert "secret-generation-key" not in serialized
    assert "secret-judge-key" not in serialized


def test_contract_hash_changes_when_llm_prompt_source_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Unit-level check: T22's prompt/parse source hashing actually differs."""
    manager = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    same_class_hash_a = manager._contract_hash("generation", _PromptStageLLMV1([], []))
    same_class_hash_b = manager._contract_hash("generation", _PromptStageLLMV1([], []))
    changed_prompt_hash = manager._contract_hash("generation", _PromptStageLLMV2([], []))
    no_llm_hash = manager._contract_hash("generation")

    assert same_class_hash_a == same_class_hash_b
    assert same_class_hash_a != changed_prompt_hash
    assert same_class_hash_a != no_llm_hash


def test_prompt_source_change_invalidates_completed_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Reproduces Critical Issue #9: a prompt-wording-only edit must not
    silently reuse a completed checkpoint produced under the old wording."""
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    first.start()
    calls: list[str] = []
    v1_result = [{"record_id": "one", "wording": "v1"}]
    assert (
        first.execute_llm_stage(
            stage="generation",
            role="generation",
            llm=_PromptStageLLMV1(v1_result, calls),
            inputs=[{"planned_request_id": "one"}],
        )
        == v1_result
    )
    first.finish("partial")

    # Same run, same config, same model identity, same STAGE_CONTRACT_VERSIONS
    # entry — only the prompt() wording differs (the scenario a developer would
    # forget to bump a manual version number for).
    second = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    second.start()
    v2_result = [{"record_id": "one", "wording": "v2"}]
    result = second.execute_llm_stage(
        stage="generation",
        role="generation",
        llm=_PromptStageLLMV2(v2_result, calls),
        inputs=[{"planned_request_id": "one"}],
    )

    assert result == v2_result
    assert len(calls) == 2
    assert second.summary()["stage_events"]["generation"]["status"] == "executed"


def test_unchanged_prompt_source_still_reuses_completed_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Companion to the invalidation test: an unrelated rerun of the exact
    same prompt()/parse() source must still hit the completed checkpoint."""
    first = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    first.start()
    calls: list[str] = []
    expected = [{"record_id": "one", "wording": "v1"}]
    assert (
        first.execute_llm_stage(
            stage="generation",
            role="generation",
            llm=_PromptStageLLMV1(expected, calls),
            inputs=[{"planned_request_id": "one"}],
        )
        == expected
    )
    first.finish("partial")

    second = _resume_manager(
        tmp_path,
        monkeypatch,
        generation_model="model-a",
        generation_url="http://10.0.0.1:8000/v1",
        generation_deployment="deployment-a",
    )
    second.start()
    reused = second.execute_llm_stage(
        stage="generation",
        role="generation",
        llm=_PromptStageLLMV1([], calls, fail=True),
        inputs=[{"planned_request_id": "one"}],
    )

    assert reused == expected
    assert len(calls) == 1
    assert second.summary()["stage_events"]["generation"]["status"] == "reused_checkpoint"
    assert second.summary()["stage_events"]["generation"]["compatibility"] == "current_contract"
