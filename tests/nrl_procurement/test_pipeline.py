"""Focused tests for the local procurement pipeline."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

import generate as generation_pipeline  # noqa: E402
from corpus import (  # noqa: E402
    corpus_quality_report,
    generation_text,
    load_corpus,
    representative_rows,
)
from cross_document import build_bundles  # noqa: E402
from cross_stage import (  # noqa: E402
    CrossDocumentGenerator,
    CrossDocumentJudge,
    SingularCrossDocumentJudge,
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
)
from export import assert_unique_record_ids, assign_splits, export_records  # noqa: E402
from generate import (  # noqa: E402
    ProcurementGenerator,
    ProcurementJudge,
    SingularProcurementJudge,
    _budget_judge_rows,
    _judge_prompt_budget,
    _singular_judge_batch_size,
    plan_cross_document_requests,
    plan_single_document_requests,
    request_coverage,
)
from path_qa import (  # noqa: E402
    SourceAblationAnswerGenerator,
    ablation_trial_validation_issues,
    adjudicate_ablation_trials,
    answer_validation_issues,
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
from reasoning_paths import build_reasoning_paths, validate_reasoning_path  # noqa: E402
from schemas import (  # noqa: E402
    CandidateBatch,
    CrossCandidateBatch,
    CrossJudgeBatch,
    CrossJudgedCandidate,
    DraftingBlock,
    DraftingResult,
    JudgeBatch,
    JudgedCandidate,
    PropositionBatch,
)
from source_windows import (  # noqa: E402
    build_source_windows,
    resolve_component_references,
)
from validation import (  # noqa: E402
    deduplicate,
    judge_batch_identity_issues,
    judge_quotes_are_grounded,
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
    assert (
        "judgments"
        not in SingularProcurementJudge.response_format.model_json_schema()[
            "properties"
        ]
    )
    assert (
        "judgments"
        not in SingularCrossDocumentJudge.response_format.model_json_schema()[
            "properties"
        ]
    )
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
    row = {
        "judge_items": [
            {"record_id": record["record_id"], "record": record}
            for record in records
        ]
    }
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
    assert all(
        "duplicate_judge_record_ids:record-a" in record["judge"]["issues"]
        for record in judged
    )
    assert all(
        "missing_judge_record_ids:record-b" in record["judge"]["issues"]
        for record in judged
    )


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
    prompts = [
        SourceAblationAnswerGenerator.prompt(SimpleNamespace(), trial)
        for trial in trials
    ]
    assert all("Canonical claim" not in prompt for prompt in prompts)
    assert all("source_a_only" not in prompt and "source_b_only" not in prompt for prompt in prompts)


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
                "claims": (
                    [
                        {
                            "evidence": [
                                {"proposition_id": proposition_id}
                                for proposition_id in proposition_ids
                            ]
                        }
                    ]
                    if proposition_ids
                    else []
                ),
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
            {"statement": "Apply the stated rule.", "evidence_quotes": [passage]},
            {"statement": "Apply the stated rule.", "evidence_quotes": [passage]},
        ],
    }
    issues = validate_record(record, passage)
    assert "cot_repeats_reasoning_step" in issues
    assert "cot_reuses_identical_evidence_for_all_steps" in issues


def test_empty_generation_materializes_terminal_lineage() -> None:
    row = {
        "planned_request_id": "single-request",
        "planned_task_type": "qa",
    }
    result = ProcurementGenerator.parse(
        SimpleNamespace(model_name="generator"),
        row,
        CandidateBatch(examples=[]),
    )
    assert result == [
        {
            "parent_request_id": "single-request",
            "planned_task_type": "qa",
            "terminal_state": "empty_generation",
            "generation_model": "generator",
            "deterministic_checks": {
                "passed": False,
                "issues": ["generator_returned_no_examples"],
            },
        }
    ]


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
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert captured["payload"]["tools"][0]["function"]["name"] == "Result"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 30.0


def test_model_context_window_is_explicit_and_profile_local() -> None:
    nemotron = generation_pipeline.CONFIG["model_profiles"]["nemotron"]
    source_windows = generation_pipeline.CONFIG["source_windows"]

    assert configured_context_window(nemotron) == 131072
    assert nemotron["structured_output_mode"] == "tools_auto"
    assert (
        generation_pipeline.CONFIG["model_profiles"]["glm"][
            "structured_output_mode"
        ]
        == "json_schema"
    )
    assert (
        generation_pipeline.CONFIG["model_profiles"]["gemma"][
            "structured_output_mode"
        ]
        == "json_schema"
    )
    assert source_windows["max_input_tokens"] == 8192
    for invalid in ({}, {"context_window": 0}, {"context_window": True}):
        try:
            configured_context_window(invalid)
        except ValueError as exc:
            assert "positive context_window" in str(exc)
        else:
            raise AssertionError(
                "missing or invalid model context must fail closed"
            )


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
    assert rejected[0]["judge"]["issues"] == [
        "judge_prompt_exceeds_context_window"
    ]
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
    assert all(result["complete"] is False for result in path["structural_ablation"].values())


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
    assert semantic_support_issues(
        "There is no provision for consortium registration.",
        "There is no provision for registration of Consortium.",
    ) == []
    assert semantic_support_issues(
        "The supplier must deliver the goods.",
        "The supplier shall deliver the goods.",
    ) == []


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
                    "instruction_evidence_quotes": [
                        "liquidated damages clause"
                    ],
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
    assert (
        "block_1:strengthened_modality:permission_to_obligation"
        in issues
    )


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
                "statement": "Both editions cap damages.",
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
        "answer": (
            "NRL may cancel the order, and the supplier shall replace rejected "
            "goods."
        ),
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
        "answer": (
            "The buyer may cancel the bid, and the supplier shall replace "
            "rejected goods."
        ),
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
    assert (
        "claim_strengthened_modality:permission_to_obligation"
        in validate_cross_record(record, documents)
    )


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
    manuals = [
        {"manual_id": "base"},
        {"manual_id": "amendment", "amends": ["base"]},
        {"manual_id": "other"},
    ]
    assign_splits(records[:2], manuals, 0.8, 0.1, "test")
    assert records[0]["split"] == records[1]["split"]


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
                "statement": "Identify the tender requirement.",
                "evidence": [{"source_id": "source_a", "quote": left_quote}],
            },
            {
                "statement": "Combine it with the submission method.",
                "evidence": [{"source_id": "source_b", "quote": right_quote}],
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
        "planned_answerable": True,
    }
    single_response = CandidateBatch.model_validate(
        {
            "examples": [
                {
                    "task_type": "qa",
                    "task": "compliance_and_audit",
                    "persona": "auditor",
                    "question_type": "threshold",
                    "question": "How long must the record be retained?",
                    "answer": "The record must be retained for 10 years.",
                    "answerable": True,
                    "evidence": [{"quote": passage}],
                    "reasoning_steps": [],
                }
            ]
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
                            "evidence": [{"source_id": "source_b", "quote": passage}],
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
    assert cross[0]["deterministic_checks"] == {
        "passed": False,
        "issues": ["planned_task_type_mismatch:cross_document_qa"],
    }


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
                instruction_evidence_quotes=[
                    "Draft the delayed-delivery clause."
                ],
            ),
            DraftingBlock(
                text="LD is 0.5% per week and capped at 5% of delayed goods.",
                manual_evidence_quotes=[
                    "LD is 0.5% per week and capped at 5% of delayed goods."
                ],
            ),
            DraftingBlock(
                text="Tender mode: Limited.",
                tender_facts_used=["Tender mode: Limited."],
            ),
        ],
        manual_evidence_quotes=["LD is 0.5% per week and capped at 5% of delayed goods."],
        tender_facts_used=["Tender mode: Limited."],
    )
    assert drafting_validation_issues(inputs[0], result) == []
    compact = compact_drafting(
        {
            **inputs[0],
            "citations": ["chunk-1", "tender-1"],
            "context": [*inputs[0]["tender_context"], *result.manual_evidence_quotes],
            "response": "\n\n".join(block.text for block in result.document_blocks),
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
                    "citation_id": "tender-1",
                    "source_type": "tender_seed",
                    "tender_id": "tender-1",
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
        "citation_details",
        "citations",
    ]
    assert compact["citations"] == ["chunk-1", "tender-1"]
    assert inputs[0]["candidate_citation_ids"] == ["chunk-1", "tender-1"]


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
            "citation_id": "tender-1",
            "source_type": "tender_seed",
            "tender_id": "tender-1",
        },
    ]
    assert (
        drafting_citation_integrity_issues(
            ["chunk-used", "tender-1"],
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
        "planned_answerable": True,
    }
    prompt = ProcurementGenerator.prompt(None, row)
    for required in (
        "TASK",
        "PLANNED CONTRACT",
        "underlying procurement work",
        "nit_filling",
        "persona",
        "SOURCE POLICY",
        "CONSTRAINTS",
        "OUTPUT CONTRACT",
        "FINAL CHECK",
        "qa_cot",
        "two to four",
        "private hidden chain-of-thought",
        "evidence_quotes",
        "Not answerable from the provided sources.",
        "Government",
        "guidance as NRL policy",
        "---BEGIN UNTRUSTED SOURCE PASSAGE---",
        "---END UNTRUSTED SOURCE PASSAGE---",
    ):
        assert required in prompt

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
                "reasoning_steps": ([{"statement": "Apply the stated rule."}] if task_type.endswith("_cot") else []),
                "evidence": [],
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
        assert list(json.loads(line))[-1] == "citations"


def test_run_layout_and_curator_cache_are_project_local(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generation_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generation_pipeline, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(generation_pipeline, "CACHE_ROOT", tmp_path / ".curator_working")
    fixed_time = datetime(2026, 7, 28, 15, 30, 12, 123456, tzinfo=timezone.utc)

    run_id, files_dir = generation_pipeline._run_layout(None, fixed_time)

    assert run_id == "run-20260728T153012-123456Z"
    assert files_dir == tmp_path / "outputs" / run_id / "files"
    assert files_dir.is_dir()
    assert generation_pipeline._working_dir(run_id, "generation") == str(tmp_path / ".curator_working" / run_id / "generation")


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
        assert "already exists and is not empty" in str(exc)
    else:
        raise AssertionError("non-empty run output should not be overwritten")
