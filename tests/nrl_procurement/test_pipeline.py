"""Focused tests for the local procurement pipeline."""

import sys
from datetime import datetime, timezone
from pathlib import Path

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
from cross_stage import CrossDocumentGenerator, CrossDocumentJudge  # noqa: E402
from drafting import (  # noqa: E402
    TenderDraftingGenerator,
    TenderDraftingJudge,
    build_drafting_inputs,
    compact_drafting,
    drafting_validation_issues,
    normalize_drafting_response,
    read_drafting_seeds,
)
from export import assign_splits, export_records  # noqa: E402
from generate import (  # noqa: E402
    ProcurementGenerator,
    ProcurementJudge,
    plan_cross_document_requests,
    plan_single_document_requests,
    request_coverage,
)
from schemas import DraftingResult  # noqa: E402
from validation import deduplicate, validate_cross_record, validate_record  # noqa: E402


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
        response=("Delayed Delivery & Liquidated Damages\n" "LD is 0.5% per week and capped at 5% of delayed goods."),
        manual_evidence_quotes=["LD is 0.5% per week and capped at 5% of delayed goods."],
        tender_facts_used=["Tender mode: Limited."],
    )
    assert drafting_validation_issues(inputs[0], result) == []
    compact = compact_drafting(
        {
            **inputs[0],
            "context": [*inputs[0]["tender_context"], *result.manual_evidence_quotes],
            "response": result.response,
        }
    )
    assert list(compact) == [
        "id",
        "tender_id",
        "task",
        "instruction",
        "context",
        "response",
        "citations",
    ]
    assert compact["citations"] == ["chunk-1", "tender-1"]


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
        response="The cap is 10%. Contact invented@example.com.",
        manual_evidence_quotes=["The cap is 5%."],
        tender_facts_used=["Tender mode: Limited."],
    )
    issues = drafting_validation_issues(row, result)
    assert "unsupported_number:10%" in issues
    assert "unsupported_email:invented@example.com" in issues


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
        "newline characters",
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
    monkeypatch.setitem(generation_pipeline.QUALITY, "unanswerable_fraction", 0.2)
    rows = [
        {
            "chunk_id": f"chunk-{index}",
            "generation_passage": ("The buyer shall act if the stated condition applies. " f"Rule {index}. " * 8),
        }
        for index in range(5)
    ]
    planned = plan_single_document_requests(rows, "seed")
    assert sum(row["planned_task_type"] == "qa_cot" for row in planned) == 2
    assert sum(not row["planned_answerable"] for row in planned) == 1
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
        response=("1. Scope\nTender ID: NRL-GOODS-CRANE-1009379-V2.\n" "Organization: NUMALIGARH REFINERY LIMITED.\nThe cap is 5%."),
        manual_evidence_quotes=["The cap is 5%."],
        tender_facts_used=["Tender ID: NRL-GOODS-CRANE-1009379-V2."],
    )
    assert drafting_validation_issues(row, valid) == []
    invalid = valid.model_copy(update={"response": ("<b>Scope</b>\nOrganization: Invented Division.\nThe cap is 10%.")})
    issues = drafting_validation_issues(row, invalid)
    assert "draft_contains_html_markup" in issues
    assert "unsupported_authority:Invented Division" in issues
    assert "unsupported_number:10%" in issues


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
