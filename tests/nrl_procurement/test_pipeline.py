"""Focused tests for the local procurement pipeline."""

import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from corpus import load_corpus  # noqa: E402
from cross_document import build_bundles  # noqa: E402
from export import assign_splits  # noqa: E402
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
    assert "unsupported_number:10 years" in validate_record(
        record, "The buyer shall retain it for 5 years."
    )


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
    assert "misattributed_or_non_verbatim_evidence" in validate_cross_record(
        record, bundle["source_documents"]
    )
