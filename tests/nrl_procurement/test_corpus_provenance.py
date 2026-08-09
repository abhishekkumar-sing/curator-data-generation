"""Source registry, OCR lineage, and pilot coverage regressions."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from corpus import (  # noqa: E402
    _ocr_provenance,
    generation_text,
    image_artifact_count,
    load_corpus,
    selection_coverage_report,
    source_quality_issues,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manual(manual_id: str, file: str) -> dict:
    return {
        "manual_id": manual_id,
        "title": manual_id,
        "file": file,
        "revision_date": "2026",
        "as_of_date": "2026",
    }


def test_duplicate_registered_source_content_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    text = "# Policy\n\n" + "A supported procurement requirement. " * 20
    (source / "a.md").write_text(text, encoding="utf-8")
    (source / "b.md").write_text(text, encoding="utf-8")
    (source / "manuals.yaml").write_text(
        yaml.safe_dump(
            {"manuals": [_manual("a", "a.md"), _manual("b", "b.md")]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate registered source content"):
        load_corpus(source, tmp_path / "ocr")


def test_registered_hash_pin_is_enforced(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.md").write_text("Policy text. " * 30, encoding="utf-8")
    manual = _manual("a", "a.md")
    manual["official_verification"] = {"source_sha256": "0" * 64}
    (source / "manuals.yaml").write_text(
        yaml.safe_dump({"manuals": [manual]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_corpus(source, tmp_path / "ocr")


def test_ocr_v2_requires_revisions_and_matching_page_counts(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    markdown = tmp_path / "manual.md"
    source.write_bytes(b"pdf")
    markdown.write_text("page one", encoding="utf-8")
    cache_dir = tmp_path / source.stem
    cache_dir.mkdir()
    payload = {
        "contract_version": "nrl-ocr-provenance-v2",
        "source_sha256": _sha(source),
        "markdown_sha256": _sha(markdown),
        "model": "chandra",
        "model_revision": "revision-1",
        "engine": "chandra",
        "chandra_ocr_version": "0.2.0",
        "package_revision": "package-revision-1",
        "source_page_count": 1,
        "markdown_page_count": 1,
    }
    (cache_dir / ".chandra-cache.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert _ocr_provenance(tmp_path, source, markdown)["status"] == "complete"
    payload["markdown_page_count"] = 2
    (cache_dir / ".chandra-cache.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Incomplete OCR output"):
        _ocr_provenance(tmp_path, source, markdown)


def test_image_caption_line_is_removed_and_counted() -> None:
    passage = (
        "Policy before.\n"
        "![Refinery](image.webp)A generated caption about the refinery.\n"
        "Policy after."
    )
    assert image_artifact_count(passage) == 1
    cleaned = generation_text(passage)
    assert "generated caption" not in cleaned
    assert cleaned == "Policy before.\n\nPolicy after."


def test_registered_corpus_gives_every_manual_at_least_one_eligible_chunk() -> None:
    """Regression test for the front-matter heuristic zeroing out short manuals.

    Every manual registered in data/source/manuals.yaml must contribute at
    least one chunk with no source_quality_issues, otherwise it can never
    produce a single-document QA/drafting record.
    """
    source_dir = REPO_ROOT / "data" / "source"
    ocr_dir = REPO_ROOT / "data" / "interim" / "ocr"
    rows, manuals = load_corpus(source_dir, ocr_dir)
    eligible_manual_ids = {
        row["manual_id"] for row in rows if not source_quality_issues(row)
    }
    registered_manual_ids = {manual["manual_id"] for manual in manuals}
    missing = sorted(registered_manual_ids - eligible_manual_ids)
    assert not missing, (
        "Manuals with zero eligible (answer-bearing) chunks: "
        f"{missing} -- every registered manual must contribute at least one "
        "chunk that source_quality_issues does not reject."
    )


def test_selection_coverage_reports_strata() -> None:
    rows = [
        {
            "manual_id": "a",
            "source_category": "government_manual",
            "content_class": "policy",
            "generation_passage": "Policy text. " * 30,
        },
        {
            "manual_id": "b",
            "source_category": "company_manual",
            "content_class": "table",
            "generation_passage": "Policy text. " * 30,
        },
    ]
    report = selection_coverage_report(rows, rows[:1])
    assert report["strategy"] == "deterministic_weighted_round_robin"
    assert report["dimensions"]["manual"]["coverage"] == 0.5
