"""Manifest-aware, page-preserving procurement corpus loading."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PAGE_PATTERN = re.compile(r"<!--\s*Page\s+(\d+)\s*-->", re.IGNORECASE)
CHANDRA_PAGE_PATTERN = re.compile(r"(?m)^\s*(\d+)-{20,}\s*$")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
IMAGE_LINE_PATTERN = re.compile(r"(?m)^\s*!\[[^\]]*]\([^)]+\).*$")
HTML_IMAGE_PATTERN = re.compile(r"(?is)<(?:img|figure)\b[^>]*>.*?</figure>|<img\b[^>]*>")
TOC_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:\|\s*)?(?:\d+(?:\.\d+)*\s+)?[^\n|]{3,100}?"
    r"(?:\.{3,}|\|\s*)\s*\d{1,4}\s*(?:\|)?\s*$"
)
ABBREVIATION_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:\|\s*)?(?:<b>)?[A-Z][A-Z&/.-]{1,14}(?:</b>)?\s*"
    r"(?:[-:|]|\s{2,})\s*[A-Za-z][^\n]{1,100}$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ocr_markdown(ocr_dir: Path, source: Path) -> Path:
    matches = sorted(path for path in ocr_dir.rglob("*.md") if path.stem == source.stem or path.parent.name == source.stem)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Chandra Markdown output for {source.name} under " f"{ocr_dir}; found {len(matches)}. Run preprocess_pdfs.py first."
        )
    return matches[0]


def _ocr_provenance(ocr_dir: Path, source: Path, markdown: Path) -> dict[str, Any]:
    cache = ocr_dir / source.stem / ".chandra-cache.json"
    if not cache.is_file():
        return {"status": "missing", "cache_file": str(cache)}
    payload = json.loads(cache.read_text(encoding="utf-8"))
    if payload.get("source_sha256") != _sha256(source):
        raise ValueError(f"Stale OCR cache for {source.name}: source hash mismatch")
    markdown_hash = payload.get("markdown_sha256")
    if markdown_hash and markdown_hash != _sha256(markdown):
        raise ValueError(f"Stale OCR cache for {source.name}: Markdown hash mismatch")
    source_pages = payload.get("source_page_count")
    markdown_pages = payload.get("markdown_page_count")
    if source_pages and markdown_pages and int(source_pages) != int(markdown_pages):
        raise ValueError(
            f"Incomplete OCR output for {source.name}: source has {source_pages} "
            f"pages but Markdown records {markdown_pages}"
        )
    required = (
        "contract_version",
        "model",
        "model_revision",
        "chandra_ocr_version",
        "package_revision",
        "source_page_count",
        "markdown_page_count",
        "markdown_sha256",
    )
    return {
        "status": (
            "complete"
            if payload.get("contract_version") == "nrl-ocr-provenance-v2"
            and all(payload.get(field) for field in required)
            else "legacy"
        ),
        "cache_file": str(cache),
        "contract_version": payload.get("contract_version"),
        "model": payload.get("model"),
        "model_revision": payload.get("model_revision"),
        "engine": payload.get("engine"),
        "chandra_ocr_version": payload.get("chandra_ocr_version"),
        "package_revision": payload.get("package_revision"),
        "generated_at": payload.get("generated_at"),
        "source_page_count": source_pages,
        "markdown_page_count": markdown_pages,
        "markdown_sha256": markdown_hash,
    }


def _pages(text: str) -> list[tuple[int | None, str]]:
    markers = list(PAGE_PATTERN.finditer(text))
    if not markers:
        # Chandra's official --paginate_output format places a zero-based page
        # index before every page after the first.
        chandra_markers = list(CHANDRA_PAGE_PATTERN.finditer(text))
        if not chandra_markers:
            return [(None, text.strip())]
        pages = [(1, text[: chandra_markers[0].start()].strip())]
        pages.extend(
            (
                int(marker.group(1)) + 1,
                text[marker.end() : chandra_markers[index + 1].start() if index + 1 < len(chandra_markers) else len(text)].strip(),
            )
            for index, marker in enumerate(chandra_markers)
        )
        return pages
    return [
        (
            int(marker.group(1)),
            text[marker.end() : markers[index + 1].start() if index + 1 < len(markers) else len(text)].strip(),
        )
        for index, marker in enumerate(markers)
    ]


def _chunks(text: str, maximum: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    result: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        if current and length + len(paragraph) + 2 > maximum:
            result.append("\n\n".join(current))
            current, length = [], 0
        current.append(paragraph)
        length += len(paragraph) + 2
    if current:
        result.append("\n\n".join(current))
    return result


def _updated_heading_stack(
    stack: list[str],
    passage: str,
) -> list[str]:
    """Apply explicit Markdown headings and return the current breadcrumb."""
    updated = list(stack)
    for marks, title in HEADING_PATTERN.findall(passage):
        level = len(marks)
        updated = updated[: level - 1]
        updated.append(title.strip())
    return updated


def generation_text(passage: str) -> str:
    """Remove non-policy image descriptions while preserving tables and prose."""
    value = IMAGE_LINE_PATTERN.sub("", passage)
    value = HTML_IMAGE_PATTERN.sub("", value)
    value = re.sub(r"\n[ \t]+\n", "\n\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def image_artifact_count(passage: str) -> int:
    """Count model-generated image/figure blocks removed from generation text."""
    return len(IMAGE_LINE_PATTERN.findall(passage)) + len(
        HTML_IMAGE_PATTERN.findall(passage)
    )


def _document_family(manual_id: str) -> str:
    for family in ("goods", "works", "services", "consultancy"):
        if family in manual_id:
            return family
    return "other"


# Inclusive width (in pages) of the position-based front-matter detection
# window starting at a manual's configured start_page.
FRONT_MATTER_WINDOW_PAGES = 8


def _content_class(row: dict[str, Any]) -> str:
    passage = row["generation_passage"]
    section = str(row.get("section") or "").casefold()
    page = int(row.get("page") or 1)
    start_page = int(row.get("start_page", 1))
    manual_page_count = row.get("manual_page_count")
    # Introductory policy chapters later in a manual are substantive. Only
    # position-independent labels that are unambiguously apparatus belong here;
    # the early-page rule already covers a manual's opening introduction.
    #
    # The position rule assumes a manual has real content beyond its opening
    # window (cover/Foreword/Disclaimer/Table of Contents, tuned against
    # goods_2017's ~270 pages). A manual whose entire loaded content fits
    # inside that window -- e.g. a one- or two-page OM/amendment/corrigendum
    # notice -- has no front matter to skip; applying the rule there would
    # wrongly classify 100% of a substantive policy letter as front matter
    # and leave the manual with zero eligible chunks. Only apply the position
    # rule once we know the manual actually extends past the window (or when
    # the caller hasn't supplied a page count, to preserve prior behavior).
    positional_front_matter = page <= start_page + 7 and (
        manual_page_count is None
        or manual_page_count > FRONT_MATTER_WINDOW_PAGES
    )
    if positional_front_matter or any(
        label in section for label in ("foreword", "preface", "contents")
    ):
        return "front_matter"
    if "<table" in passage.casefold():
        return "table"
    return "policy"


def source_quality_issues(row: dict[str, Any]) -> list[str]:
    """Identify high-confidence non-answer-bearing source chunks.

    The classifier intentionally fails open for mixed prose/tables. It removes
    only structures that repeatedly produce retrieval failures: front matter,
    contents listings, abbreviation-only pages, and effectively blank forms.
    """
    passage = str(row.get("generation_passage") or "").strip()
    issues: list[str] = []
    if len(passage) < 200:
        issues.append("source_too_short")
    if row.get("content_class") == "front_matter":
        issues.append("front_matter")

    nonempty_lines = [line.strip() for line in passage.splitlines() if line.strip()]
    toc_lines = TOC_LINE_PATTERN.findall(passage)
    abbreviation_lines = ABBREVIATION_LINE_PATTERN.findall(passage)
    prose_sentences = re.findall(r"[A-Z][^.!?\n]{25,}[.!?](?:\s|$)", passage)
    normalized_prefix = re.sub(r"<[^>]+>|\s+", " ", passage[:800]).casefold()
    if "table of contents" in normalized_prefix and (
        passage.casefold().count("<tr") >= 4 or not prose_sentences
    ):
        issues.append("table_of_contents_only")
    if len(toc_lines) >= 5 and len(toc_lines) / max(len(nonempty_lines), 1) >= 0.45:
        issues.append("table_of_contents_only")
    if (
        len(abbreviation_lines) >= 6
        and len(abbreviation_lines) / max(len(nonempty_lines), 1) >= 0.55
        and not prose_sentences
    ):
        issues.append("abbreviation_glossary_only")

    table_cells = re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", passage)
    empty_cells = sum(not re.sub(r"<[^>]+>|\s|&nbsp;", "", cell) for cell in table_cells)
    if len(table_cells) >= 8 and empty_cells / len(table_cells) >= 0.6 and not prose_sentences:
        issues.append("blank_form_or_table")
    return sorted(set(issues))


def representative_rows(rows: list[dict[str, Any]], limit: int | None, seed: str) -> list[dict[str, Any]]:
    """Select a deterministic diversity-first pilot instead of a corpus prefix."""
    eligible = [row for row in rows if not source_quality_issues(row)]
    if limit is None or limit >= len(eligible):
        return eligible
    if limit < 1:
        return []

    maximum_page = {manual_id: max(int(row.get("page") or 1) for row in eligible) for manual_id in {str(row["manual_id"]) for row in eligible}}
    remaining = sorted(
        eligible,
        key=lambda row: hashlib.sha256(f"{seed}:{row['chunk_id']}".encode()).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    covered: dict[str, set[str | int]] = {
        "source_category": set(),
        "family": set(),
        "manual": set(),
        "content_class": set(),
        "page_band": set(),
    }
    while remaining and len(selected) < limit:

        def score(row: dict[str, Any]) -> tuple[int, str]:
            page = int(row.get("page") or 1)
            maximum = maximum_page[str(row["manual_id"])]
            dimensions: dict[str, str | int] = {
                "source_category": str(row["source_category"]),
                "family": _document_family(str(row["manual_id"])),
                "manual": str(row["manual_id"]),
                "content_class": str(row["content_class"]),
                "page_band": min(3, (page - 1) * 4 // max(maximum, 1)),
            }
            value = sum(
                weight
                for name, weight in (
                    ("source_category", 100),
                    ("family", 40),
                    ("manual", 20),
                    ("content_class", 8),
                    ("page_band", 4),
                )
                if dimensions[name] not in covered[name]
            )
            tie = hashlib.sha256(f"{seed}:{row['chunk_id']}".encode()).hexdigest()
            return value, tie

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        page = int(chosen.get("page") or 1)
        maximum = maximum_page[str(chosen["manual_id"])]
        covered["source_category"].add(str(chosen["source_category"]))
        covered["family"].add(_document_family(str(chosen["manual_id"])))
        covered["manual"].add(str(chosen["manual_id"]))
        covered["content_class"].add(str(chosen["content_class"]))
        covered["page_band"].add(min(3, (page - 1) * 4 // max(maximum, 1)))
    return selected


def corpus_quality_report(rows: list[dict[str, Any]], manuals: list[dict[str, Any]]) -> dict[str, Any]:
    """Return deterministic corpus coverage and extraction-risk metrics."""
    source_rejections = [
        issue
        for row in rows
        for issue in source_quality_issues(row)
    ]
    return {
        "manuals": len(manuals),
        "chunks": len(rows),
        "ocr_provenance_status": dict(sorted(Counter(manual["ocr_provenance"]["status"] for manual in manuals if manual.get("ocr_provenance")).items())),
        "chunks_by_manual": dict(sorted(Counter(row["manual_id"] for row in rows).items())),
        "chunks_by_source_category": dict(sorted(Counter(row["source_category"] for row in rows).items())),
        "chunks_by_content_class": dict(sorted(Counter(row["content_class"] for row in rows).items())),
        "chunks_with_image_markdown": sum("![" in row["passage"] for row in rows),
        "removed_image_or_caption_blocks": sum(
            image_artifact_count(row["passage"]) for row in rows
        ),
        "chunks_with_html_tables": sum("<table" in row["passage"].casefold() for row in rows),
        "chunks_with_replacement_characters": sum("\ufffd" in row["passage"] for row in rows),
        "empty_generation_chunks": sum(not row["generation_passage"] for row in rows),
        "answer_bearing_chunks": sum(not source_quality_issues(row) for row in rows),
        "source_quality_rejected_chunks": sum(bool(source_quality_issues(row)) for row in rows),
        "source_quality_rejection_reasons": dict(sorted(Counter(source_rejections).items())),
    }


def selection_coverage_report(
    all_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report stratified pilot coverage rather than only selected IDs."""

    def values(rows: list[dict[str, Any]], field: str) -> set[str]:
        return {str(row.get(field, "missing")) for row in rows}

    dimensions = {
        "manual": (values(all_rows, "manual_id"), values(selected_rows, "manual_id")),
        "source_category": (
            values(all_rows, "source_category"),
            values(selected_rows, "source_category"),
        ),
        "content_class": (
            values(all_rows, "content_class"),
            values(selected_rows, "content_class"),
        ),
    }
    return {
        "strategy": "deterministic_weighted_round_robin",
        "eligible_chunks": sum(not source_quality_issues(row) for row in all_rows),
        "selected_chunks": len(selected_rows),
        "dimensions": {
            name: {
                "available": sorted(available),
                "selected": sorted(selected),
                "covered": len(selected),
                "total": len(available),
                "coverage": round(len(selected) / len(available), 4)
                if available
                else 1.0,
            }
            for name, (available, selected) in dimensions.items()
        },
    }


def load_corpus(
    source_dir: Path,
    ocr_dir: Path,
    *,
    maximum_chars: int = 12000,
    minimum_chars: int = 200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only sources registered in manuals.yaml and return chunks plus manuals."""
    manifest_path = source_dir / "manuals.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manuals = payload.get("manuals", []) if isinstance(payload, dict) else []
    if not manuals:
        raise ValueError(f"No manuals registered in {manifest_path}")

    seen_manuals: set[str] = set()
    seen_source_paths: dict[Path, str] = {}
    seen_source_hashes: dict[str, str] = {}
    seen_chunks: set[str] = set()
    rows: list[dict[str, Any]] = []
    normalized_manuals: list[dict[str, Any]] = []
    for raw in manuals:
        manual = dict(raw)
        manual_id = str(manual["manual_id"])
        if manual_id in seen_manuals:
            raise ValueError(f"Duplicate manual_id: {manual_id}")
        seen_manuals.add(manual_id)
        source_path = (source_dir / manual["file"]).resolve()
        try:
            source_path.relative_to(source_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Registered source escapes source_dir: {manual['file']}"
            ) from exc
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        content_path = _ocr_markdown(ocr_dir, source_path) if source_path.suffix.lower() == ".pdf" else source_path
        ocr_provenance = _ocr_provenance(ocr_dir, source_path, content_path) if source_path.suffix.lower() == ".pdf" else None
        source_hash = _sha256(source_path)
        if source_path in seen_source_paths:
            raise ValueError(
                "Duplicate registered source path for manuals "
                f"{seen_source_paths[source_path]} and {manual_id}: {source_path}"
            )
        if source_hash in seen_source_hashes:
            raise ValueError(
                "Duplicate registered source content for manuals "
                f"{seen_source_hashes[source_hash]} and {manual_id}: {source_hash}"
            )
        expected_hash = str(
            (manual.get("official_verification") or {}).get("source_sha256", "")
        ).strip()
        if expected_hash and expected_hash != source_hash:
            raise ValueError(
                f"Registered source hash mismatch for {manual_id}: "
                f"expected {expected_hash}, got {source_hash}"
            )
        seen_source_paths[source_path] = manual_id
        seen_source_hashes[source_hash] = manual_id
        content_hash = _sha256(content_path)
        defaults = {
            "source_category": "government_manual",
            "issuing_organization": "Government of India",
            "policy_scope": "government_reference",
        }
        metadata = {**defaults, **manual}
        metadata.update(
            {
                "source_file": str(source_path.relative_to(source_dir)),
                "source_sha256": source_hash,
                "content_file": str(content_path),
                "content_sha256": content_hash,
                "ocr_provenance": ocr_provenance,
            }
        )
        normalized_manuals.append(metadata)
        start_page = int(manual.get("start_page", 1))
        excluded = {int(page) for page in manual.get("exclude_pages", [])}
        heading_stack: list[str] = []
        document_order = 0
        manual_row_start = len(rows)
        manual_pages_seen: set[int | None] = set()
        for page, page_text in _pages(content_path.read_text(encoding="utf-8")):
            if page is not None and (page < start_page or page in excluded):
                continue
            manual_pages_seen.add(page)
            for index, passage in enumerate(_chunks(page_text, maximum_chars), 1):
                if len(passage) < minimum_chars:
                    continue
                heading_stack = _updated_heading_stack(heading_stack, passage)
                document_order += 1
                digest = hashlib.sha256(passage.encode()).hexdigest()[:12]
                page_label = f"p{page:04d}" if page is not None else "pnone"
                chunk_id = f"{manual_id}-{page_label}-c{index:02d}-{digest}"
                if chunk_id in seen_chunks:
                    raise ValueError(f"Duplicate chunk_id: {chunk_id}")
                seen_chunks.add(chunk_id)
                rows.append(
                    {
                        **metadata,
                        "page": page,
                        "page_chunk_index": index,
                        "document_order": document_order,
                        "section": heading_stack[-1] if heading_stack else None,
                        "section_path": list(heading_stack),
                        "chunk_id": chunk_id,
                        "passage": passage,
                        "generation_passage": generation_text(passage),
                    }
                )
        manual_page_count = len(manual_pages_seen)
        for row in rows[manual_row_start:]:
            row["manual_page_count"] = manual_page_count
    if not rows:
        raise ValueError("The registered corpus produced no usable chunks")
    for row in rows:
        row["content_class"] = _content_class(row)
    return rows, normalized_manuals
