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
    return {
        "status": "complete" if markdown_hash and payload.get("model") else "legacy",
        "cache_file": str(cache),
        "model": payload.get("model"),
        "engine": payload.get("engine"),
        "chandra_ocr_version": payload.get("chandra_ocr_version"),
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


def generation_text(passage: str) -> str:
    """Remove non-policy image descriptions while preserving tables and prose."""
    value = IMAGE_LINE_PATTERN.sub("", passage)
    value = HTML_IMAGE_PATTERN.sub("", value)
    value = re.sub(r"\n[ \t]+\n", "\n\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _document_family(manual_id: str) -> str:
    for family in ("goods", "works", "services", "consultancy"):
        if family in manual_id:
            return family
    return "other"


def _content_class(row: dict[str, Any]) -> str:
    passage = row["generation_passage"]
    section = str(row.get("section") or "").casefold()
    if "<table" in passage.casefold():
        return "table"
    if int(row.get("page") or 1) <= int(row.get("start_page", 1)) + 7 or any(label in section for label in ("foreword", "preface", "introduction", "contents")):
        return "front_matter"
    return "policy"


def representative_rows(rows: list[dict[str, Any]], limit: int | None, seed: str) -> list[dict[str, Any]]:
    """Select a deterministic diversity-first pilot instead of a corpus prefix."""
    eligible = [
        row
        for row in rows
        if len(row["generation_passage"]) >= 200
        and row["content_class"] != "front_matter"
    ]
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
    return {
        "manuals": len(manuals),
        "chunks": len(rows),
        "ocr_provenance_status": dict(sorted(Counter(manual["ocr_provenance"]["status"] for manual in manuals if manual.get("ocr_provenance")).items())),
        "chunks_by_manual": dict(sorted(Counter(row["manual_id"] for row in rows).items())),
        "chunks_by_source_category": dict(sorted(Counter(row["source_category"] for row in rows).items())),
        "chunks_by_content_class": dict(sorted(Counter(row["content_class"] for row in rows).items())),
        "chunks_with_image_markdown": sum("![" in row["passage"] for row in rows),
        "chunks_with_html_tables": sum("<table" in row["passage"].casefold() for row in rows),
        "chunks_with_replacement_characters": sum("\ufffd" in row["passage"] for row in rows),
        "empty_generation_chunks": sum(not row["generation_passage"] for row in rows),
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
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        content_path = _ocr_markdown(ocr_dir, source_path) if source_path.suffix.lower() == ".pdf" else source_path
        ocr_provenance = _ocr_provenance(ocr_dir, source_path, content_path) if source_path.suffix.lower() == ".pdf" else None
        source_hash = _sha256(source_path)
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
        current_section: str | None = None
        for page, page_text in _pages(content_path.read_text(encoding="utf-8")):
            if page is not None and (page < start_page or page in excluded):
                continue
            for index, passage in enumerate(_chunks(page_text, maximum_chars), 1):
                if len(passage) < minimum_chars:
                    continue
                headings = HEADING_PATTERN.findall(passage)
                if headings:
                    current_section = headings[-1][1].strip()
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
                        "section": current_section,
                        "chunk_id": chunk_id,
                        "passage": passage,
                        "generation_passage": generation_text(passage),
                    }
                )
    if not rows:
        raise ValueError("The registered corpus produced no usable chunks")
    for row in rows:
        row["content_class"] = _content_class(row)
    return rows, normalized_manuals
