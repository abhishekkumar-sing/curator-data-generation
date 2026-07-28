"""Manifest-aware, page-preserving procurement corpus loading."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

PAGE_PATTERN = re.compile(r"<!--\s*Page\s+(\d+)\s*-->", re.IGNORECASE)
CHANDRA_PAGE_PATTERN = re.compile(r"(?m)^\s*(\d+)-{20,}\s*$")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ocr_markdown(ocr_dir: Path, source: Path) -> Path:
    matches = sorted(
        path
        for path in ocr_dir.rglob("*.md")
        if path.stem == source.stem or path.parent.name == source.stem
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Chandra Markdown output for {source.name} under "
            f"{ocr_dir}; found {len(matches)}. Run preprocess_pdfs.py first."
        )
    return matches[0]


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
                text[
                    marker.end() : chandra_markers[index + 1].start()
                    if index + 1 < len(chandra_markers)
                    else len(text)
                ].strip(),
            )
            for index, marker in enumerate(chandra_markers)
        )
        return pages
    return [
        (
            int(marker.group(1)),
            text[
                marker.end() : markers[index + 1].start()
                if index + 1 < len(markers)
                else len(text)
            ].strip(),
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
        content_path = (
            _ocr_markdown(ocr_dir, source_path)
            if source_path.suffix.lower() == ".pdf"
            else source_path
        )
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
                    }
                )
    if not rows:
        raise ValueError("The registered corpus produced no usable chunks")
    return rows, normalized_manuals
