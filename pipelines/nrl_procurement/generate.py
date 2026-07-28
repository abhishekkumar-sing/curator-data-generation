"""Generate grounded synthetic procurement QA data with a hosted Nemotron model."""

# ruff: noqa: I001

import argparse
import os
import re
from pathlib import Path

from datasets import Dataset
from pydantic import BaseModel, Field

from settings import CONFIG, PROJECT_ROOT, require_private_endpoint, require_setting

# settings applies privacy controls before this import.
from bespokelabs import curator

PATH_CONFIG = CONFIG["paths"]
GENERATION_CONFIG = CONFIG["models"]["generation"]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / PATH_CONFIG["source_dir"]
DEFAULT_OCR_DIR = PROJECT_ROOT / PATH_CONFIG["ocr_dir"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / PATH_CONFIG["output_dir"]
PAGE_PATTERN = re.compile(r"<!-- Page (\d+) -->")


class QuestionAnswer(BaseModel):
    """One evidence-grounded synthetic question and answer."""

    question: str = Field(description="A standalone procurement question")
    answer: str = Field(description="An answer supported only by the supplied source")
    evidence_quote: str = Field(description="An exact supporting quotation from the source")


class SyntheticExamples(BaseModel):
    """Synthetic examples generated from one source passage."""

    examples: list[QuestionAnswer]


def markdown_pages(path: Path, source_dir: Path, source_prefix: str = "") -> list[dict]:
    """Split a converted procurement manual into page-preserving inputs."""
    text = path.read_text(encoding="utf-8")
    matches = list(PAGE_PATTERN.finditer(text))
    source_file = str(path.relative_to(source_dir))
    if source_prefix:
        source_file = f"{source_prefix}/{source_file}"

    if not matches:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        chunks = []
        current = []
        current_length = 0
        for paragraph in paragraphs:
            if current and current_length + len(paragraph) > 12000:
                chunks.append("\n\n".join(current))
                current = []
                current_length = 0
            current.append(paragraph)
            current_length += len(paragraph) + 2
        if current:
            chunks.append("\n\n".join(current))
        return [
            {
                "source_file": source_file,
                "page": None,
                "source_chunk": index,
                "passage": chunk,
            }
            for index, chunk in enumerate(chunks, start=1)
            if len(chunk) >= 200
        ]

    rows = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        passage = text[match.end() : end].strip()
        if len(passage) < 200:
            continue
        rows.append(
            {
                "source_file": source_file,
                "page": int(match.group(1)),
                "source_chunk": index + 1,
                "passage": passage[:12000],
            }
        )
    return rows


def load_sources(source_dir: Path, ocr_dir: Path, limit: int | None) -> Dataset:
    """Load page-level inputs from all Markdown sources."""
    rows = []
    for path in sorted(source_dir.rglob("*.md")):
        rows.extend(markdown_pages(path, source_dir))
    if ocr_dir.is_dir():
        for path in sorted(ocr_dir.rglob("*.md")):
            rows.extend(markdown_pages(path, ocr_dir, "chandra"))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No Markdown source pages found under {source_dir}")
    return Dataset.from_list(rows)


class ProcurementQAGenerator(curator.LLM):
    """Generate auditable QA examples from procurement source passages."""

    response_format = SyntheticExamples

    def prompt(self, row: dict) -> str:
        """Build a strictly grounded generation prompt."""
        return f"""Create up to three diverse procurement training examples from the source passage.

Requirements:
- Use only facts explicitly stated in the passage.
- Make every question standalone and unambiguous.
- Preserve all thresholds, dates, exceptions, and qualifications.
- evidence_quote must be copied verbatim from the passage.
- Do not infer that Government guidance is NRL policy.

Source: {row["source_file"]}, page {row["page"]}, source chunk {row["source_chunk"]}

Passage:
{row["passage"]}
"""

    def parse(self, row: dict, response: SyntheticExamples) -> list[dict]:
        """Attach source provenance and reject non-verbatim evidence."""
        results = []
        for example in response.examples:
            if example.evidence_quote not in row["passage"]:
                continue
            results.append(
                {
                    "question": example.question,
                    "answer": example.answer,
                    "evidence_quote": example.evidence_quote,
                    "source_file": row["source_file"],
                    "page": row["page"],
                    "source_chunk": row["source_chunk"],
                }
            )
        return results


def main() -> None:
    """Run grounded synthetic QA generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="Limit source pages for a pilot run")
    args = parser.parse_args()

    model = require_setting(GENERATION_CONFIG["served_model_env"])
    base_url_name = GENERATION_CONFIG["base_url_env"]
    base_url = (
        require_private_endpoint(base_url_name)
        if GENERATION_CONFIG.get("private_endpoint_only", True)
        else require_setting(base_url_name)
    )
    api_key = require_setting(GENERATION_CONFIG["api_key_env"])
    os.environ["HOSTED_VLLM_API_KEY"] = api_key

    generator = ProcurementQAGenerator(
        model_name=f"hosted_vllm/{model}",
        backend="litellm",
        response_format=SyntheticExamples,
        generation_params=GENERATION_CONFIG["generation_params"],
        backend_params={
            "base_url": base_url,
            "api_key": api_key,
            "request_timeout": GENERATION_CONFIG["request_timeout"],
            "max_concurrent_requests": GENERATION_CONFIG["max_concurrent_requests"],
            "require_all_responses": False,
        },
    )

    dataset = load_sources(
        args.source_dir.resolve(),
        args.ocr_dir.resolve(),
        args.limit,
    )
    result = generator(dataset, working_dir=str(args.output_dir.resolve()))
    output = args.output_dir / "procurement_qa"
    result.dataset.save_to_disk(str(output))
    print(f"Saved {len(result.dataset)} examples to {output}")


if __name__ == "__main__":
    main()
