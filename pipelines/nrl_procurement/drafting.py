"""Seed-driven, source-grounded procurement drafting generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from pydantic import ValidationError

from schemas import DraftingJudgeDecision, DraftingResult, DraftingSeed

from bespokelabs import curator

NUMBER = re.compile(r"(?<![\w@])(?:₹|Rs\.?\s*)?\d[\d,.]*(?:\s*%|\s*[A-Za-z]+)?")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def read_drafting_seeds(path: Path) -> list[DraftingSeed]:
    """Load strict, unique JSONL seed records."""
    if not path.is_file():
        raise FileNotFoundError(f"Drafting seed file not found: {path}")
    seeds: list[DraftingSeed] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            seed = DraftingSeed.model_validate_json(raw_line)
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"Invalid drafting seed at {path}:{line_number}") from exc
        if seed.id in seen:
            raise ValueError(f"Duplicate drafting seed id at {path}:{line_number}: {seed.id}")
        seen.add(seed.id)
        seeds.append(seed)
    if not seeds:
        raise ValueError(f"No drafting seeds found in {path}")
    return seeds


def build_drafting_inputs(
    seeds: list[DraftingSeed], corpus_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve every authored seed to exact chunks in the current corpus."""
    chunks = {str(row["chunk_id"]): row for row in corpus_rows}
    if len(chunks) != len(corpus_rows):
        raise ValueError("Corpus contains duplicate chunk IDs")
    inputs: list[dict[str, Any]] = []
    for seed in seeds:
        selected: list[dict[str, Any]] = []
        for chunk_id in seed.manual_chunk_ids:
            if chunk_id not in chunks:
                raise ValueError(
                    f"Drafting seed {seed.id} references unknown chunk: {chunk_id}"
                )
            selected.append(chunks[chunk_id])
        tender_context = list(seed.tender_context)
        manual_context = "\n\n".join(
            (
                f"[{chunk['chunk_id']} | page {chunk['page']} | "
                f"{chunk.get('section') or 'section unavailable'}]\n{chunk['passage']}"
            )
            for chunk in selected
        )
        inputs.append(
            {
                **seed.model_dump(),
                "manual_context": manual_context,
                "manual_passages": [chunk["passage"] for chunk in selected],
                "combined_source_text": "\n".join(tender_context) + "\n\n" + manual_context,
                "citations": [*seed.manual_chunk_ids, seed.tender_id],
            }
        )
    return inputs


def drafting_validation_issues(
    row: dict[str, Any], result: DraftingResult
) -> list[str]:
    """Return deterministic grounding failures for a drafting response."""
    issues: list[str] = []
    manual_text = "\n\n".join(row["manual_passages"])
    combined = row["combined_source_text"]
    if "\n" not in result.response:
        issues.append("draft_has_no_document_line_structure")
    for quote in result.manual_evidence_quotes:
        if quote not in manual_text:
            issues.append("non_verbatim_manual_evidence")
    tender_facts = set(row["tender_context"])
    for fact in result.tender_facts_used:
        if fact not in tender_facts:
            issues.append("unknown_tender_fact")
    for value in NUMBER.findall(result.response):
        if value.strip() and value.strip().casefold() not in combined.casefold():
            issues.append(f"unsupported_number:{value.strip()}")
    for value in EMAIL.findall(result.response):
        if value.casefold() not in combined.casefold():
            issues.append(f"unsupported_email:{value}")
    return sorted(set(issues))


class TenderDraftingGenerator(curator.LLM):
    """Generate one grounded procurement draft for each authored request."""

    response_format = DraftingResult

    def prompt(self, row: dict[str, Any]) -> str:
        """Render a source-separated drafting request."""
        return f"""Draft the requested NRL procurement text.

Rules:
- Use only the supplied tender facts and manual context.
- Produce the complete, ready-to-use text requested by INSTRUCTION. A heading,
  outline, summary, commentary, or drafting advice is not an acceptable response.
- Format it as a usable document: include newline characters between the heading,
  every labelled field, body paragraph, contact, and footer. Never join adjacent
  values, labels, sentences, or email addresses without whitespace.
- Tender facts provide instance-specific names, references, contacts, and particulars.
- Include every tender fact required by the instruction and every applicable fact
  needed to make the requested text complete. Do not silently omit supplied required
  fields, organization identity, references, bidding structure, contacts, or footer.
- Manual context provides governing procurement rules.
- Do not invent, broaden, or weaken a rule, threshold, exception, remedy, or authority.
- If tender facts conflict with the manual, state the conflict instead of blending them.
- Omit an optional field, or write [NOT PROVIDED] for a required field, when its value is absent.
- Return every manual_evidence_quote verbatim from MANUAL CONTEXT.
- Return every tender_facts_used entry verbatim as one complete item from TENDER FACTS.
- Before returning, check the response against each requirement in INSTRUCTION and
  each applicable TENDER FACTS item.
- The response must contain no citation list; citations are attached deterministically.

TENDER ID: {row['tender_id']}
TASK: {row['task']}
INSTRUCTION: {row['instruction']}

TENDER FACTS:
---BEGIN TENDER FACTS---
{json.dumps(row['tender_context'], ensure_ascii=False, indent=2)}
---END TENDER FACTS---

MANUAL CONTEXT:
---BEGIN MANUAL CONTEXT---
{row['manual_context']}
---END MANUAL CONTEXT---
"""

    def parse(self, row: dict[str, Any], response: DraftingResult) -> list[dict[str, Any]]:
        """Retain full lineage while marking deterministic acceptance."""
        issues = drafting_validation_issues(row, response)
        context = [*row["tender_context"], *response.manual_evidence_quotes]
        return [
            {
                "id": row["id"],
                "tender_id": row["tender_id"],
                "task": row["task"],
                "instruction": row["instruction"],
                "context": context,
                "response": response.response,
                "citations": row["citations"],
                "evidence_quotes": response.manual_evidence_quotes,
                "tender_facts_used": response.tender_facts_used,
                "manual_chunk_ids": row["manual_chunk_ids"],
                "generation_model": self.model_name,
                "deterministic_checks": {
                    "passed": not issues,
                    "issues": issues,
                },
                "_combined_source_text": row["combined_source_text"],
            }
        ]


class TenderDraftingJudge(curator.LLM):
    """Judge drafting utility after deterministic source checks."""

    response_format = DraftingJudgeDecision

    def prompt(self, row: dict[str, Any]) -> str:
        """Render a criterion-based single-record judgment."""
        return f"""Judge this procurement draft only against its instruction and sources.

Reject invented content, broadened rules, dropped conditions or exceptions, incorrect
authority, unsafe resolution of source conflicts, or failure to perform the requested
drafting task. Score 4-5 only when it is suitable as grounded drafting supervision.

INSTRUCTION:
{row['instruction']}

SOURCES:
{row['_combined_source_text']}

DRAFT:
{row['response']}
"""

    def parse(
        self, row: dict[str, Any], response: DraftingJudgeDecision
    ) -> dict[str, Any]:
        """Attach a transparent acceptance decision."""
        decision = response.model_dump()
        accepted = (
            all(
                decision[field]
                for field in (
                    "supported",
                    "follows_instruction",
                    "preserves_policy_qualifications",
                    "resolves_source_conflicts_safely",
                )
            )
            and decision["score"] >= int(row.get("_minimum_judge_score", 4))
        )
        return {
            **row,
            "judge": {
                **decision,
                "accepted": accepted,
                "model": self.model_name,
            },
        }


def compact_drafting(row: dict[str, Any]) -> dict[str, Any]:
    """Return the requested stable drafting JSONL contract."""
    return {
        "id": row["id"],
        "tender_id": row["tender_id"],
        "task": row["task"],
        "instruction": row["instruction"],
        "context": row["context"],
        "response": row["response"],
        "citations": row["citations"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
