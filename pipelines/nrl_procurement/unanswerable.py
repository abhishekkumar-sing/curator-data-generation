"""Construct and independently verify adversarial unanswerable QA."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from schemas import AnswerabilityDecision, UnanswerableQuestionDraft
from settings import CONFIG

from bespokelabs import curator

ABSTENTION = "Not answerable from the provided sources."


def answer_type(record: dict[str, Any]) -> str:
    """Infer a coarse answer type used only to choose plausible distractors."""
    answer = str(record.get("answer", ""))
    question = str(record.get("question", "")).casefold()
    if re.search(r"\b\d+(?:[.,]\d+)?\b|%|₹|rs\.?\b", answer, re.I):
        return "number_or_threshold"
    if re.search(r"\b(?:who|which authority|which officer|which committee)\b", question):
        return "actor_or_authority"
    if record.get("question_type") in {"procedure", "sequence"}:
        return "procedure"
    if record.get("question_type") in {"exception", "negative_rule"}:
        return "condition_or_exception"
    if record.get("question_type") == "definition":
        return "definition"
    return "policy_fact"


def build_unanswerable_inputs(
    records: list[dict[str, Any]],
    fraction: float,
    seed: str,
) -> list[dict[str, Any]]:
    """Pair answerable seeds with deterministic, same-type non-golden evidence."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("quality.unanswerable_fraction must be in [0, 1)")
    answerable = [
        row
        for row in records
        if row.get("answerable", True)
        and row.get("task_type") in {"qa", "qa_cot"}
        and row.get("evidence")
        and row.get("_source_passage")
    ]
    if not answerable or fraction == 0:
        return []
    target = math.ceil(len(answerable) * fraction / (1.0 - fraction))
    ordered = sorted(
        answerable,
        key=lambda row: hashlib.sha256(
            f"{seed}:negative-seed:{row['record_id']}".encode()
        ).hexdigest(),
    )
    inputs: list[dict[str, Any]] = []
    for source in ordered:
        source_type = answer_type(source)
        alternatives = [
            row
            for row in answerable
            if row["record_id"] != source["record_id"]
            and answer_type(row) == source_type
            and not set(row.get("source_chunk_ids", []))
            & set(source.get("source_chunk_ids", []))
        ]
        if not alternatives:
            continue
        alternatives.sort(
            key=lambda row: (
                row.get("manual_id") != source.get("manual_id"),
                hashlib.sha256(
                    f"{seed}:distractor:{source['record_id']}:{row['record_id']}".encode()
                ).hexdigest(),
            )
        )
        distractor = alternatives[0]
        distractor_evidence = distractor["evidence"][0]
        identity = json.dumps(
            [source["record_id"], distractor["record_id"], source_type],
            sort_keys=True,
        )
        inputs.append(
            {
                "construction_id": "unans-"
                + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "seed_record": source,
                "distractor_record_id": distractor["record_id"],
                "expected_answer_type": source_type,
                "distractor": {
                    "manual_id": distractor["manual_id"],
                    "chunk_id": distractor_evidence["chunk_id"],
                    "page": distractor_evidence.get("page"),
                    "section": distractor_evidence.get("section"),
                    "quote": distractor_evidence["quote"],
                    "source_passage": distractor["_source_passage"],
                },
            }
        )
        if len(inputs) >= target:
            break
    return inputs


def unanswerable_fraction_gate(
    total_records: int,
    unanswerable_count: int,
    target_fraction: float,
    *,
    relative_tolerance: float = 0.20,
) -> dict[str, Any]:
    """Flag when the achieved unanswerable share deviates materially from target.

    `build_unanswerable_inputs` can silently return fewer candidates than its
    own target (or none at all) when too few seeds have an eligible
    same-type, non-overlapping distractor. Nothing previously compared the
    achieved share of the final exported pool back to the configured
    `quality.unanswerable_fraction`, so that shortfall was invisible. A
    `target_fraction <= 0` means no target was configured, so it trivially
    passes rather than dividing by zero.
    """
    achieved_fraction = (unanswerable_count / total_records) if total_records else 0.0
    if target_fraction <= 0:
        return {
            "target_fraction": target_fraction,
            "achieved_fraction": round(achieved_fraction, 4),
            "relative_tolerance": relative_tolerance,
            "relative_deviation": None,
            "within_tolerance": True,
        }
    relative_deviation = abs(achieved_fraction - target_fraction) / target_fraction
    return {
        "target_fraction": target_fraction,
        "achieved_fraction": round(achieved_fraction, 4),
        "relative_tolerance": relative_tolerance,
        "relative_deviation": round(relative_deviation, 4),
        "within_tolerance": relative_deviation <= relative_tolerance,
    }


class AdversarialUnanswerableGenerator(curator.LLM):
    """Generate a plausible missing-premise question, never its answer label."""

    response_format = UnanswerableQuestionDraft

    def prompt(self, row: dict[str, Any]) -> str:
        """Render one answerable seed and a same-type hard distractor."""
        seed = row["seed_record"]
        distractor = row["distractor"]
        return f"""TASK
Create one plausible procurement question that is NOT answerable from either supplied
passage. Start from the answerable seed's information need, but alter exactly one
material actor, condition, threshold, exception, date, or authority premise. The
altered premise must be absent from both passages.

The distractor contains a plausible answer of the same coarse type
({row['expected_answer_type']}) but must not answer the new question. Do not merely
ask about missing page text, OCR damage, retrieval failure, or where text appears.
Return only the new question and a concise description of the missing premise.

ANSWERABLE SEED QUESTION: {seed['question']}
ANSWERABLE SEED ANSWER: {seed['answer']}

---BEGIN SEED SOURCE---
{seed['_source_passage']}
---END SEED SOURCE---

---BEGIN SAME-TYPE DISTRACTOR SOURCE---
{distractor['source_passage']}
---END SAME-TYPE DISTRACTOR SOURCE---
"""

    def parse(
        self,
        row: dict[str, Any],
        response: UnanswerableQuestionDraft,
    ) -> dict[str, Any]:
        """Materialize a negative candidate with complete construction lineage."""
        seed = row["seed_record"]
        question = response.question.strip()
        missing = response.missing_premise.strip()
        combined = (
            seed["_source_passage"] + "\n\n" + row["distractor"]["source_passage"]
        )
        issues = []
        if question.casefold() == str(seed["question"]).strip().casefold():
            issues.append("unanswerable_question_unchanged")
        if missing.casefold() in combined.casefold():
            issues.append("claimed_missing_premise_present_verbatim")
        identity = json.dumps(
            [row["construction_id"], question], ensure_ascii=False
        )
        return {
            "record_id": "nrlqa-"
            + hashlib.sha256(identity.encode()).hexdigest()[:20],
            "source_construction_id": row["construction_id"],
            "task_type": "qa",
            "task": seed["task"],
            "persona": seed["persona"],
            "question_type": "unanswerable",
            "question": question,
            "answer": ABSTENTION,
            "answerable": False,
            "claims": [],
            "reasoning_steps": [],
            "answer_format": "concise_direct",
            "evidence": [],
            "manual_id": seed["manual_id"],
            "manual_title": seed["manual_title"],
            "issuing_organization": seed["issuing_organization"],
            "policy_scope": seed["policy_scope"],
            "revision_date": seed["revision_date"],
            "as_of_date": seed["as_of_date"],
            "source_file": seed["source_file"],
            "source_sha256": seed["source_sha256"],
            "source_chunk_ids": [
                *seed.get("source_chunk_ids", []),
                row["distractor"]["chunk_id"],
            ],
            "citations": [],
            "parent_request_id": seed["parent_request_id"],
            "unanswerable_construction": {
                "source_answerable_record_id": seed["record_id"],
                "distractor_record_id": row["distractor_record_id"],
                "expected_answer_type": row["expected_answer_type"],
                "missing_premise": missing,
                "distractor": {
                    key: value
                    for key, value in row["distractor"].items()
                    if key != "source_passage"
                },
            },
            "_source_passage": combined,
            "generation_model": self.model_name,
            "deterministic_checks": {
                "passed": not issues,
                "issues": issues,
            },
        }


class IndependentAnswerabilityJudge(curator.LLM):
    """Independently decide whether the constructed question is answerable."""

    response_format = AnswerabilityDecision

    def prompt(self, row: dict[str, Any]) -> str:
        """Render the entire candidate context without generator conclusions."""
        construction = row["unanswerable_construction"]
        return f"""TASK
Independently assess whether the question can be answered completely from the full
supplied context. Do not trust the proposed abstention or missing-premise label.

- full_context_answerable=true if the context directly supports a complete answer.
- altered_premise_absent=true only if the alleged altered premise is genuinely absent.
- distractor_is_same_type=true only if the recorded distractor is a plausible answer
  of the expected coarse type but does not answer this question.
- abstention_is_appropriate=true only if refusing is the correct response.
- Preserve record_id exactly and list concrete issues.

record_id: {row['record_id']}
question: {row['question']}
expected_answer_type: {construction['expected_answer_type']}
alleged_missing_premise: {construction['missing_premise']}
distractor_quote: {construction['distractor']['quote']}

---BEGIN FULL UNTRUSTED CONTEXT---
{row['_source_passage']}
---END FULL UNTRUSTED CONTEXT---
"""

    def parse(
        self,
        row: dict[str, Any],
        response: AnswerabilityDecision,
    ) -> dict[str, Any]:
        """Attach a fail-closed answerability promotion decision."""
        decision = response.model_dump()
        identity_correct = decision["record_id"] == row["record_id"]
        accepted = (
            identity_correct
            and not decision["full_context_answerable"]
            and decision["altered_premise_absent"]
            and decision["distractor_is_same_type"]
            and decision["abstention_is_appropriate"]
            and decision["score"] >= int(CONFIG["quality"].get("minimum_judge_score", 4))
        )
        return {
            **row,
            "answerability_judge": {
                **decision,
                "record_id_correct": identity_correct,
                "accepted": accepted,
                "model": self.model_name,
            },
        }
