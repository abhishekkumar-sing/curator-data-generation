"""Typed records shared by generation, validation, and export."""

from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal[
    "direct_fact",
    "definition",
    "authority",
    "threshold",
    "conditional_rule",
    "exception",
    "procedure",
    "scenario",
    "multi_section",
    "temporal",
    "unanswerable",
]


class EvidenceDraft(BaseModel):
    quote: str = Field(min_length=8)


class ReasoningStepDraft(BaseModel):
    statement: str = Field(min_length=8)
    evidence_quotes: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    task_type: Literal["qa", "qa_cot"]
    question_type: QuestionType
    question: str = Field(min_length=12)
    answer: str = Field(min_length=1)
    answerable: bool = True
    evidence: list[EvidenceDraft] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStepDraft] = Field(default_factory=list)


class CandidateBatch(BaseModel):
    examples: list[Candidate]


class JudgeDecision(BaseModel):
    supported: bool
    relevant: bool
    preserves_qualifications: bool
    authority_correct: bool
    reasoning_valid: bool
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)


class JudgedCandidate(BaseModel):
    record_id: str
    decision: JudgeDecision


class JudgeBatch(BaseModel):
    judgments: list[JudgedCandidate]
