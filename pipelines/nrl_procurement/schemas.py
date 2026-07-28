"""Typed records shared by generation, validation, and export."""

# ruff: noqa: D101

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class DraftingSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    tender_id: str = Field(min_length=1)
    task: Literal["drafting"]
    instruction: str = Field(min_length=12)
    tender_context: list[str] = Field(min_length=1)
    manual_chunk_ids: list[str] = Field(min_length=1)


class DraftingResult(BaseModel):
    response: str = Field(
        min_length=40,
        description=(
            "Complete ready-to-use text requested by the instruction, not a title, "
            "outline, summary, or explanation of what should be drafted. Preserve "
            "document formatting with newline characters between headings, fields, "
            "paragraphs, contacts, and footers."
        ),
    )
    manual_evidence_quotes: list[str] = Field(
        min_length=1,
        description="Exact manual quotations that govern the completed draft.",
    )
    tender_facts_used: list[str] = Field(
        min_length=1,
        description=(
            "Every applicable tender fact used in the response, copied as complete "
            "verbatim items from TENDER FACTS."
        ),
    )


class DraftingJudgeDecision(BaseModel):
    supported: bool
    follows_instruction: bool
    preserves_policy_qualifications: bool
    resolves_source_conflicts_safely: bool
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)


CrossRelationship = Literal[
    "same_authority_temporal",
    "government_company_comparison",
    "company_cross_domain",
    "complementary_procedure",
]


class CrossEvidenceDraft(BaseModel):
    source_id: str
    quote: str = Field(min_length=8)


class CrossClaimDraft(BaseModel):
    statement: str = Field(min_length=8)
    evidence: list[CrossEvidenceDraft] = Field(min_length=1)


class CrossReasoningStepDraft(BaseModel):
    operation: Literal[
        "lookup",
        "compare",
        "apply_condition",
        "resolve_authority",
        "resolve_time",
        "combine",
        "calculate",
        "conclude",
    ]
    statement: str = Field(min_length=8)
    evidence: list[CrossEvidenceDraft] = Field(min_length=1)


class CrossCandidate(BaseModel):
    task_type: Literal["cross_document_qa", "cross_document_qa_cot"]
    question_type: Literal[
        "comparison",
        "temporal",
        "complementary",
        "bridge",
        "cross_domain",
        "unanswerable",
    ]
    question: str = Field(min_length=12)
    answer: str = Field(min_length=1)
    answerable: bool = True
    claims: list[CrossClaimDraft] = Field(min_length=1)
    reasoning_steps: list[CrossReasoningStepDraft] = Field(default_factory=list)


class CrossCandidateBatch(BaseModel):
    examples: list[CrossCandidate]


class CrossJudgeDecision(JudgeDecision):
    full_context_supported: bool
    unsupported_without_source_ids: list[str]
    connected_reasoning: bool
    relationship_correct: bool


class CrossJudgedCandidate(BaseModel):
    record_id: str
    decision: CrossJudgeDecision


class CrossJudgeBatch(BaseModel):
    judgments: list[CrossJudgedCandidate]
