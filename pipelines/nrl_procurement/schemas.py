"""Typed records shared by generation, validation, and export."""

# ruff: noqa: D101

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProcurementPersona = Literal[
    "indenting_officer",
    "end_user_department_head",
    "procurement_officer",
    "tendering_officer",
    "technical_evaluator",
    "commercial_evaluator",
    "tender_committee_member",
    "tender_committee_chair",
    "finance_officer",
    "stores_rep_in_tender",
    "legal_officer_pre_award",
    "vigilance_officer_pre_award",
    "hope_head_of_procuring_entity",
    "approving_authority",
    "engineer_in_charge",
    "project_manager",
    "inspection_officer",
    "quality_control_officer",
    "stores_officer",
    "asset_manager",
    "contract_manager",
    "billing_officer",
    "payment_processing_officer",
    "performance_monitoring_officer",
    "vendor_performance_evaluator",
    "auditor",
    "internal_auditor",
    "cag_auditor",
    "vigilance_officer",
    "ethics_compliance_officer",
    "legal_officer",
    "independent_external_monitor",
    "ombudsman_grievance_officer",
    "gem_primary_user",
    "gem_buyer",
    "gem_bid_opening_officer",
    "gem_forwarding_officer",
    "eproc_admin",
    "system_audit_officer",
    "bidder",
    "supplier",
    "contractor",
    "consultant",
    "service_provider",
    "msme_startup_bidder",
    "foreign_bidder",
    "joint_venture_bidder",
    "integrity_pact_signatory_bidder",
    "holiday_listed_vendor",
    "external_reviewer",
    "general_user",
]

ProcurementTask = Literal[
    "general_reference",
    "need_and_planning",
    "market_analysis",
    "sourcing_and_mode",
    "tendering",
    "nit_filling",
    "clarifications_and_corrigenda",
    "bid_handling",
    "evaluation_and_award",
    "preference_policy_application",
    "security_and_guarantees",
    "contract_management",
    "execution_and_quality",
    "framework_and_rate_contracts",
    "disposal",
    "vendor_registration_and_empanelment",
    "supplier_governance",
    "compliance_and_audit",
    "ethics_and_risk_management",
    "grievance_handling",
    "drafting",
    "process_diagnosis",
    "cross_rule_application",
    "currentness",
]

QuestionType = Literal[
    "direct_fact",
    "definition",
    "procedure",
    "sequence",
    "threshold",
    "exception",
    "negative_rule",
    "role_responsibility",
    "comparison",
    "compliance_check",
    "drafting_knowledge",
    "currentness",
]


class EvidenceDraft(BaseModel):
    quote: str = Field(min_length=8)


class AnswerClaimDraft(BaseModel):
    """One material answer claim and its exact supporting source spans."""

    statement: str = Field(min_length=8)
    evidence: list[EvidenceDraft] = Field(min_length=1)


class ReasoningStepDraft(BaseModel):
    statement: str = Field(min_length=8)
    evidence_quotes: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    task_type: Literal["qa", "qa_cot"]
    task: ProcurementTask
    persona: ProcurementPersona
    question_type: QuestionType
    question: str = Field(min_length=12)
    answer: str = Field(min_length=1)
    answerable: bool = True
    claims: list[AnswerClaimDraft] = Field(default_factory=list)
    evidence: list[EvidenceDraft] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStepDraft] = Field(default_factory=list)


class CandidateBatch(BaseModel):
    examples: list[Candidate]


class PropositionDraft(BaseModel):
    """One source-language procurement proposition and its complete witness."""

    subject: str = Field(min_length=1)
    action: str = Field(min_length=1)
    object: str = Field(min_length=1)
    modality: Literal[
        "mandatory",
        "recommended",
        "permitted",
        "prohibited",
        "declarative",
    ]
    polarity: Literal["positive", "negative"]
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    threshold_value: str = ""
    threshold_unit: str = ""
    temporal_scope: str = ""
    evidence_quote: str = Field(min_length=8)


class PropositionBatch(BaseModel):
    propositions: list[PropositionDraft]


PathQuestionType = Literal[
    "comparison",
    "bridge",
    "temporal",
    "complementary",
    "condition_exception",
    "cross_domain",
]


class PathQuestionDraft(BaseModel):
    task: ProcurementTask
    persona: ProcurementPersona
    question_type: PathQuestionType
    difficulty: Literal["moderate", "advanced"]
    question: str = Field(min_length=12)


class PathQuestionBatch(BaseModel):
    questions: list[PathQuestionDraft] = Field(max_length=1)


class PathAnswerEvidenceDraft(BaseModel):
    proposition_id: str = Field(min_length=1)
    quote: str = Field(min_length=8)


class PathAnswerClaimDraft(BaseModel):
    statement: str = Field(min_length=8)
    evidence: list[PathAnswerEvidenceDraft] = Field(min_length=1)


class PathAnswerDraft(BaseModel):
    answer: str = Field(min_length=1)
    claims: list[PathAnswerClaimDraft] = Field(min_length=2)
    rationale_steps: list[str] = Field(default_factory=list, max_length=4)


class AblationTrialDraft(BaseModel):
    """One answer attempt under an explicitly bounded evidence context."""

    answerable: bool
    answer: str = ""
    claims: list[PathAnswerClaimDraft] = Field(default_factory=list)
    limitation_reason: str = ""


class JudgeDecision(BaseModel):
    supported: bool
    relevant: bool
    preserves_qualifications: bool
    authority_correct: bool
    reasoning_valid: bool
    recommended_task: ProcurementTask
    recommended_persona: ProcurementPersona
    answer_found_in_source: bool
    answer_quotes: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Zero to three independent verbatim source spans supporting an answer. " "Never concatenate separate spans or insert ellipses inside a span."
        ),
    )
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
    task: ProcurementTask
    instruction: str = Field(min_length=12)
    tender_context: list[str] = Field(min_length=1)
    manual_chunk_ids: list[str] = Field(min_length=1)


class DraftingBlock(BaseModel):
    text: str = Field(
        min_length=1,
        description=(
            "One ready-to-use document line or paragraph. Do not combine unrelated " "headings, fields, contacts, clauses, or footer lines in one block."
        ),
    )
    manual_evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Exact manual quotations supporting this block.",
    )
    tender_facts_used: list[str] = Field(
        default_factory=list,
        description="Complete verbatim tender facts supporting this block.",
    )
    instruction_evidence_quotes: list[str] = Field(
        default_factory=list,
        description=(
            "Exact instruction substrings supporting requested headings or layout; "
            "not a substitute for factual or policy evidence."
        ),
    )


class DraftingResult(BaseModel):
    document_blocks: list[DraftingBlock] = Field(
        min_length=2,
        description=(
            "Ordered ready-to-use document blocks. The caller renders one blank line " "between blocks; do not return an outline or drafting commentary."
        ),
    )
    manual_evidence_quotes: list[str] = Field(
        min_length=1,
        description="Exact manual quotations that govern the completed draft.",
    )
    tender_facts_used: list[str] = Field(
        min_length=1,
        description=("Every applicable tender fact used in the response, copied as complete " "verbatim items from TENDER FACTS."),
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
    task: ProcurementTask
    persona: ProcurementPersona
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
