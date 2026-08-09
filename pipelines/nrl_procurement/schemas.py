"""Typed records shared by generation, validation, and export."""

# ruff: noqa: D101

import json
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue


class AuditedListModel(BaseModel):
    """Recover only valid JSON-encoded lists and retain an audit marker.

    Some OpenAI-compatible tool servers return a schema list as a JSON string.
    This narrow repair accepts only a string that decodes to an actual JSON
    list. It deliberately does not use ``eval``, coerce prose, or map enums.
    The audit field is excluded from serialization and from the tool schema.
    """

    json_list_fields: ClassVar[tuple[str, ...]] = ()
    scalar_string_list_fields: ClassVar[tuple[str, ...]] = ()
    list_max_items: ClassVar[dict[str, int]] = {}
    structural_repairs: list[str] = Field(
        default_factory=list,
        exclude=True,
        alias="_structural_repairs",
    )
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def recover_json_lists(cls, value: Any) -> Any:
        """Decode only configured fields containing valid JSON arrays."""
        if not isinstance(value, dict):
            return value
        repaired = dict(value)
        repairs = list(repaired.get("_structural_repairs", []))
        for field_name in cls.json_list_fields:
            candidate = repaired.get(field_name)
            if isinstance(candidate, str):
                try:
                    decoded = json.loads(candidate)
                except json.JSONDecodeError:
                    decoded = candidate.strip()
                if isinstance(decoded, list):
                    repaired[field_name] = decoded
                    repairs.append(f"stringified_json_list:{field_name}")
                elif (
                    field_name in cls.scalar_string_list_fields
                    and isinstance(decoded, str)
                    and decoded.strip()
                ):
                    repaired[field_name] = [decoded.strip()]
                    repairs.append(f"scalar_string_to_list:{field_name}")
            maximum = cls.list_max_items.get(field_name)
            current = repaired.get(field_name)
            if maximum is not None and isinstance(current, list):
                if len(current) > maximum:
                    repaired[field_name] = current[:maximum]
                    repairs.append(
                        f"list_clipped:{field_name}:{len(current)}>{maximum}"
                    )
        if repairs:
            repaired["_structural_repairs"] = repairs
        return repaired

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Hide the pipeline-only repair ledger from model-facing schemas."""
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        schema.get("properties", {}).pop("_structural_repairs", None)
        required = schema.get("required", [])
        schema["required"] = [
            name for name in required if name != "_structural_repairs"
        ]
        if not schema.get("required"):
            schema.pop("required", None)
        return schema


def collect_structural_repairs(value: Any, path: str = "") -> list[str]:
    """Collect nested structural repair markers with stable field paths."""
    repairs: list[str] = []
    if isinstance(value, AuditedListModel):
        repairs.extend(
            f"{path}:{repair}" if path else repair
            for repair in value.structural_repairs
        )
    if isinstance(value, BaseModel):
        for field_name in value.__class__.model_fields:
            if field_name == "structural_repairs":
                continue
            child_path = f"{path}.{field_name}" if path else field_name
            repairs.extend(
                collect_structural_repairs(getattr(value, field_name), child_path)
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            repairs.extend(collect_structural_repairs(item, f"{path}[{index}]"))
    return list(dict.fromkeys(repairs))

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

IssueText = Annotated[str, Field(max_length=160)]

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
    # Short source labels such as "Buyer" and "Seller" can be complete,
    # material evidence. Grounding and claim support are validated separately;
    # an arbitrary character minimum only turns valid short spans into schema
    # failures before those checks can run.
    quote: str = Field(min_length=1)


class AnswerClaimDraft(AuditedListModel):
    """One material answer claim and its exact supporting source spans."""

    # Parse incomplete model objects so deterministic validation can persist a
    # precise rejection instead of losing the entire request to Pydantic.
    statement: str = ""
    evidence: list[EvidenceDraft] = Field(default_factory=list)
    json_list_fields = ("evidence",)


class ReasoningStepDraft(AuditedListModel):
    # Intermediate model output is intentionally permissive. The validator
    # enforces the operation vocabulary and grounded inputs while preserving
    # malformed rows in the audit trail.
    operation: str = ""
    statement: str = ""
    evidence_quotes: list[str] = Field(default_factory=list)
    json_list_fields = ("evidence_quotes",)


class Candidate(AuditedListModel):
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
    json_list_fields = ("claims", "evidence", "reasoning_steps")


class CandidateBatch(AuditedListModel):
    examples: list[Candidate]
    json_list_fields = ("examples",)


class QABlueprintDraft(AuditedListModel):
    """A grounded plan produced before final question wording."""

    # Keep the allowed vocabulary in the prompt and validate it in parse. This
    # prevents one invented label from erasing the whole response before an
    # auditable rejection can be written.
    task: str
    persona: str
    persona_need: str = Field(
        min_length=12,
        description=(
            "Concrete work decision, check, or action for which this persona "
            "needs the answer; never a role-play preamble."
        ),
    )
    instruction_goal: str = Field(min_length=12)
    must_cover: list[str] = Field(min_length=1, max_length=4)
    evidence: list[EvidenceDraft] = Field(min_length=1, max_length=4)
    json_list_fields = ("must_cover", "evidence")
    scalar_string_list_fields = ("must_cover",)
    list_max_items = {"must_cover": 4, "evidence": 4}


class GroundedCandidateDraft(AuditedListModel):
    """Final wording for one already-fixed blueprint.

    Contract labels deliberately do not appear here. The pipeline injects the
    blueprint's task, persona, question type, answer format, task type, and
    answerability so the final model cannot rename or swap them.
    """

    question: str = Field(min_length=12)
    answer: str = Field(min_length=1)
    claims: list[AnswerClaimDraft] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStepDraft] = Field(default_factory=list)
    json_list_fields = ("claims", "reasoning_steps")


class UnanswerableQuestionDraft(BaseModel):
    """One adversarial question derived from a valid answerable seed."""

    question: str = Field(min_length=12)
    missing_premise: str = Field(min_length=4)


class AnswerabilityDecision(AuditedListModel):
    """Independent full-context verification of an adversarial negative."""

    record_id: str = Field(min_length=1)
    full_context_answerable: bool
    altered_premise_absent: bool
    distractor_is_same_type: bool
    abstention_is_appropriate: bool
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    json_list_fields = ("issues",)


class PropositionDraft(AuditedListModel):
    """One source-language procurement proposition and its complete witness."""

    subject: str = Field(min_length=1)
    action: str = Field(min_length=1)
    object: str = ""
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
    json_list_fields = ("conditions", "exceptions")


class PropositionBatch(AuditedListModel):
    propositions: list[PropositionDraft]
    json_list_fields = ("propositions",)


PathQuestionType = Literal[
    "comparison",
    "bridge",
    "temporal",
    "complementary",
    "condition_exception",
    "cross_domain",
]


class PathQuestionDraft(BaseModel):
    # Parse unknown labels so deterministic validation can preserve the request
    # as a rejection instead of dropping it at the transport boundary.
    task: str
    persona: str
    question_type: str
    difficulty: str
    question: str = Field(min_length=12)


class PathQuestionBatch(AuditedListModel):
    questions: list[PathQuestionDraft] = Field(max_length=1)
    json_list_fields = ("questions",)


class PathAnswerEvidenceDraft(BaseModel):
    proposition_id: str = Field(min_length=1)
    quote: str = Field(min_length=8)


class PathAnswerClaimDraft(AuditedListModel):
    statement: str = Field(min_length=8)
    evidence: list[PathAnswerEvidenceDraft] = Field(min_length=1)
    json_list_fields = ("evidence",)


class PathAnswerDraft(AuditedListModel):
    answer: str = Field(min_length=1)
    claims: list[PathAnswerClaimDraft] = Field(min_length=2)
    rationale_steps: list[str] = Field(default_factory=list, max_length=4)
    json_list_fields = ("claims", "rationale_steps")


class AblationTrialDraft(AuditedListModel):
    """One answer attempt under an explicitly bounded evidence context."""

    answerable: bool
    answer: str = ""
    claims: list[PathAnswerClaimDraft] = Field(default_factory=list)
    limitation_reason: str = ""
    json_list_fields = ("claims",)


class AblationJudgeDecision(AuditedListModel):
    """Independent review of persisted full/A-only/B-only answer attempts."""

    record_id: str
    full_context_supported: bool
    source_a_only_incomplete: bool
    source_b_only_incomplete: bool
    comparison_valid: bool
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    json_list_fields = ("issues",)


class JudgeDecision(AuditedListModel):
    supported: bool
    relevant: bool
    preserves_qualifications: bool
    authority_correct: bool
    reasoning_valid: bool
    question_natural: bool
    persona_relevant: bool
    recommended_task: str
    recommended_persona: str
    answer_found_in_source: bool
    answer_quotes: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Zero to three independent verbatim source spans supporting an answer. " "Never concatenate separate spans or insert ellipses inside a span."
        ),
    )
    score: int = Field(ge=1, le=5)
    issues: list[IssueText] = Field(default_factory=list, max_length=4)
    json_list_fields = ("answer_quotes", "issues")


class JudgedCandidate(BaseModel):
    record_id: str
    decision: JudgeDecision


class JudgeBatch(AuditedListModel):
    judgments: list[JudgedCandidate]
    json_list_fields = ("judgments",)


class DraftingSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    tender_id: str = Field(min_length=1)
    task: ProcurementTask
    instruction: str = Field(min_length=12)
    tender_context: list[str] = Field(min_length=1)
    manual_chunk_ids: list[str] = Field(min_length=1)


class DraftingBlock(AuditedListModel):
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
    json_list_fields = (
        "manual_evidence_quotes",
        "tender_facts_used",
        "instruction_evidence_quotes",
    )


class DraftingFieldClaim(AuditedListModel):
    """One material draft value bound to its exact block-local support."""

    block_index: int = Field(ge=0)
    field_name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    manual_evidence_quotes: list[str] = Field(default_factory=list)
    tender_facts_used: list[str] = Field(default_factory=list)
    instruction_evidence_quotes: list[str] = Field(default_factory=list)
    json_list_fields = (
        "manual_evidence_quotes",
        "tender_facts_used",
        "instruction_evidence_quotes",
    )


class DraftingResult(AuditedListModel):
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
    field_claims: list[DraftingFieldClaim] = Field(
        default_factory=list,
        description=(
            "Atomic material values/claims, each tied to one document block and "
            "the exact manual, tender, or instruction support used for it."
        ),
    )
    json_list_fields = (
        "document_blocks",
        "manual_evidence_quotes",
        "tender_facts_used",
        "field_claims",
    )


class DraftingJudgeDecision(AuditedListModel):
    supported: bool
    follows_instruction: bool
    preserves_policy_qualifications: bool
    resolves_source_conflicts_safely: bool
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    json_list_fields = ("issues",)


CrossRelationship = Literal[
    "supersedes",
    "amends",
    "carries_forward",
    "adds_requirement",
    "removes_requirement",
    "changes_threshold",
    "changes_scope",
    "organization_deviation",
    "cross_reference_change",
    "complementary_procedure",
]


class CrossEvidenceDraft(BaseModel):
    source_id: str
    quote: str = Field(min_length=8)


class CrossClaimDraft(AuditedListModel):
    statement: str = Field(min_length=8)
    evidence: list[CrossEvidenceDraft] = Field(min_length=1)
    json_list_fields = ("evidence",)


class CrossReasoningStepDraft(AuditedListModel):
    operation: str
    statement: str = Field(min_length=8)
    evidence: list[CrossEvidenceDraft] = Field(min_length=1)
    json_list_fields = ("evidence",)


class CrossCandidate(AuditedListModel):
    task_type: str
    task: str
    persona: str
    question_type: str
    question: str = Field(min_length=12)
    answer: str = Field(min_length=1)
    answerable: bool = True
    claims: list[CrossClaimDraft] = Field(min_length=1)
    reasoning_steps: list[CrossReasoningStepDraft] = Field(default_factory=list)
    json_list_fields = ("claims", "reasoning_steps")


class CrossCandidateBatch(AuditedListModel):
    examples: list[CrossCandidate]
    json_list_fields = ("examples",)


class CrossAblationTrialDraft(AuditedListModel):
    """One cross-document answer attempt under an explicitly bounded source context."""

    answerable: bool
    answer: str = ""
    claims: list[CrossClaimDraft] = Field(default_factory=list)
    limitation_reason: str = ""
    json_list_fields = ("claims",)


class CrossJudgeDecision(JudgeDecision):
    full_context_supported: bool
    unsupported_without_source_ids: list[str]
    connected_reasoning: bool
    relationship_correct: bool
    json_list_fields = ("answer_quotes", "issues", "unsupported_without_source_ids")


class CrossJudgedCandidate(BaseModel):
    record_id: str
    decision: CrossJudgeDecision


class CrossJudgeBatch(AuditedListModel):
    judgments: list[CrossJudgedCandidate]
    json_list_fields = ("judgments",)
