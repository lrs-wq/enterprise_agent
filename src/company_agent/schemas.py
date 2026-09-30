from typing import Literal, TypeAlias

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

ResponseStatus: TypeAlias = Literal[
    "grounded",
    "general_knowledge",
    "partial_answer",
    "out_of_scope",
    "clarification_needed",
    "insufficient_evidence",
    "service_unavailable",
    "conversational",
]
EvidenceStatus: TypeAlias = Literal[
    "supported", "partial", "missing", "not_applicable", "model_knowledge"
]
KnowledgeDomain: TypeAlias = Literal["company_public", "product_public"]
FallbackKind: TypeAlias = Literal[
    "policy_boundary",
    "clarification",
    "missing_evidence",
    "service_error",
    "scope_only",
    "verified_extractive",
]


class SourceRef(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    document_title: str
    filename: str
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    score: float = Field(ge=-1, le=1)
    excerpt: str
    source_version: int | None = None
    source_kind: Literal["text", "media"] = "text"


class ThinkingStep(BaseModel):
    stage: Literal["intent", "retrieval", "evidence", "conclusion"]
    summary: str = Field(min_length=1, max_length=300)
    evidence_ids: list[str] = Field(default_factory=list)


class AnswerSection(BaseModel):
    heading: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=4000)


class GeneratedSection(BaseModel):
    heading: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=4000)
    source_ids: list[str] = Field(max_length=8)

    @field_validator("source_ids")
    @classmethod
    def unique_source_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class GeneratedAnswer(BaseModel):
    status: Literal["grounded", "insufficient_evidence"]
    intent: Literal[
        "company_intro",
        "product_intro",
        "product_comparison",
        "faq",
        "pet_care_general",
        "other",
    ]
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=1200)
    summary_source_ids: list[str] = Field(max_length=8)
    sections: list[GeneratedSection] = Field(default_factory=list, max_length=8)
    thinking_summary: list[ThinkingStep] = Field(default_factory=list, max_length=6)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("summary_source_ids")
    @classmethod
    def unique_summary_source_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class GeneralKnowledgeAnswer(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=1200)
    sections: list[AnswerSection] = Field(default_factory=list, max_length=8)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=4)


class QuestionTopicClassification(BaseModel):
    kind: Literal[
        "company",
        "pet_product",
        "pet_care",
        "medical",
        "third_party",
        "pet_other",
        "non_pet",
        "uncertain",
    ]
    use_prior_pet_subject: bool = False
    referenced_pet_subject: str | None = Field(default=None, max_length=40)


class AnswerPayload(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=1200)
    refusal_reason: str | None = Field(default=None, min_length=1, max_length=1600)
    sections: list[AnswerSection] = Field(default_factory=list, max_length=8)


class ResponseImage(BaseModel):
    id: str
    url: str
    alt_text: str = Field(min_length=1, max_length=500)
    caption: str = Field(default="", max_length=500)
    source_id: str | None = None
    document_id: str
    document_title: str
    document_filename: str = ""
    product_ids: list[str] = Field(default_factory=list)
    product_keys: list[str] = Field(default_factory=list)
    source_version: int | None = None
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    width: int
    height: int


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


class DecisionView(BaseModel):
    reason_code: str
    policy_version: str
    route_reason_code: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    status: ResponseStatus
    intent: str
    evidence_status: EvidenceStatus
    fallback_kind: FallbackKind | None = None
    decision: DecisionView
    answer: AnswerPayload
    images: list[ResponseImage] = Field(default_factory=list)
    thinking_summary: list[ThinkingStep]
    sources: list[SourceRef]
    confidence: float | None = None
    follow_up_questions: list[str]
    metrics: dict[str, int | float | str]


class ChunkUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    enabled: bool | None = None


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    knowledge_domain: KnowledgeDomain | None = None
    owner_company_id: str | None = Field(default=None, min_length=1, max_length=120)
    brand_id: str | None = Field(default=None, max_length=120)
    product_ids: list[str] | None = Field(default=None, max_length=100)
    visibility: Literal["public", "internal", "private"] | None = None
    publication_status: Literal["draft", "published", "withdrawn"] | None = None
    review_status: Literal["pending", "approved", "rejected"] | None = None
    reviewer_id: str | None = Field(default=None, max_length=120)
    effective_from: AwareDatetime | None = None
    effective_to: AwareDatetime | None = None
    evidence_kind: Literal["official_material", "operator_curated", "external_reference"] | None = (
        None
    )

    @model_validator(mode="after")
    def effective_window_is_ordered(self) -> "DocumentUpdate":
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from > self.effective_to
        ):
            raise ValueError("生效开始时间不能晚于结束时间")
        return self


class DocumentView(BaseModel):
    id: str
    title: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    status: str
    knowledge_domain: KnowledgeDomain
    owner_company_id: str
    brand_id: str | None = None
    product_ids: list[str] = Field(default_factory=list)
    visibility: str
    publication_status: str
    review_status: str
    reviewer_id: str | None = None
    reviewed_at: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    source_version: int
    active_version: int | None = None
    is_withdrawn: bool = False
    evidence_kind: str
    chunk_count: int = 0
    media_count: int = 0
    error_message: str | None = None
    created_at: str
    updated_at: str


class ChunkView(BaseModel):
    id: str
    document_id: str
    position: int
    content: str
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    enabled: bool
    created_at: str
    updated_at: str


class MediaAssetUpdate(BaseModel):
    caption: str | None = Field(default=None, max_length=500)
    alt_text: str | None = Field(default=None, min_length=1, max_length=500)
    product_ids: list[str] | None = Field(default=None, max_length=100)
    enabled: bool | None = None
    review_status: Literal["pending", "approved", "rejected"] | None = None


class MediaAssetView(BaseModel):
    id: str
    document_id: str
    image_key: str
    filename: str
    mime_type: str
    sha256: str
    width: int
    height: int
    source_kind: str
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    cell_ref: str | None = None
    caption: str
    alt_text: str
    product_ids: list[str] = Field(default_factory=list)
    enabled: bool
    review_status: str
    extraction_status: str = "current"
    extracted_caption: str = ""
    created_at: str
    updated_at: str
