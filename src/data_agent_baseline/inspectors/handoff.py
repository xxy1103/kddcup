from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Confidence = Literal["low", "medium", "high"]
ConceptRole = Literal["answer_entity", "metric", "filter", "time", "operation", "unknown"]
RowPolicy = Literal["single", "preserve_all_ties", "multiple", "unknown"]
MetricOperation = Literal["min", "max", "sum", "count", "average", "lookup", "unknown"]


class GroundedField(BaseModel):
    asset: str
    field: str
    confidence: Confidence = "medium"
    reason: str = ""


class RejectedField(BaseModel):
    asset: str
    field: str
    reason: str


class GroundedConcept(BaseModel):
    term: str
    role: ConceptRole = "unknown"
    grounded_fields: list[GroundedField] = Field(default_factory=list)
    rejected_fields: list[RejectedField] = Field(default_factory=list)


class JoinEdge(BaseModel):
    from_field: str
    to_field: str
    confidence: Confidence = "medium"
    reason: str = ""


class JoinPath(BaseModel):
    source: str
    target: str
    path: list[JoinEdge] = Field(default_factory=list)
    confidence: Confidence = "medium"


class QuestionGrounding(BaseModel):
    concepts: list[GroundedConcept] = Field(default_factory=list)


class DataFabric(BaseModel):
    join_paths: list[JoinPath] = Field(default_factory=list)


class AnswerContract(BaseModel):
    columns: list[str] = Field(default_factory=list)
    row_policy: RowPolicy = "unknown"
    metric_operation: MetricOperation = "unknown"
    metric_field: str | None = None


class HandoffUncertainty(BaseModel):
    risk: str
    instruction: str
    evidence_refs: list[str] = Field(default_factory=list)


class SemanticNotesDraft(BaseModel):
    semantic_notes: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class DataUnderstandingHandoff(BaseModel):
    brief_markdown: str
    question_grounding: QuestionGrounding = Field(default_factory=QuestionGrounding)
    data_fabric: DataFabric = Field(default_factory=DataFabric)
    answer_contract: AnswerContract = Field(default_factory=AnswerContract)
    uncertainties: list[HandoffUncertainty] = Field(default_factory=list)
    llm_notes: list[str] = Field(default_factory=list)
    llm_notes_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
