from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Confidence = Literal["low", "medium", "high"]
ConceptRole = Literal["answer_entity", "metric", "filter", "time", "operation", "unknown"]
RowPolicy = Literal["single", "preserve_all_ties", "multiple", "unknown"]
MetricOperation = Literal["min", "max", "sum", "count", "average", "lookup", "unknown"]
DistinctPolicy = Literal["preserve", "deduplicate", "unknown"]
HandoffStatus = Literal["complete", "partial", "fallback"]
JoinPolicy = Literal["inner", "left", "preserve_left", "unknown"]


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


class AnswerColumn(BaseModel):
    name: str
    source_field: str = ""
    reason: str = ""


class AnswerContract(BaseModel):
    answer_columns: list[AnswerColumn] = Field(default_factory=list)
    row_policy: RowPolicy = "unknown"
    metric_operation: MetricOperation = "unknown"
    metric_field: str | None = None
    filters: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    metric_fields: list[str] = Field(default_factory=list)
    output_grain: str = ""
    row_source: str = ""
    join_policy: JoinPolicy = "unknown"
    enrichment_fields: list[str] = Field(default_factory=list)
    distinct_policy: DistinctPolicy = "unknown"


def get_answer_column_names(contract: AnswerContract) -> list[str]:
    return [column.name for column in contract.answer_columns]


class InspectorStep(BaseModel):
    phase: str
    prompt_type: str = ""
    tool_requests: list[dict[str, object]] = Field(default_factory=list)
    tool_results: list[dict[str, object]] = Field(default_factory=list)
    validation_error: str | None = None
    model_request_retry: dict[str, object] | None = None
    accepted_draft: dict[str, object] | None = None
    # 模型响应元数据（reasoning_content、token 用量等），与主 agent 的 trace 格式对齐
    model_response: dict[str, object] | None = None


class DataUnderstandingHandoff(BaseModel):
    brief_markdown: str
    question_grounding: QuestionGrounding = Field(default_factory=QuestionGrounding)
    data_fabric: DataFabric = Field(default_factory=DataFabric)
    answer_contract: AnswerContract = Field(default_factory=AnswerContract)
    handoff_status: HandoffStatus = "complete"
    validation_errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
