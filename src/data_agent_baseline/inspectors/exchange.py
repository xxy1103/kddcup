from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


MessageType = Literal["perception_result", "semantic_context"]


class ResourceRef(BaseModel):
    id: str
    kind: str
    path: str
    schema_ref: str | None = None
    recommended_tools: list[str] = Field(default_factory=list)


class ArtifactRef(BaseModel):
    id: str
    kind: str
    path: str


class SemanticClaim(BaseModel):
    claim: str
    confidence: Literal["low", "medium", "high"] = "medium"
    evidence_refs: list[str] = Field(default_factory=list)


class Uncertainty(BaseModel):
    risk: str
    instruction: str
    evidence_refs: list[str] = Field(default_factory=list)


class AgentEnvelopeContent(BaseModel):
    summary: str = ""
    resources: list[ResourceRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    semantic_claims: list[SemanticClaim] = Field(default_factory=list)
    uncertainties: list[Uncertainty] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentEnvelope(BaseModel):
    protocol_version: Literal["agent-exchange/v1"] = "agent-exchange/v1"
    task_id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: AgentEnvelopeContent

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
