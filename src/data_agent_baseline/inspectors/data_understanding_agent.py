from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors.exchange import (
    AgentEnvelope,
    AgentEnvelopeContent,
    ArtifactRef,
    ResourceRef,
    SemanticClaim,
    Uncertainty,
)
from data_agent_baseline.inspectors.handoff import (
    AnswerColumn,
    AnswerContract,
    DataFabric,
    DataUnderstandingHandoff,
    GroundedConcept,
    GroundedField,
    InspectorStep,
    JoinEdge,
    JoinPath,
    QuestionGrounding,
    RejectedField,
    get_answer_column_names,
)
from data_agent_baseline.inspectors.prompts import (
    GUIDED_UNDERSTANDING_SYSTEM_PROMPT,
    build_guided_phase_prompt,
    build_guided_retry_prompt,
)
from data_agent_baseline.model_retry import invoke_model_with_retries, summarize_model_retry_events
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
from data_agent_baseline.inspectors.semantic_index import build_semantic_index
from data_agent_baseline.inspectors.semantic_query import SemanticQueryTools


@dataclass(frozen=True, slots=True)
class DataUnderstandingResult:
    perception: dict[str, Any]
    semantic_catalog: dict[str, Any]
    semantic_index: dict[str, Any]
    perception_envelope: dict[str, Any]
    semantic_context_envelope: dict[str, Any]
    data_understanding_handoff: dict[str, Any]
    summary: str
    inspector_steps: list[dict[str, Any]]
    handoff_status: str
    validation_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "perception": self.perception,
            "semantic_catalog": self.semantic_catalog,
            "semantic_index": self.semantic_index,
            "perception_envelope": self.perception_envelope,
            "semantic_context_envelope": self.semantic_context_envelope,
            "data_understanding_handoff": self.data_understanding_handoff,
            "summary": self.summary,
            "inspector_steps": self.inspector_steps,
            "handoff_status": self.handoff_status,
            "validation_errors": self.validation_errors,
        }


class ToolRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class OverviewDraft(BaseModel):
    task_intent: str = ""
    concepts: list[dict[str, str]] = Field(default_factory=list)
    ambiguity_targets: list[str] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)


class AcceptedFieldDraft(BaseModel):
    field_ref: str
    confidence: str = "medium"
    reason: str = ""


class RejectedFieldDraft(BaseModel):
    field_ref: str
    reason: str = ""


class GroundedConceptDraft(BaseModel):
    term: str
    role: str = "unknown"
    accepted_fields: list[AcceptedFieldDraft] = Field(default_factory=list)
    rejected_fields: list[RejectedFieldDraft] = Field(default_factory=list)


class GroundingDraft(BaseModel):
    grounded_concepts: list[GroundedConceptDraft] = Field(default_factory=list)
    remaining_uncertainties: list[str] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)


class JoinEdgeDraft(BaseModel):
    from_field: str
    to_field: str


class JoinPathDraft(BaseModel):
    purpose: str = ""
    path: list[JoinEdgeDraft] = Field(default_factory=list)
    confidence: str = "medium"


class FabricDraft(BaseModel):
    join_paths: list[JoinPathDraft] = Field(default_factory=list)
    data_grain: str = ""
    relationship_risks: list[str] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)


class AnswerColumnDraft(BaseModel):
    name: str
    source_field: str = ""
    reason: str = ""


class ContractDraft(BaseModel):
    answer_columns: list[AnswerColumnDraft] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    metric_operation: str = "unknown"
    metric_fields: list[str] = Field(default_factory=list)
    row_policy: str = "unknown"
    distinct_policy: str = "unknown"
    output_grain: str = ""
    row_source: str = ""
    join_policy: str = "unknown"
    enrichment_fields: list[str] = Field(default_factory=list)
    remaining_uncertainties: list[str] = Field(default_factory=list)


class RepairDraft(BaseModel):
    grounded_concepts: list[GroundedConceptDraft] = Field(default_factory=list)
    join_paths: list[JoinPathDraft] = Field(default_factory=list)
    contract_patch: ContractDraft | None = None
    remaining_uncertainties: list[str] = Field(default_factory=list)


class DataUnderstandingAgent:
    def __init__(self, *, model: Any | None, config: DataInspectorConfig) -> None:
        self.model = model
        self.config = config

    def run(self, task: PublicTask, perception_envelope: AgentEnvelope) -> DataUnderstandingResult:
        perception_payload = dict(perception_envelope.content.payload)
        catalog = build_semantic_catalog(task, budget=self.config.sample_budget)
        semantic_index = build_semantic_index(
            question=task.question,
            catalog=catalog,
            perception_payload=perception_payload,
        )
        query_tools = SemanticQueryTools(
            catalog=catalog,
            semantic_index=semantic_index,
            limit=self.config.context_bundle_limit,
            max_join_hops=self.config.max_join_hops,
        )
        context_bundle = (
            query_tools.build_context_bundle(task.question)
            if self.config.enable_semantic_tools
            else {
                "query": task.question,
                "field_candidates": catalog.get("query_relevance", {}).get("relevant_fields", []),
                "join_paths": catalog.get("relationships", []),
                "knowledge_hits": [],
                "rejected_field_hints": [],
                "risk_candidates": semantic_index.get("risk_index", {}),
            }
        )
        field_whitelist = sorted(query_tools.known_field_refs()) if self.config.enable_semantic_tools else []
        deterministic_handoff = _build_handoff(
            task=task,
            perception_payload=perception_payload,
            context_bundle=context_bundle,
        )
        handoff = deterministic_handoff
        inspector_steps: list[dict[str, Any]] = [
            InspectorStep(
                phase="deterministic_context",
                prompt_type="rules",
                accepted_draft={
                    "asset_count": len(catalog.get("assets", [])),
                    "schema_count": len(catalog.get("schemas", [])),
                    "field_candidate_count": len(context_bundle.get("field_candidates", [])),
                    "join_path_count": len(context_bundle.get("join_paths", [])),
                },
            ).model_dump(mode="json")
        ]
        validation_errors: list[str] = []
        if self.config.mode == "hybrid" and self.model is not None and self.config.max_agent_steps > 0:
            guided_result = GuidedDataUnderstandingLoop(
                model=self.model,
                query_tools=query_tools,
                max_steps=self.config.max_agent_steps,
                max_phase_retries=self.config.max_phase_retries,
            ).run(
                task=task,
                perception_payload=perception_payload,
                context_bundle=context_bundle,
                field_whitelist=field_whitelist,
                deterministic_handoff=deterministic_handoff,
            )
            handoff = guided_result.handoff
            inspector_steps.extend(guided_result.steps)
            validation_errors = guided_result.validation_errors
        else:
            validation_errors = _validate_handoff_quality(task.question, deterministic_handoff, context_bundle)
            if validation_errors:
                handoff = _mark_handoff_status(deterministic_handoff, "partial", validation_errors)

        envelope = self._build_semantic_context_envelope(
            task=task,
            catalog=catalog,
            semantic_index=semantic_index,
            perception_envelope=perception_envelope,
            handoff=handoff,
        )
        return DataUnderstandingResult(
            perception=perception_payload,
            semantic_catalog=catalog,
            semantic_index=semantic_index,
            perception_envelope=perception_envelope.to_dict(),
            semantic_context_envelope=envelope.to_dict(),
            data_understanding_handoff=handoff.to_dict(),
            summary=envelope.content.summary,
            inspector_steps=inspector_steps,
            handoff_status=handoff.handoff_status,
            validation_errors=handoff.validation_errors,
        )

    def _build_semantic_context_envelope(
        self,
        *,
        task: PublicTask,
        catalog: dict[str, Any],
        semantic_index: dict[str, Any],
        perception_envelope: AgentEnvelope,
        handoff: DataUnderstandingHandoff,
    ) -> AgentEnvelope:
        resources = [
            ResourceRef(
                id=f"res_{index}",
                kind=str(asset.get("kind", "file")),
                path=str(asset.get("path", "")),
                schema_ref=f"schema:{asset.get('path')}",
                recommended_tools=list(asset.get("recommended_tools", [])),
            )
            for index, asset in enumerate(catalog.get("assets", []))
        ]
        artifacts = [
            ArtifactRef(id="perception", kind="json", path="perception.json"),
            ArtifactRef(id="semantic_catalog", kind="json", path="semantic_catalog.json"),
            ArtifactRef(id="semantic_index", kind="json", path="semantic_index.json"),
            ArtifactRef(id="data_understanding_handoff", kind="json", path="data_understanding_handoff.json"),
        ]
        claims = list(perception_envelope.content.semantic_claims)
        uncertainties = list(perception_envelope.content.uncertainties)
        claims.extend(_claims_from_relationships(catalog))
        uncertainties.extend(_uncertainties_from_catalog(catalog))

        summary = handoff.brief_markdown
        return AgentEnvelope(
            task_id=task.task_id,
            sender="DataUnderstandingAgent",
            recipient="MainSolvingAgent",
            message_type="semantic_context",
            content=AgentEnvelopeContent(
                summary=summary,
                resources=resources,
                artifacts=artifacts,
                semantic_claims=claims,
                uncertainties=uncertainties,
                payload={
                    "catalog_asset_count": len(catalog.get("assets", [])),
                    "catalog_schema_count": len(catalog.get("schemas", [])),
                    "answer_contract": handoff.answer_contract.model_dump(mode="json"),
                    "handoff": handoff.to_dict(),
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class GuidedLoopResult:
    handoff: DataUnderstandingHandoff
    steps: list[dict[str, Any]]
    validation_errors: list[str]


class GuidedDataUnderstandingLoop:
    def __init__(
        self,
        *,
        model: Any,
        query_tools: SemanticQueryTools,
        max_steps: int,
        max_phase_retries: int,
    ) -> None:
        self.model = model
        self.query_tools = query_tools
        self.max_steps = max(0, max_steps)
        self.max_phase_retries = max(0, max_phase_retries)
        self._step_count = 0

    def run(
        self,
        *,
        task: PublicTask,
        perception_payload: dict[str, Any],
        context_bundle: dict[str, Any],
        field_whitelist: list[str],
        deterministic_handoff: DataUnderstandingHandoff,
    ) -> GuidedLoopResult:
        steps: list[dict[str, Any]] = []
        tool_observations: list[dict[str, Any]] = []
        working_memory: dict[str, Any] = {
            "deterministic_handoff": deterministic_handoff.to_dict(),
            "overview": None,
            "grounding": None,
            "fabric": None,
            "contract": None,
        }
        errors: list[str] = []
        accepted_grounding: GroundingDraft | None = None
        accepted_fabric: FabricDraft | None = None
        accepted_contract: ContractDraft | None = None

        try:
            overview = self._run_phase_with_tool_refinement(
                phase="overview",
                draft_model=OverviewDraft,
                task=task,
                perception_payload=perception_payload,
                context_bundle=context_bundle,
                field_whitelist=field_whitelist,
                working_memory=working_memory,
                tool_observations=tool_observations,
                steps=steps,
            )
            working_memory["overview"] = overview.model_dump(mode="json")

            grounding = self._run_phase_with_tool_refinement(
                phase="grounding",
                draft_model=GroundingDraft,
                task=task,
                perception_payload=perception_payload,
                context_bundle=context_bundle,
                field_whitelist=field_whitelist,
                working_memory=working_memory,
                tool_observations=tool_observations,
                steps=steps,
            )
            _validate_grounding_draft(grounding, field_whitelist)
            accepted_grounding = grounding
            working_memory["grounding"] = grounding.model_dump(mode="json")

            fabric = self._run_phase_with_tool_refinement(
                phase="fabric",
                draft_model=FabricDraft,
                task=task,
                perception_payload=perception_payload,
                context_bundle=context_bundle,
                field_whitelist=field_whitelist,
                working_memory=working_memory,
                tool_observations=tool_observations,
                steps=steps,
            )
            _validate_fabric_draft(fabric, field_whitelist)
            accepted_fabric = fabric
            working_memory["fabric"] = fabric.model_dump(mode="json")

            contract = self._run_phase(
                phase="contract",
                draft_model=ContractDraft,
                task=task,
                perception_payload=perception_payload,
                context_bundle=context_bundle,
                field_whitelist=field_whitelist,
                working_memory=working_memory,
                tool_observations=tool_observations,
                steps=steps,
                draft_validator=lambda draft: _validate_contract_draft(
                    cast(ContractDraft, draft),
                    grounding,
                    field_whitelist,
                ),
            )
            accepted_contract = contract
            working_memory["contract"] = contract.model_dump(mode="json")

            handoff = _build_guided_handoff(
                deterministic_handoff=deterministic_handoff,
                grounding=grounding,
                fabric=fabric,
                contract=contract,
            )
            validation_errors = _validate_handoff_quality(task.question, handoff, context_bundle)
            if validation_errors and self._can_call_model():
                repair = self._run_phase(
                    phase="repair_or_critique",
                    draft_model=RepairDraft,
                    task=task,
                    perception_payload=perception_payload,
                    context_bundle=context_bundle,
                    field_whitelist=field_whitelist,
                    working_memory={**working_memory, "current_handoff": handoff.to_dict()},
                    tool_observations=tool_observations,
                    steps=steps,
                    validation_errors=validation_errors,
                )
                grounding, fabric, contract = _apply_repair_draft(grounding, fabric, contract, repair, field_whitelist)
                handoff = _build_guided_handoff(
                    deterministic_handoff=deterministic_handoff,
                    grounding=grounding,
                    fabric=fabric,
                    contract=contract,
                )
                validation_errors = _validate_handoff_quality(task.question, handoff, context_bundle)

            status = "complete" if not validation_errors else "partial"
            handoff = _mark_handoff_status(handoff, status, validation_errors)
            return GuidedLoopResult(
                handoff=handoff,
                steps=steps,
                validation_errors=validation_errors,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            fallback_source = _build_partial_guided_handoff(
                deterministic_handoff=deterministic_handoff,
                grounding=accepted_grounding,
                fabric=accepted_fabric,
                contract=accepted_contract,
            )
            fallback_errors = [*errors, *_validate_handoff_quality(task.question, fallback_source, context_bundle)]
            fallback = _mark_handoff_status(
                fallback_source,
                "fallback",
                fallback_errors,
            )
            steps.append(
                InspectorStep(
                    phase="fallback",
                    prompt_type="error",
                    validation_error=str(exc),
                    accepted_draft={"handoff_status": "fallback"},
                ).model_dump(mode="json")
            )
            return GuidedLoopResult(
                handoff=fallback,
                steps=steps,
                validation_errors=fallback_errors,
            )

    def _run_phase_with_tool_refinement(
        self,
        *,
        phase: str,
        draft_model: type[BaseModel],
        task: PublicTask,
        perception_payload: dict[str, Any],
        context_bundle: dict[str, Any],
        field_whitelist: list[str],
        working_memory: dict[str, Any],
        tool_observations: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        validation_errors: list[str] | None = None,
    ) -> BaseModel:
        draft = self._run_phase(
            phase=phase,
            draft_model=draft_model,
            task=task,
            perception_payload=perception_payload,
            context_bundle=context_bundle,
            field_whitelist=field_whitelist,
            working_memory=working_memory,
            tool_observations=tool_observations,
            steps=steps,
            validation_errors=validation_errors,
            prompt_type="phase_probe",
        )
        requests = getattr(draft, "tool_requests", [])
        if not requests:
            if steps and steps[-1].get("phase") == phase and steps[-1].get("prompt_type") == "phase_probe":
                steps[-1]["prompt_type"] = "phase"
            return draft

        new_observations = self._execute_tool_requests(requests, steps, phase)
        tool_observations.extend(new_observations)
        if not new_observations or not self._can_refine_phase(phase):
            return draft

        refined_memory = {**working_memory, f"{phase}_probe": draft.model_dump(mode="json")}
        return self._run_phase(
            phase=phase,
            draft_model=draft_model,
            task=task,
            perception_payload=perception_payload,
            context_bundle=context_bundle,
            field_whitelist=field_whitelist,
            working_memory=refined_memory,
            tool_observations=tool_observations,
            steps=steps,
            validation_errors=validation_errors,
            prompt_type="phase",
        )

    def _can_refine_phase(self, phase: str) -> bool:
        required_later_calls = {
            "overview": 3,  # grounding, fabric, contract
            "grounding": 2,  # fabric, contract
            "fabric": 1,  # contract
        }.get(phase, 0)
        return (self.max_steps - (self._step_count + 1)) >= required_later_calls

    def _run_phase(
        self,
        *,
        phase: str,
        draft_model: type[BaseModel],
        task: PublicTask,
        perception_payload: dict[str, Any],
        context_bundle: dict[str, Any],
        field_whitelist: list[str],
        working_memory: dict[str, Any],
        tool_observations: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        validation_errors: list[str] | None = None,
        prompt_type: str = "phase",
        draft_validator: Callable[[BaseModel], None] | None = None,
    ) -> BaseModel:
        errors: list[str] = []
        attempts = self.max_phase_retries + 1
        for attempt in range(attempts):
            if not self._can_call_model():
                raise RuntimeError(f"Inspector step budget exhausted before `{phase}`.")
            prompt = (
                build_guided_phase_prompt(
                    phase=phase,
                    question=task.question,
                    perception_payload=perception_payload,
                    context_bundle=context_bundle,
                    allowed_field_refs=field_whitelist,
                    working_memory=working_memory,
                    tool_observations=tool_observations,
                    validation_errors=validation_errors,
                    phase_mode="probe" if prompt_type == "phase_probe" else "final",
                )
                if attempt == 0
                else build_guided_retry_prompt(
                    phase=phase,
                    previous_error=errors[-1],
                    question=task.question,
                    allowed_field_refs=field_whitelist,
                    working_memory=working_memory,
                    tool_observations=tool_observations,
                )
            )
            self._step_count += 1
            request_retry_events: list[dict[str, Any]] = []
            try:
                draft = _invoke_guided_phase(
                    model=self.model,
                    prompt=prompt,
                    draft_model=draft_model,
                    request_retry_events=request_retry_events,
                )
                _validate_draft_field_refs(draft, field_whitelist)
                if draft_validator is not None:
                    draft_validator(draft)
                model_request_retry = summarize_model_retry_events(request_retry_events, succeeded=True)
                steps.append(
                    InspectorStep(
                        phase=phase,
                        prompt_type="retry" if attempt else prompt_type,
                        tool_requests=_tool_requests_from_draft(draft),
                        accepted_draft=draft.model_dump(mode="json"),
                        model_request_retry=model_request_retry,
                    ).model_dump(mode="json")
                )
                return draft
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                errors.append(str(exc))
                model_request_retry = summarize_model_retry_events(request_retry_events, succeeded=True)
                steps.append(
                    InspectorStep(
                        phase=phase,
                        prompt_type="retry" if attempt else prompt_type,
                        validation_error=str(exc),
                        model_request_retry=model_request_retry,
                    ).model_dump(mode="json")
                )
            except Exception as exc:
                model_request_retry = summarize_model_retry_events(request_retry_events, succeeded=False)
                if model_request_retry is not None:
                    steps.append(
                        InspectorStep(
                            phase=phase,
                            prompt_type="retry" if attempt else prompt_type,
                            validation_error=str(exc),
                            model_request_retry=model_request_retry,
                        ).model_dump(mode="json")
                    )
                raise
        raise ValueError(f"`{phase}` failed validation: {' | '.join(errors)}")

    def _can_call_model(self) -> bool:
        return self._step_count < self.max_steps

    def _execute_tool_requests(
        self,
        requests: list[ToolRequest],
        steps: list[dict[str, Any]],
        phase: str,
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for request in requests[:4]:
            result = _execute_semantic_tool(self.query_tools, request)
            observations.append(result)
        if observations:
            steps.append(
                InspectorStep(
                    phase=f"{phase}_tools",
                    prompt_type="tool",
                    tool_requests=[request.model_dump(mode="json") for request in requests[:4]],
                    tool_results=observations,
                ).model_dump(mode="json")
            )
        return observations


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _invoke_guided_phase(
    *,
    model: Any,
    prompt: str,
    draft_model: type[BaseModel],
    request_retry_events: list[dict[str, Any]] | None = None,
) -> BaseModel:
    def record_request_retry(event: dict[str, Any]) -> None:
        if request_retry_events is not None:
            request_retry_events.append(dict(event))

    response = invoke_model_with_retries(
        model,
        [
            SystemMessage(content=GUIDED_UNDERSTANDING_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ],
        on_retry_event=record_request_retry,
    )
    text = _message_text(getattr(response, "content", response))
    payload = _extract_json_object(text)
    return draft_model.model_validate(payload)


def _tool_requests_from_draft(draft: BaseModel) -> list[dict[str, object]]:
    requests = getattr(draft, "tool_requests", [])
    return [request.model_dump(mode="json") for request in requests]


def _execute_semantic_tool(query_tools: SemanticQueryTools, request: ToolRequest) -> dict[str, Any]:
    name = request.tool
    args = request.args
    try:
        if name == "search_semantic_index":
            result = query_tools.search_semantic_index(str(args.get("query", "")))
        elif name == "lookup_knowledge":
            result = query_tools.lookup_knowledge(str(args.get("term") or args.get("query") or ""))
        elif name == "get_asset_schema":
            result = query_tools.get_asset_schema(str(args.get("asset_path", "")), include_samples=True)
        elif name == "find_join_paths":
            result = query_tools.find_join_paths(
                str(args.get("source", "")),
                str(args.get("target", "")),
            )
        else:
            return {"ok": False, "tool": name, "error": f"Unsupported semantic tool: {name}"}
        return {"ok": True, "tool": name, "args": args, "content": _compact_tool_result(result)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "tool": name, "args": args, "error": str(exc)}


def _compact_tool_result(result: Any) -> Any:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= 4000:
        return result
    return {"preview": text[:4000], "truncated": True}


def _validate_draft_field_refs(draft: BaseModel, field_whitelist: list[str]) -> None:
    if not field_whitelist:
        return
    allowed = set(field_whitelist)
    payload = draft.model_dump(mode="json")
    refs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"field_ref", "from_field", "to_field", "source_field"} and isinstance(child, str):
                    refs.append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    unknown = [ref for ref in refs if ref and ref not in allowed]
    if unknown:
        raise ValueError(_format_unknown_field_refs_error("Draft referenced fields outside whitelist", unknown, field_whitelist))


def _validate_grounding_draft(draft: GroundingDraft, field_whitelist: list[str]) -> None:
    _validate_draft_field_refs(draft, field_whitelist)
    for concept in draft.grounded_concepts:
        accepted = {item.field_ref for item in concept.accepted_fields}
        rejected = {item.field_ref for item in concept.rejected_fields}
        conflict = accepted & rejected
        if conflict:
            raise ValueError(f"Accepted and rejected fields conflict for `{concept.term}`: {sorted(conflict)}")


def _validate_fabric_draft(draft: FabricDraft, field_whitelist: list[str]) -> None:
    _validate_draft_field_refs(draft, field_whitelist)


def _validate_contract_draft(
    draft: ContractDraft,
    grounding: GroundingDraft,
    field_whitelist: list[str],
) -> None:
    _validate_draft_field_refs(draft, field_whitelist)
    _validate_answer_column_names(draft.answer_columns)
    _validate_contract_field_refs(draft, field_whitelist)
    accepted = {
        accepted.field_ref
        for concept in grounding.grounded_concepts
        for accepted in concept.accepted_fields
    }
    rejected = {
        rejected.field_ref
        for concept in grounding.grounded_concepts
        for rejected in concept.rejected_fields
    } - accepted
    used = (
        set(draft.group_by)
        | set(draft.metric_fields)
        | set(draft.enrichment_fields)
        | {column.source_field for column in draft.answer_columns if column.source_field}
    )
    if draft.row_source in field_whitelist:
        used.add(draft.row_source)
    for filter_text in draft.filters:
        used.update(ref for ref in field_whitelist if ref in filter_text)
    conflict = used & rejected
    if conflict:
        raise ValueError(f"Contract uses rejected fields: {sorted(conflict)}")


def _validate_answer_column_names(answer_columns: list[AnswerColumnDraft]) -> None:
    invalid = [column.name for column in answer_columns if _looks_like_data_reference(column.name)]
    if invalid:
        raise ValueError(
            "answer_columns[].name must be final answer headers, not source field references: "
            f"{invalid[:5]}"
        )


def _validate_contract_field_refs(draft: ContractDraft, field_whitelist: list[str]) -> None:
    if not field_whitelist:
        return
    allowed_fields = set(field_whitelist)
    allowed_row_sources = _allowed_row_sources(field_whitelist)
    refs = [
        *draft.group_by,
        *draft.metric_fields,
        *draft.enrichment_fields,
        *(column.source_field for column in draft.answer_columns if column.source_field),
    ]
    unknown = [ref for ref in refs if ref and ref not in allowed_fields]
    if unknown:
        raise ValueError(_format_unknown_field_refs_error("Contract referenced fields outside whitelist", unknown, field_whitelist))

    row_source = draft.row_source.strip()
    if row_source and _looks_like_data_reference(row_source):
        if row_source not in allowed_fields and row_source not in allowed_row_sources:
            raise ValueError(f"Contract row_source is outside whitelist: {row_source}")
    bad_filter_refs = [
        filter_text
        for filter_text in draft.filters
        if _looks_like_data_reference(filter_text) and not any(ref in filter_text for ref in field_whitelist)
    ]
    if bad_filter_refs:
        raise ValueError(f"Contract filters must contain exact allowed field references: {bad_filter_refs[:3]}")


def _allowed_row_sources(field_refs: list[str]) -> set[str]:
    allowed: set[str] = set()
    for ref in field_refs:
        asset = _asset_ref_from_field_ref(ref)
        allowed.add(asset)
        suffix = ref[len(asset) + 1 :] if ref.startswith(f"{asset}.") else ""
        if "." in suffix:
            allowed.add(f"{asset}.{suffix.rsplit('.', 1)[0]}")
    return allowed


def _format_unknown_field_refs_error(message: str, unknown_refs: list[str], field_whitelist: list[str]) -> str:
    unique_unknown = list(dict.fromkeys(unknown_refs))
    suggestions = _field_ref_repair_suggestions(unique_unknown, field_whitelist)
    if not suggestions:
        return f"{message}: {unique_unknown[:5]}"
    return f"{message}: {unique_unknown[:5]}; repair_hints: {suggestions[:5]}"


def _field_ref_repair_suggestions(unknown_refs: list[str], field_whitelist: list[str]) -> list[str]:
    suggestions: list[str] = []
    for unknown_ref in unknown_refs:
        unknown_field = _field_suffix(unknown_ref)
        if not unknown_field:
            continue
        exact_field_matches = [ref for ref in field_whitelist if _field_suffix(ref).lower() == unknown_field.lower()]
        leaf_matches = [
            ref
            for ref in field_whitelist
            if _field_leaf(ref).lower() == _field_leaf(unknown_ref).lower() and ref not in exact_field_matches
        ]
        candidates = exact_field_matches + leaf_matches[:3]
        if candidates:
            suggestions.append(
                f"{unknown_ref} is not available; use one of {candidates[:4]} if it matches the requested concept, "
                "and join from the filter table when needed."
            )
    return suggestions


def _field_suffix(field_ref: str) -> str:
    parts = field_ref.split(".", 2)
    return parts[2] if len(parts) == 3 else ""


def _field_leaf(field_ref: str) -> str:
    suffix = _field_suffix(field_ref) or field_ref
    return suffix.rsplit(".", 1)[-1]


def _looks_like_data_reference(value: str) -> bool:
    return bool(re.search(r"\b(?:csv|json|db|doc)/", value))


def _apply_repair_draft(
    grounding: GroundingDraft,
    fabric: FabricDraft,
    contract: ContractDraft,
    repair: RepairDraft,
    field_whitelist: list[str],
) -> tuple[GroundingDraft, FabricDraft, ContractDraft]:
    _validate_draft_field_refs(repair, field_whitelist)
    if repair.grounded_concepts:
        grounding = GroundingDraft(
            grounded_concepts=repair.grounded_concepts,
            remaining_uncertainties=repair.remaining_uncertainties,
        )
        _validate_grounding_draft(grounding, field_whitelist)
    if repair.join_paths:
        fabric = FabricDraft(
            join_paths=repair.join_paths,
            data_grain=fabric.data_grain,
            relationship_risks=fabric.relationship_risks,
        )
        _validate_fabric_draft(fabric, field_whitelist)
    if repair.contract_patch is not None:
        contract = repair.contract_patch
        _validate_contract_draft(contract, grounding, field_whitelist)
    return grounding, fabric, contract


def _claims_from_relationships(catalog: dict[str, Any]) -> list[SemanticClaim]:
    claims: list[SemanticClaim] = []
    for relationship in catalog.get("relationships", [])[:10]:
        field = relationship.get("field")
        locations = relationship.get("locations", [])
        claims.append(
            SemanticClaim(
                claim=f"Field `{field}` is a possible join key across {len(locations)} assets.",
                confidence=str(relationship.get("confidence", "medium")),
                evidence_refs=[
                    f"field:{item.get('asset_path')}:{item.get('field')}" for item in locations
                ],
            )
        )
    return claims


def _uncertainties_from_catalog(catalog: dict[str, Any]) -> list[Uncertainty]:
    return [
        Uncertainty(
            risk=str(item.get("risk", "catalog_uncertainty")),
            instruction=str(item.get("instruction", item)),
            evidence_refs=[str(item.get("asset_path"))] if item.get("asset_path") else [],
        )
        for item in catalog.get("semantic_uncertainties", [])
    ]


def _asset_ref_from_field_ref(field_ref: str) -> str:
    for extension in (".csv", ".json", ".db", ".md", ".txt"):
        marker = field_ref.find(extension)
        if marker >= 0:
            return field_ref[: marker + len(extension)]
    return field_ref.rsplit(".", 1)[0]


def _build_handoff(
    *,
    task: PublicTask,
    perception_payload: dict[str, Any],
    context_bundle: dict[str, Any],
) -> DataUnderstandingHandoff:
    concepts = _build_grounded_concepts(task.question, context_bundle)
    join_paths = _build_join_paths(context_bundle)
    answer_contract = _build_answer_contract(task.question, concepts)
    handoff = DataUnderstandingHandoff(
        brief_markdown="",
        question_grounding=QuestionGrounding(concepts=concepts),
        data_fabric=DataFabric(join_paths=join_paths),
        answer_contract=answer_contract,
    )
    return handoff.model_copy(update={"brief_markdown": _render_handoff_brief(handoff)})


def _build_guided_handoff(
    *,
    deterministic_handoff: DataUnderstandingHandoff,
    grounding: GroundingDraft,
    fabric: FabricDraft,
    contract: ContractDraft,
) -> DataUnderstandingHandoff:
    concepts = [_concept_from_guided_draft(item) for item in grounding.grounded_concepts]
    if not concepts:
        concepts = deterministic_handoff.question_grounding.concepts

    join_paths = [_join_path_from_guided_draft(item) for item in fabric.join_paths if item.path]
    if not join_paths:
        join_paths = deterministic_handoff.data_fabric.join_paths

    metric_fields = list(contract.metric_fields)
    metric_field = metric_fields[0] if metric_fields else deterministic_handoff.answer_contract.metric_field
    answer_columns = [
        AnswerColumn(name=column.name, source_field=column.source_field, reason=column.reason)
        for column in contract.answer_columns
    ] or deterministic_handoff.answer_contract.answer_columns
    answer_contract = AnswerContract(
        answer_columns=answer_columns,
        row_policy=_coerce_row_policy(contract.row_policy, deterministic_handoff.answer_contract.row_policy),
        metric_operation=_coerce_metric_operation(
            contract.metric_operation,
            deterministic_handoff.answer_contract.metric_operation,
        ),
        metric_field=metric_field,
        filters=list(contract.filters),
        group_by=list(contract.group_by),
        metric_fields=metric_fields or ([metric_field] if metric_field else []),
        output_grain=contract.output_grain or fabric.data_grain,
        row_source=contract.row_source,
        join_policy=_coerce_join_policy(contract.join_policy),
        enrichment_fields=list(contract.enrichment_fields),
        distinct_policy=_coerce_distinct_policy(contract.distinct_policy),
    )
    handoff = DataUnderstandingHandoff(
        brief_markdown="",
        question_grounding=QuestionGrounding(concepts=concepts),
        data_fabric=DataFabric(join_paths=join_paths),
        answer_contract=answer_contract,
    )
    return handoff.model_copy(update={"brief_markdown": _render_handoff_brief(handoff)})


def _build_partial_guided_handoff(
    *,
    deterministic_handoff: DataUnderstandingHandoff,
    grounding: GroundingDraft | None,
    fabric: FabricDraft | None,
    contract: ContractDraft | None,
) -> DataUnderstandingHandoff:
    if contract is not None:
        return _build_guided_handoff(
            deterministic_handoff=deterministic_handoff,
            grounding=grounding or GroundingDraft(),
            fabric=fabric or FabricDraft(),
            contract=contract,
        )
    concepts = (
        [_concept_from_guided_draft(item) for item in grounding.grounded_concepts]
        if grounding is not None
        else deterministic_handoff.question_grounding.concepts
    )
    join_paths = (
        [_join_path_from_guided_draft(item) for item in fabric.join_paths if item.path]
        if fabric is not None
        else deterministic_handoff.data_fabric.join_paths
    )
    handoff = DataUnderstandingHandoff(
        brief_markdown="",
        question_grounding=QuestionGrounding(concepts=concepts),
        data_fabric=DataFabric(join_paths=join_paths),
        answer_contract=AnswerContract(),
    )
    return handoff.model_copy(update={"brief_markdown": _render_handoff_brief(handoff)})


def _concept_from_guided_draft(draft: GroundedConceptDraft) -> GroundedConcept:
    return GroundedConcept(
        term=draft.term,
        role=_coerce_concept_role(draft.role),
        grounded_fields=[
            GroundedField(
                asset=_asset_ref_from_field_ref(item.field_ref),
                field=_field_name_from_ref(item.field_ref),
                confidence=_coerce_confidence(item.confidence),
                reason=item.reason,
            )
            for item in draft.accepted_fields
        ],
        rejected_fields=[
            RejectedField(
                asset=_asset_ref_from_field_ref(item.field_ref),
                field=_field_name_from_ref(item.field_ref),
                reason=item.reason,
            )
            for item in draft.rejected_fields
        ],
    )


def _join_path_from_guided_draft(draft: JoinPathDraft) -> JoinPath:
    edges = [
        JoinEdge(
            from_field=edge.from_field,
            to_field=edge.to_field,
            confidence=_coerce_confidence(draft.confidence),
            reason=draft.purpose,
        )
        for edge in draft.path
    ]
    return JoinPath(
        source=edges[0].from_field if edges else "",
        target=edges[-1].to_field if edges else "",
        path=edges,
        confidence=_coerce_confidence(draft.confidence),
    )


def _field_name_from_ref(field_ref: str) -> str:
    asset = _asset_ref_from_field_ref(field_ref)
    prefix = f"{asset}."
    return field_ref[len(prefix) :] if field_ref.startswith(prefix) else field_ref


def _coerce_confidence(value: str) -> str:
    return value if value in {"low", "medium", "high"} else "medium"


def _coerce_concept_role(value: str) -> str:
    allowed = {"answer_entity", "metric", "filter", "time", "operation", "unknown"}
    return value if value in allowed else "unknown"


def _coerce_row_policy(value: str, fallback: str) -> str:
    allowed = {"single", "preserve_all_ties", "multiple", "unknown"}
    return value if value in allowed else fallback if fallback in allowed else "unknown"


def _coerce_metric_operation(value: str, fallback: str) -> str:
    allowed = {"min", "max", "sum", "count", "average", "lookup", "unknown"}
    return value if value in allowed else fallback if fallback in allowed else "unknown"


def _coerce_distinct_policy(value: str) -> str:
    return value if value in {"preserve", "deduplicate", "unknown"} else "unknown"


def _coerce_join_policy(value: str) -> str:
    return value if value in {"inner", "left", "preserve_left", "unknown"} else "unknown"


def _validate_handoff_quality(
    question: str,
    handoff: DataUnderstandingHandoff,
    context_bundle: dict[str, Any],
) -> list[str]:
    tokens = set(_tokenize(question))
    errors: list[str] = []
    has_candidates = bool(context_bundle.get("field_candidates"))
    if has_candidates and not handoff.question_grounding.concepts:
        errors.append("question_grounding.concepts is empty despite available field candidates.")
    if _question_expects_columns(question) and not handoff.answer_contract.answer_columns and has_candidates:
        errors.append("answer_contract.answer_columns is empty despite an explicit requested output.")
    invalid_column_names = [
        column.name for column in handoff.answer_contract.answer_columns if _looks_like_data_reference(column.name)
    ]
    if invalid_column_names:
        errors.append(f"answer_contract.answer_columns contains source refs as final column names: {invalid_column_names[:5]}")
    if tokens & {"total", "sum", "count", "average", "avg", "lowest", "highest", "minimum", "maximum"}:
        if handoff.answer_contract.metric_operation == "unknown":
            errors.append("answer_contract.metric_operation is unknown despite an aggregate or extreme-value question.")
    if tokens & {"total", "sum", "average", "avg", "lowest", "highest", "minimum", "maximum"}:
        metric_fields = handoff.answer_contract.metric_fields or (
            [handoff.answer_contract.metric_field] if handoff.answer_contract.metric_field else []
        )
        if not metric_fields:
            errors.append("answer_contract.metric_fields is empty despite a metric question.")
    if _question_has_filter(question) and not handoff.answer_contract.filters:
        if not any(concept.role in {"filter", "time"} for concept in handoff.question_grounding.concepts):
            errors.append("No filters or filter grounding found despite apparent filter terms in the question.")
    if handoff.answer_contract.metric_fields or handoff.answer_contract.filters:
        if not handoff.answer_contract.row_source:
            errors.append("answer_contract.row_source is empty despite metric or filter constraints.")
    if handoff.data_fabric.join_paths and handoff.answer_contract.join_policy == "unknown":
        errors.append("answer_contract.join_policy is unknown despite available join paths.")
    return errors


def _question_expects_columns(question: str) -> bool:
    tokens = set(_tokenize(question))
    return bool(tokens & {"which", "what", "identify", "list", "show", "give", "find", "country", "countries", "time"})


def _question_has_filter(question: str) -> bool:
    lowered = question.lower()
    if re.search(r"'[^']+'|\"[^\"]+\"|\b\d{4}\b", question):
        return True
    return any(term in lowered for term in ["approved", "ranked", "second", "during", "for ", "in june", "status"])


def _mark_handoff_status(
    handoff: DataUnderstandingHandoff,
    status: str,
    validation_errors: list[str],
) -> DataUnderstandingHandoff:
    updated = handoff.model_copy(
        update={
            "handoff_status": status,
            "validation_errors": validation_errors,
        }
    )
    return updated.model_copy(update={"brief_markdown": _render_handoff_brief(updated)})


def _build_grounded_concepts(question: str, context_bundle: dict[str, Any]) -> list[GroundedConcept]:
    question_tokens = set(_tokenize(question))
    field_candidates = context_bundle.get("field_candidates", [])
    rejected_hints = context_bundle.get("rejected_field_hints", [])
    concepts: list[GroundedConcept] = []

    if "event" in question_tokens:
        concepts.append(
            GroundedConcept(
                term="event",
                role="answer_entity",
                grounded_fields=_grounded_fields_for_terms(field_candidates, ["event_name", "event_id", "event"]),
            )
        )
    if "cost" in question_tokens:
        concepts.append(
            GroundedConcept(
                term="cost",
                role="metric",
                grounded_fields=_grounded_fields_for_terms(field_candidates, ["cost"]),
                rejected_fields=[
                    RejectedField(
                        asset=str(item.get("asset", "")),
                        field=str(item.get("field", "")),
                        reason=str(item.get("reason", "")),
                    )
                    for item in rejected_hints
                ],
            )
        )
    if question_tokens & {"lowest", "minimum", "min"}:
        concepts.append(
            GroundedConcept(
                term="lowest",
                role="operation",
                grounded_fields=[],
            )
        )
    if question_tokens & {"highest", "maximum", "max"}:
        concepts.append(
            GroundedConcept(
                term="highest",
                role="operation",
                grounded_fields=[],
            )
        )
    return concepts


def _grounded_fields_for_terms(field_candidates: list[dict[str, Any]], terms: list[str]) -> list[GroundedField]:
    grounded: list[GroundedField] = []
    normalized_terms = {_normalize_text(term) for term in terms}
    for item in sorted(field_candidates, key=lambda value: _field_priority(str(value.get("field", "")), normalized_terms)):
        field = str(item.get("field", ""))
        normalized_field = _normalize_text(field)
        if not any(term in normalized_field for term in normalized_terms):
            continue
        grounded_item = GroundedField(
            asset=str(item.get("asset_path", "")),
            field=field,
            confidence="high" if _normalize_text(field).endswith(tuple(normalized_terms)) else "medium",
            reason="Matched question term to catalog/index field candidate.",
        )
        if grounded_item not in grounded:
            grounded.append(grounded_item)
    return grounded[:6]


def _field_priority(field: str, normalized_terms: set[str]) -> int:
    normalized_field = _normalize_text(field)
    if normalized_field.endswith("eventname"):
        return 0
    if normalized_field.endswith("cost"):
        return 0
    if any(normalized_field.endswith(term) for term in normalized_terms):
        return 1
    return 2


def _build_join_paths(context_bundle: dict[str, Any]) -> list[JoinPath]:
    paths: list[JoinPath] = []
    for path_payload in context_bundle.get("join_paths", []):
        edges = [
            JoinEdge(
                from_field=str(edge.get("from", "")),
                to_field=str(edge.get("to", "")),
                confidence=str(edge.get("confidence", "medium")),
                reason=str(edge.get("reason", "")),
            )
            for edge in path_payload.get("path", [])
        ]
        if not edges:
            continue
        paths.append(
            JoinPath(
                source=str(path_payload.get("source", "")),
                target=str(path_payload.get("target", "")),
                path=edges,
                confidence=str(path_payload.get("confidence", "medium")),
            )
        )
    return paths[:5]


def _build_answer_contract(question: str, concepts: list[GroundedConcept]) -> AnswerContract:
    tokens = set(_tokenize(question))
    metric_operation = "unknown"
    row_policy = "unknown"
    if tokens & {"lowest", "minimum", "min"}:
        metric_operation = "min"
        row_policy = "preserve_all_ties"
    elif tokens & {"highest", "maximum", "max"}:
        metric_operation = "max"
        row_policy = "preserve_all_ties"
    elif question.lower().startswith(("what ", "what's ", "which ")):
        row_policy = "single"

    answer_columns = _answer_columns_from_concepts(concepts)

    metric_field = None
    for concept in concepts:
        if concept.role != "metric":
            continue
        for field in concept.grounded_fields:
            metric_field = f"{field.asset}.{field.field}"
            break
    row_source = _asset_ref_from_field_ref(answer_columns[0].source_field) if answer_columns else ""
    if not row_source and metric_field:
        row_source = _asset_ref_from_field_ref(metric_field)
    join_policy = "inner" if answer_columns and metric_field else "unknown"
    return AnswerContract(
        answer_columns=answer_columns,
        row_policy=row_policy,
        metric_operation=metric_operation,
        metric_field=metric_field,
        metric_fields=[metric_field] if metric_field else [],
        row_source=row_source,
        join_policy=join_policy,
    )


def _answer_columns_from_concepts(concepts: list[GroundedConcept]) -> list[AnswerColumn]:
    answer_columns: list[AnswerColumn] = []
    for concept in concepts:
        if concept.role != "answer_entity":
            continue
        for field in concept.grounded_fields:
            source_field = f"{field.asset}.{field.field}"
            answer_columns.append(
                AnswerColumn(
                    name=field.field.rsplit(".", 1)[-1],
                    source_field=source_field,
                    reason=f"Grounded output concept `{concept.term}` to this same-entity field.",
                )
            )
            break
    return answer_columns


def _render_handoff_brief(handoff: DataUnderstandingHandoff) -> str:
    lines = ["## Data Understanding Brief", "", f"Status: {handoff.handoff_status}"]
    answer_column_names = get_answer_column_names(handoff.answer_contract)
    if answer_column_names:
        lines.append(f"Question asks for: {', '.join(answer_column_names)}")
    if handoff.answer_contract.metric_field:
        lines.append(f"Metric field: {handoff.answer_contract.metric_field}")
    elif handoff.answer_contract.metric_fields:
        lines.append(f"Metric fields: {', '.join(handoff.answer_contract.metric_fields)}")
    if handoff.answer_contract.metric_operation != "unknown":
        lines.append(
            f"Metric operation: {handoff.answer_contract.metric_operation}; row policy: {handoff.answer_contract.row_policy}"
        )

    lines.extend(["", "Grounding:"])
    for concept in handoff.question_grounding.concepts:
        if concept.grounded_fields:
            rendered_fields = ", ".join(f"{field.asset}.{field.field}" for field in concept.grounded_fields[:4])
            lines.append(f"- {concept.term} -> {rendered_fields}")
        for rejected in concept.rejected_fields[:4]:
            lines.append(f"- Do not use {rejected.asset}.{rejected.field}: {rejected.reason}")

    if handoff.data_fabric.join_paths:
        lines.extend(["", "Join path:"])
        for join_path in handoff.data_fabric.join_paths[:3]:
            rendered_edges = " -> ".join(f"{edge.from_field} => {edge.to_field}" for edge in join_path.path)
            lines.append(f"- {rendered_edges}")

    if answer_column_names:
        lines.extend(["", "Answer contract:"])
        lines.append(f"- Submit columns exactly as: {', '.join(answer_column_names)}")
        for column in handoff.answer_contract.answer_columns:
            if column.source_field:
                lines.append(f"- Use source field: {column.name} <- {column.source_field}")
        if handoff.answer_contract.row_source:
            lines.append(f"- Driving row source: {handoff.answer_contract.row_source}")
        if handoff.answer_contract.filters:
            lines.append(f"- Filters: {'; '.join(handoff.answer_contract.filters[:6])}")
        if handoff.answer_contract.group_by:
            lines.append(f"- Group by: {', '.join(handoff.answer_contract.group_by)}")
        if handoff.answer_contract.metric_fields:
            lines.append(f"- Metric fields: {', '.join(handoff.answer_contract.metric_fields)}")
        if handoff.answer_contract.enrichment_fields:
            lines.append(f"- Enrichment fields: {', '.join(handoff.answer_contract.enrichment_fields)}")
        if handoff.answer_contract.join_policy != "unknown":
            lines.append(f"- Join policy: {handoff.answer_contract.join_policy}")
        if handoff.answer_contract.output_grain:
            lines.append(f"- Output grain: {handoff.answer_contract.output_grain}")
        if handoff.answer_contract.distinct_policy != "unknown":
            lines.append(f"- Distinct policy: {handoff.answer_contract.distinct_policy}")
        if handoff.answer_contract.row_policy == "preserve_all_ties":
            lines.append("- Preserve all rows tied at the extreme value.")
    if handoff.validation_errors:
        lines.extend(["", "Validation warnings:"])
        lines.extend(f"- {error}" for error in handoff.validation_errors[:5])
    lines.append("")
    if handoff.handoff_status == "complete":
        lines.append("Use this handoff as trusted guidance; call tools to compute the requested result.")
    else:
        lines.append("Use this partial handoff as a candidate route; resolve warnings before finalizing.")
    return "\n".join(lines)[:4000]


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", text.lower()) if token]


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
