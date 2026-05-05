from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.inspectors.exchange import (
    AgentEnvelope,
    AgentEnvelopeContent,
    SemanticClaim,
    Uncertainty,
)
from data_agent_baseline.model_retry import invoke_model_with_retries


PerceptionRetryEventCallback = Callable[[dict[str, Any]], None]


PERCEPTION_SYSTEM_PROMPT = """
You are PerceptionAgent. Your job is to analyze the user's data question before data exploration.
You do not solve the task, compute answers, inspect files, or choose concrete dataset fields.

Return only one valid JSON object matching the requested schema.
Do not return Markdown, code fences, comments, explanations, or extra keys.

Semantic rules:
1. Preserve complete filter phrases and noun phrases. Do not shrink a multi-word filter, modifier, or scope phrase to only a proper name.
2. Separate output entities, filter scopes, metric concepts, operations, and answer shape.
3. Identify the head entity being modified by qualifiers. For example, distinguish a named geographic qualifier from the entity level it modifies.
4. Mark answer-changing ambiguity in high_risk_terms, especially geography/scope, entity level, aggregation grain, metric operation, joins/keys, ranks/positions, type/category labels, units, and ties.
5. Unless explicitly asked for an ID, always set column_hint to human-readable display fields (e.g., name, title, text) for "what is" queries, prioritizing content over system identifiers.
6. Keep the analysis compact and task-focused. Use short strings.
""".strip()

_PERCEPTION_SCHEMA: dict[str, Any] = {
    "question": "exact original question string",
    "difficulty": "task difficulty string",
    "entities": ["output entities, named entities, and important entity-level nouns"],
    "metrics": ["metric names or operations explicitly requested"],
    "filter_phrases": ["complete filter/scope phrases from the question"],
    "expected_answer_shape": {
        "row_shape": "single_row|multiple_rows|unknown",
        "column_hint": "short description of requested output columns",
        "only_requested_columns": True,
    },
    "high_risk_terms": [
        "short risk labels such as aggregation_grain, geographic_scope_ambiguity, entity_level_ambiguity"
    ],
}

_RISK_LABEL_GUIDANCE = {
    "aggregation_grain": "The grouping or row grain could change the answer.",
    "geographic_scope_ambiguity": "A place-like phrase could map to multiple geographic or administrative levels.",
    "entity_level_ambiguity": "The question may refer to different entity levels, such as item versus group.",
    "metric_operation_ambiguity": "The metric operation or comparison target is not fully explicit.",
    "filter_scope_ambiguity": "A filter phrase could apply to different entities or scopes.",
    "join_key_ambiguity": "The task likely requires linking assets and the key choice may matter.",
    "rank_position_ambiguity": "Rank and position/order may not be interchangeable.",
    "type_category_level_ambiguity": "A requested type/category label may belong to different entity levels.",
    "per_unit_ratio": "A per-unit phrase may require a ratio instead of a total.",
    "tie_handling_ambiguity": "An extreme/ranking question may require preserving ties.",
}


class ExpectedAnswerShapeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_shape: str = "unknown"
    column_hint: str = "unknown"
    only_requested_columns: bool = True


class PerceptionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = None
    difficulty: str | None = None
    entities: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    filter_phrases: list[str] = Field(default_factory=list)
    expected_answer_shape: ExpectedAnswerShapeDraft = Field(default_factory=ExpectedAnswerShapeDraft)
    high_risk_terms: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PerceptionAttempt:
    messages: list[BaseMessage]
    response: BaseMessage | None = None
    raw_output: str | None = None
    error: str | None = None
    validation_error: str | None = None
    request_retry_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PerceptionInvocationResult:
    envelope: AgentEnvelope
    attempts: list[PerceptionAttempt]


class PerceptionBuildError(RuntimeError):
    def __init__(self, message: str, *, attempts: list[PerceptionAttempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


def build_perception_retry_prompt(
    *,
    task: PublicTask,
    previous_error: str,
    previous_output: str | None,
) -> str:
    payload = {
        "instruction": (
            "Fix the previous perception response format. Return only one valid JSON object matching "
            "required_json_schema. Do not use Markdown, code fences, explanatory prose, comments, or extra keys."
        ),
        "task": {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "question": task.question,
        },
        "previous_error": previous_error,
        "previous_output": previous_output or "",
        "required_json_schema": _PERCEPTION_SCHEMA,
        "risk_label_guidance": _RISK_LABEL_GUIDANCE,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_perception_envelope(task: PublicTask, model: Any) -> AgentEnvelope:
    return invoke_perception_agent(task, model).envelope


def invoke_perception_agent(
    task: PublicTask,
    model: Any,
    *,
    retry_event_callback: PerceptionRetryEventCallback | None = None,
) -> PerceptionInvocationResult:
    if model is None:
        raise PerceptionBuildError("Perception model is not available.", attempts=[])

    attempts: list[PerceptionAttempt] = []
    first_messages = _build_perception_messages(task)
    draft, attempt = _invoke_and_parse(
        model=model,
        messages=first_messages,
        retry_event_callback=retry_event_callback,
    )
    attempts.append(attempt)
    if draft is not None:
        return PerceptionInvocationResult(envelope=_build_envelope(task, draft), attempts=attempts)
    if attempt.error is not None:
        raise PerceptionBuildError(f"Perception request failed: {attempt.error}", attempts=attempts)

    previous_error = attempt.error or attempt.validation_error or "Unknown perception validation error."
    retry_messages = [
        SystemMessage(content=PERCEPTION_SYSTEM_PROMPT),
        HumanMessage(
            content=build_perception_retry_prompt(
                task=task,
                previous_error=previous_error,
                previous_output=attempt.raw_output,
            )
        ),
    ]
    retry_draft, retry_attempt = _invoke_and_parse(
        model=model,
        messages=retry_messages,
        retry_event_callback=retry_event_callback,
    )
    attempts.append(retry_attempt)
    if retry_draft is not None:
        return PerceptionInvocationResult(envelope=_build_envelope(task, retry_draft), attempts=attempts)

    final_error = retry_attempt.error or retry_attempt.validation_error or "Unknown perception validation error."
    raise PerceptionBuildError(f"Perception failed after retry: {final_error}", attempts=attempts)


def _build_perception_messages(task: PublicTask) -> list[BaseMessage]:
    payload = {
        "task": {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "question": task.question,
        },
        "instructions": [
            "Analyze only the question text and task difficulty.",
            "Extract output entities, important named entities, metrics, filters, and answer shape.",
            "Preserve complete filter and scope phrases from the wording.",
            "For content-bearing outputs such as comments, posts, questions, answers, users, tags, products, or places, prefer human-readable text/body/title/name/display fields in column_hint unless the question explicitly asks for an id or identifier.",
            "Use high_risk_terms for any ambiguity that could change the final answer.",
            "Return only valid JSON matching required_json_schema.",
        ],
        "required_json_schema": _PERCEPTION_SCHEMA,
        "risk_label_guidance": _RISK_LABEL_GUIDANCE,
    }
    return [
        SystemMessage(content=PERCEPTION_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]


def _invoke_and_parse(
    *,
    model: Any,
    messages: list[BaseMessage],
    retry_event_callback: PerceptionRetryEventCallback | None = None,
) -> tuple[PerceptionDraft | None, PerceptionAttempt]:
    request_retry_events: list[dict[str, Any]] = []

    def record_request_retry(event: dict[str, Any]) -> None:
        event_payload = dict(event)
        request_retry_events.append(event_payload)
        if retry_event_callback is not None:
            retry_event_callback(event_payload)

    try:
        response = invoke_model_with_retries(
            model,
            messages,
            on_retry_event=record_request_retry,
        )
    except Exception as exc:  # noqa: BLE001
        return None, PerceptionAttempt(
            messages=messages,
            error=str(exc),
            request_retry_events=request_retry_events,
        )

    raw_output = _message_text(response)
    if not raw_output:
        return None, PerceptionAttempt(
            messages=messages,
            response=response,
            raw_output=raw_output,
            validation_error="Model returned empty perception content.",
            request_retry_events=request_retry_events,
        )

    try:
        payload = _extract_json_object(raw_output)
        draft = PerceptionDraft.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        return None, PerceptionAttempt(
            messages=messages,
            response=response,
            raw_output=raw_output,
            validation_error=str(exc),
            request_retry_events=request_retry_events,
        )
    return draft, PerceptionAttempt(
        messages=messages,
        response=response,
        raw_output=raw_output,
        request_retry_events=request_retry_events,
    )


def _message_text(message: Any) -> str | None:
    content = getattr(message, "content", message)
    if content in (None, ""):
        return None
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
        raise ValueError("Perception response must be a JSON object.")
    return payload


def _build_envelope(task: PublicTask, draft: PerceptionDraft) -> AgentEnvelope:
    payload = _normalized_payload(task, draft)
    risks = [
        Uncertainty(
            risk=risk,
            instruction=_risk_instruction(risk),
        )
        for risk in payload["high_risk_terms"]
    ]
    claims = [
        SemanticClaim(
            claim=f"Expected answer shape: {payload['expected_answer_shape']}",
            confidence="medium",
        )
    ]
    if payload["filter_phrases"]:
        claims.append(
            SemanticClaim(
                claim=f"Question filters: {payload['filter_phrases']}",
                confidence="medium",
            )
        )
    summary = (
        f"Question entities: {payload['entities']}. "
        f"Metrics: {payload['metrics']}. "
        f"Filters: {payload['filter_phrases']}. "
        f"Expected answer shape: {payload['expected_answer_shape']}. "
        f"High-risk terms: {payload['high_risk_terms']}."
    )
    return AgentEnvelope(
        task_id=task.task_id,
        sender="PerceptionAgent",
        recipient="DataUnderstandingAgent",
        message_type="perception_result",
        content=AgentEnvelopeContent(
            summary=summary,
            semantic_claims=claims,
            uncertainties=risks,
            payload=payload,
        ),
    )


def _normalized_payload(task: PublicTask, draft: PerceptionDraft) -> dict[str, Any]:
    shape = draft.expected_answer_shape
    return {
        "question": task.question,
        "difficulty": task.difficulty,
        "entities": _clean_list(draft.entities),
        "metrics": _clean_list(draft.metrics),
        "filter_phrases": _clean_list(draft.filter_phrases),
        "expected_answer_shape": {
            "row_shape": (shape.row_shape or "unknown").strip() or "unknown",
            "column_hint": (shape.column_hint or "unknown").strip() or "unknown",
            "only_requested_columns": bool(shape.only_requested_columns),
        },
        "high_risk_terms": _clean_list(draft.high_risk_terms),
    }


def _clean_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _risk_instruction(risk: str) -> str:
    detail = _RISK_LABEL_GUIDANCE.get(risk)
    if detail:
        return detail
    return f"Verify high-risk term before final answer: {risk}"
