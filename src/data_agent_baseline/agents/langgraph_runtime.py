from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from data_agent_baseline.agents.prompt import build_system_prompt, build_task_prompt
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.agents.state import AgentGraphState
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors import DataUnderstandingAgent
from data_agent_baseline.inspectors.exchange import AgentEnvelope
from data_agent_baseline.inspectors.perception import PerceptionAttempt, PerceptionBuildError, invoke_perception_agent
from data_agent_baseline.model_retry import invoke_model_with_retries, summarize_model_retry_events
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import ToolRegistry, ToolRuntimeContext


TraceCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class LangGraphAgentConfig:
    max_steps: int = 16
    empty_stop_retry_limit: int = 1
    react_retry_limit: int = 2
    enable_data_inspector: bool = False
    data_inspector: DataInspectorConfig = field(default_factory=DataInspectorConfig)


EMPTY_STOP_REPAIR_PROMPT = (
    "Your previous response stopped with no content and no tool call. "
    "Continue solving the task. You must either call another tool or call `answer`. "
    "Do not end the turn with an empty response."
)

REACT_CONTINUATION_PROMPT = (
    "Your previous response was recorded as a brief working note. "
    "Now continue the task by taking the next concrete action: call a tool, or call `answer` if the result is ready. "
    "Do not repeat the same note."
)


def _render_message_content(content: Any) -> str | None:
    if content in (None, ""):
        return None
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}

def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        normalized.append(
            {
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "args": _coerce_dict(tool_call.get("args")),
            }
        )
    return normalized


def _preview_text(text: str | None, *, limit: int = 180) -> str | None:
    if text in (None, ""):
        return None
    return text if len(text) <= limit else f"{text[:limit]}..."


def _summarize_message(message: BaseMessage) -> dict[str, Any]:
    rendered_content = _render_message_content(message.content)
    payload = {
        "type": message.type,
        "name": getattr(message, "name", None),
        "content_preview": _preview_text(rendered_content),
        "content_length": 0 if rendered_content is None else len(rendered_content),
    }
    if isinstance(message, AIMessage):
        payload["tool_call_names"] = [call["name"] for call in _normalize_tool_calls(message.tool_calls)]
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        payload["status"] = getattr(message, "status", None)
    return payload


def _summarize_model_request(
    *,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    tool_choice: str,
    parallel_tool_calls: bool,
) -> dict[str, Any]:
    last_message = messages[-1] if messages else None
    return {
        "message_count": len(messages),
        "last_message": None if last_message is None else _summarize_message(last_message),
        "tool_names": [tool.name for tool in tools],
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
    }


def _summarize_ai_message(ai_message: AIMessage) -> dict[str, Any]:
    response_metadata = _coerce_dict(getattr(ai_message, "response_metadata", None))
    usage_metadata = _coerce_dict(getattr(ai_message, "usage_metadata", None))
    token_usage = _coerce_dict(response_metadata.get("token_usage"))
    completion_token_details = _coerce_dict(token_usage.get("completion_tokens_details"))
    output_token_details = _coerce_dict(usage_metadata.get("output_token_details"))
    rendered_content = _render_message_content(ai_message.content)
    reasoning_content = _render_message_content(
        ai_message.additional_kwargs.get("reasoning_content")
    )
    payload = {
        "response_id": response_metadata.get("id") or ai_message.id,
        "model_name": response_metadata.get("model_name"),
        "finish_reason": response_metadata.get("finish_reason"),
        "tool_call_names": [call["name"] for call in _normalize_tool_calls(ai_message.tool_calls)],
        "content_preview": _preview_text(rendered_content),
        "content_length": 0 if rendered_content is None else len(rendered_content),
        "input_tokens": usage_metadata.get("input_tokens", token_usage.get("prompt_tokens")),
        "output_tokens": usage_metadata.get("output_tokens", token_usage.get("completion_tokens")),
        "reasoning_tokens": output_token_details.get(
            "reasoning",
            completion_token_details.get("reasoning_tokens"),
        ),
    }
    if reasoning_content is not None:
        payload["reasoning_content"] = reasoning_content
        payload["reasoning_content_length"] = len(reasoning_content)
    return payload


def _summarize_perception_requests(attempts: list[PerceptionAttempt]) -> dict[str, Any] | None:
    if not attempts:
        return None
    return {
        "attempts": [
            _summarize_model_request(
                messages=attempt.messages,
                tools=[],
                tool_choice="none",
                parallel_tool_calls=False,
            )
            for attempt in attempts
        ]
    }


def _summarize_perception_responses(
    attempts: list[PerceptionAttempt],
    *,
    final_error: str | None = None,
) -> dict[str, Any] | None:
    if not attempts and final_error is None:
        return None
    rendered_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        response_summary: dict[str, Any] | None = None
        if isinstance(attempt.response, AIMessage):
            response_summary = _summarize_ai_message(attempt.response)
        elif attempt.response is not None:
            rendered_content = _render_message_content(getattr(attempt.response, "content", None))
            response_summary = {
                "type": getattr(attempt.response, "type", None),
                "content_preview": _preview_text(rendered_content),
                "content_length": 0 if rendered_content is None else len(rendered_content),
            }
        rendered_attempts.append(
            {
                "response": response_summary,
                "raw_output_preview": _preview_text(attempt.raw_output),
                "raw_output_length": 0 if attempt.raw_output is None else len(attempt.raw_output),
                "error": attempt.error,
                "validation_error": attempt.validation_error,
                "request_retry": summarize_model_retry_events(attempt.request_retry_events),
            }
        )
    payload: dict[str, Any] = {"attempts": rendered_attempts}
    if final_error is not None:
        payload["error"] = final_error
    return payload


def _is_empty_stop(ai_message: AIMessage) -> bool:
    response_metadata = _coerce_dict(getattr(ai_message, "response_metadata", None))
    finish_reason = str(response_metadata.get("finish_reason", "")).lower()
    return _render_message_content(ai_message.content) is None and not ai_message.tool_calls and finish_reason == "stop"


def _has_reasoning_note(ai_message: AIMessage) -> bool:
    return _render_message_content(ai_message.content) is not None and not ai_message.tool_calls


class LangGraphAgent:
    def __init__(
        self,
        *,
        model: BaseChatModel | Any,
        tools: ToolRegistry,
        config: LangGraphAgentConfig | None = None,
        trace_callback: TraceCallback | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or LangGraphAgentConfig()
        self.trace_callback = trace_callback

    def run(self, task: PublicTask) -> AgentRunResult:
        python_workspace = TaskContextWorkspace(task.context_dir)
        runtime_context = ToolRuntimeContext(task=task, python_workspace=python_workspace)
        bound_tools = self.tools.bind(runtime_context)
        langchain_tools = bound_tools.langchain_tools()
        tool_choice = "auto"
        parallel_tool_calls = False
        model_with_tools = self.model.bind_tools(
            langchain_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        def trace_timestamp() -> str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def answer_payload(answer: Any) -> dict[str, Any] | None:
            if answer is None:
                return None
            if hasattr(answer, "to_dict"):
                return answer.to_dict()
            return dict(answer) if isinstance(answer, dict) else None

        def emit_trace(
            state: AgentGraphState,
            update: AgentGraphState | None = None,
            *,
            partial: bool = True,
        ) -> None:
            if self.trace_callback is None:
                return
            update = update or {}
            answer = update.get("answer", state.get("answer"))
            failure_reason = update.get("failure_reason", state.get("failure_reason"))
            payload = {
                "task_id": task.task_id,
                "answer": answer_payload(answer),
                "steps": [*list(state.get("steps", [])), *list(update.get("steps", []))],
                "failure_reason": failure_reason,
                "succeeded": answer is not None and failure_reason is None,
                "inspector": update.get("inspector", state.get("inspector")),
                "partial": partial,
                "started_at": state.get("started_at"),
                "updated_at": trace_timestamp(),
            }
            self.trace_callback(payload)

        def emit_in_progress_trace(
            state: AgentGraphState,
            *,
            node: str,
            assistant_message: str | None = None,
            tool_calls: list[dict[str, Any]] | None = None,
            tool_results: list[dict[str, Any]] | None = None,
            model_request: dict[str, Any] | None = None,
            model_response: dict[str, Any] | None = None,
        ) -> None:
            step_payload = StepRecord(
                step_index=next_step_index(state),
                node=node,
                assistant_message=assistant_message,
                tool_calls=tool_calls or [],
                tool_results=tool_results or [{"ok": None, "status": "in_progress"}],
                ok=False,
                model_request=model_request,
                model_response=model_response,
            ).to_dict()
            step_payload["status"] = "in_progress"
            step_payload["started_at"] = trace_timestamp()
            emit_trace(state, {"steps": [step_payload]})

        def next_step_index(state: AgentGraphState) -> int:
            return len(state.get("steps", [])) + 1

        def init_state(_: AgentGraphState) -> AgentGraphState:
            return {
                "task_id": task.task_id,
                "messages": [
                    SystemMessage(content=build_system_prompt()),
                    HumanMessage(content=build_task_prompt(task)),
                ],
                "step_count": 0,
                "empty_stop_retry_count": 0,
                "react_retry_count": 0,
                "answer": None,
                "failure_reason": None,
                "steps": [],
                "tool_events": [],
                "temp_workspace": None,
                "inspector": None,
                "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        def perceive_task(state: AgentGraphState) -> AgentGraphState:
            if not self.config.enable_data_inspector:
                return {}
            perception_request_payload = {
                "attempts": [
                    {
                        "message_count": 2,
                        "last_message": {
                            "type": "human",
                            "content_preview": _preview_text(task.question),
                            "content_length": len(task.question),
                        },
                        "tool_names": [],
                        "tool_choice": "none",
                        "parallel_tool_calls": False,
                    }
                ]
            }
            emit_in_progress_trace(
                state,
                node="perceive_task",
                assistant_message="Perception agent request is in progress.",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "perception_model_request"}],
                model_request=perception_request_payload,
            )
            perception_retry_events: list[dict[str, Any]] = []

            def record_perception_retry(event: dict[str, Any]) -> None:
                perception_retry_events.append(dict(event))
                retry_status = "retrying" if event.get("will_retry") else "failed"
                emit_in_progress_trace(
                    state,
                    node="perceive_task",
                    assistant_message="Perception agent request is in progress.",
                    tool_results=[
                        {
                            "ok": False,
                            "status": retry_status,
                            "phase": "perception_model_request",
                            "attempt": event.get("attempt"),
                            "max_attempts": event.get("max_attempts"),
                            "error_type": event.get("error_type"),
                            "error": event.get("error"),
                            "next_retry_delay_seconds": event.get("next_retry_delay_seconds"),
                        }
                    ],
                    model_request=perception_request_payload,
                    model_response={
                        "request_retry": summarize_model_retry_events(perception_retry_events),
                    },
                )

            try:
                perception_result = invoke_perception_agent(
                    task,
                    self.model,
                    retry_event_callback=record_perception_retry,
                )
                perception_envelope = perception_result.envelope
                inspector_payload = {
                    "perception": perception_envelope.content.payload,
                    "perception_envelope": perception_envelope.to_dict(),
                }
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="perceive_task",
                    assistant_message=perception_envelope.content.summary,
                    tool_calls=[],
                    tool_results=[{"ok": True, "content": inspector_payload["perception"]}],
                    ok=True,
                    model_request=_summarize_perception_requests(perception_result.attempts),
                    model_response=_summarize_perception_responses(perception_result.attempts),
                )
                update: AgentGraphState = {
                    "inspector": inspector_payload,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update
            except PerceptionBuildError as exc:
                attempts = list(exc.attempts)
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="perceive_task",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=_summarize_perception_requests(attempts),
                    model_response=_summarize_perception_responses(attempts, final_error=str(exc)),
                )
                update = {
                    "inspector": {"error": f"Perception failed: {exc}"},
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="perceive_task",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=None,
                    model_response=None,
                )
                update = {
                    "inspector": {"error": f"Perception failed: {exc}"},
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

        def understand_and_explore_data(state: AgentGraphState) -> AgentGraphState:
            if not self.config.enable_data_inspector:
                return {}
            inspector_payload = dict(state.get("inspector") or {})
            if inspector_payload.get("error"):
                return {}
            emit_in_progress_trace(
                state,
                node="understand_and_explore_data",
                assistant_message="Data understanding agent is in progress.",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "data_understanding"}],
            )
            try:
                perception_envelope = AgentEnvelope.model_validate(inspector_payload.get("perception_envelope"))
                understanding_agent = DataUnderstandingAgent(
                    model=self.model,
                    config=self.config.data_inspector,
                )
                result = understanding_agent.run(task, perception_envelope)
                result_payload = result.to_dict()
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="understand_and_explore_data",
                    assistant_message=result.summary,
                    tool_calls=[],
                    tool_results=[
                        {
                            "ok": True,
                            "content": {
                                "asset_count": len(result.semantic_catalog.get("assets", [])),
                                "schema_count": len(result.semantic_catalog.get("schemas", [])),
                                "handoff_status": result.handoff_status,
                                "validation_errors": result.validation_errors,
                                "inspector_steps": result.inspector_steps
                                if self.config.data_inspector.include_inspector_trace
                                else [],
                            },
                        }
                    ],
                    ok=True,
                    model_request=None,
                    model_response=None,
                )
                update: AgentGraphState = {
                    "inspector": result_payload,
                    "steps": [step_record.to_dict()],
                }
                if self.config.data_inspector.inject_summary_to_agent and result.summary:
                    handoff_json = json.dumps(
                        result.data_understanding_handoff,
                        ensure_ascii=False,
                        indent=2,
                    )
                    update["messages"] = [
                        HumanMessage(
                            content=(
                                f"{result.summary}\n\n"
                                "Full data_understanding_handoff.json:\n"
                                "```json\n"
                                f"{handoff_json}\n"
                                "```\n\n"
                                + (
                                    "Treat this handoff as trusted guidance from a separate data understanding agent. "
                                    "Use the full JSON for structured fields, join paths, answer contract, rejected fields, "
                                    "row source, filters, join policy, and validation status. Do not re-verify it by default; "
                                    "call tools only to compute the requested result, resolve validation warnings or missing details, or investigate a clear conflict."
                                    if result.handoff_status == "complete"
                                    else "Treat this partial/fallback handoff as a candidate route. Use the JSON to focus exploration, "
                                    "but resolve validation warnings before finalizing the answer."
                                )
                            )
                        )
                    ]
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                merged_payload = {
                    **inspector_payload,
                    "error": f"Data understanding failed: {exc}",
                }
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="understand_and_explore_data",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=None,
                    model_response=None,
                )
                update = {
                    "inspector": merged_payload,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

        def model_step(state: AgentGraphState) -> AgentGraphState:
            if state.get("failure_reason") is not None or state.get("answer") is not None:
                return {}
            if state.get("step_count", 0) >= self.config.max_steps:
                return {"failure_reason": "Agent did not submit an answer within max_steps."}

            request_payload = _summarize_model_request(
                messages=list(state["messages"]),
                tools=langchain_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
            emit_in_progress_trace(
                state,
                node="model",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "model_request"}],
                model_request=request_payload,
            )
            retry_events: list[dict[str, Any]] = []

            def record_model_retry(event: dict[str, Any]) -> None:
                retry_events.append(dict(event))
                retry_status = "retrying" if event.get("will_retry") else "failed"
                emit_in_progress_trace(
                    state,
                    node="model",
                    tool_results=[
                        {
                            "ok": False,
                            "status": retry_status,
                            "phase": "model_request",
                            "attempt": event.get("attempt"),
                            "max_attempts": event.get("max_attempts"),
                            "error_type": event.get("error_type"),
                            "error": event.get("error"),
                            "next_retry_delay_seconds": event.get("next_retry_delay_seconds"),
                        }
                    ],
                    model_request=request_payload,
                    model_response={
                        "request_retry": summarize_model_retry_events(retry_events),
                    },
                )

            try:
                ai_message = invoke_model_with_retries(
                    model_with_tools,
                    state["messages"],
                    on_retry_event=record_model_retry,
                )
            except Exception as exc:
                model_response = {"error": str(exc)}
                request_retry = summarize_model_retry_events(retry_events, succeeded=False)
                if request_retry is not None:
                    model_response["request_retry"] = request_retry
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="model",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=request_payload,
                    model_response=model_response,
                )
                update = {
                    "failure_reason": f"Model request failed: {exc}",
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

            model_response = _summarize_ai_message(ai_message)
            request_retry = summarize_model_retry_events(retry_events, succeeded=True)
            if request_retry is not None:
                model_response["request_retry"] = request_retry
            step_record = StepRecord(
                step_index=next_step_index(state),
                node="model",
                assistant_message=_render_message_content(ai_message.content),
                tool_calls=_normalize_tool_calls(ai_message.tool_calls),
                tool_results=[],
                ok=True,
                model_request=request_payload,
                model_response=model_response,
            )
            update = {
                "messages": [ai_message],
                "step_count": state.get("step_count", 0) + 1,
                "steps": [step_record.to_dict()],
            }
            emit_trace(state, update)
            return update

        def tool_step(state: AgentGraphState) -> AgentGraphState:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                return {"failure_reason": "Tool execution requested without a preceding AI tool call."}

            emit_in_progress_trace(
                state,
                node="tool",
                tool_calls=_normalize_tool_calls(last_message.tool_calls),
                tool_results=[{"ok": None, "status": "in_progress", "phase": "tool_execution"}],
            )

            tool_results: list[dict[str, Any]] = []
            tool_messages: list[ToolMessage] = []
            terminal_answer = state.get("answer")
            overall_ok = True

            for tool_call in last_message.tool_calls:
                tool_name = str(tool_call.get("name"))
                tool_args = _coerce_dict(tool_call.get("args"))
                tool_call_id = str(tool_call.get("id"))
                try:
                    result = bound_tools.execute(tool_name, tool_args)
                    payload = {
                        "ok": result.ok,
                        "tool": tool_name,
                        "content": result.content,
                    }
                    if result.answer is not None:
                        terminal_answer = result.answer
                    overall_ok = overall_ok and result.ok
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "tool": tool_name,
                        "error": str(exc),
                    }
                    overall_ok = False

                tool_results.append(payload)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        status="success" if payload.get("ok") else "error",
                    )
                )

            step_record = StepRecord(
                step_index=next_step_index(state),
                node="tool",
                assistant_message=None,
                tool_calls=_normalize_tool_calls(last_message.tool_calls),
                tool_results=tool_results,
                ok=overall_ok,
                model_response=None,
            )

            update: AgentGraphState = {
                "messages": tool_messages,
                "empty_stop_retry_count": 0,
                "react_retry_count": 0,
                "steps": [step_record.to_dict()],
                "tool_events": list(tool_results),
                "temp_workspace": runtime_context.temp_workspace,
            }
            if terminal_answer is not None:
                update["answer"] = terminal_answer
            emit_trace(state, update)
            return update

        def react_step(state: AgentGraphState) -> AgentGraphState:
            step_record = StepRecord(
                step_index=next_step_index(state),
                node="react",
                assistant_message=REACT_CONTINUATION_PROMPT,
                tool_calls=[],
                tool_results=[],
                ok=True,
                model_request=None,
                model_response=None,
            )
            update = {
                "messages": [HumanMessage(content=REACT_CONTINUATION_PROMPT)],
                "react_retry_count": state.get("react_retry_count", 0) + 1,
                "steps": [step_record.to_dict()],
            }
            emit_trace(state, update)
            return update

        def repair_step(state: AgentGraphState) -> AgentGraphState:
            step_record = StepRecord(
                step_index=next_step_index(state),
                node="repair",
                assistant_message=EMPTY_STOP_REPAIR_PROMPT,
                tool_calls=[],
                tool_results=[],
                ok=True,
                model_request=None,
                model_response=None,
            )
            update = {
                "messages": [HumanMessage(content=EMPTY_STOP_REPAIR_PROMPT)],
                "empty_stop_retry_count": state.get("empty_stop_retry_count", 0) + 1,
                "steps": [step_record.to_dict()],
            }
            emit_trace(state, update)
            return update

        def finalize(state: AgentGraphState) -> AgentGraphState:
            failure_reason = state.get("failure_reason")
            if state.get("answer") is None and failure_reason is None:
                last_message = state["messages"][-1] if state.get("messages") else None
                if state.get("step_count", 0) >= self.config.max_steps:
                    failure_reason = "Agent did not submit an answer within max_steps."
                elif isinstance(last_message, AIMessage) and _has_reasoning_note(last_message):
                    failure_reason = "Model kept reasoning without taking a tool action or submitting an answer."
                elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    failure_reason = "Model did not request a tool or submit an answer."
                else:
                    failure_reason = "Agent finished without submitting an answer."

            update = {
                "failure_reason": failure_reason,
                "temp_workspace": runtime_context.temp_workspace,
            }
            emit_trace(state, update, partial=False)
            return update

        def route_after_model(state: AgentGraphState) -> str:
            if state.get("failure_reason") is not None or state.get("answer") is not None:
                return "finalize"
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tool_step"
            if (
                isinstance(last_message, AIMessage)
                and _has_reasoning_note(last_message)
                and state.get("react_retry_count", 0) < self.config.react_retry_limit
            ):
                return "react_step"
            if (
                isinstance(last_message, AIMessage)
                and _is_empty_stop(last_message)
                and state.get("empty_stop_retry_count", 0) < self.config.empty_stop_retry_limit
            ):
                return "repair_step"
            return "finalize"

        def route_after_tool(state: AgentGraphState) -> str:
            if state.get("answer") is not None or state.get("failure_reason") is not None:
                return "finalize"
            if state.get("step_count", 0) >= self.config.max_steps:
                return "finalize"
            return "model_step"

        graph_builder = StateGraph(AgentGraphState)
        graph_builder.add_node("init_state", init_state)
        graph_builder.add_node("perceive_task", perceive_task)
        graph_builder.add_node("understand_and_explore_data", understand_and_explore_data)
        graph_builder.add_node("model_step", model_step)
        graph_builder.add_node("tool_step", tool_step)
        graph_builder.add_node("react_step", react_step)
        graph_builder.add_node("repair_step", repair_step)
        graph_builder.add_node("finalize", finalize)
        graph_builder.add_edge(START, "init_state")
        graph_builder.add_edge("init_state", "perceive_task")
        graph_builder.add_edge("perceive_task", "understand_and_explore_data")
        graph_builder.add_edge("understand_and_explore_data", "model_step")
        graph_builder.add_conditional_edges(
            "model_step",
            route_after_model,
            {
                "tool_step": "tool_step",
                "react_step": "react_step",
                "repair_step": "repair_step",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("react_step", "model_step")
        graph_builder.add_edge("repair_step", "model_step")
        graph_builder.add_conditional_edges(
            "tool_step",
            route_after_tool,
            {
                "model_step": "model_step",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("finalize", END)
        graph = graph_builder.compile()

        try:
            final_state = graph.invoke({})
        finally:
            python_workspace.cleanup()

        steps = [StepRecord(**step_payload) for step_payload in final_state.get("steps", [])]
        return AgentRunResult(
            task_id=task.task_id,
            answer=final_state.get("answer"),
            steps=steps,
            failure_reason=final_state.get("failure_reason"),
            inspector=final_state.get("inspector"),
        )
