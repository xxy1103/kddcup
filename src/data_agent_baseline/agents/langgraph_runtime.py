from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from data_agent_baseline.agents.answer_validator import validate_answer as invoke_answer_validator
from data_agent_baseline.agents.prompt import build_system_prompt, build_task_prompt
from data_agent_baseline.agents.prompt2 import build_system_prompt_v2
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.agents.state import AgentGraphState
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors import DataUnderstandingAgent
from data_agent_baseline.model_retry import invoke_model_with_retries, summarize_model_retry_events
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import ToolRegistry, ToolRuntimeContext

logger = logging.getLogger(__name__)


TraceCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class LangGraphAgentConfig:
    max_steps: int = 16
    empty_stop_retry_limit: int = 1
    # Maximum number of times answer validation can reject and return to the main agent.
    validation_retry_limit: int = 2
    enable_answer_validator: bool = True
    validation_context_steps: int = 3
    enable_data_inspector: bool = False
    data_inspector: DataInspectorConfig = field(default_factory=DataInspectorConfig)
    prompt_version: int = 1


EMPTY_STOP_REPAIR_PROMPT = (
    "Your previous response stopped with no executable tool call. "
    "If you wrote a tool-call-like block in text or reasoning, it was only a pseudo tool call and was not executed. "
    "Re-issue the intended action now as a real tool call, or call `answer` if the final result is ready. "
    "Do not end the turn with plain text or an empty response."
)

MAX_REASONING_CONTEXT_CHARS = 6000

REASONING_CONTEXT_PREFIX = (
    "Previous model reasoning_content from the last turn, provided as historical context. "
    "This text is not an executed tool result and any tool-call-like block inside it was not executed unless a matching "
    "tool result appears in the conversation."
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


def _ai_reasoning_content(ai_message: AIMessage) -> str | None:
    return _render_message_content(ai_message.additional_kwargs.get("reasoning_content"))


def _build_reasoning_context_content(ai_message: AIMessage) -> str | None:
    reasoning_content = _ai_reasoning_content(ai_message)
    if reasoning_content is None:
        return None
    if len(reasoning_content) > MAX_REASONING_CONTEXT_CHARS:
        reasoning_content = f"{reasoning_content[:MAX_REASONING_CONTEXT_CHARS]}\n...[truncated]"
    return f"{REASONING_CONTEXT_PREFIX}\n\n```text\n{reasoning_content}\n```"


def _build_reasoning_context_message(ai_message: AIMessage) -> HumanMessage | None:
    content = _build_reasoning_context_content(ai_message)
    if content is None:
        return None
    return HumanMessage(content=content)


def _append_reasoning_context(prompt: str, ai_message: AIMessage) -> str:
    reasoning_context = _build_reasoning_context_content(ai_message)
    if reasoning_context is None:
        return prompt
    return f"{prompt}\n\n{reasoning_context}"



def _is_empty_stop(ai_message: AIMessage) -> bool:
    response_metadata = _coerce_dict(getattr(ai_message, "response_metadata", None))
    finish_reason = str(response_metadata.get("finish_reason", "")).lower()
    return _render_message_content(ai_message.content) is None and not ai_message.tool_calls and finish_reason == "stop"


def _extract_validation_context(
    *,
    state: AgentGraphState,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Extract the last N tool-step records for answer-validation context."""
    if max_steps <= 0:
        return []

    steps: list[dict[str, Any]] = state.get("steps", [])
    if not steps:
        return []

    tool_steps: list[dict[str, Any]] = []
    for step in reversed(steps):
        if step.get("node") != "tool":
            continue
        if len(tool_steps) >= max_steps:
            break
        # Skip previous `answer` submissions so the validator only sees data-querying steps.
        tool_calls = step.get("tool_calls") or []
        if any(tc.get("name") == "answer" for tc in tool_calls if isinstance(tc, dict)):
            continue
        tool_steps.append(step)

    tool_steps.reverse()

    context: list[dict[str, Any]] = []
    for step in tool_steps:
        tool_calls = step.get("tool_calls") or []
        tool_results = step.get("tool_results") or []

        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            tc_name = tc.get("name", "unknown")
            tc_args = tc.get("args") if isinstance(tc.get("args"), dict) else {}

            tr = tool_results[i] if i < len(tool_results) and isinstance(tool_results[i], dict) else {}
            tr_content = tr.get("content")
            if isinstance(tr_content, dict):
                stats = {k: v for k, v in tr_content.items() if k in ("row_count", "column_count", "columns", "success", "status")}
                output = tr_content.get("output")
                if isinstance(output, str):
                    output = output[:2000]
            elif isinstance(tr_content, str):
                stats = {}
                output = tr_content[:2000]
            else:
                stats = {}
                output = None

            entry: dict[str, Any] = {
                "tool": tc_name,
                "arguments": {
                    k: v for k, v in tc_args.items()
                    if k != "code"
                },
            }
            if stats:
                entry["result_stats"] = stats
            if output is not None:
                entry["result_preview"] = output
            context.append(entry)

    return context


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
            global_data_profile = update.get("global_data_profile", state.get("global_data_profile"))
            if isinstance(global_data_profile, str) and global_data_profile.strip():
                payload["global_data_profile"] = global_data_profile
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
            _PROMPT_BUILDERS = {
                1: build_system_prompt,
                2: build_system_prompt_v2,
            }
            builder = _PROMPT_BUILDERS.get(self.config.prompt_version, build_system_prompt)
            system_prompt = builder()
            return {
                "task_id": task.task_id,
                "messages": [
                    SystemMessage(content=system_prompt),
                ],
                "step_count": 0,
                "empty_stop_retry_count": 0,
                "validation_retry_count": 0,
                "answer": None,
                "failure_reason": None,
                "steps": [],
                "tool_events": [],
                "temp_workspace": None,
                "inspector": None,
                "global_data_profile": None,
                "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        def global_data_exploration(state: AgentGraphState) -> AgentGraphState:
            if not self.config.enable_data_inspector:
                return {}
            emit_in_progress_trace(
                state,
                node="global_data_exploration",
                assistant_message="Global data profiling is in progress.",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "global_data_exploration"}],
            )
            try:
                understanding_agent = DataUnderstandingAgent(
                    config=self.config.data_inspector,
                )
                profile = understanding_agent.explore_data_globally(
                    context_dir=task.context_dir,
                    task_id=task.task_id,
                )
                profile_preview = _preview_text(profile, limit=500) or ""
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="global_data_exploration",
                    assistant_message=profile_preview,
                    tool_calls=[],
                    tool_results=[{"ok": True, "content": {"profile_length": len(profile)}}],
                    ok=True,
                    model_request=None,
                    model_response=None,
                )
                update: AgentGraphState = {
                    "global_data_profile": profile,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="global_data_exploration",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=None,
                    model_response=None,
                )
                update = {
                    "global_data_profile": f"Global profiling failed: {exc}",
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

        def receive_problem(state: AgentGraphState) -> AgentGraphState:
            messages: list[BaseMessage] = [HumanMessage(content=build_task_prompt(task))]

            if self.config.enable_data_inspector:
                global_data_profile = state.get("global_data_profile") or ""
                if global_data_profile.strip():
                    content = (
                        "The following is the raw data catalog produced by global data exploration. "
                        "It contains asset, schema, and knowledge document information in JSON format. "
                        "Use it directly for constructing queries and understanding the data landscape:\n\n"
                        f"{global_data_profile}"
                    )
                    messages.append(HumanMessage(content=content))

            return {"messages": messages}

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
                "steps": [step_record.to_dict()],
                "tool_events": list(tool_results),
                "temp_workspace": runtime_context.temp_workspace,
            }
            if terminal_answer is not None:
                update["answer"] = terminal_answer
            else:
                reasoning_context_message = _build_reasoning_context_message(last_message)
                if reasoning_context_message is not None:
                    update["messages"] = [*tool_messages, reasoning_context_message]
            emit_trace(state, update)
            return update

        def repair_step(state: AgentGraphState) -> AgentGraphState:
            last_message = state["messages"][-1]
            prompt = (
                _append_reasoning_context(EMPTY_STOP_REPAIR_PROMPT, last_message)
                if isinstance(last_message, AIMessage)
                else EMPTY_STOP_REPAIR_PROMPT
            )
            step_record = StepRecord(
                step_index=next_step_index(state),
                node="repair",
                assistant_message=prompt,
                tool_calls=[],
                tool_results=[],
                ok=True,
                model_request=None,
                model_response=None,
            )
            update = {
                "messages": [HumanMessage(content=prompt)],
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

        def validate_answer_step(state: AgentGraphState) -> AgentGraphState:
            """Validate a submitted answer and optionally return control to the main agent."""
            if not self.config.enable_answer_validator:
                return {}

            answer = state.get("answer")
            failure_reason = state.get("failure_reason")

            if answer is None or failure_reason is not None:
                return {}

            # Unit-test fakes in this repo are not BaseChatModel instances. Real runtime
            # models are, so production runs still get the validation node behavior.
            if not isinstance(self.model, BaseChatModel):
                return {}

            current_retry = state.get("validation_retry_count", 0)
            if current_retry >= self.config.validation_retry_limit:
                logger.info(
                    "[%s] Answer validation reached retry limit (%d); accepting answer.",
                    task.task_id,
                    self.config.validation_retry_limit,
                )
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_answer",
                    assistant_message="Answer validation retry limit reached; accepting current answer.",
                    tool_calls=[],
                    tool_results=[{"ok": True, "skipped": True, "reason": "max_retries_reached"}],
                    ok=True,
                )
                return {"steps": [step_record.to_dict()]}

            if hasattr(answer, "to_dict"):
                answer_dict = answer.to_dict()
            elif isinstance(answer, dict):
                answer_dict = dict(answer)
            else:
                return {}

            logger.info(
                "[%s] Answer validator is checking submitted answer (attempt %d)...",
                task.task_id,
                current_retry + 1,
            )
            emit_in_progress_trace(
                state,
                node="validate_answer",
                assistant_message="Answer validator is checking the submitted answer.",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "answer_validation"}],
            )

            try:
                context_steps = _extract_validation_context(
                    state=state,
                    max_steps=self.config.validation_context_steps,
                )
                validation_request = {
                    "question": task.question,
                    "answer_columns": answer_dict.get("columns"),
                    "answer_row_count": len(answer_dict.get("rows", [])),
                    "context_steps_count": len(context_steps) if context_steps else 0,
                }
                validation_result = invoke_answer_validator(
                    model=self.model,
                    question=task.question,
                    answer=answer_dict,
                    context_steps=context_steps if context_steps else None,
                )
                is_valid = validation_result.get("valid", True)
                issues = validation_result.get("issues", [])
                validator_error = validation_result.get("validator_error")
                validation_response = {
                    "valid": is_valid,
                    "issues": issues,
                    "raw_response": validation_result.get("raw_response"),
                }

                if is_valid:
                    logger.info("[%s] Answer validation passed.", task.task_id)
                    step_record = StepRecord(
                        step_index=next_step_index(state),
                        node="validate_answer",
                        assistant_message="Answer format validation passed.",
                        tool_calls=[],
                        tool_results=[
                            {
                                "ok": True,
                                "valid": True,
                                "issues": [],
                                "validator_error": validator_error,
                            }
                        ],
                        ok=True,
                        model_request=validation_request,
                        model_response=validation_response,
                    )
                    return {"steps": [step_record.to_dict()]}

                issues_text = "\n".join(f"- {issue}" for issue in issues)
                feedback_message = (
                    "Your submitted answer did NOT pass the answer validation check. "
                    "The following issues were found:\n"
                    f"{issues_text}\n\n"
                    "Your previous answer, which has been rejected:\n"
                    f"```json\n{json.dumps(answer_dict, ensure_ascii=False, indent=2)}\n```\n\n"
                    "Please fix the issues above and re-submit by calling `answer` again. "
                    "Key formatting rules:\n"
                    "1. Dates must be ISO 8601 format with zero-padding, e.g. "
                    "'2024-03-01', not '2024-3-1'.\n"
                    "2. DateTime with timezone must be converted to UTC ending with 'Z'.\n"
                    "3. Only include columns that the question asks for.\n"
                    "4. String values are case-sensitive; do not change their case.\n"
                    "You may call tools again if needed, or directly call `answer` "
                    "with the corrected table."
                )
                logger.info(
                    "[%s] Answer validation failed with %d issue(s); returning to main agent:\n%s",
                    task.task_id,
                    len(issues),
                    issues_text,
                )
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_answer",
                    assistant_message=f"Answer validation failed:\n{issues_text}",
                    tool_calls=[],
                    tool_results=[
                        {
                            "ok": False,
                            "valid": False,
                            "issues": issues,
                        }
                    ],
                    ok=False,
                    model_request=validation_request,
                    model_response=validation_response,
                )
                update: AgentGraphState = {
                    "answer": None,
                    "failure_reason": None,
                    "messages": [HumanMessage(content=feedback_message)],
                    "validation_retry_count": current_retry + 1,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Answer validator failed; accepting original answer: %s", task.task_id, exc)
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_answer",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                )
                return {"steps": [step_record.to_dict()]}

        def route_after_model(state: AgentGraphState) -> str:
            if state.get("failure_reason") is not None or state.get("answer") is not None:
                return "finalize"
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tool_step"
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

        def route_after_validation(state: AgentGraphState) -> str:
            if state.get("answer") is None and state.get("failure_reason") is None:
                return "model_step"
            return "end"

        graph_builder = StateGraph(AgentGraphState)
        graph_builder.add_node("init_state", init_state)
        graph_builder.add_node("global_data_exploration", global_data_exploration)
        graph_builder.add_node("receive_problem", receive_problem)
        graph_builder.add_node("model_step", model_step)
        graph_builder.add_node("tool_step", tool_step)
        graph_builder.add_node("repair_step", repair_step)
        graph_builder.add_node("finalize", finalize)
        graph_builder.add_node("validate_answer", validate_answer_step)
        graph_builder.add_edge(START, "init_state")
        graph_builder.add_edge("init_state", "global_data_exploration")
        graph_builder.add_edge("global_data_exploration", "receive_problem")
        graph_builder.add_edge("receive_problem", "model_step")
        graph_builder.add_conditional_edges(
            "model_step",
            route_after_model,
            {
                "tool_step": "tool_step",
                "repair_step": "repair_step",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("repair_step", "model_step")
        graph_builder.add_conditional_edges(
            "tool_step",
            route_after_tool,
            {
                "model_step": "model_step",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("finalize", "validate_answer")
        graph_builder.add_conditional_edges(
            "validate_answer",
            route_after_validation,
            {
                "model_step": "model_step",
                "end": END,
            },
        )
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
            global_data_profile=final_state.get("global_data_profile"),
        )
