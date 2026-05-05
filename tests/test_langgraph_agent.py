from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


def _create_task(tmp_path: Path, task_id: str = "task_demo") -> PublicTask:
    task_dir = tmp_path / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample.csv").write_text("value\n1\n2\n", encoding="utf-8")
    (context_dir / "notes.md").write_text("# Notes\nhello\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id=task_id, difficulty="easy", question="List the value column."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class ScriptedToolCallingModel:
    def __init__(self, responses: list[AIMessage | BaseException]) -> None:
        self._responses = list(responses)
        self.invoke_count = 0
        self.bound_tools = []
        self.tool_choice = None
        self.parallel_tool_calls = None

    def bind_tools(self, tools, tool_choice="auto", parallel_tool_calls=False):  # noqa: ANN001
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        self.invoke_count += 1
        if not self._responses:
            raise RuntimeError("No scripted responses remaining.")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_langgraph_agent_executes_tool_call_loop_and_submits_answer(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                id="response_1",
                additional_kwargs={"provider_note": "first"},
                response_metadata={"finish_reason": "tool_calls"},
                tool_calls=[
                    {"name": "list_context", "args": {"max_depth": 2}, "id": "call_1", "type": "tool_call"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["value"], "rows": [["1"], ["2"]]},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.answer.columns == ["value"]
    assert result.answer.rows == [["1"], ["2"]]
    assert [step.node for step in result.steps] == ["model", "tool", "model", "tool"]
    assert model.tool_choice == "auto"
    assert model.parallel_tool_calls is False
    assert result.steps[0].model_request is not None
    assert result.steps[0].model_request["tool_choice"] == "auto"
    assert result.steps[0].model_request["parallel_tool_calls"] is False
    assert "answer" in result.steps[0].model_request["tool_names"]
    assert result.steps[0].model_response is not None
    assert result.steps[0].model_response["response_id"] == "response_1"
    assert result.steps[0].model_response["finish_reason"] == "tool_calls"
    assert result.steps[0].model_response["tool_call_names"] == ["list_context"]


def test_langgraph_agent_emits_live_trace_updates(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["ok"]]},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [len(update["steps"]) for update in trace_updates] == [1, 2, 2]
    assert trace_updates[0]["partial"] is True
    assert trace_updates[-1]["partial"] is False
    assert trace_updates[-1]["succeeded"] is True
    assert trace_updates[-1]["answer"] == {"columns": ["status"], "rows": [["ok"]]}
    json.dumps(trace_updates[-1], ensure_ascii=False)


def test_langgraph_agent_retries_model_request_errors_with_backoff(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = ScriptedToolCallingModel(
        responses=[
            RuntimeError("temporary request failure 1"),
            RuntimeError("temporary request failure 2"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["recovered"]]},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert model.invoke_count == 3
    assert sleep_delays == [15, 30]
    assert [step.node for step in result.steps] == ["model", "tool"]


def test_langgraph_agent_finalizes_after_request_retries_are_exhausted(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = ScriptedToolCallingModel(
        responses=[
            RuntimeError("temporary request failure 1"),
            RuntimeError("temporary request failure 2"),
            RuntimeError("temporary request failure 3"),
            RuntimeError("temporary request failure 4"),
            RuntimeError("temporary request failure 5"),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )
    result = agent.run(task)

    assert result.succeeded is False
    assert result.failure_reason == "Model request failed: temporary request failure 5"
    assert model.invoke_count == 5
    assert sleep_delays == [15, 30, 45, 60]
    assert [step.node for step in result.steps] == ["model"]
    assert result.steps[0].ok is False


def test_langgraph_agent_handles_nullable_completion_token_details(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                id="response_nullable",
                response_metadata={
                    "id": "response_nullable",
                    "finish_reason": "tool_calls",
                    "token_usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 22,
                        "completion_tokens_details": None,
                    },
                },
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["ok"]]},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert result.steps[0].model_response is not None
    assert result.steps[0].model_response["response_id"] == "response_nullable"
    assert result.steps[0].model_response["input_tokens"] == 11
    assert result.steps[0].model_response["output_tokens"] == 22
    assert result.steps[0].model_response["reasoning_tokens"] is None


def test_langgraph_agent_records_tool_errors_without_crashing(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_csv", "args": {}, "id": "call_1", "type": "tool_call"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["recovered"]]},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    tool_steps = [step for step in result.steps if step.node == "tool"]
    assert tool_steps[0].ok is False
    assert "error" in tool_steps[0].tool_results[0]
    recovery_model_step = next(step for step in result.steps if step.node == "model" and step.step_index == 3)
    assert recovery_model_step.model_request is not None
    assert recovery_model_step.model_request["last_message"]["type"] == "tool"
    assert recovery_model_step.model_request["last_message"]["name"] == "read_csv"
    assert recovery_model_step.model_request["last_message"]["status"] == "error"


def test_langgraph_agent_uses_temporary_workspace_for_python(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python",
                        "args": {"code": "from pathlib import Path\nPath('generated.txt').write_text('ok')\nprint('done')"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["ok"]]},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert not (task.context_dir / "generated.txt").exists()
    python_tool_step = next(step for step in result.steps if step.node == "tool")
    tool_result = python_tool_step.tool_results[0]
    assert tool_result["ok"] is True
    assert "done" in json.dumps(tool_result["content"], ensure_ascii=False)


def test_langgraph_agent_allows_reasoning_turn_before_tool_call(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(content="I should inspect the file tree first, then read the most relevant file.", tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_context", "args": {"max_depth": 2}, "id": "call_1", "type": "tool_call"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["done"]]},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=6),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "react", "model", "tool", "model", "tool"]
    assert result.steps[0].assistant_message == "I should inspect the file tree first, then read the most relevant file."
    assert result.steps[2].model_request is not None
    assert result.steps[2].model_request["last_message"]["content_preview"].startswith(
        "Your previous response was recorded as a brief working note."
    )
    assert result.steps[2].model_request["last_message"]["content_length"] > len(
        result.steps[2].model_request["last_message"]["content_preview"]
    )


def test_langgraph_agent_fails_after_too_many_reasoning_only_turns(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(responses=[AIMessage(content="I should think more about the plan.", tool_calls=[])])
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2, react_retry_limit=0),
    )

    result = agent.run(task)

    assert result.succeeded is False
    assert result.failure_reason == "Model kept reasoning without taking a tool action or submitting an answer."
    assert [step.node for step in result.steps] == ["model"]


def test_langgraph_agent_records_reasoning_content_in_model_response(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "I should inspect the context first."},
                tool_calls=[
                    {"name": "list_context", "args": {"max_depth": 2}, "id": "call_1", "type": "tool_call"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["done"]]},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    first_step = result.steps[0]
    assert first_step.assistant_message is None
    assert first_step.model_response is not None
    assert first_step.model_response["reasoning_content"] == "I should inspect the context first."
    assert first_step.model_response["reasoning_content_length"] == 35


def test_langgraph_agent_retries_once_after_empty_stop(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["recovered"]]},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "repair", "model", "tool"]
    second_model_step = result.steps[2]
    assert second_model_step.model_request is not None
    assert second_model_step.model_request["last_message"]["content_preview"].startswith(
        "Your previous response stopped with no content and no tool call."
    )
    assert second_model_step.model_request["last_message"]["content_length"] > len(
        second_model_step.model_request["last_message"]["content_preview"]
    )


def test_langgraph_agent_resets_empty_stop_retry_after_tool_progress(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_doc",
                        "args": {"path": "missing.md"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="", response_metadata={"finish_reason": "stop"}, tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["status"], "rows": [["recovered_twice"]]},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=8, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "repair", "model", "tool", "model", "repair", "model", "tool"]
    assert sum(1 for step in result.steps if step.node == "repair") == 2
