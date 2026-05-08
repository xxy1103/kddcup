from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import DataInspectorConfig
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


def _perception_response(task: PublicTask) -> str:
    return json.dumps(
        {
            "question": task.question,
            "difficulty": task.difficulty,
            "entities": ["value"],
            "metrics": [],
            "filter_phrases": [],
            "expected_answer_shape": {
                "row_shape": "multiple_rows",
                "column_hint": "value",
                "only_requested_columns": True,
            },
            "high_risk_terms": [],
        }
    )


class ScriptedToolCallingModel:
    def __init__(self, responses: list[AIMessage | BaseException]) -> None:
        self._responses = list(responses)
        self.invoke_count = 0
        self.invocations = []
        self.bound_tools = []
        self.tool_choice = None
        self.parallel_tool_calls = None

    def bind_tools(self, tools, tool_choice="auto", parallel_tool_calls=False):  # noqa: ANN001
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.invocations.append(list(messages))
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
    assert [len(update["steps"]) for update in trace_updates] == [1, 1, 2, 2, 2]
    assert trace_updates[0]["partial"] is True
    assert trace_updates[0]["steps"][-1]["node"] == "model"
    assert trace_updates[0]["steps"][-1]["status"] == "in_progress"
    assert trace_updates[1]["steps"][-1]["node"] == "model"
    assert "status" not in trace_updates[1]["steps"][-1]
    assert trace_updates[2]["steps"][-1]["node"] == "tool"
    assert trace_updates[2]["steps"][-1]["status"] == "in_progress"
    assert trace_updates[-1]["partial"] is False
    assert trace_updates[-1]["succeeded"] is True
    assert trace_updates[-1]["answer"] == {"columns": ["status"], "rows": [["ok"]]}
    json.dumps(trace_updates[-1], ensure_ascii=False)


def test_langgraph_agent_emits_in_progress_trace_before_model_invoke(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []

    class ObservingModel(ScriptedToolCallingModel):
        def invoke(self, messages):  # noqa: ANN001
            pending_step = trace_updates[-1]["steps"][-1]
            assert pending_step["node"] == "model"
            assert pending_step["status"] == "in_progress"
            assert pending_step["model_request"] is not None
            return super().invoke(messages)

    model = ObservingModel(
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
    assert result.steps[0].model_response is not None
    request_retry = result.steps[0].model_response["request_retry"]
    assert request_retry["status"] == "succeeded_after_retry"
    assert request_retry["retry_count"] == 2
    assert request_retry["request_error_count"] == 2
    assert "last_error_type" not in request_retry
    assert [event["error"] for event in request_retry["errors"]] == [
        "temporary request failure 1",
        "temporary request failure 2",
    ]
    assert all("error_type" not in event for event in request_retry["errors"])


def test_langgraph_agent_live_trace_records_model_retry_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []
    sleep_delays: list[int] = []
    monkeypatch.setattr("data_agent_baseline.model_retry.time.sleep", sleep_delays.append)
    model = ScriptedToolCallingModel(
        responses=[
            RuntimeError("temporary request failure"),
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
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    retry_updates = []
    for update in trace_updates:
        model_response = update["steps"][-1].get("model_response") or {}
        if model_response.get("request_retry"):
            retry_updates.append(update)
    assert retry_updates
    live_retry = retry_updates[0]["steps"][-1]["model_response"]["request_retry"]
    assert live_retry["status"] == "retrying"
    assert live_retry["retry_count"] == 1
    assert live_retry["request_error_count"] == 1
    assert "last_error_type" not in live_retry
    assert live_retry["errors"][0]["error"] == "temporary request failure"
    assert "error_type" not in live_retry["errors"][0]
    assert live_retry["errors"][0]["next_retry_delay_seconds"] == 15
    assert sleep_delays == [15]


def test_langgraph_agent_live_trace_records_global_exploration_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    trace_updates: list[dict[str, object]] = []

    def fail_explore(self, *, context_dir, task_id="", llm_enabled=True):
        raise RuntimeError("synthetic profiling failure")

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fail_explore,
    )
    model = ScriptedToolCallingModel(
        responses=[
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
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(max_agent_steps=0),
        ),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    fail_steps = [
        update
        for update in trace_updates
        if update["steps"][-1]["node"] == "global_data_exploration"
        and update["steps"][-1]["ok"] is False
        and update["steps"][-1].get("status") != "in_progress"
    ]
    assert fail_steps
    fail_step = fail_steps[0]["steps"][-1]
    assert "synthetic profiling failure" in fail_step["tool_results"][0]["error"]


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
    assert result.steps[0].model_response is not None
    request_retry = result.steps[0].model_response["request_retry"]
    assert request_retry["status"] == "failed_after_retries"
    assert request_retry["retry_count"] == 4
    assert request_retry["request_error_count"] == 5
    assert "last_error_type" not in request_retry
    assert request_retry["errors"][-1]["error"] == "temporary request failure 5"
    assert "error_type" not in request_retry["errors"][-1]
    assert request_retry["errors"][-1]["will_retry"] is False


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


def test_langgraph_agent_adds_reasoning_content_after_tool_results(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "I should inspect available files before answering."},
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
    assert len(model.invocations) == 2
    second_request_messages = model.invocations[1]
    assert second_request_messages[-1].content.startswith("Previous model reasoning_content from the last turn")
    assert "I should inspect available files before answering." in second_request_messages[-1].content
    assert result.steps[2].model_request is not None
    assert result.steps[2].model_request["last_message"]["content_preview"].startswith(
        "Previous model reasoning_content from the last turn"
    )


def test_langgraph_agent_repair_includes_previous_reasoning_content(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "I need to read doc/budget.md next.\n"
                        "<tool_call><function=read_doc><parameter=path>doc/budget.md</parameter></function></tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_doc",
                        "args": {"path": "doc/budget.md"},
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
        config=LangGraphAgentConfig(max_steps=6, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps[:4]] == ["model", "repair", "model", "tool"]
    assert len(model.invocations) >= 2
    repair_request_messages = model.invocations[1]
    assert repair_request_messages[-1].content.startswith("Your previous response stopped with no executable tool call.")
    assert "doc/budget.md" in repair_request_messages[-1].content
    assert "pseudo tool call" in repair_request_messages[-1].content


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
        "Your previous response stopped with no executable tool call."
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
