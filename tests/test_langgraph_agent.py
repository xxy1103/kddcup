from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import DataInspectorConfig, ToolConfig
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


def test_langgraph_agent_validates_answer_before_final_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
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

    monkeypatch.setattr("data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel)
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "tool", "validate_answer"]
    assert result.steps[-1].ok is True
    assert trace_updates[-2]["steps"][-1]["node"] == "validate_answer"
    assert "status" not in trace_updates[-2]["steps"][-1]
    assert trace_updates[-1]["partial"] is False
    assert trace_updates[-1]["steps"][-1]["node"] == "validate_answer"
    assert trace_updates[-1]["succeeded"] is True


def test_langgraph_agent_validation_failure_returns_to_model_step(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["extra"], "rows": [["bad"]]},
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
    validation_results = [
        {"valid": False, "issues": ["extra column"], "raw_response": '{"valid": false}'},
        {"valid": True, "issues": [], "raw_response": '{"valid": true, "issues": []}'},
    ]

    monkeypatch.setattr("data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel)
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator",
        lambda **_: validation_results.pop(0),
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=4),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == [
        "model",
        "tool",
        "validate_answer",
        "model",
        "tool",
        "validate_answer",
    ]
    assert result.steps[2].ok is False
    assert result.steps[-1].ok is True
    assert model.invocations[1][-1].content.startswith("Your submitted answer did NOT pass")


def test_langgraph_agent_accepts_answer_when_validator_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
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

    def fail_validator(**_: object) -> dict[str, object]:
        raise RuntimeError("validator unavailable")

    monkeypatch.setattr("data_agent_baseline.agents.langgraph_runtime.BaseChatModel", ScriptedToolCallingModel)
    monkeypatch.setattr("data_agent_baseline.agents.langgraph_runtime.invoke_answer_validator", fail_validator)
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(max_steps=2),
        trace_callback=trace_updates.append,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.steps[-1].node == "validate_answer"
    assert result.steps[-1].ok is False
    assert result.steps[-1].tool_results[0]["error"] == "validator unavailable"
    assert trace_updates[-1]["partial"] is False
    assert trace_updates[-1]["steps"][-1]["tool_results"][0]["error"] == "validator unavailable"


def test_langgraph_agent_receives_problem_in_sft_aligned_user_message(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task = _create_task(tmp_path)

    def fake_explore(self, *, context_dir, task_id=""):  # noqa: ANN001
        return '{"task_id": "task_demo", "assets": [{"path": "sample.csv", "kind": "csv", "size": 10}], "schemas": [{"asset_path": "sample.csv", "kind": "csv", "fields": [{"name": "value", "type": "integer"}]}], "knowledge_documents": []}'

    def fake_analyze_question(*, model, question):  # noqa: ANN001
        return {
            "entities": ["value"],
            "filters": [],
            "requested_output": "value column",
        }

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fake_explore,
    )
    monkeypatch.setattr(
        "data_agent_baseline.agents.langgraph_runtime.analyze_question",
        fake_analyze_question,
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "answer",
                        "args": {"columns": ["value"], "rows": [["1"], ["2"]]},
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
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            enable_question_analysis=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    first_request = model.invocations[0]
    assert [message.type for message in first_request] == ["system", "human"]
    user_content = first_request[1].content
    user_query_index = user_content.index("<user_query>")
    context_index = user_content.index("<context_injection>")
    analysis_index = user_content.index("<question_analysis>")
    catalog_index = user_content.index("<data_catalog>")
    action_index = user_content.index("<action_trigger>")
    assert user_query_index < context_index < analysis_index < catalog_index < action_index
    assert "User Question: List the value column." in user_content
    assert "\nQuestion: List the value column." not in user_content
    assert "## Task Input" not in user_content
    assert "## Problem Analysis" not in user_content
    assert "## Global Data Profile" not in user_content
    assert "## Action Instruction" not in user_content
    assert "All tool file paths are relative" not in user_content
    assert "Each turn should make progress" not in user_content
    assert "call `answer`" not in user_content
    assert "<data_catalog>" in user_content
    assert "</data_catalog>" in user_content
    assert '"path": "sample.csv"' in user_content
    assert "<question_analysis>" in user_content
    assert "</question_analysis>" in user_content
    assert '"requested_output": "value column"' in user_content
    assert "formulate your first thought and execute the most appropriate tool" in user_content


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

    def fail_explore(self, *, context_dir, task_id=""):
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
            data_inspector=DataInspectorConfig(),
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


def test_langgraph_agent_truncates_tool_step_results_for_trace_and_messages(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python",
                        "args": {"code": "print('x' * 200)"},
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
        tools=create_default_tool_registry(ToolConfig(max_output_chars=40, max_list_items=200)),
        config=LangGraphAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    python_tool_step = next(step for step in result.steps if step.node == "tool")
    output = python_tool_step.tool_results[0]["content"]["output"]
    assert output.startswith("x" * 40)
    assert "内容已被截断" in output
    assert len(output) < 200

    second_request_messages = model.invocations[1]
    tool_message = next(message for message in second_request_messages if getattr(message, "name", None) == "execute_python")
    tool_payload = json.loads(str(tool_message.content))
    assert tool_payload["content"]["output"] == output
    assert "内容已被截断" in tool_payload["content"]["output"]


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
    assert first_step.assistant_message == "I should inspect the context first."
    assert first_step.model_response is not None
    assert first_step.model_response["reasoning_content"] == "I should inspect the context first."
    assert first_step.model_response["reasoning_content_length"] == 35


def test_langgraph_agent_adds_clean_reasoning_content_to_history(tmp_path: Path) -> None:
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
    assert [message.type for message in second_request_messages] == ["system", "human", "ai", "tool"]
    assistant_message = second_request_messages[-2]
    tool_message = second_request_messages[-1]
    assert isinstance(tool_message, ToolMessage)
    assert assistant_message.content == "I should inspect available files before answering."
    assert assistant_message.tool_calls[0]["name"] == "list_context"
    assert assistant_message.additional_kwargs["reasoning_content"] == (
        "I should inspect available files before answering."
    )
    assert all(
        not str(getattr(message, "content", "")).startswith("Previous model reasoning_content from the last turn")
        for message in second_request_messages
    )
    assert result.steps[2].model_request is not None
    assert result.steps[2].model_request["last_message"]["type"] == "tool"


def test_langgraph_agent_recovers_pseudo_tool_call_from_reasoning_content(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "I need to read notes.md next.\n"
                        "<tool_call><function=read_doc><parameter=path>notes.md</parameter></function></tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
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
    assert [step.node for step in result.steps] == ["model", "tool", "model", "tool"]
    assert result.steps[0].tool_calls == [
        {
            "id": result.steps[0].tool_calls[0]["id"],
            "name": "read_doc",
            "args": {"path": "notes.md"},
        }
    ]
    assert result.steps[0].model_response is not None
    assert result.steps[0].model_response["recovered_tool_call"] is True
    assert result.steps[0].model_response["recovered_tool_call_source"] == "reasoning_content"
    assert result.steps[0].model_response["recovered_tool_call_name"] == "read_doc"
    second_request_messages = model.invocations[1]
    assert [message.type for message in second_request_messages] == ["system", "human", "ai", "tool"]
    assert second_request_messages[-2].content == "I need to read notes.md next."
    assert "<tool_call>" not in second_request_messages[-2].content
    assert second_request_messages[-2].tool_calls[0]["name"] == "read_doc"
    assert second_request_messages[-1].name == "read_doc"


def test_langgraph_agent_recovers_pseudo_sql_tool_call_with_typed_args(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "<tool_call>\n"
                        "<function=execute_context_sql>\n"
                        "<parameter=path>\ndb/example.db\n</parameter>\n"
                        "<parameter=sql>\nSELECT COUNT(*) FROM demo\n</parameter>\n"
                        "<parameter=limit>\n50\n</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
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
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "tool", "model", "tool"]
    recovered_args = result.steps[0].tool_calls[0]["args"]
    assert recovered_args == {
        "path": "db/example.db",
        "sql": "SELECT COUNT(*) FROM demo",
        "limit": 50,
    }
    assert isinstance(recovered_args["limit"], int)


def test_langgraph_agent_recovers_multiline_python_pseudo_tool_call(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    code = "value = 1\nprint(value)\n"
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "<tool_call>\n"
                        "<function=execute_python>\n"
                        f"<parameter=code>\n{code}</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
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
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert [step.node for step in result.steps] == ["model", "tool", "model", "tool"]
    assert result.steps[0].tool_calls[0]["name"] == "execute_python"
    assert result.steps[0].tool_calls[0]["args"]["code"] == code.strip()


def test_langgraph_agent_does_not_recover_unknown_pseudo_tool_call(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "I should use the tool next.\n"
                        "<tool_call><function=unknown_tool><parameter=path>notes.md</parameter></function></tool_call>"
                    )
                },
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
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
    assert result.steps[0].model_response is not None
    assert "recovered_tool_call" not in result.steps[0].model_response
    assert model.invocations[1][-2].type == "ai"
    assert model.invocations[1][-2].content == "I should use the tool next."
    assert "<tool_call>" not in model.invocations[1][-2].content
    assert model.invocations[1][-1].type == "system"


def test_langgraph_agent_does_not_override_native_tool_call_with_pseudo_tool_call(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "<tool_call><function=execute_python><parameter=code>print('wrong')</parameter></function></tool_call>"
                    )
                },
                tool_calls=[
                    {"name": "read_doc", "args": {"path": "notes.md"}, "id": "call_1", "type": "tool_call"}
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
        config=LangGraphAgentConfig(max_steps=4, empty_stop_retry_limit=1),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.steps[0].tool_calls[0]["name"] == "read_doc"
    assert result.steps[0].model_response is not None
    assert "recovered_tool_call" not in result.steps[0].model_response


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
    assert second_model_step.model_request["last_message"]["type"] == "system"
    assert second_model_step.model_request["last_message"]["content_preview"].startswith(
        "Your previous response did not call a tool."
    )
    assert "immediately call a tool" in model.invocations[1][-1].content
    assert second_model_step.model_request["last_message"]["content_length"] == len(
        model.invocations[1][-1].content
    )


def test_langgraph_agent_retries_after_non_tool_stop_with_content(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="I should inspect the context next, then continue.",
                response_metadata={"finish_reason": "stop"},
                tool_calls=[],
            ),
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
    assert result.steps[0].assistant_message == "I should inspect the context next, then continue."
    assert "immediately call a tool" in model.invocations[1][-1].content


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
