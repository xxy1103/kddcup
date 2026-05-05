from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig, load_app_config
from data_agent_baseline.run import runner as runner_module
from data_agent_baseline.run.runner import TaskRunArtifacts, run_benchmark


def _create_task(input_root: Path, task_id: str, difficulty: str = "easy") -> None:
    task_dir = input_root / task_id
    (task_dir / "context").mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "question": f"Question for {task_id}",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class ScriptedToolCallingModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def bind_tools(self, tools, tool_choice="auto", parallel_tool_calls=False):  # noqa: ANN001
        del tools, tool_choice, parallel_tool_calls
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        if not self._responses:
            raise RuntimeError("No scripted responses remaining.")
        return self._responses.pop(0)


def test_run_benchmark_summary_includes_runtime_and_agent_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2", difficulty="hard")

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=48, temperature=0.3),
        run=RunConfig(
            output_dir=output_root,
            run_id="summary-test-run",
            max_workers=7,
            task_timeout_seconds=321,
        ),
    )

    def fake_run_single_task(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        prediction_output_root: Path | None = None,
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
        del prediction_output_root
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = task_output_dir / "trace.json"
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=(task_id == "task_1"),
            failure_reason=None if task_id == "task_1" else "failed",
        )

    monkeypatch.setattr(runner_module, "run_single_task", fake_run_single_task)

    run_output_dir, artifacts = run_benchmark(config=config, model=object())

    assert run_output_dir.name == "summary-test-run"
    assert len(artifacts) == 2

    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["max_workers"] == 1
    assert summary_payload["task_timeout_seconds"] == 321
    assert summary_payload["max_steps"] == 48
    assert summary_payload["temperature"] == 0.3
    assert summary_payload["model_request_timeout_seconds"] == 60.0
    assert summary_payload["succeeded_task_count"] == 1
    assert summary_payload["output_layout"] == "run_dir"
    assert (run_output_dir / "task_status.jsonl").exists()


def test_load_app_config_parses_model_request_timeout_seconds(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  model_request_timeout_seconds: 12.5
""",
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.agent.model_request_timeout_seconds == 12.5


def test_run_benchmark_uses_configured_task_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2")
    _create_task(dataset_root, "task_3")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            run_id="selected-task-run",
            max_workers=1,
            task_ids=("task_2", "task_3"),
        ),
    )
    attempted_task_ids: list[str] = []

    def fake_run_single_task(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        prediction_output_root: Path | None = None,
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
        del config, prediction_output_root, model, tools
        attempted_task_ids.append(task_id)
        task_output_dir = run_output_dir / task_id
        trace_path = task_output_dir / "trace.json"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(runner_module, "run_single_task", fake_run_single_task)

    _, artifacts = run_benchmark(config=config, model=object())

    assert attempted_task_ids == ["task_2", "task_3"]
    assert [artifact.task_id for artifact in artifacts] == ["task_2", "task_3"]


def test_run_benchmark_flat_layout_writes_predictions_to_output_and_logs_to_log_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    _create_task(dataset_root, "task_1")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            log_dir=log_root,
            output_layout="flat",
            run_id="docker-style-run",
            max_workers=1,
        ),
    )

    def fake_execute_task(
        *,
        task_id: str,
        config: AppConfig,
        model=None,
        tools=None,
        trace_path=None,
    ) -> dict[str, object]:
        del config, model, tools, trace_path
        return {
            "task_id": task_id,
            "answer": {"columns": ["value"], "rows": [["ok"]]},
            "steps": [{"node": "model"}],
            "failure_reason": None,
            "succeeded": True,
        }

    monkeypatch.setattr(runner_module, "execute_task", fake_execute_task)

    run_output_dir, artifacts = run_benchmark(config=config, model=object(), tools=object())

    assert run_output_dir == log_root / "docker-style-run"
    assert artifacts[0].prediction_csv_path == output_root / "task_1" / "prediction.csv"
    assert (output_root / "task_1" / "prediction.csv").exists()
    assert not (output_root / "summary.json").exists()
    assert not (output_root / "task_1" / "trace.json").exists()
    assert (log_root / "docker-style-run" / "summary.json").exists()
    assert (log_root / "docker-style-run" / "task_status.jsonl").exists()
    assert (log_root / "docker-style-run" / "task_1" / "trace.json").exists()


def test_run_benchmark_writes_trace_and_summary_with_explicit_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            run_id="utf8-trace-run",
            max_workers=1,
            task_timeout_seconds=60,
        ),
    )

    def fake_execute_task(
        *,
        task_id: str,
            config: AppConfig,
            model=None,
            tools=None,
            trace_path=None,
        ) -> dict[str, object]:
        del config, model, tools, trace_path
        return {
            "task_id": task_id,
            "answer": None,
            "steps": [{"content": "emoji 🙂 and symbol \u2260"}],
            "failure_reason": None,
            "succeeded": True,
        }

    write_encodings: list[str | None] = []
    real_open = Path.open

    def tracking_open(self: Path, mode: str = "r", *args, **kwargs):
        if "w" in mode and self.is_relative_to(output_root):
            write_encodings.append(kwargs.get("encoding"))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(runner_module, "execute_task", fake_execute_task)
    monkeypatch.setattr(Path, "open", tracking_open)

    run_output_dir, artifacts = run_benchmark(config=config, model=object(), tools=object())

    assert len(artifacts) == 1
    trace_payload = json.loads((run_output_dir / "task_1" / "trace.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert trace_payload["steps"][0]["content"] == "emoji 🙂 and symbol \u2260"
    assert summary_payload["tasks"][0]["task_id"] == "task_1"
    assert write_encodings
    assert set(write_encodings) == {"utf-8"}


def test_live_trace_writer_rewrites_valid_json_each_update(tmp_path: Path) -> None:
    trace_path = tmp_path / "task_1" / "trace.json"
    writer = runner_module.LiveTraceWriter(trace_path=trace_path, task_id="task_1")

    writer.start()
    first_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    writer.update({"steps": [{"node": "model"}], "failure_reason": "still running"})
    second_payload = json.loads(trace_path.read_text(encoding="utf-8"))

    assert first_payload["task_id"] == "task_1"
    assert first_payload["partial"] is True
    assert first_payload["steps"] == []
    assert second_payload["steps"] == [{"node": "model"}]
    assert second_payload["failure_reason"] == "still running"
    assert second_payload["partial"] is True


def test_write_task_outputs_preserves_partial_trace_steps_for_empty_failure(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    trace_path = run_output_dir / "task_1" / "trace.json"
    runner_module._write_json(
        trace_path,
        {
            "task_id": "task_1",
            "answer": None,
            "steps": [{"step_index": 1, "node": "model"}],
            "failure_reason": None,
            "succeeded": False,
            "inspector": {"perception": {"ok": True}},
            "partial": True,
        },
    )

    artifact = runner_module._write_task_outputs(
        "task_1",
        run_output_dir,
        {
            "task_id": "task_1",
            "answer": None,
            "steps": [],
            "failure_reason": "Task timed out after 60 seconds.",
            "succeeded": False,
        },
    )

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert artifact.succeeded is False
    assert trace_payload["steps"] == [{"step_index": 1, "node": "model"}]
    assert trace_payload["failure_reason"] == "Task timed out after 60 seconds."
    assert trace_payload["succeeded"] is False
    assert trace_payload["finalized_from_partial_trace"] is True
    assert "partial" not in trace_payload
    assert not (run_output_dir / "task_1" / "prediction.csv").exists()


def test_write_task_outputs_success_overwrites_partial_trace(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    trace_path = run_output_dir / "task_1" / "trace.json"
    runner_module._write_json(
        trace_path,
        {
            "task_id": "task_1",
            "answer": None,
            "steps": [{"step_index": 1, "node": "old"}],
            "failure_reason": None,
            "succeeded": False,
            "partial": True,
        },
    )

    runner_module._write_task_outputs(
        "task_1",
        run_output_dir,
        {
            "task_id": "task_1",
            "answer": {"columns": ["status"], "rows": [["ok"]]},
            "steps": [{"step_index": 1, "node": "model"}],
            "failure_reason": None,
            "succeeded": True,
        },
    )

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_payload["steps"] == [{"step_index": 1, "node": "model"}]
    assert trace_payload["succeeded"] is True
    assert "partial" not in trace_payload
    assert "finalized_from_partial_trace" not in trace_payload
    assert (run_output_dir / "task_1" / "prediction.csv").exists()


def test_run_benchmark_preserves_trace_for_max_steps_failure(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=1, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            run_id="max-steps-trace-run",
            max_workers=1,
            task_timeout_seconds=60,
        ),
    )
    model = ScriptedToolCallingModel(
        responses=[AIMessage(content="I should think more before using a tool.", tool_calls=[])]
    )

    run_output_dir, artifacts = run_benchmark(config=config, model=model)

    trace_payload = json.loads((run_output_dir / "task_1" / "trace.json").read_text(encoding="utf-8"))
    assert len(artifacts) == 1
    assert artifacts[0].succeeded is False
    assert trace_payload["failure_reason"] == "Agent did not submit an answer within max_steps."
    assert [step["node"] for step in trace_payload["steps"]] == ["model", "react"]
    assert "partial" not in trace_payload


def test_write_json_is_atomic_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_path = tmp_path / "trace.json"
    target_path.write_text('{"status":"old"}\n', encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError(f"synthetic replace failure for {target.name}")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        runner_module._write_json(target_path, {"status": "new"})

    assert target_path.read_text(encoding="utf-8") == '{"status":"old"}\n'
    assert list(tmp_path.iterdir()) == [target_path]
