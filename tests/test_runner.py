from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
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
        agent=AgentConfig(
            max_steps=48,
            temperature=0.3,
            strip_reasoning_history=True,
            reasoning_history_limit=2,
        ),
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
    assert summary_payload["strip_reasoning_history"] is True
    assert summary_payload["reasoning_history_limit"] == 2
    assert summary_payload["succeeded_task_count"] == 1
    assert summary_payload["output_layout"] == "run_dir"
    assert (run_output_dir / "task_status.jsonl").exists()


def test_load_app_config_accepts_empty_reasoning_history_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  reasoning_history_limit:
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.agent.reasoning_history_limit is None


def test_load_app_config_rejects_negative_reasoning_history_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  reasoning_history_limit: -1
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    with pytest.raises(ValueError, match="agent.reasoning_history_limit"):
        load_app_config(config_path)


def test_load_app_config_supports_process_validator_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent:\n  model: test-model\n", encoding="utf-8")

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.agent.enable_process_validator is False
    assert config.process_validator.checkpoint_model_interval == 10
    assert config.process_validator.retry_limit == 5
    assert config.process_validator.recent_step_limit == 8


def test_load_app_config_supports_process_validator_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  enable_process_validator: true
process_validator:
  checkpoint_model_interval: 5
  retry_limit: 2
  recent_step_limit: 12
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.agent.enable_process_validator is True
    assert config.process_validator.checkpoint_model_interval == 5
    assert config.process_validator.retry_limit == 2
    assert config.process_validator.recent_step_limit == 12


def test_task_timeout_wrapper_keeps_result_when_subprocess_cleanup_lags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(output_dir=output_root, task_timeout_seconds=60),
    )

    class FakeQueue:
        def get(self, timeout):  # noqa: ANN001
            assert timeout == 60
            return {
                "ok": True,
                "run_result": {
                    "task_id": "task_1",
                    "answer": {"columns": ["status"], "rows": [["ok"]]},
                    "steps": [{"node": "validate_answer"}],
                    "failure_reason": None,
                    "succeeded": True,
                },
            }

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeProcess:
        def __init__(self, target, args):  # noqa: ANN001
            self.target = target
            self.args = args
            self.terminated = False
            self.killed = False

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:  # noqa: ANN001
            pass

        def is_alive(self) -> bool:
            return not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.terminated = True

    class FakeContext:
        def Queue(self):  # noqa: N802
            return FakeQueue()

        def Process(self, target, args):  # noqa: N802, ANN001
            return FakeProcess(target, args)

    monkeypatch.setattr(runner_module.multiprocessing, "get_context", lambda _: FakeContext())

    result = runner_module._run_single_task_with_timeout(task_id="task_1", config=config)

    assert result["succeeded"] is True
    assert result["answer"] == {"columns": ["status"], "rows": [["ok"]]}
    assert "cleanup_warning" in result


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


def test_run_benchmark_writes_summary_when_sequential_run_is_interrupted(
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
        run=RunConfig(output_dir=output_root, run_id="interrupted-run", max_workers=1),
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
        del config, prediction_output_root, model, tools
        if task_id == "task_2":
            raise KeyboardInterrupt
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = task_output_dir / "trace.json"
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

    run_output_dir, artifacts = run_benchmark(config=config, model=object())

    assert [artifact.task_id for artifact in artifacts] == ["task_1", "task_2", "task_3"]
    assert artifacts[0].succeeded is True
    assert [artifact.failure_reason for artifact in artifacts[1:]] == [
        runner_module.INTERRUPTED_FAILURE_REASON,
        runner_module.INTERRUPTED_FAILURE_REASON,
    ]
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["interrupted"] is True
    assert summary_payload["succeeded_task_count"] == 1
    assert [task["failure_reason"] for task in summary_payload["tasks"][1:]] == [
        runner_module.INTERRUPTED_FAILURE_REASON,
        runner_module.INTERRUPTED_FAILURE_REASON,
    ]
    assert (run_output_dir / "task_status.jsonl").exists()


def test_run_benchmark_writes_summary_when_parallel_run_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(output_dir=output_root, run_id="parallel-interrupted-run", max_workers=2),
    )

    class FakeFuture:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            self.cancelled = False

        def cancel(self) -> bool:
            self.cancelled = True
            return True

    class FakeExecutor:
        futures: list[FakeFuture] = []
        shutdown_calls: list[dict[str, object]] = []

        def __init__(self, max_workers: int) -> None:
            assert max_workers == 2
            self.futures = []
            FakeExecutor.futures = self.futures
            FakeExecutor.shutdown_calls = []

        def submit(self, fn, **kwargs):  # noqa: ANN001
            del fn
            future = FakeFuture(kwargs["task_id"])
            self.futures.append(future)
            return future

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            FakeExecutor.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    def interrupted_as_completed(_futures):  # noqa: ANN001
        raise KeyboardInterrupt
        yield  # pragma: no cover

    monkeypatch.setattr(runner_module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(runner_module, "as_completed", interrupted_as_completed)

    run_output_dir, artifacts = run_benchmark(config=config)

    assert [artifact.task_id for artifact in artifacts] == ["task_1", "task_2"]
    assert all(artifact.failure_reason == runner_module.INTERRUPTED_FAILURE_REASON for artifact in artifacts)
    assert all(future.cancelled for future in FakeExecutor.futures)
    assert {"wait": False, "cancel_futures": True} in FakeExecutor.shutdown_calls
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["interrupted"] is True
    assert summary_payload["succeeded_task_count"] == 0


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
    assert (output_root / "task_1" / "trace.json").exists()
    assert (log_root / "docker-style-run" / "summary.json").exists()
    assert (log_root / "docker-style-run" / "task_status.jsonl").exists()
    assert not (log_root / "docker-style-run" / "task_1" / "trace.json").exists()


def test_run_benchmark_skip_completed_uses_flat_prediction_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2")
    completed_prediction_path = output_root / "task_1" / "prediction.csv"
    completed_prediction_path.parent.mkdir(parents=True)
    completed_prediction_path.write_text("value\nalready-done\n", encoding="utf-8")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            log_dir=log_root,
            output_layout="flat",
            run_id="skip-completed-run",
            max_workers=1,
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

    run_output_dir, artifacts = run_benchmark(config=config, model=object(), skip_completed=True)

    assert run_output_dir == log_root / "skip-completed-run"
    assert attempted_task_ids == ["task_2"]
    assert [artifact.task_id for artifact in artifacts] == ["task_2"]
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["skipped_task_count"] == 1
    assert summary_payload["skipped_task_ids"] == ["task_1"]


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
    assert [step["node"] for step in trace_payload["steps"]] == ["model"]
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
