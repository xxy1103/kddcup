from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
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
    assert summary_payload["succeeded_task_count"] == 1


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
    ) -> dict[str, object]:
        del config, model, tools
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
