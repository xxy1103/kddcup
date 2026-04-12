from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_agent_baseline import cli as cli_module
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig, SubmissionConfig
from data_agent_baseline import config as config_module
from data_agent_baseline.run import submission as submission_module

runner = CliRunner()


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


def _write_submission_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_load_submission_config_from_env_ignores_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    submission_config_path = tmp_path / "submission.yaml"
    _write_submission_config(
        submission_config_path,
        """
        agent:
          max_steps: 20
          temperature: 0.1
          enable_thinking: false
        run:
          max_workers: 2
          task_timeout_seconds: 30
          soft_runtime_limit_seconds: 900
        """,
    )

    monkeypatch.setenv("DABENCH_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("DABENCH_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("DABENCH_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("DABENCH_SUBMISSION_CONFIG", str(submission_config_path))
    monkeypatch.setenv("MODEL_API_URL", "http://submission-model/v1")
    monkeypatch.setenv("MODEL_API_KEY", "submission-key")
    monkeypatch.setenv("MODEL_NAME", "submission-model")
    monkeypatch.setenv("DABENCH_MAX_STEPS", "24")
    monkeypatch.setenv("DABENCH_TEMPERATURE", "0.2")
    monkeypatch.setenv("DABENCH_ENABLE_THINKING", "true")
    monkeypatch.setenv("DABENCH_MAX_WORKERS", "3")
    monkeypatch.setenv("DABENCH_TASK_TIMEOUT_SECONDS", "45")
    monkeypatch.setattr(
        config_module,
        "_dotenv_value",
        lambda dotenv_path, env_var_name: (_ for _ in ()).throw(AssertionError("dotenv should not be read")),
    )

    config = config_module.load_submission_config_from_env()

    assert config.input_root == input_root
    assert config.output_dir == tmp_path / "output"
    assert config.log_dir == tmp_path / "logs"
    assert config.parameter_config_path == submission_config_path
    assert config.app_config.agent.api_base == "http://submission-model/v1"
    assert config.app_config.agent.api_key == "submission-key"
    assert config.app_config.agent.model == "submission-model"
    assert config.app_config.agent.max_steps == 24
    assert config.app_config.agent.temperature == pytest.approx(0.2)
    assert config.app_config.agent.enable_thinking is True
    assert config.app_config.run.max_workers == 3
    assert config.app_config.run.task_timeout_seconds == 45
    assert config.app_config.run.soft_runtime_limit_seconds == 900


def test_load_submission_config_uses_yaml_for_non_sensitive_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    submission_config_path = tmp_path / "submission.yaml"
    _write_submission_config(
        submission_config_path,
        """
        agent:
          max_steps: 28
          temperature: 0.4
          enable_thinking: true
        run:
          max_workers: 6
          task_timeout_seconds: 123
          soft_runtime_limit_seconds: 456
        """,
    )

    monkeypatch.setenv("DABENCH_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("DABENCH_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("DABENCH_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("DABENCH_SUBMISSION_CONFIG", str(submission_config_path))
    monkeypatch.setenv("MODEL_API_URL", "http://submission-model/v1")
    monkeypatch.setenv("MODEL_API_KEY", "submission-key")
    monkeypatch.setenv("MODEL_NAME", "submission-model")
    monkeypatch.delenv("DABENCH_MAX_STEPS", raising=False)
    monkeypatch.delenv("DABENCH_TEMPERATURE", raising=False)
    monkeypatch.delenv("DABENCH_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("DABENCH_MAX_WORKERS", raising=False)
    monkeypatch.delenv("DABENCH_TASK_TIMEOUT_SECONDS", raising=False)

    config = config_module.load_submission_config_from_env()

    assert config.app_config.agent.max_steps == 28
    assert config.app_config.agent.temperature == pytest.approx(0.4)
    assert config.app_config.agent.enable_thinking is True
    assert config.app_config.run.max_workers == 6
    assert config.app_config.run.task_timeout_seconds == 123
    assert config.app_config.run.soft_runtime_limit_seconds == 456


def test_load_submission_config_env_overrides_yaml_non_sensitive_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    submission_config_path = tmp_path / "submission.yaml"
    _write_submission_config(
        submission_config_path,
        """
        agent:
          max_steps: 28
          temperature: 0.4
          enable_thinking: false
        run:
          max_workers: 6
          task_timeout_seconds: 123
          soft_runtime_limit_seconds: 456
        """,
    )

    monkeypatch.setenv("DABENCH_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("DABENCH_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("DABENCH_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("DABENCH_SUBMISSION_CONFIG", str(submission_config_path))
    monkeypatch.setenv("MODEL_API_URL", "http://submission-model/v1")
    monkeypatch.setenv("MODEL_API_KEY", "submission-key")
    monkeypatch.setenv("MODEL_NAME", "submission-model")
    monkeypatch.setenv("DABENCH_MAX_STEPS", "40")
    monkeypatch.setenv("DABENCH_TEMPERATURE", "0.9")
    monkeypatch.setenv("DABENCH_ENABLE_THINKING", "true")
    monkeypatch.setenv("DABENCH_MAX_WORKERS", "8")
    monkeypatch.setenv("DABENCH_TASK_TIMEOUT_SECONDS", "222")
    monkeypatch.setenv("DABENCH_SOFT_RUNTIME_LIMIT_SECONDS", "999")

    config = config_module.load_submission_config_from_env()

    assert config.app_config.agent.max_steps == 40
    assert config.app_config.agent.temperature == pytest.approx(0.9)
    assert config.app_config.agent.enable_thinking is True
    assert config.app_config.run.max_workers == 8
    assert config.app_config.run.task_timeout_seconds == 222
    assert config.app_config.run.soft_runtime_limit_seconds == 999


def test_load_submission_config_rejects_sensitive_yaml_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    submission_config_path = tmp_path / "submission.yaml"
    _write_submission_config(
        submission_config_path,
        """
        agent:
          model: should-not-be-here
          max_steps: 28
        """,
    )

    monkeypatch.setenv("DABENCH_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("DABENCH_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("DABENCH_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("DABENCH_SUBMISSION_CONFIG", str(submission_config_path))
    monkeypatch.setenv("MODEL_API_URL", "http://submission-model/v1")
    monkeypatch.setenv("MODEL_API_KEY", "submission-key")
    monkeypatch.setenv("MODEL_NAME", "submission-model")

    with pytest.raises(ValueError, match="non-sensitive `agent` keys"):
        config_module.load_submission_config_from_env()


def test_run_submission_writes_prediction_and_logs_without_dev_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    _create_task(input_root, "task_2")
    _create_task(input_root, "task_alpha")

    config = SubmissionConfig(
        app_config=AppConfig(
            dataset=DatasetConfig(root_path=input_root),
            agent=AgentConfig(max_steps=12, temperature=0.0),
            run=RunConfig(output_dir=output_root, max_workers=1, task_timeout_seconds=60),
        ),
        log_dir=log_root,
    )

    def fake_execute_task(*, task_id: str, config: AppConfig, model=None, tools=None) -> dict[str, object]:
        del config, model, tools
        if task_id == "task_2":
            return {
                "task_id": task_id,
                "answer": {"columns": ["value"], "rows": [["ok"]]},
                "failure_reason": None,
                "succeeded": True,
                "e2e_elapsed_seconds": 1.5,
            }
        return {
            "task_id": task_id,
            "answer": None,
            "failure_reason": "synthetic failure",
            "succeeded": False,
            "e2e_elapsed_seconds": 2.5,
        }

    monkeypatch.setattr(submission_module, "execute_task", fake_execute_task)

    artifacts = submission_module.run_submission(config=config, model=object(), tools=object())

    success_prediction = output_root / "task_2" / "prediction.csv"
    assert success_prediction.exists()
    with success_prediction.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["value"], ["ok"]]
    assert not (output_root / "task_2" / "trace.json").exists()
    assert not (output_root / "task_alpha" / "prediction.csv").exists()
    assert not (output_root / "summary.json").exists()

    assert artifacts.task_count == 2
    assert artifacts.succeeded_task_count == 1
    assert artifacts.failed_task_count == 1
    assert artifacts.runtime_log_path.exists()
    assert artifacts.task_status_path.exists()
    assert artifacts.run_summary_path.exists()

    summary_payload = json.loads(artifacts.run_summary_path.read_text(encoding="utf-8"))
    assert summary_payload["task_count"] == 2
    assert summary_payload["succeeded_task_count"] == 1
    assert summary_payload["failed_task_count"] == 1
    assert summary_payload["soft_runtime_limit_seconds"] == 42300
    assert summary_payload["tasks"][0]["task_id"] == "task_2"
    assert "Question for" not in artifacts.runtime_log_path.read_text(encoding="utf-8")


def test_run_submission_stops_launching_new_tasks_after_soft_runtime_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    _create_task(input_root, "task_1")
    _create_task(input_root, "task_2")
    _create_task(input_root, "task_3")

    config = SubmissionConfig(
        app_config=AppConfig(
            dataset=DatasetConfig(root_path=input_root),
            agent=AgentConfig(max_steps=12, temperature=0.0),
            run=RunConfig(
                output_dir=output_root,
                max_workers=1,
                task_timeout_seconds=60,
                soft_runtime_limit_seconds=1,
            ),
        ),
        log_dir=log_root,
    )

    perf_counter_values = iter([0.0, 0.1, 2.0, 2.1])
    monkeypatch.setattr(submission_module, "perf_counter", lambda: next(perf_counter_values))

    def fake_execute_task(*, task_id: str, config: AppConfig, model=None, tools=None) -> dict[str, object]:
        del config, model, tools
        return {
            "task_id": task_id,
            "answer": {"columns": ["value"], "rows": [[task_id]]},
            "failure_reason": None,
            "succeeded": True,
            "e2e_elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(submission_module, "execute_task", fake_execute_task)

    artifacts = submission_module.run_submission(config=config, model=object(), tools=object())

    assert artifacts.task_count == 1
    assert artifacts.tasks[0].task_id == "task_1"
    summary_payload = json.loads(artifacts.run_summary_path.read_text(encoding="utf-8"))
    assert summary_payload["skipped_task_count"] == 2
    assert summary_payload["drain_reason"] == "soft_runtime_limit"


def test_submit_command_missing_required_model_env_writes_runtime_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"

    monkeypatch.setenv("DABENCH_INPUT_ROOT", str(input_root))
    monkeypatch.setenv("DABENCH_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("DABENCH_LOG_ROOT", str(log_root))
    monkeypatch.delenv("MODEL_API_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    result = runner.invoke(cli_module.app, ["submit"])

    assert result.exit_code == 1
    runtime_log = log_root / "runtime.log"
    assert runtime_log.exists()
    assert "Missing required environment variable" in runtime_log.read_text(encoding="utf-8")


def test_dataset_task_sorting_falls_back_to_lexicographic_for_non_numeric_suffixes(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _create_task(input_root, "task_10")
    _create_task(input_root, "task_alpha")
    _create_task(input_root, "task_2")

    dataset = DABenchPublicDataset(input_root)

    assert dataset.list_task_ids() == ["task_2", "task_10", "task_alpha"]
