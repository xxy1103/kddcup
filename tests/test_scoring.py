from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_agent_baseline import cli as cli_module
from data_agent_baseline.scoring import normalize_cell, score_run_outputs

runner = CliRunner()


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _create_task(input_root: Path, task_id: str, difficulty: str) -> None:
    task_dir = input_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        task_dir / "task.json",
        {
            "task_id": task_id,
            "difficulty": difficulty,
            "question": f"Question for {task_id}",
        },
    )


def _write_prediction(run_output_dir: Path, task_id: str, columns: list[str], rows: list[list[object]]) -> None:
    _write_csv(run_output_dir / task_id / "prediction.csv", columns, rows)


def _write_trace(
    run_output_dir: Path,
    task_id: str,
    *,
    succeeded: bool,
    failure_reason: str | None,
    elapsed_seconds: float,
    step_count: int,
) -> None:
    _write_json(
        run_output_dir / task_id / "trace.json",
        {
            "task_id": task_id,
            "answer": None,
            "steps": [{"step_index": index + 1} for index in range(step_count)],
            "failure_reason": failure_reason,
            "succeeded": succeeded,
            "e2e_elapsed_seconds": elapsed_seconds,
        },
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("NULL", ""),
        ("  nan  ", ""),
        ("4200000", "4200000.00"),
        ("0.005", "0.01"),
        ("2024-3-1", "2024-03-01"),
        ("2024-03-01T10:30:00+08:00", "2024-03-01T02:30:00Z"),
        ("2024-03-01 10:30:00", "2024-03-01T10:30:00"),
        ("  East Asia\r\n", "East Asia"),
    ],
)
def test_normalize_cell_rules(raw_value: str, expected: str) -> None:
    assert normalize_cell(raw_value) == expected


def test_score_run_outputs_supports_name_field_equivalence_and_prefers_less_redundancy(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["full_name"], [["John Smith"], ["Jane Doe"]])
    _write_prediction(
        run_output_dir,
        "task_1",
        ["first_name", "last_name", "full_name"],
        [["John", "Smith", "John Smith"], ["Jane", "Doe", "Jane Doe"]],
    )

    _create_task(input_root, "task_2", "medium")
    _write_csv(gold_root / "task_2" / "gold.csv", ["first_name", "last_name"], [["Jane", "Doe"], ["John", "Smith"]])
    _write_prediction(run_output_dir, "task_2", ["full_name"], [["John Smith"], ["Jane Doe"]])

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    task_1 = next(task for task in summary.tasks if task.task_id == "task_1")
    assert task_1.full_cover is True
    assert task_1.covered_gold_columns == 1
    assert task_1.matched_prediction_columns == 2
    assert task_1.extra_prediction_columns == 1
    assert task_1.redundancy_rate == pytest.approx(1 / 3)

    task_2 = next(task for task in summary.tasks if task.task_id == "task_2")
    assert task_2.full_cover is True
    assert task_2.covered_gold_columns == 2
    assert task_2.matched_prediction_columns == 1
    assert task_2.extra_prediction_columns == 0
    assert task_2.recall == pytest.approx(1.0)


def test_score_run_outputs_aggregates_metrics_and_generates_report(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["value"], [[1]])
    _write_prediction(run_output_dir, "task_1", ["value"], [[1.004]])
    _write_trace(run_output_dir, "task_1", succeeded=True, failure_reason=None, elapsed_seconds=10.0, step_count=3)

    _create_task(input_root, "task_2", "hard")
    _write_csv(gold_root / "task_2" / "gold.csv", ["full_name"], [["John Smith"], ["Jane Doe"]])
    _write_prediction(
        run_output_dir,
        "task_2",
        ["first_name", "last_name", "full_name"],
        [["John", "Smith", "John Smith"], ["Jane", "Doe", "Jane Doe"]],
    )
    _write_trace(run_output_dir, "task_2", succeeded=True, failure_reason=None, elapsed_seconds=20.0, step_count=5)

    _create_task(input_root, "task_3", "medium")
    _write_csv(gold_root / "task_3" / "gold.csv", ["answer"], [["42"]])
    _write_trace(
        run_output_dir,
        "task_3",
        succeeded=False,
        failure_reason="Agent did not submit an answer within max_steps.",
        elapsed_seconds=90.0,
        step_count=32,
    )

    _write_json(
        run_output_dir / "summary.json",
        {
            "tasks": [
                {"task_id": "task_1", "succeeded": True, "failure_reason": None},
                {"task_id": "task_2", "succeeded": True, "failure_reason": None},
                {
                    "task_id": "task_3",
                    "succeeded": False,
                    "failure_reason": "Agent did not submit an answer within max_steps.",
                },
            ]
        },
    )

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    assert summary.task_count == 3
    assert summary.prediction_task_count == 2
    assert summary.full_cover_count == 2
    assert summary.total_score == 2
    assert summary.full_cover_rate == pytest.approx(2 / 3)
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.mean_recall == pytest.approx(2 / 3)
    assert summary.mean_redundancy_rate == pytest.approx((0 + (1 / 3) + 0) / 3)
    assert summary.proxy_scores["0.5"] == pytest.approx((1 + (1 - (0.5 / 3)) + 0) / 3)
    assert summary.failure_breakdown == {"Agent did not submit an answer within max_steps.": 1}
    assert summary.difficulty_breakdown["easy"]["full_cover_count"] == 1
    assert summary.runtime_summary["available_runtime_count"] == 3
    assert summary.runtime_summary["max_step_count"] == 32
    assert summary.score_path.exists()
    assert summary.score_report_path.exists()
    report_text = summary.score_report_path.read_text(encoding="utf-8")
    assert "多 λ 代理分数" in report_text
    assert "| 任务 | 难度 | Gold列数 | 预测列数 | 覆盖Gold列数 | 冗余列数 | Recall | Redundancy | Full Cover | 耗时(秒) | 失败/备注 |" in report_text
    assert "| task_1 | easy | 1 | 1 | 1 | 0 | 1.0000 | 0.0000 | yes | 10.000 | - |" in report_text
    assert "| task_3 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | no | 90.000 | Agent did not submit an answer within max_steps. |" in report_text

    score_payload = json.loads(summary.score_path.read_text(encoding="utf-8"))
    assert score_payload["metadata"]["run_id"] == "sample-run"
    assert score_payload["overview"]["full_cover_count"] == 2
    assert score_payload["total_score"] == 2
    assert score_payload["accuracy"] == pytest.approx(2 / 3)


def test_score_run_outputs_reports_invalid_csv_width(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["a", "b"], [[1, 2]])
    prediction_path = run_output_dir / "task_1" / "prediction.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("a,b\n1\n", encoding="utf-8")

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    assert summary.tasks[0].full_cover is False
    assert "CSV row width mismatch" in (summary.tasks[0].reason or "")


def test_cli_score_run_supports_custom_lambda_and_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "artifacts" / "runs"
    run_output_dir = runs_root / "sample-run"
    gold_root = tmp_path / "data" / "public" / "output"
    input_root = tmp_path / "data" / "public" / "input"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["value"], [[1]])
    _write_prediction(run_output_dir, "task_1", ["value", "extra"], [[1, "noise"]])

    monkeypatch.setattr(cli_module, "ARTIFACT_RUNS_DIR", runs_root)
    monkeypatch.setattr(cli_module, "PUBLIC_GOLD_DIR", gold_root)

    result = runner.invoke(
        cli_module.app,
        ["score-run", "sample-run", "--lambda", "0.25", "--lambda", "0.5"],
    )

    assert result.exit_code == 0, result.output
    assert "full_cover_rate" in result.output
    assert "mean_recall" in result.output
    assert "score_report" in result.output
    assert "0.25" in result.output
    assert (run_output_dir / "score_report.md").exists()

    score_payload = json.loads((run_output_dir / "score.json").read_text(encoding="utf-8"))
    assert score_payload["metadata"]["lambda_grid"] == [0.25, 0.5]
