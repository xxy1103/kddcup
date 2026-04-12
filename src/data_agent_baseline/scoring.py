from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_LAMBDA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
RULES_URL = "https://dataagent.top/rules"
NULL_SYNONYMS = frozenset({"", "null", "none", "nan", "nat", "<na>"})
NUMERIC_QUANTIZER = Decimal("0.01")
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
DATETIME_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[T ].+$")


@dataclass(frozen=True, slots=True)
class ColumnVector:
    index: int
    values: tuple[str, ...]
    signature: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    gold_mask: int
    prediction_mask: int
    covered_gold_columns: int
    matched_prediction_columns: int
    mode: str


@dataclass(frozen=True, slots=True)
class TaskDiagnostics:
    difficulty: str | None
    succeeded: bool | None
    failure_reason: str | None
    e2e_elapsed_seconds: float | None
    step_count: int | None


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    difficulty: str | None
    gold_csv_path: Path
    prediction_csv_path: Path | None
    score: int
    full_cover: bool
    covered_gold_columns: int
    matched_prediction_columns: int
    gold_column_count: int
    prediction_column_count: int
    extra_prediction_columns: int
    recall: float
    redundancy_rate: float
    proxy_scores: dict[str, float]
    reason: str | None
    succeeded: bool | None
    failure_reason: str | None
    e2e_elapsed_seconds: float | None
    step_count: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "gold_csv_path": str(self.gold_csv_path),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "score": self.score,
            "full_cover": self.full_cover,
            "covered_gold_columns": self.covered_gold_columns,
            "matched_prediction_columns": self.matched_prediction_columns,
            "gold_column_count": self.gold_column_count,
            "prediction_column_count": self.prediction_column_count,
            "extra_prediction_columns": self.extra_prediction_columns,
            "recall": _round_metric(self.recall),
            "redundancy_rate": _round_metric(self.redundancy_rate),
            "proxy_scores": {key: _round_metric(value) for key, value in self.proxy_scores.items()},
            "reason": self.reason,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
            "e2e_elapsed_seconds": _round_metric(self.e2e_elapsed_seconds),
            "step_count": self.step_count,
        }


@dataclass(frozen=True, slots=True)
class RunScoreSummary:
    run_id: str
    run_output_dir: Path
    score_path: Path
    score_report_path: Path
    rules_url: str
    lambda_grid: list[float]
    task_count: int
    prediction_task_count: int
    full_cover_count: int
    full_cover_rate: float
    total_score: int
    accuracy: float
    mean_recall: float
    mean_redundancy_rate: float
    proxy_scores: dict[str, float]
    difficulty_breakdown: dict[str, dict[str, object]]
    failure_breakdown: dict[str, int]
    runtime_summary: dict[str, object]
    tasks: list[TaskScore]

    def to_dict(self) -> dict[str, object]:
        metadata = {
            "run_id": self.run_id,
            "run_output_dir": str(self.run_output_dir),
            "score_path": str(self.score_path),
            "score_report_path": str(self.score_report_path),
            "rules_url": self.rules_url,
            "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lambda_grid": list(self.lambda_grid),
        }
        overview = {
            "task_count": self.task_count,
            "prediction_task_count": self.prediction_task_count,
            "full_cover_count": self.full_cover_count,
            "full_cover_rate": _round_metric(self.full_cover_rate),
            "mean_recall": _round_metric(self.mean_recall),
            "mean_redundancy_rate": _round_metric(self.mean_redundancy_rate),
            "proxy_scores": {key: _round_metric(value) for key, value in self.proxy_scores.items()},
            "total_score": self.total_score,
            "accuracy": _round_metric(self.accuracy),
        }
        return {
            "metadata": metadata,
            "overview": overview,
            "difficulty_breakdown": self.difficulty_breakdown,
            "failure_breakdown": self.failure_breakdown,
            "runtime_summary": self.runtime_summary,
            "tasks": [task.to_dict() for task in self.tasks],
            "run_id": self.run_id,
            "run_output_dir": str(self.run_output_dir),
            "score_path": str(self.score_path),
            "score_report_path": str(self.score_report_path),
            "scored_at_utc": metadata["scored_at_utc"],
            "rules_url": self.rules_url,
            "lambda_grid": list(self.lambda_grid),
            "task_count": self.task_count,
            "prediction_task_count": self.prediction_task_count,
            "full_cover_count": self.full_cover_count,
            "full_cover_rate": _round_metric(self.full_cover_rate),
            "mean_recall": _round_metric(self.mean_recall),
            "mean_redundancy_rate": _round_metric(self.mean_redundancy_rate),
            "proxy_scores": {key: _round_metric(value) for key, value in self.proxy_scores.items()},
            "total_score": self.total_score,
            "accuracy": _round_metric(self.accuracy),
        }


def normalize_lambda_grid(lambda_values: list[float] | tuple[float, ...] | None = None) -> list[float]:
    raw_values = list(DEFAULT_LAMBDA_GRID if not lambda_values else lambda_values)
    normalized: list[float] = []
    seen: set[float] = set()
    for value in raw_values:
        candidate = float(value)
        if not math.isfinite(candidate):
            raise ValueError(f"lambda must be finite, got {value!r}.")
        if candidate < 0:
            raise ValueError(f"lambda must be non-negative, got {value!r}.")
        if candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    if not normalized:
        raise ValueError("lambda grid must not be empty.")
    return normalized


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _lambda_label(value: float) -> str:
    text = format(value, ".6f").rstrip("0").rstrip(".")
    return text or "0"


def _proxy_scores(*, recall: float, redundancy_rate: float, lambda_grid: list[float]) -> dict[str, float]:
    return {
        _lambda_label(lambda_value): max(recall - (lambda_value * redundancy_rate), 0.0)
        for lambda_value in lambda_grid
    }


def normalize_cell(raw_value: str) -> str:
    value = raw_value.strip()
    if value.lower() in NULL_SYNONYMS:
        return ""

    normalized_datetime = _normalize_datetime(value)
    if normalized_datetime is not None:
        return normalized_datetime

    normalized_date = _normalize_date(value)
    if normalized_date is not None:
        return normalized_date

    normalized_numeric = _normalize_numeric(value)
    if normalized_numeric is not None:
        return normalized_numeric

    return value


def _normalize_numeric(value: str) -> str | None:
    if not NUMERIC_PATTERN.fullmatch(value):
        return None
    try:
        quantized = Decimal(value).quantize(NUMERIC_QUANTIZER, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    return format(quantized, "f")


def _normalize_date(value: str) -> str | None:
    if not DATE_PATTERN.fullmatch(value):
        return None
    try:
        year_str, month_str, day_str = value.split("-")
        normalized = date(int(year_str), int(month_str), int(day_str))
    except ValueError:
        return None
    return normalized.isoformat()


def _normalize_datetime(value: str) -> str | None:
    if not DATETIME_PATTERN.fullmatch(value):
        return None

    parse_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError:
        return None

    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        normalized = parsed.astimezone(timezone.utc).isoformat()
        return normalized.removesuffix("+00:00") + "Z"
    return parsed.isoformat()


def _column_signature(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(values))


def _combine_name_columns(left: ColumnVector, right: ColumnVector) -> tuple[str, ...]:
    return tuple(" ".join(part for part in (first, last) if part) for first, last in zip(left.values, right.values))


def _load_csv_columns(path: Path) -> list[ColumnVector]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    expected_width = len(rows[0])
    columns = [[] for _ in range(expected_width)]
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != expected_width:
            raise ValueError(
                f"CSV row width mismatch in {path} at line {row_index}: "
                f"expected {expected_width}, got {len(row)}."
            )
        for column_index, value in enumerate(row):
            columns[column_index].append(normalize_cell(value))

    return [
        ColumnVector(index=index, values=tuple(values), signature=_column_signature(tuple(values)))
        for index, values in enumerate(columns)
    ]


def _build_match_candidates(
    gold_columns: list[ColumnVector],
    prediction_columns: list[ColumnVector],
) -> list[MatchCandidate]:
    candidates: list[MatchCandidate] = []

    for gold_column in gold_columns:
        for prediction_column in prediction_columns:
            if gold_column.signature == prediction_column.signature:
                candidates.append(
                    MatchCandidate(
                        gold_mask=1 << gold_column.index,
                        prediction_mask=1 << prediction_column.index,
                        covered_gold_columns=1,
                        matched_prediction_columns=1,
                        mode="1to1",
                    )
                )

    for left_gold, right_gold in combinations(gold_columns, 2):
        combined_signature = _column_signature(_combine_name_columns(left_gold, right_gold))
        for prediction_column in prediction_columns:
            if combined_signature == prediction_column.signature:
                candidates.append(
                    MatchCandidate(
                        gold_mask=(1 << left_gold.index) | (1 << right_gold.index),
                        prediction_mask=1 << prediction_column.index,
                        covered_gold_columns=2,
                        matched_prediction_columns=1,
                        mode="2to1_name",
                    )
                )

    for gold_column in gold_columns:
        for left_prediction, right_prediction in combinations(prediction_columns, 2):
            combined_signature = _column_signature(_combine_name_columns(left_prediction, right_prediction))
            if gold_column.signature == combined_signature:
                candidates.append(
                    MatchCandidate(
                        gold_mask=1 << gold_column.index,
                        prediction_mask=(1 << left_prediction.index) | (1 << right_prediction.index),
                        covered_gold_columns=1,
                        matched_prediction_columns=2,
                        mode="1to2_name",
                    )
                )

    candidates.sort(
        key=lambda item: (
            -item.covered_gold_columns,
            -item.matched_prediction_columns,
            item.gold_mask,
            item.prediction_mask,
            item.mode,
        )
    )
    return candidates


def _select_best_matches(
    gold_columns: list[ColumnVector],
    prediction_columns: list[ColumnVector],
) -> tuple[int, int]:
    candidates = _build_match_candidates(gold_columns, prediction_columns)

    @lru_cache(maxsize=None)
    def search(used_gold_mask: int, used_prediction_mask: int) -> tuple[int, int, tuple[int, ...]]:
        best_covered = 0
        best_prediction_columns = 0
        best_path: tuple[int, ...] = ()

        for index, candidate in enumerate(candidates):
            if candidate.gold_mask & used_gold_mask:
                continue
            if candidate.prediction_mask & used_prediction_mask:
                continue

            covered, matched_prediction_columns, path = search(
                used_gold_mask | candidate.gold_mask,
                used_prediction_mask | candidate.prediction_mask,
            )
            covered += candidate.covered_gold_columns
            matched_prediction_columns += candidate.matched_prediction_columns
            candidate_path = (index,) + path

            if covered > best_covered:
                best_covered = covered
                best_prediction_columns = matched_prediction_columns
                best_path = candidate_path
                continue
            if covered == best_covered and matched_prediction_columns > best_prediction_columns:
                best_prediction_columns = matched_prediction_columns
                best_path = candidate_path
                continue
            if (
                covered == best_covered
                and matched_prediction_columns == best_prediction_columns
                and candidate_path < best_path
            ):
                best_path = candidate_path

        return best_covered, best_prediction_columns, best_path

    covered_gold_columns, matched_prediction_columns, _ = search(0, 0)
    return covered_gold_columns, matched_prediction_columns


def _load_task_difficulty(input_root: Path, task_id: str) -> str | None:
    task_json_path = input_root / task_id / "task.json"
    if not task_json_path.exists():
        return None
    try:
        payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    difficulty = payload.get("difficulty")
    return str(difficulty) if difficulty is not None else None


def _load_summary_task_map(run_output_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return {}
    task_map: dict[str, dict[str, Any]] = {}
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if isinstance(task_id, str):
            task_map[task_id] = item
    return task_map


def _load_trace_payload(task_output_dir: Path) -> dict[str, Any]:
    trace_path = task_output_dir / "trace.json"
    if not trace_path.exists():
        return {}
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_task_diagnostics(
    *,
    task_id: str,
    run_output_dir: Path,
    input_root: Path,
    summary_task_map: dict[str, dict[str, Any]],
) -> TaskDiagnostics:
    summary_item = summary_task_map.get(task_id, {})
    task_output_dir = run_output_dir / task_id
    trace_payload = _load_trace_payload(task_output_dir)
    step_count = trace_payload.get("steps")
    runtime = trace_payload.get("e2e_elapsed_seconds")

    succeeded = summary_item.get("succeeded")
    if not isinstance(succeeded, bool):
        trace_succeeded = trace_payload.get("succeeded")
        succeeded = trace_succeeded if isinstance(trace_succeeded, bool) else None

    failure_reason = summary_item.get("failure_reason")
    if failure_reason is None:
        trace_failure_reason = trace_payload.get("failure_reason")
        failure_reason = trace_failure_reason if isinstance(trace_failure_reason, str) else None
    elif not isinstance(failure_reason, str):
        failure_reason = None

    return TaskDiagnostics(
        difficulty=_load_task_difficulty(input_root, task_id),
        succeeded=succeeded,
        failure_reason=failure_reason,
        e2e_elapsed_seconds=float(runtime) if isinstance(runtime, (int, float)) else None,
        step_count=len(step_count) if isinstance(step_count, list) else None,
    )


def _build_task_reason(
    *,
    covered_gold_columns: int,
    gold_column_count: int,
    extra_prediction_columns: int,
    prediction_column_count: int,
) -> str | None:
    if gold_column_count == 0 and prediction_column_count == 0:
        return "Both gold and prediction contain no columns."
    if covered_gold_columns == gold_column_count and extra_prediction_columns == 0:
        return None
    if covered_gold_columns == gold_column_count:
        return f"All gold columns covered, with {extra_prediction_columns} extra prediction column(s)."
    if covered_gold_columns == 0:
        return (
            f"No gold columns matched; {prediction_column_count} prediction column(s) were produced and "
            f"{extra_prediction_columns} are redundant."
        )
    return (
        f"Covered {covered_gold_columns}/{gold_column_count} gold column(s), with "
        f"{extra_prediction_columns} extra prediction column(s)."
    )


def _score_task(
    *,
    task_id: str,
    gold_csv_path: Path,
    prediction_csv_path: Path | None,
    lambda_grid: list[float],
    diagnostics: TaskDiagnostics,
) -> TaskScore:
    gold_columns = _load_csv_columns(gold_csv_path)
    gold_column_count = len(gold_columns)

    if prediction_csv_path is None or not prediction_csv_path.exists():
        recall = 0.0
        redundancy_rate = 0.0
        return TaskScore(
            task_id=task_id,
            difficulty=diagnostics.difficulty,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=prediction_csv_path,
            score=0,
            full_cover=False,
            covered_gold_columns=0,
            matched_prediction_columns=0,
            gold_column_count=gold_column_count,
            prediction_column_count=0,
            extra_prediction_columns=0,
            recall=recall,
            redundancy_rate=redundancy_rate,
            proxy_scores=_proxy_scores(recall=recall, redundancy_rate=redundancy_rate, lambda_grid=lambda_grid),
            reason="prediction.csv is missing.",
            succeeded=diagnostics.succeeded,
            failure_reason=diagnostics.failure_reason,
            e2e_elapsed_seconds=diagnostics.e2e_elapsed_seconds,
            step_count=diagnostics.step_count,
        )

    try:
        prediction_columns = _load_csv_columns(prediction_csv_path)
    except ValueError as exc:
        recall = 0.0
        redundancy_rate = 0.0
        return TaskScore(
            task_id=task_id,
            difficulty=diagnostics.difficulty,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=prediction_csv_path,
            score=0,
            full_cover=False,
            covered_gold_columns=0,
            matched_prediction_columns=0,
            gold_column_count=gold_column_count,
            prediction_column_count=0,
            extra_prediction_columns=0,
            recall=recall,
            redundancy_rate=redundancy_rate,
            proxy_scores=_proxy_scores(recall=recall, redundancy_rate=redundancy_rate, lambda_grid=lambda_grid),
            reason=str(exc),
            succeeded=diagnostics.succeeded,
            failure_reason=diagnostics.failure_reason,
            e2e_elapsed_seconds=diagnostics.e2e_elapsed_seconds,
            step_count=diagnostics.step_count,
        )

    prediction_column_count = len(prediction_columns)
    covered_gold_columns, matched_prediction_columns = _select_best_matches(gold_columns, prediction_columns)
    extra_prediction_columns = max(prediction_column_count - matched_prediction_columns, 0)
    recall = covered_gold_columns / gold_column_count if gold_column_count > 0 else 0.0
    redundancy_rate = (
        extra_prediction_columns / prediction_column_count if prediction_column_count > 0 else 0.0
    )
    full_cover = gold_column_count > 0 and covered_gold_columns == gold_column_count

    return TaskScore(
        task_id=task_id,
        difficulty=diagnostics.difficulty,
        gold_csv_path=gold_csv_path,
        prediction_csv_path=prediction_csv_path,
        score=int(full_cover),
        full_cover=full_cover,
        covered_gold_columns=covered_gold_columns,
        matched_prediction_columns=matched_prediction_columns,
        gold_column_count=gold_column_count,
        prediction_column_count=prediction_column_count,
        extra_prediction_columns=extra_prediction_columns,
        recall=recall,
        redundancy_rate=redundancy_rate,
        proxy_scores=_proxy_scores(recall=recall, redundancy_rate=redundancy_rate, lambda_grid=lambda_grid),
        reason=_build_task_reason(
            covered_gold_columns=covered_gold_columns,
            gold_column_count=gold_column_count,
            extra_prediction_columns=extra_prediction_columns,
            prediction_column_count=prediction_column_count,
        ),
        succeeded=diagnostics.succeeded,
        failure_reason=diagnostics.failure_reason,
        e2e_elapsed_seconds=diagnostics.e2e_elapsed_seconds,
        step_count=diagnostics.step_count,
    )


def resolve_score_run_dir(runs_root: Path, run_id: str | None = None) -> tuple[str, Path]:
    if run_id is not None:
        normalized = run_id.strip()
        if not normalized:
            raise ValueError("run_id must not be empty.")
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("run_id must be a single directory name, not a path.")
        run_output_dir = runs_root / normalized
        if not run_output_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_output_dir}")
        return normalized, run_output_dir

    candidate_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")

    latest_run_dir = max(candidate_dirs, key=lambda path: path.name)
    return latest_run_dir.name, latest_run_dir


def _aggregate_proxy_scores(tasks: list[TaskScore], lambda_grid: list[float]) -> dict[str, float]:
    if not tasks:
        return {_lambda_label(lambda_value): 0.0 for lambda_value in lambda_grid}
    return {
        _lambda_label(lambda_value): mean(
            max(task.recall - (lambda_value * task.redundancy_rate), 0.0) for task in tasks
        )
        for lambda_value in lambda_grid
    }


def _build_difficulty_breakdown(tasks: list[TaskScore], lambda_grid: list[float]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[TaskScore]] = {}
    for task in tasks:
        difficulty = task.difficulty or "unknown"
        grouped.setdefault(difficulty, []).append(task)

    breakdown: dict[str, dict[str, object]] = {}
    for difficulty in sorted(grouped):
        group_tasks = grouped[difficulty]
        task_count = len(group_tasks)
        prediction_task_count = sum(1 for task in group_tasks if task.prediction_csv_path is not None)
        full_cover_count = sum(1 for task in group_tasks if task.full_cover)
        breakdown[difficulty] = {
            "task_count": task_count,
            "prediction_task_count": prediction_task_count,
            "full_cover_count": full_cover_count,
            "full_cover_rate": _round_metric(full_cover_count / task_count if task_count > 0 else 0.0),
            "mean_recall": _round_metric(mean(task.recall for task in group_tasks) if group_tasks else 0.0),
            "mean_redundancy_rate": _round_metric(
                mean(task.redundancy_rate for task in group_tasks) if group_tasks else 0.0
            ),
            "proxy_scores": {
                key: _round_metric(value)
                for key, value in _aggregate_proxy_scores(group_tasks, lambda_grid).items()
            },
        }
    return breakdown


def _build_failure_breakdown(tasks: list[TaskScore]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for task in tasks:
        if task.failure_reason is None:
            continue
        breakdown[task.failure_reason] = breakdown.get(task.failure_reason, 0) + 1
    return dict(sorted(breakdown.items(), key=lambda item: (-item[1], item[0])))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _build_runtime_summary(tasks: list[TaskScore]) -> dict[str, object]:
    runtimes = [task.e2e_elapsed_seconds for task in tasks if task.e2e_elapsed_seconds is not None]
    step_counts = [float(task.step_count) for task in tasks if task.step_count is not None]
    return {
        "available_runtime_count": len(runtimes),
        "mean_e2e_elapsed_seconds": _round_metric(mean(runtimes) if runtimes else 0.0),
        "median_e2e_elapsed_seconds": _round_metric(median(runtimes) if runtimes else 0.0),
        "p95_e2e_elapsed_seconds": _round_metric(_percentile(runtimes, 0.95) if runtimes else 0.0),
        "max_e2e_elapsed_seconds": _round_metric(max(runtimes) if runtimes else 0.0),
        "available_step_count": len(step_counts),
        "mean_step_count": _round_metric(mean(step_counts) if step_counts else 0.0),
        "max_step_count": int(max(step_counts)) if step_counts else 0,
    }


def _review_task_priority(task: TaskScore) -> tuple[object, ...]:
    return (
        0 if task.failure_reason is not None else 1,
        0 if not task.full_cover else 1,
        task.recall,
        -task.redundancy_rate,
        -(task.e2e_elapsed_seconds or 0.0),
        task.task_id,
    )


def _select_review_tasks(tasks: list[TaskScore], limit: int = 8) -> list[TaskScore]:
    candidates = [task for task in tasks if task.failure_reason is not None or not task.full_cover]
    return sorted(candidates, key=_review_task_priority)[:limit]


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _build_score_report(summary: RunScoreSummary) -> str:
    proxy_rows = [[label, f"{score:.4f}"] for label, score in summary.proxy_scores.items()]
    overview_rows = [
        ["任务总数", str(summary.task_count)],
        ["生成 prediction.csv 的任务数", str(summary.prediction_task_count)],
        ["Full Cover 任务数", str(summary.full_cover_count)],
        ["Full Cover Rate", f"{summary.full_cover_rate:.4f}"],
        ["Mean Recall", f"{summary.mean_recall:.4f}"],
        ["Mean Redundancy Rate", f"{summary.mean_redundancy_rate:.4f}"],
        ["兼容 total_score", str(summary.total_score)],
        ["兼容 accuracy", f"{summary.accuracy:.4f}"],
    ]

    difficulty_rows = []
    for difficulty, payload in summary.difficulty_breakdown.items():
        difficulty_rows.append(
            [
                difficulty,
                str(payload["task_count"]),
                str(payload["prediction_task_count"]),
                str(payload["full_cover_count"]),
                f"{float(payload['full_cover_rate']):.4f}",
                f"{float(payload['mean_recall']):.4f}",
                f"{float(payload['mean_redundancy_rate']):.4f}",
            ]
        )

    failure_rows = [
        [reason, str(count)] for reason, count in summary.failure_breakdown.items()
    ] or [["无", "0"]]

    runtime = summary.runtime_summary
    runtime_rows = [
        ["可用耗时任务数", str(runtime["available_runtime_count"])],
        ["平均耗时（秒）", f"{float(runtime['mean_e2e_elapsed_seconds']):.3f}"],
        ["中位耗时（秒）", f"{float(runtime['median_e2e_elapsed_seconds']):.3f}"],
        ["P95 耗时（秒）", f"{float(runtime['p95_e2e_elapsed_seconds']):.3f}"],
        ["最长耗时（秒）", f"{float(runtime['max_e2e_elapsed_seconds']):.3f}"],
        ["可用 step_count 任务数", str(runtime["available_step_count"])],
        ["平均 step_count", f"{float(runtime['mean_step_count']):.2f}"],
        ["最大 step_count", str(runtime["max_step_count"])],
    ]

    review_rows = []
    for task in _select_review_tasks(summary.tasks):
        review_rows.append(
            [
                task.task_id,
                task.difficulty or "unknown",
                f"{task.recall:.4f}",
                f"{task.redundancy_rate:.4f}",
                "yes" if task.full_cover else "no",
                task.failure_reason or (task.reason or "-"),
                "-" if task.e2e_elapsed_seconds is None else f"{task.e2e_elapsed_seconds:.3f}",
            ]
        )
    if not review_rows:
        review_rows = [["无", "-", "-", "-", "-", "-", "-"]]

    appendix_rows = [
        [
            task.task_id,
            task.difficulty or "unknown",
            str(task.gold_column_count),
            str(task.prediction_column_count),
            str(task.covered_gold_columns),
            str(task.extra_prediction_columns),
            f"{task.recall:.4f}",
            f"{task.redundancy_rate:.4f}",
            "yes" if task.full_cover else "no",
            "-" if task.e2e_elapsed_seconds is None else f"{task.e2e_elapsed_seconds:.3f}",
            task.failure_reason or (task.reason or "-"),
        ]
        for task in summary.tasks
    ]

    sections = [
        f"# {summary.run_id} 评测报告",
        "",
        "## 执行摘要",
        "",
        f"- 评分规则来源：[{summary.rules_url}]({summary.rules_url})",
        f"- 本地结果采用“双指标 + 多 λ 代理”体系，`λ` 网格为 `{', '.join(_lambda_label(item) for item in summary.lambda_grid)}`。",
        "- 本地分数用于全面评估 Recall 与冗余惩罚敏感度，不代表官方未公开 λ 下的唯一得分。",
        "",
        "## 双指标总览",
        "",
        _render_markdown_table(["指标", "值"], overview_rows),
        "",
        "## 多 λ 代理分数",
        "",
        _render_markdown_table(["λ", "代理分数"], proxy_rows),
        "",
        "## 按难度拆分表现",
        "",
        _render_markdown_table(
            ["难度", "任务数", "有预测", "Full Cover", "Full Cover Rate", "Mean Recall", "Mean Redundancy"],
            difficulty_rows or [["无", "0", "0", "0", "0.0000", "0.0000", "0.0000"]],
        ),
        "",
        "## 失败原因与耗时分析",
        "",
        "### 失败原因",
        "",
        _render_markdown_table(["失败原因", "任务数"], failure_rows),
        "",
        "### 运行时摘要",
        "",
        _render_markdown_table(["指标", "值"], runtime_rows),
        "",
        "## 最值得复盘的任务",
        "",
        _render_markdown_table(
            ["任务", "难度", "Recall", "Redundancy", "Full Cover", "失败/备注", "耗时(秒)"],
            review_rows,
        ),
        "",
        "## 全量任务附录",
        "",
        _render_markdown_table(
            [
                "任务",
                "难度",
                "Gold列数",
                "预测列数",
                "覆盖Gold列数",
                "冗余列数",
                "Recall",
                "Redundancy",
                "Full Cover",
                "耗时(秒)",
                "失败/备注",
            ],
            appendix_rows,
        ),
        "",
    ]
    return "\n".join(sections)


def score_run_outputs(
    *,
    run_output_dir: Path,
    gold_root: Path,
    lambda_values: list[float] | tuple[float, ...] | None = None,
    rules_url: str = RULES_URL,
) -> RunScoreSummary:
    if not gold_root.is_dir():
        raise FileNotFoundError(f"Gold output directory not found: {gold_root}")

    input_root = gold_root.parent / "input"
    lambda_grid = normalize_lambda_grid(lambda_values)
    summary_task_map = _load_summary_task_map(run_output_dir)

    tasks: list[TaskScore] = []
    for gold_task_dir in sorted(path for path in gold_root.iterdir() if path.is_dir()):
        task_id = gold_task_dir.name
        gold_csv_path = gold_task_dir / "gold.csv"
        if not gold_csv_path.exists():
            continue

        prediction_csv_path = run_output_dir / task_id / "prediction.csv"
        if not prediction_csv_path.exists():
            prediction_csv_path = None

        diagnostics = _build_task_diagnostics(
            task_id=task_id,
            run_output_dir=run_output_dir,
            input_root=input_root,
            summary_task_map=summary_task_map,
        )
        tasks.append(
            _score_task(
                task_id=task_id,
                gold_csv_path=gold_csv_path,
                prediction_csv_path=prediction_csv_path,
                lambda_grid=lambda_grid,
                diagnostics=diagnostics,
            )
        )

    task_count = len(tasks)
    prediction_task_count = sum(1 for task in tasks if task.prediction_csv_path is not None)
    full_cover_count = sum(1 for task in tasks if task.full_cover)
    full_cover_rate = full_cover_count / task_count if task_count > 0 else 0.0
    mean_recall = mean(task.recall for task in tasks) if tasks else 0.0
    mean_redundancy_rate = mean(task.redundancy_rate for task in tasks) if tasks else 0.0
    proxy_scores = _aggregate_proxy_scores(tasks, lambda_grid)
    score_path = run_output_dir / "score.json"
    score_report_path = run_output_dir / "score_report.md"

    summary = RunScoreSummary(
        run_id=run_output_dir.name,
        run_output_dir=run_output_dir,
        score_path=score_path,
        score_report_path=score_report_path,
        rules_url=rules_url,
        lambda_grid=lambda_grid,
        task_count=task_count,
        prediction_task_count=prediction_task_count,
        full_cover_count=full_cover_count,
        full_cover_rate=full_cover_rate,
        total_score=full_cover_count,
        accuracy=full_cover_rate,
        mean_recall=mean_recall,
        mean_redundancy_rate=mean_redundancy_rate,
        proxy_scores=proxy_scores,
        difficulty_breakdown=_build_difficulty_breakdown(tasks, lambda_grid),
        failure_breakdown=_build_failure_breakdown(tasks),
        runtime_summary=_build_runtime_summary(tasks),
        tasks=tasks,
    )

    score_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_report_path.write_text(_build_score_report(summary), encoding="utf-8")
    return summary
