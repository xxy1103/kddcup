from __future__ import annotations

import csv
import json
import signal
import threading
from collections import deque
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import SubmissionConfig
from data_agent_baseline.run.runner import build_chat_model, execute_task
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SubmissionLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_log_path = self.log_dir / "runtime.log"
        self.task_status_path = self.log_dir / "task_status.jsonl"
        self.run_summary_path = self.log_dir / "run_summary.json"
        self._lock = Lock()

    def log(self, level: str, message: str) -> None:
        line = f"{_utc_timestamp()} [{level}] {message}"
        with self._lock:
            with self.runtime_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        print(line, flush=True)

    def record_task_status(self, payload: dict[str, Any]) -> None:
        with self._lock:
            with self.task_status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_run_summary(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.run_summary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


@dataclass(frozen=True, slots=True)
class SubmissionTaskArtifacts:
    task_id: str
    prediction_csv_path: Path | None
    succeeded: bool
    failure_reason: str | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class SubmissionRunArtifacts:
    task_count: int
    succeeded_task_count: int
    failed_task_count: int
    runtime_log_path: Path
    task_status_path: Path
    run_summary_path: Path
    tasks: list[SubmissionTaskArtifacts]


@dataclass(slots=True)
class SubmissionStopController:
    soft_runtime_limit_seconds: int
    _drain_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    drain_reason: str | None = None

    def request_drain(self, reason: str) -> None:
        if self.drain_reason is None:
            self.drain_reason = reason
        self._drain_event.set()

    @property
    def drain_requested(self) -> bool:
        return self._drain_event.is_set()

    def should_stop_new_tasks(self, started_at_monotonic: float) -> bool:
        if self.drain_requested:
            return True
        if self.soft_runtime_limit_seconds > 0:
            elapsed_seconds = perf_counter() - started_at_monotonic
            if elapsed_seconds >= self.soft_runtime_limit_seconds:
                self.request_drain("soft_runtime_limit")
                return True
        return False


@contextmanager
def _install_sigterm_drain(stop_controller: SubmissionStopController) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame) -> None:  # type: ignore[no-untyped-def]
        del signum, frame
        stop_controller.request_drain("sigterm")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _write_prediction_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_submission_prediction(task_id: str, output_root: Path, run_result: dict[str, Any]) -> Path | None:
    answer = run_result.get("answer")
    if not isinstance(answer, dict):
        return None

    prediction_path = output_root / task_id / "prediction.csv"
    _write_prediction_csv(
        prediction_path,
        list(answer.get("columns", [])),
        [list(row) for row in answer.get("rows", [])],
    )
    return prediction_path


def _run_submission_task(
    *,
    task_id: str,
    config: SubmissionConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> SubmissionTaskArtifacts:
    try:
        run_result = execute_task(
            task_id=task_id,
            config=config.app_config,
            model=model,
            tools=tools,
        )
        prediction_csv_path = _write_submission_prediction(task_id, config.output_dir, run_result)
        return SubmissionTaskArtifacts(
            task_id=task_id,
            prediction_csv_path=prediction_csv_path,
            succeeded=bool(run_result.get("succeeded")),
            failure_reason=run_result.get("failure_reason"),
            elapsed_seconds=float(run_result.get("e2e_elapsed_seconds", 0.0)),
        )
    except BaseException as exc:  # noqa: BLE001
        return SubmissionTaskArtifacts(
            task_id=task_id,
            prediction_csv_path=None,
            succeeded=False,
            failure_reason=f"Submission task handling failed: {exc}",
            elapsed_seconds=0.0,
        )


def _record_completed_task(
    *,
    artifact: SubmissionTaskArtifacts,
    active_logger: SubmissionLogger,
) -> None:
    active_logger.record_task_status(artifact.to_dict())
    status = "ok" if artifact.succeeded else "fail"
    active_logger.log(
        "INFO",
        f"Completed {artifact.task_id}: status={status} elapsed={artifact.elapsed_seconds:.3f}s",
    )


def run_submission(
    *,
    config: SubmissionConfig,
    model=None,
    tools: ToolRegistry | None = None,
    logger: SubmissionLogger | None = None,
) -> SubmissionRunArtifacts:
    dataset = DABenchPublicDataset(config.input_root)
    if not dataset.exists:
        raise FileNotFoundError(f"Submission input root does not exist: {config.input_root}")

    effective_workers = config.app_config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    config.output_dir.mkdir(parents=True, exist_ok=True)
    active_logger = logger or SubmissionLogger(config.log_dir)
    task_ids = [task.task_id for task in dataset.iter_tasks()]
    started_at = _utc_timestamp()
    started_at_monotonic = perf_counter()
    stop_controller = SubmissionStopController(
        soft_runtime_limit_seconds=config.app_config.run.soft_runtime_limit_seconds
    )
    active_logger.log(
        "INFO",
        (
            f"Starting submission run: tasks={len(task_ids)} "
            f"input_root={config.input_root} output_root={config.output_dir} "
            f"max_workers={effective_workers} "
            f"parameter_config={config.parameter_config_path or 'none'}"
        ),
    )

    task_artifacts: list[SubmissionTaskArtifacts] = []
    pending_task_ids = deque((index, task_id) for index, task_id in enumerate(task_ids))

    with _install_sigterm_drain(stop_controller):
        if effective_workers == 1:
            shared_model = model or build_chat_model(config.app_config)
            shared_tools = tools or create_default_tool_registry()
            while pending_task_ids:
                if stop_controller.should_stop_new_tasks(started_at_monotonic):
                    break

                _, task_id = pending_task_ids.popleft()
                artifact = _run_submission_task(
                    task_id=task_id,
                    config=config,
                    model=shared_model,
                    tools=shared_tools,
                )
                task_artifacts.append(artifact)
                _record_completed_task(artifact=artifact, active_logger=active_logger)
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_index: dict[Future[SubmissionTaskArtifacts], int] = {}
                indexed_artifacts: list[SubmissionTaskArtifacts | None] = [None] * len(task_ids)

                def submit_available_tasks() -> None:
                    while pending_task_ids and len(future_to_index) < effective_workers:
                        if stop_controller.should_stop_new_tasks(started_at_monotonic):
                            return
                        index, task_id = pending_task_ids.popleft()
                        future = executor.submit(_run_submission_task, task_id=task_id, config=config)
                        future_to_index[future] = index

                submit_available_tasks()
                while future_to_index:
                    done_futures, _ = wait(
                        tuple(future_to_index),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done_futures:
                        artifact = future.result()
                        index = future_to_index.pop(future)
                        indexed_artifacts[index] = artifact
                        _record_completed_task(artifact=artifact, active_logger=active_logger)
                    submit_available_tasks()

                task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    if stop_controller.drain_reason is not None:
        active_logger.log("WARNING", f"Stopped scheduling new tasks: reason={stop_controller.drain_reason}")

    succeeded_task_count = sum(1 for artifact in task_artifacts if artifact.succeeded)
    failed_task_count = len(task_artifacts) - succeeded_task_count
    skipped_task_count = len(task_ids) - len(task_artifacts)
    summary_payload = {
        "started_at": started_at,
        "completed_at": _utc_timestamp(),
        "input_root": str(config.input_root),
        "output_root": str(config.output_dir),
        "log_root": str(config.log_dir),
        "parameter_config_path": str(config.parameter_config_path) if config.parameter_config_path else None,
        "task_count": len(task_artifacts),
        "succeeded_task_count": succeeded_task_count,
        "failed_task_count": failed_task_count,
        "skipped_task_count": skipped_task_count,
        "drain_reason": stop_controller.drain_reason,
        "max_workers": effective_workers,
        "task_timeout_seconds": config.app_config.run.task_timeout_seconds,
        "soft_runtime_limit_seconds": config.app_config.run.soft_runtime_limit_seconds,
        "max_steps": config.app_config.agent.max_steps,
        "temperature": config.app_config.agent.temperature,
        "tasks": [artifact.to_dict() for artifact in task_artifacts],
    }
    active_logger.write_run_summary(summary_payload)
    active_logger.log(
        "INFO",
        (
            f"Finished submission run: succeeded={succeeded_task_count} "
            f"failed={failed_task_count} skipped={skipped_task_count}"
        ),
    )
    return SubmissionRunArtifacts(
        task_count=len(task_artifacts),
        succeeded_task_count=succeeded_task_count,
        failed_task_count=failed_task_count,
        runtime_log_path=active_logger.runtime_log_path,
        task_status_path=active_logger.task_status_path,
        run_summary_path=active_logger.run_summary_path,
        tasks=task_artifacts,
    )
