from __future__ import annotations

import csv
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Any
from uuid import uuid4

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.agents.model import create_chat_model
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


# 每个任务运行结束后需要落盘保存的产物元信息。
@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


# 使用 UTC 时间戳作为 run_id，便于排序且不受本地时区影响。
def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# 接受显式传入的 run_id；如果没有提供则自动生成，同时禁止路径形式的输入。
def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


# 为一次 benchmark 运行创建独立的输出目录，用来存放所有产物。
def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


# 根据配置构造聊天模型实例。
def build_chat_model(config: AppConfig):
    return create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=config.agent.temperature,
        enable_thinking=config.agent.enable_thinking,
    )


# 供任务产物落盘复用的简单文件写入辅助函数。
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        temp_path.replace(path)
    except BaseException:  # noqa: BLE001
        temp_path.unlink(missing_ok=True)
        raise


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
        temp_path.replace(path)
    except BaseException:  # noqa: BLE001
        temp_path.unlink(missing_ok=True)
        raise


# 统一失败结果的结构，便于后续按相同流程写出任务产物。
def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


# 单任务执行的核心路径：加载任务、构造 agent，并返回完整运行结果。
def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    agent = LangGraphAgent(
        model=model or build_chat_model(config),
        tools=tools or create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=config.agent.max_steps,
            enable_data_inspector=config.agent.enable_data_inspector,
            data_inspector=config.data_inspector,
        ),
    )
    run_result = agent.run(task)
    return run_result.to_dict()


# 在子进程中执行单任务核心逻辑，便于父进程施加硬超时控制。
def _run_single_task_in_subprocess(task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


# 为单任务执行增加进程级超时控制和异常退出处理。
def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    # 子进程隔离了模型和工具执行，任务卡住时父进程可以直接终止它。
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()
    try:
        # 先等待子进程把结果放进队列，再回收子进程；否则在某些平台上会因为
        # 大对象仍滞留在 Queue 管道中，导致 join() 误判为超时。
        result = queue.get(timeout=timeout_seconds)
    except Empty:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

        process.join(timeout=1.0)
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")
    finally:
        # 父进程负责关闭自身持有的队列句柄，避免后台 feeder 线程悬挂。
        queue.close()
        queue.join_thread()

    process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _failure_run_result_payload(task_id, "Task returned a result but did not exit cleanly.")

    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


# 为每个任务写出结构化 trace；只有产生有效答案时才写 prediction.csv。
def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    inspector = run_result.get("inspector")
    if isinstance(inspector, dict):
        inspector_outputs = {
            "perception.json": inspector.get("perception"),
            "semantic_catalog.json": inspector.get("semantic_catalog"),
            "semantic_index.json": inspector.get("semantic_index"),
            "data_understanding_handoff.json": inspector.get("data_understanding_handoff"),
        }
        for filename, payload in inspector_outputs.items():
            if isinstance(payload, dict):
                _write_json(task_output_dir / filename, payload)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


# 公开的任务执行入口：负责运行单个任务并补齐端到端耗时，但不直接写盘。
def execute_task(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return run_result


# 公开的单任务运行入口，负责端到端执行并记录总耗时。
def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    run_result = execute_task(task_id=task_id, config=config, model=model, tools=tools)
    return _write_task_outputs(task_id, run_output_dir, run_result)


# 运行选中的任务集合，可按需并行执行，并在最后写出本次运行的汇总信息。
def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks(task_ids=task_ids)
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    # 如果外部直接传入了 model 或 tools，会复用这些实例，因此强制退回单 worker 执行。
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        # 顺序执行时复用共享实例，避免每个任务重复构造模型和工具注册表。
        shared_model = model or build_chat_model(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        # 并行执行时，每个任务各自处理超时控制与结果落盘。
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "task_timeout_seconds": config.run.task_timeout_seconds,
            "soft_runtime_limit_seconds": config.run.soft_runtime_limit_seconds,
            "max_steps": config.agent.max_steps,
            "temperature": config.agent.temperature,
            "enable_data_inspector": config.agent.enable_data_inspector,
            "data_inspector": {
                "mode": config.data_inspector.mode,
                "inject_summary_to_agent": config.data_inspector.inject_summary_to_agent,
                "max_agent_steps": config.data_inspector.max_agent_steps,
                "enable_semantic_tools": config.data_inspector.enable_semantic_tools,
                "context_bundle_limit": config.data_inspector.context_bundle_limit,
                "max_join_hops": config.data_inspector.max_join_hops,
            },
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts


def run_selected_tasks_from_config(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    selected_task_ids = list(config.run.task_ids or ())
    if not selected_task_ids:
        raise ValueError("`run.task_ids` must contain at least one task id for selected-task runs.")

    return run_benchmark(
        config=config,
        model=model,
        tools=tools,
        task_ids=selected_task_ids,
        limit=limit,
        progress_callback=progress_callback,
    )
