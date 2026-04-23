from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import (
    load_app_config,
    load_submission_config_from_env,
    resolve_submission_log_dir_from_env,
)
from data_agent_baseline.run.submission import SubmissionLogger, run_submission
from data_agent_baseline.run.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_selected_tasks_from_config,
    run_single_task,
)
from data_agent_baseline.scoring import resolve_score_run_dir, score_run_outputs
from data_agent_baseline.tools.filesystem import list_context_tree

# 约定好的项目目录入口，CLI 会基于这些路径展示状态和写出产物。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"
PUBLIC_GOLD_DIR = DATA_DIR / "public" / "output"

# Typer 应用入口和统一的 rich 控制台输出对象。
app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


# 将路径存在性压缩成简短的状态文案。
def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


# 以 task/min 的格式展示当前吞吐速率，便于进度条紧凑展示。
def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


# 在进度条右侧显示最近完成任务的编号和状态。
def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


# 为 rich Progress 统一构造紧凑字段，避免在多处重复拼接展示逻辑。
def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


# CLI 根命令，仅用于挂载子命令。
@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


# 查看本地项目路径、配置路径和公开数据集是否就绪。
@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(app_config.dataset.root_path), _status_value(app_config.dataset.root_path))
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


# 查看单个任务的元信息以及 context/ 下可访问的文件。
@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


# 运行单个任务，并打印输出目录、预测文件和失败原因。
@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(app_config.run.output_dir, run_id=app_config.run.run_id)
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


# 批量运行 benchmark，并用 rich 进度条显示完成情况和吞吐。
@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_total = len(dataset.iter_tasks())
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    # 进度条字段刻意做得紧凑，方便在终端中同时展示成功/失败、速度和最近任务。
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        # 每完成一个任务就刷新一次统计信息和进度条展示。
        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


# 从配置文件的 run.task_ids 执行指定任务集合。
@app.command("run-selected-tasks")
def run_selected_tasks_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of selected tasks to run."),
) -> None:
    """Run only tasks listed in run.task_ids from the config file."""
    app_config = load_app_config(config)
    selected_task_ids = list(app_config.run.task_ids or ())
    if not selected_task_ids:
        raise typer.BadParameter(
            "`run.task_ids` is empty. Please set a non-empty task id list in the config file.",
            param_hint="run.task_ids",
        )

    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    selected_tasks = dataset.iter_tasks(task_ids=selected_task_ids)
    if not selected_tasks:
        raise typer.BadParameter(
            "No tasks matched `run.task_ids` in the dataset root.",
            param_hint="run.task_ids",
        )

    task_total = len(selected_tasks)
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Selected Tasks",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Selected Tasks",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_selected_tasks_from_config(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc

        progress.update(
            progress_task_id,
            completed=task_total,
            description="Selected Tasks",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Selected tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


# 提交模式入口：只依赖环境变量，面向未来 Docker ENTRYPOINT 使用。
@app.command("submit")
def submit_command() -> None:
    """Run the benchmark in submission mode with env-only configuration."""
    logger = SubmissionLogger(resolve_submission_log_dir_from_env())
    try:
        submission_config = load_submission_config_from_env()
    except (FileNotFoundError, ValueError) as exc:
        logger.log("ERROR", f"Submission configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        artifacts = run_submission(config=submission_config, logger=logger)
    except Exception as exc:
        logger.log("ERROR", f"Submission run failed: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Submission tasks attempted: {artifacts.task_count}")
    console.print(f"Submission tasks succeeded: {artifacts.succeeded_task_count}")
    console.print(f"Submission tasks failed: {artifacts.failed_task_count}")
    console.print(f"Runtime log: {artifacts.runtime_log_path}")


# 对某次 run 的 prediction.csv 按官方列匹配规则打分。
@app.command("score-run")
def score_run_command(
    run_id: str | None = typer.Argument(
        None,
        help=(
            "Optional run directory name under artifacts/runs. Latest run is used when omitted. "
            "The run must include summary.json because score-run only evaluates task ids recorded there."
        ),
    ),
    lambda_values: list[float] | None = typer.Option(
        None,
        "--lambda",
        help="Optional proxy lambda values. Repeat the flag to evaluate multiple lambda settings.",
    ),
) -> None:
    """Score one run directory using only the task ids recorded in summary.json."""
    try:
        effective_run_id, run_output_dir = resolve_score_run_dir(ARTIFACT_RUNS_DIR, run_id=run_id)
        summary = score_run_outputs(
            run_output_dir=run_output_dir,
            gold_root=PUBLIC_GOLD_DIR,
            lambda_values=lambda_values,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run_id") from exc

    summary_table = Table(title=f"Score Summary for {effective_run_id}")
    summary_table.add_column("Item")
    summary_table.add_column("Value")
    summary_table.add_row("run_id", effective_run_id)
    summary_table.add_row("run_output", str(run_output_dir))
    summary_table.add_row("public_gold_dir", str(PUBLIC_GOLD_DIR))
    summary_table.add_row("task_source", summary.task_source)
    summary_table.add_row("task_count", str(summary.task_count))
    summary_table.add_row("prediction_task_count", str(summary.prediction_task_count))
    summary_table.add_row("full_cover_count", str(summary.full_cover_count))
    summary_table.add_row("full_cover_rate", f"{summary.full_cover_rate:.4f}")
    summary_table.add_row("mean_recall", f"{summary.mean_recall:.4f}")
    summary_table.add_row("mean_redundancy_rate", f"{summary.mean_redundancy_rate:.4f}")
    summary_table.add_row("compat_total_score", str(summary.total_score))
    summary_table.add_row("compat_accuracy", f"{summary.accuracy:.4f}")
    summary_table.add_row("score_json", str(summary.score_path))
    summary_table.add_row("score_report", str(summary.score_report_path))
    console.print(summary_table)

    proxy_table = Table(title="Proxy Scores (Recall - λ * Redundancy)")
    proxy_table.add_column("λ")
    proxy_table.add_column("score")
    for label, score in summary.proxy_scores.items():
        proxy_table.add_row(label, f"{score:.4f}")
    console.print(proxy_table)

    if summary.failure_breakdown:
        failure_table = Table(title="Failure Breakdown")
        failure_table.add_column("Failure")
        failure_table.add_column("Count")
        for reason, count in summary.failure_breakdown.items():
            failure_table.add_row(reason, str(count))
        console.print(failure_table)


# 供 pyproject 或脚本入口直接调用的主函数。
def main() -> None:
    app()
