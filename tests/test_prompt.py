from __future__ import annotations

from pathlib import Path

from data_agent_baseline.agents.prompt import build_system_prompt, build_task_prompt
from data_agent_baseline.agents.prompt2 import build_system_prompt_v2
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord


def _create_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_demo"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    return PublicTask(
        record=TaskRecord(task_id="task_demo", difficulty="easy", question="List all IDs."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_system_prompt_emphasizes_tool_turns_and_answer_schema() -> None:
    prompt = build_system_prompt()

    assert "EVERY non-terminal turn MUST conclude with an executable tool call" in prompt
    assert "FIRST tool call for any structural-data task SHOULD be" in prompt
    assert "lookup_schema" in prompt
    assert "even when a catalog is already present" not in prompt  # catalog is now lightweight
    assert "`answer.rows` must be a list of rows, and every row must itself be a list" in prompt
    assert "If the correct result is empty, call `answer`" in prompt
    assert "Do not drop numeric zeros" in prompt
    assert "lightweight index" in prompt
    assert "JSON field name convention" in prompt


def test_task_prompt_emphasizes_relative_paths_and_no_stop(tmp_path: Path) -> None:
    task = _create_task(tmp_path)

    prompt = build_task_prompt(task)

    assert task.question in prompt
    assert "All tool file paths are relative to the task context directory" in prompt
    assert "action-oriented working note" in prompt
    assert "Each turn must make progress through a tool call or the final answer call" in prompt
    assert "catalog or as returned" in prompt


def test_system_prompt_v2_handoff_preserved_for_comparison() -> None:
    """Old v1 prompt (Handoff-era) is now in prompt2 for comparison."""
    prompt = build_system_prompt_v2()

    assert "Data Understanding Handoff" in prompt
    assert "handoff JSON as trusted" in prompt
    assert "Do not drop numeric zero values" in prompt
    assert "answer_contract.answer_columns" in prompt
