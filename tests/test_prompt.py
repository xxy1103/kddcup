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


def test_system_prompt_emphasizes_non_stop_tool_turns_and_answer_schema() -> None:
    prompt = build_system_prompt()

    assert "may either call a tool immediately" in prompt
    assert "After a working-note turn, continue the task on the next turn" in prompt
    assert "Never end a turn with empty content and no tool call." in prompt
    assert "`answer.rows` must be a list of rows, and every row must itself be a list." in prompt
    assert "If the correct result is empty, call `answer`" in prompt
    assert "Do not drop numeric zero values" in prompt
    assert "missing_count" in prompt
    assert "included in computations by default" in prompt


def test_task_prompt_emphasizes_relative_paths_and_no_stop(tmp_path: Path) -> None:
    task = _create_task(tmp_path)

    prompt = build_task_prompt(task)

    assert task.question in prompt
    assert "All tool file paths are relative to the task context directory" in prompt
    assert "you may briefly state what you learned and what you will inspect next" in prompt
    assert "Do not stop without either continuing the task or calling `answer`." in prompt


def test_system_prompt_v2_preserves_observed_values_by_default() -> None:
    prompt = build_system_prompt_v2()

    assert "Do not drop numeric zero values" in prompt
    assert "missing_count" in prompt
    assert "included in computations by default" in prompt
