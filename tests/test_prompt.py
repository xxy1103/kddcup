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

    assert "Every non-terminal turn must end with an executable tool call" in prompt
    assert "For structural data, first use `lookup_schema`" in prompt
    assert "lookup_schema" in prompt
    assert "even when a catalog is already present" not in prompt  # catalog is now lightweight
    assert "Cells must be JSON-compatible" in prompt
    assert "rows: []" in prompt
    assert "Do not drop zeros" in prompt
    assert "read the catalog" in prompt
    assert "JSON rule" in prompt
    assert "Ambiguity rule" in prompt
    assert "Probe each with `execute_python` or `execute_context_sql`" in prompt
    assert "If one candidate is non-empty" in prompt
    assert "For large outputs" in prompt
    assert "stable order" in prompt
    assert "verify full coverage with no gaps or duplicates" in prompt


def test_task_prompt_emphasizes_relative_paths_and_no_stop(tmp_path: Path) -> None:
    task = _create_task(tmp_path)

    prompt = build_task_prompt(task)

    assert task.question in prompt
    assert "All tool file paths are relative to the task context directory" in prompt
    assert "action-oriented working note" in prompt
    assert "Each turn must make progress through a tool call or the final answer call" in prompt
    assert "catalog or as returned" in prompt
    assert "probe every plausible interpretation against real data" in prompt
    assert "deterministic batch export with stable ordering" in prompt


def test_system_prompt_v2_handoff_preserved_for_comparison() -> None:
    """Old v1 prompt (Handoff-era) is now in prompt2 for comparison."""
    prompt = build_system_prompt_v2()

    assert "Data Understanding Handoff" in prompt
    assert "handoff JSON as trusted" in prompt
    assert "Do not drop numeric zero values" in prompt
    assert "answer_contract.answer_columns" in prompt


def test_prompts_do_not_reference_removed_tools() -> None:
    prompts = [build_system_prompt(), build_system_prompt_v2()]
    removed_tool_names = [
        "read_csv",
        "read_json",
        "inspect_all_schema",
        "inspect_sqlite_schema",
    ]

    for prompt in prompts:
        for tool_name in removed_tool_names:
            assert tool_name not in prompt
