from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import ToolConfig
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntimeContext,
    create_default_tool_registry,
)


def _create_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_demo"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "users.csv").write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    (context_dir / "events.json").write_text(
        json.dumps({"records": [{"Id": 10, "UserId": 1}, {"Id": 11, "UserId": 2}]}),
        encoding="utf-8",
    )
    (context_dir / "notes.md").write_text("# Notes\nhello\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id="task_demo", difficulty="easy", question="Inspect schema."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_format_result_truncates_string_content() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_chars=8, max_list_items=200),
    )

    payload = registry.format_result(
        "execute_python",
        ToolExecutionResult(ok=True, content={"output": "x" * 20}),
    )

    assert payload["ok"] is True
    assert str(payload["content"]["output"]).startswith("x" * 8)
    assert "内容已被截断" in str(payload["content"]["output"])


def test_format_result_truncates_list_content() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_chars=8000, max_list_items=2),
    )

    payload = registry.format_result(
        "read_csv",
        ToolExecutionResult(ok=True, content={"rows": [[1], [2], [3]]}),
    )

    assert payload["content"]["rows"][:2] == [[1], [2]]
    assert "内容已被截断" in str(payload["content"]["rows"][2])


def test_format_result_does_not_truncate_answer_content_and_keeps_answer() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_chars=4, max_list_items=1),
    )

    payload = registry.format_result(
        "answer",
        ToolExecutionResult(
            ok=True,
            content={"status": "submitted", "detail": "x" * 20},
            answer=AnswerTable(columns=["value"], rows=[["x" * 20]]),
        ),
    )

    assert payload["content"]["detail"] == "x" * 20
    assert payload["answer"] == {"columns": ["value"], "rows": [["x" * 20]]}


def test_default_registry_exposes_lookup_schema_and_hides_legacy_tools() -> None:
    registry = create_default_tool_registry()

    assert "lookup_schema" in registry.specs
    assert "inspect_all_schema" not in registry.specs
    assert "read_doc" in registry.specs
    assert "read_csv" not in registry.specs
    assert "read_json" not in registry.specs
    assert "inspect_sqlite_schema" not in registry.specs
    assert "lookup_schema" in registry.handlers
    assert "inspect_sqlite_schema" not in registry.handlers
    assert "read_csv" in registry.handlers
    assert "read_json" in registry.handlers


def test_lookup_schema_returns_field_details_for_csv(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(runtime_context, "lookup_schema", {"field_ref": "users.csv.id"})

    assert result.ok is True
    content = result.content
    assert content["field"] == "users.csv.id"
    assert content["resolved_to"] == "users.csv.id"
    assert content["field_details"]["name"] == "id"
    assert content["field_details"]["type"] == "integer"
    assert "related_fields" in content
    assert isinstance(content["related_fields"], list)
    assert "join_hints" in content
    assert isinstance(content["join_hints"], list)


def test_lookup_schema_caches_full_catalog(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    assert runtime_context._catalog_cache is None
    result1 = registry.execute(runtime_context, "lookup_schema", {"field_ref": "users.csv.id"})
    assert result1.ok is True
    assert runtime_context._catalog_cache is not None

    cache_after_first = runtime_context._catalog_cache
    result2 = registry.execute(runtime_context, "lookup_schema", {"field_ref": "users.csv.name"})
    assert result2.ok is True
    assert runtime_context._catalog_cache is cache_after_first


def test_lookup_schema_bad_ref_returns_error(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(runtime_context, "lookup_schema", {"field_ref": "nonexistent.column"})

    assert result.ok is False
    assert "No field found matching" in result.content["error"]


def test_lookup_schema_join_hints(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_join"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "users.csv").write_text("id,name\n1,Alice\n2,Bob\n3,Chen\n", encoding="utf-8")
    (context_dir / "orders.csv").write_text(
        "order_id,user_id,total\n10,1,25\n11,2,40\n12,3,50\n",
        encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_join", difficulty="easy", question="Find orders."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(runtime_context, "lookup_schema", {"field_ref": "orders.csv.user_id"})

    assert result.ok is True
    assert result.content["field_details"]["name"] == "user_id"
    # Join hints should connect orders.csv.user_id to users.csv.id
    join_hints = result.content["join_hints"]
    assert len(join_hints) >= 1
    hint_text = join_hints[0]
    assert "↔" in hint_text
    assert "users.csv" in hint_text
    assert "orders.csv" in hint_text


def test_lookup_schema_resolves_field_by_basename(tmp_path: Path) -> None:
    """Model may use only the filename (e.g. member.csv) even when the
    asset_path includes a directory prefix (e.g. csv/member.csv)."""
    task_dir = tmp_path / "task_subdir"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "csv"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "member.csv").write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_subdir", difficulty="easy", question="List names."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(runtime_context, "lookup_schema", {"field_ref": "member.csv.name"})

    assert result.ok is True
    assert result.content["resolved_to"] == "csv/member.csv.name"
