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
        "execute_python",
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

    assert set(registry.handlers) == set(registry.specs)
    assert "lookup_schema" in registry.specs
    assert "inspect_all_schema" not in registry.specs
    assert "read_doc" in registry.specs
    assert "read_csv" not in registry.specs
    assert "read_json" not in registry.specs
    assert "inspect_sqlite_schema" not in registry.specs
    assert "lookup_schema" in registry.handlers
    assert "inspect_all_schema" not in registry.handlers
    assert "inspect_sqlite_schema" not in registry.handlers
    assert "read_csv" not in registry.handlers
    assert "read_json" not in registry.handlers


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


def test_lookup_schema_resolves_slash_replaced_path(tmp_path: Path) -> None:
    """Model may replace / with . in the asset_path, e.g. csv.trans.csv.type."""
    task_dir = tmp_path / "task_slash_dot"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "csv"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "trans.csv").write_text("type,amount\nA,100\nB,200\n", encoding="utf-8")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_slash_dot", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(runtime_context, "lookup_schema", {"field_ref": "csv.trans.csv.type"})

    assert result.ok is True
    assert result.content["resolved_to"] == "csv/trans.csv.type"
    assert result.content["field_details"]["name"] == "type"


def test_lookup_schema_resolves_stripped_extension(tmp_path: Path) -> None:
    """Model may strip .csv from basename, e.g. trans.type instead of trans.csv.type."""
    task_dir = tmp_path / "task_noext"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "csv"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "trans.csv").write_text("type,amount\nA,100\nB,200\n", encoding="utf-8")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_noext", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(runtime_context, "lookup_schema", {"field_ref": "trans.type"})

    assert result.ok is True
    assert result.content["resolved_to"] == "csv/trans.csv.type"
    assert result.content["field_details"]["name"] == "type"


def test_lookup_schema_slash_replaced_sqlite(tmp_path: Path) -> None:
    """Model may replace / with . in SQLite asset_path, e.g. data.events.db.races.raceId."""
    import sqlite3
    task_dir = tmp_path / "task_sqlite_dot"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "data"
    sub_dir.mkdir(parents=True, exist_ok=True)
    db_path = sub_dir / "events.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE races (raceId INTEGER, name TEXT)")
        conn.execute("INSERT INTO races VALUES (1, 'GP')")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_sqlite_dot", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "lookup_schema", {"field_ref": "data.events.db.races.raceId"}
    )

    assert result.ok is True
    assert result.content["resolved_to"] == "data/events.db.races.raceId"
    assert result.content["field_details"]["name"] == "raceId"


def test_lookup_schema_json_with_bare_short_name(tmp_path: Path) -> None:
    """Catalog field names for JSON with records wrapper are short (no prefix),
    so a bare field name like 'Thrombosis' should resolve."""
    task_dir = tmp_path / "task_json_short"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "json"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "exam.json").write_text(
        json.dumps({"records": [{"Thrombosis": 2, "ID": 1}]}),
        encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_json_short", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(runtime_context, "lookup_schema", {"field_ref": "Thrombosis"})

    assert result.ok is True
    assert result.content["field"] == "Thrombosis"
    assert result.content["resolved_to"] == "json/exam.json.Thrombosis"
    assert result.content["field_details"]["name"] == "Thrombosis"


def test_lookup_schema_json_with_full_asset_path(tmp_path: Path) -> None:
    """Full path like 'json/exam.json.Thrombosis' should resolve with short name."""
    task_dir = tmp_path / "task_json_full"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "json"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "exam.json").write_text(
        json.dumps({"records": [{"Thrombosis": 2, "ID": 1}]}),
        encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_json_full", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "lookup_schema", {"field_ref": "json/exam.json.Thrombosis"}
    )

    assert result.ok is True
    assert result.content["field"] == "json/exam.json.Thrombosis"
    assert result.content["resolved_to"] == "json/exam.json.Thrombosis"
    assert result.content["field_details"]["name"] == "Thrombosis"


def test_lookup_schema_json_records_prefix_rejected(tmp_path: Path) -> None:
    """'records.Thrombosis' should NOT match — the wrapper key is not a field name."""
    task_dir = tmp_path / "task_json_reject"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "json"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "exam.json").write_text(
        json.dumps({"records": [{"Thrombosis": 2, "ID": 1}]}),
        encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_json_reject", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "lookup_schema", {"field_ref": "records.Thrombosis"}
    )

    assert result.ok is False
    assert "No field found matching" in result.content["error"]


def test_lookup_schema_json_slash_replaced_path(tmp_path: Path) -> None:
    """Model may replace / with . in JSON path, e.g. json.exam.json.Thrombosis."""
    task_dir = tmp_path / "task_json_slash"
    context_dir = task_dir / "context"
    sub_dir = context_dir / "json"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "exam.json").write_text(
        json.dumps({"records": [{"Thrombosis": 2, "ID": 1}]}),
        encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_json_slash", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "lookup_schema", {"field_ref": "json.exam.json.Thrombosis"}
    )

    assert result.ok is True
    assert result.content["resolved_to"] == "json/exam.json.Thrombosis"
    assert result.content["field_details"]["name"] == "Thrombosis"
