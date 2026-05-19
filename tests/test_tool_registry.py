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
        tool_config=ToolConfig(max_output_tokens=2, max_list_items=200),
    )

    payload = registry.format_result(
        "execute_python",
        ToolExecutionResult(ok=True, content={"output": "x" * 20}),
    )

    assert payload["ok"] is True
    output_str = str(payload["content"]["output"])
    assert output_str.startswith("x")
    assert output_str != "x" * 20  # truncated, not the full 20
    assert "内容已被截断" in output_str


def test_format_result_truncates_list_content() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_tokens=2000, max_list_items=2),
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
        tool_config=ToolConfig(max_output_tokens=2, max_list_items=1),
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


def test_default_registry_exposes_probe_tools_and_hides_legacy_tools() -> None:
    registry = create_default_tool_registry()

    assert set(registry.handlers) == set(registry.specs)
    assert "execute_probe_query" in registry.specs
    assert "get_column_distinct_values" in registry.specs
    assert "inspect_all_schema" not in registry.specs
    assert "read_doc" in registry.specs
    assert "read_csv" not in registry.specs
    assert "read_json" not in registry.specs
    assert "inspect_sqlite_schema" not in registry.specs
    assert "lookup_schema" not in registry.specs
    assert "execute_probe_query" in registry.handlers
    assert "get_column_distinct_values" in registry.handlers
    assert "inspect_all_schema" not in registry.handlers
    assert "inspect_sqlite_schema" not in registry.handlers
    assert "read_csv" not in registry.handlers
    assert "read_json" not in registry.handlers
    assert "lookup_schema" not in registry.handlers


# ---------------------------------------------------------------------------
# execute_probe_query / get_column_distinct_values
# ---------------------------------------------------------------------------


def test_default_registry_exposes_probe_tools() -> None:
    registry = create_default_tool_registry()

    assert "execute_probe_query" in registry.specs
    assert "execute_probe_query" in registry.handlers
    assert "get_column_distinct_values" in registry.specs
    assert "get_column_distinct_values" in registry.handlers


def test_execute_probe_query_csv_select(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM users", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["id", "name"]
    assert first["rows"] == [[1, "Alice"], [2, "Bob"]]
    assert first["row_count"] == 2
    assert first["truncated"] is False


def test_execute_probe_query_csv_where_filter(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_filter"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "events.csv").write_text(
        "id,type,amount\n1,A,100\n2,B,200\n3,A,300\n", encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_filter", difficulty="easy", question="Filter."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM events WHERE type = 'A'", "limit": 10},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["row_count"] == 2
    assert [1, "A", 100] in first["rows"]
    assert [3, "A", 300] in first["rows"]


def test_execute_probe_query_json_records_select(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM events", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["Id", "UserId", "records"]
    assert [row[0] for row in first["rows"]] == [10, 11]
    assert [row[1] for row in first["rows"]] == [1, 2]
    assert first["row_count"] == 2


def test_execute_probe_query_json_records_using_asset_path(tmp_path: Path) -> None:
    """SQL can reference the asset path and it will be normalized."""
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT Id FROM events.json", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["Id"]
    assert [row[0] for row in first["rows"]] == [10, 11]


def test_execute_probe_query_with_group_by_and_aggregate(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_agg"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "orders.csv").write_text(
        "customer,amount\nAlice,100\nBob,200\nAlice,150\n", encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_agg", difficulty="easy", question="Aggregate."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["customer", "total"]
    rows = first["rows"]
    rows_as_tuples = sorted((row[0], float(row[1])) for row in rows)
    assert rows_as_tuples == [("Alice", 250.0), ("Bob", 200.0)]


def test_execute_probe_query_invalid_sql_rejected(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "INSERT INTO users VALUES (3, 'Eve')"},
    )

    assert result.ok is False
    assert "Only SELECT/WITH" in result.content["results"][0]["error"]


def test_execute_probe_query_nonexistent_table(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM ghost"},
    )

    assert result.ok is False
    assert "error" in result.content["results"][0]


def test_execute_probe_query_limit_truncation(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_trunc"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{i},val{i}" for i in range(20))
    (context_dir / "big.csv").write_text(f"id,val\n{rows}\n", encoding="utf-8")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_trunc", difficulty="easy", question="Trunc."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM big", "limit": 3},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["row_count"] == 3
    assert first["truncated"] is True


def test_get_column_distinct_values_csv(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "get_column_distinct_values",
        {"table": "users", "column": "name", "top_n": 10},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    assert result.content["table"] == "users"
    assert result.content["column"] == "name"
    values = result.content["values"]
    assert {"value": "Alice", "count": 1} in values
    assert {"value": "Bob", "count": 1} in values


def test_get_column_distinct_values_nonexistent_table(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "get_column_distinct_values",
        {"table": "ghost", "column": "x"},
    )

    assert result.ok is False
    assert "not found" in result.content.get("error", "")


def test_execute_probe_query_lazy_builds_catalog(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    assert runtime_context._catalog_cache is None
    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM users", "limit": 1},
    )
    assert result.ok is True
    assert runtime_context._catalog_cache is not None

    cache_after_first = runtime_context._catalog_cache
    result2 = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM events", "limit": 1},
    )
    assert result2.ok is True
    assert runtime_context._catalog_cache is cache_after_first


def test_execute_probe_query_sqlite(tmp_path: Path) -> None:
    import sqlite3
    task_dir = tmp_path / "task_probe_sqlite"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    db_path = context_dir / "data.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE races (raceId INTEGER, name TEXT)")
        conn.execute("INSERT INTO races VALUES (1, 'GP')")
        conn.execute("INSERT INTO races VALUES (2, 'WRC')")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_probe_sqlite", difficulty="easy", question="Probe."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "execute_probe_query",
        {"sql": "SELECT * FROM races", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["raceId", "name"]
    assert first["rows"] == [[1, "GP"], [2, "WRC"]]


def test_get_column_distinct_values_sqlite(tmp_path: Path) -> None:
    import sqlite3
    task_dir = tmp_path / "task_dist_sqlite"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    db_path = context_dir / "data.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE items (id INTEGER, category TEXT)")
        conn.execute("INSERT INTO items VALUES (1, 'A')")
        conn.execute("INSERT INTO items VALUES (2, 'A')")
        conn.execute("INSERT INTO items VALUES (3, 'B')")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_dist_sqlite", difficulty="easy", question="Dist."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "get_column_distinct_values",
        {"table": "items", "column": "category", "top_n": 10},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    values = result.content["values"]
    assert len(values) == 2
    assert {"value": "A", "count": 2} in values
    assert {"value": "B", "count": 1} in values


# ---------------------------------------------------------------------------
# search_doc pagination
# ---------------------------------------------------------------------------


def _create_doc_task(tmp_path: Path, doc_name: str, lines: list[str]) -> PublicTask:
    task_dir = tmp_path / f"task_{doc_name}"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / f"{doc_name}.md").write_text("\n".join(lines), encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id=f"task_{doc_name}", difficulty="easy", question="Search."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_search_doc_default_pagination(tmp_path: Path) -> None:
    lines = ["line alpha"] * 5 + ["line beta"] * 3
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "search_doc",
        {"query": "alpha", "context_lines": 0},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 5
    assert content["page"] == 1
    assert content["page_size"] == 20
    assert content["total_pages"] == 1
    assert len(content["results"]) == 1
    assert len(content["results"][0]["matches"]) == 5


def test_search_doc_page2(tmp_path: Path) -> None:
    lines = []
    for i in range(25):
        lines.append(f"item_{i} alpha")
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "search_doc",
        {"query": "alpha", "context_lines": 0, "page": 2, "page_size": 10},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 25
    assert content["page"] == 2
    assert content["page_size"] == 10
    assert content["total_pages"] == 3
    assert len(content["results"]) == 1
    page_matches = content["results"][0]["matches"]
    assert len(page_matches) == 10
    assert page_matches[0]["line_number"] == 11
    assert page_matches[-1]["line_number"] == 20


def test_search_doc_page_out_of_range(tmp_path: Path) -> None:
    lines = ["alpha one", "beta two", "alpha three"]
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "search_doc",
        {"query": "alpha", "context_lines": 0, "page": 99, "page_size": 10},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 2
    assert content["page"] == 99
    assert content["page_size"] == 10
    assert content["total_pages"] == 1
    assert content["results"] == []


def test_search_doc_page_size_zero(tmp_path: Path) -> None:
    lines = ["alpha one", "alpha two", "alpha three", "delta four", "alpha five"]
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context, "search_doc",
        {"query": "alpha", "context_lines": 0, "page_size": 0},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 4
    assert content["page"] == 1
    assert content["total_pages"] == 1
    assert len(content["results"][0]["matches"]) == 4
