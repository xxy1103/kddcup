"""Tests for submit_tool_result tool."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data_agent_baseline.benchmark.schema import PublicTask, TaskRecord, TaskAssets, ContextView
from data_agent_baseline.tools.registry import (
    ToolRuntimeContext,
    ToolRegistry,
    create_default_tool_registry,
    _extract_answer_from_probe_query,
    _extract_answer_from_python,
    _extract_answer_from_context_sql,
    _submit_tool_result,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace


# ---------------------------------------------------------------------------
# Extractor unit tests
# ---------------------------------------------------------------------------


def test_extract_answer_from_probe_query_last_successful():
    content = {
        "ok": True,
        "results": [
            {"ok": True, "columns": ["a"], "rows": [[1]], "row_count": 1},
            {"ok": True, "columns": ["x", "y"], "rows": [[10, 20], [30, 40]], "row_count": 2},
        ],
        "query_count": 2,
    }
    columns, rows = _extract_answer_from_probe_query(content)
    assert columns == ["x", "y"]
    assert rows == [[10, 20], [30, 40]]


def test_extract_answer_from_probe_query_skips_failed():
    content = {
        "ok": False,
        "results": [
            {"ok": False, "error": "table not found", "sql": "SELECT * FROM bad"},
            {"ok": True, "columns": ["name"], "rows": [["Alice"]], "row_count": 1},
        ],
        "query_count": 2,
    }
    columns, rows = _extract_answer_from_probe_query(content)
    assert columns == ["name"]
    assert rows == [["Alice"]]


def test_extract_answer_from_probe_query_all_failed():
    content = {
        "ok": False,
        "results": [{"ok": False, "error": "bad"}],
        "query_count": 1,
    }
    with pytest.raises(ValueError, match="No successful query result"):
        _extract_answer_from_probe_query(content)


def test_extract_answer_from_python_valid_json():
    output = json.dumps({"columns": ["id", "name"], "rows": [[1, "Alice"], [2, "Bob"]]})
    content = {"success": True, "output": output, "stderr": ""}
    columns, rows = _extract_answer_from_python(content)
    assert columns == ["id", "name"]
    assert rows == [[1, "Alice"], [2, "Bob"]]


def test_extract_answer_from_python_with_extra_output():
    content = {
        "success": True,
        "output": "Processing...\nDone.\n" + json.dumps({"columns": ["v"], "rows": [[42]]}),
        "stderr": "",
    }
    columns, rows = _extract_answer_from_python(content)
    assert columns == ["v"]
    assert rows == [[42]]


def test_extract_answer_from_python_no_json():
    content = {"success": True, "output": "Just some text without JSON", "stderr": ""}
    with pytest.raises(ValueError, match="does not contain a valid JSON"):
        _extract_answer_from_python(content)


def test_extract_answer_from_python_invalid_json():
    content = {"success": True, "output": "{bad json}", "stderr": ""}
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        _extract_answer_from_python(content)


def test_extract_answer_from_python_missing_keys():
    content = {"success": True, "output": json.dumps({"data": []}), "stderr": ""}
    with pytest.raises(ValueError, match="must contain 'columns'"):
        _extract_answer_from_python(content)


def test_extract_answer_from_context_sql():
    content = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
    columns, rows = _extract_answer_from_context_sql(content)
    assert columns == ["a", "b"]
    assert rows == [[1, 2], [3, 4]]


def test_extract_answer_from_context_sql_missing():
    with pytest.raises(ValueError, match="did not return columns/rows"):
        _extract_answer_from_context_sql({})


# ---------------------------------------------------------------------------
# Handler integration test (using a mock registry)
# ---------------------------------------------------------------------------


def _create_task(tmp_path: Path) -> PublicTask:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    return PublicTask(
        record=TaskRecord(task_id="test_task", difficulty="easy", question="test?"),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )


def test_submit_tool_result_unsupported_tool(tmp_path: Path):
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(task=task, python_workspace=workspace, registry=registry)

    result = _submit_tool_result(
        runtime_context,
        {"tool_name": "nonexistent_tool", "tool_args": {}},
    )
    assert result.ok is False
    assert "Unsupported source tool" in result.content["error"]


def test_submit_tool_result_no_registry(tmp_path: Path):
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)
    runtime_context = ToolRuntimeContext(task=task, python_workspace=workspace, registry=None)

    result = _submit_tool_result(
        runtime_context,
        {"tool_name": "execute_probe_query", "tool_args": {"queries": ["SELECT 1"]}},
    )
    assert result.ok is False
    assert "not available" in result.content["error"]


def test_submit_tool_result_column_count_mismatch(tmp_path: Path):
    """When the user specifies columns with wrong count, it should error."""
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)

    # Create a mock registry that returns a valid result
    mock_registry = MagicMock(spec=ToolRegistry)
    from data_agent_baseline.tools.registry import ToolExecutionResult
    mock_registry.execute.return_value = ToolExecutionResult(
        ok=True,
        content={
            "ok": True,
            "results": [
                {"ok": True, "columns": ["a", "b"], "rows": [[1, 2]], "row_count": 1}
            ],
            "query_count": 1,
        },
    )

    runtime_context = ToolRuntimeContext(
        task=task, python_workspace=workspace, registry=mock_registry
    )

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_probe_query",
            "tool_args": {"queries": ["SELECT a, b FROM t"]},
            "columns": ["only_one"],  # Wrong count
        },
    )
    assert result.ok is False
    assert "Column count mismatch" in result.content["error"]


def test_submit_tool_result_success(tmp_path: Path):
    """End-to-end test with mock registry returning a valid answer."""
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)

    mock_registry = MagicMock(spec=ToolRegistry)
    from data_agent_baseline.tools.registry import ToolExecutionResult
    mock_registry.execute.return_value = ToolExecutionResult(
        ok=True,
        content={
            "ok": True,
            "results": [
                {"ok": True, "columns": ["name", "score"], "rows": [["Alice", 95], ["Bob", 87]], "row_count": 2}
            ],
            "query_count": 1,
        },
    )

    runtime_context = ToolRuntimeContext(
        task=task, python_workspace=workspace, registry=mock_registry
    )

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_probe_query",
            "tool_args": {"queries": ["SELECT name, score FROM students ORDER BY score DESC"], "limit": 200},
        },
    )

    assert result.ok is True
    assert result.is_terminal is True
    assert result.answer is not None
    assert result.answer.columns == ["name", "score"]
    assert result.answer.rows == [["Alice", 95], ["Bob", 87]]
    assert result.content["status"] == "submitted"
    assert result.content["source_tool"] == "execute_probe_query"
    assert result.content["column_count"] == 2
    assert result.content["row_count"] == 2


def test_submit_tool_result_with_column_override(tmp_path: Path):
    """Test that columns can be renamed via the columns parameter."""
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)

    mock_registry = MagicMock(spec=ToolRegistry)
    from data_agent_baseline.tools.registry import ToolExecutionResult
    mock_registry.execute.return_value = ToolExecutionResult(
        ok=True,
        content={
            "ok": True,
            "results": [
                {"ok": True, "columns": ["first_name", "last_name"], "rows": [["John", "Doe"]], "row_count": 1}
            ],
            "query_count": 1,
        },
    )

    runtime_context = ToolRuntimeContext(
        task=task, python_workspace=workspace, registry=mock_registry
    )

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_probe_query",
            "tool_args": {"queries": ["SELECT first_name, last_name FROM people"]},
            "columns": ["given_name", "family_name"],
        },
    )

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["given_name", "family_name"]
    assert result.answer.rows == [["John", "Doe"]]
