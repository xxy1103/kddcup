from __future__ import annotations

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.config import ToolConfig
from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry


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
