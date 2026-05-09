from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.langgraph_tools import (
    AnswerArgs,
    ExecuteContextSqlArgs,
    ExecutePythonArgs,
    InspectSqliteSchemaArgs,
    ListContextArgs,
    ReadCsvArgs,
    ReadDocArgs,
    ReadJsonArgs,
    create_structured_tool,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace, execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema
from data_agent_baseline.config import ToolConfig
from data_agent_baseline.tools.truncation import truncate_content

# Python 执行工具的固定超时时间，避免模型生成的脚本长时间卡住。
EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


@dataclass(slots=True)
class ToolRuntimeContext:
    task: PublicTask
    python_workspace: TaskContextWorkspace

    @property
    def temp_workspace(self) -> str | None:
        return None if self.python_workspace.path is None else str(self.python_workspace.path)


ToolHandler = Callable[[ToolRuntimeContext, dict[str, Any]], ToolExecutionResult]


def _list_context(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(
        ok=True,
        content=list_context_tree(runtime_context.task, max_depth=max_depth),
    )


def _read_csv(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(
        ok=True,
        content=read_csv_preview(runtime_context.task, path, max_rows=max_rows),
    )


def _read_json(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(
        ok=True,
        content=read_json_preview(runtime_context.task, path, max_chars=max_chars),
    )


def _read_doc(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(runtime_context.task, path, max_chars=max_chars),
    )


def _inspect_sqlite_schema(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(runtime_context.task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(runtime_context.task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    workspace_root = runtime_context.python_workspace.materialize()
    content = execute_python_code(
        context_root=workspace_root,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class BoundToolRegistry:
    registry: ToolRegistry
    runtime_context: ToolRuntimeContext
    tools: dict[str, BaseTool]

    def langchain_tools(self) -> list[BaseTool]:
        return [self.tools[name] for name in sorted(self.tools)]

    def execute(self, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        return self.registry.execute(self.runtime_context, action, action_input)


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
    tool_config: ToolConfig = field(default_factory=ToolConfig)

    def bind(self, runtime_context: ToolRuntimeContext) -> BoundToolRegistry:
        tools: dict[str, BaseTool] = {}
        for name, spec in self.specs.items():
            tools[name] = create_structured_tool(
                name=spec.name,
                description=spec.description,
                args_schema=spec.args_schema,
                invoke=self._build_tool_wrapper(runtime_context, name),
            )
        return BoundToolRegistry(registry=self, runtime_context=runtime_context, tools=tools)

    def _build_tool_wrapper(
        self,
        runtime_context: ToolRuntimeContext,
        action: str,
    ) -> Callable[..., dict[str, Any]]:
        def invoke(**kwargs: Any) -> dict[str, Any]:
            result = self.execute(runtime_context, action, kwargs)
            payload = {
                "ok": result.ok,
                "content": result.content,
            }
            if result.answer is not None:
                payload["answer"] = result.answer.to_dict()
            if action != "answer":
                payload["content"] = truncate_content(
                    payload["content"],
                    max_str_chars=self.tool_config.max_output_chars,
                    max_list_items=self.tool_config.max_list_items,
                )
            return payload

        return invoke

    def execute(
        self,
        runtime_context: ToolRuntimeContext,
        action: str,
        action_input: dict[str, Any],
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](runtime_context, action_input)


def create_default_tool_registry(tool_config: ToolConfig | None = None) -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            args_schema=AnswerArgs,
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            args_schema=ExecuteContextSqlArgs,
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute Python code inside a per-task temporary copy of the context directory. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            args_schema=ExecutePythonArgs,
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            args_schema=InspectSqliteSchemaArgs,
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            args_schema=ListContextArgs,
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            args_schema=ReadCsvArgs,
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            args_schema=ReadDocArgs,
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            args_schema=ReadJsonArgs,
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(
        specs=specs,
        handlers=handlers,
        tool_config=tool_config if tool_config is not None else ToolConfig(),
    )
