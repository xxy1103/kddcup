from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.config import DataInspectorSampleBudget, ToolConfig
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    normalize_context_relative_path,
    read_doc_preview,
    resolve_context_path,
    search_doc_text,
)
from data_agent_baseline.tools.langgraph_tools import (
    AnswerArgs,
    ExecuteContextSqlArgs,
    ExecuteProbeQueryArgs,
    ExecutePythonArgs,
    GetColumnDistinctValuesArgs,
    ListContextArgs,
    LookupDocOutlineArgs,
    ReadDocArgs,
    SearchDocArgs,
    create_structured_tool,
)
from data_agent_baseline.tools.probe_engine import (
    execute_probe_query,
    get_column_distinct_values,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace, execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql
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
    budget: DataInspectorSampleBudget = field(default_factory=DataInspectorSampleBudget)
    _catalog_cache: dict[str, Any] | None = field(default=None, repr=False)

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


def _lookup_doc_outline(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    doc_path = str(action_input["path"])

    if runtime_context._catalog_cache is None:
        runtime_context._catalog_cache = build_semantic_catalog(
            runtime_context.task,
            budget=runtime_context.budget,
            max_depth=20,
            include_relationships=True,
        )

    catalog = runtime_context._catalog_cache
    normalized_path = normalize_context_relative_path(doc_path)

    for schema in catalog["schemas"]:
        if schema.get("kind") != "document":
            continue
        if schema.get("asset_path") == normalized_path:
            headings = schema.get("headings", [])
            return ToolExecutionResult(
                ok=True,
                content={
                    "path": normalized_path,
                    "head_count": len(headings),
                    "headings": headings,
                },
            )

    for asset in catalog.get("assets", []):
        if asset.get("asset_path") == normalized_path:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": f"'{normalized_path}' is not a text document (kind={asset.get('kind')}). lookup_doc_outline only supports .md, .txt, .rst files.",
                },
            )

    return ToolExecutionResult(
        ok=False,
        content={
            "error": f"No document found matching path '{doc_path}'.",
            "hint": "Use list_context or check the catalog for available documents.",
        },
    )


def _search_doc(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    query = str(action_input["query"])
    context_lines = int(action_input.get("context_lines", 3))
    path = action_input.get("path")
    return ToolExecutionResult(
        ok=True,
        content=search_doc_text(
            runtime_context.task,
            query,
            context_lines=context_lines,
            path=path,
        ),
    )


def _read_doc(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    heading = action_input.get("heading")
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(runtime_context.task, path, heading=heading),
    )


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


def _execute_probe_query(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    if runtime_context._catalog_cache is None:
        runtime_context._catalog_cache = build_semantic_catalog(
            runtime_context.task,
            budget=runtime_context.budget,
            max_depth=20,
            include_relationships=True,
        )
    catalog = runtime_context._catalog_cache
    queries = [str(q) for q in action_input["queries"]]
    limit = min(int(action_input.get("limit", 5)), 200)
    try:
        result = execute_probe_query(
            context_dir=runtime_context.task.context_dir,
            catalog=catalog,
            queries=queries,
            limit=limit,
        )
    except ValueError as exc:
        return ToolExecutionResult(ok=False, content={"error": str(exc)})
    return ToolExecutionResult(
        ok=bool(result.get("ok")),
        content=result,
    )


def _get_column_distinct_values(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    if runtime_context._catalog_cache is None:
        runtime_context._catalog_cache = build_semantic_catalog(
            runtime_context.task,
            budget=runtime_context.budget,
            max_depth=20,
            include_relationships=True,
        )
    catalog = runtime_context._catalog_cache
    table = str(action_input["table"])
    column = str(action_input["column"])
    top_n = min(int(action_input.get("top_n", 20)), 200)
    result = get_column_distinct_values(
        context_dir=runtime_context.task.context_dir,
        catalog=catalog,
        table=table,
        column=column,
        top_n=top_n,
    )
    return ToolExecutionResult(
        ok=bool(result.get("ok")),
        content=result,
    )


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
            return self.format_result(action, result)

        return invoke

    def format_result(self, action: str, result: ToolExecutionResult) -> dict[str, Any]:
        payload = {
            "ok": result.ok,
            "content": result.content,
        }
        if result.answer is not None:
            payload["answer"] = result.answer.to_dict()
        if action != "answer":
            payload["content"] = truncate_content(
                payload["content"],
                max_str_tokens=self.tool_config.max_output_tokens,
                max_list_items=self.tool_config.max_list_items,
            )
        return payload

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
        "execute_probe_query": ToolSpec(
            name="execute_probe_query",
            description=(
                "Execute read-only SQL queries against task data files (CSV, JSON, SQLite) "
                "using DuckDB. MANDATORY: pack multiple independent queries into ONE call "
                "whenever possible — combine COUNTs, DISTINCT scans, sample rows, and "
                "parallel aggregations instead of sending them one by one. "
                "Each query must be SELECT or WITH. "
                "CSV/JSON files are accessed by their file-name stem (e.g., 'member') or "
                "by asset path (e.g., 'csv/member.csv'). "
                "SQLite tables by their table name. "
                "Do NOT wrap table references in single quotes in SQL — "
                "use FROM qualifying, not FROM 'qualifying'. "
                "Returns a results list with up to <limit> rows per query."
            ),
            args_schema=ExecuteProbeQueryArgs,
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute Python code inside a per-task temporary copy of the context directory. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            args_schema=ExecutePythonArgs,
        ),
        "get_column_distinct_values": ToolSpec(
            name="get_column_distinct_values",
            description=(
                "Get the most frequent distinct values for a specific column/field, "
                "ranked by frequency. Supports CSV, JSON, and SQLite. "
                "Use this to quickly understand what values a field contains, verify "
                "candidate field mapping, or identify filter values. "
                "For CSV/JSON, 'table' is the file-name stem (e.g., 'member' for "
                "'csv/member.csv'). For SQLite, 'table' is the table name."
            ),
            args_schema=GetColumnDistinctValuesArgs,
        ),
        "lookup_doc_outline": ToolSpec(
            name="lookup_doc_outline",
            description=(
                "Look up the table of contents / heading structure of a markdown "
                "or text document from the catalog. "
                "Returns headings with level (1 = '#', 2 = '##', etc.). "
                "Use this before read_doc to discover section names."
            ),
            args_schema=LookupDocOutlineArgs,
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            args_schema=ListContextArgs,
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            args_schema=ReadDocArgs,
        ),
        "search_doc": ToolSpec(
            name="search_doc",
            description=(
                "Search text documents (.md, .txt, .rst) inside context for a regex "
                "pattern or plain keyword. Returns matching lines with surrounding "
                "context lines and their locations. "
                "Use this before read_doc to locate relevant sections when you "
                "don't know where the information lives, instead of writing Python "
                "code to grep through documents."
            ),
            args_schema=SearchDocArgs,
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_probe_query": _execute_probe_query,
        "execute_python": _execute_python,
        "get_column_distinct_values": _get_column_distinct_values,
        "lookup_doc_outline": _lookup_doc_outline,
        "list_context": _list_context,
        "read_doc": _read_doc,
        "search_doc": _search_doc,
    }
    return ToolRegistry(
        specs=specs,
        handlers=handlers,
        tool_config=tool_config if tool_config is not None else ToolConfig(),
    )
