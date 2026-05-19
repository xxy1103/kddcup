from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    MemAgentArgs,
    ReadDocArgs,
    SearchDocArgs,
    create_structured_tool,
)
from data_agent_baseline.tools.memagent import (
    MEMORY_MAX_TOKENS,
    RECURRENT_CHUNK_SIZE,
    RECURRENT_MAX_CONTEXT_LEN,
    _build_llm_fn,
    make_pattern_analyzer,
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
    model: object | None = field(default=None, repr=False)

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
    context_lines = int(action_input.get("context_lines", 5))
    path = action_input.get("path")
    page = int(action_input.get("page", 1))
    page_size = int(action_input.get("page_size", 20))
    return ToolExecutionResult(
        ok=True,
        content=search_doc_text(
            runtime_context.task,
            query,
            context_lines=context_lines,
            path=path,
            page=page,
            page_size=page_size,
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
    if "queries" in action_input:
        queries = [str(q) for q in action_input["queries"]]
    elif "sql" in action_input:
        # Backward compatibility for older traces/tests and for models that
        # still emit the pre-batching argument name.
        queries = [str(action_input["sql"])]
    else:
        raise ValueError("execute_probe_query requires `queries` (list[str]).")
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


def _lookup_distinct_values_from_catalog(
    catalog: dict[str, Any], table: str, column: str, top_n: int,
) -> dict[str, Any] | None:
    normalized_table = table.strip().strip('"').replace("\\", "/")
    for schema in catalog.get("schemas", []):
        asset_path = schema.get("asset_path", "")
        kind = schema.get("kind", "")
        stem = Path(asset_path).stem

        if kind == "sqlite":
            for t in schema.get("tables", []):
                if t.get("name") != normalized_table:
                    continue
                for field in t.get("fields", []):
                    if field.get("name") == column:
                        distinct = field.get("distinct_values")
                        if isinstance(distinct, list) and distinct:
                            values = distinct[:top_n]
                            return {
                                "ok": True,
                                "table": table,
                                "column": column,
                                "values": values,
                                "value_count": len(values),
                            }
                        return None
            continue

        if kind in ("csv", "json") and normalized_table in {stem, asset_path, f"{asset_path}.records"}:
            for field in schema.get("fields", []):
                if field.get("name") == column:
                    distinct = field.get("distinct_values")
                    if isinstance(distinct, list) and distinct:
                        values = distinct[:top_n]
                        return {
                            "ok": True,
                            "table": table,
                            "column": column,
                            "values": values,
                            "value_count": len(values),
                        }
                    return None
    return None


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

    # Attempt to reuse precomputed distinct_values from the catalog.
    catalog_result = _lookup_distinct_values_from_catalog(catalog, table, column, top_n)
    if catalog_result is not None:
        return ToolExecutionResult(ok=True, content=catalog_result)

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


def _memagent(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    question = str(action_input["question"])
    full_path = runtime_context.task.context_dir / path

    if runtime_context.model is None:
        return ToolExecutionResult(
            ok=False,
            content={"error": "memagent tool requires model access; ensure the agent runtime provides a model."},
        )

    per_call_timeout = float(action_input.get("per_call_timeout_seconds", 180.0))
    llm = _build_llm_fn(runtime_context.model, timeout_seconds=per_call_timeout)
    pattern_analyzer = make_pattern_analyzer(
        llm,
        recurrent_max_context_len=int(action_input.get("recurrent_max_context_len", RECURRENT_MAX_CONTEXT_LEN)),
        recurrent_chunk_size=int(action_input.get("recurrent_chunk_size", RECURRENT_CHUNK_SIZE)),
        max_memory_tokens=int(action_input.get("max_memory_tokens", MEMORY_MAX_TOKENS)),
        per_call_timeout_seconds=per_call_timeout,
        total_timeout_seconds=float(action_input.get("total_timeout_seconds", 300.0)),
        keep_trace=bool(action_input.get("keep_trace", False)),
        use_deterministic_engine=bool(action_input.get("use_deterministic_engine", True)),
        repair_rounds=int(action_input.get("repair_rounds", 2)),
        min_field_coverage=float(action_input.get("min_field_coverage", 0.75)),
        emit_engine_records=bool(action_input.get("emit_engine_records", False)),
    )

    try:
        content = pattern_analyzer(question, full_path)
    except ValueError as exc:
        return ToolExecutionResult(ok=False, content={"error": str(exc)})
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            content={"error": f"MemAgent failed to process document: {exc}"},
        )

    result_content: dict[str, Any] = {
        "extraction_guide": content["answer"],
        "source": path,
        "diagnostics": content.get("diagnostics", {}),
    }
    if "steps" in content:
        result_content["steps"] = content["steps"]
    if "chunk_count" in content:
        result_content["chunk_count"] = content["chunk_count"]
    if "pattern_spec" in content:
        result_content["pattern_spec"] = content["pattern_spec"]
    if "engine_diagnostics" in content:
        result_content["engine_diagnostics"] = content["engine_diagnostics"]
    if "engine_records" in content:
        result_content["engine_records"] = content["engine_records"]
    return ToolExecutionResult(ok=True, content=result_content)


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
            description=(
                "Submit the final answer table and terminate the task. Use only after "
                "the required evidence has been inspected and the final rows are fully "
                "computed. Do not use this for intermediate notes or previews."
            ),
            args_schema=AnswerArgs,
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description=(
                "Run a read-only SQL query against one specific SQLite/.db file inside "
                "context. Use when the relevant source is a known SQLite database and "
                "you need exact SQL over its native tables. Prefer execute_probe_query "
                "for first-pass probing across CSV/JSON/SQLite or when the catalog "
                "already exposes convenient DuckDB views."
            ),
            args_schema=ExecuteContextSqlArgs,
        ),
        "execute_probe_query": ToolSpec(
            name="execute_probe_query",
            description=(
                "Execute batched read-only SQL probes against task data files (CSV, "
                "JSON, SQLite) using DuckDB. Use this as the default tool for "
                "understanding structured data: candidate-field checks, COUNTs, "
                "DISTINCT scans, sample rows, filters, joins that DuckDB can express, "
                "and quick aggregations. MANDATORY: pack multiple independent queries "
                "into ONE call whenever possible instead of sending them one by one. "
                "Each query in queries must be SELECT or WITH. "
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
                "Use when SQL tools are not enough: complex multi-file transformations, "
                "custom parsing, iterative logic, exact final row construction, or "
                "machine-readable JSON export of a large/intermediate result. Avoid using "
                "Python just to list files, grep text documents, or run simple COUNT/"
                "DISTINCT/sample probes that execute_probe_query can handle. "
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
                "or text document from the catalog. Use before read_doc so you can "
                "choose a targeted heading instead of reading a full document. "
                "Returns headings with level (1 = '#', 2 = '##', etc.). "
            ),
            args_schema=LookupDocOutlineArgs,
        ),
        "list_context": ToolSpec(
            name="list_context",
            description=(
                "List files and directories available under context. Use to discover "
                "available assets, resolve an unknown/missing path, or inspect "
                "non-structural files. If the catalog already names the needed "
                "CSV/JSON/SQLite assets, go directly to execute_probe_query instead."
            ),
            args_schema=ListContextArgs,
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document inside context, optionally restricted to "
                "one heading. Use after lookup_doc_outline when you know the relevant "
                "section. Prefer search_doc first when you do not know which document "
                "or heading contains the needed fact."
            ),
            args_schema=ReadDocArgs,
        ),
        "search_doc": ToolSpec(
            name="search_doc",
            description=(
                "Search text documents (.md, .txt, .rst) inside context for a regex "
                "pattern or plain keyword. Returns matching lines with surrounding "
                "context lines and their locations. "
                "Results are paginated (default 20 matches per page, 1-indexed). "
                "Use 'page' to request different pages; check 'total_pages' and "
                "'total_matches' in the response to know how many pages are available. "
                "Use this before read_doc to locate relevant sections when you "
                "don't know where the information lives, instead of writing Python "
                "code to grep through documents."
            ),
            args_schema=SearchDocArgs,
        ),
        "memagent": ToolSpec(
            name="memagent",
            description=(
                "Analyze the structure of a long markdown/text document and produce "
                "regex/Python extraction patterns. This tool scans the document "
                "chunk-by-chunk using an internal LLM, identifies repeating text "
                "patterns around data fields, and returns a structured extraction "
                "guide with regexes and complete Python code. "
                "Use this when `read_doc` truncation hides critical sections of a "
                "large document, or when data fields are embedded in prose across "
                "multiple sections. The returned code should be executed via "
                "`execute_python` to extract the actual data with 100% accuracy. "
                "The question should describe what data you need, e.g.: "
                "'Identify where height, weight, ID, and publisher fields appear in "
                "this document and produce regex patterns to extract them all.'"
            ),
            args_schema=MemAgentArgs,
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
        "memagent": _memagent,
    }
    return ToolRegistry(
        specs=specs,
        handlers=handlers,
        tool_config=tool_config if tool_config is not None else ToolConfig(),
    )
