from __future__ import annotations

import base64
from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.config import DataInspectorSampleBudget, ToolConfig
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog, iter_logical_tables
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
    GetFieldProfileArgs,
    GetTableProfileArgs,
    GetTableRelationshipsArgs,
    ListContextArgs,
    LookupDocOutlineArgs,
    ReadContextImageArgs,
    ReadDocArgs,
    SearchDocArgs,
    SearchSemanticCatalogArgs,
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
    model_content_parts: list[dict[str, Any]] = field(default_factory=list)


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


def _ensure_catalog(runtime_context: ToolRuntimeContext) -> dict[str, Any]:
    if runtime_context._catalog_cache is None:
        runtime_context._catalog_cache = build_semantic_catalog(
            runtime_context.task,
            budget=runtime_context.budget,
            max_depth=20,
            include_relationships=True,
        )
    return runtime_context._catalog_cache


def _logical_tables_by_name(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(table["table"]): table for table in iter_logical_tables(catalog)}


def _resolve_logical_table(catalog: dict[str, Any], table_name: str) -> dict[str, Any] | None:
    normalized = table_name.strip().strip('"')
    tables = _logical_tables_by_name(catalog)
    if normalized in tables:
        return tables[normalized]
    lowered = normalized.lower()
    for table, entry in tables.items():
        if table.lower() == lowered:
            return entry
    return None


def _strip_source_details(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_source_details(item)
            for key, item in value.items()
            if key not in {"asset_path", "source_asset_path"}
        }
    if isinstance(value, list):
        return [_strip_source_details(item) for item in value]
    return value


def _schema_for_logical_table(catalog: dict[str, Any], logical_table: dict[str, Any]) -> dict[str, Any] | None:
    asset_path = logical_table.get("source_asset_path")
    source_table = logical_table.get("source_table")
    for schema in catalog.get("schemas", []):
        if schema.get("asset_path") != asset_path:
            continue
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                if table.get("name") == source_table:
                    return {
                        "table": logical_table["table"],
                        "kind": "sqlite",
                        "row_count": table.get("row_count"),
                        "fields": table.get("fields", []),
                    }
            return None
        return {
            "table": logical_table["table"],
            "kind": schema.get("kind"),
            "row_count": schema.get("row_count"),
            "json_structure": schema.get("json_structure"),
            "fields": schema.get("fields", []),
        }
    return None


def _search_semantic_catalog(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    query = str(action_input["query"]).strip().lower()
    scope = str(action_input.get("scope", "all")).strip().lower()
    limit = max(1, min(int(action_input.get("limit", 20)), 100))
    matches: list[dict[str, Any]] = []

    def in_scope(name: str) -> bool:
        return scope in {"all", name}

    if in_scope("tables") or in_scope("fields"):
        for logical in iter_logical_tables(catalog):
            table_name = str(logical["table"])
            if in_scope("tables") and query in table_name.lower():
                matches.append({"type": "table", "table": table_name, "row_count": logical.get("row_count")})
            if in_scope("fields"):
                for column in logical.get("columns", []):
                    column_name = str(column.get("name", ""))
                    if query in column_name.lower() or query in table_name.lower():
                        matches.append(
                            {
                                "type": "field",
                                "table": table_name,
                                "column": column_name,
                                "field_type": column.get("type"),
                            }
                        )

    if in_scope("documents"):
        for asset in catalog.get("assets", []):
            if asset.get("kind") != "document":
                continue
            path = str(asset.get("asset_path", ""))
            if query in path.lower():
                matches.append({"type": "document", "path": path, "size": asset.get("size")})

    if in_scope("relationships"):
        table_lookup = _logical_tables_by_name(catalog)
        source_to_table: dict[tuple[str, str | None], str] = {}
        for name, logical in table_lookup.items():
            source_to_table[(str(logical.get("source_asset_path")), logical.get("source_table"))] = name
        for rel in catalog.get("relationships", []):
            source = rel.get("source", {})
            target = rel.get("target", {})
            source_table = source_to_table.get((str(source.get("asset_path")), source.get("table")))
            target_table = source_to_table.get((str(target.get("asset_path")), target.get("table")))
            text = " ".join(
                [
                    str(source_table or ""),
                    str(target_table or ""),
                    " ".join(str(v) for v in source.get("fields", [])),
                    " ".join(str(v) for v in target.get("fields", [])),
                    str(rel.get("relationship_type", "")),
                ]
            ).lower()
            if query in text:
                matches.append(
                    {
                        "type": "relationship",
                        "source_table": source_table,
                        "source_fields": source.get("fields", []),
                        "target_table": target_table,
                        "target_fields": target.get("fields", []),
                        "relationship_type": rel.get("relationship_type"),
                        "confidence": rel.get("confidence"),
                    }
                )

    if in_scope("uncertainties"):
        for item in catalog.get("semantic_uncertainties", []):
            text = " ".join(str(v) for v in item.values()).lower()
            if query in text:
                matches.append({"type": "uncertainty", **item})

    return ToolExecutionResult(
        ok=True,
        content={"query": query, "scope": scope, "matches": matches[:limit], "match_count": len(matches)},
    )


def _get_table_profile(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    table_name = str(action_input["table"])
    logical = _resolve_logical_table(catalog, table_name)
    if logical is None:
        return ToolExecutionResult(ok=False, content={"error": f"Unknown logical table: {table_name}"})
    schema = _schema_for_logical_table(catalog, logical)
    return ToolExecutionResult(ok=schema is not None, content=_strip_source_details(schema or {}))


def _get_field_profile(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    table_result = _get_table_profile(runtime_context, {"table": action_input["table"]})
    if not table_result.ok:
        return table_result
    column = str(action_input["column"]).strip().lower()
    for field in table_result.content.get("fields", []):
        names = {str(field.get("name", "")).lower(), str(field.get("json_path", "")).lower()}
        if column in names:
            return ToolExecutionResult(
                ok=True,
                content={
                    "table": table_result.content.get("table"),
                    "field": field,
                },
            )
    return ToolExecutionResult(
        ok=False,
        content={"error": f"Unknown column {action_input['column']!r} on table {action_input['table']!r}"},
    )


def _get_table_relationships(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    table_name = str(action_input["table"])
    logical = _resolve_logical_table(catalog, table_name)
    if logical is None:
        return ToolExecutionResult(ok=False, content={"error": f"Unknown logical table: {table_name}"})
    source_key = (str(logical.get("source_asset_path")), logical.get("source_table"))
    source_to_table = {
        (str(item.get("source_asset_path")), item.get("source_table")): str(item["table"])
        for item in iter_logical_tables(catalog)
    }
    relationships: list[dict[str, Any]] = []
    for rel in catalog.get("relationships", []):
        source = rel.get("source", {})
        target = rel.get("target", {})
        rel_source_key = (str(source.get("asset_path")), source.get("table"))
        rel_target_key = (str(target.get("asset_path")), target.get("table"))
        if source_key not in {rel_source_key, rel_target_key}:
            continue
        relationships.append(
            {
                "source_table": source_to_table.get(rel_source_key),
                "source_fields": source.get("fields", []),
                "target_table": source_to_table.get(rel_target_key),
                "target_fields": target.get("fields", []),
                "relationship_type": rel.get("relationship_type"),
                "cardinality": rel.get("cardinality"),
                "confidence": rel.get("confidence"),
                "evidence": rel.get("evidence"),
            }
        )
    return ToolExecutionResult(ok=True, content={"table": logical["table"], "relationships": relationships})


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _read_context_image(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    image_path = str(action_input["path"])
    normalized_path = normalize_context_relative_path(image_path)
    path = resolve_context_path(runtime_context.task, normalized_path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return ToolExecutionResult(
            ok=False,
            content={"error": f"Unsupported image type: {normalized_path}. Supported: jpg, jpeg, png, webp."},
        )
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    detail = str(action_input.get("detail", "auto") or "auto")
    return ToolExecutionResult(
        ok=True,
        content={
            "path": normalized_path,
            "mime_type": mime_type,
            "size": path.stat().st_size,
            "status": "image attached to next model request",
        },
        model_content_parts=[
            {"type": "text", "text": f"Image from `{normalized_path}`:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}",
                    "detail": detail,
                },
            },
        ],
    )


def _list_context(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(
        ok=True,
        content=list_context_tree(runtime_context.task, max_depth=max_depth),
    )


def _lookup_doc_outline(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    doc_path = str(action_input["path"])

    catalog = _ensure_catalog(runtime_context)
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
    catalog = _ensure_catalog(runtime_context)
    if "queries" in action_input:
        queries = [str(q) for q in action_input["queries"]]
    elif "sql" in action_input:
        # Backward compatibility for older traces/tests and for models that
        # still emit the pre-batching argument name.
        queries = [str(action_input["sql"])]
    else:
        raise ValueError("execute_probe_query requires `queries` (list[str]).")
    limit = min(int(action_input.get("limit", 200)), 200)
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
    catalog = _ensure_catalog(runtime_context)
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
            description=(
                "Submit the final answer table and terminate the task. Use only after "
                "the required evidence has been inspected and the final rows are fully "
                "computed. Do not use this for intermediate notes or previews."
            ),
            args_schema=AnswerArgs,
        ),
        "execute_probe_query": ToolSpec(
            name="execute_probe_query",
            description=(
                "Execute batched read-only SQL probes against logical tables using DuckDB. "
                "Use this as the default tool for "
                "understanding structured data: candidate-field checks, COUNTs, "
                "DISTINCT scans, sample rows, filters, joins that DuckDB can express, "
                "and quick aggregations. MANDATORY: pack multiple independent queries "
                "into ONE call whenever possible instead of sending them one by one. "
                "Each query in queries must be SELECT or WITH. "
                "Reference tables by the logical table names shown in the lightweight catalog. "
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
                "candidate field mapping, or identify filter values. Use logical table names."
            ),
            args_schema=GetColumnDistinctValuesArgs,
        ),
        "search_semantic_catalog": ToolSpec(
            name="search_semantic_catalog",
            description=(
                "Search the full semantic catalog by keyword. Use this to find candidate "
                "logical tables, fields, documents, relationships, or catalog warnings "
                "without loading the entire catalog into the prompt."
            ),
            args_schema=SearchSemanticCatalogArgs,
        ),
        "get_table_profile": ToolSpec(
            name="get_table_profile",
            description=(
                "Return the full semantic profile for one logical table, including fields, "
                "types, missing counts, cardinalities, top distinct values, and numeric ranges."
            ),
            args_schema=GetTableProfileArgs,
        ),
        "get_field_profile": ToolSpec(
            name="get_field_profile",
            description=(
                "Return the full semantic profile for one field on a logical table, including "
                "type, missing count, cardinality, top distinct values, and numeric range."
            ),
            args_schema=GetFieldProfileArgs,
        ),
        "get_table_relationships": ToolSpec(
            name="get_table_relationships",
            description=(
                "Return inferred and explicit semantic relationships involving one logical table, "
                "including source/target fields, relationship type, confidence, and evidence."
            ),
            args_schema=GetTableRelationshipsArgs,
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
        "read_context_image": ToolSpec(
            name="read_context_image",
            description=(
                "Attach an image from context to the next model request. Use this for stable "
                "video frames or other image evidence after inspecting the timeline or file list."
            ),
            args_schema=ReadContextImageArgs,
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
    }
    handlers = {
        "answer": _answer,
        "execute_probe_query": _execute_probe_query,
        "execute_python": _execute_python,
        "get_column_distinct_values": _get_column_distinct_values,
        "search_semantic_catalog": _search_semantic_catalog,
        "get_table_profile": _get_table_profile,
        "get_field_profile": _get_field_profile,
        "get_table_relationships": _get_table_relationships,
        "lookup_doc_outline": _lookup_doc_outline,
        "list_context": _list_context,
        "read_doc": _read_doc,
        "read_context_image": _read_context_image,
        "search_doc": _search_doc,
    }
    return ToolRegistry(
        specs=specs,
        handlers=handlers,
        tool_config=tool_config if tool_config is not None else ToolConfig(),
    )
