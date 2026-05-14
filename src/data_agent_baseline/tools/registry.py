from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.config import DataInspectorSampleBudget, ToolConfig
from data_agent_baseline.inspectors.semantic_catalog import STRUCTURAL_KINDS, build_semantic_catalog
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
    ExecutePythonArgs,
    ListContextArgs,
    LookupDocOutlineArgs,
    LookupSchemaArgs,
    ReadDocArgs,
    SearchDocArgs,
    create_structured_tool,
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
    _catalog_cache: dict[str, Any] | None = field(default=None, repr=False)
    _enrichment_map: dict[str, dict[str, str]] | None = field(default=None, repr=False)

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
            budget=DataInspectorSampleBudget(),
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


def _resolve_field_ref(
    ref: str,
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve a field reference against catalog schemas.

    Supports:
      - "asset.field" for CSV/JSON
      - "asset.table.field" for SQLite
      - "field" partial match across all assets
    """
    matches: list[dict[str, Any]] = []
    parts = ref.split(".")
    for schema in schemas:
        asset = schema.get("asset_path", "")
        asset_name = asset.rsplit("/", 1)[-1]  # "member.csv" from "csv/member.csv"
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                table_name = table.get("name", "")
                for field in table.get("fields", []):
                    field_name = field.get("name", "")
                    candidates = [
                        f"{asset}.{table_name}.{field_name}",
                        f"{asset_name}.{table_name}.{field_name}",
                        f"{asset.replace('/', '.')}.{table_name}.{field_name}",
                        f"{asset_name.rsplit('.', 1)[0]}.{table_name}.{field_name}",
                        f"{table_name}.{field_name}",
                        field_name,
                    ]
                    if ref in candidates or (len(parts) == 1 and field_name == ref):
                        matches.append({
                            "asset_path": asset,
                            "table": table_name,
                            "field": field,
                        })
        elif schema.get("kind") in {"csv", "json"}:
            for field in schema.get("fields", []):
                field_name = field.get("name", "")
                candidates = [
                    f"{asset}.{field_name}",
                    f"{asset_name}.{field_name}",
                    f"{asset.replace('/', '.')}.{field_name}",
                    f"{asset_name.rsplit('.', 1)[0]}.{field_name}",
                    field_name,
                ]
                if ref in candidates or (len(parts) == 1 and field_name == ref):
                    matches.append({
                        "asset_path": asset,
                        "table": None,
                        "field": field,
                    })
    return matches


def _format_field_ref(asset: str, table: str | None, field: str) -> str:
    if table:
        return f"{asset}.{table}.{field}"
    return f"{asset}.{field}"


def _build_field_enrichment_map(enriched_catalog: dict[str, Any]) -> dict[str, dict[str, str]]:
    enrichment_map: dict[str, dict[str, str]] = {}
    for schema in enriched_catalog.get("schemas", []):
        asset_path = schema.get("asset_path", "")
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                table_name = table.get("name", "")
                for field in table.get("fields", []):
                    entry = _extract_enrichment_entry(field)
                    if entry:
                        enrichment_map[f"{asset_path}.{table_name}.{field.get('name', '')}"] = entry
        else:
            for field in schema.get("fields", []):
                entry = _extract_enrichment_entry(field)
                if entry:
                    enrichment_map[f"{asset_path}.{field.get('name', '')}"] = entry
    return enrichment_map


def _extract_enrichment_entry(field: dict[str, Any]) -> dict[str, str]:
    entry: dict[str, str] = {}
    desc = field.get("description")
    if isinstance(desc, str) and desc.strip():
        entry["description"] = desc.strip()
    note = field.get("note")
    if isinstance(note, str) and note.strip():
        entry["note"] = note.strip()
    return entry


def _collect_asset_field_refs(
    asset_path: str,
    table_filter: str | None,
    schemas: list[dict[str, Any]],
) -> list[str]:
    refs: list[str] = []
    for schema in schemas:
        if schema.get("asset_path") != asset_path:
            continue
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                if table_filter is not None and table["name"] != table_filter:
                    continue
                for field in table.get("fields", []):
                    r = _format_field_ref(asset_path, table["name"], field["name"])
                    if r not in refs:
                        refs.append(r)
        elif schema.get("kind") in {"csv", "json"}:
            for field in schema.get("fields", []):
                r = _format_field_ref(asset_path, None, field["name"])
                if r not in refs:
                    refs.append(r)
    return refs


def _get_related_field_refs(
    asset_path: str,
    table: str | None,
    field_name: str,
    schemas: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> list[str]:
    related: list[str] = []

    # 1. Co-located fields from same asset/table (excluding the queried field)
    for ref in _collect_asset_field_refs(asset_path, table, schemas):
        field_part = ref.rsplit(".", 1)[-1]
        if (table and ".".join(ref.split(".")[1:]) != f"{table}.{field_name}") \
                or (not table and field_part != field_name):
            if ref not in related:
                related.append(ref)

    # 2. Join-connected assets (1-hop BFS over relationships)
    seen_assets: set[str] = {asset_path}
    for rel in relationships:
        source = rel.get("source", {})
        target = rel.get("target", {})
        src_path = source.get("asset_path", "")
        tgt_path = target.get("asset_path", "")

        if src_path in seen_assets and tgt_path not in seen_assets:
            seen_assets.add(tgt_path)
            for ref in _collect_asset_field_refs(tgt_path, None, schemas):
                if ref not in related:
                    related.append(ref)
        elif tgt_path in seen_assets and src_path not in seen_assets:
            seen_assets.add(src_path)
            for ref in _collect_asset_field_refs(src_path, None, schemas):
                if ref not in related:
                    related.append(ref)

    return related


def _build_join_hints(
    asset_path: str,
    relationships: list[dict[str, Any]],
) -> list[str]:
    hints: list[str] = []
    for rel in relationships:
        source = rel.get("source", {})
        target = rel.get("target", {})
        src_path = source.get("asset_path", "")
        tgt_path = target.get("asset_path", "")

        if asset_path not in (src_path, tgt_path):
            continue

        src_fields = source.get("fields", [])
        tgt_fields = target.get("fields", [])
        src_table = source.get("table")
        tgt_table = target.get("table")
        rel_type = rel.get("relationship_type", "unknown")
        confidence = rel.get("confidence", 0)
        cardinality = rel.get("cardinality", "unknown")

        src_ref = _format_field_ref(src_path, src_table, src_fields[0] if src_fields else "?")
        tgt_ref = _format_field_ref(tgt_path, tgt_table, tgt_fields[0] if tgt_fields else "?")

        hint = f"{src_ref} ↔ {tgt_ref} ({rel_type}, confidence {confidence:.2f})"
        if cardinality and cardinality != "unknown":
            hint += f" [{cardinality}]"

        # Add type-based normalisation tip
        evidence = rel.get("evidence", {})
        if evidence.get("explicit_sqlite_foreign_key"):
            hint += " [explicit FK]"
        if evidence.get("value_profile_capped"):
            hint += " [large cardinality: verify with sample]"

        hints.append(hint)

    return hints


def _lookup_schema(runtime_context: ToolRuntimeContext, action_input: dict[str, Any]) -> ToolExecutionResult:
    field_ref = str(action_input["field_ref"])

    # 1. Build and cache the full catalog on first call
    if runtime_context._catalog_cache is None:
        runtime_context._catalog_cache = build_semantic_catalog(
            runtime_context.task,
            budget=DataInspectorSampleBudget(),
            max_depth=20,
            include_relationships=True,
        )

    catalog = runtime_context._catalog_cache
    schemas = [s for s in catalog["schemas"] if s.get("kind") in STRUCTURAL_KINDS]
    relationships = catalog.get("relationships", [])

    # 2. Resolve field reference
    matches = _resolve_field_ref(field_ref, schemas)

    if not matches:
        return ToolExecutionResult(
            ok=False,
            content={
                "error": f"No field found matching '{field_ref}'.",
                "hint": (
                    "Use one of these formats:\n"
                    "- 'csv/data.csv.field'  (asset_path + '.' + field name) for CSV/JSON\n"
                    "- 'data/db.sqlite.table.field'  (asset_path + '.' + table + '.' + field name) for SQLite\n"
                    "- 'field'  partial field name, searched across all assets\n\n"
                    "Do NOT replace '/' with '.' in the path. "
                    "The asset_path includes the directory prefix and file extension. "
                    "Use list_context to discover available assets."
                ),
            },
        )

    # Use the most specific match: prefer exact over partial
    match = matches[0]
    field_detail: dict[str, Any] = dict(match["field"])

    # Inject description/note from semantic enrichment if available
    if runtime_context._enrichment_map:
        field_key = _format_field_ref(
            match["asset_path"], match["table"],
            field_detail.get("name", ""),
        )
        enrichment = runtime_context._enrichment_map.get(field_key)
        if enrichment:
            if "description" in enrichment:
                field_detail["description"] = enrichment["description"]
            if "note" in enrichment:
                field_detail["note"] = enrichment["note"]

    # 3. Compute related fields and join hints
    related = _get_related_field_refs(
        match["asset_path"], match["table"],
        field_detail.get("name", ""), schemas, relationships,
    )
    join_hints = _build_join_hints(match["asset_path"], relationships)

    resolved = _format_field_ref(match["asset_path"], match["table"], field_detail.get("name", ""))

    return ToolExecutionResult(
        ok=True,
        content={
            "field": field_ref,
            "resolved_to": resolved,
            "field_details": field_detail,
            "related_fields": related,
            "join_hints": join_hints,
        },
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
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute Python code inside a per-task temporary copy of the context directory. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            args_schema=ExecutePythonArgs,
        ),
        "lookup_schema": ToolSpec(
            name="lookup_schema",
            description=(
                "Look up full schema details for a specific field reference. "
                "Returns field type, cardinality, distinct values, min/max, "
                "related fields from the same table and join-connected tables, "
                "and join hints. "
                "Accepts 'path/to/file.csv.field' (CSV/JSON), "
                "'path/to/file.db.table.field' (SQLite), "
                "or just 'field' to search across all assets. "
                "Use the asset_path from the catalog with slashes, e.g., "
                "'csv/trans.csv.type'."
            ),
            args_schema=LookupSchemaArgs,
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
        "execute_python": _execute_python,
        "lookup_doc_outline": _lookup_doc_outline,
        "lookup_schema": _lookup_schema,
        "list_context": _list_context,
        "read_doc": _read_doc,
        "search_doc": _search_doc,
    }
    return ToolRegistry(
        specs=specs,
        handlers=handlers,
        tool_config=tool_config if tool_config is not None else ToolConfig(),
    )
