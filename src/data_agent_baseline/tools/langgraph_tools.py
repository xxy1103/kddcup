from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class AnswerArgs(BaseModel):
    columns: list[str] = Field(
        description="Exact final answer column names requested by the question, with no extra evidence or helper columns."
    )
    rows: list[list[Any]] = Field(
        description="Fully computed final data rows aligned to columns. Each row must have exactly len(columns) cells."
    )


class SubmitToolResultArgs(BaseModel):
    tool_name: str = Field(
        description=(
            "The data tool to execute for generating the answer. "
            "Supported: 'execute_probe_query', 'execute_python', 'execute_context_sql'."
        )
    )
    tool_args: dict[str, Any] = Field(
        description=(
            "The arguments to pass to the specified tool. "
            "Use the same argument format as calling the tool directly. "
            "For final submission, execute_probe_query and execute_context_sql results "
            "are fetched completely; any limit value here is ignored."
        )
    )
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Optional: override or reorder the answer columns. "
            "If omitted, columns are extracted from the tool's output automatically."
        ),
    )


class ExecuteContextSqlArgs(BaseModel):
    path: str = Field(
        description="Relative path to a sqlite/db file under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )
    sql: str = Field(description="A read-only SQL query. Only SELECT, WITH, and PRAGMA are allowed.")
    limit: int = Field(default=200, description="Maximum number of rows to return.")


class ExecutePythonArgs(BaseModel):
    code: str = Field(
        description=(
            "Python code to execute inside the task's temporary context workspace. "
            "Use query(sql) or query_rows(sql) to query the same logical tables exposed "
            "to execute_probe_query; do not use a bare duckdb.connect(':memory:') for "
            "logical tables. "
            "Read files by paths relative to context. For final results, print full "
            "machine-readable JSON rather than pandas previews."
        )
    )


class ExecuteProbeQueryArgs(BaseModel):
    queries: list[str] = Field(
        description=(
            "A list of read-only SQL queries to execute against task data files. "
            "Each query must be SELECT or WITH. "
            "All queries share the same in-memory DuckDB connection and views, "
            "so you can batch multiple lookups into one efficient call — "
            "e.g., send COUNT + DISTINCT + sample rows together instead of "
            "three separate calls. "
            "CSV/JSON files are accessible by their file-name stem (no extension), "
            "e.g., 'trans' or 'member'. SQLite tables by their table name, "
            "or by 'asset_path_stem__table_name' if the name collides. "
            "Asset paths with directory prefixes can also be used, e.g., "
            "'csv/trans.csv' is resolved to the view 'trans'. "
            "IMPORTANT: do NOT wrap table references in single quotes in SQL. "
            "Write FROM qualifying, not FROM 'qualifying' or FROM 'csv/qualifying.csv'. "
            "Single quotes create string literals, not table references."
        ),
    )
    limit: int = Field(default=5, description="Preview row limit per query (default 5, max 200). Final submission via submit_tool_result is not limited by this.")


class GetColumnDistinctValuesArgs(BaseModel):
    table: str = Field(
        description=(
            "The table name to inspect. For CSV/JSON this is the file-name stem "
            "(e.g., 'member' for 'csv/member.csv'). For SQLite this is the table "
            "name (e.g., 'users')."
        ),
    )
    column: str = Field(description="The column/field name to inspect.")
    top_n: int = Field(default=20, description="Maximum number of distinct values to return, ranked by frequency.")


class SearchSemanticCatalogArgs(BaseModel):
    query: str = Field(description="Keyword to search across logical table names, columns, documents, and relationships.")
    scope: str = Field(
        default="all",
        description="One of all, tables, fields, documents, relationships, uncertainties.",
    )
    limit: int = Field(default=20, description="Maximum number of matches to return.")


class GetTableProfileArgs(BaseModel):
    table: str = Field(description="Logical table name from the lightweight catalog.")


class GetFieldProfileArgs(BaseModel):
    table: str = Field(description="Logical table name from the lightweight catalog.")
    column: str = Field(description="Column name to inspect.")


class GetTableRelationshipsArgs(BaseModel):
    table: str = Field(description="Logical table name from the lightweight catalog.")


class ReadContextImageArgs(BaseModel):
    path: str = Field(description="Relative path to an image under the task context directory.")
    detail: str = Field(default="auto", description="Image detail hint: auto, low, or high.")


class LookupDocOutlineArgs(BaseModel):
    path: str = Field(
        description="Relative path to a text document under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )


class ListContextArgs(BaseModel):
    max_depth: int = Field(
        default=4,
        description="Maximum directory recursion depth to list. Use a small depth first unless paths are still missing.",
    )


class ReadDocArgs(BaseModel):
    path: str = Field(
        description="Relative path to a text-like document under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )
    heading: str | None = Field(
        default=None,
        description="Optional heading name to read a specific section. Prefer the exact heading text from lookup_doc_outline. If numbering is omitted, read_doc can also match the corresponding numbered heading case-insensitively. Content is extracted from the matched heading until the next heading of the same or higher level.",
    )


class SearchDocArgs(BaseModel):
    query: str = Field(
        description="Regex pattern or plain text keyword to search for. Case-insensitive. Use character classes like \\d, \\w, \\s for flexible matching. Examples: 'patient \\d{6,7}', 'creatinine', 'normal range', 'TR\\d{3}'."
    )
    context_lines: int = Field(
        default=5,
        description="Number of lines before and after each matching line to include for context.",
    )
    path: str | None = Field(
        default=None,
        description="Optional: restrict search to a single document file by its relative path. If omitted, searches all .md, .txt, and .rst files under context.",
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed) of results to return. Use this along with page_size to paginate through large result sets.",
    )
    page_size: int = Field(
        default=20,
        ge=0,
        description="Number of matches per page. Set to 0 to return all matches (no pagination). Default is 20.",
    )


def create_structured_tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    invoke: Callable[..., dict[str, Any]],
) -> BaseTool:
    return StructuredTool.from_function(
        func=invoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )
