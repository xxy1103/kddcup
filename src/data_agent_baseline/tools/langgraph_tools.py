from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class AnswerArgs(BaseModel):
    columns: list[str] = Field(description="Header row of the final answer table.")
    rows: list[list[Any]] = Field(description="Data rows of the final answer table.")


class ExecuteContextSqlArgs(BaseModel):
    path: str = Field(
        description="Relative path to a sqlite/db file under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )
    sql: str = Field(description="A read-only SQL query. Only SELECT, WITH, and PRAGMA are allowed.")
    limit: int = Field(default=200, description="Maximum number of rows to return.")


class ExecutePythonArgs(BaseModel):
    code: str = Field(description="Python code to execute inside the task's temporary context workspace.")


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
            "Use lookup_schema to discover available tables and fields. "
            "IMPORTANT: do NOT wrap table references in single quotes in SQL. "
            "Write FROM qualifying, not FROM 'qualifying' or FROM 'csv/qualifying.csv'. "
            "Single quotes create string literals, not table references."
        ),
    )
    limit: int = Field(default=5, description="Maximum number of rows to return per query (default 5, max 200).")


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


class LookupSchemaArgs(BaseModel):
    field_refs: list[str] = Field(
        description=(
            "List of field references to look up in one batch call. "
            "Each reference can be in one of these formats:\n"
            "- 'path/to/file.csv.field_name'  for CSV or JSON files\n"
            "- 'path/to/file.db.table.field_name'  for SQLite databases\n"
            "- 'field_name'  partial match, searched across all assets\n\n"
            "The path prefix is the asset_path from the catalog (e.g., 'csv/trans.csv'). "
            "Copy it verbatim, keeping the slashes. "
            "Batch multiple fields into one call for efficiency. "
            "Examples: ['csv/trans.csv.type', 'csv/trans.csv.amount', 'data/events.db.races.raceId']"
        ),
    )


class LookupDocOutlineArgs(BaseModel):
    path: str = Field(
        description="Relative path to a text document under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )


class ListContextArgs(BaseModel):
    max_depth: int = Field(default=4, description="Maximum directory recursion depth to list.")


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
        default=3,
        description="Number of lines before and after each matching line to include for context.",
    )
    path: str | None = Field(
        default=None,
        description="Optional: restrict search to a single document file by its relative path. If omitted, searches all .md, .txt, and .rst files under context.",
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
