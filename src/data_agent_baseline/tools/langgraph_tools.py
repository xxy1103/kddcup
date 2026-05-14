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


class LookupSchemaArgs(BaseModel):
    field_ref: str = Field(
        description=(
            "Field reference in one of these formats:\n"
            "- 'path/to/file.csv.field_name'  for CSV or JSON files\n"
            "- 'path/to/file.db.table.field_name'  for SQLite databases\n"
            "- 'field_name'  partial match, searched across all assets\n\n"
            "The path prefix is the asset_path from the catalog (e.g., 'csv/trans.csv'). "
            "Copy it verbatim, keeping the slashes. "
            "Examples: 'csv/trans.csv.type', 'data/events.db.races.raceId', 'type'"
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
