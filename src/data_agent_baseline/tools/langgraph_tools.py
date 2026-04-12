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


class InspectSqliteSchemaArgs(BaseModel):
    path: str = Field(
        description="Relative path to a sqlite/db file under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )


class ListContextArgs(BaseModel):
    max_depth: int = Field(default=4, description="Maximum directory recursion depth to list.")


class ReadCsvArgs(BaseModel):
    path: str = Field(
        description="Relative path to a CSV file under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )
    max_rows: int = Field(default=20, description="Maximum number of data rows to preview.")


class ReadDocArgs(BaseModel):
    path: str = Field(
        description="Relative path to a text-like document under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )
    max_chars: int = Field(default=4000, description="Maximum number of characters to preview.")


class ReadJsonArgs(BaseModel):
    path: str = Field(
        description="Relative path to a JSON file under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )
    max_chars: int = Field(default=4000, description="Maximum number of characters to preview.")


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
