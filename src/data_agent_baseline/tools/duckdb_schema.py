"""Shared DuckDB view registration and schema inspection helpers."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import duckdb


DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE = 16 * 1024 * 1024


def quote_duckdb_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_duckdb_path(path: Path) -> str:
    return str(path).replace("'", "''")


def asset_view_name(asset_path: str) -> str:
    return Path(asset_path).stem


def json_schema_has_records_fields(schema: dict[str, Any]) -> bool:
    if schema.get("json_structure") == "object_with_records":
        return True
    return any(
        str(field.get("name", "")).startswith("records.")
        or str(field.get("json_path", "")).startswith("records.")
        for field in schema.get("fields", [])
    )


def sqlite_view_names(catalog: dict[str, Any]) -> dict[tuple[str, str], str]:
    reserved_names = {
        asset_view_name(str(schema.get("asset_path", "")))
        for schema in catalog.get("schemas", [])
        if schema.get("kind") in {"csv", "json"} and schema.get("asset_path")
    }
    sqlite_tables: list[tuple[str, str]] = []
    table_counts: dict[str, int] = {}
    for schema in catalog.get("schemas", []):
        if schema.get("kind") != "sqlite":
            continue
        asset_path = str(schema.get("asset_path", ""))
        for table in schema.get("tables", []):
            table_name = str(table.get("name", ""))
            if not asset_path or not table_name:
                continue
            sqlite_tables.append((asset_path, table_name))
            table_counts[table_name] = table_counts.get(table_name, 0) + 1

    used_names = set(reserved_names)
    view_names: dict[tuple[str, str], str] = {}
    for asset_path, table_name in sqlite_tables:
        if table_counts.get(table_name, 0) == 1 and table_name not in used_names:
            view_name = table_name
        else:
            view_name = f"{asset_view_name(asset_path)}__{table_name}"
        base_name = view_name
        suffix = 2
        while view_name in used_names:
            view_name = f"{base_name}__{suffix}"
            suffix += 1
        used_names.add(view_name)
        view_names[(asset_path, table_name)] = view_name
    return view_names


def connect_sqlite_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _json_maximum_object_size(path: Path) -> int:
    try:
        return max(DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE, path.stat().st_size + 1024 * 1024)
    except OSError:
        return DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE


def _register_sqlite_view(
    conn: duckdb.DuckDBPyConnection,
    file_path: Path,
    sqlite_table: str,
    view_name: str,
) -> None:
    import pandas as pd

    with connect_sqlite_read_only(file_path) as sqlite_conn:
        dataframe = pd.read_sql_query(
            f"SELECT * FROM {quote_duckdb_identifier(sqlite_table)}",
            sqlite_conn,
        )
    temp_name = (
        f"__sqlite_probe_{len(view_name)}_"
        f"{abs(hash((str(file_path), sqlite_table))) & 0xFFFFFFFF}"
    )
    conn.register(temp_name, dataframe)
    conn.execute(
        f"CREATE VIEW {quote_duckdb_identifier(view_name)} AS "
        f"SELECT * FROM {quote_duckdb_identifier(temp_name)}"
    )


def _sql_references_view(sql: str, view_name: str) -> bool:
    if quote_duckdb_identifier(view_name) in sql:
        return True
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(view_name)}(?![A-Za-z0-9_])"
    return re.search(pattern, sql) is not None


def create_duckdb_views(
    conn: duckdb.DuckDBPyConnection,
    context_dir: Path,
    catalog: dict[str, Any],
    *,
    logical_tables: list[dict[str, Any]],
    sql: str | None = None,
    register_all_sqlite: bool = False,
) -> None:
    logical_by_source = {
        (str(table.get("source_asset_path")), table.get("source_table")): str(table.get("table"))
        for table in logical_tables
    }
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        if kind not in ("csv", "json"):
            continue
        file_path = context_dir / asset_path
        table_name = asset_view_name(asset_path)
        quoted_table = quote_duckdb_identifier(table_name)
        safe_path = quote_duckdb_path(file_path)
        try:
            if kind == "csv":
                conn.execute(
                    f"CREATE VIEW {quoted_table} AS "
                    f"SELECT * FROM read_csv_auto('{safe_path}')"
                )
                logical_name = logical_by_source.get((asset_path, None))
                if logical_name and logical_name != table_name:
                    conn.execute(
                        f"CREATE VIEW {quote_duckdb_identifier(logical_name)} "
                        f"AS SELECT * FROM {quoted_table}"
                    )
                continue
            max_object_size = _json_maximum_object_size(file_path)
            if json_schema_has_records_fields(schema):
                conn.execute(
                    f"CREATE VIEW {quoted_table} AS "
                    f"SELECT r.*, r AS records "
                    f"FROM read_json_auto('{safe_path}', maximum_object_size={max_object_size}), "
                    "UNNEST(records) AS t(r)"
                )
                logical_name = logical_by_source.get((asset_path, None))
                if logical_name and logical_name != table_name:
                    conn.execute(
                        f"CREATE VIEW {quote_duckdb_identifier(logical_name)} "
                        f"AS SELECT * FROM {quoted_table}"
                    )
                continue
            conn.execute(
                f"CREATE VIEW {quoted_table} AS "
                f"SELECT * FROM read_json_auto('{safe_path}', maximum_object_size={max_object_size})"
            )
            logical_name = logical_by_source.get((asset_path, None))
            if logical_name and logical_name != table_name:
                conn.execute(
                    f"CREATE VIEW {quote_duckdb_identifier(logical_name)} "
                    f"AS SELECT * FROM {quoted_table}"
                )
        except Exception:
            continue

    view_names = sqlite_view_names(catalog)
    for schema in catalog.get("schemas", []):
        if schema.get("kind") != "sqlite":
            continue
        asset_path = str(schema.get("asset_path", ""))
        file_path = context_dir / asset_path
        for table in schema.get("tables", []):
            table_name = str(table.get("name", ""))
            view_name = view_names.get((asset_path, table_name))
            if not view_name:
                continue
            if not register_all_sqlite and (sql is None or not _sql_references_view(sql, view_name)):
                continue
            try:
                _register_sqlite_view(conn, file_path, table_name, view_name)
                logical_name = logical_by_source.get((asset_path, table_name))
                if logical_name and logical_name != view_name:
                    conn.execute(
                        f"CREATE VIEW {quote_duckdb_identifier(logical_name)} AS "
                        f"SELECT * FROM {quote_duckdb_identifier(view_name)}"
                    )
            except Exception:
                continue


def inspect_duckdb_logical_schemas(
    context_dir: Path,
    catalog: dict[str, Any],
    *,
    logical_tables: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Return DuckDB column types for every logical table that can be registered."""
    conn = duckdb.connect(":memory:")
    try:
        create_duckdb_views(
            conn,
            context_dir,
            catalog,
            logical_tables=logical_tables,
            register_all_sqlite=True,
        )
        schemas: dict[str, dict[str, str]] = {}
        for logical in logical_tables:
            table_name = str(logical.get("table", ""))
            if not table_name:
                continue
            try:
                rows = conn.execute(
                    f"DESCRIBE {quote_duckdb_identifier(table_name)}"
                ).fetchall()
            except Exception:
                continue
            schemas[table_name] = {str(row[0]): str(row[1]) for row in rows}
        return schemas
    finally:
        conn.close()
