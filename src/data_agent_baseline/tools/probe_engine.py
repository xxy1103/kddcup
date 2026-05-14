"""Probe engine for executing data-level queries against task data files."""

from __future__ import annotations

import re
import sqlite3
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import duckdb


DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE = 16 * 1024 * 1024


def _validate_read_only_sql(sql: str) -> None:
    normalized = sql.strip().lstrip("(").lstrip()
    if not normalized.upper().startswith(("SELECT", "WITH")):
        raise ValueError(f"Only SELECT/WITH statements are allowed. Got: {sql[:120]}")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _json_maximum_object_size(path: Path) -> int:
    try:
        return max(DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE, path.stat().st_size + 1024 * 1024)
    except OSError:
        return DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE


def _connect_sqlite_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _asset_view_name(asset_path: str) -> str:
    return Path(asset_path).stem


def _json_schema_has_records_fields(schema: dict[str, Any]) -> bool:
    if schema.get("json_structure") == "object_with_records":
        return True
    return any(
        str(field.get("name", "")).startswith("records.")
        or str(field.get("json_path", "")).startswith("records.")
        for field in schema.get("fields", [])
    )


def _sqlite_view_names(catalog: dict[str, Any]) -> dict[tuple[str, str], str]:
    reserved_names = {
        _asset_view_name(str(schema.get("asset_path", "")))
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
            view_name = f"{_asset_view_name(asset_path)}__{table_name}"
        base_name = view_name
        suffix = 2
        while view_name in used_names:
            view_name = f"{base_name}__{suffix}"
            suffix += 1
        used_names.add(view_name)
        view_names[(asset_path, table_name)] = view_name
    return view_names


def _register_sqlite_view(
    conn: duckdb.DuckDBPyConnection,
    file_path: Path,
    sqlite_table: str,
    view_name: str,
) -> None:
    import pandas as pd

    with _connect_sqlite_read_only(file_path) as sqlite_conn:
        dataframe = pd.read_sql_query(
            f"SELECT * FROM {_quote_identifier(sqlite_table)}",
            sqlite_conn,
        )
    temp_name = f"__sqlite_probe_{len(view_name)}_{abs(hash((str(file_path), sqlite_table))) & 0xFFFFFFFF}"
    conn.register(temp_name, dataframe)
    conn.execute(
        f"CREATE VIEW {_quote_identifier(view_name)} AS "
        f"SELECT * FROM {_quote_identifier(temp_name)}"
    )


def _sql_references_view(sql: str, view_name: str) -> bool:
    if _quote_identifier(view_name) in sql:
        return True
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(view_name)}(?![A-Za-z0-9_])"
    return re.search(pattern, sql) is not None


def _create_duckdb_views(
    conn: duckdb.DuckDBPyConnection,
    context_dir: Path,
    catalog: dict[str, Any],
    *,
    sql: str | None = None,
) -> None:
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        if kind not in ("csv", "json"):
            continue
        file_path = context_dir / asset_path
        table_name = _asset_view_name(asset_path)
        quoted_table = _quote_identifier(table_name)
        safe_path = _quote_path(file_path)
        try:
            if kind == "csv":
                conn.execute(
                    f"CREATE VIEW {quoted_table} AS "
                    f"SELECT * FROM read_csv_auto('{safe_path}')"
                )
                continue
            max_object_size = _json_maximum_object_size(file_path)
            if _json_schema_has_records_fields(schema):
                conn.execute(
                    f"CREATE VIEW {quoted_table} AS "
                    f"SELECT r.*, r AS records "
                    f"FROM read_json_auto('{safe_path}', maximum_object_size={max_object_size}), "
                    "UNNEST(records) AS t(r)"
                )
                continue
            conn.execute(
                f"CREATE VIEW {quoted_table} AS "
                f"SELECT * FROM read_json_auto('{safe_path}', maximum_object_size={max_object_size})"
            )
        except Exception:
            continue
    sqlite_view_names = _sqlite_view_names(catalog)
    for schema in catalog.get("schemas", []):
        if schema.get("kind") != "sqlite":
            continue
        asset_path = str(schema.get("asset_path", ""))
        file_path = context_dir / asset_path
        for table in schema.get("tables", []):
            table_name = str(table.get("name", ""))
            view_name = sqlite_view_names.get((asset_path, table_name))
            if not view_name:
                continue
            if sql is None or not _sql_references_view(sql, view_name):
                continue
            try:
                _register_sqlite_view(conn, file_path, table_name, view_name)
            except Exception:
                continue


def _asset_sql_replacements(catalog: dict[str, Any]) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    sqlite_view_names = _sqlite_view_names(catalog)
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        if not asset_path:
            continue
        if kind in ("csv", "json"):
            table_ref = _quote_identifier(_asset_view_name(asset_path))
            if kind == "json" and _json_schema_has_records_fields(schema):
                replacements.append((f"{asset_path}.records", table_ref))
            replacements.append((asset_path, table_ref))
            continue
        if kind == "sqlite":
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                view_name = sqlite_view_names.get((asset_path, table_name))
                if table_name and view_name:
                    replacements.append((f"{asset_path}.{table_name}", _quote_identifier(view_name)))
    return sorted(replacements, key=lambda item: len(item[0]), reverse=True)


def _normalize_probe_sql(catalog: dict[str, Any], sql: str) -> str:
    normalized = sql
    for source, target in _asset_sql_replacements(catalog):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])"
        normalized = re.sub(pattern, target, normalized)
    return normalized


def _column_expression(kind: str, column: str) -> str:
    if kind == "json" and column.startswith("records."):
        return ".".join(_quote_identifier(part) for part in column.split("."))
    return _quote_identifier(column)


def _make_json_safe(value: Any) -> Any:
    """Convert DuckDB-returned types to JSON-serializable values (recursive)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    return value


def execute_probe_query(
    context_dir: Path,
    catalog: dict[str, Any],
    queries: list[str],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Execute multiple read-only SQL queries against task data files.

    Creates in-memory DuckDB views for CSV/JSON/SQLite assets from the catalog
    once, then executes each query in *queries*. Only SELECT / WITH statements
    are allowed. Table names must match file-name stems for CSV/JSON assets, or
    SQLite table names when they do not collide with existing view names.

    Returns a list of per-query results under ``results`` so the caller can
    map each query to its output. A single failing query does not abort the
    batch.
    """
    conn = duckdb.connect(":memory:")
    try:
        all_sql = " ".join(queries)
        _create_duckdb_views(conn, context_dir, catalog, sql=all_sql)
        results: list[dict[str, Any]] = []
        for sql in queries:
            try:
                _validate_read_only_sql(sql)
                normalized_sql = _normalize_probe_sql(catalog, sql)
                result = conn.execute(normalized_sql)
                columns = [desc[0] for desc in result.description or []]
                rows = result.fetchmany(limit + 1)
                truncated = len(rows) > limit
                payload: dict[str, Any] = {
                    "ok": True,
                    "columns": columns,
                    "rows": [[_make_json_safe(cell) for cell in row] for row in rows[:limit]],
                    "row_count": len(rows[:limit]),
                    "truncated": truncated,
                }
                if normalized_sql != sql:
                    payload["normalized_sql"] = normalized_sql
                results.append(payload)
            except Exception as exc:
                results.append({"ok": False, "error": str(exc), "sql": sql})
        return {"ok": True, "results": results, "query_count": len(results)}
    finally:
        conn.close()


def get_column_distinct_values(
    context_dir: Path,
    catalog: dict[str, Any],
    table: str,
    column: str,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Return the most frequent distinct values for a column.

    *table* is the file-name stem for CSV/JSON assets, or the SQLite
    table name for ``.db`` assets.
    """
    resolved = _resolve_table(catalog, table)
    if resolved is None:
        return {"ok": False, "error": f"Table '{table}' not found in catalog"}

    asset_path: str = resolved["asset_path"]
    kind: str = resolved["kind"]
    file_path = context_dir / asset_path

    if kind in ("csv", "json"):
        return _distinct_values_duckdb(context_dir, catalog, resolved, column, top_n)
    if kind == "sqlite":
        return _distinct_values_sqlite(file_path, resolved["sqlite_table"], column, top_n)
    return {"ok": False, "error": f"Unsupported asset kind: {kind}"}


def _resolve_table(
    catalog: dict[str, Any], table: str,
) -> dict[str, Any] | None:
    normalized_table = table.strip().strip('"').replace("\\", "/")
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        stem = Path(asset_path).stem

        if kind in ("csv", "json") and normalized_table in {stem, asset_path, f"{asset_path}.records"}:
            return {"asset_path": asset_path, "kind": kind}

        if kind == "sqlite":
            for t in schema.get("tables", []):
                if t.get("name") == normalized_table:
                    return {
                        "asset_path": asset_path,
                        "kind": "sqlite",
                        "sqlite_table": table,
                    }
    return None


def _distinct_values_duckdb(
    context_dir: Path, catalog: dict[str, Any], resolved: dict[str, Any], column: str, top_n: int,
) -> dict[str, Any]:
    conn = duckdb.connect(":memory:")
    try:
        _create_duckdb_views(conn, context_dir, catalog)
        asset_path: str = resolved["asset_path"]
        kind: str = resolved["kind"]
        table_name = _asset_view_name(asset_path)
        quoted_table = _quote_identifier(table_name)
        column_expr = _column_expression(kind, column)
        result = conn.execute(
            f"SELECT {column_expr} AS value, COUNT(*) AS cnt "
            f"FROM {quoted_table} "
            f"GROUP BY {column_expr} "
            f"ORDER BY cnt DESC "
            f"LIMIT ?",
            [top_n],
        )
        values = [{"value": _make_json_safe(row[0]), "count": row[1]} for row in result.fetchall()]
        return {"ok": True, "table": table_name, "column": column, "values": values, "value_count": len(values)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()


def _distinct_values_sqlite(
    file_path: Path, table: str, column: str, top_n: int,
) -> dict[str, Any]:
    try:
        with _connect_sqlite_read_only(file_path) as conn:
            quoted_table = _quote_identifier(table)
            quoted_col = _quote_identifier(column)
            rows = conn.execute(
                f"SELECT {quoted_col} AS value, COUNT(*) AS cnt "
                f"FROM {quoted_table} "
                f"GROUP BY {quoted_col} "
                f"ORDER BY cnt DESC "
                f"LIMIT ?",
                (top_n,),
            ).fetchall()
            values = [{"value": _make_json_safe(row[0]), "count": row[1]} for row in rows]
            return {"ok": True, "table": table, "column": column, "values": values, "value_count": len(values)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
