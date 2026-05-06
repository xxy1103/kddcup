"""Probe engine for executing data-level queries against task data files."""

from __future__ import annotations

import sqlite3
import re
from pathlib import Path
from typing import Any

import duckdb


def _validate_read_only_sql(sql: str) -> None:
    normalized = sql.strip().lstrip("(").lstrip()
    if not normalized.upper().startswith(("SELECT", "WITH")):
        raise ValueError(f"Only SELECT/WITH statements are allowed. Got: {sql[:120]}")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _connect_sqlite_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _asset_view_name(asset_path: str) -> str:
    return Path(asset_path).stem


def _json_schema_has_records_fields(schema: dict[str, Any]) -> bool:
    return any(str(field.get("name", "")).startswith("records.") for field in schema.get("fields", []))


def _create_duckdb_views(conn: duckdb.DuckDBPyConnection, context_dir: Path, catalog: dict[str, Any]) -> None:
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
            if _json_schema_has_records_fields(schema):
                conn.execute(
                    f"CREATE VIEW {quoted_table} AS "
                    f"SELECT r.*, r AS records "
                    f"FROM read_json_auto('{safe_path}'), UNNEST(records) AS t(r)"
                )
                continue
            conn.execute(
                f"CREATE VIEW {quoted_table} AS "
                f"SELECT * FROM read_json_auto('{safe_path}')"
            )
        except Exception:
            continue


def _asset_sql_replacements(catalog: dict[str, Any]) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        if kind not in ("csv", "json") or not asset_path:
            continue
        table_ref = _quote_identifier(_asset_view_name(asset_path))
        if kind == "json" and _json_schema_has_records_fields(schema):
            replacements.append((f"{asset_path}.records", table_ref))
        replacements.append((asset_path, table_ref))
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


def execute_probe_query(
    context_dir: Path,
    catalog: dict[str, Any],
    sql: str,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Execute a read-only SQL query against task data files.

    Creates in-memory DuckDB views for CSV/JSON assets from the catalog,
    then executes *sql*.  Only SELECT / WITH statements are allowed.
    Table names must match file-name stems (e.g. ``drivers`` for
    ``drivers.csv``).
    """
    _validate_read_only_sql(sql)
    normalized_sql = _normalize_probe_sql(catalog, sql)
    conn = duckdb.connect(":memory:")
    try:
        _create_duckdb_views(conn, context_dir, catalog)

        result = conn.execute(normalized_sql)
        columns = [desc[0] for desc in result.description or []]
        rows = result.fetchmany(limit + 1)
        truncated = len(rows) > limit
        payload = {
            "ok": True,
            "columns": columns,
            "rows": [list(row) for row in rows[:limit]],
            "row_count": len(rows[:limit]),
            "truncated": truncated,
        }
        if normalized_sql != sql:
            payload["normalized_sql"] = normalized_sql
        return payload
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
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
        values = [{"value": row[0], "count": row[1]} for row in result.fetchall()]
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
            values = [{"value": row[0], "count": row[1]} for row in rows]
            return {"ok": True, "table": table, "column": column, "values": values, "value_count": len(values)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
