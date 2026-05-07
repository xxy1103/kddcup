from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorSampleBudget


TEXT_SUFFIXES = {".md", ".txt", ".rst"}
CSV_SUFFIXES = {".csv"}
JSON_SUFFIXES = {".json"}
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _asset_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return "csv"
    if suffix in JSON_SUFFIXES:
        return "json"
    if suffix in SQLITE_SUFFIXES:
        return "sqlite"
    if suffix in TEXT_SUFFIXES:
        return "document"
    return "file"


def _recommended_tools(kind: str) -> list[str]:
    if kind == "csv":
        return ["read_csv", "execute_python"]
    if kind == "json":
        return ["read_json", "execute_python"]
    if kind == "sqlite":
        return ["inspect_sqlite_schema", "execute_context_sql"]
    if kind == "document":
        return ["read_doc"]
    return ["list_context"]


def _guess_type(values: list[Any]) -> str:
    non_empty = [value for value in values if value not in (None, "")]
    if not non_empty:
        return "unknown"
    if all(str(value).isdigit() for value in non_empty):
        return "integer"
    try:
        for value in non_empty:
            float(str(value))
    except ValueError:
        return "string"
    return "number"


def _field_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {token for token in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if token}


def _read_csv_schema(path: Path, rel_path: str, budget: DataInspectorSampleBudget) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        sample_rows: list[list[str]] = []
        sample_values: dict[str, list[str]] = {column: [] for column in header}
        distinct_sets: dict[str, set[str]] = {column: set() for column in header}
        missing_counts: dict[str, int] = {column: 0 for column in header}
        numeric_min: dict[str, float] = {}
        numeric_max: dict[str, float] = {}
        all_numeric: dict[str, bool] = {column: True for column in header}
        max_distinct = budget.catalog_max_distinct_values
        row_count = 0
        for row in reader:
            row_count += 1
            if len(sample_rows) < budget.catalog_sample_rows:
                sample_rows.append(row)
            for index, column in enumerate(header):
                value = row[index] if index < len(row) else ""
                if value == "":
                    missing_counts[column] += 1
                else:
                    if len(sample_values[column]) < budget.catalog_sample_rows:
                        sample_values[column].append(value)
                    if len(distinct_sets[column]) <= max_distinct:
                        distinct_sets[column].add(value)
                    if all_numeric[column]:
                        try:
                            num = float(value)
                        except ValueError:
                            all_numeric[column] = False
                        else:
                            if column not in numeric_min or num < numeric_min[column]:
                                numeric_min[column] = num
                            if column not in numeric_max or num > numeric_max[column]:
                                numeric_max[column] = num

    fields: list[dict[str, Any]] = []
    for column in header:
        distinct_count = len(distinct_sets[column])
        overflow = distinct_count > max_distinct
        field: dict[str, Any] = {
            "name": column,
            "type": _guess_type(sample_values[column]),
            "sample_values": sample_values[column],
            "missing_count": missing_counts[column],
            "cardinality": distinct_count if not overflow else None,
            "distinct_values": sorted(distinct_sets[column]) if not overflow else [],
        }
        if all_numeric[column] and column in numeric_min:
            field["min_value"] = numeric_min[column]
            field["max_value"] = numeric_max[column]
        fields.append(field)
    return {
        "asset_path": rel_path,
        "kind": "csv",
        "row_count": row_count,
        "fields": fields,
        "sample_rows": sample_rows,
    }


def _flatten_json_fields(value: Any, prefix: str = "") -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = defaultdict(list)
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            for field, values in _flatten_json_fields(item, child_prefix).items():
                fields[field].extend(values)
    elif isinstance(value, list):
        for item in value[:20]:
            for field, values in _flatten_json_fields(item, prefix).items():
                fields[field].extend(values)
    elif prefix:
        fields[prefix].append(value)
    return dict(fields)


def _read_json_schema(path: Path, rel_path: str, budget: DataInspectorSampleBudget) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    payload = json.loads(text)
    flattened = _flatten_json_fields(payload)
    max_distinct = budget.catalog_max_distinct_values
    fields: list[dict[str, Any]] = []
    for field, values in sorted(flattened.items()):
        distinct_set: set[str] = set()
        numeric_min: float | None = None
        numeric_max: float | None = None
        all_numeric = True
        for value in values:
            if value is not None and str(value) != "":
                if len(distinct_set) <= max_distinct:
                    distinct_set.add(str(value))
                if all_numeric:
                    try:
                        num = float(value)
                    except (ValueError, TypeError):
                        all_numeric = False
                    else:
                        if numeric_min is None or num < numeric_min:
                            numeric_min = num
                        if numeric_max is None or num > numeric_max:
                            numeric_max = num
        distinct_count = len(distinct_set)
        overflow = distinct_count > max_distinct
        field_dict: dict[str, Any] = {
            "name": field,
            "type": _guess_type(values),
            "sample_values": [value for value in values[: budget.catalog_sample_rows]],
            "missing_count": None,
            "cardinality": distinct_count if not overflow else None,
            "distinct_values": sorted(distinct_set) if not overflow else [],
        }
        if all_numeric and numeric_min is not None:
            field_dict["min_value"] = numeric_min
            field_dict["max_value"] = numeric_max
        fields.append(field_dict)
    row_count = len(payload) if isinstance(payload, list) else 1
    return {
        "asset_path": rel_path,
        "kind": "json",
        "row_count": row_count,
        "fields": fields,
        "preview": text[: budget.max_json_chars],
        "truncated": len(text) > budget.max_json_chars,
    }


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _quote_sqlite_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _stringify_sqlite_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _read_sqlite_table_samples(
    conn: sqlite3.Connection,
    table_name: str,
    column_names: list[str],
    budget: DataInspectorSampleBudget,
) -> tuple[list[list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, int | None], dict[str, dict[str, float]], list[str]]:
    sample_values: dict[str, list[str]] = {column: [] for column in column_names}
    distinct_values: dict[str, list[str]] = {column: [] for column in column_names}
    cardinalities: dict[str, int | None] = {column: None for column in column_names}
    min_max: dict[str, dict[str, float]] = {}
    sample_rows: list[list[str]] = []
    warnings: list[str] = []
    limit = max(int(budget.catalog_sample_rows), 0)
    max_distinct = budget.catalog_max_distinct_values
    if not column_names or limit == 0:
        return sample_rows, sample_values, distinct_values, cardinalities, min_max, warnings

    try:
        selected_columns = ", ".join(_quote_sqlite_identifier(column) for column in column_names)
        rows = conn.execute(
            f"SELECT {selected_columns} FROM {_quote_sqlite_identifier(table_name)} LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.Error as exc:
        warnings.append(f"Could not sample table `{table_name}`: {exc}")
        return sample_rows, sample_values, distinct_values, cardinalities, min_max, warnings

    for row in rows:
        rendered_row = [_stringify_sqlite_value(value) for value in row]
        sample_rows.append(rendered_row)
        for index, column in enumerate(column_names):
            value = rendered_row[index] if index < len(rendered_row) else ""
            if value and len(sample_values[column]) < limit:
                sample_values[column].append(value)

    for column in column_names:
        try:
            distinct_rows = conn.execute(
                f"SELECT DISTINCT {_quote_sqlite_identifier(column)} FROM {_quote_sqlite_identifier(table_name)} LIMIT ?",
                (max_distinct + 1,),
            ).fetchall()
        except sqlite3.Error:
            continue
        distinct_count = len(distinct_rows)
        overflow = distinct_count > max_distinct
        cardinalities[column] = distinct_count if not overflow else None
        distinct_values[column] = sorted(
            _stringify_sqlite_value(row[0]) for row in distinct_rows[:max_distinct]
            if _stringify_sqlite_value(row[0])
        )
        col_type = _guess_type(sample_values[column])
        if col_type in ("integer", "number"):
            try:
                min_max_rows = conn.execute(
                    f"SELECT MIN(CAST({_quote_sqlite_identifier(column)} AS REAL)), MAX(CAST({_quote_sqlite_identifier(column)} AS REAL)) FROM {_quote_sqlite_identifier(table_name)} WHERE {_quote_sqlite_identifier(column)} IS NOT NULL AND {_quote_sqlite_identifier(column)} != ''"
                ).fetchone()
                if min_max_rows and min_max_rows[0] is not None:
                    min_max[column] = {"min_value": float(min_max_rows[0]), "max_value": float(min_max_rows[1])}
            except sqlite3.Error:
                pass

    return sample_rows, sample_values, distinct_values, cardinalities, min_max, warnings


def _read_sqlite_schema(path: Path, rel_path: str, budget: DataInspectorSampleBudget) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    with _connect_read_only(path) as conn:
        table_rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for (table_name,) in table_rows:
            table_warnings: list[str] = []
            try:
                columns = conn.execute(
                    f"PRAGMA table_info({_quote_sqlite_identifier(table_name)})"
                ).fetchall()
            except sqlite3.Error as exc:
                tables.append(
                    {
                        "name": table_name,
                        "fields": [],
                        "sample_rows": [],
                        "scan_warnings": [f"Could not inspect table `{table_name}`: {exc}"],
                    }
                )
                continue

            column_names = [str(row[1]) for row in columns]
            sample_rows, sample_values, distinct_values, cardinalities, min_max, sample_warnings = _read_sqlite_table_samples(
                conn,
                str(table_name),
                column_names,
                budget,
            )
            table_warnings.extend(sample_warnings)
            table_row_count: int | None = None
            try:
                count_row = conn.execute(
                    f"SELECT COUNT(*) FROM {_quote_sqlite_identifier(str(table_name))}"
                ).fetchone()
                if count_row is not None:
                    table_row_count = int(count_row[0])
            except sqlite3.Error:
                pass
            tables.append(
                {
                    "name": table_name,
                    "fields": [
                        {
                            "name": row[1],
                            "type": row[2] or "unknown",
                            "sample_values": sample_values.get(str(row[1]), []),
                            "missing_count": None,
                            "cardinality": cardinalities.get(str(row[1])),
                            "distinct_values": distinct_values.get(str(row[1]), []),
                            **min_max.get(str(row[1]), {}),
                        }
                        for row in columns
                    ],
                    "sample_rows": sample_rows,
                    **({"row_count": table_row_count} if table_row_count is not None else {}),
                    **({"scan_warnings": table_warnings} if table_warnings else {}),
                }
            )
    return {
        "asset_path": rel_path,
        "kind": "sqlite",
        "tables": tables,
    }


def _read_document_schema(path: Path, rel_path: str, budget: DataInspectorSampleBudget) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    headings = [
        line.lstrip("#").strip()
        for line in text.splitlines()
        if line.lstrip().startswith("#") and line.lstrip("#").strip()
    ]
    return {
        "asset_path": rel_path,
        "kind": "document",
        "char_count": len(text),
        "headings": headings[:20],
        "content": text,
        "preview": text[: budget.max_doc_chars],
        "truncated": len(text) > budget.max_doc_chars,
    }


def _build_relationships(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field_locations: dict[str, list[dict[str, str]]] = defaultdict(list)
    for schema in schemas:
        asset_path = str(schema.get("asset_path"))
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                for field in table.get("fields", []):
                    field_locations[_relationship_field_key(str(field["name"]))].append(
                        {"asset_path": asset_path, "table": table["name"], "field": field["name"]}
                    )
            continue
        for field in schema.get("fields", []):
            field_locations[_relationship_field_key(str(field["name"]))].append(
                {"asset_path": asset_path, "field": field["name"]}
            )

    relationships: list[dict[str, Any]] = []
    for field_name, locations in sorted(field_locations.items()):
        if len(locations) > 1 and ("id" in field_name or field_name.endswith("_id")):
            relationships.append(
                {
                    "kind": "join_candidate",
                    "field": field_name,
                    "locations": locations,
                    "confidence": "medium",
                }
            )
    return relationships


def _relationship_field_key(field_name: str) -> str:
    basename = field_name.rsplit(".", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", basename.lower())


def _score_query_relevance(question: str, assets: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> dict[str, Any]:
    query_tokens = _field_tokens(question)
    relevant_assets: list[dict[str, Any]] = []
    relevant_fields: list[dict[str, Any]] = []

    for asset in assets:
        tokens = _field_tokens(str(asset["path"]))
        score = len(query_tokens & tokens)
        if score:
            relevant_assets.append({"path": asset["path"], "score": score})

    for schema in schemas:
        asset_path = str(schema.get("asset_path"))
        field_groups = []
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                field_groups.extend((table.get("name"), field) for field in table.get("fields", []))
        else:
            field_groups.extend((None, field) for field in schema.get("fields", []))

        for table_name, field in field_groups:
            tokens = _field_tokens(str(field.get("name", "")))
            score = len(query_tokens & tokens)
            if score:
                relevant_fields.append(
                    {
                        "asset_path": asset_path,
                        "table": table_name,
                        "field": field.get("name"),
                        "score": score,
                    }
                )

    return {
        "query_tokens": sorted(query_tokens),
        "relevant_assets": sorted(relevant_assets, key=lambda item: item["score"], reverse=True)[:10],
        "relevant_fields": sorted(relevant_fields, key=lambda item: item["score"], reverse=True)[:20],
    }


def build_semantic_catalog(
    task: PublicTask,
    *,
    budget: DataInspectorSampleBudget,
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    uncertainties: list[dict[str, Any]] = []

    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(task.context_dir).as_posix()
        kind = _asset_kind(path)
        assets.append(
            {
                "path": rel_path,
                "kind": kind,
                "size": path.stat().st_size,
                "recommended_tools": _recommended_tools(kind),
            }
        )
        try:
            if kind == "csv":
                schemas.append(_read_csv_schema(path, rel_path, budget))
            elif kind == "json":
                schemas.append(_read_json_schema(path, rel_path, budget))
            elif kind == "sqlite":
                schemas.append(_read_sqlite_schema(path, rel_path, budget))
            elif kind == "document":
                schemas.append(_read_document_schema(path, rel_path, budget))
        except Exception as exc:  # noqa: BLE001
            uncertainties.append(
                {
                    "risk": "asset_scan_failed",
                    "asset_path": rel_path,
                    "instruction": f"Scanner could not parse this asset: {exc}",
                }
            )

    relationships = _build_relationships(schemas)
    return {
        "task_id": task.task_id,
        "assets": assets,
        "schemas": schemas,
        "semantic_entities": [],
        "field_meanings": [],
        "relationships": relationships,
        "query_relevance": _score_query_relevance(task.question, assets, schemas),
        "semantic_uncertainties": uncertainties,
    }
