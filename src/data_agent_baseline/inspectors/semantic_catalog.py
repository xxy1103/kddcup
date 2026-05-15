from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorSampleBudget
from data_agent_baseline.token_utils import count_tokens, truncate_by_tokens


TEXT_SUFFIXES = {".md", ".txt", ".rst"}
CSV_SUFFIXES = {".csv"}
JSON_SUFFIXES = {".json"}
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
STRUCTURAL_KINDS = {"csv", "json", "sqlite"}

MAX_RELATION_DISTINCT_VALUES = 200_000
MIN_INFERRED_SOURCE_NON_NULL = 3
MIN_INFERRED_SOURCE_DISTINCT = 2
MIN_INFERRED_MATCH_RATIO = 0.95
MIN_INFERRED_CONFIDENCE = 0.80

_ROLE_TOKENS = {
    "owner",
    "author",
    "creator",
    "created",
    "editor",
    "last",
    "parent",
    "child",
    "source",
    "target",
    "from",
    "to",
    "by",
}
_ID_TOKENS = {"id", "key", "code"}
_METRIC_TOKENS = {
    "age",
    "amount",
    "average",
    "avg",
    "body",
    "count",
    "date",
    "description",
    "downvotes",
    "favorite",
    "favorites",
    "name",
    "number",
    "price",
    "quantity",
    "score",
    "sum",
    "text",
    "time",
    "timestamp",
    "title",
    "total",
    "upvotes",
    "value",
    "view",
    "views",
}


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
    if kind in ("csv", "json"):
        return ["execute_python", "execute_probe_query"]
    if kind == "sqlite":
        return ["execute_context_sql", "execute_probe_query"]
    if kind == "document":
        return ["read_doc"]
    return ["list_context"]


def _is_integer_str(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


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


def _singularize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 2 and token.endswith("s"):
        return token[:-1]
    return token


def _normalized_tokens(name: str | None) -> set[str]:
    return {_singularize_token(token) for token in _field_tokens(name or "")}


def _read_csv_schema(path: Path, rel_path: str, budget: DataInspectorSampleBudget) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        freq_counters: dict[str, Counter[str]] = {column: Counter() for column in header}
        missing_counts: dict[str, int] = {column: 0 for column in header}
        numeric_min: dict[str, float] = {}
        numeric_max: dict[str, float] = {}
        all_numeric: dict[str, bool] = {column: True for column in header}
        all_integer: dict[str, bool] = {column: True for column in header}
        top_n = budget.catalog_top_distinct_values
        row_count = 0
        for row in reader:
            row_count += 1
            for index, column in enumerate(header):
                value = row[index] if index < len(row) else ""
                if value == "":
                    missing_counts[column] += 1
                else:
                    freq_counters[column][value] += 1
                    if all_numeric[column]:
                        try:
                            num = float(value)
                        except ValueError:
                            all_numeric[column] = False
                            all_integer[column] = False
                        else:
                            if all_integer[column]:
                                if _is_integer_str(value):
                                    pass
                                else:
                                    all_integer[column] = False
                            if column not in numeric_min or num < numeric_min[column]:
                                numeric_min[column] = num
                            if column not in numeric_max or num > numeric_max[column]:
                                numeric_max[column] = num

    fields: list[dict[str, Any]] = []
    for column in header:
        counter = freq_counters[column]
        cardinality = len(counter)
        top_values = [value for value, _ in counter.most_common(top_n)]
        if cardinality == 0:
            col_type = "unknown"
        elif all_integer[column]:
            col_type = "integer"
        elif all_numeric[column]:
            col_type = "number"
        else:
            col_type = "string"
        field: dict[str, Any] = {
            "name": column,
            "type": col_type,
            "missing_count": missing_counts[column],
            "cardinality": cardinality,
            "distinct_values": top_values,
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
    row_count, json_structure = _json_row_count(payload)

    prefix = ""
    if json_structure == "object_with_records":
        prefix = "records."
    elif json_structure.startswith("object_with_array["):
        key = json_structure[len("object_with_array["):-1]
        prefix = f"{key}."

    top_n = budget.catalog_top_distinct_values
    fields: list[dict[str, Any]] = []
    for field, values in sorted(flattened.items()):
        freq_counter: Counter[str] = Counter()
        numeric_min: float | None = None
        numeric_max: float | None = None
        all_numeric = True
        for value in values:
            if value is not None and str(value) != "":
                freq_counter[str(value)] += 1
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
        cardinality = len(freq_counter)
        top_values = [value for value, _ in freq_counter.most_common(top_n)]
        short_name = field
        if prefix and field.startswith(prefix):
            short_name = field[len(prefix):]
        field_dict: dict[str, Any] = {
            "name": short_name,
            "json_path": field,
            "type": _guess_type(values),
            "missing_count": None,
            "cardinality": cardinality,
            "distinct_values": top_values,
        }
        if all_numeric and numeric_min is not None:
            field_dict["min_value"] = numeric_min
            field_dict["max_value"] = numeric_max
        fields.append(field_dict)
    return {
        "asset_path": rel_path,
        "kind": "json",
        "row_count": row_count,
        "json_structure": json_structure,
        "fields": fields,
    }


def _json_row_count(payload: Any) -> tuple[int | None, str]:
    if isinstance(payload, list):
        return len(payload), "list_of_objects"
    if isinstance(payload, dict):
        array_keys: dict[str, int] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                array_keys[key] = len(value)
        if len(array_keys) == 1:
            key, count = next(iter(array_keys.items()))
            if key == "records":
                return count, "object_with_records"
            return count, f"object_with_array[{key}]"
        if len(array_keys) > 1:
            return None, f"object_with_arrays[{','.join(array_keys)}]"
        return 1, "object"
    return None, "unknown"


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
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, dict[str, float]], list[str]]:
    distinct_values: dict[str, list[str]] = {column: [] for column in column_names}
    cardinalities: dict[str, int] = {column: 0 for column in column_names}
    min_max: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    top_n = budget.catalog_top_distinct_values
    if not column_names:
        return distinct_values, cardinalities, min_max, warnings

    for column in column_names:
        quoted_col = _quote_sqlite_identifier(column)
        quoted_table = _quote_sqlite_identifier(table_name)
        try:
            freq_rows = conn.execute(
                f"SELECT {quoted_col}, COUNT(*) AS cnt FROM {quoted_table} GROUP BY {quoted_col} ORDER BY cnt DESC LIMIT ?",
                (top_n,),
            ).fetchall()
        except sqlite3.Error:
            continue
        try:
            cardinality_row = conn.execute(
                f"SELECT COUNT(DISTINCT {quoted_col}) FROM {quoted_table}"
            ).fetchone()
            cardinalities[column] = int(cardinality_row[0]) if cardinality_row else 0
        except sqlite3.Error:
            cardinalities[column] = 0
        distinct_values[column] = [
            _stringify_sqlite_value(row[0]) for row in freq_rows
            if _stringify_sqlite_value(row[0])
        ]
        col_type = _guess_type(distinct_values[column])
        if col_type in ("integer", "number"):
            try:
                min_max_rows = conn.execute(
                    f"SELECT MIN(CAST({_quote_sqlite_identifier(column)} AS REAL)), MAX(CAST({_quote_sqlite_identifier(column)} AS REAL)) FROM {_quote_sqlite_identifier(table_name)} WHERE {_quote_sqlite_identifier(column)} IS NOT NULL AND {_quote_sqlite_identifier(column)} != ''"
                ).fetchone()
                if min_max_rows and min_max_rows[0] is not None:
                    min_max[column] = {"min_value": float(min_max_rows[0]), "max_value": float(min_max_rows[1])}
            except sqlite3.Error:
                pass

    return distinct_values, cardinalities, min_max, warnings


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
                        "scan_warnings": [f"Could not inspect table `{table_name}`: {exc}"],
                    }
                )
                continue

            foreign_keys: list[dict[str, Any]] = []
            try:
                fk_rows = conn.execute(
                    f"PRAGMA foreign_key_list({_quote_sqlite_identifier(table_name)})"
                ).fetchall()
                grouped_fks: dict[int, dict[str, Any]] = {}
                for row in fk_rows:
                    fk_id = int(row[0])
                    entry = grouped_fks.setdefault(
                        fk_id,
                        {
                            "columns": [],
                            "target_table": row[2],
                            "target_columns": [],
                        },
                    )
                    entry["columns"].append(row[3])
                    entry["target_columns"].append(row[4])
                foreign_keys = list(grouped_fks.values())
            except sqlite3.Error:
                foreign_keys = []

            column_names = [str(row[1]) for row in columns]
            distinct_values, cardinalities, min_max, sample_warnings = _read_sqlite_table_samples(
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
                            "missing_count": None,
                            "cardinality": cardinalities.get(str(row[1])),
                            "distinct_values": distinct_values.get(str(row[1]), []),
                            "primary_key": bool(row[5]),
                            **min_max.get(str(row[1]), {}),
                        }
                        for row in columns
                    ],
                    **({"row_count": table_row_count} if table_row_count is not None else {}),
                    **({"foreign_keys": foreign_keys} if foreign_keys else {}),
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
    headings: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and stripped.lstrip("#").strip():
            level = len(line) - len(line.lstrip("#"))
            headings.append({"level": level, "text": stripped.lstrip("#").strip()})
    headings = headings[:20]
    token_count = count_tokens(text)
    is_truncated = token_count > budget.max_doc_tokens
    result: dict[str, Any] = {
        "asset_path": rel_path,
        "kind": "document",
        "token_count": token_count,
        "headings": headings,
        "preview": truncate_by_tokens(text, budget.max_doc_tokens) if is_truncated else text,
        "truncated": is_truncated,
    }
    if path.name.lower() == "knowledge.md":
        result["content"] = text
        del result["preview"]
    return result


@dataclass(frozen=True, slots=True)
class FieldRef:
    asset_path: str
    kind: str
    table: str | None
    field: str
    field_type: str
    cardinality: int | None
    row_count: int | None
    is_primary_key: bool = False


@dataclass(slots=True)
class ValueProfile:
    non_null_count: int
    value_counts: Counter[str]
    capped: bool = False

    @property
    def distinct_count(self) -> int:
        return len(self.value_counts)


def _iter_schema_fields(schemas: list[dict[str, Any]]) -> list[FieldRef]:
    refs: list[FieldRef] = []
    for schema in schemas:
        kind = str(schema.get("kind", ""))
        asset_path = str(schema.get("asset_path", ""))
        if kind == "sqlite":
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                row_count = table.get("row_count")
                for field in table.get("fields", []):
                    refs.append(
                        FieldRef(
                            asset_path=asset_path,
                            kind=kind,
                            table=table_name,
                            field=str(field.get("name", "")),
                            field_type=str(field.get("type", "unknown")),
                            cardinality=_optional_int(field.get("cardinality")),
                            row_count=_optional_int(row_count),
                            is_primary_key=bool(field.get("primary_key")),
                        )
                    )
        elif kind in {"csv", "json"}:
            row_count = schema.get("row_count")
            for field in schema.get("fields", []):
                refs.append(
                    FieldRef(
                        asset_path=asset_path,
                        kind=kind,
                        table=None,
                        field=str(field.get("json_path", field.get("name", ""))),
                        field_type=str(field.get("type", "unknown")),
                        cardinality=_optional_int(field.get("cardinality")),
                        row_count=_optional_int(row_count),
                    )
                )
    return refs


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _entity_tokens(ref: FieldRef) -> set[str]:
    tokens = set()
    if ref.table:
        tokens |= _normalized_tokens(ref.table)
    stem = Path(ref.asset_path).stem
    tokens |= _normalized_tokens(stem)
    return tokens


def _field_name_tokens(ref: FieldRef) -> set[str]:
    return _normalized_tokens(ref.field.split(".")[-1])


def _reference_tokens(ref: FieldRef) -> set[str]:
    tokens = _field_name_tokens(ref)
    return {token for token in tokens if token not in _ID_TOKENS and token not in _ROLE_TOKENS}


def _field_type_family(field_type: str) -> str:
    normalized = field_type.lower()
    if "int" in normalized:
        return "integer"
    if normalized in {"integer", "number"} or any(token in normalized for token in ("real", "float", "double", "numeric", "decimal")):
        return "number"
    if any(token in normalized for token in ("char", "clob", "text", "string", "varchar")):
        return "string"
    return "unknown"


def _types_compatible(source: FieldRef, target: FieldRef) -> bool:
    source_family = _field_type_family(source.field_type)
    target_family = _field_type_family(target.field_type)
    if source_family == "unknown" or target_family == "unknown":
        return True
    if source_family == target_family:
        return True
    return {source_family, target_family} <= {"integer", "string"}


def _looks_like_source_key(ref: FieldRef) -> bool:
    tokens = _field_name_tokens(ref)
    if not (tokens & _ID_TOKENS):
        return False
    if tokens <= {"id"}:
        return False
    if tokens & _METRIC_TOKENS:
        return False
    return True


def _looks_like_target_key(ref: FieldRef) -> bool:
    tokens = _field_name_tokens(ref)
    if ref.is_primary_key:
        return True
    if tokens in ({"id"}, {"key"}, {"code"}):
        return True
    if tokens & _METRIC_TOKENS:
        return False
    if tokens & _ID_TOKENS and ref.cardinality is not None and ref.row_count:
        return ref.cardinality / max(ref.row_count, 1) >= 0.80
    return False


def _relationship_name_score(source: FieldRef, target: FieldRef) -> tuple[float, str | None]:
    source_tokens = _field_name_tokens(source)
    source_refs = _reference_tokens(source)
    target_entity = _entity_tokens(target)
    target_field = _field_name_tokens(target)

    if "parent" in source_tokens and source.asset_path == target.asset_path and source.table == target.table:
        if target_field <= {"id"} or target.is_primary_key:
            return 0.90, "parent self-reference field targets the same entity key"

    if source_refs and source_refs <= (target_entity | target_field):
        return 0.86, "source key tokens match target entity/key tokens"

    if source.field.lower() == target.field.lower() and _looks_like_target_key(target):
        return 0.78, "source and target key fields share the same name"

    return 0.0, None


def _relationship_type(source: FieldRef, target: FieldRef) -> str:
    if source.asset_path == target.asset_path and source.table == target.table:
        return "self_reference"
    if "code" in _field_name_tokens(source) or "code" in _field_name_tokens(target):
        return "lookup_code"
    if source.field.lower() == target.field.lower():
        return "same_key"
    return "foreign_key"


def _cardinality(source: FieldRef, target: FieldRef, target_profile: ValueProfile) -> str:
    if target.row_count and target_profile.distinct_count >= int(target.row_count * 0.98):
        if source.cardinality == target.cardinality and source.row_count == target.row_count:
            return "one_to_one"
        return "many_to_one"
    return "unknown"


def _schema_by_asset(schemas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(schema.get("asset_path")): schema for schema in schemas}


def _profile_csv_field(path: Path, field: str) -> ValueProfile:
    counts: Counter[str] = Counter()
    non_null_count = 0
    capped = False
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = _normalize_relation_value(row.get(field))
            if value == "":
                continue
            non_null_count += 1
            if value in counts or len(counts) < MAX_RELATION_DISTINCT_VALUES:
                counts[value] += 1
            else:
                capped = True
    return ValueProfile(non_null_count=non_null_count, value_counts=counts, capped=capped)


def _collect_json_path_values(value: Any, parts: list[str], out: list[Any]) -> None:
    if len(out) >= MAX_RELATION_DISTINCT_VALUES:
        return
    if isinstance(value, list):
        for item in value:
            _collect_json_path_values(item, parts, out)
            if len(out) >= MAX_RELATION_DISTINCT_VALUES:
                return
        return
    if not parts:
        out.append(value)
        return
    if isinstance(value, dict) and parts[0] in value:
        _collect_json_path_values(value[parts[0]], parts[1:], out)


def _profile_json_field(path: Path, field: str) -> ValueProfile:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    values: list[Any] = []
    _collect_json_path_values(payload, field.split("."), values)
    counts: Counter[str] = Counter()
    non_null_count = 0
    for raw_value in values:
        value = _normalize_relation_value(raw_value)
        if value == "":
            continue
        non_null_count += 1
        counts[value] += 1
    return ValueProfile(
        non_null_count=non_null_count,
        value_counts=counts,
        capped=len(values) >= MAX_RELATION_DISTINCT_VALUES,
    )


def _profile_sqlite_field(path: Path, table: str, field: str) -> ValueProfile:
    quoted_table = _quote_sqlite_identifier(table)
    quoted_field = _quote_sqlite_identifier(field)
    counts: Counter[str] = Counter()
    capped = False
    with _connect_read_only(path) as conn:
        count_row = conn.execute(
            f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_field} IS NOT NULL AND {quoted_field} != ''"
        ).fetchone()
        non_null_count = int(count_row[0]) if count_row else 0
        rows = conn.execute(
            f"SELECT {quoted_field}, COUNT(*) AS cnt FROM {quoted_table} "
            f"WHERE {quoted_field} IS NOT NULL AND {quoted_field} != '' "
            f"GROUP BY {quoted_field} LIMIT ?",
            (MAX_RELATION_DISTINCT_VALUES + 1,),
        ).fetchall()
    for value, count in rows[:MAX_RELATION_DISTINCT_VALUES]:
        normalized = _normalize_relation_value(value)
        if normalized:
            counts[normalized] = int(count)
    if len(rows) > MAX_RELATION_DISTINCT_VALUES:
        capped = True
    return ValueProfile(non_null_count=non_null_count, value_counts=counts, capped=capped)


def _normalize_relation_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    text = str(value).strip()
    if text.endswith(".0") and _is_integer_str(text[:-2]):
        return text[:-2]
    return text


def _load_value_profile(
    task: PublicTask,
    ref: FieldRef,
    cache: dict[tuple[str, str | None, str], ValueProfile],
) -> ValueProfile:
    key = (ref.asset_path, ref.table, ref.field)
    if key in cache:
        return cache[key]
    path = task.context_dir / ref.asset_path
    if ref.kind == "csv":
        profile = _profile_csv_field(path, ref.field)
    elif ref.kind == "json":
        profile = _profile_json_field(path, ref.field)
    elif ref.kind == "sqlite" and ref.table is not None:
        profile = _profile_sqlite_field(path, ref.table, ref.field)
    else:
        profile = ValueProfile(non_null_count=0, value_counts=Counter())
    cache[key] = profile
    return profile


def _validate_relationship(
    source: FieldRef,
    target: FieldRef,
    *,
    task: PublicTask,
    value_cache: dict[tuple[str, str | None, str], ValueProfile],
    explicit_fk: bool,
    name_score: float,
    name_evidence: str,
) -> dict[str, Any] | None:
    if not _types_compatible(source, target):
        return None
    source_profile = _load_value_profile(task, source, value_cache)
    target_profile = _load_value_profile(task, target, value_cache)
    if not explicit_fk:
        if source_profile.non_null_count < MIN_INFERRED_SOURCE_NON_NULL:
            return None
        if source_profile.distinct_count < MIN_INFERRED_SOURCE_DISTINCT:
            return None
    if not source_profile.value_counts or not target_profile.value_counts:
        return None

    target_values = set(target_profile.value_counts)
    matched_rows = sum(
        count for value, count in source_profile.value_counts.items()
        if value in target_values
    )
    matched_distinct = sum(1 for value in source_profile.value_counts if value in target_values)
    observed_source_rows = sum(source_profile.value_counts.values())
    row_ratio = matched_rows / max(observed_source_rows, 1)
    distinct_ratio = matched_distinct / max(source_profile.distinct_count, 1)
    target_uniqueness_ratio = target_profile.distinct_count / max(target_profile.non_null_count, 1)

    if not explicit_fk and row_ratio < MIN_INFERRED_MATCH_RATIO and distinct_ratio < MIN_INFERRED_MATCH_RATIO:
        return None

    confidence = 0.55
    confidence += min(name_score, 0.30)
    confidence += min(max(row_ratio, distinct_ratio) * 0.18, 0.18)
    confidence += min(target_uniqueness_ratio * 0.08, 0.08)
    if explicit_fk:
        confidence = max(confidence, 0.95)
    if source_profile.capped or target_profile.capped:
        confidence -= 0.05
    confidence = round(min(confidence, 0.99), 3)
    if confidence < MIN_INFERRED_CONFIDENCE and not explicit_fk:
        return None

    return {
        "source": {
            "asset_path": source.asset_path,
            "table": source.table,
            "fields": [source.field],
        },
        "target": {
            "asset_path": target.asset_path,
            "table": target.table,
            "fields": [target.field],
        },
        "relationship_type": "foreign_key" if explicit_fk else _relationship_type(source, target),
        "cardinality": _cardinality(source, target, target_profile),
        "confidence": confidence,
        "evidence": {
            "explicit_sqlite_foreign_key": explicit_fk,
            "name_match": name_evidence,
            "source_non_null_count": source_profile.non_null_count,
            "source_distinct_count": source_profile.distinct_count,
            "target_distinct_count": target_profile.distinct_count,
            "matched_source_row_ratio": round(row_ratio, 4),
            "matched_source_distinct_ratio": round(distinct_ratio, 4),
            "target_uniqueness_ratio": round(target_uniqueness_ratio, 4),
            "value_profile_capped": source_profile.capped or target_profile.capped,
        },
    }


def _explicit_sqlite_fk_candidates(
    schemas: list[dict[str, Any]],
    field_refs: list[FieldRef],
) -> list[tuple[FieldRef, FieldRef]]:
    by_key = {
        (ref.asset_path, ref.table, ref.field): ref
        for ref in field_refs
    }
    candidates: list[tuple[FieldRef, FieldRef]] = []
    for schema in schemas:
        if schema.get("kind") != "sqlite":
            continue
        asset_path = str(schema.get("asset_path"))
        for table in schema.get("tables", []):
            table_name = str(table.get("name", ""))
            for fk in table.get("foreign_keys", []):
                source_columns = [str(col) for col in fk.get("columns", [])]
                target_table = str(fk.get("target_table", ""))
                target_columns = [str(col) for col in fk.get("target_columns", [])]
                if len(source_columns) != 1 or len(target_columns) != 1:
                    continue
                source = by_key.get((asset_path, table_name, source_columns[0]))
                target = by_key.get((asset_path, target_table, target_columns[0]))
                if source and target:
                    candidates.append((source, target))
    return candidates


def infer_schema_relationships(
    task: PublicTask,
    schemas: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    field_refs = _iter_schema_fields(schemas)
    target_refs = [ref for ref in field_refs if _looks_like_target_key(ref)]
    relationships: list[dict[str, Any]] = []
    warnings: list[str] = []
    value_cache: dict[tuple[str, str | None, str], ValueProfile] = {}
    seen: set[tuple[str, str | None, str, str, str | None, str]] = set()

    for source, target in _explicit_sqlite_fk_candidates(schemas, field_refs):
        key = (source.asset_path, source.table, source.field, target.asset_path, target.table, target.field)
        seen.add(key)
        relationship = _validate_relationship(
            source,
            target,
            task=task,
            value_cache=value_cache,
            explicit_fk=True,
            name_score=0.95,
            name_evidence="explicit SQLite foreign key",
        )
        if relationship is not None:
            relationships.append(relationship)

    for source in field_refs:
        if not _looks_like_source_key(source):
            continue
        for target in target_refs:
            if source == target:
                continue
            key = (source.asset_path, source.table, source.field, target.asset_path, target.table, target.field)
            if key in seen:
                continue
            name_score, name_evidence = _relationship_name_score(source, target)
            if name_score <= 0 or name_evidence is None:
                continue
            relationship = _validate_relationship(
                source,
                target,
                task=task,
                value_cache=value_cache,
                explicit_fk=False,
                name_score=name_score,
                name_evidence=name_evidence,
            )
            if relationship is not None:
                relationships.append(relationship)

    relationships.sort(
        key=lambda item: (
            -float(item.get("confidence", 0)),
            str(item["source"]["asset_path"]),
            str(item["source"]["fields"]),
            str(item["target"]["asset_path"]),
        )
    )
    return relationships, warnings


def _score_query_relevance(question: str, assets: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> dict[str, Any]:
    query_tokens = _field_tokens(question)
    relevant_assets: list[dict[str, Any]] = []
    relevant_fields: list[dict[str, Any]] = []

    for asset in assets:
        tokens = _field_tokens(str(asset["asset_path"]))
        score = len(query_tokens & tokens)
        if score:
            relevant_assets.append({"asset_path": asset["asset_path"], "score": score})

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
    max_depth: int | None = None,
    include_relationships: bool = True,
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    uncertainties: list[dict[str, Any]] = []

    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(task.context_dir).as_posix()
        if max_depth is not None and len(Path(rel_path).parts) > max_depth:
            continue
        kind = _asset_kind(path)
        assets.append(
            {
                "asset_path": rel_path,
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

    relationship_warnings: list[str] = []
    relationships: list[dict[str, Any]] = []
    if include_relationships:
        relationships, relationship_warnings = infer_schema_relationships(task, schemas)

    return {
        "task_id": task.task_id,
        "assets": assets,
        "schemas": schemas,
        "semantic_entities": [],
        "field_meanings": [],
        "relationships": relationships,
        "relationship_warnings": relationship_warnings,
        "query_relevance": _score_query_relevance(task.question, assets, schemas),
        "semantic_uncertainties": uncertainties,
    }
