from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.tools.sqlite import execute_read_only_sql


TEXT_TABLE_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    block_id: int
    text: str
    start_line: int
    end_line: int
    heading: str | None = None
    section: str | None = None
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class FieldEvent:
    table: str
    key: tuple[Any, ...]
    field: str
    value: Any
    source_block: int
    confidence: float
    marker: str = "observed"


@dataclass(slots=True)
class ParagraphLedgerEntry:
    block_id: int
    path: str
    start_line: int
    end_line: int
    status: str
    reason: str
    heading: str | None
    text: str


@dataclass(slots=True)
class ExtractionTable:
    name: str
    primary_key: list[str]
    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[ParagraphLedgerEntry] = field(default_factory=list)


@dataclass(slots=True)
class MemAgentStoreInfo:
    store_id: str
    sqlite_path: str
    tables: dict[str, dict[str, Any]]


@dataclass(slots=True)
class ExtractionResult:
    store_id: str
    sqlite_path: Path
    tables: list[ExtractionTable]
    diagnostics: dict[str, Any]


RuleProvider = Callable[[dict[str, Any]], dict[str, Any] | None]


CORRECTION_MARKERS = {
    "corrected": 4,
    "verified": 3,
    "final": 3,
    "confirmed": 3,
    "rectified": 4,
    "updated": 2,
    "amended": 2,
    "initial": 0,
    "initially": 0,
    "preliminary": 0,
    "observed": 1,
}


def default_store_id(task_id: str, paths: list[str]) -> str:
    stems = "_".join(Path(p).stem for p in paths) or "documents"
    raw = f"{task_id}_{stems}"
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    digest = hashlib.sha1("|".join(paths).encode("utf-8")).hexdigest()[:8]
    return f"{safe[:80]}_{digest}"


def parse_markdown_blocks(text: str) -> list[MarkdownBlock]:
    blocks: list[MarkdownBlock] = []
    heading_stack: list[tuple[int, str]] = []
    current: list[str] = []
    start_line = 1
    line_no = 0

    def current_heading() -> str | None:
        return heading_stack[-1][1] if heading_stack else None

    def current_section() -> str | None:
        return heading_stack[0][1] if heading_stack else None

    def current_phase() -> str | None:
        for _, title in reversed(heading_stack):
            if re.search(r"\b(initial|preliminary|corrected|verified|final|follow[- ]?up)\b", title, re.I):
                return title
        return None

    def flush(end_line: int) -> None:
        nonlocal current, start_line
        block_text = "\n".join(current).strip()
        if block_text:
            blocks.append(
                MarkdownBlock(
                    block_id=len(blocks),
                    text=block_text,
                    start_line=start_line,
                    end_line=end_line,
                    heading=current_heading(),
                    section=current_section(),
                    phase=current_phase(),
                )
            )
        current = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush(line_no - 1)
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [(lvl, h) for lvl, h in heading_stack if lvl < level]
            heading_stack.append((level, title))
            start_line = line_no + 1
            continue

        if not line.strip():
            flush(line_no - 1)
            start_line = line_no + 1
            continue

        if not current:
            start_line = line_no
        current.append(line)

    flush(line_no)
    return blocks


def extract_tables_from_documents(
    *,
    sources: list[tuple[str, Path]],
    workspace_root: Path,
    task_id: str,
    store_id: str | None = None,
    goal: str = "",
    rule_provider: RuleProvider | None = None,
    repair_rounds: int = 2,
) -> ExtractionResult:
    if not sources:
        raise ValueError("memagent requires at least one document path.")
    rel_paths = [rel for rel, _ in sources]
    resolved_store_id = store_id or default_store_id(task_id, rel_paths)
    store_dir = workspace_root / ".memagent" / resolved_store_id
    store_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = store_dir / "extracted.sqlite"

    tables_by_name: dict[str, ExtractionTable] = {}
    diagnostics: dict[str, Any] = {
        "store_id": resolved_store_id,
        "source_paths": rel_paths,
        "block_count": 0,
        "rules": [],
        "repair_rounds_used": 0,
    }

    for rel_path, full_path in sources:
        if full_path.suffix.lower() not in TEXT_TABLE_SUFFIXES:
            raise ValueError(f"Unsupported document type for memagent ETL: {rel_path}")
        text = full_path.read_text(encoding="utf-8", errors="replace")
        blocks = parse_markdown_blocks(text)
        diagnostics["block_count"] += len(blocks)
        schema = _schema_for_stem(Path(rel_path).stem)
        rule_pack = _build_rule_pack(
            schema=schema,
            rel_path=rel_path,
            blocks=blocks,
            goal=goal,
            rule_provider=rule_provider,
        )
        doc_table = _extract_with_rule_pack(rel_path, blocks, rule_pack)
        repair_used = 0
        for round_idx in range(repair_rounds):
            residual = [
                entry
                for entry in doc_table.unresolved
                if entry.status in {"partial", "conflict", "unmatched"}
            ]
            if not residual or rule_provider is None:
                break
            patch = rule_provider(
                {
                    "kind": "repair",
                    "goal": goal,
                    "schema": schema,
                    "current_rule_pack": rule_pack,
                    "diagnostics": _table_diagnostics(doc_table),
                    "residual_blocks": [asdict(entry) for entry in residual[:20]],
                }
            )
            if not patch:
                break
            rule_pack = _merge_rule_pack(rule_pack, patch)
            repaired = _extract_with_rule_pack(rel_path, blocks, rule_pack)
            if _extracted_cell_count(repaired) <= _extracted_cell_count(doc_table):
                break
            doc_table = repaired
            repair_used = round_idx + 1
        diagnostics["repair_rounds_used"] = max(
            int(diagnostics["repair_rounds_used"]),
            repair_used,
        )
        diagnostics["rules"].append(
            {
                "path": rel_path,
                "table": rule_pack["table"],
                "source": rule_pack.get("source", "unknown"),
                "field_count": len(rule_pack.get("field_patterns", {})),
                "key_pattern_count": len(rule_pack.get("key_patterns", [])),
            }
        )
        existing = tables_by_name.get(doc_table.name)
        if existing is None:
            tables_by_name[doc_table.name] = doc_table
        else:
            existing.rows.extend(doc_table.rows)
            existing.evidence.extend(doc_table.evidence)
            existing.unresolved.extend(doc_table.unresolved)

    tables = list(tables_by_name.values())
    _write_sqlite(sqlite_path, tables, diagnostics)
    return ExtractionResult(
        store_id=resolved_store_id,
        sqlite_path=sqlite_path,
        tables=tables,
        diagnostics=diagnostics,
    )


def list_store_tables(sqlite_path: Path, *, sample_limit: int = 3) -> dict[str, Any]:
    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if not str(row[0]).startswith("sqlite_")
        ]
        tables = []
        for name in table_names:
            cols = [dict(row) for row in conn.execute(f'PRAGMA table_info("{name}")')]
            count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            sample_rows = [
                dict(row)
                for row in conn.execute(f'SELECT * FROM "{name}" LIMIT ?', (sample_limit,))
            ]
            tables.append(
                {
                    "name": name,
                    "columns": [col["name"] for col in cols],
                    "schema": cols,
                    "row_count": count,
                    "sample_rows": sample_rows,
                }
            )
    return {"path": str(sqlite_path), "tables": tables}


def query_store(sqlite_path: Path, sql: str, *, limit: int = 200) -> dict[str, Any]:
    return execute_read_only_sql(sqlite_path, sql, limit=limit)


def read_unresolved(
    sqlite_path: Path,
    *,
    status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    sql = (
        "SELECT path, block_id, start_line, end_line, status, reason, heading, text "
        "FROM _unresolved_blocks"
    )
    params: tuple[Any, ...] = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY path, block_id LIMIT ?"
    params = (*params, limit)
    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params)]
    return {"path": str(sqlite_path), "status": status, "row_count": len(rows), "rows": rows}


def _schema_for_stem(stem: str) -> dict[str, Any]:
    table_name = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or "Extracted"
    return {
        "table": table_name,
        "columns": ["id"],
        "primary_key": ["id"],
        "kind": "entity",
    }


def _build_rule_pack(
    *,
    schema: dict[str, Any],
    rel_path: str,
    blocks: list[MarkdownBlock],
    goal: str,
    rule_provider: RuleProvider | None,
) -> dict[str, Any]:
    if rule_provider is not None:
        generated = rule_provider(
            {
                "kind": "initial",
                "goal": goal,
                "path": rel_path,
                "schema": schema,
                "sample_blocks": [asdict(block) for block in blocks[:25]],
            }
        )
        if generated:
            return _normalize_rule_pack(schema, generated, source="llm")
    raise ValueError(
        "memagent extract_tables requires model-generated rule JSON; no bootstrap rules are bundled."
    )


def _extract_with_rule_pack(
    rel_path: str,
    blocks: list[MarkdownBlock],
    rule_pack: dict[str, Any],
) -> ExtractionTable:
    table_name = str(rule_pack["table"])
    columns = list(rule_pack["columns"])
    primary_key = list(rule_pack["primary_key"])
    events: list[FieldEvent] = []
    ledger: list[ParagraphLedgerEntry] = []
    for block in blocks:
        key_values = _extract_key_values(block.text, rule_pack)
        fields = _extract_rule_fields(block.text, rule_pack)
        status = "unmatched"
        reason = f"No {table_name} key or recognized fields."
        if key_values and fields:
            status = "extracted"
            reason = f"{table_name} key and fields extracted."
            for key_col, key_value in zip(primary_key, key_values):
                events.append(
                    FieldEvent(
                        table_name,
                        tuple(key_values),
                        key_col,
                        key_value,
                        block.block_id,
                        1.0,
                        _marker(block.text, block.phase),
                    )
                )
            for field, value in fields.items():
                if field not in columns:
                    columns.append(field)
                events.append(
                    FieldEvent(
                        table_name,
                        tuple(key_values),
                        field,
                        value,
                        block.block_id,
                        0.9,
                        _marker(block.text, block.phase),
                    )
                )
        elif key_values:
            status = "partial"
            reason = f"{table_name} key found without recognized fields."
        elif fields:
            status = "partial"
            reason = f"{table_name} fields found without key."
        elif _looks_narrative(block.text):
            status = "ignored_narrative"
            reason = f"Narrative paragraph without extractable {table_name} facts."
        if status != "extracted":
            ledger.append(_ledger(rel_path, block, status, reason))
    rows, evidence = _resolve_events(table_name, primary_key, columns, events)
    return ExtractionTable(table_name, primary_key, columns, rows, evidence, ledger)


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {k: (v if isinstance(v, list) else [v]) for k, v in value.items()}
    if isinstance(value, list):
        result: dict[str, Any] = {}
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                key = str(item[0])
                val = list(item[1:]) if len(item) > 2 else item[1]
                result[key] = val if isinstance(val, list) else [val]
        return result
    return {}


def _normalize_rule_pack(
    schema: dict[str, Any],
    data: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    pack = {
        "table": str(data.get("table") or schema["table"]),
        "columns": list(data.get("columns") or schema["columns"]),
        "primary_key": list(data.get("primary_key") or schema["primary_key"]),
        "key_patterns": list(data.get("key_patterns") or []),
        "field_patterns": _safe_dict(data.get("field_patterns")),
        "field_types": _safe_dict(data.get("field_types")),
        "null_values": list(data.get("null_values") or ["None", "NaN", "-", ""]),
        "paired_fields": [
            item for item in (data.get("paired_fields") or []) if isinstance(item, dict)
        ],
        "source": source,
    }
    if len(pack["primary_key"]) > 1:
        pack["date_patterns"] = list(data.get("date_patterns") or [])
    return pack


def _merge_rule_pack(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for item in patch.get("key_patterns", []):
        if item not in merged["key_patterns"]:
            merged["key_patterns"].append(item)
    for item in patch.get("date_patterns", []):
        merged.setdefault("date_patterns", [])
        if item not in merged["date_patterns"]:
            merged["date_patterns"].append(item)
    for field, patterns in (patch.get("field_patterns") or {}).items():
        current = merged.setdefault("field_patterns", {}).setdefault(field, [])
        for pattern in patterns if isinstance(patterns, list) else [patterns]:
            if pattern not in current:
                current.append(pattern)
    merged.setdefault("field_types", {}).update(patch.get("field_types") or {})
    for item in patch.get("paired_fields", []):
        if item not in merged.setdefault("paired_fields", []):
            merged["paired_fields"].append(item)
    merged["source"] = "llm_repaired"
    return merged


def _extract_key_values(text: str, rule_pack: dict[str, Any]) -> list[Any] | None:
    key_patterns = list(rule_pack.get("key_patterns", []))
    first = _first_capture(key_patterns, text)
    if first is None:
        return None
    key_values = [_normalize_value(first)]
    if len(rule_pack.get("primary_key", [])) > 1:
        date = _first_capture(list(rule_pack.get("date_patterns", [])), text)
        if date is None:
            return None
        key_values.append(_normalize_date(str(date)))
    return key_values


def _extract_rule_fields(text: str, rule_pack: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    null_lower = {str(item).lower() for item in rule_pack.get("null_values", [])}
    for field, patterns in rule_pack.get("field_patterns", {}).items():
        matches: list[tuple[int, int, Any]] = []
        for pattern_index, pattern in enumerate(patterns if isinstance(patterns, list) else [patterns]):
            for match in re.finditer(str(pattern), text, flags=re.I):
                raw = _last_capture(match)
                if raw is None:
                    continue
                value = _normalize_value(raw)
                if value is None and str(raw).strip().lower() in null_lower:
                    value = None
                matches.append((match.start(), -pattern_index, value))
        if matches:
            fields[str(field)] = sorted(matches, key=lambda item: (item[0], item[1]))[-1][2]
    for pair in rule_pack.get("paired_fields", []):
        if not isinstance(pair, dict):
            continue
        pattern = str(pair.get("pattern", ""))
        targets = list(pair.get("fields", []))
        if not pattern or not targets:
            continue
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        value = _normalize_value(_last_capture(match) or "")
        for target in targets:
            fields[str(target)] = value
    for field, type_hint in rule_pack.get("field_types", {}).items():
        hint = type_hint[0] if isinstance(type_hint, list) and type_hint else type_hint
        if field in fields and hint in {"int", "float"}:
            fields[field] = _coerce_number(fields[field])
    return fields


def _first_capture(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return _last_capture(match)
    return None


def _last_capture(match: re.Match[str]) -> str | None:
    if not match.groups():
        return None
    for value in reversed(match.groups()):
        if value is not None:
            return value.strip()
    return None


def _normalize_date(raw: str) -> str:
    match = re.search(r"\b(20\d{2}|19\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
    if not match:
        return raw
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _table_diagnostics(table: ExtractionTable) -> dict[str, Any]:
    return {
        "table": table.name,
        "row_count": len(table.rows),
        "evidence_count": len(table.evidence),
        "unresolved_count": len(table.unresolved),
        "unresolved_by_status": {
            status: sum(1 for entry in table.unresolved if entry.status == status)
            for status in sorted({entry.status for entry in table.unresolved})
        },
    }


def _extracted_cell_count(table: ExtractionTable) -> int:
    return sum(
        1
        for row in table.rows
        for value in row.values()
        if value is not None and value != ""
    )


def _resolve_events(
    table: str,
    key_columns: list[str],
    columns: list[str],
    events: list[FieldEvent],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, list[FieldEvent]]] = {}
    for event in events:
        grouped.setdefault(event.key, {}).setdefault(event.field, []).append(event)

    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for key, fields in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = {column: None for column in columns}
        for idx, column in enumerate(key_columns):
            row[column] = key[idx] if idx < len(key) else None
        for field, field_events in fields.items():
            chosen = max(
                field_events,
                key=lambda ev: (CORRECTION_MARKERS.get(ev.marker, 1), ev.source_block, ev.confidence),
            )
            if field in row:
                row[field] = chosen.value
            conflict = len({json.dumps(ev.value, sort_keys=True) for ev in field_events}) > 1
            evidence.append(
                {
                    "table_name": table,
                    "record_key": json.dumps(list(key), ensure_ascii=False),
                    "field": field,
                    "value": chosen.value,
                    "source_block": chosen.source_block,
                    "confidence": chosen.confidence,
                    "marker": chosen.marker,
                    "event_count": len(field_events),
                    "conflict": int(conflict),
                }
            )
        rows.append(row)
    return rows, evidence


def _write_sqlite(path: Path, tables: list[ExtractionTable], diagnostics: dict[str, Any]) -> None:
    if path.exists():
        path.unlink()
    with closing(sqlite3.connect(path)) as conn:
        for table in tables:
            _create_data_table(conn, table)
        conn.execute(
            "CREATE TABLE _extraction_evidence ("
            "table_name TEXT, record_key TEXT, field TEXT, value TEXT, "
            "source_block INTEGER, confidence REAL, marker TEXT, event_count INTEGER, conflict INTEGER)"
        )
        conn.execute(
            "CREATE TABLE _unresolved_blocks ("
            "path TEXT, block_id INTEGER, start_line INTEGER, end_line INTEGER, "
            "status TEXT, reason TEXT, heading TEXT, text TEXT)"
        )
        conn.execute(
            "CREATE TABLE _extraction_diagnostics (key TEXT PRIMARY KEY, value TEXT)"
        )
        for table in tables:
            conn.executemany(
                "INSERT INTO _extraction_evidence VALUES (:table_name, :record_key, :field, :value, :source_block, :confidence, :marker, :event_count, :conflict)",
                [{**item, "value": _sqlite_value(item.get("value"))} for item in table.evidence],
            )
            conn.executemany(
                "INSERT INTO _unresolved_blocks VALUES (:path, :block_id, :start_line, :end_line, :status, :reason, :heading, :text)",
                [
                    {
                        "path": entry.path,
                        "block_id": entry.block_id,
                        "start_line": entry.start_line,
                        "end_line": entry.end_line,
                        "status": entry.status,
                        "reason": entry.reason,
                        "heading": entry.heading,
                        "text": entry.text,
                    }
                    for entry in table.unresolved
                ],
            )
        diag = {
            **diagnostics,
            "tables": {
                table.name: {
                    "columns": table.columns,
                    "primary_key": table.primary_key,
                    "row_count": len(table.rows),
                    "unresolved_count": len(table.unresolved),
                }
                for table in tables
            },
        }
        for key, value in diag.items():
            conn.execute(
                "INSERT INTO _extraction_diagnostics VALUES (?, ?)",
                (str(key), json.dumps(value, ensure_ascii=False)),
            )
        conn.commit()


def _create_data_table(conn: sqlite3.Connection, table: ExtractionTable) -> None:
    column_defs = []
    for column in table.columns:
        values = [row.get(column) for row in table.rows if row.get(column) is not None]
        sql_type = _infer_sql_type(values)
        column_defs.append(f'"{column}" {sql_type}')
    pk = ", ".join(f'"{col}"' for col in table.primary_key)
    pk_clause = f", PRIMARY KEY ({pk})" if pk else ""
    conn.execute(f'CREATE TABLE "{table.name}" ({", ".join(column_defs)}{pk_clause})')
    if not table.rows:
        return
    placeholders = ", ".join("?" for _ in table.columns)
    quoted_columns = ", ".join(f'"{col}"' for col in table.columns)
    conn.executemany(
        f'INSERT OR REPLACE INTO "{table.name}" ({quoted_columns}) VALUES ({placeholders})',
        [[row.get(col) for col in table.columns] for row in table.rows],
    )


def _infer_sql_type(values: list[Any]) -> str:
    if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "INTEGER"
    if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return "REAL"
    return "TEXT"


def _normalize_value(value: str) -> Any:
    cleaned = value.strip().strip("'\"` ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned.lower() in {"none", "nan", "null", "-"}:
        return None
    return _coerce_number(cleaned)


def _coerce_number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _marker(text: str, phase: str | None) -> str:
    source = f"{phase or ''} {text}".lower()
    for marker in CORRECTION_MARKERS:
        if re.search(rf"\b{re.escape(marker)}\b", source):
            return marker
    return "observed"


def _looks_narrative(text: str) -> bool:
    return len(text.split()) >= 6


def _ledger(rel_path: str, block: MarkdownBlock, status: str, reason: str) -> ParagraphLedgerEntry:
    return ParagraphLedgerEntry(
        block_id=block.block_id,
        path=rel_path,
        start_line=block.start_line,
        end_line=block.end_line,
        status=status,
        reason=reason,
        heading=block.heading,
        text=block.text,
    )


def _sqlite_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def summarize_extraction(result: ExtractionResult) -> dict[str, Any]:
    table_summaries = []
    for table in result.tables:
        table_summaries.append(
            {
                "name": table.name,
                "primary_key": table.primary_key,
                "columns": table.columns,
                "row_count": len(table.rows),
                "sample_rows": table.rows[:3],
                "unresolved_count": len(table.unresolved),
            }
        )
    return {
        "store_id": result.store_id,
        "sqlite_path": str(result.sqlite_path),
        "tables": table_summaries,
        "diagnostics": result.diagnostics,
    }


def store_info_from_result(result: ExtractionResult) -> MemAgentStoreInfo:
    return MemAgentStoreInfo(
        store_id=result.store_id,
        sqlite_path=str(result.sqlite_path),
        tables={
            table.name: {
                "columns": table.columns,
                "primary_key": table.primary_key,
                "row_count": len(table.rows),
                "unresolved_count": len(table.unresolved),
            }
            for table in result.tables
        },
    )


def store_info_to_dict(info: MemAgentStoreInfo) -> dict[str, Any]:
    return asdict(info)
