from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.inspectors.semantic_catalog import iter_logical_tables
from data_agent_baseline.model_retry import invoke_model_with_retries
from data_agent_baseline.tools.filesystem import normalize_context_relative_path, resolve_context_path
from data_agent_baseline.tools.python_exec import TaskContextWorkspace


GENERATED_STRUCTURED_DOC_DIR = ".generated/structured_doc"
STRUCTURED_DOC_MANIFEST = "manifest.json"
MAX_MODEL_CALLS = 20


@dataclass(frozen=True, slots=True)
class StructuredDocExtraction:
    columns: list[str]
    rows: list[list[Any]]
    metadata: dict[str, Any]


class StructuredDocExtractionError(RuntimeError):
    def __init__(self, message: str, *, log_summary: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.log_summary = log_summary


class StructuredDocLogger:
    def __init__(
        self,
        workspace_root: Path,
        registered_table: str,
        *,
        log_dir: Path | None = None,
    ) -> None:
        self._started_at = perf_counter()
        self._events: list[dict[str, Any]] = []
        self.relative_log_file: str | None = None
        self.path: Path | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.relative_log_file = f"structured_doc_{registered_table}.log.jsonl"
            self.path = log_dir / self.relative_log_file
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **details: Any) -> None:
        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(perf_counter() - self._started_at, 3),
            "details": details,
        }
        self._events.append(payload)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        chunks: list[dict[str, Any]] = []
        schema_fields: list[str] = []
        for event in self._events:
            name = str(event.get("event", ""))
            counts[name] = counts.get(name, 0) + 1
            details = event.get("details", {})
            if name == "schema_done" and isinstance(details, dict):
                schema_fields = [str(field) for field in details.get("fields", [])]
            if name in {"chunk_done", "chunk_failed", "chunk_repair_done"} and isinstance(details, dict):
                chunks.append(
                    {
                        "event": name,
                        "chunk_index": details.get("chunk_index"),
                        "line_start": details.get("line_start"),
                        "line_end": details.get("line_end"),
                        "input_line_count": details.get("input_line_count"),
                        "record_count": details.get("record_count"),
                        "elapsed_seconds": details.get("elapsed_seconds"),
                        "error": details.get("error"),
                    }
                )
        return {
            "log_file": self.relative_log_file,
            "event_count": len(self._events),
            "events": counts,
            "elapsed_seconds": round(perf_counter() - self._started_at, 3),
            "schema_fields": schema_fields,
            "chunk_events": chunks,
            "failed_chunk_count": counts.get("chunk_failed", 0),
            "repair_count": counts.get("chunk_repair_done", 0),
        }


def _short_error(exc: Exception | str, *, limit: int = 300) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _records_for_log(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "line_id": record.get("line_id"),
            "values": record.get("values", {}),
        }
        for record in records
    ]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            candidate, end = decoder.raw_decode(stripped)
            if isinstance(candidate, dict) and not stripped[end:].strip():
                return candidate
        except json.JSONDecodeError:
            pass
    parsed: dict[str, Any] | None = None
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and (
            "records" in candidate or "fields" in candidate or parsed is None
        ):
            parsed = candidate
    if parsed is None:
        raise ValueError("Model response did not contain a JSON object.")
    return parsed


def _normalize_field_name(value: str) -> str:
    return value.strip().strip("`").strip()


def _schema_from_model(
    model: Any,
    *,
    knowledge_text: str,
    target_table: str,
    requested_fields: list[str] | None,
) -> tuple[list[dict[str, str]], int]:
    prompt = {
        "target_table": target_table,
        "requested_fields": requested_fields,
        "instruction": (
            "Extract the field schema for target_table from the knowledge document. "
            "Return only JSON: {\"fields\":[{\"name\":\"...\",\"description\":\"...\"}]}. "
            "Use exact field names from the document. If requested_fields is provided, "
            "include only those fields, preserving their requested names if missing."
        ),
        "knowledge": knowledge_text,
    }
    response = invoke_model_with_retries(
        model,
        [
            SystemMessage(content="You extract compact machine-readable schemas from knowledge docs."),
            HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
        ],
    )
    parsed = _last_json_object(_message_text(response))
    raw_fields = parsed.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError("Schema response must contain fields list.")
    fields: list[dict[str, str]] = []
    for item in raw_fields:
        if not isinstance(item, dict):
            continue
        name = _normalize_field_name(str(item.get("name") or ""))
        if not name:
            continue
        fields.append({"name": name, "description": str(item.get("description") or "")})
    if not fields:
        raise ValueError("Schema response did not contain usable fields.")
    return fields, 1


def _split_record_lines(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped:
            lines.append({"line_id": index, "text": stripped})
    return lines


def _chunk_lines(lines: list[dict[str, Any]], available_calls: int) -> list[list[dict[str, Any]]]:
    if not lines:
        return []
    extraction_calls = max(1, available_calls // 2) if available_calls > 1 else 1
    chunk_count = max(1, min(extraction_calls, len(lines)))
    chunk_size = max(1, math.ceil(len(lines) / chunk_count))
    return [lines[index:index + chunk_size] for index in range(0, len(lines), chunk_size)]


def _extract_chunk(
    model: Any,
    *,
    target_table: str,
    fields: list[dict[str, str]],
    lines: list[dict[str, Any]],
    repair_error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "target_table": target_table,
        "fields": fields,
        "lines": lines,
        "rules": [
            "Each non-empty source line is an independent candidate record.",
            "Return one item per input line with the same line_id.",
            "Use null when a field is absent; do not infer or invent missing values.",
            "Preserve source values exactly. Do not convert units.",
            "If a line contains an earlier mistaken value and a final confirmed/corrected value, choose the final confirmed/corrected value.",
            "Set is_record=false for headings, introductions, summaries, or lines that contain no record data.",
        ],
        "output_format": (
            "Return only JSON: {\"records\":[{\"line_id\":1,\"is_record\":true,"
            "\"values\":{\"field\":value}}]}."
        ),
    }
    if repair_error:
        payload["repair_error"] = repair_error
    response = invoke_model_with_retries(
        model,
        [
            SystemMessage(content="You extract structured records from line-oriented markdown."),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    return _last_json_object(_message_text(response))


def _validate_chunk_records(
    payload: dict[str, Any],
    *,
    fields: list[str],
    expected_line_ids: set[int],
) -> list[dict[str, Any]]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Chunk response must contain records list.")
    allowed_fields = set(fields)
    seen: set[int] = set()
    records: list[dict[str, Any]] = []
    for item in raw_records:
        if not isinstance(item, dict):
            raise ValueError("Each record item must be an object.")
        line_id = item.get("line_id")
        if not isinstance(line_id, int) or line_id not in expected_line_ids:
            raise ValueError(f"Invalid or unexpected line_id: {line_id!r}.")
        if line_id in seen:
            raise ValueError(f"Duplicate line_id: {line_id}.")
        seen.add(line_id)
        if item.get("is_record") is False:
            continue
        values = item.get("values")
        if not isinstance(values, dict):
            raise ValueError(f"Record {line_id} must contain values object.")
        unknown = set(values) - allowed_fields
        if unknown:
            raise ValueError(f"Record {line_id} returned unknown fields: {sorted(unknown)}.")
        records.append({"line_id": line_id, "values": values})
    missing = expected_line_ids - seen
    if missing:
        raise ValueError(f"Missing line_id values: {sorted(missing)[:10]}.")
    return records


def _catalog_table_names(catalog: dict[str, Any]) -> set[str]:
    return {str(table.get("table", "")) for table in iter_logical_tables(catalog) if table.get("table")}


def _registered_table_name(path: str, catalog: dict[str, Any]) -> str:
    stem = PurePosixPath(path).stem
    existing = {name.lower() for name in _catalog_table_names(catalog)}
    if stem.lower() in existing:
        return f"{stem}_extracted"
    return stem


def _manifest_path(workspace_root: Path) -> Path:
    return workspace_root / GENERATED_STRUCTURED_DOC_DIR / STRUCTURED_DOC_MANIFEST


def _load_manifest(workspace_root: Path) -> dict[str, Any]:
    path = _manifest_path(workspace_root)
    if not path.exists():
        return {"tables": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"tables": []}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tables"), list):
        return {"tables": []}
    return manifest


def _write_manifest(workspace_root: Path, entry: dict[str, Any]) -> None:
    path = _manifest_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(workspace_root)
    tables = [
        item for item in manifest.get("tables", [])
        if isinstance(item, dict) and item.get("registered_table") != entry["registered_table"]
    ]
    tables.append(entry)
    manifest["tables"] = tables
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(
    *,
    path: str,
    knowledge_path: str,
    target_table: str,
    fields: list[str],
    doc_hash: str,
    knowledge_hash: str,
) -> str:
    payload = {
        "path": path,
        "knowledge_path": knowledge_path,
        "target_table": target_table,
        "fields": fields,
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_cached_extraction(
    workspace_root: Path,
    *,
    cache_key: str,
    logger: StructuredDocLogger,
) -> StructuredDocExtraction | None:
    manifest = _load_manifest(workspace_root)
    for entry in manifest.get("tables", []):
        if not isinstance(entry, dict) or entry.get("cache_key") != cache_key:
            continue
        file_name = str(entry.get("file") or "")
        data_path = workspace_root / GENERATED_STRUCTURED_DOC_DIR / file_name
        if not data_path.exists():
            continue
        columns = [str(column) for column in entry.get("columns", [])]
        rows: list[list[Any]] = []
        for line in data_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append([record.get(column) for column in columns])
        metadata = {
            "registered_table": entry.get("registered_table"),
            "source_path": entry.get("source_path"),
            "knowledge_path": entry.get("knowledge_path"),
            "target_table": entry.get("target_table"),
            "row_count": len(rows),
            "model_call_count": 0,
            "cache_hit": True,
            "log_file": logger.relative_log_file,
        }
        logger.emit(
            "cache_hit",
            cache_key=cache_key,
            registered_table=entry.get("registered_table"),
            row_count=len(rows),
        )
        metadata["log_summary"] = logger.summary()
        return StructuredDocExtraction(columns=columns, rows=rows, metadata=metadata)
    return None


def _persist_extraction(
    workspace_root: Path,
    *,
    entry: dict[str, Any],
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    output_dir = workspace_root / GENERATED_STRUCTURED_DOC_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / str(entry["file"])
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {column: value for column, value in zip(columns, row, strict=False)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_manifest(workspace_root, entry)


def extract_structured_doc(
    *,
    task: PublicTask,
    workspace: TaskContextWorkspace,
    catalog: dict[str, Any],
    model: Any,
    path: str,
    knowledge_path: str = "knowledge.md",
    target_table: str | None = None,
    fields: list[str] | None = None,
    max_model_calls: int = MAX_MODEL_CALLS,
    log_dir: Path | None = None,
) -> StructuredDocExtraction:
    if model is None:
        raise ValueError("extract_structured_doc requires an available model.")
    max_calls = min(max(1, int(max_model_calls)), MAX_MODEL_CALLS)
    normalized_path = normalize_context_relative_path(path)
    normalized_knowledge_path = normalize_context_relative_path(knowledge_path)
    resolved_doc = resolve_context_path(task, normalized_path)
    resolved_knowledge = resolve_context_path(task, normalized_knowledge_path)
    doc_text = resolved_doc.read_text(encoding="utf-8", errors="replace")
    knowledge_text = resolved_knowledge.read_text(encoding="utf-8", errors="replace")
    target = target_table or PurePosixPath(normalized_path).stem
    requested_fields = [_normalize_field_name(field) for field in fields] if fields else None

    doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
    knowledge_hash = hashlib.sha256(knowledge_text.encode("utf-8")).hexdigest()
    field_names_for_cache = requested_fields or []
    key = _cache_key(
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        fields=field_names_for_cache,
        doc_hash=doc_hash,
        knowledge_hash=knowledge_hash,
    )
    workspace_root = workspace.materialize()
    registered_table = _registered_table_name(normalized_path, catalog)
    logger = StructuredDocLogger(workspace_root, registered_table, log_dir=log_dir)
    logger.emit(
        "start",
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        requested_fields=requested_fields,
        max_model_calls=max_calls,
        registered_table=registered_table,
    )
    cached = _read_cached_extraction(workspace_root, cache_key=key, logger=logger)
    if cached is not None:
        return cached
    logger.emit("cache_miss", cache_key=key, registered_table=registered_table)

    model_calls = 0
    try:
        schema_start = perf_counter()
        logger.emit("schema_start", target_table=target, requested_fields=requested_fields)
        schema_fields, schema_calls = _schema_from_model(
            model,
            knowledge_text=knowledge_text,
            target_table=target,
            requested_fields=requested_fields,
        )
        model_calls += schema_calls
        columns = [field["name"] for field in schema_fields]
        if not columns:
            raise ValueError(f"No fields were found for target table {target!r}.")
        logger.emit(
            "schema_done",
            field_count=len(columns),
            fields=columns,
            elapsed_seconds=round(perf_counter() - schema_start, 3),
            model_call_count=model_calls,
        )
        if model_calls >= max_calls:
            raise ValueError("Model call budget exhausted before record extraction.")
    except Exception as exc:
        logger.emit("schema_failed", error=_short_error(exc), model_call_count=model_calls)
        logger.emit("failed", error=_short_error(exc), model_call_count=model_calls)
        raise StructuredDocExtractionError(str(exc), log_summary=logger.summary()) from exc

    lines = _split_record_lines(doc_text)
    available_calls = max_calls - model_calls
    chunks = _chunk_lines(lines, available_calls)
    chunk_sizes = [len(chunk) for chunk in chunks]
    logger.emit(
        "chunk_plan",
        input_line_count=len(lines),
        chunk_count=len(chunks),
        chunk_sizes=chunk_sizes,
        available_model_calls=available_calls,
        reserved_repair_budget=max(0, available_calls - len(chunks)),
    )
    extracted_records: list[dict[str, Any]] = []
    errors: list[str] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        if model_calls >= max_calls:
            exc = ValueError("Model call budget exhausted during record extraction.")
            logger.emit("failed", error=_short_error(exc), model_call_count=model_calls)
            raise StructuredDocExtractionError(str(exc), log_summary=logger.summary()) from exc
        expected_line_ids = {int(line["line_id"]) for line in chunk}
        line_ids = sorted(expected_line_ids)
        line_start = line_ids[0] if line_ids else None
        line_end = line_ids[-1] if line_ids else None
        logger.emit(
            "chunk_start",
            chunk_index=chunk_index,
            line_start=line_start,
            line_end=line_end,
            input_line_count=len(chunk),
            model_call_count=model_calls,
        )
        chunk_start = perf_counter()
        try:
            payload = _extract_chunk(model, target_table=target, fields=schema_fields, lines=chunk)
            model_calls += 1
            chunk_records = _validate_chunk_records(
                payload, fields=columns, expected_line_ids=expected_line_ids
            )
            extracted_records.extend(chunk_records)
            logger.emit(
                "chunk_done",
                chunk_index=chunk_index,
                line_start=line_start,
                line_end=line_end,
                input_line_count=len(chunk),
                record_count=len(chunk_records),
                extracted_records=_records_for_log(chunk_records),
                elapsed_seconds=round(perf_counter() - chunk_start, 3),
                model_call_count=model_calls,
            )
        except Exception as exc:
            errors.append(str(exc))
            logger.emit(
                "chunk_failed",
                chunk_index=chunk_index,
                line_start=line_start,
                line_end=line_end,
                input_line_count=len(chunk),
                elapsed_seconds=round(perf_counter() - chunk_start, 3),
                model_call_count=model_calls,
                error=_short_error(exc),
            )
            if model_calls >= max_calls:
                error = f"Chunk extraction failed and no repair budget remains: {exc}"
                logger.emit("failed", error=_short_error(error), model_call_count=model_calls)
                raise StructuredDocExtractionError(error, log_summary=logger.summary()) from exc
            repair_start = perf_counter()
            try:
                payload = _extract_chunk(
                    model,
                    target_table=target,
                    fields=schema_fields,
                    lines=chunk,
                    repair_error=str(exc),
                )
                model_calls += 1
                chunk_records = _validate_chunk_records(
                    payload, fields=columns, expected_line_ids=expected_line_ids
                )
                extracted_records.extend(chunk_records)
                logger.emit(
                    "chunk_repair_done",
                    chunk_index=chunk_index,
                    line_start=line_start,
                    line_end=line_end,
                    input_line_count=len(chunk),
                    record_count=len(chunk_records),
                    extracted_records=_records_for_log(chunk_records),
                    elapsed_seconds=round(perf_counter() - repair_start, 3),
                    model_call_count=model_calls,
                )
            except Exception as repair_exc:
                logger.emit(
                    "failed",
                    error=_short_error(repair_exc),
                    chunk_index=chunk_index,
                    model_call_count=model_calls,
                )
                raise StructuredDocExtractionError(
                    str(repair_exc), log_summary=logger.summary()
                ) from repair_exc

    extracted_records.sort(key=lambda item: int(item["line_id"]))
    rows = [[record["values"].get(column) for column in columns] for record in extracted_records]
    file_name = f"{registered_table}.jsonl"
    log_file = logger.relative_log_file
    entry = {
        "source_path": normalized_path,
        "knowledge_path": normalized_knowledge_path,
        "target_table": target,
        "registered_table": registered_table,
        "columns": columns,
        "row_count": len(rows),
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
        "cache_key": key,
        "file": file_name,
        "log_file": log_file,
    }
    _persist_extraction(workspace_root, entry=entry, columns=columns, rows=rows)
    logger.emit(
        "persist_done",
        data_file=f"{GENERATED_STRUCTURED_DOC_DIR}/{file_name}",
        manifest_file=f"{GENERATED_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}",
        log_file=log_file,
        registered_table=registered_table,
        row_count=len(rows),
    )
    logger.emit("done", row_count=len(rows), model_call_count=model_calls)
    metadata = {
        "registered_table": registered_table,
        "source_path": normalized_path,
        "knowledge_path": normalized_knowledge_path,
        "target_table": target,
        "input_line_count": len(lines),
        "row_count": len(rows),
        "skipped_line_count": len(lines) - len(rows),
        "model_call_count": model_calls,
        "chunk_count": len(chunks),
        "cache_hit": False,
        "errors": errors,
        "log_file": log_file,
    }
    metadata["log_summary"] = logger.summary()
    return StructuredDocExtraction(columns=columns, rows=rows, metadata=metadata)
