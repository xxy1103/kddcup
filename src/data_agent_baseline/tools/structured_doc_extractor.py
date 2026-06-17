from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import StructuredDocToolConfig
from data_agent_baseline.inspectors.semantic_catalog import iter_logical_tables
from data_agent_baseline.model_retry import invoke_model_with_retries
from data_agent_baseline.tools.doc_structure import STRUCTURE_VERSION, load_doc_structure
from data_agent_baseline.tools.filesystem import normalize_context_relative_path, resolve_context_path
from data_agent_baseline.tools.python_exec import TaskContextWorkspace


GENERATED_STRUCTURED_DOC_DIR = ".generated/structured_doc"
VISIBLE_STRUCTURED_DOC_DIR = "structured_doc"
STRUCTURED_DOC_MANIFEST = "manifest.json"
MAX_MODEL_CALLS = 20
PLAN_VERSION = 3
CHUNKING_VERSION = 1
LINE_ID_KEY = "line_id"


@dataclass(frozen=True, slots=True)
class StructuredDocExtraction:
    columns: list[str]
    rows: list[list[Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    target_fields: list[dict[str, str]]
    entity_key_fields: list[str]
    fallback_entity_key: str
    merge_grain: str
    field_hints: dict[str, str]


@dataclass(frozen=True, slots=True)
class MergeResult:
    rows: list[list[Any]]
    fact_count: int
    merged_row_count: int
    dropped_empty_fact_count: int
    column_non_null_counts: dict[str, int]
    field_conflicts: list[dict[str, Any]]
    missing_entity_key_count: int


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    chunks: list[list[dict[str, Any]]]
    chunking_reason: str
    repair_budget: int


class StructuredDocExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        log_summary: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.log_summary = log_summary
        self.details = details or {}


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
            visible_dir = log_dir / VISIBLE_STRUCTURED_DOC_DIR
            visible_dir.mkdir(parents=True, exist_ok=True)
            self.relative_log_file = (
                f"{VISIBLE_STRUCTURED_DOC_DIR}/structured_doc_{registered_table}.log.jsonl"
            )
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
        quality_warnings: list[str] = []
        merge_summary: dict[str, Any] | None = None
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
                        "fact_count": details.get("fact_count"),
                        "elapsed_seconds": details.get("elapsed_seconds"),
                        "error": details.get("error"),
                    }
                )
            if name == "merge_done" and isinstance(details, dict):
                merge_summary = {
                    "fact_count": details.get("fact_count"),
                    "merged_row_count": details.get("merged_row_count"),
                    "column_non_null_counts": details.get("column_non_null_counts"),
                    "field_conflict_count": details.get("field_conflict_count"),
                    "dropped_empty_fact_count": details.get("dropped_empty_fact_count"),
                    "missing_entity_key_count": details.get("missing_entity_key_count"),
                }
            if name == "quality_warning" and isinstance(details, dict):
                quality_warnings.append(str(details.get("warning", "")))
        return {
            "log_file": self.relative_log_file,
            "event_count": len(self._events),
            "events": counts,
            "elapsed_seconds": round(perf_counter() - self._started_at, 3),
            "schema_fields": schema_fields,
            "chunk_events": chunks,
            "merge_summary": merge_summary,
            "quality_warnings": quality_warnings,
            "failed_chunk_count": counts.get("chunk_failed", 0),
            "repair_count": counts.get("chunk_repair_done", 0),
        }


def _short_error(exc: Exception | str, *, limit: int = 300) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


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
            "facts" in candidate
            or "target_fields" in candidate
            or "fields" in candidate
            or parsed is None
        ):
            parsed = candidate
    if parsed is None:
        raise ValueError("Model response did not contain a JSON object.")
    return parsed


def _normalize_field_name(value: str) -> str:
    return value.strip().strip("`").strip()


def _normalize_key_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _split_record_lines(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped:
            lines.append({"line_id": index, "text": stripped})
    return lines


def _normalize_line_ranges(value: Any) -> list[tuple[int, int]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("line_ranges must be a list of [start, end] pairs.")
    ranges: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
        ):
            raise ValueError("Each line_ranges item must be [start, end] integers.")
        start, end = int(item[0]), int(item[1])
        if start < 1 or end < start:
            raise ValueError("line_ranges must be 1-based inclusive ranges with end >= start.")
        ranges.append((start, end))
    return ranges


def _line_in_ranges(line_id: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= line_id <= end for start, end in ranges)


def _normalize_block_ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("block_ids must be a list of strings.")
    return [str(item).strip() for item in value if str(item).strip()]


def _load_selected_blocks(
    workspace_root: Path,
    *,
    path: str,
    block_ids: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    if not block_ids:
        return [], None
    doc_stem = PurePosixPath(path).stem
    structure = load_doc_structure(workspace_root, doc_stem)
    if structure is None:
        raise ValueError(
            "block_ids were provided but no persisted document structure was found. "
            "Call inspect_doc_structure first, or pass explicit line_ranges."
        )
    blocks = [block for block in structure.get("blocks", []) if isinstance(block, dict)]
    by_id = {str(block.get("block_id")): block for block in blocks}
    missing = [block_id for block_id in block_ids if block_id not in by_id]
    if missing:
        raise ValueError(f"Unknown block_ids for {path!r}: {missing}.")
    selected = [by_id[block_id] for block_id in block_ids]
    structure_hash = None
    raw_metadata = structure.get("structure")
    if isinstance(raw_metadata, dict):
        value = raw_metadata.get("structure_hash")
        structure_hash = None if value is None else str(value)
    return selected, structure_hash


def _load_doc_structure_blocks(
    workspace_root: Path,
    *,
    path: str,
) -> tuple[list[dict[str, Any]], str | None]:
    doc_stem = PurePosixPath(path).stem
    structure = load_doc_structure(workspace_root, doc_stem)
    if structure is None:
        return [], None
    blocks = [block for block in structure.get("blocks", []) if isinstance(block, dict)]
    structure_hash = None
    raw_metadata = structure.get("structure")
    if isinstance(raw_metadata, dict):
        value = raw_metadata.get("structure_hash")
        structure_hash = None if value is None else str(value)
    return blocks, structure_hash


def _block_candidate_fields(block: dict[str, Any]) -> set[str]:
    raw_fields = block.get("candidate_fields", [])
    if not isinstance(raw_fields, list):
        return set()
    return {_normalize_field_name(str(field)) for field in raw_fields if str(field).strip()}


def _continuation_ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _auto_select_blocks_for_fields(
    blocks: list[dict[str, Any]],
    *,
    requested_fields: list[str] | None,
) -> dict[str, Any]:
    by_id = {str(block.get("block_id")): block for block in blocks if block.get("block_id")}
    field_to_blocks: dict[str, list[str]] = {}
    selected_ids: set[str] = set()
    requested = [_normalize_field_name(field) for field in requested_fields or []]

    if requested:
        for field in requested:
            matches = [
                str(block.get("block_id"))
                for block in blocks
                if field in _block_candidate_fields(block)
            ]
            field_to_blocks[field] = matches
            selected_ids.update(matches)
        missing_fields = [field for field, matches in field_to_blocks.items() if not matches]
    else:
        missing_fields = []
        for block in blocks:
            block_id = str(block.get("block_id") or "").strip()
            if block_id and _block_candidate_fields(block):
                selected_ids.add(block_id)

    pending = list(selected_ids)
    while pending:
        block_id = pending.pop()
        block = by_id.get(block_id)
        if block is None:
            continue
        block_scope = block.get("scope_id") or block.get("section_scope")
        for parent_id in _continuation_ids(block.get("continuation_of")):
            parent = by_id.get(parent_id)
            if parent is None:
                continue
            parent_scope = parent.get("scope_id") or parent.get("section_scope")
            # Only follow continuation within the same extraction scope.
            # Different scopes may be adjacent in document reading order
            # but are not data continuations of each other.
            if parent_scope and block_scope and parent_scope != block_scope:
                continue
            if parent_id not in selected_ids:
                selected_ids.add(parent_id)
                pending.append(parent_id)

    selected_blocks = [
        block for block in blocks if str(block.get("block_id") or "").strip() in selected_ids
    ]
    selected_block_ids = [str(block.get("block_id")) for block in selected_blocks]
    return {
        "selected_blocks": selected_blocks,
        "selected_block_ids": selected_block_ids,
        "field_to_blocks": field_to_blocks,
        "missing_fields": missing_fields,
    }


def _selected_line_context(
    lines: list[dict[str, Any]],
    *,
    selected_blocks: list[dict[str, Any]],
    line_ranges: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[tuple[int, int]]]:
    ranges: list[tuple[int, int]] = list(line_ranges)
    line_context: dict[int, dict[str, Any]] = {}
    for block in selected_blocks:
        start = int(block["data_start_line"])
        end = int(block["data_end_line"])
        ranges.append((start, end))
        for line_id in range(start, end + 1):
            line_context[line_id] = block
    if not ranges:
        return lines, line_context, []
    selected: list[dict[str, Any]] = []
    for line in lines:
        line_id = int(line["line_id"])
        if not _line_in_ranges(line_id, ranges):
            continue
        copied = dict(line)
        block = line_context.get(line_id)
        if block is not None:
            copied["block_id"] = block.get("block_id")
            copied["section_scope"] = block.get("section_scope") or block.get("scope_id")
            copied["scope_name"] = block.get("scope_name")
            copied["candidate_fields"] = block.get("candidate_fields", [])
        selected.append(copied)
    return selected, line_context, ranges


def _sample_lines_for_plan(
    lines: list[dict[str, Any]], target_count: int = 30
) -> list[dict[str, Any]]:
    """Sample lines evenly across blocks so the LLM can see block→field relationships.

    Returns a list of block summaries, each containing block metadata and
    evenly-spaced sample lines from that block.
    """
    # Group lines by block_id (or "_ungrouped" when no block attribution exists)
    groups: dict[str, dict[str, Any]] = {}
    ordered_block_ids: list[str] = []
    for line in lines:
        block_id = str(line.get("block_id") or "_ungrouped")
        if block_id not in groups:
            groups[block_id] = {
                "block_id": line.get("block_id"),
                "scope_id": line.get("section_scope") or line.get("scope_id"),
                "scope_name": line.get("scope_name"),
                "candidate_fields": line.get("candidate_fields", []),
                "all_lines": [],
            }
            ordered_block_ids.append(block_id)
        groups[block_id]["all_lines"].append(line)

    total_lines = len(lines)
    if total_lines <= target_count:
        # Small enough — include every line grouped by block.
        return [
            {
                "block_id": groups[bid]["block_id"],
                "scope_id": groups[bid]["scope_id"],
                "scope_name": groups[bid]["scope_name"],
                "candidate_fields": groups[bid]["candidate_fields"],
                "sample_lines": [
                    {"line_id": int(ln["line_id"]), "text": str(ln.get("text", ""))}
                    for ln in groups[bid]["all_lines"]
                ],
            }
            for bid in ordered_block_ids
        ]

    # Allocate sample quota per block proportional to its share of total lines.
    block_count = len(ordered_block_ids)
    allocated: list[int] = []
    for i, bid in enumerate(ordered_block_ids):
        share = len(groups[bid]["all_lines"]) / total_lines
        quota = max(1, round(target_count * share))
        allocated.append(quota)

    # Adjust so total matches target_count (may be off due to rounding).
    while sum(allocated) < target_count:
        # Give extra to the largest block that is under its line count.
        for i in sorted(
            range(block_count),
            key=lambda j: len(groups[ordered_block_ids[j]]["all_lines"]),
            reverse=True,
        ):
            bid = ordered_block_ids[i]
            if allocated[i] < len(groups[bid]["all_lines"]):
                allocated[i] += 1
                if sum(allocated) >= target_count:
                    break
    while sum(allocated) > target_count:
        # Shrink the largest-quota block that still has room above 1.
        for i in sorted(range(block_count), key=lambda j: allocated[j], reverse=True):
            if allocated[i] > 1:
                allocated[i] -= 1
                if sum(allocated) <= target_count:
                    break

    # Evenly sample within each block.
    result: list[dict[str, Any]] = []
    for i, bid in enumerate(ordered_block_ids):
        block_lines = groups[bid]["all_lines"]
        k = min(allocated[i], len(block_lines))
        if k <= 0:
            continue
        if k == 1:
            indices = [0]
        else:
            step = (len(block_lines) - 1) / (k - 1) if k > 1 else 1.0
            indices = [min(len(block_lines) - 1, round(j * step)) for j in range(k)]
        result.append(
            {
                "block_id": groups[bid]["block_id"],
                "scope_id": groups[bid]["scope_id"],
                "scope_name": groups[bid]["scope_name"],
                "candidate_fields": groups[bid]["candidate_fields"],
                "sample_lines": [
                    {"line_id": int(block_lines[idx]["line_id"]), "text": str(block_lines[idx].get("text", ""))}
                    for idx in indices
                ],
            }
        )
    return result


def _lines_from_sizes(lines: list[dict[str, Any]], sizes: list[int]) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    offset = 0
    for size in sizes:
        if size <= 0:
            continue
        chunks.append(lines[offset:offset + size])
        offset += size
    return chunks


def _balanced_chunk_sizes(line_count: int, chunk_count: int) -> list[int]:
    base_size, extra = divmod(line_count, chunk_count)
    return [base_size + (1 if index < extra else 0) for index in range(chunk_count)]


def _chunk_lines(
    lines: list[dict[str, Any]],
    available_calls: int,
    *,
    config: StructuredDocToolConfig | None = None,
) -> ChunkPlan:
    structured_doc_config = config or StructuredDocToolConfig()
    if not lines:
        return ChunkPlan(chunks=[], chunking_reason="empty_input", repair_budget=max(0, available_calls))
    if available_calls <= 0:
        raise ValueError("No model calls remain for fact extraction.")

    line_count = len(lines)
    min_chunk_lines = structured_doc_config.min_chunk_lines
    max_chunk_lines = structured_doc_config.max_chunk_lines
    required_chunks = max(1, math.ceil(line_count / max_chunk_lines))
    if required_chunks > available_calls:
        raise ValueError(
            "Selected document lines require more extraction chunks than the model-call "
            f"budget allows: selected_line_count={line_count}, "
            f"required_chunks={required_chunks}, available_model_calls={available_calls}, "
            f"max_chunk_lines={max_chunk_lines}."
        )

    if line_count <= max_chunk_lines:
        chunk_count = 1
    else:
        preferred_chunks = max(1, available_calls // 2) if available_calls > 1 else 1
        max_useful_chunks = max(1, math.ceil(line_count / min_chunk_lines))
        chunk_count = min(max(preferred_chunks, required_chunks), max_useful_chunks, line_count)
        if chunk_count < required_chunks:
            chunk_count = required_chunks

    if chunk_count == 1:
        sizes = [line_count]
        reason = "single_chunk"
    elif line_count < chunk_count * min_chunk_lines:
        sizes = [min_chunk_lines] * (chunk_count - 1)
        sizes.append(line_count - sum(sizes))
        reason = "min_chunk_with_remainder"
    else:
        sizes = _balanced_chunk_sizes(line_count, chunk_count)
        reason = "balanced_by_line_count"

    if any(size > max_chunk_lines for size in sizes):
        raise ValueError(
            "Chunk planner produced an oversized chunk: "
            f"chunk_sizes={sizes}, max_chunk_lines={max_chunk_lines}."
        )
    repair_budget = max(0, available_calls - len(sizes))
    return ChunkPlan(
        chunks=_lines_from_sizes(lines, sizes),
        chunking_reason=reason,
        repair_budget=repair_budget,
    )


def _build_extraction_plan(
    model: Any,
    *,
    target_table: str,
    requested_fields: list[str] | None,
    sample_blocks: list[dict[str, Any]],
) -> tuple[ExtractionPlan, int]:
    prompt = {
        "target_table": target_table,
        "requested_fields": requested_fields,
        "plan_version": PLAN_VERSION,
        "instruction": (
            "Create a fact-first extraction plan for converting a markdown/text "
            "document into a structured table. Return only JSON with: "
            "target_fields=[{name,description}], entity_key_fields=[...], "
            "fallback_entity_key, merge_grain, field_hints={field:hint}. "
            "entity_key_fields are internal merge keys and do not need to be final output "
            "columns. When target fields for the same entity are spread across selected "
            "blocks/sections, choose source-document entity anchors visible across those "
            "blocks, such as archive_id/doc id/record id/item id. Do not choose a final "
            "output field as the primary entity key if that field is visible only in one "
            "kind of block. Put answer fields in values; put cross-line/cross-section "
            "linking identifiers in entity_key. If no stable natural entity key is "
            "visible, use line_id fallback."
        ),
        "entity_key_rules": [
            "entity_key_fields are for deterministic merging only; they may be non-output source identifiers.",
            "Prefer a source identifier that appears in every selected block type needed for the requested fields.",
            "Reject keys that are visible only in identity blocks, metric blocks, or any single block family.",
            "Do not put target answer values into entity_key; target fields belong in values.",
            "candidate_fields metadata limits values extraction only; it must not prevent extracting entity_key.",
        ],
        "sample_blocks": sample_blocks,
    }
    response = invoke_model_with_retries(
        model,
        [
            SystemMessage(content="You design compact machine-readable fact extraction plans."),
            HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
        ],
    )
    parsed = _last_json_object(_message_text(response))
    raw_fields = parsed.get("target_fields", parsed.get("fields"))
    if not isinstance(raw_fields, list):
        raise ValueError("Extraction plan must contain target_fields list.")
    fields: list[dict[str, str]] = []
    for item in raw_fields:
        if not isinstance(item, dict):
            continue
        name = _normalize_field_name(str(item.get("name") or ""))
        if not name:
            continue
        fields.append({"name": name, "description": str(item.get("description") or "")})
    if requested_fields:
        field_by_name = {field["name"].lower(): field for field in fields}
        fields = [
            field_by_name.get(field.lower(), {"name": field, "description": ""})
            for field in requested_fields
        ]
    if not fields:
        raise ValueError("Extraction plan did not contain usable target fields.")
    raw_key_fields = parsed.get("entity_key_fields", [])
    entity_key_fields = [
        _normalize_field_name(str(value))
        for value in raw_key_fields
        if _normalize_field_name(str(value))
    ] if isinstance(raw_key_fields, list) else []
    field_hints_raw = parsed.get("field_hints", {})
    field_hints = {
        str(key): str(value)
        for key, value in field_hints_raw.items()
    } if isinstance(field_hints_raw, dict) else {}
    fallback = _normalize_field_name(str(parsed.get("fallback_entity_key") or LINE_ID_KEY))
    if not fallback:
        fallback = LINE_ID_KEY
    return (
        ExtractionPlan(
            target_fields=fields,
            entity_key_fields=entity_key_fields,
            fallback_entity_key=fallback,
            merge_grain=str(parsed.get("merge_grain") or ""),
            field_hints=field_hints,
        ),
        1,
    )


def _extract_chunk_facts(
    model: Any,
    *,
    target_table: str,
    plan: ExtractionPlan,
    lines: list[dict[str, Any]],
    repair_error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "target_table": target_table,
        "target_fields": plan.target_fields,
        "entity_key_fields": plan.entity_key_fields,
        "fallback_entity_key": plan.fallback_entity_key,
        "merge_grain": plan.merge_grain,
        "field_hints": plan.field_hints,
        "lines": lines,
        "rules": [
            "Extract facts visible in each source line; do not infer missing values.",
            "A source line may contribute only some target fields for an entity.",
            "Return only facts with at least one visible target field or useful entity key.",
            "Use null only when a visible fact explicitly has no value; omit absent fields.",
            "Preserve source values and units exactly; do not convert units.",
            "If a line contains earlier mistaken values and a final confirmed/corrected value, choose the final confirmed/corrected value.",
            "Use entity_key to link facts about the same entity across different lines/sections. Follow the plan's entity_key_fields whenever that key is visible.",
            "entity_key is an internal merge key and may be a source identifier that is not one of the target output fields.",
            "Even when a block only allows one target value field, still extract the same cross-block entity anchor into entity_key when it is visible.",
            "Do not put target answer values into entity_key; target fields belong in values.",
            "When a line includes block_id/section_scope/candidate_fields metadata, extract only values compatible with that block's candidate_fields.",
            "candidate_fields restrict values only; they do not restrict entity_key extraction.",
        ],
        "output_format": (
            "Return only JSON: {\"facts\":[{\"line_id\":1,\"is_fact\":true,"
            "\"entity_key\":{\"archive_id\":\"...\"},\"values\":{\"field\":value},"
            "\"evidence_fields\":[\"field\"]}]}."
        ),
    }
    if repair_error:
        payload["repair_error"] = repair_error
    response = invoke_model_with_retries(
        model,
        [
            SystemMessage(content="You extract visible structured facts from markdown/text documents."),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    return _last_json_object(_message_text(response))


def _validate_facts(
    payload: dict[str, Any],
    *,
    fields: list[str],
    expected_line_ids: set[int],
    fallback_entity_key: str,
    line_context: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        raise ValueError("Chunk response must contain facts list.")
    allowed_fields = set(fields)
    facts: list[dict[str, Any]] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            raise ValueError("Each fact item must be an object.")
        line_id = item.get("line_id")
        if not isinstance(line_id, int) or line_id not in expected_line_ids:
            raise ValueError(f"Invalid or unexpected line_id: {line_id!r}.")
        if item.get("is_fact") is False:
            continue
        values = item.get("values", {})
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError(f"Fact {line_id} must contain values object.")
        unknown = set(values) - allowed_fields
        if unknown:
            raise ValueError(f"Fact {line_id} returned unknown fields: {sorted(unknown)}.")
        entity_key = item.get("entity_key", {})
        if entity_key is None:
            entity_key = {}
        if not isinstance(entity_key, dict):
            raise ValueError(f"Fact {line_id} must contain entity_key object.")
        normalized_entity_key = {
            _normalize_field_name(str(key)): str(value).strip()
            for key, value in entity_key.items()
            if _normalize_field_name(str(key)) and _normalize_key_value(value) is not None
        }
        if not normalized_entity_key:
            normalized_entity_key = {fallback_entity_key: str(line_id)}
        evidence_fields = item.get("evidence_fields", [])
        if not isinstance(evidence_fields, list):
            evidence_fields = []
        facts.append(
            {
                "line_id": line_id,
                "entity_key": normalized_entity_key,
                "values": values,
                "evidence_fields": [str(field) for field in evidence_fields],
                "block_id": (line_context or {}).get(line_id, {}).get("block_id"),
                "section_scope": (
                    (line_context or {}).get(line_id, {}).get("section_scope")
                    or (line_context or {}).get(line_id, {}).get("scope_id")
                ),
            }
        )
    return facts


def _filter_facts_by_scope(
    facts: list[dict[str, Any]],
    *,
    line_context: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not line_context:
        return facts, []
    filtered_facts: list[dict[str, Any]] = []
    filtered_values: list[dict[str, Any]] = []
    for fact in facts:
        line_id = int(fact["line_id"])
        block = line_context.get(line_id)
        values = dict(fact.get("values", {}))
        if not block:
            filtered_facts.append(fact)
            continue
        candidate_fields = {
            str(field)
            for field in block.get("candidate_fields", [])
            if str(field).strip()
        }
        if not candidate_fields:
            if values:
                filtered_values.append(
                    {
                        "line_id": line_id,
                        "block_id": block.get("block_id"),
                        "section_scope": block.get("section_scope") or block.get("scope_id"),
                        "filtered_fields": sorted(values),
                        "reason": "block has no candidate_fields",
                    }
                )
            values = {}
        else:
            disallowed = sorted(field for field in values if field not in candidate_fields)
            if disallowed:
                for field in disallowed:
                    values.pop(field, None)
                filtered_values.append(
                    {
                        "line_id": line_id,
                        "block_id": block.get("block_id"),
                        "section_scope": block.get("section_scope") or block.get("scope_id"),
                        "filtered_fields": disallowed,
                        "allowed_fields": sorted(candidate_fields),
                    }
                )
        copied = dict(fact)
        copied["values"] = values
        copied["evidence_fields"] = [
            field for field in fact.get("evidence_fields", []) if field in values
        ]
        filtered_facts.append(copied)
    return filtered_facts, filtered_values


def _facts_for_log(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logged: list[dict[str, Any]] = []
    for fact in facts:
        item = {
            "line_id": fact.get("line_id"),
            "entity_key": fact.get("entity_key", {}),
            "values": fact.get("values", {}),
            "evidence_fields": fact.get("evidence_fields", []),
        }
        if fact.get("block_id") is not None:
            item["block_id"] = fact.get("block_id")
        if fact.get("section_scope") is not None:
            item["section_scope"] = fact.get("section_scope")
        logged.append(item)
    return logged


def _fact_scope_counts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for fact in facts:
        key = (
            str(fact.get("block_id") or ""),
            str(fact.get("section_scope") or ""),
        )
        counts[key] = counts.get(key, 0) + 1
    return [
        {"block_id": block_id or None, "section_scope": scope or None, "fact_count": count}
        for (block_id, scope), count in sorted(counts.items())
    ]


def _entity_sort_key(entity_key: dict[str, str], entity_key_fields: list[str]) -> tuple[tuple[str, str], ...]:
    if entity_key_fields:
        ordered = [
            (field, entity_key[field])
            for field in entity_key_fields
            if field in entity_key
        ]
        remaining = sorted((key, value) for key, value in entity_key.items() if key not in entity_key_fields)
        return tuple(ordered + remaining)
    return tuple(sorted(entity_key.items()))


def _merge_facts(
    facts: list[dict[str, Any]],
    *,
    columns: list[str],
    entity_key_fields: list[str],
) -> MergeResult:
    grouped: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    entity_order: list[tuple[tuple[str, str], ...]] = []
    field_conflicts: list[dict[str, Any]] = []
    dropped_empty_fact_count = 0
    missing_entity_key_count = 0
    allowed_columns = set(columns)
    for fact in sorted(facts, key=lambda item: int(item["line_id"])):
        values = {
            key: value
            for key, value in fact.get("values", {}).items()
            if key in allowed_columns and value is not None
        }
        if not values:
            dropped_empty_fact_count += 1
            continue
        entity_key = fact.get("entity_key", {})
        if not isinstance(entity_key, dict) or not entity_key:
            entity_key = {LINE_ID_KEY: str(fact["line_id"])}
            missing_entity_key_count += 1
        normalized_entity_key = {str(key): str(value) for key, value in entity_key.items()}
        key = _entity_sort_key(normalized_entity_key, entity_key_fields)
        if key not in grouped:
            grouped[key] = {"entity_key": normalized_entity_key, "values": {}, "line_ids": []}
            entity_order.append(key)
        row_values = grouped[key]["values"]
        grouped[key]["line_ids"].append(fact["line_id"])
        for field, value in values.items():
            old_value = row_values.get(field)
            if old_value is not None and old_value != value:
                field_conflicts.append(
                    {
                        "entity_key": normalized_entity_key,
                        "field": field,
                        "old_value": old_value,
                        "new_value": value,
                        "line_id": fact["line_id"],
                    }
                )
            row_values[field] = value
    rows = [
        [grouped[key]["values"].get(column) for column in columns]
        for key in entity_order
    ]
    column_non_null_counts = {
        column: sum(row[index] is not None for row in rows)
        for index, column in enumerate(columns)
    }
    return MergeResult(
        rows=rows,
        fact_count=len(facts),
        merged_row_count=len(rows),
        dropped_empty_fact_count=dropped_empty_fact_count,
        column_non_null_counts=column_non_null_counts,
        field_conflicts=field_conflicts,
        missing_entity_key_count=missing_entity_key_count,
    )


def _quality_warnings(
    *,
    columns: list[str],
    merge_result: MergeResult,
    entity_key_fields: list[str],
) -> list[str]:
    warnings: list[str] = []
    if not entity_key_fields:
        warnings.append("No natural entity key was identified; line_id fallback was used.")
    if merge_result.merged_row_count == 0:
        warnings.append("No valid merged rows were produced from extracted facts.")
    for column in columns:
        non_null = merge_result.column_non_null_counts.get(column, 0)
        if merge_result.merged_row_count >= 5 and non_null == 0:
            warnings.append(f"Column {column!r} is null for every merged row.")
    if merge_result.missing_entity_key_count:
        warnings.append(
            f"{merge_result.missing_entity_key_count} facts used fallback entity keys."
        )
    return warnings


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
    plan_version: int,
    block_ids: list[str] | None = None,
    line_ranges: list[tuple[int, int]] | None = None,
    doc_structure_hash: str | None = None,
    structure_version: int | None = None,
    chunking_version: int | None = None,
    chunking_config: dict[str, Any] | None = None,
    target_fields: list[str] | None = None,
    entity_key_fields: list[str] | None = None,
) -> str:
    payload = {
        "path": path,
        "knowledge_path": knowledge_path,
        "target_table": target_table,
        "fields": fields,
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
        "plan_version": plan_version,
        "block_ids": block_ids or [],
        "line_ranges": line_ranges or [],
        "doc_structure_hash": doc_structure_hash,
        "structure_version": structure_version,
        "chunking_version": chunking_version,
        "chunking_config": chunking_config or {},
        "target_fields": target_fields or [],
        "entity_key_fields": entity_key_fields or [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _chunking_config_for_cache(config: StructuredDocToolConfig) -> dict[str, int]:
    return {
        "min_chunk_lines": config.min_chunk_lines,
        "max_chunk_lines": config.max_chunk_lines,
        "max_selected_lines_for_llm_extraction": config.max_selected_lines_for_llm_extraction,
        "default_max_model_calls": config.default_max_model_calls,
        "hard_max_model_calls": config.hard_max_model_calls,
    }


def _read_cached_extraction(
    workspace_root: Path,
    *,
    preplan_cache_key: str,
    logger: StructuredDocLogger,
    visible_dir: Path | None = None,
) -> StructuredDocExtraction | None:
    manifest = _load_manifest(workspace_root)
    for entry in manifest.get("tables", []):
        if not isinstance(entry, dict) or entry.get("preplan_cache_key") != preplan_cache_key:
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
            "visible_data_file": (
                f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}" if visible_dir is not None else None
            ),
            "visible_manifest_file": (
                f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
                if visible_dir is not None
                else None
            ),
            "plan_version": entry.get("plan_version"),
            "selected_blocks": entry.get("selected_blocks", []),
            "selected_line_ranges": entry.get("selected_line_ranges", []),
            "auto_selected_blocks": entry.get("auto_selected_blocks", False),
            "auto_block_selection": entry.get("auto_block_selection"),
            "block_scopes": entry.get("block_scopes", []),
            "doc_structure_hash": entry.get("doc_structure_hash"),
            "structure_version": entry.get("structure_version"),
            "scope_filtered_fact_count": entry.get("scope_filtered_fact_count", 0),
            "entity_key_fields": entry.get("entity_key_fields", []),
            "merge_grain": entry.get("merge_grain"),
            "fact_count": entry.get("fact_count"),
            "merged_row_count": entry.get("merged_row_count"),
            "column_non_null_counts": entry.get("column_non_null_counts", {}),
            "field_conflict_count": entry.get("field_conflict_count", 0),
        }
        logger.emit(
            "cache_hit",
            cache_key=entry.get("cache_key"),
            preplan_cache_key=preplan_cache_key,
            registered_table=entry.get("registered_table"),
            row_count=len(rows),
        )
        if visible_dir is not None:
            visible_dir.mkdir(parents=True, exist_ok=True)
            (visible_dir / file_name).write_text(
                data_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (visible_dir / STRUCTURED_DOC_MANIFEST).write_text(
                json.dumps(_load_manifest(workspace_root), ensure_ascii=False, indent=2),
                encoding="utf-8",
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
    visible_dir: Path | None = None,
) -> None:
    output_dir = workspace_root / GENERATED_STRUCTURED_DOC_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / str(entry["file"])
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {column: value for column, value in zip(columns, row, strict=False)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_manifest(workspace_root, entry)
    if visible_dir is not None:
        visible_dir.mkdir(parents=True, exist_ok=True)
        visible_data_path = visible_dir / str(entry["file"])
        visible_data_path.write_text(data_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest_payload = _load_manifest(workspace_root)
        (visible_dir / STRUCTURED_DOC_MANIFEST).write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
    block_ids: list[str] | None = None,
    line_ranges: list[list[int]] | None = None,
    max_model_calls: int = MAX_MODEL_CALLS,
    structured_doc_config: StructuredDocToolConfig | None = None,
    log_dir: Path | None = None,
) -> StructuredDocExtraction:
    if model is None:
        raise ValueError("extract_structured_doc requires an available model.")
    config = structured_doc_config or StructuredDocToolConfig()
    max_calls = min(max(1, int(max_model_calls)), config.hard_max_model_calls)
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
    normalized_block_ids = _normalize_block_ids(block_ids)
    normalized_line_ranges = _normalize_line_ranges(line_ranges)
    workspace_root = workspace.materialize()
    registered_table = _registered_table_name(normalized_path, catalog)
    logger = StructuredDocLogger(workspace_root, registered_table, log_dir=log_dir)
    visible_structured_doc_dir = (
        log_dir / VISIBLE_STRUCTURED_DOC_DIR if log_dir is not None else None
    )
    logger.emit(
        "start",
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        requested_fields=requested_fields,
        max_model_calls=max_calls,
        registered_table=registered_table,
        plan_version=PLAN_VERSION,
        chunking_version=CHUNKING_VERSION,
        structured_doc_config=_chunking_config_for_cache(config),
        block_ids=normalized_block_ids,
        line_ranges=normalized_line_ranges,
    )

    auto_selected_blocks = False
    auto_block_selection: dict[str, Any] | None = None
    if normalized_block_ids:
        selected_blocks, doc_structure_hash = _load_selected_blocks(
            workspace_root,
            path=normalized_path,
            block_ids=normalized_block_ids,
        )
    elif normalized_line_ranges:
        selected_blocks, doc_structure_hash = [], None
    else:
        structure_blocks, doc_structure_hash = _load_doc_structure_blocks(
            workspace_root,
            path=normalized_path,
        )
        if not structure_blocks:
            error = (
                "Document structure cache is required before automatic block selection. "
                "Call inspect_doc_structure for this path first, or pass explicit "
                "block_ids/line_ranges."
            )
            logger.emit(
                "missing_doc_structure",
                path=normalized_path,
                recommendation=(
                    "Call inspect_doc_structure(path, target_table, fields) before "
                    "extract_structured_doc, or pass explicit line_ranges/block_ids."
                ),
            )
            logger.emit("failed", error=error, error_code="missing_doc_structure", model_call_count=0)
            raise StructuredDocExtractionError(
                error,
                log_summary=logger.summary(),
                details={
                    "error_code": "missing_doc_structure",
                    "recommendation": (
                        "Call inspect_doc_structure for this document before "
                        "extract_structured_doc, or pass explicit line_ranges/block_ids."
                    ),
                },
            )
        logger.emit(
            "doc_structure_cache_loaded",
            path=normalized_path,
            block_count=len(structure_blocks),
            doc_structure_hash=doc_structure_hash,
        )
        auto_block_selection = _auto_select_blocks_for_fields(
            structure_blocks,
            requested_fields=requested_fields,
        )
        selected_blocks = list(auto_block_selection["selected_blocks"])
        normalized_block_ids = list(auto_block_selection["selected_block_ids"])
        auto_selected_blocks = True
        logger.emit(
            "auto_block_select_done",
            requested_fields=requested_fields,
            selected_block_ids=normalized_block_ids,
            field_to_blocks=auto_block_selection["field_to_blocks"],
            missing_fields=auto_block_selection["missing_fields"],
            selected_block_count=len(selected_blocks),
        )
        if auto_block_selection["missing_fields"]:
            missing_fields = list(auto_block_selection["missing_fields"])
            error = (
                "Document structure cache does not contain blocks for requested fields: "
                f"{missing_fields}. Call inspect_doc_structure to review candidate_fields, "
                "or pass explicit line_ranges/block_ids."
            )
            logger.emit(
                "failed",
                error=error,
                error_code="missing_fields",
                missing_fields=missing_fields,
                model_call_count=0,
            )
            raise StructuredDocExtractionError(
                error,
                log_summary=logger.summary(),
                details={
                    "error_code": "missing_fields",
                    "missing_fields": missing_fields,
                    "field_to_blocks": auto_block_selection["field_to_blocks"],
                    "recommendation": (
                        "Inspect doc_structure candidate_fields, then pass exact "
                        "line_ranges/block_ids if the field is present but was not classified."
                    ),
                },
            )
        if not selected_blocks:
            error = (
                "Document structure cache did not select any blocks for extraction. "
                "Call inspect_doc_structure to review candidate_fields, or pass explicit "
                "line_ranges/block_ids."
            )
            logger.emit("failed", error=error, error_code="no_selected_blocks", model_call_count=0)
            raise StructuredDocExtractionError(
                error,
                log_summary=logger.summary(),
                details={
                    "error_code": "no_selected_blocks",
                    "field_to_blocks": auto_block_selection["field_to_blocks"],
                },
            )

    all_lines = _split_record_lines(doc_text)
    lines, line_context, selected_ranges = _selected_line_context(
        all_lines,
        selected_blocks=selected_blocks,
        line_ranges=normalized_line_ranges,
    )
    preplan_key = _cache_key(
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        fields=field_names_for_cache,
        doc_hash=doc_hash,
        knowledge_hash=knowledge_hash,
        plan_version=PLAN_VERSION,
        block_ids=normalized_block_ids,
        line_ranges=selected_ranges,
        doc_structure_hash=doc_structure_hash,
        structure_version=STRUCTURE_VERSION if normalized_block_ids else None,
        chunking_version=CHUNKING_VERSION,
        chunking_config=_chunking_config_for_cache(config),
    )
    if selected_ranges:
        logger.emit(
            "structure_selected",
            selected_blocks=[
                {
                    "block_id": block.get("block_id"),
                    "scope_id": block.get("scope_id") or block.get("section_scope"),
                    "scope_name": block.get("scope_name"),
                    "candidate_fields": block.get("candidate_fields", []),
                    "data_start_line": block.get("data_start_line"),
                    "data_end_line": block.get("data_end_line"),
                }
                for block in selected_blocks
            ],
            selected_line_ranges=selected_ranges,
            input_line_count=len(all_lines),
            selected_line_count=len(lines),
            auto_selected_blocks=auto_selected_blocks,
        )
    cached = _read_cached_extraction(
        workspace_root,
        preplan_cache_key=preplan_key,
        logger=logger,
        visible_dir=visible_structured_doc_dir,
    )
    if cached is not None:
        return cached
    logger.emit("cache_miss", preplan_cache_key=preplan_key, registered_table=registered_table)

    available_calls_before_plan = max_calls - 1
    logger.emit(
        "input_size_check",
        selected_line_count=len(lines),
        max_selected_lines_for_llm_extraction=config.max_selected_lines_for_llm_extraction,
        max_model_calls=max_calls,
        available_extraction_calls=available_calls_before_plan,
        min_chunk_lines=config.min_chunk_lines,
        max_chunk_lines=config.max_chunk_lines,
    )
    if len(lines) > config.max_selected_lines_for_llm_extraction:
        error = (
            "Selected document range is too large for LLM structured extraction: "
            f"selected_line_count={len(lines)}, "
            f"max_selected_lines_for_llm_extraction={config.max_selected_lines_for_llm_extraction}. "
            "Use inspect_doc_structure with narrower block_ids/line_ranges, or use "
            "read_doc/search_doc plus execute_python for regex/programmatic parsing."
        )
        logger.emit(
            "input_too_large",
            selected_line_count=len(lines),
            max_selected_lines_for_llm_extraction=config.max_selected_lines_for_llm_extraction,
            recommendation=(
                "Narrow block_ids/line_ranges, or parse the document with execute_python "
                "using regex/programmatic logic."
            ),
        )
        logger.emit("failed", error=error, model_call_count=0)
        raise StructuredDocExtractionError(error, log_summary=logger.summary())
    try:
        _chunk_lines(lines, available_calls_before_plan, config=config)
    except Exception as exc:
        error = (
            "Selected document range cannot fit the configured chunk/model-call budget "
            f"before schema planning: {exc}"
        )
        logger.emit("failed", error=_short_error(error), model_call_count=0)
        raise StructuredDocExtractionError(error, log_summary=logger.summary()) from exc

    model_calls = 0
    try:
        plan_start = perf_counter()
        logger.emit("schema_start", target_table=target, requested_fields=requested_fields)
        plan, plan_calls = _build_extraction_plan(
            model,
            target_table=target,
            requested_fields=requested_fields,
            sample_blocks=_sample_lines_for_plan(lines),
        )
        model_calls += plan_calls
        columns = [field["name"] for field in plan.target_fields]
        if not columns:
            raise ValueError(f"No fields were found for target table {target!r}.")
        logger.emit(
            "schema_done",
            field_count=len(columns),
            fields=columns,
            entity_key_fields=plan.entity_key_fields,
            fallback_entity_key=plan.fallback_entity_key,
            merge_grain=plan.merge_grain,
            elapsed_seconds=round(perf_counter() - plan_start, 3),
            model_call_count=model_calls,
        )
        logger.emit(
            "extraction_plan",
            target_fields=columns,
            entity_key_fields=plan.entity_key_fields,
            fallback_entity_key=plan.fallback_entity_key,
            merge_grain=plan.merge_grain,
            field_hints=plan.field_hints,
        )
        if model_calls >= max_calls:
            raise ValueError("Model call budget exhausted before fact extraction.")
    except Exception as exc:
        logger.emit("schema_failed", error=_short_error(exc), model_call_count=model_calls)
        logger.emit("failed", error=_short_error(exc), model_call_count=model_calls)
        raise StructuredDocExtractionError(str(exc), log_summary=logger.summary()) from exc

    available_calls = max_calls - model_calls
    chunk_plan = _chunk_lines(lines, available_calls, config=config)
    chunks = chunk_plan.chunks
    chunk_sizes = [len(chunk) for chunk in chunks]
    logger.emit(
        "chunk_plan",
        input_line_count=len(lines),
        chunk_count=len(chunks),
        chunk_sizes=chunk_sizes,
        available_model_calls=available_calls,
        reserved_repair_budget=chunk_plan.repair_budget,
        repair_budget=chunk_plan.repair_budget,
        min_chunk_lines=config.min_chunk_lines,
        max_chunk_lines=config.max_chunk_lines,
        chunking_version=CHUNKING_VERSION,
        chunking_reason=chunk_plan.chunking_reason,
    )
    facts: list[dict[str, Any]] = []
    errors: list[str] = []
    scope_filtered_fact_count = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        if model_calls >= max_calls:
            exc = ValueError("Model call budget exhausted during fact extraction.")
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
            payload = _extract_chunk_facts(model, target_table=target, plan=plan, lines=chunk)
            model_calls += 1
            chunk_facts = _validate_facts(
                payload,
                fields=columns,
                expected_line_ids=expected_line_ids,
                fallback_entity_key=plan.fallback_entity_key or LINE_ID_KEY,
                line_context=line_context,
            )
            chunk_facts, filtered_values = _filter_facts_by_scope(
                chunk_facts,
                line_context=line_context,
            )
            scope_filtered_fact_count += len(filtered_values)
            facts.extend(chunk_facts)
            for filtered in filtered_values[:20]:
                logger.emit("scope_filtered_fact", **filtered)
            logger.emit(
                "chunk_done",
                chunk_index=chunk_index,
                line_start=line_start,
                line_end=line_end,
                input_line_count=len(chunk),
                fact_count=len(chunk_facts),
                scope_filtered_fact_count=len(filtered_values),
                extracted_facts=_facts_for_log(chunk_facts),
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
                error = f"Chunk fact extraction failed and no repair budget remains: {exc}"
                logger.emit("failed", error=_short_error(error), model_call_count=model_calls)
                raise StructuredDocExtractionError(error, log_summary=logger.summary()) from exc
            repair_start = perf_counter()
            try:
                payload = _extract_chunk_facts(
                    model,
                    target_table=target,
                    plan=plan,
                    lines=chunk,
                    repair_error=str(exc),
                )
                model_calls += 1
                chunk_facts = _validate_facts(
                    payload,
                    fields=columns,
                    expected_line_ids=expected_line_ids,
                    fallback_entity_key=plan.fallback_entity_key or LINE_ID_KEY,
                    line_context=line_context,
                )
                chunk_facts, filtered_values = _filter_facts_by_scope(
                    chunk_facts,
                    line_context=line_context,
                )
                scope_filtered_fact_count += len(filtered_values)
                facts.extend(chunk_facts)
                for filtered in filtered_values[:20]:
                    logger.emit("scope_filtered_fact", **filtered)
                logger.emit(
                    "chunk_repair_done",
                    chunk_index=chunk_index,
                    line_start=line_start,
                    line_end=line_end,
                    input_line_count=len(chunk),
                    fact_count=len(chunk_facts),
                    scope_filtered_fact_count=len(filtered_values),
                    extracted_facts=_facts_for_log(chunk_facts),
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

    merge_result = _merge_facts(facts, columns=columns, entity_key_fields=plan.entity_key_fields)
    logger.emit(
        "merge_done",
        fact_count=merge_result.fact_count,
        merged_row_count=merge_result.merged_row_count,
        column_non_null_counts=merge_result.column_non_null_counts,
        field_conflict_count=len(merge_result.field_conflicts),
        field_conflicts=merge_result.field_conflicts[:50],
        dropped_empty_fact_count=merge_result.dropped_empty_fact_count,
        missing_entity_key_count=merge_result.missing_entity_key_count,
        scope_filtered_fact_count=scope_filtered_fact_count,
        block_scope_counts=_fact_scope_counts(facts),
    )
    warnings = _quality_warnings(columns=columns, merge_result=merge_result, entity_key_fields=plan.entity_key_fields)
    for warning in warnings:
        logger.emit("quality_warning", warning=warning)
    if not merge_result.rows:
        error = "No valid merged rows were produced from extracted facts."
        logger.emit("failed", error=error, model_call_count=model_calls)
        raise StructuredDocExtractionError(error, log_summary=logger.summary())

    file_name = f"{registered_table}.jsonl"
    log_file = logger.relative_log_file
    full_cache_key = _cache_key(
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        fields=field_names_for_cache,
        doc_hash=doc_hash,
        knowledge_hash=knowledge_hash,
        plan_version=PLAN_VERSION,
        block_ids=normalized_block_ids,
        line_ranges=selected_ranges,
        doc_structure_hash=doc_structure_hash,
        structure_version=STRUCTURE_VERSION if normalized_block_ids else None,
        chunking_version=CHUNKING_VERSION,
        chunking_config=_chunking_config_for_cache(config),
        target_fields=columns,
        entity_key_fields=plan.entity_key_fields,
    )
    block_scopes = [
        {
            "block_id": block.get("block_id"),
            "scope_id": block.get("scope_id") or block.get("section_scope"),
            "scope_name": block.get("scope_name"),
            "candidate_fields": block.get("candidate_fields", []),
            "data_start_line": block.get("data_start_line"),
            "data_end_line": block.get("data_end_line"),
        }
        for block in selected_blocks
    ]
    entry = {
        "source_path": normalized_path,
        "knowledge_path": normalized_knowledge_path,
        "target_table": target,
        "registered_table": registered_table,
        "columns": columns,
        "row_count": len(merge_result.rows),
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
        "preplan_cache_key": preplan_key,
        "cache_key": full_cache_key,
        "file": file_name,
        "log_file": log_file,
        "visible_data_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "visible_manifest_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "plan_version": PLAN_VERSION,
        "selected_blocks": normalized_block_ids,
        "selected_line_ranges": selected_ranges,
        "auto_selected_blocks": auto_selected_blocks,
        "auto_block_selection": auto_block_selection,
        "block_scopes": block_scopes,
        "doc_structure_hash": doc_structure_hash,
        "structure_version": STRUCTURE_VERSION if normalized_block_ids else None,
        "chunking_version": CHUNKING_VERSION,
        "chunking_config": _chunking_config_for_cache(config),
        "scope_filtered_fact_count": scope_filtered_fact_count,
        "entity_key_fields": plan.entity_key_fields,
        "merge_grain": plan.merge_grain,
        "fact_count": merge_result.fact_count,
        "merged_row_count": merge_result.merged_row_count,
        "column_non_null_counts": merge_result.column_non_null_counts,
        "field_conflict_count": len(merge_result.field_conflicts),
    }
    _persist_extraction(
        workspace_root,
        entry=entry,
        columns=columns,
        rows=merge_result.rows,
        visible_dir=visible_structured_doc_dir,
    )
    logger.emit(
        "persist_done",
        data_file=f"{GENERATED_STRUCTURED_DOC_DIR}/{file_name}",
        manifest_file=f"{GENERATED_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}",
        visible_data_file=(
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}"
            if visible_structured_doc_dir is not None
            else None
        ),
        visible_manifest_file=(
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
            if visible_structured_doc_dir is not None
            else None
        ),
        log_file=log_file,
        registered_table=registered_table,
        row_count=len(merge_result.rows),
    )
    logger.emit("done", row_count=len(merge_result.rows), model_call_count=model_calls)
    metadata = {
        "registered_table": registered_table,
        "source_path": normalized_path,
        "knowledge_path": normalized_knowledge_path,
        "target_table": target,
        "input_line_count": len(lines),
        "row_count": len(merge_result.rows),
        "skipped_line_count": max(0, len(lines) - merge_result.fact_count),
        "model_call_count": model_calls,
        "chunk_count": len(chunks),
        "cache_hit": False,
        "errors": errors,
        "log_file": log_file,
        "visible_data_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "visible_manifest_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "plan_version": PLAN_VERSION,
        "selected_blocks": normalized_block_ids,
        "selected_line_ranges": selected_ranges,
        "auto_selected_blocks": auto_selected_blocks,
        "auto_block_selection": auto_block_selection,
        "block_scopes": block_scopes,
        "doc_structure_hash": doc_structure_hash,
        "structure_version": STRUCTURE_VERSION if normalized_block_ids else None,
        "chunking_version": CHUNKING_VERSION,
        "chunking_config": _chunking_config_for_cache(config),
        "scope_filtered_fact_count": scope_filtered_fact_count,
        "entity_key_fields": plan.entity_key_fields,
        "merge_grain": plan.merge_grain,
        "fact_count": merge_result.fact_count,
        "merged_row_count": merge_result.merged_row_count,
        "column_non_null_counts": merge_result.column_non_null_counts,
        "field_conflict_count": len(merge_result.field_conflicts),
        "quality_warnings": warnings,
    }
    metadata["log_summary"] = logger.summary()
    return StructuredDocExtraction(columns=columns, rows=merge_result.rows, metadata=metadata)
