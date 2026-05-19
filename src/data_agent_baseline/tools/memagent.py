from __future__ import annotations

import concurrent.futures
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

import tiktoken

LLMFn = Callable[[str], str]

NO_MEMORY = "No previous memory"
RECURRENT_MAX_CONTEXT_LEN = 16384
RECURRENT_CHUNK_SIZE = 8192
RECURRENT_CHUNK_OVERLAP = 256
TIKTOKEN_ENCODING_NAME = "o200k_base"
MEMORY_MAX_TOKENS = 4096
TEXT_DOCUMENT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".xml"}

TEMPLATE_CONTEXT_UPDATE = """\
You are scanning a long document chunk by chunk to collect knowledge that helps answer the task.

The document may include a direct answer to the task or other information that helps complete it.
Gather whatever is relevant from this <section> and merge it into the updated text that replaces
<memory> below; you may use Markdown for structure.

<task>
{question}
</task>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated task context:
"""

PATTERN_MEMORY_INIT = """\
No patterns identified yet.  This document is being scanned chunk-by-chunk in
order.  Early chunks may NOT contain all the fields -- a field that is not in the
current section may appear many chunks later.

Build the extraction guide incrementally.  Your goal is NOT to extract the actual
data, but to identify text patterns so a deterministic extraction engine can
extract the data accurately.

CRITICAL RULES:
- NEVER invent regex patterns for a field you have NOT personally observed in a
  section.  If a field is listed in the question but you have not seen it yet,
  mark it as PENDING -- do NOT write a placeholder regex.
- When you DO encounter a data field, copy one complete original sentence as an
  example, then write the regex that captures the value from that exact sentence.
- The regex must be anchored to surrounding marker words so it does NOT match
  unrelated numbers or text elsewhere in the document.
- Mark every regex with a confidence level: CONFIRMED (seen in >=2 paragraphs),
  TENTATIVE (seen in 1 paragraph only), PENDING (not seen yet).

Final output structure:

## Document Sections
[For each logical section: topic, approximate paragraph range, data fields found]

## Record Join Strategy
[Identify the primary record key observed in the document, such as ID,
reference code, registry number, or another stable key. Explain how fields from
separate sections should be merged by that key.]

## Fields Inventory
[For every field the question asks about, even if not yet seen:
  PENDING  | field_name | (not yet observed in any chunk)
  TENTATIVE| field_name | regex: <pattern> | example: <one full sentence>
  CONFIRMED| field_name | regex: <pattern> | examples: <2+ sentences>]

## PatternSpec
[Provide a JSON code block with the extraction patterns.  Include it ONLY when
ALL requested fields are at least TENTATIVE.  If any field is still PENDING,
note it and continue scanning.  The JSON must have this structure:
```json
{{
  "join_key_patterns": ["regex that captures the record ID"],
  "field_patterns": {{"field_name": ["regex that captures the field value"]}},
  "field_types": {{"field_name": "float|int|str"}},
  "null_values": ["None", "NaN", "-", ""],
  "correction_markers": ["corrected", "updated", "rectified", "confirmed"]
}}
```
Each regex must have exactly ONE capture group for the value.]

## Validation Warnings
[List known extraction risks: missing keys, duplicate keys, conflicting values,
unmatched field mentions, or fields seen without a join key.]
"""

TEMPLATE_PATTERN_ANALYSIS = """\
You are scanning a long document chunk by chunk in document order.  Your job is
to identify text patterns so that a deterministic engine can extract data
accurately.

CRITICAL: you are seeing only ONE chunk of the document right now.  Fields that
the <question> asks about may appear in earlier chunks (already recorded in
<patterns>) or in future chunks (not seen yet).  Do NOT assume a field doesn't
exist just because it is absent from this <section>.

Rules:
1. For each field you ACTUALLY observe in this <section>:
   - Copy one full original sentence as an example.
   - Write a regex with capture groups that extracts the value from that sentence.
   - Anchor the regex to surrounding marker words (e.g. "height is recorded as",
     "publisher affiliation is recorded as", "registered under the unique
     identifier").
   - Mark it CONFIRMED if you have seen it in >=2 paragraphs, otherwise TENTATIVE.
2. For fields in <patterns> that are already CONFIRMED, keep them as-is unless
   this <section> contradicts them.
3. For fields the <question> asks about but you have NOT observed in ANY chunk
   (including previous ones), keep them marked PENDING.
4. NEVER write a regex for a field you have not seen.  PENDING fields stay in the
   Fields Inventory with no regex.
5. Identify a stable record join key. Good join keys include "ID", "reference
   code", "registry number", or another repeated identifier.
6. Documents may split one entity across separate chapters.  The engine merges
   fields by join key paragraph-by-paragraph.  Do NOT extract independent lists
   and zip them together.
   - The engine finds the local join key in each paragraph, extracts fields
     present there, and merges them into records[join_key].
   - Do NOT suggest global nearest-ID matching or fixed character windows.
   - If a paragraph/sentence has no local join key, the engine skips its fields.
7. Update the PatternSpec section every turn.  The JSON block must contain
   join_key_patterns and field_patterns for CONFIRMED and TENTATIVE fields.
   Each regex must have exactly ONE capture group that extracts the value.
   If any requested field is still PENDING, note it in Validation Warnings.
8. Include validation warnings for missing keys, duplicate keys, conflicting
   values, unmatched field mentions, and field sentences that do not contain a
   usable join key.
9. For prose numeric fields, prefer flexible but locally anchored regexes over
   brittle verb lists. For example, match a field label, a short non-sentence
   gap, then a value and unit:
   `height[^.]{{0,160}}?(\\d+(?:\\.\\d+)?)\\s*(?:cm|centimeters)`.

<question>
{question}
</question>

<patterns>
{memory}
</patterns>

<section>
{chunk}
</section>

Output the complete updated <patterns> document with all five sections:
Document Sections, Record Join Strategy, Fields Inventory, PatternSpec,
and Validation Warnings.

The PatternSpec section must contain a JSON code block like this:
```json
{{
  "join_key_patterns": ["regex with one capture group for the record ID"],
  "field_patterns": {{"field_name": ["regex with one capture group for the value"]}},
  "field_types": {{"field_name": "float|int|str"}},
  "null_values": ["None", "NaN", "-", ""],
  "correction_markers": ["corrected", "updated", "rectified", "confirmed"]
}}
```

"""

TEMPLATE_PATTERN_JSON_FINAL = """\
You have scanned a document chunk by chunk and built a detailed extraction
guide.  Now produce a machine-readable PatternSpec JSON that a deterministic
engine can use to extract records from the FULL document text.

The engine works like this:
1. Split text into paragraphs (double-newline separated).
2. For each paragraph, find the join key (record ID) using join_key_patterns.
3. Extract field values from that same paragraph using field_patterns.
4. Merge all values by join key into records.
5. Type-convert fields according to field_types.

CRITICAL RULES:
- Each join_key_pattern must have EXACTLY ONE capture group for the ID value.
- Each field_pattern must have EXACTLY ONE capture group for the field value.
- Anchor patterns to surrounding marker words so they do NOT match unrelated
  numbers or text.
- Patterns must be reusable across ALL paragraphs, not just one example.
- If a field value is sometimes "None", "NaN", "0.0", or "-", add those to
  null_values so the engine treats them as missing.
- If you observed correction/update language (words like "corrected", "updated",
  "rectified", "confirmed", "adjusted"), include those in correction_markers.
- field_types: "float" for decimal numbers, "int" for integer IDs/codes,
  "str" for text values, "int|str" when both are possible.

Output ONLY the JSON block.  No explanation.

```json
{{
  "join_key_patterns": ["regex1", "regex2"],
  "field_patterns": {{"field_name": ["regex1", "regex2"]}},
  "field_types": {{"field_name": "float"}},
  "null_values": ["None", "NaN", "-", ""],
  "correction_markers": ["corrected", "updated", "rectified", "confirmed"]
}}
```

<question>
{question}
</question>

<patterns>
{memory}
</patterns>
"""

TEMPLATE_REPAIR_PATTERN_SPEC = """\
The deterministic extraction engine ran with your PatternSpec and produced
the diagnostics below.  Some fields have low coverage.  Please fix the
PatternSpec JSON so that extraction improves.

Engine Diagnostics:
{diagnostics_json}

Current PatternSpec:
{pattern_spec_json}

<question>
{question}
</question>

Common causes of low coverage:
1. Join key patterns are too narrow and miss some ID formats.
2. Field patterns are too specific and don't match all paragraph variants.
3. Field patterns capture the wrong group (e.g. a different number in the
   same sentence).
4. Patterns match placeholder/null values that should be excluded -- add
   them to null_values.
5. A required field pattern is missing entirely.

Output ONLY the updated JSON block.  No explanation.

```json
{{
  "join_key_patterns": [...],
  "field_patterns": {{...}},
  "field_types": {{...}},
  "null_values": [...],
  "correction_markers": [...]
}}
```
"""

__all__ = [
    "MemAgent",
    "MemAgentConfig",
    "MemAgentResult",
    "PatternSpec",
    "ExtractionDiagnostics",
    "extract_records_with_spec",
    "_build_llm_fn",
    "_extract_json_block",
    "_json_to_pattern_spec",
    "make_pattern_analyzer",
    "make_process_long_doc",
    "PATTERN_MEMORY_INIT",
    "TEMPLATE_PATTERN_ANALYSIS",
    "TEXT_DOCUMENT_SUFFIXES",
]


@dataclass
class MemAgentConfig:
    recurrent_max_context_len: int = RECURRENT_MAX_CONTEXT_LEN
    recurrent_chunk_size: int = RECURRENT_CHUNK_SIZE
    recurrent_chunk_overlap: int = RECURRENT_CHUNK_OVERLAP
    max_memory_tokens: int = MEMORY_MAX_TOKENS
    max_retries: int = 0
    sleep_between_calls: float = 0.0
    keep_trace: bool = True
    per_call_timeout_seconds: float = 180.0
    total_timeout_seconds: float = 300.0
    use_deterministic_engine: bool = True
    repair_rounds: int = 2
    min_field_coverage: float = 0.75
    emit_engine_records: bool = False


@dataclass
class MemoryStep:
    index: int
    chunk_start: int
    chunk_end: int
    memory: str


@dataclass
class MemAgentDiagnostics:
    original_token_count: int = 0
    processed_token_count: int = 0
    chunk_count: int = 0
    token_coverage: float = 0.0
    complete_scan: bool = True
    timed_out: bool = False
    timeout_message: str | None = None
    pattern_spec_parse_failed: bool = False
    repair_rounds_used: int = 0
    engine_diagnostics: dict | None = None


@dataclass
class MemAgentResult:
    question: str
    answer: str
    steps: List[MemoryStep] = field(default_factory=list)
    diagnostics: MemAgentDiagnostics = field(default_factory=MemAgentDiagnostics)
    pattern_spec: Optional[object] = None
    engine_diagnostics: Optional[object] = None
    engine_records: Optional[list[dict]] = None


@dataclass
class PatternSpec:
    """Structured extraction specification produced by the LLM.

    The deterministic engine consumes this to extract records from full text
    without code generation.
    """

    join_key_patterns: list[str] = field(default_factory=list)
    field_patterns: dict[str, list[str]] = field(default_factory=dict)
    field_types: dict[str, str] = field(default_factory=dict)
    null_values: list[str] = field(default_factory=lambda: ["None", "NaN", "-", ""])
    correction_markers: list[str] = field(
        default_factory=lambda: [
            "corrected",
            "updated",
            "rectified",
            "confirmed",
            "adjusted",
            "amended",
        ]
    )


@dataclass
class ExtractionDiagnostics:
    """Diagnostics from a deterministic engine run."""

    record_count: int = 0
    field_coverage: dict[str, float] = field(default_factory=dict)
    field_hit_counts: dict[str, int] = field(default_factory=dict)
    orphan_field_sentences: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    unmatched_samples: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class TextChunk:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ParagraphSpan:
    index: int
    start: int
    end: int
    text: str
    tokens: tuple[int, ...]


def _get_tiktoken_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TIKTOKEN_ENCODING_NAME)


def _truncate_text_to_max_tokens(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = encoding.encode(text)
    if len(ids) <= max_tokens:
        return text
    return encoding.decode(ids[:max_tokens])


def extract_records_with_spec(
    text: str,
    spec: PatternSpec,
) -> tuple[list[dict], ExtractionDiagnostics]:
    """Deterministic extraction engine driven by a PatternSpec.

    Scans *text* paragraph-by-paragraph, finds record join keys via
    ``spec.join_key_patterns``, extracts field values via
    ``spec.field_patterns``, and merges records by join key.

    Returns ``(records, diagnostics)`` where *records* is a list of dicts and
    *diagnostics* captures coverage, orphans, and conflicts.
    """

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    records: dict[str, dict[str, Any]] = {}
    diagnostics = ExtractionDiagnostics()

    join_regexes = [re.compile(p, re.I) for p in spec.join_key_patterns]
    field_regex_map: dict[str, list[re.Pattern]] = {}
    for fname, patterns in spec.field_patterns.items():
        field_regex_map[fname] = [re.compile(p, re.I) for p in patterns]

    null_lower = {v.lower() for v in spec.null_values}

    def _is_null(raw: str) -> bool:
        return raw.strip().lower() in null_lower

    def _record_conflict(rid: str, fname: str, old: object, new: object) -> None:
        diagnostics.conflicts.append(
            {"id": rid, "field": fname, "old_value": old, "new_value": new}
        )
        if len(diagnostics.conflicts) > 100:
            diagnostics.conflicts.pop(0)

    for para in paragraphs:
        # --- locate all join keys in this paragraph --------------------------------
        id_positions: list[tuple[str, int]] = []
        for regex in join_regexes:
            for m in regex.finditer(para):
                id_positions.append((m.group(1).strip(), m.start()))

        # --- locate all field values with their positions --------------------------
        field_matches: dict[str, list[tuple[str, int]]] = {}
        for fname, regexes in field_regex_map.items():
            fmatches: list[tuple[str, int]] = []
            for regex in regexes:
                for m in regex.finditer(para):
                    val = m.group(m.lastindex or 0).strip()
                    fmatches.append((val, m.start()))
            if fmatches:
                field_matches[fname] = fmatches

        # --- no join key in this paragraph → track as orphan / unmatched ----------
        if not id_positions:
            if field_matches:
                diagnostics.orphan_field_sentences.append(para[:200])
                if len(diagnostics.orphan_field_sentences) > 50:
                    diagnostics.orphan_field_sentences.pop(0)
            continue

        # --- best-effort field value per field (last positional match wins) -------
        field_best: dict[str, str] = {}
        for fname, matches in field_matches.items():
            matches.sort(key=lambda x: x[1])
            field_best[fname] = matches[-1][0]

        # --- merge into records ---------------------------------------------------
        for rid, _pos in id_positions:
            if rid not in records:
                records[rid] = {"id": rid}

            for fname, raw_val in field_best.items():
                if _is_null(raw_val):
                    continue

                existing = records[rid].get(fname)
                if existing is not None and str(existing) != raw_val:
                    _record_conflict(rid, fname, existing, raw_val)

                records[rid][fname] = raw_val

    # --- type conversion ----------------------------------------------------------
    for rid, record in records.items():
        # convert the record id to int when it looks numeric
        rid_int = None
        try:
            rid_int = int(rid)
        except (ValueError, TypeError):
            pass
        if rid_int is not None:
            record["id"] = rid_int

        for fname, type_hint in spec.field_types.items():
            if fname not in record:
                continue
            raw = record[fname]
            try:
                if type_hint == "float":
                    record[fname] = float(raw)
                elif type_hint == "int":
                    record[fname] = int(float(raw))
                elif type_hint == "int|str":
                    try:
                        record[fname] = int(float(raw))
                    except (ValueError, TypeError):
                        pass
            except (ValueError, TypeError):
                pass

    # --- compute coverage diagnostics ---------------------------------------------
    diagnostics.record_count = len(records)
    if records:
        for fname in spec.field_patterns:
            hits = sum(
                1
                for r in records.values()
                if r.get(fname) is not None and r.get(fname) != ""
            )
            diagnostics.field_coverage[fname] = hits / len(records)
            diagnostics.field_hit_counts[fname] = hits
    else:
        for fname in spec.field_patterns:
            diagnostics.field_coverage[fname] = 0.0
            diagnostics.field_hit_counts[fname] = 0

    return list(records.values()), diagnostics


def _extract_json_block(markdown: str) -> dict | None:
    """Return the parsed contents of the first ```json fenced block in *markdown*."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", markdown, flags=re.I | re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _json_to_pattern_spec(data: dict) -> PatternSpec | None:
    """Convert a parsed JSON dict into a :class:`PatternSpec`, with validation."""
    try:
        join_key_patterns = list(data.get("join_key_patterns", []))
        if not join_key_patterns:
            return None
        field_patterns: dict[str, list[str]] = {}
        for fname, pats in data.get("field_patterns", {}).items():
            if isinstance(pats, list):
                field_patterns[str(fname)] = [str(p) for p in pats]
            else:
                field_patterns[str(fname)] = [str(pats)]
        field_types: dict[str, str] = {}
        for fname, ft in data.get("field_types", {}).items():
            field_types[str(fname)] = str(ft)
        null_values = list(data.get("null_values", ["None", "NaN", "-", ""]))
        correction_markers = list(
            data.get(
                "correction_markers",
                ["corrected", "updated", "rectified", "confirmed", "adjusted", "amended"],
            )
        )
        return PatternSpec(
            join_key_patterns=join_key_patterns,
            field_patterns=field_patterns,
            field_types=field_types,
            null_values=null_values,
            correction_markers=correction_markers,
        )
    except (TypeError, KeyError, ValueError):
        return None


def _extraction_diagnostics_to_dict(diag: ExtractionDiagnostics) -> dict:
    return {
        "record_count": diag.record_count,
        "field_coverage": dict(diag.field_coverage),
        "field_hit_counts": dict(diag.field_hit_counts),
        "orphan_field_count": len(diag.orphan_field_sentences),
        "conflict_count": len(diag.conflicts),
        "conflict_samples": diag.conflicts[:10],
        "error": diag.error,
    }


def _build_llm_fn(model: object, timeout_seconds: float = 120.0) -> LLMFn:
    """Wrap a langchain BaseChatModel into the simple str->str interface MemAgent expects.

    Each call runs in a thread with a per-call timeout; if the model hangs the
    call is interrupted and the exception propagates to the retry loop.
    """

    def _llm(prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        def _call() -> str:
            response = model.invoke([HumanMessage(content=prompt)])
            content = response.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text_parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
                joined = "".join(text_parts)
                if joined.strip():
                    return joined
            return ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            try:
                result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"LLM call exceeded {timeout_seconds}s timeout"
                )
            return result

    return _llm


class MemAgent:
    def __init__(self, llm: LLMFn, config: Optional[MemAgentConfig] = None) -> None:
        self.llm = llm
        self.config = config or MemAgentConfig()
        self._encoding = _get_tiktoken_encoding()

    def build_task_context(self, question: str, document: str) -> MemAgentResult:
        memory = NO_MEMORY
        steps: List[MemoryStep] = []
        started_at = time.monotonic()
        total_timeout = self.config.total_timeout_seconds
        chunks = list(self._chunk_text(document))
        diagnostics = self._initial_diagnostics(document, chunks)
        last_processed_end = 0

        for idx, chunk_info in enumerate(chunks, start=1):
            elapsed = time.monotonic() - started_at
            if elapsed >= total_timeout:
                diagnostics.timed_out = True
                diagnostics.complete_scan = False
                diagnostics.timeout_message = (
                    "MemAgent: total timeout reached, returning partial results."
                )
                memory += f"\n\n[{diagnostics.timeout_message}]"
                break

            prompt = self._build_context_update_prompt(
                question=question,
                memory=memory,
                chunk=chunk_info.text,
            )
            memory = self._call_with_retry(prompt)
            memory = self._clean_context_memory(memory)
            memory = _truncate_text_to_max_tokens(
                memory, self._encoding, self.config.max_memory_tokens
            )
            last_processed_end = max(last_processed_end, chunk_info.end)

            if self.config.keep_trace:
                steps.append(
                    MemoryStep(
                        index=idx,
                        chunk_start=chunk_info.start,
                        chunk_end=chunk_info.end,
                        memory=memory,
                    )
                )

            if self.config.sleep_between_calls:
                time.sleep(self.config.sleep_between_calls)

        answer = memory.strip()
        if answer == NO_MEMORY:
            answer = ""

        return MemAgentResult(
            question=question,
            answer=answer,
            steps=steps,
            diagnostics=self._finalize_diagnostics(diagnostics, last_processed_end),
        )

    def build_extraction_patterns(self, question: str, document: str) -> MemAgentResult:
        """Scan the document chunk-by-chunk to identify text patterns and produce
        regex/Python extraction code, NOT the extracted data itself.

        The iteratively accumulated memory guides the LLM to record field locations,
        surrounding text patterns, and eventually complete extraction code.
        """
        memory = PATTERN_MEMORY_INIT
        steps: List[MemoryStep] = []
        started_at = time.monotonic()
        total_timeout = self.config.total_timeout_seconds
        chunks, diagnostics = self._prepare_pattern_chunks(question, document)
        last_processed_end = 0

        for idx, chunk_info in enumerate(chunks, start=1):
            elapsed = time.monotonic() - started_at
            if elapsed >= total_timeout:
                diagnostics.timed_out = True
                diagnostics.complete_scan = False
                diagnostics.timeout_message = (
                    "MemAgent: total timeout reached, returning partial patterns."
                )
                memory += f"\n\n[{diagnostics.timeout_message}]"
                break

            prompt = TEMPLATE_PATTERN_ANALYSIS.format(
                question=question,
                memory=memory,
                chunk=chunk_info.text,
            )
            memory = self._call_with_retry(prompt)
            memory = self._clean_context_memory(memory)
            memory = _truncate_text_to_max_tokens(
                memory, self._encoding, self.config.max_memory_tokens
            )
            last_processed_end = max(last_processed_end, chunk_info.end)

            if self.config.keep_trace:
                steps.append(
                    MemoryStep(
                        index=idx,
                        chunk_start=chunk_info.start,
                        chunk_end=chunk_info.end,
                        memory=memory,
                    )
                )

            if self.config.sleep_between_calls:
                time.sleep(self.config.sleep_between_calls)

        answer = memory.strip()
        if answer == PATTERN_MEMORY_INIT.strip():
            answer = ""

        # --- deterministic engine path --------------------------------------------
        pattern_spec: PatternSpec | None = None
        engine_diag: ExtractionDiagnostics | None = None
        engine_records: list[dict] | None = None

        if self.config.use_deterministic_engine and answer:
            json_data = _extract_json_block(answer)

            # If the scanning loop did not produce a JSON block, make one
            # final conversion call.
            if json_data is None:
                try:
                    final_prompt = TEMPLATE_PATTERN_JSON_FINAL.format(
                        question=question,
                        memory=answer,
                    )
                    final_output = self._call_with_retry(final_prompt)
                    json_data = _extract_json_block(final_output)
                    if json_data is None:
                        # Also try raw JSON (no markdown fences)
                        try:
                            json_data = json.loads(final_output.strip())
                        except json.JSONDecodeError:
                            json_data = None
                except Exception:
                    json_data = None

            if json_data is not None:
                pattern_spec = _json_to_pattern_spec(json_data)
                if pattern_spec is not None:
                    try:
                        engine_records, engine_diag = extract_records_with_spec(
                            document, pattern_spec
                        )

                        # --- repair loop ------------------------------------------
                        for round_idx in range(self.config.repair_rounds):
                            needs_repair = any(
                                engine_diag.field_coverage.get(fname, 0.0)
                                < self.config.min_field_coverage
                                for fname in pattern_spec.field_patterns
                            )
                            if not needs_repair:
                                break

                            try:
                                repair_prompt = TEMPLATE_REPAIR_PATTERN_SPEC.format(
                                    diagnostics_json=json.dumps(
                                        _extraction_diagnostics_to_dict(engine_diag),
                                        indent=2,
                                    ),
                                    pattern_spec_json=json.dumps(
                                        asdict(pattern_spec), indent=2
                                    ),
                                    question=question,
                                )
                                repair_output = self._call_with_retry(repair_prompt)
                                repair_json = _extract_json_block(repair_output)
                                if repair_json is not None:
                                    repair_spec = _json_to_pattern_spec(repair_json)
                                    if repair_spec is not None:
                                        pattern_spec = repair_spec
                                        engine_records, engine_diag = (
                                            extract_records_with_spec(
                                                document, pattern_spec
                                            )
                                        )
                                        diagnostics.repair_rounds_used = round_idx + 1
                            except Exception:
                                pass

                        diagnostics.engine_diagnostics = (
                            _extraction_diagnostics_to_dict(engine_diag)
                        )
                    except Exception as exc:
                        diagnostics.engine_diagnostics = {
                            "error": str(exc),
                            "record_count": 0,
                        }
                else:
                    diagnostics.pattern_spec_parse_failed = True
            else:
                diagnostics.pattern_spec_parse_failed = True

        return MemAgentResult(
            question=question,
            answer=answer,
            steps=steps,
            diagnostics=self._finalize_diagnostics(diagnostics, last_processed_end),
            pattern_spec=pattern_spec,
            engine_diagnostics=engine_diag,
            engine_records=(
                engine_records if self.config.emit_engine_records else None
            ),
        )

    def _chunk_text(self, text: str) -> Iterable[TextChunk]:
        max_len = self.config.recurrent_max_context_len
        size = self.config.recurrent_chunk_size
        overlap = max(0, self.config.recurrent_chunk_overlap)

        if max_len <= 0:
            raise ValueError("recurrent_max_context_len must be > 0")
        if size <= 0:
            raise ValueError("recurrent_chunk_size must be > 0")

        target_size = min(size, max_len)
        overlap = min(overlap, max(target_size - 1, 0))
        paragraph_chunks = self._paragraph_token_chunks(text)
        current: list[int] = []
        current_start = 0
        current_end = 0

        for para_start, para_end, para_tokens in paragraph_chunks:
            if len(para_tokens) > target_size:
                if current:
                    yield self._make_chunk(current_start, current_end, current)
                    current = []
                for start in range(0, len(para_tokens), target_size):
                    part = para_tokens[start : start + target_size]
                    yield TextChunk(
                        start=para_start + start,
                        end=para_start + start + len(part),
                        text=self._decode_tokens(part),
                    )
                current_start = para_end
                current_end = para_end
                continue

            if current and len(current) + len(para_tokens) > target_size:
                yield self._make_chunk(current_start, current_end, current)
                prefix = current[-overlap:] if overlap else []
                current = [*prefix, *para_tokens]
                current_start = max(current_end - len(prefix), 0)
                current_end = para_end
            else:
                if not current:
                    current_start = para_start
                current.extend(para_tokens)
                current_end = para_end

        if current:
            yield TextChunk(
                start=current_start,
                end=current_end,
                text=self._decode_tokens(current),
            )

    def _prepare_pattern_chunks(
        self,
        question: str,
        document: str,
    ) -> tuple[list[TextChunk], MemAgentDiagnostics]:
        full_chunks = list(self._chunk_text(document))
        diagnostics = self._initial_diagnostics(document, full_chunks)
        return full_chunks, diagnostics

    def _paragraph_spans(self, text: str) -> list[ParagraphSpan]:
        pieces = re.split(r"(\n\s*\n)", text)
        spans: list[ParagraphSpan] = []
        token_cursor = 0
        pending = ""

        for piece in pieces:
            if piece == "":
                continue
            pending += piece
            if re.fullmatch(r"\n\s*\n", piece):
                tokens = tuple(self._encode_text(pending))
                spans.append(
                    ParagraphSpan(
                        index=len(spans),
                        start=token_cursor,
                        end=token_cursor + len(tokens),
                        text=pending.strip(),
                        tokens=tokens,
                    )
                )
                token_cursor += len(tokens)
                pending = ""

        if pending:
            tokens = tuple(self._encode_text(pending))
            spans.append(
                ParagraphSpan(
                    index=len(spans),
                    start=token_cursor,
                    end=token_cursor + len(tokens),
                    text=pending.strip(),
                    tokens=tokens,
                )
            )
        return spans

    def _paragraph_token_chunks(self, text: str) -> list[tuple[int, int, list[int]]]:
        return [(span.start, span.end, list(span.tokens)) for span in self._paragraph_spans(text)]

    def _make_chunk(
        self,
        start: int,
        end: int,
        tokens: Sequence[int],
    ) -> TextChunk:
        return TextChunk(start=start, end=end, text=self._decode_tokens(tokens))

    def _initial_diagnostics(
        self,
        document: str,
        chunks: Sequence[TextChunk],
    ) -> MemAgentDiagnostics:
        original_token_count = len(self._encode_text(document))
        complete_scan = not chunks or (chunks[0].start == 0 and chunks[-1].end >= original_token_count)
        return MemAgentDiagnostics(
            original_token_count=original_token_count,
            processed_token_count=original_token_count if complete_scan else 0,
            chunk_count=len(chunks),
            token_coverage=1.0 if complete_scan or original_token_count == 0 else 0.0,
            complete_scan=complete_scan,
        )

    def _finalize_diagnostics(
        self,
        diagnostics: MemAgentDiagnostics,
        last_processed_end: int,
    ) -> MemAgentDiagnostics:
        if diagnostics.original_token_count == 0:
            diagnostics.processed_token_count = 0
            diagnostics.token_coverage = 1.0
            diagnostics.complete_scan = not diagnostics.timed_out
            return diagnostics

        if diagnostics.timed_out:
            diagnostics.processed_token_count = min(
                last_processed_end,
                diagnostics.original_token_count,
            )
            diagnostics.token_coverage = (
                diagnostics.processed_token_count / diagnostics.original_token_count
            )
            diagnostics.complete_scan = False
        return diagnostics

    def _encode_text(self, text: str) -> list[int]:
        return list(self._encoding.encode(text))

    def _decode_tokens(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))

    def _build_context_update_prompt(
        self,
        *,
        question: str,
        memory: str,
        chunk: str,
    ) -> str:
        mem = memory.strip() if memory.strip() else NO_MEMORY
        return TEMPLATE_CONTEXT_UPDATE.format(
            question=question,
            memory=mem,
            chunk=chunk,
        )

    def _call_with_retry(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                out = self.llm(prompt)
                if not isinstance(out, str):
                    raise TypeError("llm(prompt) must return a string")
                if out.strip():
                    return out
            except TimeoutError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    time.sleep(0.5)
        raise RuntimeError(f"LLM call failed after retries: {last_error}")

    def _clean_context_memory(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^Updated task context:\s*", "", text, flags=re.I).strip()
        return text


def make_process_long_doc(
    llm: LLMFn,
    *,
    recurrent_max_context_len: int = RECURRENT_MAX_CONTEXT_LEN,
    recurrent_chunk_size: int = RECURRENT_CHUNK_SIZE,
    max_memory_tokens: int = MEMORY_MAX_TOKENS,
    per_call_timeout_seconds: float = 180.0,
    total_timeout_seconds: float = 300.0,
    keep_trace: bool = False,
) -> Callable[[str, Path], dict[str, object]]:
    agent = MemAgent(
        llm,
        config=MemAgentConfig(
            recurrent_max_context_len=recurrent_max_context_len,
            recurrent_chunk_size=recurrent_chunk_size,
            max_memory_tokens=max_memory_tokens,
            per_call_timeout_seconds=per_call_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            keep_trace=keep_trace,
        ),
    )

    def process_long_doc(question: str, doc_path: Path) -> dict[str, object]:
        if not doc_path.is_file():
            raise ValueError(f"Path is not a file: {doc_path}")
        if doc_path.suffix.lower() not in TEXT_DOCUMENT_SUFFIXES:
            raise ValueError(f"Unsupported document type: {doc_path}")

        document = doc_path.read_text(encoding="utf-8", errors="replace")
        effective_question = question.strip()
        if not document.strip():
            diagnostics = MemAgentDiagnostics(original_token_count=0, token_coverage=1.0)
            return {
                "path": str(doc_path),
                "question": effective_question,
                "answer": "",
                "chunk_count": 0,
                "diagnostics": _diagnostics_to_dict(diagnostics),
            }

        result = agent.build_task_context(effective_question, document)
        content: dict[str, object] = {
            "path": str(doc_path),
            "question": effective_question,
            "answer": result.answer,
            "diagnostics": _diagnostics_to_dict(result.diagnostics),
        }
        if keep_trace:
            content["chunk_count"] = len(result.steps)
            content["steps"] = [
                {
                    "index": step.index,
                    "chunk_start": step.chunk_start,
                    "chunk_end": step.chunk_end,
                    "memory": step.memory,
                }
                for step in result.steps
            ]
        return content

    return process_long_doc


def make_pattern_analyzer(
    llm: LLMFn,
    *,
    recurrent_max_context_len: int = RECURRENT_MAX_CONTEXT_LEN,
    recurrent_chunk_size: int = RECURRENT_CHUNK_SIZE,
    max_memory_tokens: int = MEMORY_MAX_TOKENS,
    per_call_timeout_seconds: float = 180.0,
    total_timeout_seconds: float = 300.0,
    keep_trace: bool = False,
    use_deterministic_engine: bool = True,
    repair_rounds: int = 2,
    min_field_coverage: float = 0.75,
    emit_engine_records: bool = False,
) -> Callable[[str, Path], dict[str, object]]:
    """Build a pattern-analyzer that scans a long document and returns regex/Python
    extraction guidance instead of extracted data.

    The returned callable has the same signature as `make_process_long_doc`:
        (question: str, doc_path: Path) -> dict[str, object]
    """
    agent = MemAgent(
        llm,
        config=MemAgentConfig(
            recurrent_max_context_len=recurrent_max_context_len,
            recurrent_chunk_size=recurrent_chunk_size,
            max_memory_tokens=max_memory_tokens,
            per_call_timeout_seconds=per_call_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            keep_trace=keep_trace,
            use_deterministic_engine=use_deterministic_engine,
            repair_rounds=repair_rounds,
            min_field_coverage=min_field_coverage,
            emit_engine_records=emit_engine_records,
        ),
    )

    def analyze_patterns(question: str, doc_path: Path) -> dict[str, object]:
        if not doc_path.is_file():
            raise ValueError(f"Path is not a file: {doc_path}")
        if doc_path.suffix.lower() not in TEXT_DOCUMENT_SUFFIXES:
            raise ValueError(f"Unsupported document type: {doc_path}")

        document = doc_path.read_text(encoding="utf-8", errors="replace")
        effective_question = question.strip()
        if not document.strip():
            diagnostics = MemAgentDiagnostics(original_token_count=0, token_coverage=1.0)
            return {
                "path": str(doc_path),
                "question": effective_question,
                "answer": "",
                "chunk_count": 0,
                "diagnostics": _diagnostics_to_dict(diagnostics),
            }

        result = agent.build_extraction_patterns(effective_question, document)
        content: dict[str, object] = {
            "path": str(doc_path),
            "question": effective_question,
            "answer": result.answer,
            "diagnostics": _diagnostics_to_dict(result.diagnostics),
        }
        if result.pattern_spec is not None:
            content["pattern_spec"] = asdict(result.pattern_spec)
        if result.engine_diagnostics is not None:
            content["engine_diagnostics"] = _extraction_diagnostics_to_dict(
                result.engine_diagnostics
            )
        if result.engine_records is not None:
            content["engine_records"] = result.engine_records
        if keep_trace:
            content["chunk_count"] = len(result.steps)
            content["steps"] = [
                {
                    "index": step.index,
                    "chunk_start": step.chunk_start,
                    "chunk_end": step.chunk_end,
                    "memory": step.memory,
                }
                for step in result.steps
            ]
        return content

    return analyze_patterns


def _diagnostics_to_dict(diagnostics: MemAgentDiagnostics) -> dict[str, object]:
    return {
        "original_token_count": diagnostics.original_token_count,
        "processed_token_count": diagnostics.processed_token_count,
        "chunk_count": diagnostics.chunk_count,
        "token_coverage": diagnostics.token_coverage,
        "complete_scan": diagnostics.complete_scan,
        "timed_out": diagnostics.timed_out,
        "timeout_message": diagnostics.timeout_message,
        "pattern_spec_parse_failed": diagnostics.pattern_spec_parse_failed,
        "repair_rounds_used": diagnostics.repair_rounds_used,
        "engine_diagnostics": diagnostics.engine_diagnostics,
    }
