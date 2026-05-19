"""Debug MemAgent on task_396 with detailed diagnostics.

Usage:
    uv run python scripts/debug_memagent_task396.py
    uv run python scripts/debug_memagent_task396.py --direct  # skip trace, use direct args
    uv run python scripts/debug_memagent_task396.py --oracle-only  # only run oracle
    uv run python scripts/debug_memagent_task396.py --verbose  # print LLM prompts/responses
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import create_chat_model
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.tools.memagent import (
    MEMORY_MAX_TOKENS,
    PATTERN_MEMORY_INIT,
    RECURRENT_CHUNK_SIZE,
    TEMPLATE_PATTERN_ANALYSIS,
    TEMPLATE_PATTERN_JSON_FINAL,
    TEMPLATE_REPAIR_PATTERN_SPEC,
    MemAgent,
    MemAgentConfig,
    PatternSpec,
    _build_llm_fn,
    _extract_json_block,
    _json_to_pattern_spec,
    extract_records_with_spec,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import ToolRuntimeContext, create_default_tool_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = PROJECT_ROOT / "artifacts" / "runs" / "20260519T034838Z" / "task_396" / "trace.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "easy.yaml"
DEFAULT_TASK_ID = "task_396"
EXPECTED_DENOMINATOR = 31
EXPECTED_MARVEL_COUNT = 16
EXPECTED_PERCENTAGE = 51.61290322580645


# ── banner helpers ───────────────────────────────────────────────────────────


def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _sub_banner(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── trace loading ────────────────────────────────────────────────────────────


def _load_memagent_args(trace_path: Path) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for step in trace.get("steps", []):
        if step.get("step_index") != 23:
            continue
        for call in step.get("tool_calls", []):
            if call.get("name") == "memagent":
                args = call.get("args", {})
                if isinstance(args, dict):
                    return dict(args)
    raise ValueError(f"Could not find memagent call at step 23 in {trace_path}")


# ── task 396 oracle ──────────────────────────────────────────────────────────


def _first_int(patterns: list[str], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return int(match.group(1))
    return None


def _task396_oracle(document: str) -> dict[str, Any]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document) if p.strip()]
    id_patterns = [
        r"(?:registered under the unique identifier|registered under identifier|registered with identifier|registered under ID|registered at ID|registered under the ID|registered under the unique registration number|registered under the identifier|registered under registry number|cataloged under reference code|cataloged with registry number|cataloged with identifier|cataloged under identifier|cataloged with reference number|cataloged with reference ID|cataloged under reference ID|cataloged under the reference code|cataloged with the identifier|tracked with identifier|tracked under identifier|tracked with registry number|tracked under registry number|tracked under reference ID|tracked under ID|tracked with the identifier|filed under ID|filed under identifier|filed under the unique identifier|filed under the reference ID|identified by registry number|identified by the registry number|registered with ID|registered with the ID|cataloged at ID|maintained under ID|Entry)\s*(\d+)",
        r"\(Registry Ref:\s*(\d+)\)",
        r"whose entry is referenced by ID\s*(\d+)",
        r"Subject\s*(\d+)",
    ]
    publisher_patterns = [
        r"publisher affiliation (?:is )?(?:logged with the code|logged as|recorded as|recorded|is|of)\s*(\d+)",
        r"primary publisher affiliation is logged with the code\s*(\d+)",
        r"file lists (?:his|her|its|their)?\s*publisher affiliation as\s*(\d+)",
        r"lists (?:his|her|its|their)?\s*publisher affiliation as\s*(\d+)",
        r"identifies (?:his|her|its|their)?\s*publisher affiliation as\s*(\d+)",
        r"affiliated with publisher\s*(\d+)",
        r"under the jurisdiction of publisher\s*(\d+)",
        r"classified under publisher\s*(\d+)",
        r"registered with publisher\s*(\d+)",
        r"on record with publisher\s*(\d+)",
        r"is on file with publisher\s*(\d+)",
        r"listed with publisher\s*(\d+)",
        r"documented under the oversight of publisher\s*(\d+)",
        r"designated with a publisher affiliation of\s*(\d+)",
        r"publisher affiliation was .*?rectified to .*?code of\s*(\d+)",
        r"publisher affiliation was confirmed as\s*(\d+)",
        r"publisher affiliation.*?confirmed as\s*(\d+)",
    ]

    heights: dict[int, float] = {}
    publishers: dict[int, int] = {}
    for paragraph in paragraphs:
        record_id = _first_int(id_patterns, paragraph)
        if record_id is None:
            continue

        height_values = [
            float(value)
            for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:centimeters|cm)\b", paragraph, flags=re.I)
        ]
        if height_values:
            heights[record_id] = height_values[-1]

        publisher_id = _first_int(publisher_patterns, paragraph)
        if publisher_id is not None:
            publishers[record_id] = publisher_id

    height_filtered_ids = sorted(
        record_id for record_id, height in heights.items() if 150 <= height <= 180
    )
    marvel_ids = [record_id for record_id in height_filtered_ids if publishers.get(record_id) == 13]
    denominator = len(height_filtered_ids)
    marvel_count = len(marvel_ids)
    percentage = marvel_count / denominator * 100 if denominator else None
    return {
        "denominator": denominator,
        "marvel_count": marvel_count,
        "percentage": percentage,
        "height_filtered_ids": height_filtered_ids,
        "marvel_ids": marvel_ids,
        "passed": (
            denominator == EXPECTED_DENOMINATOR
            and marvel_count == EXPECTED_MARVEL_COUNT
            and abs((percentage or 0.0) - EXPECTED_PERCENTAGE) < 1e-12
        ),
    }


def _task396_from_records(records: list[Any]) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = [record for record in records if isinstance(record, dict)]

    def as_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def as_int(value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    height_filtered: list[Any] = []
    marvel: list[Any] = []
    for record in parsed:
        height = as_float(record.get("height_cm") or record.get("height"))
        publisher_id = as_int(record.get("publisher_id") or record.get("publisher"))
        record_id = record.get("id") or record.get("join_key") or record.get("record_id")
        if height is None or not (150 <= height <= 180):
            continue
        height_filtered.append(record_id)
        if publisher_id == 13:
            marvel.append(record_id)

    denominator = len(height_filtered)
    marvel_count = len(marvel)
    percentage = marvel_count / denominator * 100 if denominator else None
    return {
        "denominator": denominator,
        "marvel_count": marvel_count,
        "percentage": percentage,
        "height_filtered_ids": height_filtered,
        "marvel_ids": marvel,
        "matches_oracle": (
            denominator == EXPECTED_DENOMINATOR
            and marvel_count == EXPECTED_MARVEL_COUNT
            and percentage is not None
            and abs(percentage - EXPECTED_PERCENTAGE) < 1e-12
        ),
    }


# ── file I/O ─────────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ── verbose LLM wrapper ──────────────────────────────────────────────────────


def _make_verbose_llm(llm_fn, output_dir: Path) -> Any:
    """Wrap an LLM function to log every prompt and response."""
    prompt_dir = output_dir / "llm_traces"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    call_counter = [0]

    def verbose_llm(prompt: str) -> str:
        call_counter[0] += 1
        idx = call_counter[0]
        prompt_preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
        print(f"  [LLM call #{idx}] prompt: {prompt_preview}")

        t0 = time.monotonic()
        response = llm_fn(prompt)
        elapsed = time.monotonic() - t0

        response_preview = response[:300] + "..." if len(response) > 300 else response
        print(f"  [LLM call #{idx}] response ({elapsed:.1f}s): {response_preview}")

        # Save full prompt/response trace
        trace_file = prompt_dir / f"call_{idx:03d}.json"
        _write_json(
            trace_file,
            {
                "call_index": idx,
                "prompt": prompt,
                "response": response,
                "elapsed_seconds": round(elapsed, 2),
                "prompt_tokens_approx": len(prompt) // 4,
                "response_tokens_approx": len(response) // 4,
            },
        )
        if not response.strip():
            print(f"  [LLM call #{idx}] WARNING: empty response!")
        return response

    return verbose_llm


# ── main debugging flow ──────────────────────────────────────────────────────


def _print_spec_summary(spec: PatternSpec) -> None:
    """Print a concise summary of a PatternSpec."""
    print(f"  join_key_patterns ({len(spec.join_key_patterns)}):")
    for p in spec.join_key_patterns[:5]:
        print(f"    {p}")
    if len(spec.join_key_patterns) > 5:
        print(f"    ... ({len(spec.join_key_patterns) - 5} more)")

    print(f"  field_patterns ({len(spec.field_patterns)} fields):")
    for fname, pats in spec.field_patterns.items():
        print(f"    {fname}: {len(pats)} pattern(s)")
        for p in pats[:2]:
            print(f"      {p}")
        if len(pats) > 2:
            print(f"      ... ({len(pats) - 2} more)")

    print(f"  field_types: {spec.field_types}")
    print(f"  null_values: {spec.null_values}")
    print(f"  correction_markers: {spec.correction_markers}")


def run_debug(
    *,
    config_path: Path,
    trace_path: Path | None,
    task_id: str,
    verbose: bool = False,
    oracle_only: bool = False,
    use_deterministic_engine: bool = True,
    repair_rounds: int = 2,
    total_timeout_seconds: float = 900.0,
    per_call_timeout_seconds: float = 240.0,
    recurrent_chunk_size: int = 8192,
    max_memory_tokens: int = 4096,
    emit_engine_records: bool = True,
    direct_path: str | None = None,
    direct_question: str | None = None,
) -> None:
    config = load_app_config(config_path)
    dataset = DABenchPublicDataset(config.dataset.root_path)
    task = dataset.get_task(task_id)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = PROJECT_ROOT / "artifacts" / "debug" / "memagent" / task_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve memagent args ─────────────────────────────────────────────
    if direct_path and direct_question:
        memagent_args: dict[str, Any] = {
            "path": direct_path,
            "question": direct_question,
        }
        print(f"Using direct args: path={direct_path}")
    elif trace_path is not None and trace_path.exists():
        memagent_args = _load_memagent_args(trace_path)
        print(f"Loaded memagent args from trace: path={memagent_args.get('path')}")
    else:
        print("ERROR: No trace file or direct args provided.")
        sys.exit(1)

    memagent_args.update({
        "keep_trace": True,
        "total_timeout_seconds": total_timeout_seconds,
        "per_call_timeout_seconds": per_call_timeout_seconds,
        "recurrent_chunk_size": recurrent_chunk_size,
        "max_memory_tokens": max_memory_tokens,
        "use_deterministic_engine": use_deterministic_engine,
        "repair_rounds": repair_rounds,
        "emit_engine_records": emit_engine_records,
    })
    _write_json(output_dir / "memagent_args.json", memagent_args)

    # ── load document ─────────────────────────────────────────────────────
    document_path = task.context_dir / str(memagent_args["path"])
    document = document_path.read_text(encoding="utf-8", errors="replace")
    question = str(memagent_args["question"])

    _banner(f"Debug MemAgent — {task_id}")
    print(f"  Document: {document_path}")
    print(f"  Document length: {len(document)} chars, ~{len(document)//4} tokens")
    print(f"  Question: {textwrap.shorten(question, width=160)}")
    print(f"  Output dir: {output_dir}")

    # ── oracle baseline ───────────────────────────────────────────────────
    _sub_banner("Task 396 Oracle (ground truth)")
    oracle = _task396_oracle(document)
    _write_json(output_dir / "task396_oracle.json", oracle)
    print(f"  denominator (150≤h≤180): {oracle['denominator']}")
    print(f"  marvel (publisher=13):   {oracle['marvel_count']}")
    print(f"  percentage:              {oracle['percentage']}")
    print(f"  matches expected:        {oracle['passed']}")

    if oracle_only:
        return

    # ── build LLM ─────────────────────────────────────────────────────────
    model = create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=config.agent.temperature,
    )
    llm_fn = _build_llm_fn(model, timeout_seconds=per_call_timeout_seconds)

    if verbose:
        llm_fn = _make_verbose_llm(llm_fn, output_dir)

    # ── create MemAgent directly (bypass registry for finer control) ──────
    agent = MemAgent(
        llm=llm_fn,
        config=MemAgentConfig(
            recurrent_max_context_len=16384,
            recurrent_chunk_size=recurrent_chunk_size,
            recurrent_chunk_overlap=256,
            max_memory_tokens=max_memory_tokens,
            per_call_timeout_seconds=per_call_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            keep_trace=True,
            use_deterministic_engine=use_deterministic_engine,
            repair_rounds=repair_rounds,
            min_field_coverage=0.75,
            emit_engine_records=emit_engine_records,
        ),
    )

    # ── step 1: chunk info ────────────────────────────────────────────────
    _banner("Step 1: Document Chunking")
    chunks = list(agent._chunk_text(document))
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunk size:   {recurrent_chunk_size} tokens")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i}: chars={len(chunk.text)}, start={chunk.start}, end={chunk.end}")
        print(f"    preview: {textwrap.shorten(chunk.text, width=120)}")
    if len(chunks) > 3:
        print(f"  ... ({len(chunks) - 3} more chunks)")
        last = chunks[-1]
        print(f"  Chunk {len(chunks) - 1}: chars={len(last.text)}, start={last.start}, end={last.end}")

    # ── step 2: run pattern scanning ──────────────────────────────────────
    _banner("Step 2: Pattern Scanning (build_extraction_patterns)")
    t0 = time.monotonic()
    result = agent.build_extraction_patterns(question, document)
    elapsed = time.monotonic() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Steps recorded: {len(result.steps)}")
    print(f"  Answer (tail 300 chars): ...{result.answer[-300:] if len(result.answer) > 300 else result.answer}")

    # ── step 3: diagnostics ───────────────────────────────────────────────
    _banner("Step 3: Diagnostics")
    diag = result.diagnostics
    print(f"  Original tokens:      {diag.original_token_count}")
    print(f"  Processed tokens:     {diag.processed_token_count}")
    print(f"  Chunk count:          {diag.chunk_count}")
    print(f"  Token coverage:       {diag.token_coverage:.2%}")
    print(f"  Complete scan:        {diag.complete_scan}")
    print(f"  Timed out:            {diag.timed_out}")
    print(f"  PatternSpec parsed:   {not diag.pattern_spec_parse_failed}")
    print(f"  Repair rounds used:   {diag.repair_rounds_used}")

    # ── step 4: pattern_spec ──────────────────────────────────────────────
    _banner("Step 4: PatternSpec Result")
    if result.pattern_spec is not None:
        _print_spec_summary(result.pattern_spec)
        _write_json(output_dir / "pattern_spec.json", asdict(result.pattern_spec))
    else:
        print("  FAILED: No PatternSpec extracted!")
        print(f"  (pattern_spec_parse_failed={diag.pattern_spec_parse_failed})")

    # ── step 5: engine diagnostics ────────────────────────────────────────
    _banner("Step 5: Engine Diagnostics")
    if result.engine_diagnostics is not None:
        ed = result.engine_diagnostics
        print(f"  Record count:     {ed.record_count}")
        print(f"  Field coverage:   {ed.field_coverage}")
        print(f"  Field hit counts: {ed.field_hit_counts}")
        print(f"  Orphan fields:    {len(ed.orphan_field_sentences)}")
        print(f"  Conflicts:        {len(ed.conflicts)}")
        if ed.error:
            print(f"  ERROR: {ed.error}")
        _write_json(output_dir / "engine_diagnostics.json", asdict(result.engine_diagnostics))
    else:
        print("  No engine diagnostics (engine not run).")

    # ── step 6: engine records ────────────────────────────────────────────
    _banner("Step 6: Engine Records")
    if result.engine_records is not None:
        records = result.engine_records
        print(f"  Records extracted: {len(records)}")
        for r in records[:5]:
            print(f"    {r}")
        if len(records) > 5:
            print(f"  ... ({len(records) - 5} more records)")
        _write_json(output_dir / "engine_records.json", records)

        from_records = _task396_from_records(records)
        _write_json(output_dir / "task396_from_engine.json", from_records)

        _sub_banner("Task 396 from Engine Records")
        print(f"  denominator (150≤h≤180): {from_records.get('denominator')}")
        print(f"  marvel (publisher=13):   {from_records.get('marvel_count')}")
        print(f"  percentage:              {from_records.get('percentage')}")
        print(f"  matches_oracle:          {from_records.get('matches_oracle')}")
    else:
        print("  No engine records (emit_engine_records=False or engine not run).")

    # ── step 7: save all steps ────────────────────────────────────────────
    if result.steps:
        _write_json(
            output_dir / "steps.json",
            [
                {
                    "index": s.index,
                    "chunk_start": s.chunk_start,
                    "chunk_end": s.chunk_end,
                    "memory_preview": s.memory[:500] + "..." if len(s.memory) > 500 else s.memory,
                }
                for s in result.steps
            ],
        )

    # ── step 8: final summary ─────────────────────────────────────────────
    _banner("Summary")
    score = 0
    if result.pattern_spec is not None:
        score += 1
        print("  [PASS] PatternSpec extracted successfully")
    else:
        print("  [FAIL] PatternSpec extraction failed")
    if result.engine_diagnostics is not None:
        score += 1
        coverage_ok = all(
            v >= 0.75 for v in result.engine_diagnostics.field_coverage.values()
        )
        if coverage_ok:
            print("  [PASS] All field coverage >= 75%")
            score += 1
        else:
            low_fields = [
                f"{k}={v:.0%}"
                for k, v in result.engine_diagnostics.field_coverage.items()
                if v < 0.75
            ]
            print(f"  [WARN] Low coverage fields: {low_fields}")
    else:
        print("  [FAIL] Engine diagnostics unavailable")
    if result.engine_records is not None:
        from_records = _task396_from_records(result.engine_records)
        if from_records.get("matches_oracle"):
            score += 2
            print("  [PASS] Engine records match oracle!")
        else:
            print(f"  [FAIL] Engine records do not match oracle: {from_records}")
    print(f"\n  Score: {score}/5")
    print(f"  All outputs saved to: {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug MemAgent on task_396",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--verbose", action="store_true", help="Log every LLM prompt and response")
    parser.add_argument("--oracle-only", action="store_true", help="Only run the oracle baseline")
    parser.add_argument("--total-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--per-call-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--recurrent-chunk-size", type=int, default=8192)
    parser.add_argument("--max-memory-tokens", type=int, default=4096)
    parser.add_argument("--use-deterministic-engine", action="store_true", default=True)
    parser.add_argument("--no-deterministic-engine", dest="use_deterministic_engine", action="store_false")
    parser.add_argument("--repair-rounds", type=int, default=2)
    parser.add_argument("--emit-engine-records", action="store_true", default=True)
    parser.add_argument("--direct", action="store_true", help="Use direct path/question instead of trace file")
    parser.add_argument("--path", type=str, default="doc/superhero.md", help="Document path (for --direct)")
    parser.add_argument(
        "--question", type=str,
        default=(
            "Extract superhero data including: superhero_name (codename), full_name, "
            "height_cm (in centimeters), weight_kg, and publisher_id (the numeric "
            "publisher code like 13, 4, 10, etc.) from each entry in this document. "
            "The document contains entries for many superheroes with their biometric "
            "and classification data. Find patterns where height is mentioned (like "
            "'height is recorded as X.X centimeters') and publisher affiliation (like "
            "'publisher affiliation is logged with the code X')."
        ),
        help="Question for memagent (for --direct)",
    )
    args = parser.parse_args()

    trace_path = None if args.direct else args.trace

    run_debug(
        config_path=args.config,
        trace_path=trace_path,
        task_id=args.task_id,
        verbose=args.verbose,
        oracle_only=args.oracle_only,
        use_deterministic_engine=args.use_deterministic_engine,
        repair_rounds=args.repair_rounds,
        total_timeout_seconds=args.total_timeout_seconds,
        per_call_timeout_seconds=args.per_call_timeout_seconds,
        recurrent_chunk_size=args.recurrent_chunk_size,
        max_memory_tokens=args.max_memory_tokens,
        emit_engine_records=args.emit_engine_records,
        direct_path=args.path if args.direct else None,
        direct_question=args.question if args.direct else None,
    )


if __name__ == "__main__":
    main()
