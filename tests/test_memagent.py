from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from data_agent_baseline.tools.memagent import (
    TEMPLATE_PATTERN_ANALYSIS,
    MemAgent,
    MemAgentConfig,
    MemAgentResult,
    PatternSpec,
    _build_llm_fn,
    _extract_json_block,
    _get_tiktoken_encoding,
    _json_to_pattern_spec,
    _truncate_text_to_max_tokens,
    extract_records_with_spec,
    make_pattern_analyzer,
)
from data_agent_baseline.tools.registry import create_default_tool_registry


def test_memagent_single_chunk() -> None:
    """A single-chunk document should produce a summary."""
    def mock_llm(prompt: str) -> str:
        return "The height of Superman is 6'3\"."

    agent = MemAgent(
        mock_llm,
        config=MemAgentConfig(recurrent_chunk_size=65536, keep_trace=False),
    )
    result = agent.build_task_context(
        "What is Superman's height?",
        "Clark Kent is Superman.\nHe is 6'3\" tall.",
    )
    assert "6'3" in result.answer
    assert isinstance(result, MemAgentResult)


def test_memagent_multi_chunk_merges_memory() -> None:
    """Multi-chunk document accumulates memory across chunks."""
    calls: list[int] = []

    def mock_llm(prompt: str) -> str:
        calls.append(len(calls))
        if len(calls) == 1:
            return "Height: 6'3\""
        return "Height: 6'3\", Publisher: DC Comics"

    agent = MemAgent(
        mock_llm,
        config=MemAgentConfig(recurrent_chunk_size=10, max_memory_tokens=4096, keep_trace=False),
    )
    text = "Superman 6'3\"\n" + "Batman 6'2\"\n" * 100
    result = agent.build_task_context("heights", text)
    assert "DC Comics" in result.answer
    assert len(calls) > 1


def test_memagent_empty_llm_retries_then_raises() -> None:
    """Empty LLM response triggers retries, then raises."""
    call_count = 0

    def mock_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return ""

    agent = MemAgent(
        mock_llm,
        config=MemAgentConfig(recurrent_chunk_size=4096, max_retries=1, keep_trace=False),
    )
    with pytest.raises(RuntimeError, match="LLM call failed"):
        agent.build_task_context("test", "some text here that is long enough")


def test_memagent_empty_document_returns_empty() -> None:
    """An empty document should produce an empty answer."""

    def mock_llm(prompt: str) -> str:
        return "should not be called"

    agent = MemAgent(mock_llm, config=MemAgentConfig(keep_trace=False))
    result = agent.build_task_context("test", "")
    assert result.answer == ""


def test_build_llm_fn_string_content() -> None:
    """Adapter extracts string content from AIMessage.content."""
    model = MagicMock()
    model.invoke.return_value.content = "Hello world"
    llm = _build_llm_fn(model)
    assert llm("prompt") == "Hello world"


def test_build_llm_fn_list_content() -> None:
    """Adapter extracts text blocks from list content."""
    model = MagicMock()
    model.invoke.return_value.content = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " world"},
    ]
    llm = _build_llm_fn(model)
    assert llm("prompt") == "Hello world"


def test_build_llm_fn_empty_response() -> None:
    """Adapter returns empty string when model returns empty content."""
    model = MagicMock()
    model.invoke.return_value.content = ""
    llm = _build_llm_fn(model)
    assert llm("prompt") == ""


def test_truncate_text_to_max_tokens() -> None:
    enc = _get_tiktoken_encoding()
    text = "hello world " * 100
    truncated = _truncate_text_to_max_tokens(text, enc, 10)
    assert len(enc.encode(truncated)) <= 10


def test_truncate_text_to_max_tokens_zero() -> None:
    enc = _get_tiktoken_encoding()
    assert _truncate_text_to_max_tokens("hello", enc, 0) == ""


def test_truncate_text_to_max_tokens_short_enough() -> None:
    enc = _get_tiktoken_encoding()
    assert _truncate_text_to_max_tokens("hi", enc, 100) == "hi"


def test_memagent_is_registered() -> None:
    """Verify memagent tool is registered in the default registry."""
    registry = create_default_tool_registry()
    assert "memagent" in registry.specs, "memagent spec should be registered"
    assert "memagent" in registry.handlers, "memagent handler should be registered"
    spec = registry.specs["memagent"]
    assert spec.name == "memagent"
    assert "chunk-by-chunk" in spec.description


def test_memagent_config_defaults() -> None:
    cfg = MemAgentConfig()
    assert cfg.recurrent_chunk_size == 8192
    assert cfg.recurrent_chunk_overlap == 256
    assert cfg.max_memory_tokens == 4096
    assert cfg.max_retries == 0
    assert cfg.keep_trace is True
    assert cfg.use_deterministic_engine is True
    assert cfg.repair_rounds == 2
    assert cfg.min_field_coverage == 0.75
    assert cfg.emit_engine_records is False


def test_memagent_long_document_chunks_do_not_drop_tail() -> None:
    """Long documents should be scanned deterministically instead of sampled."""

    agent = MemAgent(
        lambda prompt: "ok",
        config=MemAgentConfig(
            recurrent_max_context_len=30,
            recurrent_chunk_size=30,
            recurrent_chunk_overlap=0,
            keep_trace=False,
        ),
    )
    document = "\n\n".join(f"paragraph {idx} " + ("word " * 20) for idx in range(8))
    document += "\n\nTAIL_SENTINEL"

    chunks = list(agent._chunk_text(document))

    assert chunks[0].start == 0
    assert chunks[-1].end == len(agent._encode_text(document))
    assert "TAIL_SENTINEL" in chunks[-1].text
    assert "".join(chunk.text for chunk in chunks).count("TAIL_SENTINEL") == 1


def test_memagent_paragraph_chunks_preserve_order_and_overlap() -> None:
    agent = MemAgent(
        lambda prompt: "ok",
        config=MemAgentConfig(
            recurrent_max_context_len=24,
            recurrent_chunk_size=24,
            recurrent_chunk_overlap=5,
            keep_trace=False,
        ),
    )
    document = "\n\n".join(
        [
            "alpha " * 12,
            "bravo " * 12,
            "charlie " * 12,
            "delta " * 12,
        ]
    )

    chunks = list(agent._chunk_text(document))

    assert len(chunks) > 1
    assert "alpha" in chunks[0].text
    assert "delta" in chunks[-1].text
    assert all(left.start <= right.start for left, right in zip(chunks, chunks[1:]))
    assert any(left.end > right.start for left, right in zip(chunks, chunks[1:]))


def test_pattern_prompt_requires_join_key_and_record_extractor() -> None:
    assert "join key" in TEMPLATE_PATTERN_ANALYSIS
    assert "merge" in TEMPLATE_PATTERN_ANALYSIS
    assert "PatternSpec" in TEMPLATE_PATTERN_ANALYSIS
    assert "join_key_patterns" in TEMPLATE_PATTERN_ANALYSIS
    assert "Do NOT extract independent lists" in TEMPLATE_PATTERN_ANALYSIS
    assert "deterministic engine" in TEMPLATE_PATTERN_ANALYSIS


def test_pattern_analyzer_visits_all_long_document_chunks(tmp_path) -> None:
    seen_sections: list[str] = []

    def mock_llm(prompt: str) -> str:
        match = re.search(r"<section>\n(.*?)\n</section>", prompt, flags=re.S)
        assert match is not None
        seen_sections.append(match.group(1))
        return (
            "## Document Sections\nseen\n\n"
            "## Record Join Strategy\nUse ID.\n\n"
            "## Fields Inventory\nPENDING | field | not yet observed\n\n"
            "## Complete Extraction Code\n# WARNING: some fields are still PENDING\n\n"
            "## Validation Warnings\nnone"
        )

    doc_path = tmp_path / "long.md"
    doc_path.write_text(
        "\n\n".join([f"section {idx} " + ("token " * 30) for idx in range(10)])
        + "\n\nFINAL_MARKER",
        encoding="utf-8",
    )
    analyzer = make_pattern_analyzer(
        mock_llm,
        recurrent_max_context_len=40,
        recurrent_chunk_size=40,
        max_memory_tokens=4096,
        keep_trace=True,
    )

    result = analyzer("Find IDs and fields.", doc_path)

    assert result["diagnostics"]["complete_scan"] is True
    assert result["diagnostics"]["chunk_count"] == len(seen_sections)
    assert "section 0" in seen_sections[0]
    assert "FINAL_MARKER" in seen_sections[-1]


# ---------------------------------------------------------------------------
# PatternSpec + deterministic engine tests
# ---------------------------------------------------------------------------


def test_extract_records_basic() -> None:
    """Single-section doc with IDs and fields in same paragraph."""
    spec = PatternSpec(
        join_key_patterns=[r"ID\s*(\d+)"],
        field_patterns={
            "height_cm": [r"height\D*?(\d+(?:\.\d+)?)\s*(?:cm|centimeters)"],
            "publisher_id": [r"publisher.*?(\d+)"],
        },
        field_types={"height_cm": "float", "publisher_id": "int"},
    )
    text = (
        "Hero ID 1 has height 170.0 cm and publisher 13.\n\n"
        "Hero ID 2 has height 180.0 cm and publisher 4."
    )
    records, diag = extract_records_with_spec(text, spec)
    assert diag.record_count == 2
    assert len(records) == 2
    heights = sorted(r.get("height_cm") for r in records)
    assert heights == [170.0, 180.0]
    assert all(isinstance(r.get("height_cm"), float) for r in records)
    assert all(isinstance(r.get("publisher_id"), int) for r in records)


def test_extract_records_split_chapter() -> None:
    """IDs in section 1, height in section 2, publisher in section 5.

    Each ID's height and publisher are in separate paragraphs, ensuring
    the engine merges across sections by the join key.
    """
    spec = PatternSpec(
        join_key_patterns=[r"ID\s*(\d+)"],
        field_patterns={
            "height_cm": [r"height\D*?(\d+(?:\.\d+)?)\s*(?:cm|centimeters)"],
            "publisher_id": [r"publisher.*?(\d+)"],
        },
        field_types={"height_cm": "float", "publisher_id": "int"},
    )
    text = (
        "ID 1 has height 170.0 cm.\n\n"
        "ID 2 has height 180.0 cm.\n\n"
        "ID 1 has publisher 13.\n\n"
        "ID 2 has publisher 4."
    )
    records, diag = extract_records_with_spec(text, spec)
    assert len(records) == 2
    r1 = next(r for r in records if r.get("height_cm") == 170.0)
    assert r1["publisher_id"] == 13
    r2 = next(r for r in records if r.get("height_cm") == 180.0)
    assert r2["publisher_id"] == 4


def test_extract_records_correction() -> None:
    """Correction language; last positional value wins."""
    spec = PatternSpec(
        join_key_patterns=[r"ID\s*(\d+)"],
        field_patterns={
            "height_cm": [r"(\d+(?:\.\d+)?)\s*(?:cm|centimeters)"],
        },
        field_types={"height_cm": "float"},
    )
    text = (
        "ID 1 was initially 175.0 cm but later corrected to 178.0 cm."
    )
    records, diag = extract_records_with_spec(text, spec)
    assert len(records) == 1
    assert records[0]["height_cm"] == 178.0


def test_extract_records_multi_id_paragraph() -> None:
    """Multiple IDs in one paragraph share the same field values."""
    spec = PatternSpec(
        join_key_patterns=[r"ID\s*(\d+)"],
        field_patterns={
            "height_cm": [r"height\D*?(\d+(?:\.\d+)?)\s*(?:cm|centimeters)"],
        },
        field_types={"height_cm": "float"},
    )
    text = "ID 1 and ID 2 both have height 175.0 cm."
    records, diag = extract_records_with_spec(text, spec)
    assert len(records) == 2
    for r in records:
        assert r["height_cm"] == 175.0


def test_extract_records_null_handling() -> None:
    """NaN, None, 0.0, '-' should be treated as null and skipped."""
    spec = PatternSpec(
        join_key_patterns=[r"ID\s*(\d+)"],
        field_patterns={
            "height_cm": [r"height\D*?(\d+(?:\.\d+)?)\s*(?:cm|centimeters)"],
        },
        field_types={"height_cm": "float"},
        null_values=["None", "NaN", "-", ""],
    )
    text = (
        "ID 1 has height NaN cm.\n\n"
        "ID 2 has height 170.0 cm.\n\n"
        "ID 3 has height - cm."
    )
    records, diag = extract_records_with_spec(text, spec)
    records_with_height = [r for r in records if "height_cm" in r]
    assert len(records_with_height) == 1
    assert records_with_height[0]["height_cm"] == 170.0


def test_extract_records_orphan_fields() -> None:
    """Fields without a join key should be tracked as orphans."""
    spec = PatternSpec(
        join_key_patterns=[r"ID\s*(\d+)"],
        field_patterns={
            "height_cm": [r"height\D*?(\d+(?:\.\d+)?)\s*(?:cm|centimeters)"],
        },
        field_types={"height_cm": "float"},
    )
    text = (
        "ID 1 has height 170.0 cm.\n\n"
        "Some random paragraph with height 180.0 cm but no ID.\n\n"
        "ID 2 has height 165.0 cm."
    )
    records, diag = extract_records_with_spec(text, spec)
    assert len(records) == 2
    assert len(diag.orphan_field_sentences) == 1
    assert "180.0" in diag.orphan_field_sentences[0]


def test_pattern_spec_json_parsing() -> None:
    """Valid JSON blocks parse; invalid ones return None."""
    markdown = (
        "Some text\n"
        '```json\n{"join_key_patterns": ["ID (\\\\d+)"], '
        '"field_patterns": {"h": ["h (\\\\d+)"]}, '
        '"field_types": {"h": "float"}}\n'
        "```\nMore text"
    )
    data = _extract_json_block(markdown)
    assert data is not None
    spec = _json_to_pattern_spec(data)
    assert spec is not None
    assert spec.join_key_patterns == ["ID (\\d+)"]
    assert spec.field_patterns == {"h": ["h (\\d+)"]}
    assert spec.field_types == {"h": "float"}

    assert _extract_json_block("no code block") is None
    assert _extract_json_block("```json\ninvalid json\n```") is None

    # Missing join_key_patterns should fail validation
    assert _json_to_pattern_spec({"field_patterns": {"x": ["x"]}}) is None


def test_pattern_spec_json_fallback() -> None:
    """When LLM returns no JSON, pattern_spec_parse_failed is set."""
    agent = MemAgent(
        lambda prompt: "This is a markdown guide without any JSON block.",
        config=MemAgentConfig(
            recurrent_chunk_size=4096,
            keep_trace=False,
            use_deterministic_engine=True,
        ),
    )
    result = agent.build_extraction_patterns(
        "Extract ID and height.",
        "ID 1 has height 170.0 cm.",
    )
    assert result.answer != ""
    assert result.pattern_spec is None
    assert result.diagnostics.pattern_spec_parse_failed is True


def test_pattern_spec_repair_loop() -> None:
    """Low-coverage spec triggers repair and improves."""
    call_count = [0]

    def mock_llm(prompt: str) -> str:
        call_count[0] += 1
        # First call: return a spec with a non-matching regex → coverage 0
        if call_count[0] == 1:
            return (
                "## Fields Inventory\n"
                "TENTATIVE | height_cm | regex: nonexistent_field (\\d+) cm\n\n"
                "## PatternSpec\n"
                '```json\n{"join_key_patterns": ["ID (\\\\d+)"], '
                '"field_patterns": {"height_cm": ["nonexistent_field (\\\\d+) cm"]}, '
                '"field_types": {"height_cm": "float"}}\n```\n\n'
                "## Validation Warnings\nbad regex"
            )
        # Repair call: return an improved spec with a working regex
        return (
            '```json\n{"join_key_patterns": ["ID (\\\\d+)"], '
            '"field_patterns": {'
            '"height_cm": ["height\\\\D*?(\\\\d+(?:\\\\.\\\\d+)?)\\\\s*(?:cm|centimeters)"], '
            '"publisher_id": ["publisher.*?(\\\\d+)"]}, '
            '"field_types": {"height_cm": "float", "publisher_id": "int"}}\n```'
        )

    agent = MemAgent(
        mock_llm,
        config=MemAgentConfig(
            recurrent_chunk_size=4096,
            keep_trace=False,
            use_deterministic_engine=True,
            repair_rounds=1,
            min_field_coverage=0.5,
            max_retries=0,
        ),
    )
    document = (
        "ID 1 has height 170.0 cm and publisher 13.\n\n"
        "ID 2 has height 180.0 cm and publisher 4."
    )
    result = agent.build_extraction_patterns(
        "Extract ID, height, and publisher.",
        document,
    )
    # The repair should have been triggered and produced an improved spec
    assert result.pattern_spec is not None
    assert "publisher_id" in result.pattern_spec.field_patterns
    assert result.diagnostics.repair_rounds_used >= 1
    assert result.engine_records is None  # emit_engine_records defaults to False


def test_engine_disabled_skips_deterministic_path() -> None:
    """When use_deterministic_engine is False, no PatternSpec extraction occurs."""
    agent = MemAgent(
        lambda prompt: "## Fields Inventory\nCONFIRMED | h | regex\n\n## PatternSpec\n```json\n"
                        '{"join_key_patterns": ["ID (\\\\d+)"], '
                        '"field_patterns": {"h": ["h (\\\\d+)"]}, '
                        '"field_types": {"h": "float"}}\n```',
        config=MemAgentConfig(
            recurrent_chunk_size=4096,
            keep_trace=False,
            use_deterministic_engine=False,
        ),
    )
    result = agent.build_extraction_patterns(
        "Extract ID and height.",
        "ID 1 has height 170.0 cm.",
    )
    assert result.answer != ""
    # Engine path was skipped, so no pattern_spec or engine_diagnostics
    assert result.pattern_spec is None
    assert result.engine_diagnostics is None
