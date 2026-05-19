from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_agent_baseline.tools.memagent import (
    TEXT_DOCUMENT_SUFFIXES,
    MemAgent,
    MemAgentConfig,
    MemAgentResult,
    _build_llm_fn,
    _get_tiktoken_encoding,
    _truncate_text_to_max_tokens,
    make_process_long_doc,
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
    assert cfg.max_memory_tokens == 4096
    assert cfg.max_retries == 0
    assert cfg.keep_trace is True
