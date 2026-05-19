from __future__ import annotations

import concurrent.futures
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import tiktoken

LLMFn = Callable[[str], str]

NO_MEMORY = "No previous memory"
RECURRENT_MAX_CONTEXT_LEN = 240000
RECURRENT_CHUNK_SIZE = 4096
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
No patterns identified yet.

As you scan the document, build a structured extraction guide.  Your goal is NOT to
extract the actual data, but to teach a regex/Python program how to extract it.

Your final output should follow this structure:

## Document Sections
[For each logical section: describe its topic and approximate line range]

## Extraction Patterns
[For each data field, record:
  - Field name and data type
  - The exact regex with capture groups that extracts it
  - One example original text line and what the regex captures from it
  - Which document section(s) the field appears in]

## Complete Extraction Code
[At the very end, provide a complete, runnable Python code block that:
  1. Reads the file
  2. Applies all identified regexes/patterns
  3. Outputs structured data as JSON (list of dicts)
  4. Handles missing/placeholder values (0.0, NaN, None, -)]
"""

TEMPLATE_PATTERN_ANALYSIS = """\
You are scanning a long document chunk by chunk to identify text structure and data
extraction patterns.

Your job is NOT to extract data values, but to teach a program how to extract data.
For each section of the document, identify:
1. What data fields are present (IDs, measurements, codes, names, dates, etc.)
2. The exact text patterns that surround each field
3. Regular expressions that can capture each field value
4. How entries are structured (one per line? one per paragraph? mixed?)

<question>
{question}
</question>

<patterns>
{memory}
</patterns>

<section>
{chunk}
</section>

Output the complete updated <patterns> document.  Keep all previously identified
patterns that are still valid.  Add new patterns as you discover them.  Always
include the three sections: Document Sections, Extraction Patterns, and
Complete Extraction Code (update the code as patterns evolve).
"""


__all__ = [
    "MemAgent",
    "MemAgentConfig",
    "MemAgentResult",
    "_build_llm_fn",
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
    max_memory_tokens: int = MEMORY_MAX_TOKENS
    max_retries: int = 2
    sleep_between_calls: float = 0.0
    keep_trace: bool = True
    per_call_timeout_seconds: float = 120.0
    total_timeout_seconds: float = 300.0


@dataclass
class MemoryStep:
    index: int
    chunk_start: int
    chunk_end: int
    memory: str


@dataclass
class MemAgentResult:
    question: str
    answer: str
    steps: List[MemoryStep] = field(default_factory=list)


def _get_tiktoken_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TIKTOKEN_ENCODING_NAME)


def _truncate_text_to_max_tokens(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = encoding.encode(text)
    if len(ids) <= max_tokens:
        return text
    return encoding.decode(ids[:max_tokens])


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

        for idx, (start, end, chunk) in enumerate(self._chunk_text(document), start=1):
            elapsed = time.monotonic() - started_at
            if elapsed >= total_timeout:
                memory += "\n\n[MemAgent: total timeout reached, returning partial results.]"
                break

            prompt = self._build_context_update_prompt(
                question=question,
                memory=memory,
                chunk=chunk,
            )
            memory = self._call_with_retry(prompt)
            memory = self._clean_context_memory(memory)
            memory = _truncate_text_to_max_tokens(
                memory, self._encoding, self.config.max_memory_tokens
            )

            if self.config.keep_trace:
                steps.append(MemoryStep(index=idx, chunk_start=start, chunk_end=end, memory=memory))

            if self.config.sleep_between_calls:
                time.sleep(self.config.sleep_between_calls)

        answer = memory.strip()
        if answer == NO_MEMORY:
            answer = ""

        return MemAgentResult(
            question=question,
            answer=answer,
            steps=steps,
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

        for idx, (start, end, chunk) in enumerate(self._chunk_text(document), start=1):
            elapsed = time.monotonic() - started_at
            if elapsed >= total_timeout:
                memory += "\n\n[MemAgent: total timeout reached, returning partial patterns.]"
                break

            prompt = TEMPLATE_PATTERN_ANALYSIS.format(
                question=question,
                memory=memory,
                chunk=chunk,
            )
            memory = self._call_with_retry(prompt)
            memory = self._clean_context_memory(memory)
            memory = _truncate_text_to_max_tokens(
                memory, self._encoding, self.config.max_memory_tokens
            )

            if self.config.keep_trace:
                steps.append(MemoryStep(index=idx, chunk_start=start, chunk_end=end, memory=memory))

            if self.config.sleep_between_calls:
                time.sleep(self.config.sleep_between_calls)

        answer = memory.strip()
        if answer == PATTERN_MEMORY_INIT.strip():
            answer = ""

        return MemAgentResult(
            question=question,
            answer=answer,
            steps=steps,
        )

    def _chunk_text(self, text: str) -> Iterable[tuple[int, int, str]]:
        max_len = self.config.recurrent_max_context_len
        size = self.config.recurrent_chunk_size

        if max_len <= 0:
            raise ValueError("recurrent_max_context_len must be > 0")
        if size <= 0:
            raise ValueError("recurrent_chunk_size must be > 0")

        input_ids = self._encode_text(text)
        if len(input_ids) > max_len:
            sample_chunk_size = max(256, size // 4)
            chunks = [
                (idx, input_ids[i : i + sample_chunk_size])
                for idx, i in enumerate(range(0, len(input_ids), sample_chunk_size))
            ]
            # Randomly select which chunks to include, then reconstruct in
            # original document order so the LLM sees a coherent narrative.
            shuffled_indices = [c[0] for c in chunks]
            random.shuffle(shuffled_indices)
            selected: set[int] = set()
            total = 0
            for idx in shuffled_indices:
                chunk_len = len(chunks[idx][1])
                if total + chunk_len > max_len:
                    continue
                selected.add(idx)
                total += chunk_len
            sampled: list[int] = []
            for idx, (_, chunk) in enumerate(chunks):
                if idx in selected:
                    sampled.extend(chunk)
            input_ids = sampled

        for start in range(0, len(input_ids), size):
            end = min(start + size, len(input_ids))
            chunk = self._decode_tokens(input_ids[start:end])
            yield start, end, chunk

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
    per_call_timeout_seconds: float = 120.0,
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
            return {
                "path": str(doc_path),
                "question": effective_question,
                "answer": "",
                "chunk_count": 0,
            }

        result = agent.build_task_context(effective_question, document)
        content: dict[str, object] = {
            "path": str(doc_path),
            "question": effective_question,
            "answer": result.answer,
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
    per_call_timeout_seconds: float = 120.0,
    total_timeout_seconds: float = 300.0,
    keep_trace: bool = False,
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
            return {
                "path": str(doc_path),
                "question": effective_question,
                "answer": "",
                "chunk_count": 0,
            }

        result = agent.build_extraction_patterns(effective_question, document)
        content: dict[str, object] = {
            "path": str(doc_path),
            "question": effective_question,
            "answer": result.answer,
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

    return analyze_patterns
