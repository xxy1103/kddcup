from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import tiktoken

from data_agent_baseline.benchmark.schema import PublicTask

LLMFn = Callable[[str], str]

NO_MEMORY = "No previous memory"
RECURRENT_MAX_CONTEXT_LEN = 240000
RECURRENT_CHUNK_SIZE = 4096
TIKTOKEN_ENCODING_NAME = "o200k_base"
MEMORY_MAX_TOKENS = 4096

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

    def _chunk_text(self, text: str) -> Iterable[tuple[int, int, str]]:
        max_len = self.config.recurrent_max_context_len
        size = self.config.recurrent_chunk_size

        if max_len <= 0:
            raise ValueError("recurrent_max_context_len must be > 0")
        if size <= 0:
            raise ValueError("recurrent_chunk_size must be > 0")

        input_ids = self._encode_text(text)
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]

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


# ---------------------------------------------------------------------------
# Document selection helpers and build_document_context
# ---------------------------------------------------------------------------

TEXT_DOCUMENT_SUFFIXES: set[str] = {
    ".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".xml",
}
DOCUMENT_CONTEXT_MAX_DOCS = 10
DOCUMENT_CONTEXT_MAX_TOKENS = 4096


def _iter_text_document_paths(task: PublicTask) -> list[Path]:
    paths: list[Path] = []
    for path in task.context_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_DOCUMENT_SUFFIXES:
            continue
        paths.append(path)
    return paths


def _score_document_path(task: PublicTask, path: Path) -> tuple[int, int, str]:
    rel_path = path.relative_to(task.context_dir).as_posix()
    name = rel_path.lower()
    question_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9_]{3,}", task.question.lower())
        if term not in {"the", "and", "for", "with", "from", "that", "this"}
    }
    path_terms = set(re.findall(r"[a-zA-Z0-9_]{3,}", name))

    score = 0
    if name == "knowledge.md":
        score += 1000
    if name.endswith("/knowledge.md"):
        score += 900
    if "knowledge" in name:
        score += 250
    if "readme" in name or "guide" in name or "description" in name:
        score += 120
    score += 15 * len(question_terms & path_terms)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size >= 7000:
        score += 80
    elif size >= 2000:
        score += 30

    return (-score, -size, rel_path)


def _select_document_paths(task: PublicTask, *, max_docs: int) -> list[Path]:
    candidates = _iter_text_document_paths(task)
    candidates.sort(key=lambda p: _score_document_path(task, p))
    return candidates[:max_docs]


def build_document_context(
    task: PublicTask,
    model: object,
    *,
    max_docs: int = DOCUMENT_CONTEXT_MAX_DOCS,
    recurrent_max_context_len: int = RECURRENT_MAX_CONTEXT_LEN,
    recurrent_chunk_size: int = RECURRENT_CHUNK_SIZE,
    max_memory_tokens: int = MEMORY_MAX_TOKENS,
    max_total_tokens: int = DOCUMENT_CONTEXT_MAX_TOKENS,
) -> str | None:
    """Build task-relevant context from important long documents before the ReAct loop.

    This is intentionally not a normal tool call: the main agent receives the
    synthesised definitions, metric rules, and document constraints in its first
    user message.
    """
    selected_paths = _select_document_paths(task, max_docs=max_docs)
    if not selected_paths:
        return None

    llm = _build_llm_fn(model)
    agent = MemAgent(
        llm,
        config=MemAgentConfig(
            recurrent_max_context_len=recurrent_max_context_len,
            recurrent_chunk_size=recurrent_chunk_size,
            max_memory_tokens=max_memory_tokens,
            keep_trace=False,
        ),
    )

    sections: list[str] = []
    for path in selected_paths:
        rel_path = path.relative_to(task.context_dir).as_posix()
        try:
            document = path.read_text(encoding="utf-8", errors="replace")
            if not document.strip():
                continue
            result = agent.build_task_context(task.question, document)
            context = result.answer.strip()
        except Exception as exc:
            context = f"- Failed to synthesize this document: {exc}"

        if not context:
            continue
        sections.append(f"Source: {rel_path}\n{context}")

    combined = "\n\n".join(sections).strip()
    if not combined:
        return None
    enc = _get_tiktoken_encoding()
    notice = "\n[document context truncated]"
    if len(enc.encode(combined)) > max_total_tokens:
        notice_tokens = len(enc.encode(notice))
        budget = max(1, max_total_tokens - notice_tokens)
        combined = _truncate_text_to_max_tokens(combined, enc, budget) + notice
    return combined
