from __future__ import annotations

from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.truncation import truncate_str
from data_agent_baseline.token_utils import count_tokens


def normalize_context_relative_path(relative_path: str) -> str:
    normalized = relative_path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while normalized.startswith("context/"):
        normalized = normalized[len("context/") :]
    if normalized == "context":
        return ""
    return normalized.strip("/")


# 解析 context/ 下的相对路径，并阻止路径逃逸到任务目录之外。
def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    normalized_path = normalize_context_relative_path(relative_path)
    candidate = (task.context_dir / normalized_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing context asset: {normalized_path or relative_path}. "
            "All tool paths must be relative to the context directory."
        )
    return candidate


# 递归列出 context/ 目录树，用于让模型先了解有哪些可用资产。
def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    # 递归遍历目录，并记录相对路径、类型和文件大小。
    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": ".",
        "path_convention": "All paths are relative to the context directory. Use them exactly as listed and do not prefix them with `context/`.",
        "entries": entries,
    }


# 读取普通文本文件的片段，适合 markdown、txt 等说明文档。
def read_doc_preview(task: PublicTask, relative_path: str, *, max_tokens: int = 1000) -> dict[str, object]:
    normalized_path = normalize_context_relative_path(relative_path)
    path = resolve_context_path(task, normalized_path)
    text = path.read_text(errors="replace")
    preview = truncate_str(text, max_tokens=max_tokens)
    return {
        "path": normalized_path,
        "preview": preview,
        "truncated": count_tokens(text) > max_tokens,
    }
