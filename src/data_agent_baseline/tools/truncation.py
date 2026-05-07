from __future__ import annotations

from typing import Any

TRUNCATION_SUFFIX = (
    "\n\n...（内容已被截断，返回过长。"
    "如未获取到所需信息，请尝试其他方式而非直接读取全部文档。）"
)


def truncate_str(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_SUFFIX


def truncate_content(
    obj: Any,
    *,
    max_str_chars: int = 8000,
    max_list_items: int = 200,
) -> Any:
    if isinstance(obj, str):
        return truncate_str(obj, max_str_chars)

    if isinstance(obj, list):
        truncated = obj[:max_list_items]
        result: list[Any] = [
            truncate_content(item, max_str_chars=max_str_chars, max_list_items=max_list_items)
            for item in truncated
        ]
        if len(obj) > max_list_items:
            result.append(TRUNCATION_SUFFIX)
        return result

    if isinstance(obj, dict):
        return {
            key: truncate_content(value, max_str_chars=max_str_chars, max_list_items=max_list_items)
            for key, value in obj.items()
        }

    return obj
