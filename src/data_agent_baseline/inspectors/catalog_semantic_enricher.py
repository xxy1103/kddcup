from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.model_retry import invoke_model_with_retries

logger = logging.getLogger(__name__)

_ENRICHMENT_SYSTEM_PROMPT = """\
You are a data catalog enrichment assistant. Your task is to add field-level \
descriptions to a lightweight data catalog.

## Input

You will receive a JSON object (the "catalog") with these top-level keys:
- `task_id`: a string
- `assets`: list of {asset_path, kind}
- `schemas`: list of schema objects, each with {asset_path, kind, fields (or tables)}
- `relationships`: list of foreign-key relationships
- `knowledge_documents`: list of {asset_path, content, token_count, is_full_content}

The `knowledge_documents[*].content` contains human-written documentation (knowledge.md)
that explains the meaning of fields across the data assets.

## Your Task

For **every field** inside every schema's `fields` array (or nested `tables[*].fields`),
add a `description` (required) and optionally a `note` (optional):

- **description** (string, required): a single concise sentence explaining the business \
meaning of this field. Derive this from the field name, its type, and any relevant \
knowledge document content. If the knowledge documents mention this field, use that \
information. Otherwise, infer from the field name and type alone.

- **note** (string, optional): additional context such as "primary key", "foreign key \
to X.Y", or "values in range [a, b]". Only include `note` when you have specific extra \
information beyond the description. Otherwise omit it.

## Critical Rules

1. **Do NOT** add, remove, or reorder any fields, tables, schemas, or assets.
2. **Do NOT** change any existing field values (name, type, asset_path, etc.).
3. **Do NOT** add any new top-level keys.
4. **Do NOT** add any keys to field objects other than `description` and `note`.
5. Every field MUST have a `description` (string, non-empty).
6. Preserve ALL original JSON structure exactly — only add `description`/`note`.

## Output Format

Return ONLY a valid JSON object (no markdown fences, no explanation).
The output must be the complete enriched catalog.
"""

"""
您是一位数据目录丰富化助手。您的任务是为轻量级数据目录添加字段级别的描述信息。

## 输入

您将收到一个 JSON 对象（即“目录”），其顶层键如下：
- `task_id`：字符串
- `assets`：包含 `{asset_path, kind}` 的列表
- `schemas`：包含模式对象的列表，每个模式对象具有 `{asset_path, kind, fields（或 tables）}`
- `relationships`：外键关系的列表
- `knowledge_documents`：包含 `{asset_path, content, token_count, is_full_content}` 的列表

其中，`knowledge_documents[*].content` 包含由人工撰写的文档（knowledge.md），用于说明各数据资产中字段的含义。

## 您的任务

对于每个模式的 `fields` 数组中的**每一个字段**（以及嵌套的 `tables[*].fields`），请为其添加一个必填的 `description` 字段，并可选地添加一个 `note` 字段：

- **description**（字符串，必填）：用一句简洁明了的话说明该字段的业务含义。请根据字段名、字段类型以及相关知识文档的内容推导出这一描述。若知识文档中提及该字段，则优先采用文档中的信息；否则，仅依据字段名和字段类型进行推断。

- **note**（字符串，可选）：提供额外的上下文信息，例如“主键”、“外键指向 X.Y”或“取值范围为 [a, b]”。仅当您掌握超出描述之外的特定补充信息时才添加 `note`，否则应予以省略。

## 重要规则

1. **严禁**增删或重新排序任何字段、表、模式或数据资产。
2. **严禁**修改任何现有字段的属性值（如名称、类型、`asset_path` 等）。
3. **严禁**新增任何顶层键。
4. **严禁**在字段对象中添加除 `description` 和 `note` 之外的其他键。
5. 每个字段必须拥有一个非空的 `description` 字段。
6. 必须完整保留原始 JSON 的所有结构——仅添加 `description` 和 `note`。

## 输出格式

请仅返回一个合法的 JSON 对象（无需 Markdown 代码块，亦无需附加说明）。输出必须是完整的、已丰富化的数据目录。
"""


def enrich_lightweight_catalog_with_knowledge(
    *,
    model: BaseChatModel,
    lightweight_catalog: str,
) -> str:
    original: dict[str, Any] = json.loads(lightweight_catalog)

    schemas = original.get("schemas", [])
    field_count = _count_fields(schemas)
    if field_count == 0:
        return lightweight_catalog

    knowledge_preview = _build_knowledge_preview(original.get("knowledge_documents", []))
    schema_only_json = json.dumps(
        {k: v for k, v in original.items() if k != "knowledge_documents"},
        ensure_ascii=False,
        indent=2,
    )

    user_message = (
        "Below is a data catalog. The `knowledge_documents` have been extracted "
        "separately for your reference.\n\n"
        "<knowledge_documents>\n"
        f"{knowledge_preview}\n"
        "</knowledge_documents>\n\n"
        "<catalog>\n"
        f"{schema_only_json}\n"
        "</catalog>\n\n"
        "Add a `description` (and optionally `note`) to every field in the catalog. "
        "Output the complete enriched catalog as a single JSON object."
    )

    messages = [
        SystemMessage(content=_ENRICHMENT_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        ai_message = invoke_model_with_retries(model, messages)
    except Exception as exc:
        logger.warning("Catalog enrichment LLM call failed: %s", exc)
        return lightweight_catalog

    response_text = _extract_response_text(ai_message)
    parsed = _try_parse_json(response_text)
    if parsed is None:
        logger.warning("Catalog enrichment returned non-JSON; using original catalog.")
        return lightweight_catalog

    # knowledge_documents is sent separately in <knowledge_documents>,
    # so the LLM returns the catalog without it. We only validate
    # the keys that were actually sent to the LLM.
    sent_keys = set(original.keys()) - {"knowledge_documents"}
    errors = _validate_enriched_catalog(original, parsed, expected_keys=sent_keys)
    if errors:
        logger.warning(
            "Catalog enrichment validation failed (%d errors): %s",
            len(errors),
            "; ".join(errors[:5]),
        )
        return lightweight_catalog

    merged = _merge_enrichments(original, parsed)
    return json.dumps(merged, ensure_ascii=False, indent=2)


def _count_fields(schemas: list[dict[str, Any]]) -> int:
    total = 0
    for schema in schemas:
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                total += len(table.get("fields", []))
        else:
            total += len(schema.get("fields", []))
    return total


def _build_knowledge_preview(knowledge_docs: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for doc in knowledge_docs:
        asset_path = doc.get("asset_path", "")
        content = doc.get("content", "")
        parts.append(f"### {asset_path}\n\n{content}")
    return "\n\n".join(parts)


def _extract_response_text(ai_message: Any) -> str:
    if isinstance(ai_message.content, str):
        return ai_message.content
    if isinstance(ai_message.content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in ai_message.content
        )
    return ""


def _try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _validate_enriched_catalog(
    original: dict[str, Any],
    enriched: dict[str, Any],
    *,
    expected_keys: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []

    if expected_keys is None:
        expected_keys = {"task_id", "assets", "schemas", "relationships", "knowledge_documents"}
    for key in sorted(expected_keys):
        if key not in enriched:
            errors.append(f"Missing top-level key: {key}")
    if errors:
        return errors

    orig_schemas = original.get("schemas", [])
    enr_schemas = enriched.get("schemas", [])
    if len(enr_schemas) != len(orig_schemas):
        errors.append(
            f"Schema count mismatch: original={len(orig_schemas)}, "
            f"enriched={len(enr_schemas)}"
        )
        return errors

    for si, (orig_s, enr_s) in enumerate(zip(orig_schemas, enr_schemas)):
        schema_label = f"schemas[{si}] ({orig_s.get('asset_path', '?')})"
        if orig_s.get("kind") == "sqlite":
            orig_tables = orig_s.get("tables", [])
            enr_tables = enr_s.get("tables", [])
            if len(enr_tables) != len(orig_tables):
                errors.append(f"{schema_label}: table count mismatch")
                continue
            for ti, (orig_t, enr_t) in enumerate(zip(orig_tables, enr_tables)):
                table_label = f"{schema_label}.tables[{ti}] ({orig_t.get('name', '?')})"
                errors.extend(
                    _validate_fields(
                        orig_t.get("fields", []),
                        enr_t.get("fields", []),
                        table_label,
                    )
                )
        else:
            errors.extend(
                _validate_fields(
                    orig_s.get("fields", []),
                    enr_s.get("fields", []),
                    schema_label,
                )
            )

    return errors


def _validate_fields(
    orig_fields: list[dict[str, Any]],
    enr_fields: list[dict[str, Any]],
    parent_label: str,
) -> list[str]:
    errors: list[str] = []
    if len(enr_fields) != len(orig_fields):
        errors.append(
            f"{parent_label}: field count mismatch "
            f"({len(orig_fields)} vs {len(enr_fields)})"
        )
        return errors

    for fi, (orig_f, enr_f) in enumerate(zip(orig_fields, enr_fields)):
        field_label = f"{parent_label}.fields[{fi}] ({orig_f.get('name', '?')})"

        if enr_f.get("name") != orig_f.get("name"):
            errors.append(f"{field_label}: name changed")
        if enr_f.get("type") != orig_f.get("type"):
            errors.append(f"{field_label}: type changed")

        allowed_extra = {"description", "note"}
        extra_keys = set(enr_f.keys()) - set(orig_f.keys())
        if not extra_keys.issubset(allowed_extra):
            unexpected = extra_keys - allowed_extra
            errors.append(f"{field_label}: unexpected keys {unexpected}")

        desc = enr_f.get("description")
        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"{field_label}: description is missing or empty")
        note = enr_f.get("note")
        if note is not None and not isinstance(note, str):
            errors.append(f"{field_label}: note must be a string or omitted")

    return errors


def _merge_enrichments(
    original: dict[str, Any],
    enriched: dict[str, Any],
) -> dict[str, Any]:
    merged = json.loads(json.dumps(original, ensure_ascii=False))
    enr_schemas = enriched.get("schemas", [])

    for si, schema in enumerate(merged.get("schemas", [])):
        enr_s = enr_schemas[si] if si < len(enr_schemas) else None
        if enr_s is None:
            continue
        if schema.get("kind") == "sqlite":
            for ti, table in enumerate(schema.get("tables", [])):
                enr_t = enr_s.get("tables", [])[ti] if ti < len(enr_s.get("tables", [])) else None
                if enr_t is None:
                    continue
                for fi, field in enumerate(table.get("fields", [])):
                    enr_f = enr_t.get("fields", [])[fi] if fi < len(enr_t.get("fields", [])) else None
                    if enr_f is None:
                        continue
                    _copy_enrichment(field, enr_f)
        else:
            for fi, field in enumerate(schema.get("fields", [])):
                enr_f = enr_s.get("fields", [])[fi] if fi < len(enr_s.get("fields", [])) else None
                if enr_f is None:
                    continue
                _copy_enrichment(field, enr_f)

    return merged


def _copy_enrichment(field: dict[str, Any], enriched_field: dict[str, Any]) -> None:
    desc = enriched_field.get("description")
    if isinstance(desc, str) and desc.strip():
        field["description"] = desc.strip()
    note = enriched_field.get("note")
    if isinstance(note, str) and note.strip():
        field["note"] = note.strip()
