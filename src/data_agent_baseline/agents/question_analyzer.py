"""Question analysis agent.

Before the main agent receives the question, this module performs a single LLM
call to decompose the question into entities, filters, requested output, and
candidate field matches.  Candidates are *hypotheses only* — the main agent
must verify them with ``lookup_schema`` and actual data before choosing fields.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.model_retry import invoke_model_with_retries

logger = logging.getLogger(__name__)

QUESTION_ANALYZER_SYSTEM_PROMPT = """\
You are a question analysis assistant for a data analysis benchmark.

You will receive:
1. A raw user question.
2. Optionally, a schemas listing extracted from a data catalog. Each line shows \
a field path (asset_path.field_name), its type, field statistics (cardinality, \
missing count, primary key flag), sample distinct values, numeric range, and \
optionally a description and note.

Your job is to analyze the question and list plausible candidate fields from the schemas.

You MUST NOT decide the final field binding.
You MUST NOT rewrite the question.
You MUST NOT produce SQL or an execution plan.
You MUST NOT invent fields that are not present in the schemas.

## Output Format

You MUST respond with ONLY a valid JSON object (no markdown fences, no explanation):

{
  "entities": ["entity1", "entity2"],
  "filters": ["filter description 1", "filter description 2"],
  "requested_output": "what the question is asking for",
  "field_candidates": [
    {
      "phrase": "natural language phrase from the question",
      "role": "entity | filter | requested_output | join_key | ambiguous",
      "candidates": [
        {
          "field": "asset_path.field_name or asset_path.table.field_name",
          "reason": "short reason based on field name, description, note, or schema context"
        }
      ]
    }
  ],
  "filters_candidates": [
    {
      "filter": "filter description matching one from the filters array",
      "candidates": [
        "asset_path.field_name = value",
        "asset_path.table.field_name > 5"
      ]
    }
  ]
}

## Rules

- entities: concrete people, places, organizations, products, codes mentioned.
- filters: conditions that narrow down the data. Express each as a natural language condition.
- requested_output: what the answer should contain.
- For each important phrase in the question, list every reasonable candidate field from the provided schemas.
- If a phrase could plausibly refer to different entities, attributes, or data grains,
  include candidates for each plausible interpretation instead of choosing one meaning.
- Prefer recall over precision: candidate fields are hypotheses for downstream verification.
- Sort candidates from most plausible to less plausible, but do not claim any candidate is final.
- If a phrase is ambiguous, include multiple candidates and explain the ambiguity in reason.
- If no candidate field is available, use an empty candidates list.
- Keep reasons short (one sentence).
- Preserve the original question intent.
- The downstream agent will verify candidates with lookup_schema and actual data before choosing fields.
- Treat generic quantitative words as ambiguity triggers. Words such as "number", "count", "amount", "total", "quantity", "rank", "position", "order", "index", "score", "points", "level", "code", "id", "No.", "#", "top", "first", "second", "last", "less than", "greater than", "at least", and "at most" may refer to different numeric concepts.For these phrases, do not rely only on exact field-name matches. Include all schema fields whose name, type, range, description, note, table context, or sample values could plausibly represent that numeric concept.
- filters_candidates: for each filter in the filters array, list candidate field-level filter expressions. Each entry has a "filter" key matching one filter description, and a "candidates" list of strings in "field_path operator value" format (e.g. "a.csv.col > 29.00"). List every plausible field that could satisfy the filter condition. Multiple candidates for the same filter represent alternative field choices. Do NOT list a filter if no plausible field exists.
"""

"""
---

您是一个数据分析基准测试的问题分析助手。

您将收到：
1. 一个原始用户问题。
2. （可选）从数据目录中提取的模式清单。每行显示一个字段路径（asset_path.field_name）、\
其类型、字段统计信息（基数、缺失计数、主键标志）、样本唯一值、数值范围以及可选的描述和注释。

您的任务是分析问题并列出模式中可能的候选字段。

您**不得**决定最终的字段绑定。
您**不得**改写问题。
您**不得**生成 SQL 或执行计划。
您**不得**编造不存在于模式中的字段。

## 输出格式

您**必须**仅响应一个有效的 JSON 对象（没有 markdown 围栏，没有解释）：

{
  "entities": ["实体1", "实体2"],
  "filters": ["筛选条件描述 1", "筛选条件描述 2"],
  "requested_output": "问题要求输出什么",
  "field_candidates": [
    {
      "phrase": "问题中的自然语言短语",
      "role": "entity | filter | requested_output | join_key | ambiguous",
      "candidates": [
        {
          "field": "asset_path.field_name 或 asset_path.table.field_name",
          "reason": "基于字段名、描述、注释或模式上下文的简短理由"
        }
      ]
    }
  ],
  "filters_candidates": [
    {
      "filter": "与 filters 数组中某一条匹配的筛选描述",
      "candidates": [
        "asset_path.field_name = value",
        "asset_path.table.field_name > 5"
      ]
    }
  ]
}

## 规则

- entities：问题中具体提到的人物、地点、组织、产品、代码等。
- filters：用于筛选数据的条件，每条均以自然语言描述。
- requested_output：答案应包含的内容描述。
- 对于问题中的每个重要短语，从给定模式中列出所有合理候选字段。
- 如果某个短语可能对应不同实体、属性或数据粒度，应纳入每种合理解释的候选字段，而不是提前选择一个含义。
- 召回应偏向完整性而不是精确性：候选字段只是供下游验证的假设。
- 候选字段按从最可能到最不可能的顺序排列，但不要声称任何候选字段是最终确定的。
- 如果某个短语存在歧义，请列出多个候选字段并在 reason 中解释歧义。
- 如果没有可用的候选字段，请使用空的 candidates 列表。
- 保持 reason 简短（一句话）。
- 保持用户的原始意图不变。
- 下游 Agent 将在选择字段之前通过 lookup_schema 和实际数据来验证候选字段。
- 将通用的量化词视为歧义触发因素。诸如“number”“count”“amount”“total”“quantity”“rank”“position”“order”“index”“score”“points”“level”“code”“id”“No.”“#”“top”“first”“second”“last”“less than”“greater than”“at least”和“at most”等词语，可能指代不同的数值概念。对于这类短语，不应仅依赖字段名的精确匹配；应纳入所有其名称、类型、取值范围、描述、注释、所属表的上下文或样本值均有可能合理表征该数值概念的模式字段。
- filters_candidates：为 filters 数组中的每条筛选条件，列出候选的字段级筛选表达式。每项包含一个 "filter" 键（对应 filters 中的一条描述），以及一个 "candidates" 字符串列表，格式为 "字段路径 运算符 值"（如 "a.csv.col > 29.00"）。列出所有可能满足该筛选条件的字段。同一筛选条件的多个 candidate 表示不同的字段选择。如果没有合理的字段候选，则不列出该筛选条件。
"""


def analyze_question(
    *,
    model: BaseChatModel,
    question: str,
    schemas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schemas_str = _build_schemas_listing(schemas) if schemas else ""

    user_parts: list[str] = [
        "<user_question>\n"
        f"{question}\n"
        "</user_question>",
    ]
    if schemas_str:
        user_parts.append(
            "<schemas>\n"
            f"{schemas_str}\n"
            "</schemas>"
        )
    user_parts.append(
        "Analyze the question and return candidate fields only."
    )
    user_message = "\n\n".join(user_parts)

    messages = [
        SystemMessage(content=QUESTION_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        ai_message = invoke_model_with_retries(model, messages)
    except Exception as exc:
        logger.warning("Question analyzer LLM call failed; skipping analysis: %s", exc)
        return _fallback_result(question, str(exc))

    response_text = _extract_response_text(ai_message)
    parsed = _parse_analyzer_response(response_text)
    if parsed is None:
        logger.warning("Question analyzer response parsing failed; skipping analysis.")
        return _fallback_result(question, "Failed to parse analyzer response")

    if schemas:
        parsed["field_candidates"] = _validate_field_candidates(
            parsed.get("field_candidates", []), schemas
        )
        parsed["filters_candidates"] = _validate_filters_candidates(
            parsed.get("filters_candidates", []), schemas
        )

    return parsed


def _validate_filters_candidates(
    filters_candidates: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and clean filters_candidates: strip entries with invalid field paths."""
    if not isinstance(filters_candidates, list):
        return []

    valid_paths = _collect_valid_paths(schemas)
    cleaned: list[dict[str, Any]] = []

    for entry in filters_candidates:
        if not isinstance(entry, dict):
            continue
        filter_desc = entry.get("filter")
        if not isinstance(filter_desc, str) or not filter_desc.strip():
            continue

        raw_candidates = entry.get("candidates")
        if not isinstance(raw_candidates, list):
            continue

        valid_candidates: list[str] = []
        for cand in raw_candidates:
            if not isinstance(cand, str) or not cand.strip():
                continue
            field_path = _extract_field_path_from_filter(cand.strip())
            if field_path and field_path in valid_paths:
                valid_candidates.append(cand.strip())
            else:
                logger.info(
                    "Filters candidate stripped: filter=%r expr=%r field=%r not in schemas",
                    filter_desc, cand, field_path,
                )

        if not valid_candidates:
            continue
        cleaned.append({
            "filter": filter_desc.strip(),
            "candidates": valid_candidates,
        })

    return cleaned


def _extract_field_path_from_filter(expr: str) -> str | None:
    """Extract the field path part from a filter expression like 'a.b.C > 5'."""
    import re
    match = re.match(r'(\S+)\s*[=<>!]', expr)
    if match:
        return match.group(1)
    return None


def _extract_response_text(ai_message: Any) -> str:
    if isinstance(ai_message.content, str):
        return ai_message.content
    if isinstance(ai_message.content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in ai_message.content
        )
    return ""


def _build_schemas_listing(schemas: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    field_count = 0
    max_fields = 200

    for schema in schemas:
        asset = schema.get("asset_path", "?")
        kind = schema.get("kind", "?")

        if kind == "sqlite":
            for table in schema.get("tables", []):
                tname = table.get("name", "?")
                for field in table.get("fields", []):
                    if field_count >= max_fields:
                        break
                    lines.append(_format_field_line(asset, field, table_name=tname))
                    field_count += 1
        else:
            for field in schema.get("fields", []):
                if field_count >= max_fields:
                    break
                lines.append(_format_field_line(asset, field))
                field_count += 1

        if field_count >= max_fields:
            omitted = _count_total_fields(schemas) - max_fields
            lines.append(f"... ({omitted} more fields omitted)")
            break

    return "\n".join(lines)


def _format_field_line(
    asset: str, field: dict[str, Any], table_name: str | None = None
) -> str:
    name = field.get("name", "?")
    ftype = field.get("type", "?")
    prefix = f"{asset}.{table_name}" if table_name else asset
    line = f"{prefix}.{name} ({ftype})"

    stats_parts: list[str] = []
    card = field.get("cardinality")
    if card is not None:
        stats_parts.append(f"card={card}")
    missing = field.get("missing_count")
    if missing is not None and missing > 0:
        stats_parts.append(f"miss={missing}")
    if field.get("primary_key"):
        stats_parts.append("pk")
    if stats_parts:
        line += " | " + ", ".join(stats_parts)

    distinct = field.get("distinct_values")
    if isinstance(distinct, list) and distinct:
        sample = [str(v) for v in distinct[:5]]
        line += " | vals: " + ", ".join(sample)

    min_v = field.get("min_value")
    max_v = field.get("max_value")
    if min_v is not None and max_v is not None:
        line += f" | [{min_v}, {max_v}]"

    desc = field.get("description")
    if isinstance(desc, str) and desc.strip():
        line += f" — {desc.strip()}"
    note = field.get("note")
    if isinstance(note, str) and note.strip():
        line += f" [{note.strip()}]"
    return line


def _count_total_fields(schemas: list[dict[str, Any]]) -> int:
    total = 0
    for schema in schemas:
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                total += len(table.get("fields", []))
        else:
            total += len(schema.get("fields", []))
    return total


def _collect_valid_paths(schemas: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for schema in schemas:
        asset = schema.get("asset_path", "")
        if not asset:
            continue
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                tname = table.get("name", "")
                for field in table.get("fields", []):
                    fname = field.get("name", "")
                    if fname:
                        paths.add(f"{asset}.{fname}")
                        if tname:
                            paths.add(f"{asset}.{tname}.{fname}")
        else:
            for field in schema.get("fields", []):
                fname = field.get("name", "")
                if fname:
                    paths.add(f"{asset}.{fname}")
    return paths


def _validate_field_candidates(
    field_candidates: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(field_candidates, list):
        return []

    valid_paths = _collect_valid_paths(schemas)
    cleaned: list[dict[str, Any]] = []

    for entry in field_candidates:
        if not isinstance(entry, dict):
            continue
        phrase = entry.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            continue

        role = entry.get("role")
        raw_candidates = entry.get("candidates")
        if not isinstance(raw_candidates, list):
            cleaned.append(_make_candidate_entry(phrase, role, []))
            continue

        valid: list[dict[str, Any]] = []
        for cand in raw_candidates:
            if not isinstance(cand, dict):
                continue
            field = cand.get("field")
            if not isinstance(field, str) or field not in valid_paths:
                logger.info(
                    "Candidate field stripped: phrase=%r field=%r not in schemas",
                    phrase, field,
                )
                continue
            reason = cand.get("reason", "")
            entry_item: dict[str, Any] = {"field": field}
            if isinstance(reason, str) and reason.strip():
                entry_item["reason"] = reason.strip()
            valid.append(entry_item)

        if raw_candidates and not valid:
            continue
        cleaned.append(_make_candidate_entry(phrase, role, valid))

    return cleaned


def _make_candidate_entry(
    phrase: str,
    role: Any,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    entry: dict[str, Any] = {"phrase": phrase.strip(), "candidates": candidates}
    if isinstance(role, str) and role.strip():
        entry["role"] = role.strip()
    return entry


def _parse_analyzer_response(response_text: str) -> dict[str, Any] | None:
    text = response_text.strip()

    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Question analyzer returned non-JSON content: %s", text[:200])
        return None

    if not isinstance(parsed, dict):
        logger.warning("Question analyzer JSON response is not an object.")
        return None

    return {
        "entities": parsed.get("entities", []),
        "filters": parsed.get("filters", []),
        "requested_output": parsed.get("requested_output", ""),
        "field_candidates": parsed.get("field_candidates", []),
        "filters_candidates": parsed.get("filters_candidates", []),
    }


def _fallback_result(question: str, error: str) -> dict[str, Any]:
    del question, error
    return {
        "entities": [],
        "filters": [],
        "requested_output": "",
        "field_candidates": [],
        "filters_candidates": [],
    }
