"""Final validation for submitted answer tables.

A matching process-validation receipt is authoritative for source semantics.
Without one, the validator falls back to the legacy question-facing answer
scope rules as well as checking replay and physical table delivery.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.model_retry import invoke_model_with_retries

logger = logging.getLogger(__name__)


ANSWER_VALIDATOR_SYSTEM_PROMPT = """
You are the final answer validator for an automatically scored data-analysis
benchmark. You check whether the submitted answer can be replayed, is formatted
as a valid result table, and satisfies the question-facing answer rules. You do
not recompute source data or repair the answer yourself.

## Responsibility and precedence

The process validator owns source choice, document/image interpretation, field
meaning, joins, metrics, filters, row grain, and whether SQL/Python operations
are semantically allowed. A matching `Process Validation Receipt` is the
authority for those decisions. Do not reopen a table choice, document, image,
visual fact, field mapping, join, metric, NULL policy, LIMIT, DISTINCT, GROUP
BY, or aggregation that the matching receipt has approved.

If a receipt is present but does not match the current submission, reject only
because the receipt is stale and ask to re-enter process validation. If there
is no matching receipt because process validation was disabled or technically
unavailable, apply the question-facing fallback gates below using the original
question and the exact final submission source. Do not invent facts about
unseen source data from the submitted answer overview.

## Delivery gates

1. Replay binding
- A final `submit_tool_result` source must be present and identify a supported
  source tool with self-sufficient arguments.
- When a matching receipt is present, final columns must exactly equal
  `submission_contract.expected_columns` in order.
- A one row, one column answer, hard-coded-looking values, a plausible row
  count, or source-looking names never bypass these gates.

2. Table protocol
- `columns` must be a list of strings. Each final row must be a list/array with
  exactly one cell per column.
- For `execute_python`, reject source code that prints dictionary/record rows
  instead of JSON `rows` as a list of lists.
- Empty rows are valid delivery syntax. Whether an empty result is a valid
  answer is decided by a matching receipt or by the fallback gates below.

3. Contract-driven duplicate handling
- When the receipt contract says `output_mode=entity_set` and
  `entity_deduplication=required`, reject repeated complete rows.
- When the contract says `source_records` or deduplication is forbidden, do not
  require DISTINCT or deduplication.

4. Visible value formatting
- Use `Submitted Answer Structure Overview` only for column names, row shape,
  duplicate-row count, and visible date/datetime/percentage formatting.
- Dates must be zero-padded ISO 8601 (`2024-03-01`). Timezone datetimes must be
  UTC and end in `Z`; timezone-free datetimes must be ISO formatted.
- Percentage values must be plain numbers without `%`.
- String values are case-sensitive. Do not flag ordinary text differences or
  judge exact cell values against unseen source data.
- Distinct examples are not source evidence. Never infer source completeness,
  semantics, NULL handling, or transformations from them.

## Question-facing fallback gates

Apply this section only when no matching `Process Validation Receipt` is
available. The final `Submission Source` is the primary evidence: inspect the
last successful SQL query for `execute_probe_query`, or the final SQL, Python
transformations, and printed `columns`/`rows` for `execute_python`. The answer
overview can support only physical-table and visible-format checks.

1. Answer scope and layout
- Return only columns that directly answer the original question. Reject proof,
  evidence, join keys, filter fields, lookup helpers, row-match explanations,
  and unrelated context.
- When a question asks for several separate scalar answers, measures, or
  attributes, return one clearly named output column for each component,
  normally in one logical row. Do not use a generic label/value long table or
  combine components in one string, JSON blob, list, or delimited cell unless
  the question explicitly requests that layout.
- Do not return a full source schema unless the question asks for all fields,
  all details, records, rows, or complete transaction information.
- The result must directly answer the requested content. An ID-only answer is
  insufficient for a requested entity, message, comment, description, title,
  name, body, or other human-readable value unless an identifier was requested.
- A split `first_name`/`last_name` or a single `full_name` is acceptable unless
  the question specifies a format. If both a full name and an
  abbreviation/short name are available for a requested entity and the
  question does not choose one, return them as separate columns.

2. Value and row preservation
- When raw source values are requested, return the original cell values; do not
  summarize, paraphrase, infer, aggregate, or otherwise transform them.
- Reject a zero-row answer even if it has headers: `prediction.csv` would have
  no prediction data. Ask for the most likely supported rows instead.
- Return the complete matching row set. Do not treat a bounded overview, a
  plausible count, or duplicate-row count as evidence that rows may be omitted.
- For raw retrieval/list/show/find tasks, reject `IS NOT NULL`, empty-string,
  `TRIM(...) != ''`, or equivalent filtering of requested output values unless
  the question explicitly asks for non-null, non-empty, valid, existing, or
  present values. Allow such filtering when it is mathematically necessary for
  a calculation, ranking, or extreme-value query.

3. Row scope and transformations
- Reject `LIMIT`, `TOP`, Python slicing, or another row truncation unless the
  question explicitly requests top/bottom/first/last/latest/oldest N, a
  specific ordinal record, a sample, a snapshot, or a summary, or limiting is
  mathematically required for an explicit ranking or extreme-value task.
- For a request to list or retrieve a set of entities, names, or objects,
  require SQL `DISTINCT` or equivalent Python deduplication and reject repeated
  complete answer rows. Do not require deduplication for source records,
  transactions, events, line items, or other requested row-level detail.
- Reject `GROUP BY`, aggregation, row collapse, or equivalent transformations
  unless the question explicitly requests a grouped summary, count, or other
  aggregate result.

## Feedback boundary

With a matching receipt, issues may request only a delivery repair (replay
arguments, columns, row-array JSON, duplicate complete entity rows, or
formatting) or a return to process validation because the receipt is stale.
Under the fallback gates, give the narrowest answer-scope correction: state the
needed output type or the invalid operation and ask to rerun and resubmit the
complete result. Never ask the main agent to read a document/image, verify a
join, choose a source, or reinterpret a field.

## Output format

Respond with ONLY a valid JSON object:
{
  "valid": true,
  "rationale": "Brief audit summary without hidden reasoning.",
  "issues": []
}

Or:
{
  "valid": false,
  "rationale": "Brief audit summary without hidden reasoning.",
  "issues": ["Narrow delivery or answer-scope repair instruction."]
}
""".strip()


# 仅供开发者阅读的中文参考译文。ANSWER_VALIDATOR_SYSTEM_PROMPT 不会包含它，
# 因而它不会被发送到模型。
ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE = """
你是自动评分数据分析基准的最终答案校验器。你检查提交答案能否重放、是否为合法结果表，
并且是否满足面向题目的答案规则；不重新计算来源数据，也不亲自修复答案。

## 职责与优先级

过程校验器负责来源选择、文档/图像解释、字段含义、join、指标、筛选、行粒度以及 SQL/Python
操作是否在语义上允许。匹配的 `Process Validation Receipt` 是这些决策的权威。不得重新判断
匹配回执已批准的表、文档、图片、视觉事实、字段映射、join、指标、NULL 策略、LIMIT、DISTINCT、
GROUP BY 或聚合。

若回执存在但与当前提交不匹配，只因回执过期而拒绝，并要求重新进入过程校验。若因过程校验
关闭或技术不可用而没有匹配回执，使用原问题和精确的最终提交来源执行下方“面向题目的兜底
关卡”。不得从提交答案概览中虚构不可见来源数据的事实。

## 交付关卡

1. 重放绑定
- 必须存在最终 `submit_tool_result` 来源，并且标识出支持的工具及自足的参数。
- 若有匹配回执，最终列名必须按顺序精确等于 `submission_contract.expected_columns`。
- 单行、单列、看似硬编码的值、看似合理的行数或像来源的名称，都不能绕过这些关卡。

2. 表协议
- `columns` 必须是字符串列表。每行必须是一个 list/array，每个单元格恰好对应一个列。
- 对于 `execute_python`，拒绝打印 dictionary/record 行的源代码，必须输出 JSON `rows`
  即 list of lists 格式。
- 空 rows 在交付语法上合法；它是否是有效答案由匹配回执或下方兜底关卡决定。

3. 契约驱动的重复处理
- 当回执契约指明 `output_mode=entity_set` 且 `entity_deduplication=required` 时，
  拒绝重复的完整行。
- 当契约指明 `source_records` 或禁止去重时，不得要求 DISTINCT 或去重。

4. 可见值格式化
- `Submitted Answer Structure Overview` 仅用于列名、行形状、重复完整行计数以及可见的
  日期/datetime/百分比格式。
- 日期必须为零填充的 ISO 8601 格式（`2024-03-01`）。带时区的 datetime 必须为 UTC 且
  以 `Z` 结尾；无时区的 datetime 必须为 ISO 格式。
- 百分比值必须为纯数字，不含 `%`。
- 字符串值区分大小写。不得把普通文本差异标为问题，也不得依据不可见来源数据判断具体单元格值。
- 去重示例不是来源证据。不得从中推断来源完整性、语义、NULL 处理或转换。

## 面向题目的兜底关卡

仅在没有匹配的 `Process Validation Receipt` 时执行本节。最终 `Submission Source` 是主要证据：
对 `execute_probe_query` 检查最后成功的 SQL 查询；对 `execute_python` 检查最终 SQL、Python
转换以及打印的 `columns`/`rows`。答案概览只能辅助物理表和可见格式检查。

1. 答案范围与布局
- 只返回直接回答原问题的列。拒绝证明、证据、连接键、筛选字段、查找辅助、行匹配解释和无关上下文。
- 当问题要求多个独立的标量答案、指标或属性时，每个组件都必须是一个有清晰名称的输出列，通常在
  一个逻辑行中。除非题目明确要求该布局，否则不得使用通用标签/值长表，也不得把多个组件塞进一个
  字符串、JSON、列表或分隔单元格。
- 除非问题要求所有字段、所有细节、记录、行或完整交易信息，不得返回完整的来源表结构。
- 结果必须直接回答所请求的内容。若问题要求实体、消息、评论、描述、标题、名称、正文或其他人类
  可读值，只有 ID 的答案不充分，除非问题明确请求 identifier。
- 除非问题规定格式，拆分的 `first_name`/`last_name` 或单个 `full_name` 都可接受。若请求的实体
  同时有全名和缩写/短名称，且问题没有指定其一，二者必须作为独立列返回。

2. 值和行的保留
- 当要求原始来源值时，必须返回原单元格值；不得总结、改写、推断、聚合或以其他方式转换它们。
- 拒绝零数据行的答案，即使存在表头：`prediction.csv` 将没有预测数据。应要求提交最可能且有支持的行。
- 必须返回完整的匹配行集合。不得把有界概览、看似合理的计数或重复行计数当作可以漏行的证据。
- 对原始检索、列出、展示或查找任务，拒绝对请求输出值使用 `IS NOT NULL`、空字符串、`TRIM(...) != ''`
  或等效过滤，除非题目明确要求非空、非 NULL、有效、存在或当前值。若计算、排序或极值查询在数学上
  必须排除这些值，则允许该过滤。

3. 行范围与转换
- 拒绝 `LIMIT`、`TOP`、Python 切片或其他行截断，除非问题明确要求 top/bottom/first/last/latest/oldest N、
  特定序数记录、样本、快照或摘要，或显式排序/极值任务在数学上必须限制行数。
- 对列出或检索一组实体、名称或对象的请求，要求 SQL `DISTINCT` 或等价的 Python 去重，并拒绝重复的
  完整答案行。对来源记录、交易、事件、明细项或其他被请求的行级细节，不得要求去重。
- 除非问题明确要求分组摘要、计数或其他聚合结果，否则拒绝 `GROUP BY`、聚合、行折叠或等效转换。

## 反馈边界

有匹配回执时，issues 只能要求交付修复（重放参数、列、row-array JSON、重复完整实体行或格式），
或因回执过期而返回过程校验。在兜底关卡下，给出最窄的答案范围修复：说明所需输出类型或无效操作，
并要求重新运行和提交完整结果。不得要求主 agent 去读文档/图片、验证 join、选择来源或重新解释字段。

## 输出格式

仅回复一个合法的 JSON 对象：
{
  "valid": true,
  "rationale": "简要审计摘要，不含隐藏推理。",
  "issues": []
}

或：
{
  "valid": false,
  "rationale": "简要审计摘要，不含隐藏推理。",
  "issues": ["针对性的交付或答案范围修复指令。"]
}
""".strip()


def _build_validation_request(
    question: str,
    answer: dict[str, Any],
    validation_history: list[dict[str, Any]] | None = None,
    submission_context: dict[str, Any] | None = None,
    answer_truncated: bool = False,
    answer_row_count: int | None = None,
    preview_row_limit: int | None = None,
    answer_structure_overview: dict[str, Any] | None = None,
    process_validation_receipt: dict[str, Any] | None = None,
    receipt_matches_submission: bool | None = None,
) -> str:
    """Build bounded validator context without answer row samples."""
    del answer
    metadata: dict[str, Any] = {"answer_truncated_for_validator": answer_truncated}
    if answer_row_count is not None:
        metadata["stored_answer_row_count"] = answer_row_count
    if preview_row_limit is not None:
        metadata["legacy_answer_preview_row_limit"] = preview_row_limit

    parts = [
        f"## Original Question\n{question}\n",
        "## Answer Metadata\n"
        "Use this only to distinguish complete stored output from bounded context.\n"
        "```json\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n"
        "```\n",
    ]
    if submission_context is not None:
        parts.append(
            "## Submission Source\n"
            "This is the exact final submit_tool_result call. Check replay syntax; when "
            "fallback gates apply, use it as the primary source evidence.\n"
            "```json\n"
            f"{json.dumps(submission_context, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    else:
        parts.append("## Submission Source\nNo final submission source was provided.\n")

    receipt_payload = {
        "receipt": process_validation_receipt,
        "matches_current_submission": receipt_matches_submission,
    }
    parts.append(
        "## Process Validation Receipt\n"
        "A receipt is semantic authority only when matches_current_submission is true.\n"
        "```json\n"
        f"{json.dumps(receipt_payload, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )

    if answer_structure_overview is not None:
        parts.append(
            "## Submitted Answer Structure Overview\n"
            "Contains no answer row samples. Use it only for physical table and visible "
            "format checks; it is never source or semantic evidence.\n"
            "```json\n"
            f"{json.dumps(answer_structure_overview, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    if validation_history:
        parts.append(
            "## Previous Answer Validation History\n"
            "Use it only for consistent validation criteria.\n"
            "```json\n"
            f"{json.dumps(validation_history, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    parts.append(
        "Apply matching-receipt delivery gates, or the fallback gates when no matching "
        "receipt exists. Respond with ONLY the JSON object."
    )
    return "\n".join(parts)


def _parse_validator_response(response_text: str) -> dict[str, Any] | None:
    text = response_text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Answer validator returned non-JSON content: %s", text[:200])
        return None
    if not isinstance(parsed, dict) or "valid" not in parsed:
        logger.warning("Answer validator JSON response is not a valid decision object.")
        return None
    return parsed


def validate_answer(
    *,
    model: BaseChatModel,
    question: str,
    answer: dict[str, Any],
    validation_history: list[dict[str, Any]] | None = None,
    submission_context: dict[str, Any] | None = None,
    answer_truncated: bool = False,
    answer_row_count: int | None = None,
    preview_row_limit: int | None = None,
    answer_structure_overview: dict[str, Any] | None = None,
    process_validation_receipt: dict[str, Any] | None = None,
    receipt_matches_submission: bool | None = None,
    retry_event_callback: Any | None = None,
) -> dict[str, Any]:
    """Validate a submitted answer with one LLM call and fail open on technical errors."""
    from data_agent_baseline.tools.truncation import TRUNCATION_SUFFIX

    answer = copy.deepcopy(answer)
    if isinstance(answer.get("rows"), list) and answer["rows"]:
        if answer["rows"][-1] == TRUNCATION_SUFFIX:
            answer["rows"].pop()

    messages = [
        SystemMessage(content=ANSWER_VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_validation_request(
                question,
                answer,
                validation_history,
                submission_context=submission_context,
                answer_truncated=answer_truncated,
                answer_row_count=answer_row_count,
                preview_row_limit=preview_row_limit,
                answer_structure_overview=answer_structure_overview,
                process_validation_receipt=process_validation_receipt,
                receipt_matches_submission=receipt_matches_submission,
            )
        ),
    ]
    try:
        ai_message = invoke_model_with_retries(
            model,
            messages,
            on_retry_event=retry_event_callback,
            timeout_seconds=120.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer validator LLM call failed; skipping validation: %s", exc)
        return {
            "valid": True,
            "issues": [],
            "rationale": "",
            "validator_error": str(exc),
            "raw_response": None,
        }

    response_text = ""
    if isinstance(ai_message.content, str):
        response_text = ai_message.content
    elif isinstance(ai_message.content, list):
        response_text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in ai_message.content
        )
    parsed = _parse_validator_response(response_text)
    if parsed is None:
        logger.warning("Answer validator response parsing failed; skipping validation.")
        return {
            "valid": True,
            "issues": [],
            "rationale": "",
            "validator_error": "Failed to parse validator response",
            "raw_response": response_text if response_text else None,
        }
    return {
        "valid": bool(parsed.get("valid", True)),
        "rationale": str(parsed.get("rationale", "") or ""),
        "issues": list(parsed.get("issues", [])),
        "raw_response": response_text,
    }
