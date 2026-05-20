"""Answer validation agent.

After the main agent submits an answer, this module asks an LLM to check
format-level issues and reports actionable feedback. It does not fix answers
itself.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.model_retry import invoke_model_with_retries

logger = logging.getLogger(__name__)


ANSWER_VALIDATOR_SYSTEM_PROMPT = """\
You are an answer validation agent for a data analysis benchmark.
Your job is to review a submitted answer table and check for formatting and answer-scope issues.
You do NOT fix the answer. You only report whether it passes validation or not.

## Validation Rules

### 1. Strict output-column scope
- The submitted answer must contain ONLY columns that directly answer the original question.
- Reject columns that are merely proof, evidence, join keys, filter conditions, lookup helpers, or related context.
- Do NOT accept a full source-table schema unless the question explicitly asks for all fields, all details, records, rows, or complete transaction information.
- If the question asks for a specific measure, attribute, name, ID, date, count, status, category, or value, the answer should include only that requested output column or those requested output columns.
- If the answer includes columns only to explain why rows matched the filter, report them as unnecessary columns.
- Be strict: when in doubt, prefer reporting likely extra columns rather than accepting evidence columns.
- Example: for "List all the withdrawals in cash transactions that the client with the id 3356 makes", columns such as client_id, account_id, type, operation, balance, bank, account, and k_symbol are filter/proof/context columns, not direct requested output columns, unless the question explicitly asks for them.

### 2. Date/DateTime format (ISO 8601)
- All date values must be in strict ISO 8601 format.
- Dates like "2024-3-1" or "2024-1-5" are INVALID.
- Use zero-padded dates, e.g. "2024-03-01" or "2024-01-05".
- DateTime with timezone must be in UTC ending with "Z", e.g. "2024-03-01T12:00:00Z".
- DateTime without timezone should be in ISO format, e.g. "2024-03-01T12:00:00".
- Check every cell value that looks like a date or datetime.

### 3. String case sensitivity
- String values are case-sensitive. Do not flag case differences as issues.
- Only check date formatting, not text content correctness.

### 4. Name fields
- If a name field is split into first_name and last_name columns, that is acceptable.
- If a name field is a single full_name column, that is also acceptable.
- Do NOT flag name field format as an issue unless the question explicitly requires a specific format.

### 5. Requested-answer relevance
- The submitted answer must directly answer what the original question asks for, not merely identify the row that would contain the answer.
- If the question asks for an entity, item, record, message, comment, review, note, description, title, name, body, or other content-bearing object "itself", an ID-only answer is insufficient unless the question explicitly asks for an id, identifier, key, number, or code.
- If the answer contains only identifiers, join keys, filter fields, ranking metrics, or other proof/context columns while the question asks for a human-readable/content value, report it as invalid.
- When rejecting an unrelated or incomplete answer, explicitly tell the main agent what kind of answer is needed, such as a Text, Body, Content, Description, Name, Title, count, date, or other requested value inferred from the question wording.
- Do not judge exact cell-value correctness against unseen source data, but do reject answer columns whose semantics do not match the requested output.

### 6. Non-empty answer rows
- Reject any submitted answer with zero data rows, even if it has column headers.
- An empty table would write a prediction.csv with only a header row and no prediction data, which is invalid.
- When rejecting an empty answer, tell the main agent to submit the most likely data rows based on the available evidence instead of submitting an empty answer.

### 7. Percentage format
- Numeric values that represent percentages must NOT include a "%" suffix.
- Values like "12.5%", "3%", "-1.2%" are INVALID.
- Percentages must be written as plain numbers: "12.5", "3", "-1.2".
- Check every cell value that contains "%" and flag it.

## Output Format

You MUST respond with ONLY a valid JSON object (no markdown fences, no explanation):
{
  "valid": true,
  "issues": []
}

OR if there are issues:
{
  "valid": false,
  "issues": [
    "Issue description 1: explain what is wrong and how it should be fixed",
    "Issue description 2: ..."
  ]
}

- "valid": true if the answer passes all validation checks, false otherwise.
- "issues": a list of human-readable issue descriptions, empty if valid is true.
- Each issue should describe what is wrong, which column/row/value is affected, and how to fix it.

"""

"""

你是数据分析基准测试的答案验证智能体。
你的工作是检查提交的答案表格，并查找格式和答案范围问题。
你**不要**修复答案。你只报告它是否通过了验证。

## 验证规则

### 1. 严格的输出列范围
- 提交的答案必须**仅包含**直接回答原问题的列。
- 拒绝仅仅是证明、证据、连接键、过滤条件、查找辅助或相关上下文的列。
- **不要**接受完整的源表模式，除非问题明确要求所有字段、所有细节、记录、行或完整的交易信息。
- 如果问题要求特定的指标、属性、名称、ID、日期、计数、状态、类别或值，则答案应仅包含该请求的输出列或那些请求的输出列。
- 如果答案包含的列只是为了解释为什么行匹配过滤器，请将它们报告为不必要的列。
- 如果问题没有明确要求，请拒绝任何可能是过滤/证明/上下文列的列。
- 严格一点：如果有疑问，倾向于报告可能是多余的列，而不是接受证据列。

### 2. 日期/日期时间格式 (ISO 8601)
- 所有日期值必须采用严格的 ISO 8601 格式。
- 像 "2024-3-1" 或 "2024-1-5" 这样的日期是**无效的**。
- 使用补零的日期，例如 "2024-03-01" 或 "2024-01-05"。
- 带时区的日期时间必须是 UTC 且以 "Z" 结尾，例如 "2024-03-01T12:00:00Z"。
- 不带时区的日期时间应采用 ISO 格式，例如 "2024-03-01T12:00:00"。
- 检查每一个看起来像日期或日期时间的单元格值。

### 3. 字符串大小写敏感性
- 字符串值是区分大小写的。不要将大小写差异标记为问题。
- 只检查日期格式，不检查文本内容的正确性。

### 4. 姓名（Name）字段
- 如果姓名字段分成 first_name 和 last_name 两列，那是可以接受的。
- 如果姓名字段是单一的 full_name 列，那也是可以接受的。
- 除非问题明确要求特定的格式，否则**不要**将姓名字段格式标记为问题。

### 5. 请求答案相关性
- 提交的答案必须直接回答原问题所要求的内容，而不是仅仅标识“答案所在的那一行”。
- 如果问题询问某个实体、项目、记录、消息、评论、笔记、描述、标题、姓名、正文或其他承载内容的对象“本身”，则仅提交 ID 是不充分的，除非问题明确要求 id、identifier、key、number 或 code。
- 如果答案只包含标识符、连接键、过滤字段、排序指标或其他证明/上下文字段，而问题要求的是人类可读值或内容值，请判定为无效。
- 当因为答案无关或不完整而打回时，必须明确告诉主 agent 需要什么类型的答案，例如根据题目措辞推断出的 Text、Body、Content、Description、Name、Title、计数、日期或其他请求值。
- 不要在没有源数据的情况下判断单元格具体值是否正确；但如果答案列的语义与题目请求的输出不匹配，必须打回。

### 6. 非空答案行
- 如果提交的答案没有任何数据行，即使有列名，也必须打回。
- 空表会写出只有表头、没有预测数据行的 prediction.csv，这是无效提交。
- 打回空答案时，必须提醒主 agent：不要提交空答案，应根据已有证据提交最有可能的数据行。

### 7. 百分比格式
- 代表百分比的数值**不得**包含 "%" 后缀。
- 像 "12.5%"、"3%"、"-1.2%" 这样的值是**无效的**。
- 百分比必须写为纯数字："12.5"、"3"、"-1.2"。
- 检查每一个包含 "%" 的单元格值并将其标记。

## 输出格式

你**必须**仅响应一个有效的 JSON 对象（没有 markdown 围栏，没有解释）：
{
  "valid": true,
  "issues": []
}

或者，如果存在问题：
{
  "valid": false,
  "issues": [
    "问题描述 1：说明哪里错了以及如何修复",
    "问题描述 2：..."
  ]
}

- "valid": 如果答案通过了所有验证检查则为 true，否则为 false。
- "issues": 一个人类可读的问题描述列表，如果 valid 为 true，则该列表为空。
- 每个问题应描述哪里出错、影响了哪一列/行/值，以及如何修复。
"""


def _build_validation_request(
    question: str,
    answer: dict[str, Any],
    validation_history: list[dict[str, Any]] | None = None,
) -> str:
    parts = [
        f"## Original Question\n{question}\n",
        f"## Submitted Answer\n```json\n{json.dumps(answer, ensure_ascii=False, indent=2)}\n```\n",
    ]
    if validation_history:
        parts.append(
            "## Previous Validation History\n"
            "The following previous validation decisions were made in this same task run. "
            "Use them to keep the same validation criteria and judgment style for the current answer.\n"
            "```json\n"
            f"{json.dumps(validation_history, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    parts.append(
        "Please validate the answer according to the rules. "
        "Respond with ONLY a JSON object, no markdown fences."
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

    if not isinstance(parsed, dict):
        logger.warning("Answer validator JSON response is not an object.")
        return None

    if "valid" not in parsed:
        logger.warning("Answer validator JSON response is missing `valid`.")
        return None

    return parsed


def validate_answer(
    *,
    model: BaseChatModel,
    question: str,
    answer: dict[str, Any],
    validation_history: list[dict[str, Any]] | None = None,
    retry_event_callback: Any | None = None,
) -> dict[str, Any]:
    """Validate a submitted answer with one LLM call.

    If validation itself fails, the function returns ``valid=True`` so the main
    task flow is not blocked by the checker.
    """
    messages = [
        SystemMessage(content=ANSWER_VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(content=_build_validation_request(question, answer, validation_history)),
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
            "validator_error": "Failed to parse validator response",
            "raw_response": response_text if response_text else None,
        }

    return {
        "valid": bool(parsed.get("valid", True)),
        "issues": list(parsed.get("issues", [])),
        "raw_response": response_text,
    }
