"""Question analysis agent.

Before the main agent receives the question, this module performs a single LLM
call to decompose the question into entities, filters, and requested output.
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
Your job is to decompose a raw user question into three key components:
entities, filters, and requested output.

## Output Format

You MUST respond with ONLY a valid JSON object (no markdown fences, no explanation):

{
  "entities": ["entity1", "entity2"],
  "filters": ["filter description 1", "filter description 2"],
  "requested_output": "what the question is asking for"
}

Rules:
- "entities": concrete people, places, organizations, products, codes mentioned in the question.
- "filters": conditions that narrow down the data (e.g. "status = active", "date after 2023").
  Express each as a natural language condition.
- "requested_output": describe what the answer should contain (e.g. "list of customer names",
  "total transaction amount", "most recent date").
- If a field is ambiguous, make your best guess and note it briefly.
"""

"""
---

你是一个数据分析基准测试的问题分析助手。
你的工作是将原始用户问题分解为三层结构：实体、筛选条件、请求输出。

## 输出格式

你**必须**仅响应一个有效的 JSON 对象（没有 markdown 围栏，没有解释）：

{
  "entities": ["实体1", "实体2"],
  "filters": ["筛选条件描述 1", "筛选条件描述 2"],
  "requested_output": "问题要求输出什么"
}

规则：
- "entities": 问题中提到的具体人物、地点、组织、产品、代码。
- "filters": 缩小数据范围的条件（例如"状态为活跃"、"2023年之后的日期"）。每条用自然语言描述。
- "requested_output": 描述答案应包含什么（例如"客户姓名列表"、"交易总额"、"最近的日期"）。
- 如果某个字段含糊不清，做出最佳猜测并简要说明。
"""


def analyze_question(
    *,
    model: BaseChatModel,
    question: str,
) -> dict[str, Any]:
    messages = [
        SystemMessage(content=QUESTION_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze the following question:\n\n{question}"),
    ]

    try:
        ai_message = invoke_model_with_retries(model, messages)
    except Exception as exc:
        logger.warning("Question analyzer LLM call failed; skipping analysis: %s", exc)
        return _fallback_result(question, str(exc))

    response_text = ""
    if isinstance(ai_message.content, str):
        response_text = ai_message.content
    elif isinstance(ai_message.content, list):
        response_text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in ai_message.content
        )

    parsed = _parse_analyzer_response(response_text)
    if parsed is None:
        logger.warning("Question analyzer response parsing failed; skipping analysis.")
        return _fallback_result(question, "Failed to parse analyzer response")

    return parsed


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
    }


def _fallback_result(question: str, error: str) -> dict[str, Any]:
    del question, error
    return {
        "entities": [],
        "filters": [],
        "requested_output": "",
    }
