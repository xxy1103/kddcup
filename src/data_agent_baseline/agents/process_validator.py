"""Process validation agent.

This module checks whether the main agent's recent work supports its current
direction or submitted answer. It does not recompute answers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.model_retry import invoke_model_with_retries

logger = logging.getLogger(__name__)


PROCESS_VALIDATOR_SYSTEM_PROMPT = """\
You are a process validation agent for a data analysis benchmark.
Your job is to audit the main agent's recent work for semantic drift, unsupported
assumptions, unresolved ambiguities, and mismatches between evidence and the
current direction or submitted answer.

You do NOT recompute the final answer. You only decide whether the recent
process provides enough evidence to continue or accept the submitted answer.

Block when there is a high-confidence process problem:
- The agent changed the meaning of the original question.
- A key field binding, entity resolution, metric definition, grain, time range,
  join path, or filter interpretation was assumed without data evidence.
- A prior ambiguity was not resolved with actual data probes.
- The submitted answer or current conclusion contradicts tool results.
- The answer targets a different output than the original question requested.

## Strict semantic-evidence rules

You MUST reject with valid=false when a key semantic assumption can change the
set of rows, filters, joins, grouping grain, aggregation value, or final answer,
and the recent trace does not show direct evidence for that assumption.

Direct evidence means at least one of:
- A schema, data dictionary, documentation, or knowledge document explicitly
  defines the field/metric/filter meaning.
- A tool probe reads relevant source records or columns and verifies the
  interpretation against concrete data.
- A prior ambiguity analysis explicitly resolved the meaning from provided
  knowledge and the main agent used that resolution.

The following are NOT evidence and MUST NOT justify valid=true:
- "standard industry convention"
- common sense about field names
- the model's prior knowledge
- the fact that the result count looks plausible
- consistency of the output shape or row count
- an assumption being labeled "low risk"
- the absence of an alternative field

Value exclusion rule:
- The main agent must not exclude numeric zero values or values that look
  implausible, unusual, or contrary to common sense unless the question,
  knowledge document, schema, or observed rows explicitly justify the exclusion.
- If such values were excluded without explicit evidence, treat it as a material
  unsupported assumption and set valid=false.

Multiple answers for extreme value questions:
- When the question asks for a maximum, minimum, top-N, or similar extreme value,
  and multiple rows share the same extreme value, the main agent MUST submit all
  of them. Submitting only one row when ties exist is a material error.
- If the recent trace shows a tie (equal values) but the submitted answer
  contains fewer rows than the evidence supports, set valid=false and instruct
  the agent to include all tied rows.

If the semantic ledger contains any unverified assumption that is material to
the answer, you MUST set valid=false. Do not put a material unverified
assumption in "unverified_assumptions" while also returning valid=true.

Example: If the question asks for purchases at a "unit price > 29.00" and the
agent uses a field named "Price", the process is invalid unless the trace shows
evidence that Price is unit price rather than total transaction amount. A
statement such as "Price is unit price by standard industry convention" is
insufficient and must be rejected.

Do not block for minor wording issues, style issues, or missing explanations
when the tool evidence is sufficient. Do not judge exact answer correctness by
recomputing the task from scratch; judge whether the process evidence supports
the semantics the agent relied on.

You MUST respond with ONLY a valid JSON object:
{
  "valid": true,
  "issues": [],
  "required_next_actions": [],
  "semantic_ledger": {
    "intent_summary": "...",
    "verified_claims": [],
    "unverified_assumptions": [],
    "unresolved_ambiguities": [],
    "drift_risks": []
  }
}

If there are blocking process issues:
{
  "valid": false,
    "issues": [
    "Describe the unsupported material assumption, unresolved ambiguity, or drift."
  ],
  "required_next_actions": [
    "Concrete next data-probe or documentation check the main agent should take."
  ],
  "semantic_ledger": {
    "intent_summary": "...",
    "verified_claims": [],
    "unverified_assumptions": [],
    "unresolved_ambiguities": [],
    "drift_risks": []
  }
}
"""

"""
您是一位用于数据分析基准测试的过程验证专员。您的职责是审核主代理近期的工作，以识别语义漂移、未经证实的假设、未澄清的歧义，以及证据与当前决策方向或所提交答案之间的不匹配。

您无需重新计算最终答案，仅需判断近期过程是否已提供充分的证据，足以继续推进或采纳所提交的答案。

当存在高置信度的过程问题时，应予以阻断：
- 代理改变了原问题的语义内涵；
- 在缺乏数据支撑的情况下，对关键字段的绑定、实体消歧、指标定义、粒度、时间范围、连接路径或过滤条件的解释作出了默认假设；
- 前期存在的歧义未通过实际的数据探查加以澄清；
- 所提交的答案或当前结论与工具输出结果相矛盾；
- 答案所指向的输出目标与原问题的要求不符。

## 严格的语义—证据规则

当某一关键语义假设可能改变行集、过滤条件、连接方式、分组粒度、聚合值乃至最终答案，而近期追踪日志中又未见针对该假设的直接证据时，您必须判定“valid=false”并予以拒绝。

所谓“直接证据”，至少满足以下之一：
- 某个模式、数据字典、文档或知识库明确界定了该字段/指标/过滤条件的含义；
- 工具探查读取了相关源记录或列，并基于具体数据验证了其解释；
- 前期的歧义分析已依据所提供的知识明确其含义，且主代理在后续过程中采用了该解析结果。

以下情形均不属于证据，不得作为判定“valid=true”的依据：
- “行业标准惯例”；
- 对字段名称的常识性推断；
- 模型的先验知识；
- 结果条数看似合理；
- 输出形状或行数的一致性；
- 将某项假设标注为“低风险”；
- 仅因缺乏备选字段而作出的推断。

数值排除规则：
- 主代理不得排除数值为 0 的值，或看起来不合理、异常、非常识的值，除非问题、知识文档、模式或观测到的数据行明确支持该排除。
- 如果在缺乏明确证据的情况下排除了此类值，应将其视为重要的未经证实假设，并判定“valid=false”。

最值问题的多答案规则：
- 当问题要求最大值、最小值、前N名或类似的最值查询，且多行数据共享同一最值时，主代理必须提交所有并列行。仅提交其中一行而遗漏其他并列行属于重要错误。
- 若近期追踪日志中显示存在并列值，但所提交答案的行数少于证据支持的数量，则判定 valid=false，并指示主代理纳入全部并列行。

若语义台账中存在任何与答案密切相关且尚未验证的假设，您必须判定“valid=false”。切勿在判定“valid=true”的同时，将此类重要未验证假设列入“unverified_assumptions”。

当工具提供的证据已足够充分时，不应因细微的措辞问题、风格瑕疵或说明缺失而予以阻断。亦无须通过从头复算任务来评判答案的精确性，而应着重考察过程证据是否支持代理所依赖的语义逻辑。

您必须仅以一个有效的JSON对象作出回复：

```json
{
  "valid": true,
  "issues": [],
  "required_next_actions": [],
  "semantic_ledger": {
    "intent_summary": "...",
    "verified_claims": [],
    "unverified_assumptions": [],
    "unresolved_ambiguities": [],
    "drift_risks": []
  }
}
```

如存在导致阻断的过程问题，则回复格式如下：

```json
{
  "valid": false,
  "issues": [
    "详细描述所涉及的未被证实的重要假设、未澄清的歧义或漂移现象"
  ],
  "required_next_actions": [
    "主代理应采取的具体下一步数据探查或文档核查措施"
  ],
  "semantic_ledger": {
    "intent_summary": "...",
    "verified_claims": [],
    "unverified_assumptions": [],
    "unresolved_ambiguities": [],
    "drift_risks": []
  }
}
```

"""

def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    """Keep process-validator context bounded and focused."""
    payload: dict[str, Any] = {
        "step_index": step.get("step_index"),
        "node": step.get("node"),
        "assistant_message": step.get("assistant_message"),
        "tool_calls": step.get("tool_calls", []),
        "tool_results": step.get("tool_results", []),
        "ok": step.get("ok"),
    }
    model_response = step.get("model_response")
    if isinstance(model_response, dict):
        payload["model_response"] = {
            key: model_response.get(key)
            for key in (
                "finish_reason",
                "tool_call_names",
                "content_preview",
                "reasoning_content",
            )
            if key in model_response
        }
    return payload


def _build_process_validation_request(
    *,
    question: str,
    answer: dict[str, Any] | None = None,
    ambiguity_analysis: dict[str, Any] | None = None,
    recent_steps: list[dict[str, Any]] | None = None,
    semantic_ledger: dict[str, Any] | None = None,
) -> str:
    parts = [
        f"## Original Question\n{question}\n",
        "## Submitted Answer\n"
        "```json\n"
        f"{json.dumps(answer, ensure_ascii=False, indent=2) if answer is not None else 'null'}\n"
        "```\n",
    ]
    if ambiguity_analysis:
        parts.append(
            "## Prior Ambiguity Analysis\n"
            "```json\n"
            f"{json.dumps(ambiguity_analysis, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    if semantic_ledger:
        parts.append(
            "## Previous Semantic Ledger\n"
            "```json\n"
            f"{json.dumps(semantic_ledger, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    compact_steps = [_compact_step(step) for step in (recent_steps or [])]
    parts.append(
        "## Recent Trace Steps\n"
        "```json\n"
        f"{json.dumps(compact_steps, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )
    parts.append(
        "Audit the process. If the submitted answer is null, decide whether the "
        "agent can continue on its current path. If an answer is present, decide "
        "whether the process evidence supports submitting it. Respond with ONLY "
        "the JSON object."
    )
    return "\n".join(parts)


def _parse_process_validator_response(response_text: str) -> dict[str, Any] | None:
    text = response_text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Process validator returned non-JSON content: %s", text[:200])
        return None

    if not isinstance(parsed, dict):
        logger.warning("Process validator JSON response is not an object.")
        return None
    if "valid" not in parsed:
        logger.warning("Process validator JSON response is missing `valid`.")
        return None
    return parsed


def validate_process(
    *,
    model: BaseChatModel,
    question: str,
    answer: dict[str, Any] | None = None,
    ambiguity_analysis: dict[str, Any] | None = None,
    recent_steps: list[dict[str, Any]] | None = None,
    semantic_ledger: dict[str, Any] | None = None,
    retry_event_callback: Any | None = None,
) -> dict[str, Any]:
    """Validate the recent reasoning process with one LLM call.

    If validation itself fails, return valid=True so the main benchmark flow is
    not blocked by the checker.
    """
    messages = [
        SystemMessage(content=PROCESS_VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_process_validation_request(
                question=question,
                answer=answer,
                ambiguity_analysis=ambiguity_analysis,
                recent_steps=recent_steps,
                semantic_ledger=semantic_ledger,
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
        logger.warning("Process validator LLM call failed; skipping validation: %s", exc)
        return {
            "valid": True,
            "issues": [],
            "required_next_actions": [],
            "semantic_ledger": semantic_ledger or {},
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

    parsed = _parse_process_validator_response(response_text)
    if parsed is None:
        return {
            "valid": True,
            "issues": [],
            "required_next_actions": [],
            "semantic_ledger": semantic_ledger or {},
            "validator_error": "Failed to parse process validator response",
            "raw_response": response_text if response_text else None,
        }

    return {
        "valid": bool(parsed.get("valid", True)),
        "issues": list(parsed.get("issues", [])),
        "required_next_actions": list(parsed.get("required_next_actions", [])),
        "semantic_ledger": (
            parsed.get("semantic_ledger")
            if isinstance(parsed.get("semantic_ledger"), dict)
            else semantic_ledger or {}
        ),
        "raw_response": response_text,
    }
