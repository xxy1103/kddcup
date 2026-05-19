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

If the semantic ledger contains any unverified assumption that is material to
the answer, you MUST set valid=false. Do not put a material unverified
assumption in "unverified_assumptions" while also returning valid=true.

Example: If the question asks for purchases at a "unit price > 29.00" and the
agent uses a field named "Price", the process is invalid unless the trace shows
evidence that Price is unit price rather than total transaction amount. A
statement such as "Price is unit price by standard convention" is insufficient
and must be rejected.

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
