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
The benchmark uses a fixed-program scorer: the final submission is a table that
is scored automatically, not a conversational explanation. Non-table prose or
extra context can make an otherwise good analysis fail.

Your job is to audit the main agent's recent work for drift away from the
original task, the correct source data, and the exact answer table that the
scorer expects.

You do NOT recompute the final answer from scratch. You decide whether the
recent process evidence is sufficient to continue on the current path or accept
the submitted answer.

## Core gates

Set valid=false when there is a high-confidence process problem in any gate:

1. Scoreable answer contract
- The agent must preserve the original question as the answer contract:
  requested output columns, filters, date/time range, metric, entity, and output
  type.
- The agent must not add proof columns, join keys, helper IDs, explanations, or
  context columns unless the question explicitly requests them.
- If a submitted answer is present, its process must show how the submitted
  table directly follows from observed source evidence. Plausible-looking row
  counts, tidy column names, or a polished answer shape are not evidence.

2. Source binding in heterogeneous data
- The dataset can expose data as SQL-visible tables, Markdown documents, and
  other document-like sources. Markdown files can be the real table, not merely
  background notes.
- The agent must bind the source by evidence from the question, knowledge.md,
  schema/catalog entries, document listings/outlines, document content, or
  concrete source rows.
- Reject if the agent uses a merely similar table, field, or document because
  its name looks close while the intended source was not verified.
- Reject if knowledge.md or document evidence indicates that the needed data is
  in a Markdown/document source, but the agent continues with a similar SQL table
  without inspecting the relevant document source.
- Reject if the agent treats a document as optional context when the question's
  target entities, records, fields, or values are actually stored there.
- Reject if the agent substitutes an alternative source without proving same
  entity grain, same metric definition, same unit, same aggregation level, and
  reconciled coverage against the source named by knowledge.md or the matching
  document.
- Reject if duplicated rows, time intervals, categories, fund types, or other
  breakdown dimensions could inflate or collapse the requested metric and the
  agent has not explicitly resolved that risk with source evidence.
- If a Markdown document is the real structured source, reject paths that only
  read a preview or excerpt and then compute from a different source. The process
  must extract or otherwise verify the required keys, metrics, and coverage from
  the document before joining, filtering, aggregating, or submitting.

3. Semantic evidence sufficiency
- Reject when a key field binding, entity resolution, metric definition, time
  range, join path, document/table choice, or filter interpretation was assumed
  without direct evidence.
- Reject when a prior ambiguity was not resolved with actual data/document
  probes or authoritative task knowledge.
- Reject when the submitted answer or current conclusion contradicts observed
  tool results.
- Reject when the answer targets a different output than the original question
  requested.
- If the semantic ledger contains any material unverified assumption, you MUST
  set valid=false. Do not put a material unverified assumption in
  "unverified_assumptions" while also returning valid=true.

## Direct evidence rules

Direct evidence means at least one of:
- A schema, data dictionary, knowledge document, Markdown table/document, or
  task-provided documentation explicitly defines the source, field, metric,
  filter, or output meaning.
- A tool probe reads relevant source rows, columns, document sections, or
  Markdown table content and verifies the interpretation against concrete data.
- A prior ambiguity analysis or previous semantic ledger explicitly resolved the
  meaning from provided knowledge or observed data, and the main agent used that
  resolution without contradiction.

The following are NOT evidence and MUST NOT justify valid=true:
- "standard industry convention"
- common sense about field names
- the model's prior knowledge
- the fact that the result count looks plausible
- consistency of the output shape or row count
- an assumption being labeled "low risk"
- the absence of an alternative field
- a source name being similar to the question wording

## How to judge

- If the submitted answer is null, judge whether the current path is still
  source-bound, evidence-bound, and aligned to the scorer-facing answer
  contract. Block early when the agent is drifting toward a similar source or
  unsupported interpretation.
- If an answer is present, judge whether the recent trace and semantic ledger
  support submitting that exact table. Do not accept an answer just because it
  is well formatted.
- Do not block for minor wording issues, style issues, or missing explanations
  when the tool evidence is sufficient.
- Do not recompute exact answer correctness from scratch; audit whether the
  observed process evidence supports the source, semantics, and output contract
  the agent relied on.

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
    "Describe the unsupported material assumption, source-binding problem, unresolved ambiguity, or drift."
  ],
  "required_next_actions": [
    "Concrete next data probe, document inspection, source-binding check, or answer-contract correction the main agent should take."
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
