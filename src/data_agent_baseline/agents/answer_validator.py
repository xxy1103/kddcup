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
Your job is to review a submitted answer table and check for formatting issues.
You do NOT fix the answer. You only report whether it passes validation or not.

## Validation Rules

### 1. Unnecessary columns
- Compare the submitted columns with what the original question asks for.
- If there are columns that the question did NOT request, report them as issues.
- If you are unsure whether a column is needed, treat it as acceptable.

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

### 5. Hallucination / data fidelity check
- You will be given the tool execution history that produced the answer.
- Every row and value in the submitted answer MUST be traceable to the data returned by those tools.
- If a row contains values that do NOT appear in any tool output, flag it as likely hallucinated.
- If the submitted row count is far fewer than what a tool result reported (e.g. tool found 140 rows but answer has 120), flag it as missing data.
- If the answer contains rows with strictly sequential IDs with no gaps while the tool output shows natural gaps, flag them as likely fabricated.
- If the answer reuses the same non-ID column values (date/amount/balance) across many consecutive rows while only the ID changes, flag this as likely pattern-filled fabrication.
- When the tool history is empty or insufficient, skip this check rather than guessing.

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


def _build_validation_request(
    question: str, answer: dict[str, Any], context_steps: list[dict[str, Any]] | None = None
) -> str:
    parts = [
        f"## Original Question\n{question}\n",
        f"## Submitted Answer\n```json\n{json.dumps(answer, ensure_ascii=False, indent=2)}\n```\n",
    ]
    if context_steps:
        parts.append(
            "## Tool Execution History\n"
            "The following tool calls and results were used to produce the answer. "
            "Use them to verify that every value in the submitted answer is backed by real data.\n\n"
            f"```json\n{json.dumps(context_steps, ensure_ascii=False, indent=2)}\n```\n"
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
    context_steps: list[dict[str, Any]] | None = None,
    retry_event_callback: Any | None = None,
) -> dict[str, Any]:
    """Validate a submitted answer with one LLM call.

    If validation itself fails, the function returns ``valid=True`` so the main
    task flow is not blocked by the checker.
    """
    messages = [
        SystemMessage(content=ANSWER_VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(content=_build_validation_request(question, answer, context_steps)),
    ]

    try:
        ai_message = invoke_model_with_retries(
            model,
            messages,
            on_retry_event=retry_event_callback,
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
