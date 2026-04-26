from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.inspectors.exchange import (
    AgentEnvelope,
    AgentEnvelopeContent,
    SemanticClaim,
    Uncertainty,
)


_RISK_RULES: tuple[tuple[str, str, str], ...] = (
    (
        r"\bper\s+unit\b",
        "per_unit_ratio",
        "Do not map `per unit` directly to a total amount; verify whether a ratio such as price / amount is required.",
    ),
    (
        r"\branked?\s+second\b|\brank\s*=?\s*2\b",
        "rank_position_ambiguity",
        "Verify whether the wording maps to a `rank` field rather than a `position` field.",
    ),
    (
        r"\bnumber\b",
        "number_source_ambiguity",
        "Confirm which entity owns the requested number; avoid submitting helper IDs unless requested.",
    ),
    (
        r"\btype\b|\bcategory\b",
        "type_category_level_ambiguity",
        "Confirm whether the requested label belongs to an event, item, budget, or other entity level.",
    ),
    (
        r"\bdistinct\b|\bunique\b|\bcount\b|\btally\b",
        "aggregation_grain",
        "Verify the required aggregation grain before submitting the final table.",
    ),
)

_METRIC_WORDS = {
    "average",
    "count",
    "finish time",
    "highest",
    "lowest",
    "maximum",
    "minimum",
    "most",
    "fewest",
    "sum",
    "total",
}


def _answer_shape(question: str) -> dict[str, Any]:
    lowered = question.lower()
    likely_single_row = lowered.startswith(("what ", "what's ", "which ", "how many ", "when "))
    likely_list = any(token in lowered for token in ("list ", "show ", "give me all", "which patients"))
    if likely_list:
        row_shape = "multiple_rows"
    elif likely_single_row:
        row_shape = "single_row"
    else:
        row_shape = "unknown"

    column_hint = "unknown"
    list_match = re.search(r"\blist (?:their |the )?(.+?)(?:\.|\?|$)", lowered)
    if list_match:
        column_hint = list_match.group(1)
    elif "finish time" in lowered:
        column_hint = "finish time"
    elif "how many" in lowered:
        column_hint = "count"

    return {
        "row_shape": row_shape,
        "column_hint": column_hint,
        "only_requested_columns": True,
    }


def _extract_entities(question: str) -> list[str]:
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z0-9]*)(?:\s+[A-Z][A-Za-z0-9]*)*\b", question)
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    entities = [item.strip() for item in candidates + years if item.strip()]
    return list(dict.fromkeys(entities))


def _extract_metrics(question: str) -> list[str]:
    lowered = question.lower()
    return [metric for metric in sorted(_METRIC_WORDS) if metric in lowered]


def _extract_filter_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    for pattern in (r"\bwith ([^,.?]+)", r"\bin ([^,.?]+)", r"\bwhere ([^,.?]+)"):
        phrases.extend(match.strip() for match in re.findall(pattern, question, flags=re.IGNORECASE))
    return list(dict.fromkeys(phrases))


def _detect_risks(question: str) -> list[Uncertainty]:
    risks: list[Uncertainty] = []
    for pattern, risk, instruction in _RISK_RULES:
        if re.search(pattern, question, flags=re.IGNORECASE):
            risks.append(Uncertainty(risk=risk, instruction=instruction))
    return risks


def build_perception_payload(task: PublicTask) -> dict[str, Any]:
    question = task.question
    risks = _detect_risks(question)
    return {
        "question": question,
        "difficulty": task.difficulty,
        "entities": _extract_entities(question),
        "metrics": _extract_metrics(question),
        "filter_phrases": _extract_filter_phrases(question),
        "expected_answer_shape": _answer_shape(question),
        "high_risk_terms": [risk.risk for risk in risks],
    }


def build_perception_envelope(task: PublicTask) -> AgentEnvelope:
    payload = build_perception_payload(task)
    risks = _detect_risks(task.question)
    claims = [
        SemanticClaim(
            claim=f"Expected answer shape: {payload['expected_answer_shape']}",
            confidence="medium",
        )
    ]
    summary = (
        f"Question entities: {payload['entities']}. "
        f"Metrics: {payload['metrics']}. "
        f"Expected answer shape: {payload['expected_answer_shape']}. "
        f"High-risk terms: {payload['high_risk_terms']}."
    )
    return AgentEnvelope(
        task_id=task.task_id,
        sender="PerceptionAgent",
        recipient="DataUnderstandingAgent",
        message_type="perception_result",
        content=AgentEnvelopeContent(
            summary=summary,
            semantic_claims=claims,
            uncertainties=risks,
            payload=payload,
        ),
    )
