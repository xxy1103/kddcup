from __future__ import annotations

import json

from data_agent_baseline.agents.question_analyzer import (
    QUESTION_ANALYZER_SYSTEM_PROMPT,
    _fallback_result,
    _parse_analyzer_response,
)


def test_question_analyzer_prompt_requests_three_field_decomposition() -> None:
    assert '"entities"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"filters"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"requested_output"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "clarified_question" not in QUESTION_ANALYZER_SYSTEM_PROMPT


def test_parse_analyzer_response_drops_legacy_clarified_question() -> None:
    parsed = _parse_analyzer_response(
        json.dumps(
            {
                "entities": ["female superheroes"],
                "filters": ["gender is female"],
                "requested_output": "average weight",
                "clarified_question": "legacy field should be ignored",
            }
        )
    )

    assert parsed == {
        "entities": ["female superheroes"],
        "filters": ["gender is female"],
        "requested_output": "average weight",
    }


def test_question_analyzer_fallback_uses_three_fields() -> None:
    result = _fallback_result("What is the average weight?", "request failed")

    assert result == {
        "entities": [],
        "filters": [],
        "requested_output": "",
    }
