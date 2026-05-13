from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from data_agent_baseline.agents.question_analyzer import (
    QUESTION_ANALYZER_SYSTEM_PROMPT,
    _fallback_result,
    _parse_analyzer_response,
    _validate_field_candidates,
    analyze_question,
)


def test_question_analyzer_prompt_requests_field_candidates() -> None:
    assert '"entities"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"filters"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"requested_output"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"field_candidates"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "field_mappings" not in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "rewritten_question" not in QUESTION_ANALYZER_SYSTEM_PROMPT


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
        "field_candidates": [],
    }


def test_question_analyzer_fallback_uses_field_candidates() -> None:
    result = _fallback_result("What is the average weight?", "request failed")

    assert result == {
        "entities": [],
        "filters": [],
        "requested_output": "",
        "field_candidates": [],
    }


def test_parse_with_field_candidates() -> None:
    parsed = _parse_analyzer_response(
        json.dumps(
            {
                "entities": ["Alex Yoong"],
                "filters": ["track number less than 20"],
                "requested_output": "race name",
                "field_candidates": [
                    {
                        "phrase": "Alex Yoong",
                        "role": "entity",
                        "candidates": [
                            {"field": "d.json.forename", "reason": "Driver first name."},
                            {"field": "d.json.surname", "reason": "Driver last name."},
                        ],
                    }
                ],
            }
        )
    )
    assert parsed["entities"] == ["Alex Yoong"]
    assert parsed["requested_output"] == "race name"
    assert len(parsed["field_candidates"]) == 1
    assert parsed["field_candidates"][0]["phrase"] == "Alex Yoong"


def test_parse_with_three_fields_gets_defaults() -> None:
    parsed = _parse_analyzer_response(
        json.dumps(
            {
                "entities": ["value"],
                "filters": ["x > 0"],
                "requested_output": "max value",
            }
        )
    )
    assert parsed["entities"] == ["value"]
    assert parsed["filters"] == ["x > 0"]
    assert parsed["requested_output"] == "max value"
    assert parsed["field_candidates"] == []


def test_validate_field_candidates_strips_invalid_fields() -> None:
    field_candidates = [
        {
            "phrase": "name",
            "role": "entity",
            "candidates": [
                {"field": "a.csv.col_0", "reason": "valid."},
                {"field": "x.csv.no_such_field", "reason": "invalid."},
            ],
        }
    ]
    schemas = [
        {
            "asset_path": "a.csv",
            "kind": "csv",
            "fields": [{"name": "col_0", "type": "integer"}],
        }
    ]
    result = _validate_field_candidates(field_candidates, schemas)
    assert len(result) == 1
    assert result[0]["phrase"] == "name"
    assert len(result[0]["candidates"]) == 1
    assert result[0]["candidates"][0]["field"] == "a.csv.col_0"


def test_validate_field_candidates_removes_empty_phrase() -> None:
    field_candidates = [
        {"phrase": "", "role": "entity", "candidates": [{"field": "a.csv.col_0"}]},
        {"phrase": "  ", "role": "filter", "candidates": []},
        {"role": "entity", "candidates": [{"field": "a.csv.col_0"}]},
    ]
    schemas = [
        {
            "asset_path": "a.csv",
            "kind": "csv",
            "fields": [{"name": "col_0", "type": "integer"}],
        }
    ]
    result = _validate_field_candidates(field_candidates, schemas)
    assert result == []


def test_validate_field_candidates_all_invalid_returns_empty() -> None:
    field_candidates = [
        {
            "phrase": "x",
            "role": "entity",
            "candidates": [{"field": "no.good", "reason": "nope."}],
        }
    ]
    schemas = [
        {
            "asset_path": "real.csv",
            "kind": "csv",
            "fields": [{"name": "field", "type": "text"}],
        }
    ]
    result = _validate_field_candidates(field_candidates, schemas)
    assert result == []


def test_validate_field_candidates_sqlite_paths() -> None:
    field_candidates = [
        {
            "phrase": "year",
            "role": "filter",
            "candidates": [
                {"field": "db/db.sqlite.races.year", "reason": "Table-qualified."},
            ],
        }
    ]
    schemas = [
        {
            "asset_path": "db/db.sqlite",
            "kind": "sqlite",
            "tables": [
                {
                    "name": "races",
                    "fields": [{"name": "year", "type": "INTEGER"}],
                }
            ],
        }
    ]
    result = _validate_field_candidates(field_candidates, schemas)
    assert len(result) == 1
    assert result[0]["candidates"][0]["field"] == "db/db.sqlite.races.year"


def test_analyze_question_without_schemas_returns_field_candidates(monkeypatch) -> None:
    def fake_invoke(model, messages):
        return AIMessage(
            content=json.dumps(
                {
                    "entities": ["x"],
                    "filters": [],
                    "requested_output": "y",
                }
            )
        )

    monkeypatch.setattr(
        "data_agent_baseline.agents.question_analyzer.invoke_model_with_retries",
        fake_invoke,
    )
    result = analyze_question(model=MagicMock(), question="test?")
    assert result["entities"] == ["x"]
    assert result["filters"] == []
    assert result["requested_output"] == "y"
    assert result["field_candidates"] == []


def test_analyze_question_with_schemas_calls_with_candidates(monkeypatch) -> None:
    def fake_invoke(model, messages):
        return AIMessage(
            content=json.dumps(
                {
                    "entities": ["Alex Yoong"],
                    "filters": ["driver is Alex Yoong"],
                    "requested_output": "race name",
                    "field_candidates": [
                        {
                            "phrase": "Alex Yoong",
                            "role": "entity",
                            "candidates": [
                                {"field": "data/drivers.json.forename", "reason": "First name."},
                                {"field": "data/drivers.json.surname", "reason": "Last name."},
                            ],
                        }
                    ],
                }
            )
        )

    monkeypatch.setattr(
        "data_agent_baseline.agents.question_analyzer.invoke_model_with_retries",
        fake_invoke,
    )
    schemas = [
        {
            "asset_path": "data/drivers.json",
            "kind": "json",
            "fields": [
                {"name": "forename", "type": "text"},
                {"name": "surname", "type": "text"},
            ],
        },
        {
            "asset_path": "data/races.csv",
            "kind": "csv",
            "fields": [{"name": "name", "type": "text"}],
        },
    ]
    result = analyze_question(
        model=MagicMock(),
        question="Which race was Alex Yoong in?",
        schemas=schemas,
    )
    assert result["entities"] == ["Alex Yoong"]
    assert len(result["field_candidates"]) == 1
    assert result["field_candidates"][0]["phrase"] == "Alex Yoong"
    assert len(result["field_candidates"][0]["candidates"]) == 2


def test_analyze_question_with_schemas_strips_invalid_candidates(monkeypatch) -> None:
    def fake_invoke(model, messages):
        return AIMessage(
            content=json.dumps(
                {
                    "entities": ["X"],
                    "filters": [],
                    "requested_output": "Y",
                    "field_candidates": [
                        {
                            "phrase": "valid",
                            "role": "entity",
                            "candidates": [
                                {"field": "a.csv.col_0", "reason": "exists."},
                                {"field": "z.xyz.no_such", "reason": "nope."},
                            ],
                        },
                        {
                            "phrase": "bogus",
                            "role": "filter",
                            "candidates": [
                                {"field": "z.xyz.no_such", "reason": "nope."},
                            ],
                        },
                    ],
                }
            )
        )

    monkeypatch.setattr(
        "data_agent_baseline.agents.question_analyzer.invoke_model_with_retries",
        fake_invoke,
    )
    schemas = [
        {
            "asset_path": "a.csv",
            "kind": "csv",
            "fields": [{"name": "col_0", "type": "integer"}],
        }
    ]
    result = analyze_question(
        model=MagicMock(),
        question="test?",
        schemas=schemas,
    )
    assert len(result["field_candidates"]) == 1
    assert result["field_candidates"][0]["phrase"] == "valid"
    assert len(result["field_candidates"][0]["candidates"]) == 1
    assert result["field_candidates"][0]["candidates"][0]["field"] == "a.csv.col_0"
