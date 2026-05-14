from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from data_agent_baseline.agents.question_analyzer import (
    ALLOWED_DESCRIBES,
    ALLOWED_OWNER_MATCH,
    ALLOWED_ROW_GRAIN,
    QUESTION_ANALYZER_SYSTEM_PROMPT,
    _fallback_result,
    _parse_analyzer_response,
    _validate_field_candidates,
    _validate_filters_candidates,
    analyze_question,
)


def test_question_analyzer_prompt_requests_field_candidates() -> None:
    assert '"entities"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"filters"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"requested_output"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"field_candidates"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"filters_candidates"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "field_mappings" not in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "rewritten_question" not in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "up to 3" not in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "track number" not in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "Prefer recall over precision" in QUESTION_ANALYZER_SYSTEM_PROMPT
    # New semantic structure fields
    assert '"row_grain"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"describes"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"owner_match"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"risk"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"modifies"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert '"expected_row_grain"' in QUESTION_ANALYZER_SYSTEM_PROMPT
    assert "Semantic structure rules" in QUESTION_ANALYZER_SYSTEM_PROMPT


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
        "filters_candidates": [],
    }


def test_question_analyzer_fallback_uses_field_candidates() -> None:
    result = _fallback_result("What is the average weight?", "request failed")

    assert result == {
        "entities": [],
        "filters": [],
        "requested_output": "",
        "field_candidates": [],
        "filters_candidates": [],
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
                            {
                                "field": "d.json.forename",
                                "reason": "Driver first name.",
                                "row_grain": "entity",
                                "describes": "subject_entity",
                                "owner_match": "strong",
                                "risk": "May be ambiguous if multiple drivers share forename.",
                            },
                            {
                                "field": "d.json.surname",
                                "reason": "Driver last name.",
                                "row_grain": "entity",
                                "describes": "subject_entity",
                                "owner_match": "strong",
                                "risk": "May be ambiguous if multiple drivers share surname.",
                            },
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
    assert parsed["filters_candidates"] == []


def test_validate_field_candidates_strips_invalid_fields() -> None:
    field_candidates = [
        {
            "phrase": "name",
            "role": "entity",
            "candidates": [
                {
                    "field": "a.csv.col_0",
                    "reason": "valid.",
                    "row_grain": "entity",
                    "describes": "subject_entity",
                    "owner_match": "strong",
                    "risk": "may be wrong",
                },
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
    assert result[0]["candidates"][0]["row_grain"] == "entity"
    assert result[0]["candidates"][0]["describes"] == "subject_entity"
    assert result[0]["candidates"][0]["owner_match"] == "strong"
    assert result[0]["candidates"][0]["risk"] == "may be wrong"


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


def test_validate_field_candidates_sanitizes_invalid_semantic_values() -> None:
    field_candidates = [
        {
            "phrase": "test",
            "role": "entity",
            "candidates": [
                {
                    "field": "a.csv.col_0",
                    "reason": "valid.",
                    "row_grain": "bogus_grain",
                    "describes": "bogus_describes",
                    "owner_match": "bogus_match",
                    "risk": "some risk",
                },
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
    cand = result[0]["candidates"][0]
    assert cand["field"] == "a.csv.col_0"
    # Invalid values are stripped; only valid ones remain
    assert "row_grain" not in cand
    assert "describes" not in cand
    assert "owner_match" not in cand
    assert cand["risk"] == "some risk"


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
    assert result["filters_candidates"] == []


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
                                {
                                    "field": "data/drivers.json.forename",
                                    "reason": "First name.",
                                    "row_grain": "entity",
                                    "describes": "subject_entity",
                                    "owner_match": "strong",
                                    "risk": "may be ambiguous",
                                },
                                {
                                    "field": "data/drivers.json.surname",
                                    "reason": "Last name.",
                                    "row_grain": "entity",
                                    "describes": "subject_entity",
                                    "owner_match": "strong",
                                    "risk": "may be ambiguous",
                                },
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
    assert result["filters_candidates"] == []


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
    assert result["filters_candidates"] == []


def test_validate_filters_candidates_strips_invalid_paths() -> None:
    filters_candidates = [
        {
            "filter": "price > 29",
            "modifies": "the transaction record",
            "expected_row_grain": "transaction",
            "candidates": [
                {
                    "expression": "a.csv.col_0 > 29",
                    "fields": ["a.csv.col_0"],
                    "row_grain": "transaction",
                    "describes": "metric_or_measure",
                    "owner_match": "strong",
                    "risk": "may be per-unit rather than total",
                },
                {
                    "expression": "x.csv.no_such > 29",
                    "fields": ["x.csv.no_such"],
                },
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
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert len(result) == 1
    assert result[0]["filter"] == "price > 29"
    assert result[0]["modifies"] == "the transaction record"
    assert result[0]["expected_row_grain"] == "transaction"
    assert len(result[0]["candidates"]) == 1
    assert result[0]["candidates"][0]["expression"] == "a.csv.col_0 > 29"
    assert result[0]["candidates"][0]["fields"] == ["a.csv.col_0"]
    assert result[0]["candidates"][0]["row_grain"] == "transaction"
    assert result[0]["candidates"][0]["describes"] == "metric_or_measure"
    assert result[0]["candidates"][0]["owner_match"] == "strong"
    assert result[0]["candidates"][0]["risk"] == "may be per-unit rather than total"


def test_validate_filters_candidates_all_invalid_returns_empty() -> None:
    filters_candidates = [
        {
            "filter": "x < 0",
            "candidates": [
                {"expression": "no.good < 0", "fields": ["no.good"]},
            ],
        }
    ]
    schemas = [
        {
            "asset_path": "real.csv",
            "kind": "csv",
            "fields": [{"name": "field", "type": "text"}],
        }
    ]
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert result == []


def test_validate_filters_candidates_handles_empty_filter_desc() -> None:
    candidates = [{"expression": "a.csv.col_0 > 5", "fields": ["a.csv.col_0"]}]
    filters_candidates = [
        {"filter": "", "candidates": candidates},
        {"filter": "  ", "candidates": candidates},
        {"candidates": candidates},
    ]
    schemas = [
        {
            "asset_path": "a.csv",
            "kind": "csv",
            "fields": [{"name": "col_0", "type": "integer"}],
        }
    ]
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert result == []


def test_validate_filters_candidates_sqlite_paths() -> None:
    filters_candidates = [
        {
            "filter": "year = 2020",
            "candidates": [
                {
                    "expression": "db/db.sqlite.races.year = 2020",
                    "fields": ["db/db.sqlite.races.year"],
                },
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
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert len(result) == 1
    assert result[0]["candidates"][0]["expression"] == "db/db.sqlite.races.year = 2020"
    assert result[0]["candidates"][0]["fields"] == ["db/db.sqlite.races.year"]


def test_validate_filters_candidates_multiple_valid_candidates() -> None:
    filters_candidates = [
        {
            "filter": "col > 10",
            "candidates": [
                {"expression": "a.csv.col_0 > 10", "fields": ["a.csv.col_0"]},
                {"expression": "a.csv.col_1 >= 10", "fields": ["a.csv.col_1"]},
            ],
        }
    ]
    schemas = [
        {
            "asset_path": "a.csv",
            "kind": "csv",
            "fields": [
                {"name": "col_0", "type": "integer"},
                {"name": "col_1", "type": "integer"},
            ],
        }
    ]
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert len(result) == 1
    assert len(result[0]["candidates"]) == 2
    expressions = {c["expression"] for c in result[0]["candidates"]}
    assert "a.csv.col_0 > 10" in expressions
    assert "a.csv.col_1 >= 10" in expressions


def test_validate_filters_candidates_multi_field_expression() -> None:
    filters_candidates = [
        {
            "filter": "price per unit > 29",
            "candidates": [
                {
                    "expression": "a.csv.Price / a.csv.Amount > 29",
                    "fields": ["a.csv.Price", "a.csv.Amount"],
                },
                {
                    "expression": "a.csv.UnitPrice > 29",
                    "fields": ["a.csv.UnitPrice"],
                },
            ],
        }
    ]
    schemas = [
        {
            "asset_path": "a.csv",
            "kind": "csv",
            "fields": [
                {"name": "Price", "type": "REAL"},
                {"name": "Amount", "type": "INTEGER"},
                {"name": "UnitPrice", "type": "REAL"},
            ],
        }
    ]
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert len(result) == 1
    assert len(result[0]["candidates"]) == 2


def test_validate_filters_candidates_strips_invalid_field_in_multi() -> None:
    """A candidate with one valid and one invalid field is stripped entirely."""
    filters_candidates = [
        {
            "filter": "per unit > 5",
            "candidates": [
                {
                    "expression": "a.csv.Price / x.nope > 5",
                    "fields": ["a.csv.Price", "x.nope"],
                },
            ],
        }
    ]
    schemas = [
        {
            "asset_path": "a.csv",
            "kind": "csv",
            "fields": [{"name": "Price", "type": "REAL"}],
        }
    ]
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert result == []


def test_validate_filters_candidates_sanitizes_invalid_semantic_values() -> None:
    filters_candidates = [
        {
            "filter": "col > 10",
            "modifies": "the output event",
            "expected_row_grain": "bogus_grain",
            "candidates": [
                {
                    "expression": "a.csv.col_0 > 10",
                    "fields": ["a.csv.col_0"],
                    "row_grain": "bogus",
                    "describes": "nope",
                    "owner_match": "bad",
                    "risk": "some risk",
                },
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
    result = _validate_filters_candidates(filters_candidates, schemas)
    assert len(result) == 1
    assert result[0]["modifies"] == "the output event"
    # invalid expected_row_grain stripped
    assert "expected_row_grain" not in result[0]
    cand = result[0]["candidates"][0]
    # invalid semantic values are stripped
    assert "row_grain" not in cand
    assert "describes" not in cand
    assert "owner_match" not in cand
    assert cand["risk"] == "some risk"
