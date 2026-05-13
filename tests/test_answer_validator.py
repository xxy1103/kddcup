from __future__ import annotations

from data_agent_baseline.agents.answer_validator import _build_validation_request


def test_answer_validator_request_includes_validation_history() -> None:
    request = _build_validation_request(
        "List the value column.",
        {"columns": ["value"], "rows": [["1"]]},
        [
            {
                "answer_fingerprint": "abc123",
                "answer_columns": ["extra"],
                "answer_row_count": 1,
                "valid": False,
                "issues": ["extra column"],
                "validator_error": None,
            }
        ],
    )

    assert "## Previous Validation History" in request
    assert "abc123" in request
    assert "extra column" in request
    assert "keep the same validation criteria" in request


def test_answer_validator_request_without_history_matches_original_shape() -> None:
    request = _build_validation_request(
        "List the value column.",
        {"columns": ["value"], "rows": [["1"]]},
    )

    assert "## Original Question" in request
    assert "## Submitted Answer" in request
    assert "## Previous Validation History" not in request
