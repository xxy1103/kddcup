from __future__ import annotations

from data_agent_baseline.agents.answer_validator import (
    ANSWER_VALIDATOR_SYSTEM_PROMPT,
    _build_validation_request,
)
from data_agent_baseline.agents.langgraph_runtime import _build_answer_validator_context
from data_agent_baseline.agents.prompt import SYSTEM_PROMPT


def test_answer_validator_context_omits_null_counts_and_row_samples() -> None:
    rows = [[None], ["2024-01-01"], ["2024-01-02"], ["2024-01-03"]] + [
        [f"2024-02-{index:02d}"] for index in range(1, 15)
    ]
    answer = {"columns": ["value"], "rows": rows}

    first_preview, first_overview, first_bounded = _build_answer_validator_context(
        answer,
        max_str_tokens=100,
        max_list_items=50,
        distinct_examples_per_column=5,
    )
    second_preview, second_overview, second_bounded = _build_answer_validator_context(
        answer,
        max_str_tokens=100,
        max_list_items=50,
        distinct_examples_per_column=5,
    )

    profile = first_overview["column_profiles"][0]
    assert first_overview["row_count"] == 18
    assert first_overview["column_count"] == 1
    assert first_overview["row_length_counts"] == {"1": 18}
    assert first_overview["duplicate_row_count"] == 0
    assert profile["value_types"] == ["string"]
    assert len(profile["distinct_value_examples"]) == 5
    assert "null" not in str(first_overview)
    assert "row_index" not in str(first_overview)
    assert "head_rows" not in str(first_overview)
    assert first_preview == second_preview
    assert first_overview == second_overview
    assert first_bounded is True
    assert second_bounded is True


def test_answer_validator_context_counts_duplicate_complete_rows() -> None:
    _, overview, _ = _build_answer_validator_context(
        {
            "columns": ["entity", "value"],
            "rows": [["A", 1], ["A", 1], ["A", 2], ["A", 1]],
        },
        max_str_tokens=100,
        max_list_items=50,
    )

    assert overview["row_count"] == 4
    assert overview["duplicate_row_count"] == 2


def test_validation_request_uses_structure_overview_not_row_preview() -> None:
    request = _build_validation_request(
        question="List values.",
        answer={"columns": ["value"], "rows": [[None]]},
        answer_truncated=True,
        answer_row_count=10,
        answer_structure_overview={
            "columns": ["value"],
            "row_count": 10,
            "duplicate_row_count": 2,
            "column_profiles": [
                {
                    "name": "value",
                    "index": 0,
                    "value_types": ["string"],
                    "distinct_value_examples": ["2024-01-01"],
                    "examples_truncated": False,
                }
            ],
        },
        submission_risk_report={
            "detector_version": 1,
            "source_tool": "execute_probe_query",
            "detected": [
                {
                    "kind": "null_filter",
                    "confidence": "high",
                    "location": "source_tool_args.queries[0]",
                    "evidence": "WHERE value IS NOT NULL",
                    "instruction": "Check whether this was explicitly requested.",
                }
            ],
            "parse_errors": [],
            "has_high_confidence_risks": True,
        },
    )

    assert "Submitted Answer Structure Overview" in request
    assert "Programmatic Submission Risk Report" in request
    assert "not an automatic verdict" in request
    assert '"kind": "null_filter"' in request
    assert "Submitted Answer Preview" not in request
    assert "head_rows" not in request
    assert "row_index" not in request
    assert "distinct_value_examples" in request
    assert '"duplicate_row_count": 2' in request
    assert "Use duplicate_row_count only to evaluate entity-set deduplication" in request


def test_validation_request_reports_no_programmatic_risks() -> None:
    request = _build_validation_request(
        question="List values.",
        answer={"columns": ["value"], "rows": [["a"]]},
        submission_risk_report={
            "detector_version": 1,
            "source_tool": "execute_python",
            "detected": [],
            "parse_errors": [],
            "has_high_confidence_risks": False,
        },
    )

    assert "Programmatic Submission Risk Report" in request
    assert "found no obvious source-level NULL filtering" in request


def test_prompts_state_sample_and_raw_retrieval_rules() -> None:
    assert "Do not use submitted-answer structure" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Submitted Answer Structure Overview" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Never use post-submission structure facts to excuse source-level" in (
        ANSWER_VALIDATOR_SYSTEM_PROMPT
    )
    assert "Programmatic Submission Risk Report" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "raw retrieval" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Entity-set queries require DISTINCT" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "complete source records, transactions, event rows" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "except the entity-set DISTINCT requirement" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Minimal inference principle" in SYSTEM_PROMPT
    assert "Feedback about sampled NULL or empty values" in SYSTEM_PROMPT
    assert "Source substitution is forbidden unless equivalence is proven" in SYSTEM_PROMPT
    assert "When a Markdown document is the primary data source" in SYSTEM_PROMPT
    assert "Process validation failures are binding" in SYSTEM_PROMPT
    assert "skipped because `retry_limit_reached`" in SYSTEM_PROMPT


def test_answer_validator_prompt_requires_wide_columns_for_multiple_scalar_answers() -> None:
    assert "Multiple requested answers as separate columns" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "one output column per requested answer component" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "generic key-value or long-table layout" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "This rule does not apply to questions that ask for a list" in (
        ANSWER_VALIDATOR_SYSTEM_PROMPT
    )
