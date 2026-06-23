from __future__ import annotations

from data_agent_baseline.agents.answer_validator import (
    ANSWER_VALIDATOR_SYSTEM_PROMPT,
    ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE,
    _build_validation_request,
)
from data_agent_baseline.agents.langgraph_runtime import _build_answer_validator_context
from data_agent_baseline.agents.prompt3 import build_system_prompt_v3


def test_answer_validator_context_omits_row_samples() -> None:
    rows = [[None], ["2024-01-01"], ["2024-01-02"], ["2024-01-03"]] + [
        [f"2024-02-{index:02d}"] for index in range(1, 15)
    ]
    _, overview, bounded = _build_answer_validator_context(
        {"columns": ["value"], "rows": rows},
        max_str_tokens=100,
        max_list_items=50,
        distinct_examples_per_column=5,
    )

    assert overview["row_count"] == 18
    assert overview["row_length_counts"] == {"1": 18}
    assert "row_index" not in str(overview)
    assert "head_rows" not in str(overview)
    assert bounded is True


def test_answer_validator_context_counts_duplicate_complete_rows() -> None:
    _, overview, _ = _build_answer_validator_context(
        {"columns": ["entity", "value"], "rows": [["A", 1], ["A", 1], ["A", 2]]},
        max_str_tokens=100,
        max_list_items=50,
    )

    assert overview["duplicate_row_count"] == 1


def test_validation_request_is_delivery_only_and_receipt_bound() -> None:
    receipt = {
        "status": "validated",
        "receipt_version": 2,
    }
    request = _build_validation_request(
        question="List entities.",
        answer={"columns": ["entity"], "rows": [["A"]]},
        submission_context={"submission_tool": "submit_tool_result", "source_tool": "execute_probe_query"},
        answer_structure_overview={"columns": ["entity"], "row_count": 1},
        process_validation_receipt=receipt,
        receipt_matches_submission=True,
    )

    assert "Process Validation Receipt" in request
    assert '"matches_current_submission": true' in request
    assert "Programmatic Submission Risk Report" not in request
    assert "Submitted Answer Structure Overview" in request


def test_answer_prompt_keeps_process_receipt_precedence() -> None:
    assert "## Responsibility and precedence" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "A matching `Process Validation Receipt`" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "does NOT bind answer-scope decisions" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "read a document/image" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Programmatic Submission Risk Report" not in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "one row, one column" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "过程校验器负责来源选择" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "过程校验器负责来源选择" not in build_system_prompt_v3()


def test_answer_prompt_keeps_delivery_rules() -> None:
    assert "rows` as a list of lists" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Empty rows are valid delivery syntax" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Percentage values must be plain numbers" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "## Answer-scope gates" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "one clearly named output column for each component" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Return the complete matching row set" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "IS NOT NULL" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "SQL `DISTINCT` or equivalent Python deduplication" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "答案范围关卡" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "完整的匹配行集合" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
