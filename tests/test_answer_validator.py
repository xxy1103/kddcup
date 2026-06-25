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


def test_validation_request_includes_video_ranking_evidence_only_when_supplied() -> None:
    video_evidence = {
        "schema_version": 2,
        "evidence_items": [
            {
                "tool": "read_doc",
                "capabilities": ["document_text_fact"],
                "source": {"path": "video/briefing_timeline.md"},
            },
            {
                "tool": "read_context_image",
                "capabilities": ["visual_fact"],
                "source": {"path": "video/stable_006.jpg"},
                "observation": {"locator": {"frame_path": "video/stable_006.jpg"}},
            },
            {
                "tool": "record_visual_evidence",
                "capabilities": ["visual_fact_receipt"],
                "source": {"path": "video/stable_006.jpg"},
                "observation": {"evidence_excerpt": "Report scope: Top 3"},
            },
        ],
        "omitted_or_unusable": [],
    }
    request = _build_validation_request(
        question="List entities.",
        answer={"columns": ["entity"], "rows": [["A"]]},
        answer_structure_overview={"columns": ["entity"], "row_count": 1},
        supporting_source_evidence=video_evidence,
    )

    assert "Supporting Source Evidence" in request
    assert "Report scope: Top 3" in request
    assert "Programmatic Submission Risk Report" not in request
    assert "Submitted Answer Structure Overview" in request


def test_validation_request_omits_video_evidence_when_none_is_supplied() -> None:
    request = _build_validation_request(
        question="List entities.",
        answer={"columns": ["entity"], "rows": [["A"]]},
    )

    assert "Supporting Source Evidence" not in request


def test_validation_request_includes_knowledge_docs_for_identifier_rules() -> None:
    request = _build_validation_request(
        question="List records.",
        answer={"columns": ["value"], "rows": [[1]]},
        knowledge_docs=[
            {
                "path": "knowledge.md",
                "kind": "document",
                "content": "table: sample\nprimary_key: RecordNo\nfield: value",
            }
        ],
    )

    assert "Knowledge Documents (authoritative field definitions)" in request
    assert "primary_key: RecordNo" in request
    assert "do not require columns that are absent from the knowledge document" in request


def test_validation_request_explains_context_truncation_is_not_answer_truncation() -> None:
    request = _build_validation_request(
        question="List records.",
        answer={"columns": ["value"], "rows": [[1]]},
        answer_truncated=True,
        answer_row_count=1,
        preview_row_limit=50,
    )

    assert "runtime retains and scores the complete submitted answer separately" in request
    assert "They are not evidence that the submitted answer omitted rows" in request
    assert "Never reject an answer solely because of these context-bound signals" in request
    assert "Tool evidence and the answer structure overview must not be used" in request


def test_answer_prompt_keeps_independent_scope() -> None:
    assert "## Responsibility" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Process Validation Receipt" not in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Narrow video-configured ranking exception" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "matching `record_visual_evidence` receipt" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Video UI procedure names, codes, and counts are never answer data" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Do not tell it to remove or alter the limit" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Require the main agent to read the original" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "read a document/image" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Programmatic Submission Risk Report" not in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "过程校验器负责来源选择" not in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "过程校验器负责来源选择" not in build_system_prompt_v3()


def test_answer_prompt_keeps_delivery_rules() -> None:
    assert "rows` as a list of lists" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Empty rows are valid delivery syntax" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Percentage values must be plain numbers" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "## Answer-scope gates" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "one clearly named output column for each component" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Return the complete matching row set" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "IS NOT NULL" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "SQL `DISTINCT`" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "equivalent Python" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "答案范围关卡" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "完整的匹配行集合" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE


def test_answer_prompt_distinguishes_entity_sets_from_source_record_sets() -> None:
    assert "Answer grain and column scope" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "ENTITY SET or a SOURCE RECORD" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "SET. An entity set asks" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "complete source primary-key or record-" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "Knowledge Documents explicitly define it" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "do not require, recover, or invent one" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "unless the Knowledge Documents explicitly define that column" in (
        ANSWER_VALIDATOR_SYSTEM_PROMPT
    )
    assert "Two rows with equal non-key content" in ANSWER_VALIDATOR_SYSTEM_PROMPT
    assert "答案粒度和列范围" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "Knowledge Documents 为该来源表显式定义" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "不得要求、恢复或虚构标识列" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "两行的非键内容相同、但" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
    assert "主键不同，仍是两条不同记录" in ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE
