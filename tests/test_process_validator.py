from __future__ import annotations

from langchain_core.messages import AIMessage

from data_agent_baseline.agents.process_validator import (
    PROCESS_VALIDATOR_SYSTEM_PROMPT,
    _build_process_validation_request,
    _parse_process_validator_response,
    validate_process,
)


def test_process_validator_request_includes_context() -> None:
    request = _build_process_validation_request(
        question="What is the average?",
        answer={"columns": ["avg"], "rows": [[1.0]]},
        ambiguity_analysis={"ambiguities": [{"id": "amb_001"}]},
        recent_steps=[{"step_index": 1, "node": "model", "assistant_message": "I will check."}],
        semantic_ledger={"intent_summary": "average"},
    )

    assert "What is the average?" in request
    assert '"columns"' in request
    assert "amb_001" in request
    assert "Recent Trace Steps" in request
    assert "intent_summary" in request


def test_process_validator_prompt_guards_scoreable_source_binding() -> None:
    assert "fixed-program scorer" in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "Scoreable answer contract" in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "Markdown files can be the real table" in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "merely similar table" in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "same metric definition, same unit, same aggregation level" in (
        PROCESS_VALIDATOR_SYSTEM_PROMPT
    )
    assert "duplicated rows, time intervals" not in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "required keys, metrics, and coverage from" in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "No unrequested transformations" not in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "unrequested aggregation" not in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "GROUP BY" not in PROCESS_VALIDATOR_SYSTEM_PROMPT
    assert "DISTINCT" not in PROCESS_VALIDATOR_SYSTEM_PROMPT


def test_parse_process_validator_response_json() -> None:
    parsed = _parse_process_validator_response(
        '{"valid": false, "issues": ["unsupported"], '
        '"required_next_actions": ["probe data"], "semantic_ledger": {"x": 1}}'
    )

    assert parsed is not None
    assert parsed["valid"] is False
    assert parsed["issues"] == ["unsupported"]


def test_parse_process_validator_response_non_json() -> None:
    assert _parse_process_validator_response("not json") is None


def test_validate_process_defaults_to_valid_when_model_fails(monkeypatch) -> None:  # noqa: ANN001
    def fail_invoke(*args, **kwargs):  # noqa: ANN001
        del args, kwargs
        raise RuntimeError("offline")

    monkeypatch.setattr("data_agent_baseline.agents.process_validator.invoke_model_with_retries", fail_invoke)

    result = validate_process(
        model=object(),  # type: ignore[arg-type]
        question="Question?",
        semantic_ledger={"intent_summary": "old"},
    )

    assert result["valid"] is True
    assert result["validator_error"] == "offline"
    assert result["semantic_ledger"] == {"intent_summary": "old"}


def test_validate_process_parses_model_response(monkeypatch) -> None:  # noqa: ANN001
    def invoke(*args, **kwargs):  # noqa: ANN001
        del args, kwargs
        return AIMessage(
            content=(
                '{"valid": true, "issues": [], "required_next_actions": [], '
                '"semantic_ledger": {"intent_summary": "ok"}}'
            )
        )

    monkeypatch.setattr("data_agent_baseline.agents.process_validator.invoke_model_with_retries", invoke)

    result = validate_process(model=object(), question="Question?")  # type: ignore[arg-type]

    assert result["valid"] is True
    assert result["issues"] == []
    assert result["semantic_ledger"] == {"intent_summary": "ok"}
