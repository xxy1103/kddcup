from __future__ import annotations

import pytest

from data_agent_baseline.model_retry import (
    MODEL_REQUEST_RETRY_DELAYS_SECONDS,
    TIMEOUT_RETRY_PROMPT,
    invoke_model_with_retries,
)


class StatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ScriptedModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.invoke_count = 0
        self.invocations = []

    def invoke(self, messages):  # noqa: ANN001
        self.invocations.append(messages)
        self.invoke_count += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_default_model_retry_uses_three_five_second_retries() -> None:
    assert MODEL_REQUEST_RETRY_DELAYS_SECONDS == (5, 5, 5)


def test_invoke_model_retries_retryable_status_with_five_second_delay() -> None:
    model = ScriptedModel(
        [
            StatusError("busy", 503),
            StatusError("rate limited", 429),
            "ok",
        ]
    )
    events: list[dict[str, object]] = []
    sleeps: list[float] = []

    result = invoke_model_with_retries(
        model,
        messages=[],
        sleep_fn=sleeps.append,
        on_retry_event=events.append,
    )

    assert result == "ok"
    assert model.invoke_count == 3
    assert sleeps == [5, 5]
    assert [event["status_code"] for event in events] == [503, 429]
    assert all(event["retryable"] is True for event in events)
    assert all(event["will_retry"] is True for event in events)


def test_invoke_model_does_not_retry_non_429_4xx() -> None:
    model = ScriptedModel([StatusError("bad request", 400), "unexpected"])
    events: list[dict[str, object]] = []

    with pytest.raises(StatusError, match="bad request"):
        invoke_model_with_retries(model, messages=[], on_retry_event=events.append)

    assert model.invoke_count == 1
    assert events[0]["status_code"] == 400
    assert events[0]["retryable"] is False
    assert events[0]["will_retry"] is False


def test_invoke_model_retries_connection_errors() -> None:
    model = ScriptedModel([ConnectionError("server disconnected"), "ok"])
    events: list[dict[str, object]] = []

    result = invoke_model_with_retries(model, messages=[], sleep_fn=lambda _: None, on_retry_event=events.append)

    assert result == "ok"
    assert model.invoke_count == 2
    assert events[0]["error_type"] == "ConnectionError"
    assert events[0]["status_code"] is None
    assert events[0]["retryable"] is True
    assert events[0]["will_retry"] is True


def test_invoke_model_adds_short_action_prompt_after_timeout() -> None:
    model = ScriptedModel([TimeoutError("Request timed out."), "ok"])
    events: list[dict[str, object]] = []

    result = invoke_model_with_retries(
        model,
        messages=[{"role": "user", "content": "question"}],
        sleep_fn=lambda _: None,
        on_retry_event=events.append,
    )

    assert result == "ok"
    assert model.invoke_count == 2
    assert model.invocations[0] == [{"role": "user", "content": "question"}]
    assert model.invocations[1][-1] == {"role": "user", "content": TIMEOUT_RETRY_PROMPT}
    assert events[0]["error_type"] == "TimeoutError"
    assert events[0]["retryable"] is True
    assert events[0]["will_retry"] is True
    assert events[0]["retry_prompt_added"] is True


def test_invoke_model_does_not_retry_timeout_more_than_once() -> None:
    model = ScriptedModel(
        [
            TimeoutError("Request timed out."),
            TimeoutError("Request timed out again."),
            "unexpected",
        ]
    )
    events: list[dict[str, object]] = []

    with pytest.raises(TimeoutError, match="again"):
        invoke_model_with_retries(
            model,
            messages=[],
            sleep_fn=lambda _: None,
            on_retry_event=events.append,
        )

    assert model.invoke_count == 2
    assert [event["will_retry"] for event in events] == [True, False]
    assert [event["retry_prompt_added"] for event in events] == [True, False]
