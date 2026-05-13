from __future__ import annotations

import pytest

from data_agent_baseline.model_retry import (
    MODEL_REQUEST_RETRY_DELAYS_SECONDS,
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

    def invoke(self, messages):  # noqa: ANN001
        del messages
        self.invoke_count += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_default_model_retry_uses_three_zero_delay_retries() -> None:
    assert MODEL_REQUEST_RETRY_DELAYS_SECONDS == (0, 0, 0)


def test_invoke_model_retries_retryable_status_without_delay() -> None:
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
    assert sleeps == [0, 0]
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
