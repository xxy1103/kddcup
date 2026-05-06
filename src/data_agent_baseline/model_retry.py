from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any


MODEL_REQUEST_RETRY_DELAYS_SECONDS = (15, 30, 45, 60)
ModelRetryEventCallback = Callable[[dict[str, Any]], None]


def _raw_exception_content(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text:
        return response_text

    body = getattr(exc, "body", None)
    if body not in (None, ""):
        return str(body)

    return repr(exc)


def _model_retry_event(
    *,
    exc: Exception,
    attempt_index: int,
    max_attempts: int,
    retry_delay_seconds: int | None,
) -> dict[str, Any]:
    return {
        "attempt": attempt_index + 1,
        "max_attempts": max_attempts,
        "error": _raw_exception_content(exc),
        "will_retry": retry_delay_seconds is not None,
        "next_retry_delay_seconds": retry_delay_seconds,
    }


def summarize_model_retry_events(
    retry_events: list[dict[str, Any]],
    *,
    succeeded: bool | None = None,
) -> dict[str, Any] | None:
    if not retry_events:
        return None

    retry_count = sum(1 for event in retry_events if event.get("will_retry") is True)
    last_event = retry_events[-1]
    if succeeded is True:
        status = "succeeded_after_retry"
    elif succeeded is False:
        status = "failed_after_retries"
    else:
        status = "retrying" if last_event.get("will_retry") else "failed"
    return {
        "status": status,
        "retry_count": retry_count,
        "request_error_count": len(retry_events),
        "max_attempts": last_event.get("max_attempts"),
        "last_error": last_event.get("error"),
        "next_retry_delay_seconds": last_event.get("next_retry_delay_seconds"),
        "errors": retry_events,
    }


def invoke_model_with_retries(
    model: Any,
    messages: Any,
    *,
    retry_delays_seconds: Sequence[int] = MODEL_REQUEST_RETRY_DELAYS_SECONDS,
    sleep_fn: Callable[[float], None] | None = None,
    on_retry_event: ModelRetryEventCallback | None = None,
) -> Any:
    """Invoke a chat model, retrying transient request failures with fixed backoff delays."""

    sleep = sleep_fn or time.sleep
    max_attempts = len(retry_delays_seconds) + 1
    for attempt_index in range(len(retry_delays_seconds) + 1):
        try:
            return model.invoke(messages)
        except Exception as exc:
            retry_delay_seconds = (
                retry_delays_seconds[attempt_index]
                if attempt_index < len(retry_delays_seconds)
                else None
            )
            if on_retry_event is not None:
                on_retry_event(
                    _model_retry_event(
                        exc=exc,
                        attempt_index=attempt_index,
                        max_attempts=max_attempts,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                )
            if attempt_index >= len(retry_delays_seconds):
                raise
            sleep(retry_delays_seconds[attempt_index])

    raise RuntimeError("unreachable model retry state")
