from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import HumanMessage


MODEL_REQUEST_RETRY_DELAYS_SECONDS = (0, 0, 0)
ModelRetryEventCallback = Callable[[dict[str, Any]], None]
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
TIMEOUT_RETRY_PROMPT = (
    "The previous model request timed out. Keep your reasoning very brief, "
    "avoid restating prior analysis, and immediately choose the next concrete "
    "tool call or final answer."
)


def _raw_exception_content(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text:
        return response_text

    body = getattr(exc, "body", None)
    if body not in (None, ""):
        return str(body)

    return str(exc)


def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    return None


def _exception_type(exc: Exception) -> str:
    return type(exc).__name__


def _is_timeout_error(exc: Exception) -> bool:
    error_type = _exception_type(exc).lower()
    if "timeout" in error_type:
        return True
    error_text = str(exc).lower()
    return "timed out" in error_text or "timeout" in error_text


def _is_connection_error(exc: Exception) -> bool:
    error_type = _exception_type(exc).lower()
    if "timeout" in error_type:
        return False
    if "connection" in error_type or "connect" in error_type:
        return True
    return any(
        marker in str(exc).lower()
        for marker in (
            "connection aborted",
            "connection reset",
            "connection refused",
            "connection error",
            "server disconnected",
            "remote protocol error",
        )
    )


def _is_retryable_model_error(exc: Exception) -> bool:
    status_code = _exception_status_code(exc)
    if status_code is not None:
        return status_code in RETRYABLE_STATUS_CODES

    return _is_timeout_error(exc) or _is_connection_error(exc)


def _append_timeout_retry_prompt(messages: Any) -> Any:
    if isinstance(messages, list):
        if all(isinstance(message, dict) for message in messages):
            return [*messages, {"role": "user", "content": TIMEOUT_RETRY_PROMPT}]
        return [*messages, HumanMessage(content=TIMEOUT_RETRY_PROMPT)]
    if isinstance(messages, tuple):
        return (*messages, HumanMessage(content=TIMEOUT_RETRY_PROMPT))
    return messages


def _model_retry_event(
    *,
    exc: Exception,
    attempt_index: int,
    max_attempts: int,
    retry_delay_seconds: int | None,
    retryable: bool,
    retry_prompt_added: bool,
) -> dict[str, Any]:
    status_code = _exception_status_code(exc)
    return {
        "attempt": attempt_index + 1,
        "max_attempts": max_attempts,
        "error": _raw_exception_content(exc),
        "error_type": _exception_type(exc),
        "status_code": status_code,
        "retryable": retryable,
        "will_retry": retryable and retry_delay_seconds is not None,
        "next_retry_delay_seconds": retry_delay_seconds,
        "retry_prompt_added": retry_prompt_added,
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
    """Invoke a chat model, retrying only clearly transient request failures."""

    sleep = sleep_fn or time.sleep
    max_attempts = len(retry_delays_seconds) + 1
    current_messages = messages
    for attempt_index in range(len(retry_delays_seconds) + 1):
        try:
            return model.invoke(current_messages)
        except Exception as exc:
            is_timeout = _is_timeout_error(exc)
            retryable = _is_retryable_model_error(exc)
            if is_timeout and attempt_index > 0:
                retryable = False
            retry_delay_seconds = (
                retry_delays_seconds[attempt_index]
                if retryable and attempt_index < len(retry_delays_seconds)
                else None
            )
            retry_prompt_added = is_timeout and retry_delay_seconds is not None
            if on_retry_event is not None:
                on_retry_event(
                    _model_retry_event(
                        exc=exc,
                        attempt_index=attempt_index,
                        max_attempts=max_attempts,
                        retry_delay_seconds=retry_delay_seconds,
                        retryable=retryable,
                        retry_prompt_added=retry_prompt_added,
                    )
                )
            if not retryable or attempt_index >= len(retry_delays_seconds):
                raise
            if retry_prompt_added:
                current_messages = _append_timeout_retry_prompt(current_messages)
            sleep(retry_delays_seconds[attempt_index])

    raise RuntimeError("unreachable model retry state")
