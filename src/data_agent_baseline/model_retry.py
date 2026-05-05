from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any


MODEL_REQUEST_RETRY_DELAYS_SECONDS = (15, 30, 45, 60)


def invoke_model_with_retries(
    model: Any,
    messages: Any,
    *,
    retry_delays_seconds: Sequence[int] = MODEL_REQUEST_RETRY_DELAYS_SECONDS,
    sleep_fn: Callable[[float], None] | None = None,
) -> Any:
    """Invoke a chat model, retrying transient request failures with fixed backoff delays."""

    sleep = sleep_fn or time.sleep
    for attempt_index in range(len(retry_delays_seconds) + 1):
        try:
            return model.invoke(messages)
        except Exception:
            if attempt_index >= len(retry_delays_seconds):
                raise
            sleep(retry_delays_seconds[attempt_index])

    raise RuntimeError("unreachable model retry state")
