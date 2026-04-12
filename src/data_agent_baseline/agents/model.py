from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def create_chat_model(
    *,
    model: str,
    api_base: str,
    api_key: str,
    api_key_env: str | None = None,
    temperature: float,
    enable_thinking: bool = False,
) -> BaseChatModel:
    if not api_key:
        if api_key_env:
            raise RuntimeError(
                f"Missing model API key. Checked project .env for config.agent.api_key_env={api_key_env!r}."
            )
        raise RuntimeError(
            "Missing model API key in config.agent.api_key or the project .env file. "
            "Set config.agent.api_key directly or point config.agent.api_key_env at a key in .env."
        )

    request_kwargs: dict[str, object] = {
        "model": model,
        "base_url": api_base.rstrip("/"),
        "api_key": api_key,
        "temperature": temperature,
        "max_retries": 2,
    }
    if enable_thinking:
        request_kwargs["extra_body"] = {"enable_thinking": True}

    return ChatOpenAI(**request_kwargs)
