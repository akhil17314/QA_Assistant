"""Lazy Gemini model construction with bounded retry, jitter, and call tracking."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from collections.abc import Iterator
from typing import Any, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)
_active_call_counter: ContextVar[list[int] | None] = ContextVar(
    "active_llm_call_counter", default=None
)


class _CallCounter(BaseCallbackHandler):
    """Count model starts associated with the active request context."""

    def _increment(self) -> None:
        counter = _active_call_counter.get()
        if counter is not None:
            counter[0] += 1

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: object) -> None:
        self._increment()

    def on_chat_model_start(
        self, serialized: dict, messages: list[list[object]], **kwargs: object
    ) -> None:
        logger.info("LLM call started")
        self._increment()

    def on_llm_error(self, error: BaseException, **kwargs: object) -> None:
        logger.warning("LLM call failed: %s: %s", type(error).__name__, error)


_call_counter = _CallCounter()


@contextmanager
def track_llm_calls() -> Iterator[list[int]]:
    """Track actual LLM starts made inside the current synchronous request."""
    counter = [0]
    token: Token[list[int] | None] = _active_call_counter.set(counter)
    try:
        yield counter
    finally:
        _active_call_counter.reset(token)


def _create_chat_model(model: str | None, temperature: float) -> Any:
    """Create a raw Gemini chat model; fail clearly when its API key is missing."""
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for LLM-backed requests.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        temperature=temperature,
        google_api_key=settings.google_api_key,
        max_retries=0,
        timeout=60,
    )


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Create a Gemini chat model with bounded retry and call tracking."""
    chat_model = _create_chat_model(model, temperature)
    return chat_model.with_config(callbacks=[_call_counter]).with_retry(
        stop_after_attempt=5,
        wait_exponential_jitter=True,
    )


def get_structured_llm(schema: type[SchemaT], model: str | None = None) -> Any:
    """Return a Gemini model that validates responses against a Pydantic schema."""
    structured_model = _create_chat_model(model, temperature=0).with_structured_output(schema)
    return structured_model.with_config(callbacks=[_call_counter]).with_retry(
        stop_after_attempt=5,
        wait_exponential_jitter=True,
    )
