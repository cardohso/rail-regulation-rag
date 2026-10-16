"""LLM provider abstraction: generator and judge are chosen by config, never hard-coded."""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from typing import Protocol

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-120b",
}
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLM(Protocol):
    name: str  # "<provider>:<model>", recorded in traces and eval results

    def complete(self, system: str, user: str, *, schema: dict | None = None) -> str:
        """Return the model's text. With a JSON schema, the provider must return matching JSON."""


class LLMUnavailable(RuntimeError):
    """Raised when a provider keeps failing after all retries (e.g. free-tier quota exhausted)."""


def _status(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    status = _status(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    return type(exc).__name__.endswith(("ConnectionError", "TimeoutError", "Timeout"))


def with_retries[T](
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying rate limits and transient errors with exponential backoff and jitter."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            if attempt == attempts - 1:
                raise LLMUnavailable(f"gave up after {attempts} attempts: {exc}") from exc
            delay = _retry_after(exc) or min(max_delay, base_delay * 2**attempt)
            sleep(delay * random.uniform(0.8, 1.2))
    raise AssertionError("unreachable")


class GeminiLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["gemini"]):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"

    def complete(self, system: str, user: str, *, schema: dict | None = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json" if schema else None,
            response_json_schema=schema,
        )
        response = with_retries(
            lambda: self.client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        )
        return response.text or ""


class GroqLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["groq"]):
        import groq

        self.client = groq.Groq(api_key=api_key, max_retries=0)  # we retry ourselves
        self.model = model
        self.name = f"groq:{model}"

    def complete(self, system: str, user: str, *, schema: dict | None = None) -> str:
        kwargs = {}
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema},
            }
        response = with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                **kwargs,
            )
        )
        return response.choices[0].message.content or ""


PROVIDERS: dict[str, type] = {"gemini": GeminiLLM, "groq": GroqLLM}


def make_llm(provider: str, api_key: str, model: str | None = None) -> LLM:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; choose from {sorted(PROVIDERS)}")
    if not api_key:
        raise ValueError(f"no API key set for provider {provider!r} (see .env.example)")
    return PROVIDERS[provider](api_key, model or DEFAULT_MODELS[provider])


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ```json fences and surrounding prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    return json.loads(match.group(0))
