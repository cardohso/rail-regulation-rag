import pytest

from rail_rag.providers import (
    GeminiLLM,
    GroqLLM,
    LLMUnavailable,
    make_llm,
    parse_json,
    with_retries,
)


class HTTPError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = type("R", (), {"headers": {"retry-after": retry_after}})()


def flaky(failures: list[Exception], result="ok"):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if failures:
            raise failures.pop(0)
        return result

    return fn, calls


def test_retries_rate_limits_then_succeeds_with_growing_delays():
    sleeps = []
    fn, calls = flaky([HTTPError(429), HTTPError(503)])
    assert with_retries(fn, sleep=sleeps.append, base_delay=1.0) == "ok"
    assert calls["n"] == 3
    assert 0.8 <= sleeps[0] <= 1.2 and 1.6 <= sleeps[1] <= 2.4


def test_honours_retry_after_header():
    sleeps = []
    fn, _ = flaky([HTTPError(429, retry_after="7")])
    with_retries(fn, sleep=sleeps.append)
    assert 7 * 0.8 <= sleeps[0] <= 7 * 1.2


def test_does_not_retry_client_errors():
    fn, calls = flaky([HTTPError(401)])
    with pytest.raises(HTTPError):
        with_retries(fn, sleep=lambda s: None)
    assert calls["n"] == 1


def test_gives_up_with_llm_unavailable_after_max_attempts():
    fn, calls = flaky([HTTPError(429)] * 10)
    with pytest.raises(LLMUnavailable):
        with_retries(fn, attempts=3, sleep=lambda s: None)
    assert calls["n"] == 3


def test_retries_connection_errors_without_status():
    class APIConnectionError(Exception):
        pass

    fn, calls = flaky([APIConnectionError("reset")])
    assert with_retries(fn, sleep=lambda s: None) == "ok"
    assert calls["n"] == 2


def test_factory_builds_each_provider_with_default_model():
    gem = make_llm("gemini", "dummy-key")
    groq = make_llm("groq", "dummy-key")
    assert isinstance(gem, GeminiLLM) and gem.name.startswith("gemini:")
    assert isinstance(groq, GroqLLM) and groq.name == "groq:openai/gpt-oss-120b"
    assert make_llm("groq", "k", "qwen/qwen3.8-27b").name == "groq:qwen/qwen3.8-27b"


def test_factory_rejects_unknown_provider_and_missing_key():
    with pytest.raises(ValueError, match="unknown"):
        make_llm("openai", "k")
    with pytest.raises(ValueError, match="no API key"):
        make_llm("gemini", "")


@pytest.mark.parametrize(
    "reply",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here it is: {"a": 1} Hope that helps.'],
)
def test_parse_json_tolerates_fences_and_prose(reply):
    assert parse_json(reply) == {"a": 1}


def test_parse_json_raises_on_no_object():
    with pytest.raises(ValueError):
        parse_json("no json here")
