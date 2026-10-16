"""Optional Langfuse tracing. Without keys, a no-op tracer keeps the app and tests offline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from rail_rag.config import Settings
from rail_rag.providers import LLM
from rail_rag.vectorstore import SearchHit


class Observation(Protocol):
    def update(self, **kwargs: Any) -> Any: ...


class Tracer(Protocol):
    enabled: bool

    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        model: str | None = None,
        metadata: dict | None = None,
        trace_id: str | None = None,
    ) -> Any:
        """Context manager yielding an Observation. trace_id makes it the root of that trace."""

    def flush(self) -> None: ...


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        return None


class NoopTracer:
    enabled = False

    @contextmanager
    def observe(self, name: str, **kwargs: Any) -> Iterator[Observation]:
        yield _NoopObservation()

    def flush(self) -> None:
        return None


class LangfuseTracer:
    enabled = True

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        environment: str = "local",
        tracing_enabled: bool = True,
    ):
        from langfuse import Langfuse

        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=host,
            environment=environment,
            tracing_enabled=tracing_enabled,
        )

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        model: str | None = None,
        metadata: dict | None = None,
        trace_id: str | None = None,
    ) -> Iterator[Observation]:
        from langfuse import propagate_attributes

        kwargs: dict[str, Any] = {"name": name, "as_type": as_type, "input": input}
        if model:
            kwargs["model"] = model
        if trace_id:
            kwargs["trace_context"] = {"trace_id": trace_id}
        with self.client.start_as_current_observation(**kwargs) as obs:
            try:
                if trace_id:  # root observation: name the trace and tag every child span
                    with propagate_attributes(trace_name=name, metadata=metadata):
                        yield obs
                else:
                    if metadata:
                        obs.update(metadata=metadata)
                    yield obs
            except Exception as exc:
                obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
                raise

    def flush(self) -> None:
        self.client.flush()


def make_tracer(settings: Settings) -> Tracer:
    public = settings.langfuse_public_key.get_secret_value()
    secret = settings.langfuse_secret_key.get_secret_value()
    if not (public and secret):
        return NoopTracer()
    return LangfuseTracer(public, secret, settings.langfuse_host, settings.langfuse_environment)


class TracedLLM:
    """Records every LLM call as a Langfuse 'generation' with prompt, output and model."""

    def __init__(self, llm: LLM, tracer: Tracer):
        self.llm, self.tracer, self.name = llm, tracer, llm.name

    def complete(self, system: str, user: str, *, schema: dict | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with self.tracer.observe(
            "llm", as_type="generation", input=messages, model=self.llm.name,
            metadata={"structured_output": schema is not None},
        ) as gen:  # fmt: skip
            output = self.llm.complete(system, user, schema=schema)
            gen.update(output=output)
            return output


def summarize(value: Any) -> Any:
    """Make graph state JSON-friendly and compact for traces (hits -> id + score)."""
    if isinstance(value, SearchHit):
        return {"id": value.passage.id, "score": round(value.score, 4)}
    if isinstance(value, dict):
        return {k: summarize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [summarize(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value
