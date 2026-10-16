import json
from contextlib import contextmanager

import pytest

from rail_rag import prompts
from rail_rag.config import Settings
from rail_rag.graph import GraphConfig
from rail_rag.pipeline import Pipeline
from rail_rag.tracing import LangfuseTracer, NoopTracer, make_tracer
from rail_rag.vectorstore import QdrantStore
from tests.fakes import FakeLLM, HashEmbedder, HashSparseEncoder
from tests.test_graph import PASSAGES


class Obs:
    def __init__(self, name, parent, **kw):
        self.name, self.parent, self.kw, self.updates = name, parent, kw, {}

    def update(self, **kw):
        self.updates.update(kw)


class RecordingTracer:
    enabled = True

    def __init__(self):
        self.observations: list[Obs] = []
        self._stack: list[Obs] = []
        self.flushed = False

    @contextmanager
    def observe(self, name, **kw):
        obs = Obs(name, self._stack[-1].name if self._stack else None, **kw)
        self.observations.append(obs)
        self._stack.append(obs)
        try:
            yield obs
        except Exception as exc:
            obs.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            self._stack.pop()

    def flush(self):
        self.flushed = True

    def named(self, name):
        return [o for o in self.observations if o.name == name]


ANSWERING = {
    prompts.REWRITE: "emergency lighting tunnels",
    prompts.GRADE: json.dumps({"relevant": [1]}),
    prompts.GENERATE: json.dumps({"answer": "90 minutes [1].", "insufficient": False}),
}


def pipeline(tracer, replies=ANSWERING):
    e = HashEmbedder()
    store = QdrantStore(":memory:", sparse_encoder=HashSparseEncoder())
    store.reset(e.dim)
    store.upsert(PASSAGES, e.embed_documents([p.embed_text for p in PASSAGES]))
    cfg = GraphConfig(k=3, min_score=0.1)
    return Pipeline(FakeLLM(replies), e, store, cfg, store_name="qdrant", tracer=tracer)


def test_one_trace_per_question_with_the_returned_trace_id():
    tracer = RecordingTracer()
    result = pipeline(tracer).ask("How long must tunnel emergency lighting work?")
    [root] = tracer.named("rag-ask")
    assert root.parent is None
    assert root.kw["trace_id"] == result.trace_id
    assert root.kw["metadata"] == {"generator": "fake:scripted", "vector_store": "qdrant"}
    assert root.updates["output"]["citations"] == [PASSAGES[0].id]


def test_each_node_is_a_typed_child_of_the_root():
    tracer = RecordingTracer()
    pipeline(tracer).ask("tunnel lighting")
    nodes = [(o.name, o.kw["as_type"]) for o in tracer.observations if o.parent == "rag-ask"]
    assert nodes == [
        ("rewrite", "chain"),
        ("retrieve", "retriever"),
        ("grade", "evaluator"),
        ("generate", "chain"),
    ]
    [retrieve] = tracer.named("retrieve")
    top = retrieve.updates["output"]["hits"][0]  # hits are summarised as id + score
    assert top["id"] == PASSAGES[0].id and isinstance(top["score"], float)


def test_llm_calls_are_generations_with_prompt_output_and_model():
    tracer = RecordingTracer()
    pipeline(tracer).ask("tunnel lighting")
    gens = tracer.named("llm")
    assert [g.parent for g in gens] == ["rewrite", "grade", "generate"]
    assert all(g.kw["as_type"] == "generation" and g.kw["model"] == "fake:scripted" for g in gens)
    assert gens[0].kw["input"][0] == {"role": "system", "content": prompts.REWRITE}
    assert gens[0].updates["output"] == "emergency lighting tunnels"


def test_refusal_path_is_traced():
    tracer = RecordingTracer()
    pipeline(tracer, {**ANSWERING, prompts.GRADE: json.dumps({"relevant": []})}).ask("france?")
    assert tracer.named("refuse")
    assert tracer.named("rag-ask")[0].updates["output"]["refusal_reason"] == "no_relevant_passages"


def test_llm_errors_are_marked_on_the_generation_and_re_raised():
    def boom(user):
        raise RuntimeError("provider down")

    tracer = RecordingTracer()
    with pytest.raises(RuntimeError):
        pipeline(tracer, {**ANSWERING, prompts.GRADE: boom}).ask("tunnel lighting")
    failed = [g for g in tracer.named("llm") if g.parent == "grade"]
    assert failed[0].updates["level"] == "ERROR"


def test_without_keys_tracing_is_a_noop():
    assert isinstance(make_tracer(Settings(_env_file=None)), NoopTracer)
    result = pipeline(NoopTracer()).ask("tunnel lighting")
    assert result.refused is False and len(result.trace_id) == 32


def test_real_langfuse_sdk_accepts_our_calls_offline():
    # tracing_enabled=False: the SDK validates our calls but exports nothing (no network).
    tracer = LangfuseTracer("pk-lf-test", "sk-lf-test", "http://localhost:9", tracing_enabled=False)
    result = pipeline(tracer).ask("How long must tunnel emergency lighting work?")
    assert result.answer == "90 minutes [1]."
    tracer.flush()


def test_make_tracer_uses_langfuse_when_keys_are_set(monkeypatch):
    created = {}

    class FakeLangfuse:
        def __init__(self, **kw):
            created.update(kw)

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)
    s = Settings(_env_file=None, langfuse_public_key="pk", langfuse_secret_key="sk")
    assert isinstance(make_tracer(s), LangfuseTracer)
    assert created["base_url"] == "https://cloud.langfuse.com"
