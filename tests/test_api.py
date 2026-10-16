import json

import pytest
from fastapi.testclient import TestClient

from rail_rag import prompts
from rail_rag.api import create_app
from rail_rag.graph import GraphConfig
from rail_rag.pipeline import Pipeline
from rail_rag.providers import LLMUnavailable
from rail_rag.vectorstore import QdrantStore
from tests.fakes import FakeLLM, HashEmbedder, HashSparseEncoder
from tests.test_graph import PASSAGES


def make_client(replies: dict) -> TestClient:
    e = HashEmbedder()
    store = QdrantStore(":memory:", sparse_encoder=HashSparseEncoder())
    store.reset(e.dim)
    store.upsert(PASSAGES, e.embed_documents([p.embed_text for p in PASSAGES]))
    llm = FakeLLM({prompts.REWRITE: "emergency lighting tunnels", **replies})
    pipeline = Pipeline(llm, e, store, GraphConfig(k=3, min_score=0.1), store_name="qdrant")
    return TestClient(create_app(pipeline))


ANSWERING = {
    prompts.GRADE: json.dumps({"relevant": [1]}),
    prompts.GENERATE: json.dumps({"answer": "90 minutes [1].", "insufficient": False}),
}


def test_health_reports_index_and_models():
    with make_client(ANSWERING) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() | {"version": "x"} == {
        "status": "ok", "version": "x", "passages": 3, "vector_store": "qdrant",
        "search_mode": "dense", "generator": "fake:scripted",
        "tracing": False,
    }  # fmt: skip


def test_ask_returns_answer_citations_and_trace_id():
    with make_client(ANSWERING) as client:
        r = client.post("/ask", json={"question": "How long must tunnel emergency lighting work?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "90 minutes [1]." and body["refused"] is False
    [c] = body["citations"]
    assert c["n"] == 1 and c["doc_id"] == "tsi-srt" and c["passage_id"] == PASSAGES[0].id
    assert c["passage"] == PASSAGES[0].text
    assert len(body["trace_id"]) == 32


def test_ask_refusal_has_no_citations():
    with make_client({prompts.GRADE: json.dumps({"relevant": []})}) as client:
        body = client.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "no_relevant_passages"
    assert body["citations"] == [] and body["answer"] == prompts.REFUSAL


@pytest.mark.parametrize("question", ["", "hi", "x" * 501])
def test_ask_validates_question_length(question):
    with make_client(ANSWERING) as client:
        assert client.post("/ask", json={"question": question}).status_code == 422


def test_rate_limited_llm_returns_503():
    def boom(user):
        raise LLMUnavailable("quota")

    with make_client({**ANSWERING, prompts.GRADE: boom}) as client:
        r = client.post("/ask", json={"question": "tunnel lighting duration"})
    assert r.status_code == 503


def test_ui_is_served_at_root():
    with make_client(ANSWERING) as client:
        r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/ask" in r.text  # the page talks to the API


def test_ask_is_rate_limited_per_client():
    from rail_rag.api import create_app as make_app
    from rail_rag.ratelimit import RateLimiter

    e = HashEmbedder()
    store = QdrantStore(":memory:", sparse_encoder=HashSparseEncoder())
    store.reset(e.dim)
    store.upsert(PASSAGES, e.embed_documents([p.embed_text for p in PASSAGES]))
    llm = FakeLLM({prompts.REWRITE: "emergency lighting tunnels", **ANSWERING})
    app = make_app(
        Pipeline(llm, e, store, GraphConfig(k=3, min_score=0.1)),
        RateLimiter(per_client_per_minute=2, per_day=100),
    )
    with TestClient(app) as client:
        codes = [
            client.post("/ask", json={"question": "tunnel lighting"},
                        headers={"x-forwarded-for": ip}).status_code
            for ip in ["1.1.1.1", "1.1.1.1", "1.1.1.1", "2.2.2.2"]
        ]  # fmt: skip
    assert codes == [200, 200, 429, 200]
