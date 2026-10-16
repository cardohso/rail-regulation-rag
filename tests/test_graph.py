import json

import pytest

from rail_rag import prompts
from rail_rag.chunking import Passage
from rail_rag.citations import format_citations, renumber
from rail_rag.graph import GraphConfig, build_graph, passage_numbers
from rail_rag.vectorstore import QdrantStore, SearchHit
from tests.fakes import FakeLLM, HashEmbedder

PASSAGES = [
    Passage("tsi-srt:Annex §4.2.1.5.4#1", "tsi-srt", "SRT TSI", "Annex §4.2.1.5.4",
            "4.2.1.5.4. Emergency lighting", "emergency lighting in tunnels shall work 90 minutes"),
    Passage("tsi-noi:Annex §4.2.1#1", "tsi-noi", "NOI TSI", "Annex §4.2.1", "4.2.1. Stationary",
            "stationary noise limit for electric locomotives is 70 dB"),
    Passage("tsi-prm:Annex §4.2.1.2#1", "tsi-prm", "PRM TSI", "Annex §4.2.1.2", "4.2.1.2. Parking",
            "parking facilities for persons with reduced mobility"),
]  # fmt: skip


def j(**kw) -> str:
    return json.dumps(kw)


@pytest.fixture
def store():
    e = HashEmbedder()
    s = QdrantStore(":memory:")
    s.reset(e.dim)
    s.upsert(PASSAGES, e.embed_documents([p.embed_text for p in PASSAGES]))
    return s


def run(store, replies, cfg=None, question="tunnel emergency lighting"):
    cfg = cfg or GraphConfig(k=3, min_score=0.1)
    llm = FakeLLM({prompts.REWRITE: "emergency lighting tunnels", **replies})
    state = build_graph(llm, HashEmbedder(), store, cfg).invoke({"question": question})
    return state, llm


def test_answers_with_citation_mapped_to_passage(store):
    state, llm = run(store, {
        prompts.GRADE: j(relevant=[1]),
        prompts.GENERATE: j(answer="It must work for 90 minutes [1].", insufficient=False),
    })  # fmt: skip
    assert state["refused"] is False
    assert state["answer"] == "It must work for 90 minutes [1]."
    assert [h.passage.id for h in state["cited"]] == ["tsi-srt:Annex §4.2.1.5.4#1"]
    assert [c[0] for c in llm.calls] == [prompts.REWRITE, prompts.GRADE, prompts.GENERATE]


def test_low_retrieval_score_refuses_without_grading_or_generating(store):
    state, llm = run(store, {}, cfg=GraphConfig(k=3, min_score=0.99))
    assert state["refused"] and state["refusal_reason"] == "low_retrieval_score"
    assert state["answer"] == prompts.REFUSAL and state["cited"] == []
    assert len(llm.calls) == 1  # only the rewrite


def test_no_relevant_passages_refuses_without_generating(store):
    state, llm = run(store, {prompts.GRADE: j(relevant=[])})
    assert state["refusal_reason"] == "no_relevant_passages"
    assert prompts.GENERATE not in [c[0] for c in llm.calls]


def test_invalid_grader_numbers_are_ignored(store):
    state, _ = run(store, {prompts.GRADE: j(relevant=[0, 99, "x", None])})
    assert state["refusal_reason"] == "no_relevant_passages"


def test_grader_picks_keep_retrieval_order_and_cap(store):
    state, _ = run(
        store,
        {
            prompts.GRADE: j(relevant=[3, 1, 2]),
            prompts.GENERATE: j(answer="x [1]", insufficient=False),
        },
        cfg=GraphConfig(k=3, min_score=0.1, max_context=2),
    )
    assert state["context"] == state["hits"][:2]


def test_generator_insufficient_refuses(store):
    state, _ = run(store, {
        prompts.GRADE: j(relevant=[1]),
        prompts.GENERATE: j(answer="", insufficient=True),
    })  # fmt: skip
    assert state["refused"] and state["refusal_reason"] == "insufficient_context"
    assert state["answer"] == prompts.REFUSAL


def test_uncited_answer_is_refused(store):
    state, _ = run(store, {
        prompts.GRADE: j(relevant=[1]),
        prompts.GENERATE: j(answer="Ninety minutes, I believe.", insufficient=False),
    })  # fmt: skip
    assert state["refusal_reason"] == "no_citations"
    assert state["answer"] == prompts.REFUSAL


def test_rewrite_falls_back_to_question_when_empty(store):
    llm = FakeLLM({prompts.REWRITE: "  ", prompts.GRADE: j(relevant=[])})
    state = build_graph(llm, HashEmbedder(), store, GraphConfig(min_score=0.1)).invoke(
        {"question": "tunnel emergency lighting"}
    )
    assert state["query"] == "tunnel emergency lighting"


def hit(i: int) -> SearchHit:
    return SearchHit(PASSAGES[i], 0.9 - i / 10)


def test_renumber_drops_invalid_markers_and_orders_by_first_use():
    ctx = [hit(0), hit(1), hit(2)]
    text, cited = renumber("A [3]. B [1][3]. C [7].", ctx)
    assert text == "A [1]. B [2][1]. C."
    assert cited == [ctx[2], ctx[0]]


def test_format_citations_links_to_eurlex_consolidated_version():
    [c] = format_citations([hit(0)])
    assert c.n == 1
    assert (c.doc_id, c.document, c.section) == ("tsi-srt", "SRT TSI", "Annex §4.2.1.5.4")
    assert c.url.endswith("CELEX:02014R1303-20240129")
    assert c.passage == PASSAGES[0].text


@pytest.mark.parametrize(
    "reply, expected",
    [
        ([1, 3], [1, 3]),
        ([[1], [2], [3]], [1, 2, 3]),  # seen from Gemini at temperature 0
        (["2", " 3 "], [2, 3]),
        (2, [2]),
        ([1, 1, True, None, "x"], [1]),
        (None, []),
    ],
)
def test_passage_numbers_tolerates_grader_shape_drift(reply, expected):
    assert passage_numbers(reply) == expected


def test_nested_grader_reply_no_longer_causes_false_refusal(store):
    state, _ = run(store, {
        prompts.GRADE: j(relevant=[[1], [2]]),
        prompts.GENERATE: j(answer="90 minutes [1].", insufficient=False),
    })  # fmt: skip
    assert state["refused"] is False


def test_llm_calls_request_structured_output(store):
    calls = []

    class SpyLLM(FakeLLM):
        def complete(self, system, user, *, schema=None):
            calls.append((system, schema))
            return super().complete(system, user, schema=schema)

    llm = SpyLLM({
        prompts.REWRITE: "q",
        prompts.GRADE: j(relevant=[1]),
        prompts.GENERATE: j(answer="x [1]", insufficient=False),
    })  # fmt: skip
    build_graph(llm, HashEmbedder(), store, GraphConfig(min_score=0.0)).invoke(
        {"question": "tunnel"}
    )
    assert dict(calls) == {
        prompts.REWRITE: None,
        prompts.GRADE: prompts.GRADE_SCHEMA,
        prompts.GENERATE: prompts.GENERATE_SCHEMA,
    }
