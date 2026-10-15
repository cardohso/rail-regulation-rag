"""The RAG agent as an explicit LangGraph state machine.

    rewrite -> retrieve -> grade --(relevant)--> generate --(grounded)--> END
                             \\                        \\
                              +--(no)--> refuse <------+--(insufficient / uncited)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from rail_rag import prompts
from rail_rag.citations import renumber
from rail_rag.embeddings import Embedder
from rail_rag.providers import LLM, parse_json
from rail_rag.tracing import NoopTracer, TracedLLM, Tracer, summarize
from rail_rag.vectorstore import SearchHit, VectorStore

RefusalReason = Literal[
    "low_retrieval_score", "no_relevant_passages", "insufficient_context", "no_citations"
]


@dataclass(frozen=True)
class GraphConfig:
    k: int = 8  # passages retrieved
    min_score: float = 0.60  # cosine floor: below this, refuse without calling the LLM
    max_context: int = 5  # graded passages passed to the generator


class RAGState(TypedDict, total=False):
    question: str
    query: str
    hits: list[SearchHit]
    context: list[SearchHit]
    answer: str
    cited: list[SearchHit]
    refused: bool
    refusal_reason: RefusalReason | None


# Langfuse observation type for each node, so the trace view shows what each step is.
NODE_TYPES = {
    "rewrite": "chain",
    "retrieve": "retriever",
    "grade": "evaluator",
    "generate": "chain",
    "refuse": "guardrail",
}


def build_graph(
    llm: LLM,
    embedder: Embedder,
    store: VectorStore,
    cfg: GraphConfig | None = None,
    tracer: Tracer | None = None,
):
    cfg = cfg or GraphConfig()
    tracer = tracer or NoopTracer()
    llm = TracedLLM(llm, tracer) if tracer.enabled else llm

    def rewrite(state: RAGState) -> RAGState:
        query = llm.complete(prompts.REWRITE, state["question"]).strip().splitlines()
        return {"query": (query[0] if query else "") or state["question"]}

    def retrieve(state: RAGState) -> RAGState:
        return {"hits": store.search(embedder.embed_query(state["query"]), k=cfg.k)}

    def grade(state: RAGState) -> RAGState:
        hits = state["hits"]
        if not hits or hits[0].score < cfg.min_score:
            return {"context": [], "refused": True, "refusal_reason": "low_retrieval_score"}
        user = f"Question: {state['question']}\n\nPassages:\n" + prompts.numbered_passages(
            [h.passage.embed_text for h in hits]
        )
        reply = parse_json(llm.complete(prompts.GRADE, user, json_mode=True))
        picked: list[int] = []
        for i in reply.get("relevant", []):
            if isinstance(i, int) and 1 <= i <= len(hits) and i not in picked:
                picked.append(i)
        # Keep retrieval order (best score first), not the order the grader listed them.
        context = [hits[i - 1] for i in sorted(picked)][: cfg.max_context]
        if not context:
            return {"context": [], "refused": True, "refusal_reason": "no_relevant_passages"}
        return {"context": context, "refused": False}

    def generate(state: RAGState) -> RAGState:
        context = state["context"]
        user = f"Question: {state['question']}\n\nPassages:\n" + prompts.numbered_passages(
            [h.passage.embed_text for h in context]
        )
        reply = parse_json(llm.complete(prompts.GENERATE, user, json_mode=True))
        if reply.get("insufficient") or not str(reply.get("answer", "")).strip():
            return {"refused": True, "refusal_reason": "insufficient_context", "cited": []}
        answer, cited = renumber(str(reply["answer"]), context)
        if not cited:  # an answer we cannot ground in a passage is not returned
            return {"refused": True, "refusal_reason": "no_citations", "cited": []}
        return {"answer": answer, "cited": cited, "refused": False, "refusal_reason": None}

    def refuse(state: RAGState) -> RAGState:
        return {"answer": prompts.REFUSAL, "cited": []}

    def after(state: RAGState) -> str:
        return "refuse" if state.get("refused") else "continue"

    def traced(name, fn):
        def node(state: RAGState) -> RAGState:
            with tracer.observe(name, as_type=NODE_TYPES[name], input=summarize(state)) as obs:
                update = fn(state)
                obs.update(output=summarize(update))
                return update

        return node

    g = StateGraph(RAGState)
    for name, fn in [
        ("rewrite", rewrite),
        ("retrieve", retrieve),
        ("grade", grade),
        ("generate", generate),
        ("refuse", refuse),
    ]:
        g.add_node(name, traced(name, fn))
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after, {"refuse": "refuse", "continue": "generate"})
    g.add_conditional_edges("generate", after, {"refuse": "refuse", "continue": END})
    g.add_edge("refuse", END)
    return g.compile()
