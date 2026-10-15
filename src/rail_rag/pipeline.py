"""Wires settings -> LLM, embedder, vector store -> LangGraph agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from rail_rag.citations import Citation, format_citations
from rail_rag.config import Settings
from rail_rag.embeddings import Embedder, SentenceTransformerEmbedder
from rail_rag.graph import GraphConfig, build_graph
from rail_rag.providers import LLM, make_llm
from rail_rag.vectorstore import SearchHit, VectorStore, make_store


@dataclass
class AskResult:
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    trace_id: str
    query: str = ""
    contexts: list[SearchHit] = field(default_factory=list)  # passages given to the generator


class Pipeline:
    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        store: VectorStore,
        cfg: GraphConfig | None = None,
        store_name: str = "custom",
    ):
        self.llm, self.store, self.store_name = llm, store, store_name
        self.graph = build_graph(llm, embedder, store, cfg)

    @classmethod
    def from_settings(cls, s: Settings) -> Pipeline:
        llm = make_llm(s.generator_provider, s.api_key(s.generator_provider), s.generator_model)
        cfg = GraphConfig(k=s.top_k, min_score=s.min_score)
        embedder = SentenceTransformerEmbedder(s.embedding_model)
        return cls(llm, embedder, make_store(s), cfg, store_name=s.vector_store)

    def ask(self, question: str, trace_id: str | None = None) -> AskResult:
        trace_id = trace_id or uuid.uuid4().hex
        state = self.graph.invoke({"question": question})
        return AskResult(
            answer=state["answer"],
            refused=bool(state.get("refused")),
            refusal_reason=state.get("refusal_reason"),
            citations=format_citations(state.get("cited", [])),
            trace_id=trace_id,
            query=state.get("query", ""),
            contexts=state.get("context", []),
        )

    def health(self) -> dict:
        return {
            "passages": self.store.count(),
            "vector_store": self.store_name,
            "generator": self.llm.name,
        }
