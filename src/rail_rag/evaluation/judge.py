"""LLM-as-a-judge metrics via RAGAS. The judge is chosen by provider name, like the generator,
so the same answers can be scored by judges from different model families."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from rail_rag.embeddings import Embedder
from rail_rag.providers import DEFAULT_MODELS

# Both providers expose an OpenAI-compatible endpoint, which is what RAGAS' instructor adapter
# speaks. Retries/backoff for 429s are handled by the OpenAI client (honours Retry-After).
OPENAI_COMPATIBLE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
MARKERS = re.compile(r"\s*\[\d+\]")


class Judge(Protocol):
    name: str

    async def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float: ...

    async def answer_relevancy(self, question: str, answer: str) -> float: ...


def strip_markers(answer: str) -> str:
    """Citation markers are for the reader; they would only add noise to claim extraction."""
    return MARKERS.sub("", answer).strip()


class RagasJudge:
    def __init__(
        self,
        provider: str,
        api_key: str,
        embedder: Embedder,
        model: str | None = None,
        cache_dir: Path | None = None,
    ):
        from openai import AsyncOpenAI
        from ragas.cache import DiskCacheBackend
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        if provider not in OPENAI_COMPATIBLE_URLS:
            raise ValueError(f"unknown judge provider {provider!r}")
        if not api_key:
            raise ValueError(f"no API key set for judge provider {provider!r}")
        model = model or DEFAULT_MODELS[provider]
        self.name = f"{provider}:{model}"
        client = AsyncOpenAI(
            api_key=api_key, base_url=OPENAI_COMPATIBLE_URLS[provider], max_retries=10
        )
        cache = DiskCacheBackend(str(cache_dir)) if cache_dir else None
        llm = llm_factory(model, provider="openai", client=client, cache=cache, temperature=0.0)
        self._faithfulness = Faithfulness(llm=llm)
        self._relevancy = AnswerRelevancy(llm=llm, embeddings=_RagasEmbeddings(embedder))

    async def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        result = await self._faithfulness.ascore(
            user_input=question, response=strip_markers(answer), retrieved_contexts=contexts
        )
        return float(result.value)

    async def answer_relevancy(self, question: str, answer: str) -> float:
        result = await self._relevancy.ascore(user_input=question, response=strip_markers(answer))
        return float(result.value)


def _RagasEmbeddings(embedder: Embedder):
    """Adapter so RAGAS uses the same local embedding model as retrieval (no API cost)."""
    from ragas.embeddings import BaseRagasEmbedding

    class Adapter(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return embedder.embed_documents([text])[0]

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return self.embed_text(text)

    return Adapter()
