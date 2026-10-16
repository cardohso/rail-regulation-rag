"""Offline stand-ins so tests never download models or call external services."""

import hashlib
import math
import re


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector: shared words => higher cosine."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class HashSparseEncoder:
    """Term counts hashed to indices: a stand-in for BM25 term weights (Qdrant adds the IDF)."""

    def _sparse(self, text: str) -> tuple[list[int], list[float]]:
        counts: dict[int, float] = {}
        for word in re.findall(r"\w+", text.lower()):
            i = int(hashlib.md5(word.encode()).hexdigest(), 16) % 100_000
            counts[i] = counts.get(i, 0.0) + 1.0
        return list(counts), list(counts.values())

    def embed_documents(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [self._sparse(t) for t in texts]

    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        return self._sparse(text)


class FakeLLM:
    """Scripted LLM: replies are keyed by system prompt; a reply can be a str or fn(user)->str."""

    name = "fake:scripted"

    def __init__(self, replies: dict):
        self.replies = replies
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, schema: dict | None = None) -> str:
        self.calls.append((system, user))
        reply = self.replies[system]
        return reply(user) if callable(reply) else reply
