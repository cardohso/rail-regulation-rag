"""Local text embeddings (no API cost)."""

from __future__ import annotations

from typing import Protocol

# bge models are trained to embed queries with this instruction and passages without it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_embedding_dimension()
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def _encode(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            show_progress_bar=show_progress,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, show_progress=len(texts) > 100)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]
