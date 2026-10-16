"""Local text embeddings (no API cost), run with ONNX Runtime via fastembed.

Same bge-small-en-v1.5 weights as sentence-transformers (cosine 1.0000 on our corpus), without
torch: the container image went from 2.99 GB to well under 1 GB.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

# bge models are trained to embed queries with this instruction and passages without it.
# fastembed's query_embed() does NOT add it (cosine drops to 0.956), so we prepend it ourselves.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE_DIR = Path(os.environ.get("FASTEMBED_CACHE_PATH", Path.home() / ".cache" / "fastembed"))


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        from fastembed import TextEmbedding  # keep the import lazy for fast test collection

        self.model = TextEmbedding(model_name, cache_dir=str(CACHE_DIR))
        self.dim = len(self._encode(["probe"])[0])
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for v in self.model.embed(texts, batch_size=getattr(self, "batch_size", 32)):
            norm = float((v * v).sum()) ** 0.5 or 1.0  # unit length: cosine == dot product
            out.append((v / norm).tolist())
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]
