"""Swappable vector stores. Every store returns cosine similarity so thresholds are portable.

Two retrieval modes, same interface:
- dense: nearest neighbours of the query embedding;
- hybrid (pass ``text``): BM25 keyword search and dense search, fused with Reciprocal Rank
  Fusion. RRF scores are rank-based (about 0.03), not similarities, so the refusal floor can't
  use them. Hybrid hits are ranked by RRF but scored with the cosine similarity of their own
  vector, computed here from the vector the store returns with the hit (one query, no extra
  round trip).
"""

from __future__ import annotations

import base64
import contextlib
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rail_rag.chunking import Passage
from rail_rag.config import Settings
from rail_rag.embeddings import SparseEncoder

HYBRID_CANDIDATES = 50  # per retriever, before fusion (Azure's text subquery default is 50)
RRF_K = 60  # Azure's fixed RRF constant; Qdrant's local default is 2, so it is set explicitly


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float  # cosine similarity, higher is better (also in hybrid mode)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class VectorStore(Protocol):
    def reset(self, dim: int) -> None:
        """Drop and recreate the collection/index."""

    def upsert(self, passages: Sequence[Passage], vectors: Sequence[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int = 5, text: str | None = None) -> list[SearchHit]:
        """Dense search, or hybrid (BM25 + dense, RRF) when ``text`` is given. Hits come in rank
        order; scores are cosine similarities either way."""

    def count(self) -> int: ...


class QdrantStore:
    """Qdrant in local (in-process) mode: no server, persisted to a folder or kept in memory.
    Each point has a dense vector and a BM25 sparse vector (IDF applied by Qdrant at query time).
    """

    DENSE, SPARSE = "dense", "bm25"

    def __init__(
        self,
        path: Path | str = ":memory:",
        collection: str = "rail-tsi",
        sparse_encoder: SparseEncoder | None = None,
    ):
        from qdrant_client import QdrantClient

        self.collection = collection
        self._sparse = sparse_encoder
        if str(path) == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

    @property
    def sparse(self) -> SparseEncoder:
        if self._sparse is None:  # lazy: dense-only use never loads the BM25 tokenizer
            from rail_rag.embeddings import BM25Encoder

            self._sparse = BM25Encoder()
        return self._sparse

    def reset(self, dim: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                self.DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                self.SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def upsert(self, passages: Sequence[Passage], vectors: Sequence[list[float]]) -> None:
        from qdrant_client import models

        sparse = self.sparse.embed_documents([p.embed_text for p in passages])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, p.id)),
                vector={
                    self.DENSE: v,
                    self.SPARSE: models.SparseVector(indices=sv[0], values=sv[1]),
                },
                payload=p.to_dict(),
            )
            for p, v, sv in zip(passages, vectors, sparse, strict=True)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256])

    def search(self, vector: list[float], k: int = 5, text: str | None = None) -> list[SearchHit]:
        from qdrant_client import models

        if text is None:
            res = self.client.query_points(
                self.collection, query=vector, using=self.DENSE, limit=k, with_payload=True
            )
            return [SearchHit(Passage.from_dict(pt.payload), pt.score) for pt in res.points]
        indices, values = self.sparse.embed_query(text)
        res = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=vector, using=self.DENSE, limit=HYBRID_CANDIDATES),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=self.SPARSE,
                    limit=HYBRID_CANDIDATES,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
            limit=k,
            with_payload=True,
            with_vectors=[self.DENSE],
        )
        return [
            SearchHit(Passage.from_dict(pt.payload), cosine(vector, pt.vector[self.DENSE]))
            for pt in res.points
        ]

    def count(self) -> int:
        return self.client.count(self.collection).count


# Characters with operator meaning in Azure's simple or full query syntax. Stripped so a query
# like "on-board" or "(PRM)" is searched as plain words, as Qdrant's BM25 tokenizer does.
_AZURE_OPERATORS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _azure_key(passage_id: str) -> str:
    # Document keys may only contain letters, digits, '_', '-' and '='.
    return base64.urlsafe_b64encode(passage_id.encode()).decode()


class AzureSearchStore:
    """Azure AI Search (Free tier is enough: ~1,600 passages x 384 dims is a few MB)."""

    FIELDS = ("id", "doc_id", "doc_short", "section", "heading", "text")

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index: str = "rail-tsi",
        search_client: Any = None,
        index_client: Any = None,
    ):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        cred = AzureKeyCredential(api_key)
        self.index = index
        self.search_client = search_client or SearchClient(endpoint, index, cred)
        self.index_client = index_client or SearchIndexClient(endpoint, cred)

    def reset(self, dim: int) -> None:
        from azure.core.exceptions import ResourceNotFoundError
        from azure.search.documents.indexes import models as m

        with contextlib.suppress(ResourceNotFoundError):
            self.index_client.delete_index(self.index)
        text = m.SearchFieldDataType.STRING
        fields = [
            m.SimpleField(name="key", type=text, key=True),
            m.SimpleField(name="id", type=text),
            m.SimpleField(name="doc_id", type=text, filterable=True),
            m.SimpleField(name="doc_short", type=text),
            m.SimpleField(name="section", type=text),
            m.SearchableField(name="heading", type=text),
            m.SearchableField(name="text", type=text),
            m.SearchField(
                name="vector",
                type=m.SearchFieldDataType.Collection(m.SearchFieldDataType.SINGLE),
                searchable=True,
                retrievable=True,  # hybrid search recomputes cosine from the returned vector
                vector_search_dimensions=dim,
                vector_search_profile_name="hnsw-cosine",
            ),
        ]
        vector_search = m.VectorSearch(
            algorithms=[
                m.HnswAlgorithmConfiguration(
                    name="hnsw", parameters=m.HnswParameters(metric="cosine")
                )
            ],
            profiles=[
                m.VectorSearchProfile(name="hnsw-cosine", algorithm_configuration_name="hnsw")
            ],
        )
        self.index_client.create_index(
            m.SearchIndex(name=self.index, fields=fields, vector_search=vector_search)
        )

    def upsert(self, passages: Sequence[Passage], vectors: Sequence[list[float]]) -> None:
        docs = [
            {"key": _azure_key(p.id), **p.to_dict(), "vector": v}
            for p, v in zip(passages, vectors, strict=True)
        ]
        for i in range(0, len(docs), 500):  # keep request bodies well under the 16 MB limit
            results = self.search_client.merge_or_upload_documents(docs[i : i + 500])
            failed = [r for r in results if not r.succeeded]
            if failed:  # Azure reports per-document failures without raising
                raise RuntimeError(
                    f"{len(failed)} documents rejected by Azure, e.g. "
                    f"{failed[0].key}: {failed[0].error_message}"
                )

    def search(self, vector: list[float], k: int = 5, text: str | None = None) -> list[SearchHit]:
        from azure.search.documents.models import VectorizedQuery

        if text is None:
            results = self.search_client.search(
                search_text=None,
                vector_queries=[
                    VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="vector")
                ],
                select=list(self.FIELDS),
                top=k,
            )
            # For the cosine metric Azure reports score = 1 / (1 + (1 - cos)); invert it.
            return [
                SearchHit(Passage.from_dict(r), 2.0 - 1.0 / r["@search.score"]) for r in results
            ]
        # Hybrid: BM25 over the searchable fields (heading, text) + vector, fused by Azure with
        # RRF. @search.score is then the RRF score, so cosine comes from the returned vector.
        results = self.search_client.search(
            search_text=_AZURE_OPERATORS.sub(" ", text),
            vector_queries=[
                VectorizedQuery(
                    vector=vector, k_nearest_neighbors=HYBRID_CANDIDATES, fields="vector"
                )
            ],
            select=[*self.FIELDS, "vector"],
            top=k,
        )
        return [SearchHit(Passage.from_dict(r), cosine(vector, r["vector"])) for r in results]

    def count(self) -> int:
        return self.search_client.get_document_count()


def make_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "azure":
        if not settings.azure_search_endpoint:
            raise ValueError("VECTOR_STORE=azure but AZURE_SEARCH_ENDPOINT is not set")
        return AzureSearchStore(
            settings.azure_search_endpoint,
            settings.azure_search_api_key.get_secret_value(),
            settings.azure_search_index,
        )
    return QdrantStore(settings.qdrant_path)
