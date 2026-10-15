"""Swappable vector stores. Every store returns cosine similarity so thresholds are portable."""

from __future__ import annotations

import base64
import contextlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rail_rag.chunking import Passage
from rail_rag.config import Settings


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float  # cosine similarity, higher is better


class VectorStore(Protocol):
    def reset(self, dim: int) -> None:
        """Drop and recreate the collection/index."""

    def upsert(self, passages: Sequence[Passage], vectors: Sequence[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]: ...

    def count(self) -> int: ...


class QdrantStore:
    """Qdrant in local (in-process) mode: no server, persisted to a folder or kept in memory."""

    def __init__(self, path: Path | str = ":memory:", collection: str = "rail-tsi"):
        from qdrant_client import QdrantClient

        self.collection = collection
        if str(path) == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

    def reset(self, dim: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def upsert(self, passages: Sequence[Passage], vectors: Sequence[list[float]]) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, p.id)), vector=v, payload=p.to_dict()
            )
            for p, v in zip(passages, vectors, strict=True)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256])

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        res = self.client.query_points(self.collection, query=vector, limit=k, with_payload=True)
        return [SearchHit(Passage.from_dict(pt.payload), pt.score) for pt in res.points]

    def count(self) -> int:
        return self.client.count(self.collection).count


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
            self.search_client.merge_or_upload_documents(docs[i : i + 500])

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        from azure.search.documents.models import VectorizedQuery

        results = self.search_client.search(
            search_text=None,
            vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="vector")],
            select=list(self.FIELDS),
            top=k,
        )
        # For the cosine metric Azure reports score = 1 / (1 + (1 - cos)); invert it.
        return [SearchHit(Passage.from_dict(r), 2.0 - 1.0 / r["@search.score"]) for r in results]

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
