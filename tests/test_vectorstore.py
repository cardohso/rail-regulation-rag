import base64
from types import SimpleNamespace

import pytest

from rail_rag.chunking import Passage
from rail_rag.config import Settings
from rail_rag.vectorstore import AzureSearchStore, QdrantStore, make_store
from tests.fakes import HashEmbedder

PASSAGES = [
    Passage("tsi-noi:Annex §4.2.1#1", "tsi-noi", "NOI TSI", "Annex §4.2.1", "4.2.1. Stationary",
            "stationary noise limit for electric locomotives"),
    Passage("tsi-srt:Annex §4.2.1#1", "tsi-srt", "SRT TSI", "Annex §4.2.1", "4.2.1. Fire",
            "fire safety in railway tunnels and evacuation"),
    Passage("tsi-prm:Annex §4.2.1#1", "tsi-prm", "PRM TSI", "Annex §4.2.1", "4.2.1. Parking",
            "parking facilities for persons with reduced mobility"),
]  # fmt: skip


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def qdrant(embedder):
    store = QdrantStore(":memory:")
    store.reset(embedder.dim)
    store.upsert(PASSAGES, embedder.embed_documents([p.embed_text for p in PASSAGES]))
    return store


def test_qdrant_returns_most_similar_passage_first(qdrant, embedder):
    hits = qdrant.search(embedder.embed_query("fire safety in tunnels"), k=3)
    assert hits[0].passage == PASSAGES[1]
    assert hits[0].score > hits[1].score >= hits[2].score


def test_qdrant_round_trips_full_passage_and_counts(qdrant, embedder):
    assert qdrant.count() == 3
    [hit] = qdrant.search(embedder.embed_query("parking reduced mobility"), k=1)
    assert hit.passage == PASSAGES[2]
    assert 0.0 < hit.score <= 1.0


def test_qdrant_reset_empties_collection(qdrant, embedder):
    qdrant.reset(embedder.dim)
    assert qdrant.count() == 0


def test_qdrant_upsert_is_idempotent(qdrant, embedder):
    qdrant.upsert(PASSAGES, embedder.embed_documents([p.embed_text for p in PASSAGES]))
    assert qdrant.count() == 3


class FakeSearchClient:
    def __init__(self, results=(), reject=()):
        self.batches: list[list[dict]] = []
        self.results = list(results)
        self.reject = set(reject)
        self.last_query: dict = {}

    def merge_or_upload_documents(self, docs):
        self.batches.append(docs)
        return [
            SimpleNamespace(key=d["key"], succeeded=d["key"] not in self.reject,
                            error_message="too large")
            for d in docs
        ]  # fmt: skip

    def search(self, **kwargs):
        self.last_query = kwargs
        return iter(self.results)

    def get_document_count(self):
        return sum(len(b) for b in self.batches)


class FakeIndexClient:
    def __init__(self):
        self.created = None

    def delete_index(self, name):
        from azure.core.exceptions import ResourceNotFoundError

        raise ResourceNotFoundError("not there")

    def create_index(self, index):
        self.created = index


def azure(search_client=None, index_client=None):
    return AzureSearchStore(
        "https://example.search.windows.net",
        "dummy",
        search_client=search_client or FakeSearchClient(),
        index_client=index_client or FakeIndexClient(),
    )


def test_azure_reset_creates_cosine_vector_index_with_given_dim():
    idx = FakeIndexClient()
    azure(index_client=idx).reset(dim=384)
    vector = next(f for f in idx.created.fields if f.name == "vector")
    assert vector.vector_search_dimensions == 384
    assert idx.created.vector_search.algorithms[0].parameters.metric == "cosine"


def test_azure_keys_are_url_safe_and_decode_to_passage_id():
    client = FakeSearchClient()
    azure(client).upsert(PASSAGES, [[0.0] * 4] * 3)
    for doc, p in zip(client.batches[0], PASSAGES, strict=True):
        assert set(doc["key"]) <= set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_="
        )
        assert base64.urlsafe_b64decode(doc["key"]).decode() == p.id
        assert doc["text"] == p.text


def test_azure_upsert_raises_when_documents_are_rejected():
    from rail_rag.vectorstore import _azure_key

    client = FakeSearchClient(reject={_azure_key(PASSAGES[1].id)})
    with pytest.raises(RuntimeError, match="1 documents rejected"):
        azure(client).upsert(PASSAGES, [[0.0]] * 3)


def test_azure_uploads_in_batches_of_500():
    client = FakeSearchClient()
    many = [Passage(f"d:s#{i}", "d", "D", "s", "h", "t") for i in range(1201)]
    azure(client).upsert(many, [[0.0]] * 1201)
    assert [len(b) for b in client.batches] == [500, 500, 201]


def test_azure_converts_search_score_back_to_cosine():
    cos = 0.8
    row = PASSAGES[0].to_dict() | {"@search.score": 1 / (1 + (1 - cos))}
    client = FakeSearchClient(results=[row])
    [hit] = azure(client).search([0.1, 0.2], k=1)
    assert hit.passage == PASSAGES[0]
    assert hit.score == pytest.approx(cos)
    assert client.last_query["vector_queries"][0].k_nearest_neighbors == 1


def test_make_store_requires_azure_endpoint():
    with pytest.raises(ValueError, match="AZURE_SEARCH_ENDPOINT"):
        make_store(Settings(_env_file=None, vector_store="azure"))


def test_make_store_defaults_to_local_qdrant(tmp_path):
    store = make_store(Settings(_env_file=None, qdrant_path=tmp_path / "q"))
    assert isinstance(store, QdrantStore)
