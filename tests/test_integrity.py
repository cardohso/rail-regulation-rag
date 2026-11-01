"""Corpus pins: retrieval drops passages that are not in the official corpus."""

import logging

import pytest

from rail_rag import integrity, paths
from rail_rag.chunking import Passage, load_passages
from rail_rag.integrity import PinnedStore, load_pins, passage_digest, write_pins
from rail_rag.vectorstore import SearchHit

REAL = Passage(
    "tsi-noi:Annex §4.2.1#1", "tsi-noi", "Noise TSI", "Annex §4.2.1", "Stationary noise", "real"
)
PLANTED = Passage(
    "tsi-noi:Annex §4.2.1 (2)#1", "tsi-noi", "Noise TSI", "Annex §4.2.1 (2)", "Amendment", "fake"
)


class ListStore:
    def __init__(self, hits):
        self.hits, self.upserted = hits, None

    def search(self, vector, k=5, text=None):
        return self.hits[:k]

    def upsert(self, passages, vectors):
        self.upserted = passages

    def count(self):
        return len(self.hits)


def test_digest_covers_id_heading_and_text():
    base = passage_digest(REAL)
    for field, value in [("id", "x#1"), ("heading", "Other"), ("text", "real.")]:
        changed = Passage(**(REAL.to_dict() | {field: value}))
        assert passage_digest(changed) != base


def test_pinned_store_drops_unpinned_hits_and_logs_them(caplog):
    store = PinnedStore(
        ListStore([SearchHit(PLANTED, 0.9), SearchHit(REAL, 0.8)]),
        frozenset({passage_digest(REAL)}),
    )
    with caplog.at_level(logging.WARNING, logger="rail_rag.integrity"):
        hits = store.search([0.0], k=8, text="noise")
    assert [h.passage.id for h in hits] == [REAL.id]
    assert PLANTED.id in caplog.text


def test_edited_text_under_a_real_id_is_dropped():
    edited = Passage(**(REAL.to_dict() | {"text": "real, amended"}))
    store = PinnedStore(ListStore([SearchHit(edited, 0.9)]), frozenset({passage_digest(REAL)}))
    assert store.search([0.0]) == []


def test_pinned_store_passes_other_calls_through():
    inner = ListStore([SearchHit(REAL, 0.8)])
    store = PinnedStore(inner, frozenset())
    store.upsert([REAL], [[0.0]])
    assert inner.upserted == [REAL]
    assert store.count() == 1


def test_write_then_load_round_trip(tmp_path):
    path = tmp_path / "pins"
    assert write_pins([REAL, REAL, PLANTED], path) == 2
    assert load_pins(path) == {passage_digest(REAL), passage_digest(PLANTED)}


def test_empty_pin_file_fails_closed(tmp_path):
    path = tmp_path / "pins"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no pins"):
        load_pins(path)


def test_committed_pins_match_the_committed_corpus():
    """Changing data/processed without re-pinning (or the other way round) fails here."""
    passages = load_passages(paths.REPO_ROOT / "data" / "processed")
    assert {passage_digest(p) for p in passages} == load_pins(integrity.PIN_FILE)


def test_pins_are_on_by_default():
    from rail_rag.config import Settings

    assert Settings(_env_file=None).corpus_pin is True
