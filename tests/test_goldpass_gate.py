"""The goldpass retrieval gate's inputs: the converted golden set and the records script."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from rail_rag.chunking import Passage
from rail_rag.vectorstore import SearchHit

GATE = Path(__file__).parents[1] / "eval" / "goldpass"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, GATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden_rows() -> list[dict]:
    lines = (GATE.parent / "golden_set.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def test_committed_goldpass_golden_set_is_in_sync_with_the_source():
    make_golden = load("make_golden")
    expected = make_golden.render(make_golden.convert(golden_rows()))
    assert (GATE / "golden.jsonl").read_text(encoding="utf-8") == expected, (
        "eval/goldpass/golden.jsonl is stale: run `uv run python eval/goldpass/make_golden.py`"
    )


def test_conversion_keeps_answerable_questions_and_the_wrong_document_tag():
    items = load("make_golden").convert(golden_rows())
    assert len(items) == sum(r["type"] != "unanswerable" for r in golden_rows())
    assert all(i["expected"] == "answer" and i["gold_passage_ids"] for i in items)
    assert sum("wrong-document" in i.get("tags", []) for i in items) == 3


@dataclass
class FakeStore:
    calls: list

    def search(self, vector, k=5, text=None):
        self.calls.append((vector, k, text))
        passage = Passage("doc:1#1", "doc", "Doc", "1", "Heading", "text")
        return [SearchHit(passage, 0.91234)]


class FakeEmbedder:
    def embed_query(self, text):
        return [float(len(text))]


def test_records_use_the_saved_query_and_hybrid_adds_the_users_words():
    records_for = load("retrieval_records").records_for
    items = [{"id": "q1", "question": "What is the limit?"}]
    queries = {"q1": "noise limit EMU"}

    store = FakeStore([])
    dense = records_for(items, queries, FakeEmbedder(), store, k=8, hybrid=False)
    assert store.calls == [([15.0], 8, None)]  # the rewritten query, dense only

    store = FakeStore([])
    records_for(items, queries, FakeEmbedder(), store, k=8, hybrid=True)
    assert store.calls[0][2] == "noise limit EMU What is the limit?"  # rewrite + user's words

    assert dense == [
        {
            "id": "q1",
            "answer": "",
            "contexts": [{"id": "doc:1#1", "doc_id": "doc", "score": 0.9123}],
        }
    ]
