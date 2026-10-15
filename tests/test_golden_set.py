import json
from pathlib import Path

from rail_rag.chunking import load_passages

GOLDEN = Path(__file__).parents[1] / "eval" / "golden_set.jsonl"


def items() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]


def test_golden_set_has_20_items_with_the_planned_mix():
    types = [it["type"] for it in items()]
    assert len(types) == 20
    assert types.count("unanswerable") >= 3 and types.count("wrong_document") >= 3


def test_every_supporting_passage_exists_in_the_corpus():
    ids = {p.id for p in load_passages()}
    for it in items():
        assert set(it["supporting_passages"]) <= ids, it["id"]


def test_unanswerable_items_have_no_support_and_answerable_items_do():
    for it in items():
        assert (it["type"] == "unanswerable") == (it["supporting_passages"] == []), it["id"]
