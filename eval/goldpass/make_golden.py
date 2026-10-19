"""Convert eval/golden_set.jsonl to goldpass's format (https://github.com/cardohso/goldpass).

Only the answerable questions are kept: the goldpass retrieval gate (`retrieval_records.py`)
runs no LLM, so it cannot say anything about refusals or citations. tests/test_goldpass_gate.py
checks that the committed eval/goldpass/golden.jsonl is exactly this script's output.

    uv run python eval/goldpass/make_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE.parent / "golden_set.jsonl"
OUT = HERE / "golden.jsonl"


def convert(rows: list[dict]) -> list[dict]:
    items = []
    for r in rows:
        if r["type"] == "unanswerable":
            continue
        item = {"id": r["id"], "question": r["question"], "expected": "answer"}
        if r.get("reference_answer"):
            item["reference_answer"] = r["reference_answer"]
        item["gold_passage_ids"] = r["supporting_passages"]
        item["source_doc_ids"] = r["expected_docs"]
        if r["type"] == "wrong_document":
            item["tags"] = ["wrong-document"]
        if r.get("notes"):
            item["notes"] = r["notes"]
        items.append(item)
    return items


def render(items: list[dict]) -> str:
    return "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    OUT.write_text(render(convert(rows)), encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
