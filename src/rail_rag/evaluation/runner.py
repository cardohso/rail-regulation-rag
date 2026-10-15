"""Two stages: generate answers once, then score the same answers with one or more judges."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rail_rag.evaluation.judge import Judge
from rail_rag.evaluation.metrics import ANSWERABLE, score_item
from rail_rag.pipeline import Pipeline


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hit(h) -> dict:
    return {"passage_id": h.passage.id, "doc_id": h.passage.doc_id, "score": round(h.score, 4)}


def generate_answers(pipeline: Pipeline, items: list[dict], log=print) -> list[dict]:
    records = []
    for i, item in enumerate(items, start=1):
        t0 = time.perf_counter()
        result = pipeline.ask(item["question"])
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "refused": result.refused,
                "refusal_reason": result.refusal_reason,
                "query": result.query,
                "citations": [
                    {"passage_id": c.passage_id, "doc_id": c.doc_id} for c in result.citations
                ],
                "contexts": [_hit(h) | {"text": h.passage.embed_text} for h in result.contexts],
                "hits": [_hit(h) for h in result.hits],
                "trace_id": result.trace_id,
                "latency_s": round(time.perf_counter() - t0, 2),
            }
        )
        status = f"refused ({result.refusal_reason})" if result.refused else "answered"
        log(f"[{i:2}/{len(items)}] {item['id']} {status} in {records[-1]['latency_s']}s")
    return records


async def judge_answers(
    judge: Judge, items: list[dict], records: list[dict], log=print
) -> list[dict]:
    """Judge-free scores for every item, plus judge scores for answered answerable items.
    Calls run one at a time on purpose: free-tier rate limits, not latency, are the bottleneck."""
    by_id = {r["id"]: r for r in records}
    scored = []
    for item in items:
        record = by_id[item["id"]]
        scores = score_item(item, record)
        if item["type"] in ANSWERABLE and not record["refused"]:
            contexts = [c["text"] for c in record["contexts"]]
            scores["faithfulness"] = await judge.faithfulness(
                item["question"], record["answer"], contexts
            )
            scores["answer_relevancy"] = await judge.answer_relevancy(
                item["question"], record["answer"]
            )
            log(
                f"  {judge.name} {item['id']}: faithfulness={scores['faithfulness']:.2f} "
                f"relevancy={scores['answer_relevancy']:.2f}"
            )
        scored.append(scores)
    return scored
