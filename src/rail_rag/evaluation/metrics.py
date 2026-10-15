"""Judge-free metrics, aggregation and the pass/fail gate."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

ANSWERABLE = ("answerable", "wrong_document")


def context_precision(context_ids: list[str], gold: set[str]) -> float:
    """RAGAS context precision (mean of precision@k at each relevant rank), with relevance
    taken from the golden set's passage ids instead of an LLM verdict."""
    found, total = 0, 0.0
    for k, pid in enumerate(context_ids, start=1):
        if pid in gold:
            found += 1
            total += found / k
    return total / found if found else 0.0


def context_recall(context_ids: list[str], gold: set[str]) -> float:
    """Share of the golden supporting passages that reached the generator."""
    return len(gold & set(context_ids)) / len(gold) if gold else 0.0


def grounded(cited_docs: list[str], expected_docs: list[str]) -> bool:
    """An answer is grounded if it cites at least one passage and every citation comes from a
    document the golden set expects (catches 'right answer, wrong document')."""
    return bool(cited_docs) and set(cited_docs) <= set(expected_docs)


def score_item(item: dict, record: dict) -> dict:
    """Judge-free scores for one golden item and the pipeline's answer to it."""
    gold = set(item["supporting_passages"])
    context_ids = [c["passage_id"] for c in record["contexts"]]
    hit_ids = [h["passage_id"] for h in record["hits"]]
    cited_docs = [c["doc_id"] for c in record["citations"]]
    scores: dict = {"id": item["id"], "type": item["type"], "refused": record["refused"]}
    if item["type"] in ANSWERABLE:
        scores |= {
            "context_precision": context_precision(context_ids, gold),
            "context_recall": context_recall(context_ids, gold),
            "retrieval_recall_at_k": context_recall(hit_ids, gold),
            "grounded": None if record["refused"] else grounded(cited_docs, item["expected_docs"]),
            "cited_docs": sorted(set(cited_docs)),
            "expected_docs": item["expected_docs"],
        }
    return scores


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(mean(values), 4) if values else None


def aggregate(scored: list[dict]) -> dict:
    """Corpus-level metrics. Judge metrics are averaged over answered answerable items only;
    false refusals are measured separately by answered_rate."""
    ans = [s for s in scored if s["type"] in ANSWERABLE]
    answered = [s for s in ans if not s["refused"]]
    unans = [s for s in scored if s["type"] == "unanswerable"]
    wrong_doc = [s for s in answered if s["type"] == "wrong_document"]
    return {
        "faithfulness": _mean(s.get("faithfulness") for s in answered),
        "answer_relevancy": _mean(s.get("answer_relevancy") for s in answered),
        "context_precision": _mean(s["context_precision"] for s in ans),
        "context_recall": _mean(s["context_recall"] for s in ans),
        "grounding": _mean(s["grounded"] for s in answered),
        "answered_rate": _mean(not s["refused"] for s in ans),
        "refusal_rate": _mean(s["refused"] for s in unans),
        # diagnostics (no threshold)
        "retrieval_recall_at_k": _mean(s["retrieval_recall_at_k"] for s in ans),
        "wrong_document_grounding": _mean(s["grounded"] for s in wrong_doc),
        "n_answerable": len(ans),
        "n_answered": len(answered),
        "n_unanswerable": len(unans),
    }


def per_document(scored: list[dict]) -> list[dict]:
    """Grounding broken down by the document each answerable question targets."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        if s["type"] in ANSWERABLE:
            groups[s["expected_docs"][0]].append(s)
    rows = []
    for doc, items in sorted(groups.items()):
        answered = [s for s in items if not s["refused"]]
        rows.append(
            {
                "doc_id": doc,
                "questions": len(items),
                "answered": len(answered),
                "grounding": _mean(s["grounded"] for s in answered),
                "cited_other_docs": sorted({d for s in answered for d in s["cited_docs"]} - {doc}),
            }
        )
    return rows


def gate(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    """Fail if any thresholded metric is missing or below its minimum."""
    failures = []
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: no value (nothing to score)")
        elif value < rule["min"]:
            failures.append(f"{name}: {value:.3f} < {rule['min']:.3f}")
    return not failures, failures
