"""Retrieval-only comparison: dense vs hybrid, and store vs store. No LLM calls.

Reuses the rewritten queries saved in an eval results file, so both modes see exactly the
queries the full pipeline saw, and the comparison is deterministic and free.

Usage:
    uv run python -m rail_rag.evaluation.retrieval                          # local Qdrant
    uv run python -m rail_rag.evaluation.retrieval --stores qdrant,azure    # plus store parity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rail_rag import paths
from rail_rag.config import get_settings
from rail_rag.embeddings import LocalEmbedder
from rail_rag.evaluation.metrics import ANSWERABLE, context_recall
from rail_rag.evaluation.runner import load_golden
from rail_rag.graph import keyword_text
from rail_rag.vectorstore import make_store

EVAL_DIR = paths.REPO_ROOT / "eval"
MODES = ("dense", "hybrid")


def gold_rank(hit_ids: list[str], gold: set[str]) -> int | None:
    """1-based rank of the first gold passage, None if absent."""
    return next((i for i, pid in enumerate(hit_ids, start=1) if pid in gold), None)


def compare(items: list[dict], queries: dict[str, str], search, k: int, floor: float) -> dict:
    """search(item, query, mode) -> hits. Per-item ranks and scores, mean recall@k per mode."""
    rows = []
    for item in items:
        query = queries[item["id"]]
        row: dict = {"id": item["id"], "type": item["type"], "query": query}
        gold = set(item["supporting_passages"])
        for mode in MODES:
            hits = search(item, query, mode)
            ids = [h.passage.id for h in hits]
            best = max((h.score for h in hits), default=0.0)
            row[mode] = {
                "ids": ids,
                "best_cosine": round(best, 4),
                "passes_floor": best >= floor,
            }
            if item["type"] in ANSWERABLE:
                row[mode] |= {
                    "recall_at_k": context_recall(ids, gold),
                    "gold_rank": gold_rank(ids, gold),
                }
        rows.append(row)
    ans = [r for r in rows if r["type"] in ANSWERABLE]
    summary = {
        mode: {
            "retrieval_recall_at_k": round(sum(r[mode]["recall_at_k"] for r in ans) / len(ans), 4),
            "gold_in_top_k": sum(r[mode]["gold_rank"] is not None for r in ans),
            "pass_floor": sum(r[mode]["passes_floor"] for r in rows),
        }
        for mode in MODES
    }
    summary["floor_decisions_changed"] = [
        r["id"] for r in rows if r["dense"]["passes_floor"] != r["hybrid"]["passes_floor"]
    ]
    return {"k": k, "min_score": floor, "n_answerable": len(ans), "summary": summary, "items": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--answers", type=Path, default=EVAL_DIR / "results" / "latest.json",
                    help="eval results whose saved rewritten queries are reused")  # fmt: skip
    ap.add_argument("--golden", type=Path, default=EVAL_DIR / "golden_set.jsonl")
    ap.add_argument("--stores", default="qdrant", help="comma-separated: qdrant,azure")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results" / "retrieval.json")
    args = ap.parse_args(argv)

    s = get_settings()
    items = load_golden(args.golden)
    queries = {r["id"]: r["query"] for r in json.loads(args.answers.read_text())["answers"]}
    embedder = LocalEmbedder(s.embedding_model)
    vectors = {q: embedder.embed_query(q) for q in queries.values()}

    result: dict = {"answers": str(args.answers.relative_to(paths.REPO_ROOT)), "stores": {}}
    for name in [x.strip() for x in args.stores.split(",") if x.strip()]:
        store = make_store(s.model_copy(update={"vector_store": name}))

        def search(item, query, mode, store=store):  # the same texts the graph would send
            state = {"question": item["question"], "query": query}
            text = keyword_text(state) if mode == "hybrid" else None
            return store.search(vectors[query], k=s.top_k, text=text)

        result["stores"][name] = compare(items, queries, search, s.top_k, s.min_score)
        summ = result["stores"][name]["summary"]
        print(f"\n{name} (k={s.top_k}, floor {s.min_score})")
        for mode in MODES:
            print(f"  {mode:6}  {summ[mode]}")
        print(f"  floor decisions changed: {summ['floor_decisions_changed'] or 'none'}")
        for r in result["stores"][name]["items"]:
            if r["type"] in ANSWERABLE and r["dense"]["gold_rank"] != r["hybrid"]["gold_rank"]:
                print(f"  {r['id']}: gold rank {r['dense']['gold_rank']} -> "
                      f"{r['hybrid']['gold_rank']}")  # fmt: skip

    names = list(result["stores"])
    if len(names) == 2:  # store parity: same top-k passages, in the same order?
        a, b = (result["stores"][n]["items"] for n in names)
        parity = {}
        for mode in MODES:
            same_order = sum(x[mode]["ids"] == y[mode]["ids"] for x, y in zip(a, b, strict=True))
            overlap = [
                len(set(x[mode]["ids"]) & set(y[mode]["ids"])) / max(len(x[mode]["ids"]), 1)
                for x, y in zip(a, b, strict=True)
            ]
            parity[mode] = {
                "identical_top_k": f"{same_order}/{len(a)}",
                "mean_overlap": round(sum(overlap) / len(overlap), 4),
            }
        result["parity"] = parity
        print(f"\nparity {names[0]} vs {names[1]}: {parity}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
