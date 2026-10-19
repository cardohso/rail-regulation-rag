"""Retrieval-only records for the goldpass gate: no LLM, no API keys, deterministic.

For each golden question this runs the pipeline's own retrieval step (same embedder, same store,
same k, dense or hybrid as configured) with the query the pipeline last rewrote it to, taken from
a saved eval run (eval/results/latest.json). Fixing the query means a change in the score can only
come from retrieval: chunking, the index, the embedding model, hybrid search or k. A change to
the rewrite prompt is not covered here; the live eval gate (eval.yml) covers that.

Writes goldpass records: `contexts` is the raw top-k hits, so goldpass's recall@k, precision@k and
MRR are retrieval metrics, measured before grading and the cosine floor.

    uv run python eval/goldpass/retrieval_records.py --out eval/goldpass/records.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rail_rag import paths
from rail_rag.config import get_settings
from rail_rag.embeddings import LocalEmbedder
from rail_rag.graph import keyword_text

HERE = Path(__file__).parent


def records_for(items: list[dict], queries: dict[str, str], embedder, store, k: int, hybrid: bool):
    """goldpass records, one per golden item, in golden-set order."""
    records = []
    for item in items:
        query = queries[item["id"]]
        text = keyword_text({"question": item["question"], "query": query}) if hybrid else None
        hits = store.search(embedder.embed_query(query), k=k, text=text)
        records.append(
            {
                "id": item["id"],
                "answer": "",  # no generation here
                "contexts": [
                    {"id": h.passage.id, "doc_id": h.passage.doc_id, "score": round(h.score, 4)}
                    for h in hits
                ],
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    ap.add_argument(
        "--queries-from",
        type=Path,
        default=paths.REPO_ROOT / "eval/results/latest.json",
        help="eval results whose saved rewritten queries are reused",
    )
    ap.add_argument("--out", type=Path, default=HERE / "records.json")
    args = ap.parse_args(argv)

    from rail_rag.vectorstore import make_store

    s = get_settings()
    items = [
        json.loads(line) for line in args.golden.read_text(encoding="utf-8").splitlines() if line
    ]
    saved = json.loads(args.queries_from.read_text(encoding="utf-8"))["answers"]
    queries = {r["id"]: r["query"] for r in saved}
    missing = [i["id"] for i in items if i["id"] not in queries]
    if missing:
        raise SystemExit(f"no saved rewritten query for {missing}: re-run the eval and commit it")

    records = records_for(
        items,
        queries,
        LocalEmbedder(s.embedding_model),
        make_store(s),
        s.top_k,
        s.search_mode == "hybrid",
    )
    payload = {
        "system": f"rail-rag retrieval ({s.search_mode}, k={s.top_k}, {s.embedding_model})",
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
