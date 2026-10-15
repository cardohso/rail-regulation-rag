"""Embed all passages and (re)build the configured vector store.

Usage:
    uv run python -m rail_rag.index                 # store from VECTOR_STORE (default: qdrant)
    uv run python -m rail_rag.index --store azure
"""

import argparse
import time

from rail_rag.chunking import load_passages
from rail_rag.config import get_settings
from rail_rag.embeddings import SentenceTransformerEmbedder
from rail_rag.vectorstore import make_store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--store", choices=["qdrant", "azure"], help="override VECTOR_STORE")
    args = ap.parse_args()

    settings = get_settings()
    if args.store:
        settings = settings.model_copy(update={"vector_store": args.store})

    passages = load_passages()
    t0 = time.perf_counter()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vectors = embedder.embed_documents([p.embed_text for p in passages])
    t1 = time.perf_counter()

    store = make_store(settings)
    store.reset(embedder.dim)
    store.upsert(passages, vectors)
    t2 = time.perf_counter()
    print(
        f"{settings.vector_store}: {store.count()} passages indexed "
        f"(embed {t1 - t0:.0f}s, upload {t2 - t1:.0f}s, dim {embedder.dim})"
    )


if __name__ == "__main__":
    main()
