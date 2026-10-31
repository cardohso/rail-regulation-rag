"""Corpus integrity: answer only from passages whose text matches the official corpus.

Every passage comes from an EUR-Lex text parsed by ``ingest``. A passage added to the data folder
or to the vector store by anyone else (a planted "amendment", say) reads like law, and no prompt
rule can tell it apart from the real thing. So the code carries a pin per official passage, a
SHA-256 of its id, heading and text, in ``corpus.sha256`` next to this module, and retrieval drops
any hit without a pin. The pins live in the package, not in the data folder, so changing the
corpus means changing reviewed code.

Usage:
    uv run python -m rail_rag.integrity --write   # re-pin after a deliberate re-ingest
    uv run python -m rail_rag.integrity --check   # exit 1 if data/processed and the pins differ
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from rail_rag import paths
from rail_rag.chunking import Passage, load_passages
from rail_rag.vectorstore import SearchHit, VectorStore

PIN_FILE = Path(__file__).with_name("corpus.sha256")
log = logging.getLogger(__name__)


def passage_digest(p: Passage) -> str:
    return hashlib.sha256(json.dumps([p.id, p.heading, p.text]).encode("utf-8")).hexdigest()


def write_pins(passages: list[Passage], path: Path = PIN_FILE) -> int:
    digests = sorted({passage_digest(p) for p in passages})
    path.write_text("\n".join(digests) + "\n", encoding="utf-8")
    return len(digests)


def load_pins(path: Path = PIN_FILE) -> frozenset[str]:
    pins = frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    pins -= {""}
    if not pins:
        raise ValueError(f"{path} has no pins; refusing to answer from an unverified corpus")
    return pins


class PinnedStore:
    """A vector store that returns only pinned passages. Writes go straight through."""

    def __init__(self, store: VectorStore, pins: frozenset[str]):
        self.store, self.pins = store, pins

    def search(self, vector: list[float], k: int = 5, text: str | None = None) -> list[SearchHit]:
        hits = self.store.search(vector, k=k, text=text)
        kept = [h for h in hits if passage_digest(h.passage) in self.pins]
        if len(kept) < len(hits):
            dropped = [h.passage.id for h in hits if passage_digest(h.passage) not in self.pins]
            log.warning("dropped %d unpinned passage(s): %s", len(dropped), dropped)
        return kept

    def __getattr__(self, name):  # reset, upsert, count, ...
        return getattr(self.store, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="pin the passages in data/processed")
    mode.add_argument("--check", action="store_true", help="compare data/processed with the pins")
    args = ap.parse_args()

    passages = load_passages(paths.PROCESSED_DIR)
    if args.write:
        print(f"{write_pins(passages)} passages pinned -> {PIN_FILE}")
        return
    pins = load_pins()
    unpinned = [p.id for p in passages if passage_digest(p) not in pins]
    missing = len(pins - {passage_digest(p) for p in passages})
    print(f"{len(passages)} passages, {len(unpinned)} unpinned, {missing} pins with no passage")
    for pid in unpinned[:20]:
        print(f"  unpinned: {pid}")
    raise SystemExit(1 if unpinned or missing else 0)


if __name__ == "__main__":
    main()
