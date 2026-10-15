"""Map [n] markers in generated answers to passages and format them for the API."""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import BaseModel

from rail_rag.ingest import load_manifest
from rail_rag.vectorstore import SearchHit

MARKER = re.compile(r"\[(\d+)\]")
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"


class Citation(BaseModel):
    n: int  # the [n] marker used in the answer
    passage_id: str
    doc_id: str
    document: str
    section: str
    heading: str
    passage: str
    url: str
    score: float


@lru_cache
def _celex_by_doc() -> dict[str, str]:
    return {d.id: d.celex for d in load_manifest()}


def renumber(answer: str, context: list[SearchHit]) -> tuple[str, list[SearchHit]]:
    """Keep only markers that point at real passages and renumber them 1..m by first use.

    Returns the rewritten answer and the cited passages, where citations[k-1] is marker [k].
    """
    order: list[int] = []
    for m in MARKER.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= len(context) and i not in order:
            order.append(i)
    new_number = {old: new for new, old in enumerate(order, start=1)}

    def sub(m: re.Match) -> str:
        i = int(m.group(1))
        return f"[{new_number[i]}]" if i in new_number else ""

    text = re.sub(r"\s+([.,;:])", r"\1", MARKER.sub(sub, answer)).strip()
    return text, [context[i - 1] for i in order]


def format_citations(cited: list[SearchHit]) -> list[Citation]:
    celex = _celex_by_doc()
    return [
        Citation(
            n=n,
            passage_id=h.passage.id,
            doc_id=h.passage.doc_id,
            document=h.passage.doc_short,
            section=h.passage.section,
            heading=h.passage.heading,
            passage=h.passage.text,
            url=EURLEX_URL.format(celex=celex.get(h.passage.doc_id, "")),
            score=round(h.score, 4),
        )
        for n, h in enumerate(cited, start=1)
    ]
