"""Markdown for the GitHub job summary and the README."""

from __future__ import annotations

GATED_ORDER = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "grounding",
    "answered_rate",
    "refusal_rate",
]
LABELS = {
    "faithfulness": "Faithfulness (judge)",
    "answer_relevancy": "Answer relevancy (judge)",
    "context_precision": "Context precision (gold ids)",
    "context_recall": "Context recall (gold ids)",
    "grounding": "Grounding: cites only expected docs",
    "answered_rate": "Answerable questions answered",
    "refusal_rate": "Unanswerable questions refused",
    "retrieval_recall_at_k": "Retrieval recall@k, before grading",
    "wrong_document_grounding": "Grounding on wrong-document items",
}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def render(result: dict) -> str:
    meta, primary = result["meta"], result["primary_judge"]
    metrics = result["judges"][primary]["metrics"]
    thresholds = result["thresholds"]
    status = "PASSED" if result["gate"]["passed"] else "FAILED"
    lines = [
        f"## Evaluation {status}",
        "",
        f"Generator `{meta['generator']}` · judge `{primary}` · {meta['n_items']} golden items · "
        f"{meta.get('search_mode', 'dense')} retrieval · "
        f"commit `{meta['commit']}` · {meta['timestamp']}",
        "",
        "| Metric | Value | Threshold | Status |",
        "|---|---|---|---|",
    ]
    for name in GATED_ORDER:
        rule = thresholds.get(name)
        value = metrics.get(name)
        ok = rule is None or (value is not None and value >= rule["min"])
        lines.append(
            f"| {LABELS[name]} | {_fmt(value)} | {'≥ ' + _fmt(rule['min']) if rule else '—'} | "
            f"{'✅' if ok else '❌'} |"
        )
    for name in ("retrieval_recall_at_k", "wrong_document_grounding"):
        lines.append(f"| {LABELS[name]} (diagnostic) | {_fmt(metrics.get(name))} | — | |")
    if result["gate"]["failures"]:
        lines += ["", "**Failures:** " + "; ".join(result["gate"]["failures"])]

    if len(result["judges"]) > 1:
        names = list(result["judges"])
        lines += [
            "",
            "### Same answers, different judges",
            "",
            "| Metric | " + " | ".join(f"`{n}`" for n in names) + " | Δ (2nd − 1st) |",
            "|---|" + "---|" * (len(names) + 1),
        ]
        for m in ("faithfulness", "answer_relevancy"):
            vals = [result["judges"][n]["metrics"].get(m) for n in names]
            delta = vals[1] - vals[0] if None not in vals[:2] else None
            sign = "+" if delta and delta > 0 else ""
            lines.append(
                f"| {LABELS[m]} | "
                + " | ".join(_fmt(v) for v in vals)
                + f" | {sign}{_fmt(delta)} |"
            )

    lines += [
        "",
        "### Grounding per source document",
        "",
        "| Document | Questions | Answered | Grounding | Also cited |",
        "|---|---|---|---|---|",
    ]
    for row in result["judges"][primary]["per_document"]:
        lines.append(
            f"| `{row['doc_id']}` | {row['questions']} | {row['answered']} | "
            f"{_fmt(row['grounding'])} | {', '.join(row['cited_other_docs']) or '—'} |"
        )

    lines += [
        "",
        "<details><summary>Per-item scores</summary>",
        "",
        "| Item | Type | Refused | Faithfulness | Relevancy | Ctx P | Ctx R | Grounded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in result["judges"][primary]["items"]:
        lines.append(
            f"| {s['id']} | {s['type']} | {_fmt(s['refused'])} | {_fmt(s.get('faithfulness'))} | "
            f"{_fmt(s.get('answer_relevancy'))} | {_fmt(s.get('context_precision'))} | "
            f"{_fmt(s.get('context_recall'))} | {_fmt(s.get('grounded'))} |"
        )
    lines += ["", "</details>", ""]
    return "\n".join(lines)
