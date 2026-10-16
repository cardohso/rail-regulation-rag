"""Run the golden set and gate on thresholds.

Usage:
    uv run --group eval python -m rail_rag.evaluation                 # generate + judge + gate
    uv run --group eval python -m rail_rag.evaluation --judges groq,gemini
    uv run --group eval python -m rail_rag.evaluation --answers eval/results/answers.json

Exit code 1 if any thresholded metric is below its minimum.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rail_rag import paths
from rail_rag.config import get_settings
from rail_rag.embeddings import LocalEmbedder
from rail_rag.evaluation.judge import RagasJudge
from rail_rag.evaluation.metrics import aggregate, gate, per_document
from rail_rag.evaluation.report import render
from rail_rag.evaluation.runner import generate_answers, judge_answers, load_golden
from rail_rag.pipeline import Pipeline

EVAL_DIR = paths.REPO_ROOT / "eval"


def _commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--judges", help="comma-separated judge providers; first one gates "
                    "(default: JUDGE_PROVIDER)")  # fmt: skip
    ap.add_argument("--answers", type=Path, help="reuse saved answers instead of generating")
    ap.add_argument("--golden", type=Path, default=EVAL_DIR / "golden_set.jsonl")
    ap.add_argument("--thresholds", type=Path, default=EVAL_DIR / "thresholds.yaml")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results" / "run.json")
    ap.add_argument("--summary", type=Path, help="append the markdown report here "
                    "(e.g. $GITHUB_STEP_SUMMARY)")  # fmt: skip
    ap.add_argument("--allow-same-family", action="store_true",
                    help="let the gating judge share the generator's provider")  # fmt: skip
    args = ap.parse_args(argv)

    s = get_settings()
    judges = [j.strip() for j in (args.judges or s.judge_provider).split(",") if j.strip()]
    if judges[0] == s.generator_provider and not args.allow_same_family:
        sys.exit(
            f"The gating judge ({judges[0]}) is the same provider as the generator. "
            "Self-preference bias would inflate scores; pick a different family or pass "
            "--allow-same-family."
        )
    items = load_golden(args.golden)
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))["metrics"]
    embedder = LocalEmbedder(s.embedding_model)

    if args.answers:
        answers = json.loads(args.answers.read_text(encoding="utf-8"))
    else:
        pipeline = Pipeline.from_settings(s)
        answers = {
            "generator": pipeline.llm.name,
            "records": generate_answers(pipeline, items),
        }
        pipeline.tracer.flush()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        answers_path = args.out.with_name(args.out.stem + ".answers.json")
        answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"answers saved to {answers_path}")

    judged = {}
    for provider in judges:
        model = s.judge_model if provider == s.judge_provider else None
        judge = RagasJudge(
            provider, s.api_key(provider), embedder, model, cache_dir=EVAL_DIR / ".cache" / provider
        )
        scored = asyncio.run(judge_answers(judge, items, answers["records"]))
        judged[judge.name] = {
            "metrics": aggregate(scored),
            "per_document": per_document(scored),
            "items": scored,
        }

    primary = next(iter(judged))
    passed, failures = gate(judged[primary]["metrics"], thresholds)
    result = {
        "meta": {
            "generator": answers["generator"],
            "commit": _commit(),
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "n_items": len(items),
            "vector_store": s.vector_store,
            "embedding_model": s.embedding_model,
            "top_k": s.top_k,
            "min_score": s.min_score,
        },
        "primary_judge": primary,
        "thresholds": thresholds,
        "gate": {"passed": passed, "failures": failures},
        "judges": judged,
        "answers": answers["records"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render(result)
    args.out.with_suffix(".md").write_text(report, encoding="utf-8")
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(report)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
