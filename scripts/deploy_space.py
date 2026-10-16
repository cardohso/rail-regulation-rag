"""Deploy the app to a Hugging Face Space (Docker SDK, free CPU hardware).

Usage:
    uv run --group deploy python scripts/deploy_space.py --space USER/rail-regulation-rag
    uv run --group deploy python scripts/deploy_space.py --space USER/rail-regulation-rag --secrets

Needs a write token as HF_TOKEN in .env or the environment. The Space is created private;
visibility is changed separately, on purpose. --secrets copies the runtime keys from .env into
Space secrets (values are sent straight to Hugging Face and never printed).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
# Only what the Docker build needs; never .env, tests, eval caches or raw downloads.
FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "data/sources.yaml"]
DIRS = ["src", "data/processed"]
SECRETS = ["GEMINI_API_KEY", "AZURE_SEARCH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]

SPACE_README = """---
title: EU Rail Regulation Assistant
emoji: 🚆
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cited Q&A over EU rail TSIs, eval-gated in CI
---

Question answering over EU railway interoperability regulations (the Interoperability Directive
and six TSIs) with passage-level citations and refusal when the sources don't cover a question.

Source code, evaluation results and design notes: https://github.com/cardohso/rail-regulation-rag

Free Spaces sleep when idle; the first request after a pause can take about 30 seconds.
Not legal advice. Source texts: EUR-Lex, © European Union, 1998–2026.
"""


def stage(tmp: Path) -> None:
    for f in FILES:
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, tmp / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / "README.md").write_text(SPACE_README, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--space", required=True, help="USER/NAME of the Space")
    ap.add_argument("--secrets", action="store_true", help="copy runtime keys from .env")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for the build")
    args = ap.parse_args()

    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    token = env.get("HF_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN is not set (.env or environment). Create a write token on huggingface.co."
        )
    api = HfApi(token=token)
    print(f"logged in as {api.whoami()['name']}")

    url = api.create_repo(
        args.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True
    )
    print(f"space: {url}")

    if args.secrets:
        for key in SECRETS:
            if env.get(key):
                api.add_space_secret(args.space, key, env[key])
                print(f"secret set: {key}")
        azure = bool(env.get("AZURE_SEARCH_API_KEY") and env.get("AZURE_SEARCH_ENDPOINT"))
        variables = {
            "VECTOR_STORE": "azure" if azure else "qdrant",
            "AZURE_SEARCH_ENDPOINT": env.get("AZURE_SEARCH_ENDPOINT", ""),
            "AZURE_SEARCH_INDEX": env.get("AZURE_SEARCH_INDEX", "rail-tsi"),
            "LANGFUSE_HOST": env.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "LANGFUSE_ENVIRONMENT": "hf-space",
        }
        for key, value in variables.items():
            if value:
                api.add_space_variable(args.space, key, value)
                print(f"variable set: {key}={value}")

    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        info = api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=tmp,
            commit_message="Deploy from rail-regulation-rag",
            delete_patterns=["src/**", "data/**"],  # remove files deleted in the source repo
        )
    print(f"uploaded: {info.commit_url}")

    if args.no_wait:
        return 0
    last = None
    for _ in range(120):  # up to ~20 minutes
        stage_name = api.get_space_runtime(args.space).stage
        if stage_name != last:
            print(f"runtime: {stage_name}")
            last = stage_name
        if stage_name == "RUNNING":
            host = args.space.replace("/", "-").replace("_", "-").lower()
            print(f"live at https://{host}.hf.space")
            return 0
        if stage_name in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}:
            print("see the build/runtime logs on the Space page")
            return 1
        time.sleep(10)
    print("still building; check the Space page")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
