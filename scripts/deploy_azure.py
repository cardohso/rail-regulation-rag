"""Deploy the tested container image to Azure Container Apps (consumption plan, scale to zero).

Usage:
    uv run --group deploy python scripts/deploy_azure.py \
        --image ghcr.io/OWNER/rail-regulation-rag:sha-XXXXXXX

Prerequisites: `az login` (the script runs the Azure CLI through uvx, nothing installed system-wide)
and a public image on ghcr.io (built and smoke-tested by .github/workflows/docker.yml).

Cost control: min replicas 0, max 1, 0.5 vCPU / 1 GiB, and no Log Analytics workspace, so a
low-traffic demo stays inside the Container Apps monthly free grant. Runtime keys are read from
.env and stored as Container App secrets; values are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SECRETS = {  # env var -> Container App secret name
    "GEMINI_API_KEY": "gemini-api-key",
    "AZURE_SEARCH_API_KEY": "azure-search-api-key",
    "LANGFUSE_PUBLIC_KEY": "langfuse-public-key",
    "LANGFUSE_SECRET_KEY": "langfuse-secret-key",
}
AZ = ["uvx", "--from", "azure-cli", "--with", "pip", "az"]
ENV = {**os.environ, "AZURE_CORE_COLLECT_TELEMETRY": "0", "PYTHONWARNINGS": "ignore"}


def az(*args: str, capture: bool = False) -> str:
    """Run an az command. Arguments may contain secrets, so they are never echoed."""
    proc = subprocess.run([*AZ, *args], env=ENV, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        # Show az's stderr (no secret values in it), never the command line (which has secrets).
        sys.exit(f"az {args[0]} {args[1] if len(args) > 1 else ''} failed:\n{proc.stderr[-2000:]}")
    return proc.stdout if capture else ""


def exists(*args: str) -> bool:
    proc = subprocess.run([*AZ, *args], env=ENV, capture_output=True, text=True)
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--group", default="rail-rag")
    ap.add_argument("--location", default="switzerlandnorth")
    ap.add_argument("--environment", default="rail-rag-env")
    ap.add_argument("--app", default="rail-rag")
    args = ap.parse_args()

    env = dotenv_values(ROOT / ".env")
    missing = [k for k in ("GEMINI_API_KEY",) if not env.get(k)]
    if missing:
        sys.exit(f"missing in .env: {missing}")
    azure_search = bool(env.get("AZURE_SEARCH_ENDPOINT") and env.get("AZURE_SEARCH_API_KEY"))

    if not exists("containerapp", "env", "show", "-g", args.group, "-n", args.environment):
        print(f"creating environment {args.environment} in {args.location} (no log workspace)…")
        az(
            "containerapp", "env", "create", "-g", args.group, "-n", args.environment,
            "-l", args.location, "--logs-destination", "none", "-o", "none",
        )  # fmt: skip

    secrets = [f"{name}={env[key]}" for key, name in SECRETS.items() if env.get(key)]
    env_vars = [f"{key}=secretref:{name}" for key, name in SECRETS.items() if env.get(key)]
    env_vars += [
        f"VECTOR_STORE={'azure' if azure_search else 'qdrant'}",
        f"AZURE_SEARCH_ENDPOINT={env.get('AZURE_SEARCH_ENDPOINT', '')}",
        f"AZURE_SEARCH_INDEX={env.get('AZURE_SEARCH_INDEX') or 'rail-tsi'}",
        f"LANGFUSE_HOST={env.get('LANGFUSE_HOST') or 'https://cloud.langfuse.com'}",
        "LANGFUSE_ENVIRONMENT=azure-container-apps",
    ]

    if exists("containerapp", "show", "-g", args.group, "-n", args.app):
        print(f"updating app {args.app} → {args.image}")
        az("containerapp", "secret", "set", "-g", args.group, "-n", args.app,
           "--secrets", *secrets, "-o", "none")  # fmt: skip
        az("containerapp", "update", "-g", args.group, "-n", args.app, "--image", args.image,
           "--set-env-vars", *env_vars, "-o", "none")  # fmt: skip
    else:
        print(f"creating app {args.app} from {args.image}")
        az(
            "containerapp", "create", "-g", args.group, "-n", args.app,
            "--environment", args.environment, "--image", args.image,
            "--target-port", "7860", "--ingress", "external",
            "--min-replicas", "0", "--max-replicas", "1", "--cpu", "0.5", "--memory", "1.0Gi",
            "--secrets", *secrets, "--env-vars", *env_vars, "-o", "none",
        )  # fmt: skip

    info = json.loads(
        az(
            "containerapp",
            "show",
            "-g",
            args.group,
            "-n",
            args.app,
            "--query",
            "{fqdn: properties.configuration.ingress.fqdn, "
            "revision: properties.latestRevisionName, "
            "image: properties.template.containers[0].image}",
            "-o",
            "json",
            capture=True,
        )  # fmt: skip
    )
    print(f"revision {info['revision']} running {info['image']}")
    print(f"live at https://{info['fqdn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
