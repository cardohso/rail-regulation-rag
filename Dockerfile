# Single container: FastAPI API + UI, local embeddings, and a baked-in Qdrant index as fallback.
# Runs as a non-root user with uid 1000 (what Hugging Face Spaces expects).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app/rail-rag

# Dependencies first, so code changes don't reinstall torch (CPU wheels on Linux, see pyproject).
COPY --chown=app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app src ./src
COPY --chown=app data/sources.yaml ./data/sources.yaml
COPY --chown=app data/processed ./data/processed
RUN uv sync --frozen --no-dev

# Download the embedding model and build the local index at build time: the container needs no
# network for retrieval, and switching VECTOR_STORE between azure and qdrant needs no rebuild.
RUN uv run --no-sync python -m rail_rag.index --store qdrant
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"]

CMD ["uv", "run", "--no-sync", "uvicorn", "rail_rag.api:app", "--host", "0.0.0.0", "--port", "7860"]
