"""HTTP API: POST /ask, GET /health, and the single-page UI at /.

Run locally: uv run uvicorn rail_rag.api:app --port 7860
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rail_rag import __version__
from rail_rag.citations import Citation
from rail_rag.config import get_settings
from rail_rag.pipeline import Pipeline
from rail_rag.providers import LLMUnavailable
from rail_rag.ratelimit import RateLimiter

UI = Path(__file__).parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    trace_id: str


def _client_id(request: Request) -> str:
    # Behind the Hugging Face proxy the caller's address is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


def create_app(pipeline: Pipeline | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    settings = get_settings()
    limiter = limiter or RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_per_day)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or Pipeline.from_settings(settings)
        yield
        app.state.pipeline.tracer.flush()  # send buffered traces before shutdown

    app = FastAPI(title="rail-regulation-rag", version=__version__, lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(UI)

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "version": __version__, **request.app.state.pipeline.health()}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        # Sync endpoint: FastAPI runs it in a worker thread, so blocking LLM calls are fine.
        rejected = limiter.check(_client_id(request))
        if rejected:
            raise HTTPException(429, rejected)
        try:
            result = request.app.state.pipeline.ask(body.question.strip())
        except LLMUnavailable as exc:
            raise HTTPException(
                503, "The language model is rate-limited; try again shortly."
            ) from exc
        return AskResponse(
            answer=result.answer,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            citations=result.citations,
            trace_id=result.trace_id,
        )

    return app


app = create_app()
