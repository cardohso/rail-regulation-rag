"""Runtime configuration, read from environment variables and the local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from rail_rag import paths


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=paths.REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLMs: generator answers questions, judge scores answers in the eval (different families)
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    generator_provider: str = "gemini"
    generator_model: str = ""  # empty = provider default (see providers.DEFAULT_MODELS)
    judge_provider: str = "groq"
    judge_model: str = ""

    # Retrieval
    vector_store: Literal["qdrant", "azure"] = "qdrant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    qdrant_path: Path = paths.DATA_DIR / "index" / "qdrant"
    azure_search_endpoint: str = ""
    azure_search_api_key: SecretStr = SecretStr("")
    azure_search_index: str = "rail-tsi"
    top_k: int = 8
    min_score: float = 0.60

    # Tracing (optional): Langfuse Cloud Hobby tier
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "local"

    def api_key(self, provider: str) -> str:
        secret = getattr(self, f"{provider}_api_key", None)
        return secret.get_secret_value() if secret else ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
