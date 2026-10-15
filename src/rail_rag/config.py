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

    # Retrieval
    vector_store: Literal["qdrant", "azure"] = "qdrant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    qdrant_path: Path = paths.DATA_DIR / "index" / "qdrant"
    azure_search_endpoint: str = ""
    azure_search_api_key: SecretStr = SecretStr("")
    azure_search_index: str = "rail-tsi"


@lru_cache
def get_settings() -> Settings:
    return Settings()
