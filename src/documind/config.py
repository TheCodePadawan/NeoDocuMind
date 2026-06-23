"""Centralised, type-safe configuration loaded from environment variables / .env.

Using pydantic-settings means every knob is validated once, documented in one
place, and overridable per-environment without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Values come from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider selection ---
    llm_provider: str = Field(default="openai", description="openai | groq | ollama")

    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")

    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    ollama_model: str = Field(default="llama3.1")
    ollama_base_url: str = Field(default="http://localhost:11434")

    # --- Retrieval / embedding configuration ---
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    chunk_size: int = Field(default=800, ge=128, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    retrieval_top_k: int = Field(default=12, ge=1, le=100)
    rerank_top_n: int = Field(default=4, ge=1, le=50)

    # --- Persistence ---
    storage_dir: str = Field(default="storage")

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
