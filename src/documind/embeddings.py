"""Embedding model factory.

Defaults to a small, high-quality local model (BAAI/bge-small-en-v1.5) so the
whole retrieval stack runs for free with no API key. Swappable via config.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return a local sentence-transformers embedding model.

    Imported lazily so importing :mod:`documind` does not require torch.
    """
    settings = settings or get_settings()

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
