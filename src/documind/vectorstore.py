"""FAISS-backed vector store with simple persistence.

FAISS is fast, dependency-light, and installs cleanly everywhere, which makes it
a great default. In production this layer is intentionally thin so it can be
swapped for a managed store (Qdrant, pgvector, Pinecone) without touching the
rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

INDEX_NAME = "faiss_index"


def build_vectorstore(documents: list[Document], embeddings: Embeddings):
    """Create an in-memory FAISS index from documents."""
    from langchain_community.vectorstores import FAISS

    if not documents:
        raise ValueError("Cannot build a vector store from zero documents.")
    return FAISS.from_documents(documents, embeddings)


def save_vectorstore(vectorstore, storage_dir: str | Path) -> None:
    """Persist the FAISS index to disk."""
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(storage_dir), index_name=INDEX_NAME)


def load_vectorstore(storage_dir: str | Path, embeddings: Embeddings):
    """Load a previously persisted FAISS index from disk."""
    from langchain_community.vectorstores import FAISS

    storage_dir = Path(storage_dir)
    index_file = storage_dir / f"{INDEX_NAME}.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_file}. Run ingestion first "
            "(python -m scripts.ingest_sample)."
        )
    return FAISS.load_local(
        str(storage_dir),
        embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
