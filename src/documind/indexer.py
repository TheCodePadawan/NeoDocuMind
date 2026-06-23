"""Build and persist the search index from a folder of documents.

This is the offline step: load -> chunk -> embed -> persist FAISS + chunk store.
Kept separate from query-time code so ingestion can run as its own job.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings, get_settings


def build_index(source_dir: str | Path, settings: Settings | None = None) -> int:
    """Ingest every document in ``source_dir`` and persist the index.

    Returns the number of chunks indexed.
    """
    settings = settings or get_settings()

    from .embeddings import get_embeddings
    from .ingest import load_and_chunk_directory
    from .retriever import save_documents
    from .vectorstore import build_vectorstore, save_vectorstore

    documents = load_and_chunk_directory(
        source_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not documents:
        raise ValueError(f"No supported documents found in {source_dir!r}.")

    embeddings = get_embeddings(settings)
    vectorstore = build_vectorstore(documents, embeddings)

    save_vectorstore(vectorstore, settings.storage_path)
    save_documents(documents, settings.storage_path)
    return len(documents)
