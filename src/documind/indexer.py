"""Build and persist the search index from a folder of documents.

This is the offline step: load -> chunk -> embed -> persist FAISS + chunk store.
Kept separate from query-time code so ingestion can run as its own job.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import Settings, get_settings

MANIFEST_FILE = "manifest.json"


def _write_manifest(documents, storage_dir: Path) -> None:
    """Persist a small summary of what is indexed (for display and APIs)."""
    counts = Counter(d.metadata.get("source", "unknown") for d in documents)
    manifest = {
        "total_chunks": len(documents),
        "sources": dict(sorted(counts.items())),
    }
    (storage_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_manifest(storage_dir: str | Path) -> dict | None:
    """Return the index manifest (source -> chunk count), or None if absent."""
    path = Path(storage_dir) / MANIFEST_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


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
    _write_manifest(documents, settings.storage_path)
    return len(documents)
