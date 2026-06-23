"""Hybrid retrieval + cross-encoder reranking.

Two retrieval signals are combined for robustness:

* **Dense (FAISS)** captures semantic similarity ("car" ~ "automobile").
* **Sparse (BM25)** captures exact keyword / rare-term matches (IDs, acronyms).

Their results are fused with a reciprocal-rank ensemble, then a cross-encoder
reranker re-scores the top candidates by reading each (query, chunk) pair
together. This hybrid + rerank design consistently beats single-vector search,
especially on enterprise jargon.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .config import Settings, get_settings

DOCSTORE_FILE = "chunks.json"


def save_documents(documents: list[Document], storage_dir: str | Path) -> None:
    """Persist chunk text + metadata so BM25 can be rebuilt at query time."""
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in documents
    ]
    (storage_dir / DOCSTORE_FILE).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def load_documents(storage_dir: str | Path) -> list[Document]:
    """Load persisted chunks back into LangChain Documents."""
    path = Path(storage_dir) / DOCSTORE_FILE
    if not path.exists():
        raise FileNotFoundError(f"No chunk store found at {path}. Run ingestion first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload
    ]


def build_bm25_retriever(documents: list[Document], k: int) -> BaseRetriever:
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def build_dense_retriever(vectorstore, k: int) -> BaseRetriever:
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_reranker(settings: Settings):
    """Cross-encoder reranker that reads (query, chunk) pairs jointly."""
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    return CrossEncoderReranker(model=model, top_n=settings.rerank_top_n)


def load_hybrid_retriever(settings: Settings | None = None) -> BaseRetriever:
    """Load the persisted index and assemble the hybrid retriever (no LLM needed).

    Useful for retrieval-only evaluation, which runs fully offline and free.
    """
    settings = settings or get_settings()

    from .embeddings import get_embeddings
    from .vectorstore import load_vectorstore

    embeddings = get_embeddings(settings)
    vectorstore = load_vectorstore(settings.storage_path, embeddings)
    documents = load_documents(settings.storage_path)
    return build_hybrid_retriever(documents, vectorstore, settings)


def build_hybrid_retriever(
    documents: list[Document],
    vectorstore,
    settings: Settings | None = None,
) -> BaseRetriever:
    """Compose dense + sparse retrieval, fuse, then rerank to the final top-N."""
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever

    settings = settings or get_settings()
    k = settings.retrieval_top_k

    dense = build_dense_retriever(vectorstore, k)
    sparse = build_bm25_retriever(documents, k)

    ensemble = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])

    reranker = build_reranker(settings)
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble
    )
