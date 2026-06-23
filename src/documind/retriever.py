"""Hybrid retrieval + cross-encoder reranking.

Two retrieval signals are combined for robustness:

* **Dense (FAISS)** captures semantic similarity ("car" ~ "automobile").
* **Sparse (BM25)** captures exact keyword / rare-term matches (IDs, acronyms).

Their rankings are fused with Reciprocal Rank Fusion (RRF), then a cross-encoder
reranker re-scores the surviving candidates by reading each (query, chunk) pair
together. This hybrid + rerank design consistently beats single-vector search,
especially on enterprise jargon.

Implemented directly on top of ``rank_bm25`` and ``sentence-transformers`` (no
dependency on the ``langchain`` meta-package), which keeps the import graph small
and portable across Python versions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document

from .config import Settings, get_settings

DOCSTORE_FILE = "chunks.json"
RRF_K = 60  # standard Reciprocal Rank Fusion smoothing constant
_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _doc_key(doc: Document, fallback: int) -> str:
    return doc.metadata.get("citation_id") or doc.metadata.get("source") or str(fallback)


class HybridRetriever:
    """Dense + sparse retrieval fused with RRF, then cross-encoder reranked.

    Exposes :meth:`invoke` returning a list of :class:`Document`, matching the
    minimal retriever interface the rest of the pipeline depends on. Heavy models
    (BM25 index, cross-encoder) are built lazily on first use.
    """

    def __init__(
        self,
        documents: list[Document],
        vectorstore,
        settings: Settings | None = None,
    ) -> None:
        self.documents = documents
        self.vectorstore = vectorstore
        self.settings = settings or get_settings()
        self._bm25 = None
        self._reranker = None

    # --- lazy components -------------------------------------------------
    def _ensure_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            corpus = [_tokenize(d.page_content) for d in self.documents]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    def _ensure_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.settings.reranker_model)
        return self._reranker

    # --- retrieval signals ----------------------------------------------
    def _dense(self, query: str, k: int) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def _sparse(self, query: str, k: int) -> list[Document]:
        bm25 = self._ensure_bm25()
        scores = bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top]

    # --- fusion + rerank -------------------------------------------------
    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[Document]],
    ) -> list[Document]:
        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = _doc_key(doc, rank)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
                by_key.setdefault(key, doc)
        return sorted(by_key.values(), key=lambda d: scores[_doc_key(d, 0)], reverse=True)

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        reranker = self._ensure_reranker()
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        scored = zip(docs, scores, strict=False)
        ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ordered[: self.settings.rerank_top_n]]

    # --- public API ------------------------------------------------------
    def invoke(self, query: str) -> list[Document]:
        k = self.settings.retrieval_top_k
        dense = self._dense(query, k)
        sparse = self._sparse(query, k)
        fused = self._reciprocal_rank_fusion([dense, sparse])
        return self._rerank(query, fused[:k])


def build_hybrid_retriever(
    documents: list[Document],
    vectorstore,
    settings: Settings | None = None,
) -> HybridRetriever:
    """Compose dense + sparse retrieval, fuse, then rerank to the final top-N."""
    return HybridRetriever(documents, vectorstore, settings or get_settings())


def load_hybrid_retriever(settings: Settings | None = None) -> HybridRetriever:
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
