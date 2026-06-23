"""End-to-end RAG pipeline: retrieve -> build grounded prompt -> generate.

The prompt forces the model to answer **only** from retrieved context and to
cite every claim with the chunk's ``citation_id``. That grounding + citation
discipline is what makes the system trustworthy enough for real business use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from .config import Settings, get_settings

SYSTEM_PROMPT = (
    "You are NeoDocuMind, a meticulous enterprise document assistant. "
    "Answer the user's question using ONLY the numbered context passages provided. "
    "Every factual sentence must end with a citation in square brackets that "
    "reuses the passage's citation id, e.g. [handbook.pdf#3]. "
    "If the answer is not contained in the context, reply exactly: "
    "\"I could not find that in the provided documents.\" "
    "Be concise, accurate, and never invent sources."
)

ANSWER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer (with citations):"""


def format_context(documents: list[Document]) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        citation = doc.metadata.get("citation_id", doc.metadata.get("source", f"doc{i}"))
        blocks.append(f"[{citation}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_messages(question: str, documents: list[Document]) -> list[tuple[str, str]]:
    """Build the (role, content) message list sent to the chat model."""
    context = format_context(documents)
    user = ANSWER_TEMPLATE.format(context=context, question=question)
    return [("system", SYSTEM_PROMPT), ("human", user)]


@dataclass
class RAGResult:
    """Structured answer plus the evidence used to produce it."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)


class RAGPipeline:
    """Orchestrates retrieval + generation. Construct via :meth:`from_storage`."""

    def __init__(self, retriever, llm, settings: Settings | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or get_settings()

    @classmethod
    def from_storage(cls, settings: Settings | None = None) -> RAGPipeline:
        """Load the persisted index from disk and assemble a ready pipeline."""
        settings = settings or get_settings()

        from .embeddings import get_embeddings
        from .llm import get_llm
        from .retriever import build_hybrid_retriever, load_documents
        from .vectorstore import load_vectorstore

        embeddings = get_embeddings(settings)
        vectorstore = load_vectorstore(settings.storage_path, embeddings)
        documents = load_documents(settings.storage_path)
        retriever = build_hybrid_retriever(documents, vectorstore, settings)
        llm = get_llm(settings)
        return cls(retriever, llm, settings)

    def retrieve(self, question: str) -> list[Document]:
        """Return the reranked top-N chunks for a question."""
        return self.retriever.invoke(question)

    def answer(self, question: str) -> RAGResult:
        """Retrieve evidence and generate a grounded, cited answer."""
        documents = self.retrieve(question)
        if not documents:
            return RAGResult(
                question=question,
                answer="I could not find that in the provided documents.",
                sources=[],
            )
        messages = build_messages(question, documents)
        response = self.llm.invoke(messages)
        answer_text = getattr(response, "content", str(response))

        sources = [
            {
                "citation_id": d.metadata.get("citation_id"),
                "source": d.metadata.get("source"),
                "chunk": d.metadata.get("chunk"),
                "preview": d.page_content[:240].strip(),
            }
            for d in documents
        ]
        return RAGResult(question=question, answer=answer_text, sources=sources)
