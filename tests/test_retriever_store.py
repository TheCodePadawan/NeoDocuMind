"""Unit tests for chunk persistence used by the BM25 retriever."""

from langchain_core.documents import Document

from documind.retriever import load_documents, save_documents


def test_document_store_round_trip(tmp_path):
    docs = [
        Document(page_content="alpha", metadata={"source": "a.md", "chunk": 0,
                                                  "citation_id": "a.md#0"}),
        Document(page_content="beta", metadata={"source": "b.md", "chunk": 1,
                                                 "citation_id": "b.md#1"}),
    ]
    save_documents(docs, tmp_path)
    loaded = load_documents(tmp_path)

    assert len(loaded) == 2
    assert loaded[0].page_content == "alpha"
    assert loaded[1].metadata["citation_id"] == "b.md#1"
