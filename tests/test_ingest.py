"""Unit tests for document chunking (pure-python, no model downloads)."""

from documind.ingest import chunk_text, discover_files


def test_chunk_text_assigns_citation_metadata():
    text = "Alpha paragraph.\n\n" + ("Beta sentence. " * 200)
    chunks = chunk_text(text, source="handbook.md", chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    for i, doc in enumerate(chunks):
        assert doc.metadata["source"] == "handbook.md"
        assert doc.metadata["chunk"] == i
        assert doc.metadata["citation_id"] == f"handbook.md#{i}"
        assert doc.page_content.strip() != ""


def test_chunk_text_respects_chunk_size():
    text = "word " * 1000
    chunks = chunk_text(text, source="doc.txt", chunk_size=200, chunk_overlap=0)
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_chunk_text_skips_empty_input():
    assert chunk_text("   \n\n  ", source="empty.md") == []


def test_discover_files_finds_sample_docs(tmp_path):
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    found = discover_files(tmp_path)
    names = {p.name for p in found}
    assert names == {"a.md", "b.txt"}
