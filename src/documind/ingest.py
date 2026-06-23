"""Document loading and chunking.

Turns raw files (.pdf, .txt, .md) into clean, overlapping text chunks with rich
metadata. Chunking quality is one of the biggest levers on RAG accuracy, so the
splitter is configurable and every chunk keeps a stable citation id.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    # Imported lazily so the package imports without pypdf installed.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: str | Path) -> str:
    """Read a single file into raw text based on its extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_file(path)
    if suffix in SUPPORTED_SUFFIXES:
        return _read_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix!r} ({path.name})")


def discover_files(directory: str | Path) -> list[Path]:
    """Recursively find all supported documents under a directory."""
    directory = Path(directory)
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create a recursive splitter that prefers natural boundaries (paras, lines)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split raw text into LangChain Documents with citation metadata.

    Each chunk carries ``source`` and ``chunk`` metadata plus a ``citation_id``
    of the form ``filename#3`` that the answer generator uses to cite sources.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    pieces = splitter.split_text(text)
    documents: list[Document] = []
    for index, piece in enumerate(pieces):
        cleaned = piece.strip()
        if not cleaned:
            continue
        documents.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source": source,
                    "chunk": index,
                    "citation_id": f"{source}#{index}",
                },
            )
        )
    return documents


def load_and_chunk_directory(
    directory: str | Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Load every supported file in a directory and return chunked Documents."""
    documents: list[Document] = []
    for file_path in discover_files(directory):
        text = load_document(file_path)
        if not text.strip():
            continue
        documents.extend(
            chunk_text(
                text,
                source=file_path.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return documents
