"""Build the search index from the bundled sample documents.

Usage:
    python -m scripts.ingest_sample
    python -m scripts.ingest_sample --source path/to/your/docs
"""

from __future__ import annotations

import argparse

from . import _bootstrap  # noqa: F401  (adds src/ to sys.path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into NeoDocuMind.")
    parser.add_argument(
        "--source",
        default="data/sample_docs",
        help="Folder of .pdf/.txt/.md documents to index (default: data/sample_docs).",
    )
    args = parser.parse_args()

    from documind.config import get_settings
    from documind.indexer import build_index

    settings = get_settings()
    print(f"Ingesting documents from: {args.source}")
    print(f"Embedding model: {settings.embedding_model} (local, no API key needed)")
    n_chunks = build_index(args.source, settings)
    print(f"Indexed {n_chunks} chunks -> {settings.storage_path.resolve()}")
    print("Done. Launch the UI with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
