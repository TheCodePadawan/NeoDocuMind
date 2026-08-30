"""NeoDocuMind demo UI.

A clean chat interface over the RAG pipeline with expandable source citations,
so reviewers can see *why* the assistant answered the way it did. Supports
uploading your own PDF / TXT / MD documents and chatting with them.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import streamlit as st

# Make the local `src/` package importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from documind.config import get_settings  # noqa: E402
from documind.indexer import build_index, read_manifest  # noqa: E402
from documind.llm import LLMConfigError  # noqa: E402
from documind.pipeline import RAGPipeline  # noqa: E402

SAMPLE_DOCS = ROOT / "data" / "sample_docs"
UPLOAD_DIR = ROOT / "data" / "uploads"
SUPPORTED_TYPES = ["pdf", "txt", "md", "markdown"]

st.set_page_config(page_title="NeoDocuMind", layout="wide")

# On Streamlit Community Cloud, configuration is provided via the Secrets UI.
# Mirror those secrets into environment variables so pydantic Settings picks
# them up (works locally with a .env file too).
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ[_key] = _value
except Exception:
    pass


@st.cache_resource(show_spinner="Loading models and index...")
def load_pipeline() -> RAGPipeline:
    settings = get_settings()
    try:
        return RAGPipeline.from_storage(settings)
    except FileNotFoundError:
        # First run (e.g. a fresh cloud deploy): build the index from the
        # bundled sample documents so the demo works out of the box.
        build_index(SAMPLE_DOCS, settings)
        return RAGPipeline.from_storage(settings)


def _rebuild_index(source_dir: Path, corpus_label: str | None) -> None:
    """Index a folder of documents and refresh the cached pipeline.

    ``corpus_label`` of None marks the default sample corpus.
    """
    try:
        with st.spinner("Indexing documents (chunking + embedding)... this can "
                        "take a moment for large files."):
            n_chunks = build_index(source_dir, get_settings())
    except ValueError as exc:
        st.error(str(exc))
        return
    load_pipeline.clear()
    if corpus_label:
        st.session_state["active_corpus"] = corpus_label
    else:
        st.session_state.pop("active_corpus", None)
    st.session_state["history"] = []
    target = corpus_label or "the bundled sample documents"
    st.success(f"Indexed {n_chunks} chunks. You can now ask questions about {target}.")
    st.rerun()


def _index_uploaded_files(files) -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        (UPLOAD_DIR / file.name).write_bytes(file.getbuffer())
    label = files[0].name if len(files) == 1 else f"{len(files)} uploaded files"
    _rebuild_index(UPLOAD_DIR, label)


def _reset_to_samples() -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    _rebuild_index(SAMPLE_DOCS, None)


def main() -> None:
    st.title("NeoDocuMind")
    st.caption(
        "Production-grade Retrieval-Augmented Generation: hybrid search, "
        "cross-encoder reranking, and grounded answers with citations."
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            "- **Hybrid retrieval**: dense (FAISS) + sparse (BM25)\n"
            "- **Reranking**: cross-encoder re-scores candidates\n"
            "- **Grounded answers**: every claim is cited\n"
            "- **Provider-agnostic LLM**: OpenAI / Groq / Ollama"
        )
        st.divider()
        st.subheader("Chat with your own documents")
        uploaded = st.file_uploader(
            "Upload PDF / TXT / MD files (e.g. a book, report, or handbook)",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
        )
        if uploaded and st.button("Index uploaded documents", type="primary"):
            _index_uploaded_files(uploaded)
        if UPLOAD_DIR.exists():
            if st.button("Reset to sample documents"):
                _reset_to_samples()

    try:
        pipeline = load_pipeline()
    except LLMConfigError:
        st.error(
            "No language-model provider is configured. Set `GROQ_API_KEY` "
            "(free at https://console.groq.com/keys) with `LLM_PROVIDER=groq`, "
            "or `OPENAI_API_KEY`, then reload."
        )
        return
    except FileNotFoundError:
        st.warning(
            "No documents found to index. Add files to `data/sample_docs/` or run "
            "`python -m scripts.ingest_sample --source <folder>`, then reload."
        )
        return

    with st.sidebar:
        _render_indexed_docs()

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                _render_sources(turn["sources"])

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and reasoning..."):
                result = pipeline.answer(question)
        except Exception as exc:
            message = (
                f"The language model request failed: {exc}\n\n"
                "If this mentions a missing model, set `GROQ_MODEL` in Streamlit "
                "Secrets to `openai/gpt-oss-120b` (Groq retired "
                "`llama-3.3-70b-versatile` on 16 Aug 2026) and reboot the app."
            )
            st.error(message)
            st.session_state.history.append(
                {"role": "assistant", "content": message, "sources": []}
            )
            return
        st.markdown(result.answer)
        _render_sources(result.sources)

    st.session_state.history.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


def _render_indexed_docs() -> None:
    """List the documents currently in the knowledge base (survives reloads)."""
    manifest = read_manifest(get_settings().storage_path)
    if not manifest or not manifest.get("sources"):
        return
    sources = manifest["sources"]
    total = manifest.get("total_chunks", sum(sources.values()))
    st.divider()
    st.markdown(f"**Knowledge base:** {len(sources)} file(s), {total} chunks")
    for name, count in sources.items():
        st.caption(f"- {name}  ({count} chunks)")


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**`{s.get('citation_id')}`**")
            st.caption(s.get("preview", ""))


if __name__ == "__main__":
    main()
