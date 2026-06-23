"""DocuMind demo UI.

A clean chat interface over the RAG pipeline with expandable source citations,
so reviewers can see *why* the assistant answered the way it did.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make the local `src/` package importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from documind.config import get_settings  # noqa: E402
from documind.indexer import build_index  # noqa: E402
from documind.llm import LLMConfigError  # noqa: E402
from documind.pipeline import RAGPipeline  # noqa: E402

SAMPLE_DOCS = ROOT / "data" / "sample_docs"

st.set_page_config(page_title="NeoDocuMind", layout="wide")

# On Streamlit Community Cloud, configuration is provided via the Secrets UI.
# Mirror those secrets into environment variables so pydantic Settings picks
# them up (works locally with a .env file too).
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
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
        st.caption(
            "The search index over the bundled sample documents is built "
            "automatically on first launch, so you can start asking right away."
        )

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
        with st.spinner("Retrieving and reasoning..."):
            result = pipeline.answer(question)
        st.markdown(result.answer)
        _render_sources(result.sources)

    st.session_state.history.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**`{s.get('citation_id')}`**")
            st.caption(s.get("preview", ""))


if __name__ == "__main__":
    main()
