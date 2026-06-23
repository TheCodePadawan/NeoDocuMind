"""DocuMind demo UI.

A clean chat interface over the RAG pipeline with expandable source citations,
so reviewers can see *why* the assistant answered the way it did.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the local `src/` package importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from documind.config import get_settings  # noqa: E402
from documind.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def load_pipeline() -> RAGPipeline | None:
    try:
        return RAGPipeline.from_storage(get_settings())
    except FileNotFoundError:
        return None


def main() -> None:
    st.title("📄 DocuMind")
    st.caption(
        "Production-grade Retrieval-Augmented Generation — hybrid search, "
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
        st.markdown(
            "Build the index first:\n\n"
            "```bash\npython -m scripts.ingest_sample\n```"
        )

    pipeline = load_pipeline()
    if pipeline is None:
        st.warning(
            "No index found. Build it first by running "
            "`python -m scripts.ingest_sample` in your terminal, then refresh."
        )
        return

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                _render_sources(turn["sources"])

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            result = pipeline.answer(question)
        st.markdown(result.answer)
        _render_sources(result.sources)

    st.session_state.history.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**`{s.get('citation_id')}`**")
            st.caption(s.get("preview", ""))


if __name__ == "__main__":
    main()
