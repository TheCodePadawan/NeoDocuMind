"""FastAPI service exposing the RAG pipeline over HTTP.

Endpoints:
    GET  /health      liveness probe
    POST /ask         {"question": "..."} -> grounded answer + sources

The pipeline is loaded once at startup and reused across requests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import RAGPipeline

_state: dict[str, RAGPipeline | None] = {"pipeline": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["pipeline"] = RAGPipeline.from_storage(get_settings())
    except FileNotFoundError:
        # Allow the service to boot even before ingestion; /ask will report it.
        _state["pipeline"] = None
    yield
    _state["pipeline"] = None


app = FastAPI(
    title="DocuMind RAG API",
    description="Ask grounded questions over your document collection.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is the refund policy?"])


class Source(BaseModel):
    citation_id: str | None = None
    source: str | None = None
    chunk: int | None = None
    preview: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "index_loaded": _state["pipeline"] is not None}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    pipeline = _state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Run `python -m scripts.ingest_sample` first.",
        )
    result = pipeline.answer(request.question)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
    )
