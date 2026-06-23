"""Evaluation harness for NeoDocuMind.

Two layers of metrics:

1. **Retrieval quality (free, offline)**: for each labelled question we check
   whether the reranked results surface the correct source document.
     * Hit@N   : fraction of questions whose correct source is retrieved.
     * MRR      : mean reciprocal rank of the first correct source.

2. **Answer quality (optional, needs an LLM)**: with ``--with-llm`` the full
   pipeline generates answers and we measure keyword recall against the gold
   answer. Skipped automatically if no provider key is configured.

Usage:
    python -m eval.evaluate                # retrieval metrics only (free)
    python -m eval.evaluate --with-llm     # also score generated answers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def load_dataset() -> list[dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return data["examples"]


def reciprocal_rank(sources_in_order: list[str], expected: str) -> float:
    for rank, src in enumerate(sources_in_order, start=1):
        if src == expected:
            return 1.0 / rank
    return 0.0


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def evaluate(with_llm: bool = False) -> dict:
    from documind.config import get_settings
    from documind.retriever import load_hybrid_retriever

    settings = get_settings()
    examples = load_dataset()

    retriever = load_hybrid_retriever(settings)

    pipeline = None
    if with_llm:
        from documind.pipeline import RAGPipeline

        llm_settings = settings
        try:
            pipeline = RAGPipeline.from_storage(llm_settings)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] LLM unavailable ({exc}); scoring retrieval only.")
            pipeline = None

    rows = []
    hit, mrr_sum, recall_sum = 0, 0.0, 0.0
    for ex in examples:
        retrieved = retriever.invoke(ex["question"])
        sources = [d.metadata.get("source") for d in retrieved]
        expected = ex["expected_source"]
        rr = reciprocal_rank(sources, expected)
        is_hit = expected in sources
        hit += int(is_hit)
        mrr_sum += rr

        row = {
            "question": ex["question"],
            "expected_source": expected,
            "retrieved_sources": sources,
            "hit": is_hit,
            "reciprocal_rank": round(rr, 3),
        }

        if pipeline is not None:
            result = pipeline.answer(ex["question"])
            recall = keyword_recall(result.answer, ex.get("keywords", []))
            recall_sum += recall
            row["answer"] = result.answer
            row["keyword_recall"] = round(recall, 3)

        rows.append(row)

    n = len(examples)
    summary = {
        "num_questions": n,
        "hit_rate": round(hit / n, 3),
        "mrr": round(mrr_sum / n, 3),
        "rerank_top_n": settings.rerank_top_n,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
    }
    if pipeline is not None:
        summary["answer_keyword_recall"] = round(recall_sum / n, 3)
        summary["llm_provider"] = settings.llm_provider

    output = {"summary": summary, "details": rows}
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), "utf-8")

    print("\n=== NeoDocuMind Evaluation ===")
    for key, value in summary.items():
        print(f"{key:>22}: {value}")
    print(f"\nFull results written to {RESULTS_PATH}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeoDocuMind retrieval/answers.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also generate and score answers (requires a configured LLM provider).",
    )
    args = parser.parse_args()
    evaluate(with_llm=args.with_llm)


if __name__ == "__main__":
    main()
