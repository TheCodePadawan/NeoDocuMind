"""Ask DocuMind a question from the command line.

Usage:
    python -m scripts.ask "What is the PTO carryover policy?"
"""

from __future__ import annotations

import argparse

from . import _bootstrap  # noqa: F401  (adds src/ to sys.path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the DocuMind index.")
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()

    from documind.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_storage()
    result = pipeline.answer(args.question)

    print("\n=== Answer ===")
    print(result.answer)
    print("\n=== Sources ===")
    for s in result.sources:
        print(f"  - {s['citation_id']}")


if __name__ == "__main__":
    main()
