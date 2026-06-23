"""LLM-as-judge scoring for generated answers.

A separate, capable model grades each answer on two axes that keyword overlap
cannot capture:

* **Faithfulness**  - is every claim supported by the retrieved context? This is
  the anti-hallucination signal and the metric enterprises care about most.
* **Relevance**     - does the answer actually address the user's question?

Both are scored 1-5. The judge is asked to return strict JSON so scores can be
parsed deterministically; parsing is defensive because LLMs occasionally wrap
JSON in prose or code fences.
"""

from __future__ import annotations

import json
import re

JUDGE_SYSTEM = (
    "You are a strict evaluator of question-answering systems. "
    "You will be given a QUESTION, the CONTEXT passages that were retrieved, and "
    "an ANSWER produced from them. Score the ANSWER on two axes from 1 to 5:\n"
    "- faithfulness: 5 = every claim is fully supported by the CONTEXT; "
    "1 = the answer contradicts or invents information not in the CONTEXT.\n"
    "- relevance: 5 = directly and completely answers the QUESTION; "
    "1 = off-topic or non-responsive.\n"
    'Respond with ONLY a JSON object: '
    '{"faithfulness": <int>, "relevance": <int>, "reason": "<short>"}'
)

JUDGE_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Score the answer now as JSON."""


def build_judge_messages(
    question: str, answer: str, contexts: list[str]
) -> list[tuple[str, str]]:
    context = "\n\n".join(f"- {c}" for c in contexts)
    user = JUDGE_TEMPLATE.format(question=question, context=context, answer=answer)
    return [("system", JUDGE_SYSTEM), ("human", user)]


def _coerce_score(value) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return min(5, max(1, score))


def parse_judge_response(text: str) -> dict:
    """Extract {faithfulness, relevance, reason} from a model response.

    Tolerates code fences and surrounding prose. Returns None scores if the
    response cannot be parsed, so callers can exclude it from averages.
    """
    payload = None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        return {"faithfulness": None, "relevance": None, "reason": "unparseable"}

    return {
        "faithfulness": _coerce_score(payload.get("faithfulness")),
        "relevance": _coerce_score(payload.get("relevance")),
        "reason": str(payload.get("reason", ""))[:300],
    }


def judge_answer(llm, question: str, answer: str, contexts: list[str]) -> dict:
    """Run the judge model and return parsed scores."""
    messages = build_judge_messages(question, answer, contexts)
    response = llm.invoke(messages)
    text = getattr(response, "content", str(response))
    return parse_judge_response(text)
