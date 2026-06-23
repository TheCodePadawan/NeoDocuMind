"""Unit tests for prompt construction and source formatting."""

from langchain_core.documents import Document

from documind.pipeline import SYSTEM_PROMPT, build_messages, format_context


def _docs():
    return [
        Document(page_content="PTO carries over up to 5 days.",
                 metadata={"citation_id": "handbook.md#2", "source": "handbook.md"}),
        Document(page_content="Passwords rotate every 180 days.",
                 metadata={"citation_id": "security.md#1", "source": "security.md"}),
    ]


def test_format_context_includes_citation_ids():
    context = format_context(_docs())
    assert "[handbook.md#2]" in context
    assert "[security.md#1]" in context
    assert "PTO carries over" in context


def test_build_messages_structure():
    messages = build_messages("How much PTO carries over?", _docs())
    assert messages[0][0] == "system"
    assert messages[0][1] == SYSTEM_PROMPT
    assert messages[1][0] == "human"
    assert "How much PTO carries over?" in messages[1][1]
    assert "[handbook.md#2]" in messages[1][1]


def test_system_prompt_demands_grounding():
    assert "ONLY" in SYSTEM_PROMPT
    assert "citation" in SYSTEM_PROMPT.lower()
