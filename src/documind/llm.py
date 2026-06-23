"""LLM factory - one switch selects the answer-generation backend.

The rest of the pipeline only depends on the LangChain ``BaseChatModel``
interface, so swapping providers (or going fully local with Ollama) never
touches retrieval, prompting, or the API.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Settings, get_settings


class LLMConfigError(RuntimeError):
    """Raised when the selected provider is misconfigured (e.g. missing key)."""


def get_llm(settings: Settings | None = None, *, temperature: float = 0.1) -> BaseChatModel:
    """Return a chat model for the configured provider.

    Supported providers: ``openai``, ``groq``, ``ollama``. Imports are lazy so
    only the chosen provider's package needs to be installed.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMConfigError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. Add it to .env "
                "or switch to the free Groq/Ollama options."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMConfigError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. Get a free key at "
                "https://console.groq.com/keys"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    raise LLMConfigError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: openai, groq, ollama."
    )
