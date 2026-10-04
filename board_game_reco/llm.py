from __future__ import annotations


class LLM:
    """Any OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq) via LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.

    Reads a repo-root .env when present. Real environment variables win over it.
    """

    @classmethod
    def from_env(cls) -> LLM | None:
        raise NotImplementedError

    def json(self, system: str, user: str) -> dict | None:
        """None on any failure, so every caller keeps its deterministic path."""
        raise NotImplementedError

    def text(self, system: str, user: str) -> str | None:
        raise NotImplementedError

    @property
    def usage(self) -> dict[str, int]:
        """Accumulated prompt/completion tokens and call count, for the demo."""
        raise NotImplementedError
