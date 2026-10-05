from __future__ import annotations

from .llm import LLM
from .recommender import Recommendation, Recommender

__all__ = ["LLM", "Recommendation", "Recommender", "recommend"]

_default: Recommender | None = None


def recommend(text: str) -> tuple[str, str]:
    """The assignment's required signature: free text in, (game name, reason) out.

    When no game can be picked, the name is empty and the reason holds the question to ask back.
    """
    global _default
    if _default is None:
        _default = Recommender.default()
    result = _default.recommend(text)
    if result.game is None:
        return "", result.follow_up or result.reason
    return result.game.name, result.reason
