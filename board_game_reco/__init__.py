from __future__ import annotations

from .recommender import Recommendation, Recommender

__all__ = ["Recommendation", "Recommender", "recommend"]

_default: Recommender | None = None


def recommend(text: str) -> tuple[str, str]:
    """The assignment's required signature: free text in, (game name, reason) out."""
    raise NotImplementedError
