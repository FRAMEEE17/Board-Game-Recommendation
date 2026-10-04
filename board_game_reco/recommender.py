from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog, Game
from .llm import LLM


@dataclass(frozen=True)
class Recommendation:
    game: Game | None
    reason: str
    relaxed: tuple[str, ...]
    runners_up: tuple[Game, ...]


class Recommender:
    def __init__(self, catalog: Catalog, llm: LLM | None) -> None:
        self._catalog = catalog
        self._llm = llm

    @classmethod
    def default(cls) -> Recommender:
        raise NotImplementedError

    def recommend(self, text: str) -> Recommendation:
        """Relaxes only playtime (x1.5, then dropped). Players and age are never relaxed."""
        raise NotImplementedError
