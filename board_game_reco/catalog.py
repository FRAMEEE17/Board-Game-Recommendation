from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Game:
    id: int
    name: str
    year: int | None
    min_players: int
    max_players: int
    max_minutes: int | None
    min_age: int | None
    complexity: float
    rating: float
    num_ratings: int
    categories: tuple[str, ...]
    mechanics: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class HardConstraints:
    players: int | None = None
    max_minutes: int | None = None
    youngest_age: int | None = None


class Catalog:
    @classmethod
    def load(cls, csv_path: Path, cache_dir: Path) -> Catalog:
        """Rebuilds the embeddings when the manifest doesn't match the CSV or the model."""
        raise NotImplementedError

    def find(self, name: str) -> Game | None:
        """Exact name first, then the most-rated game whose name contains it."""
        raise NotImplementedError

    def eligible(self, constraints: HardConstraints) -> list[Game]:
        """Unknown playtime or age fails a constraint on that field, it never passes it."""
        raise NotImplementedError

    def similarity(self, query: str | Game, games: list[Game]) -> list[float]:
        raise NotImplementedError

    def same_family(self, anchor: Game, game: Game) -> bool:
        raise NotImplementedError
