# board_game_reco/catalog.py
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

COOP = "Cooperative Game"
SOLO = "Solo / Solitaire Game"


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


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
    families: tuple[str, ...]
    designers: tuple[str, ...]
    family_rank: int | None
    url: str
    description: str

    @property
    def lines(self) -> tuple[str, ...]:
        """BGG 'Game: X' family tags. 39% of games carry one."""
        return tuple(f for f in self.families if f.startswith("Game: "))


@dataclass(frozen=True)
class HardConstraints:
    players: int | None = None
    max_minutes: int | None = None
    youngest_age: int | None = None


@dataclass(frozen=True)
class Eligibility:
    kept: tuple[Game, ...]
    removed: dict[str, int]


def satisfies(game: Game, constraints: HardConstraints) -> str | None:
    """Name of the first constraint the game fails, or None. Unknown values fail."""
    if constraints.players is not None and not game.min_players <= constraints.players <= game.max_players:
        return "players"
    if constraints.max_minutes is not None and (game.max_minutes is None or game.max_minutes > constraints.max_minutes):
        return "max_minutes"
    if constraints.youngest_age is not None and (game.min_age is None or game.min_age > constraints.youngest_age):
        return "youngest_age"
    return None


class Catalog:
    def __init__(self, games: tuple[Game, ...], vectors: np.ndarray, encoder: Encoder) -> None:
        self.games = games
        self._vectors = vectors
        self._encoder = encoder
        self._row = {g.id: i for i, g in enumerate(games)}
        # Bayesian average: a rating backed by few votes is pulled toward the catalog mean.
        self._prior_votes = float(np.median([g.num_ratings for g in games]))
        self._prior_mean = float(np.mean([g.rating for g in games]))
        adjusted = [self.adjusted_rating(g) for g in games]
        self.rating_min, self.rating_max = min(adjusted), max(adjusted)
        self._designers = {d.casefold(): d for g in games for d in g.designers}

    @classmethod
    def load(cls, csv_path: Path, cache_dir: Path, encoder: Encoder) -> Catalog:
        """Rebuilds the vectors when the READY marker is missing or the manifest doesn't match."""
        games = _read_games(Path(csv_path))
        vectors = _cached_vectors(games, Path(csv_path), Path(cache_dir), encoder)
        return cls(games, vectors, encoder)

    def adjusted_rating(self, game: Game) -> float:
        """Raw rating stays for display and for hidden gems (F44). Scoring uses this one."""
        votes = game.num_ratings
        return (votes * game.rating + self._prior_votes * self._prior_mean) / (votes + self._prior_votes)

    def find(self, name: str) -> Game | None:
        """Exact name first, then the most-rated game whose name contains it."""
        key = name.casefold().strip()
        if not key:
            return None
        pool = [g for g in self.games if g.name.casefold() == key] or [
            g for g in self.games if key in g.name.casefold()
        ]
        return max(pool, key=lambda g: g.num_ratings, default=None)

    def has_name(self, name: str) -> bool:
        key = name.casefold().strip()
        return any(g.name.casefold() == key for g in self.games)

    def find_designer(self, name: str) -> str | None:
        return self._designers.get(name.casefold().strip())

    def eligible(self, constraints: HardConstraints) -> Eligibility:
        removed = {"players": 0, "max_minutes": 0, "youngest_age": 0}
        kept = []
        for game in self.games:
            failed = satisfies(game, constraints)
            if failed:
                removed[failed] += 1
            else:
                kept.append(game)
        return Eligibility(tuple(kept), removed)

    def similarity(self, query: str | Game, games: list[Game]) -> np.ndarray:
        if isinstance(query, Game):
            target = self._vectors[self._row[query.id]]
        else:
            target = self._encoder.encode([query])[0]
        return self._vectors[[self._row[g.id] for g in games]] @ target

    def same_family(self, anchor: Game, game: Game) -> bool:
        if anchor.id == game.id or set(anchor.lines) & set(game.lines):
            return True
        return not anchor.lines and anchor.name.casefold() in game.name.casefold()


def _known(value: str) -> int | None:
    try:
        number = int(float(value))
    except ValueError:
        return None
    return number or None


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _read_games(csv_path: Path) -> tuple[Game, ...]:
    with csv_path.open(encoding="utf-8") as f:
        return tuple(
            Game(
                id=int(row["row_id"]),
                name=row["boardgame"],
                year=_known(row["release_year"]),
                min_players=int(row["min_players"]),
                max_players=int(row["max_players"]),
                max_minutes=_known(row["max_playtime"]),
                min_age=_known(row["minimum_age"]),
                complexity=float(row["complexity"]),
                rating=float(row["avg_rating"]),
                num_ratings=int(float(row["num_ratings"])),
                categories=_split(row["categories"]),
                mechanics=_split(row["mechanics"]),
                families=_split(row["families"]),
                designers=_split(row["designers"]),
                family_rank=_known(row["rank_family"]),
                url=row["url"],
                description=row["description"],
            )
            for row in csv.DictReader(f)
        )


def _embedding_text(game: Game) -> str:
    return (
        f"{game.name}. {game.description} "
        f"Categories: {', '.join(game.categories)}. Mechanics: {', '.join(game.mechanics)}."
    )


def _cached_vectors(games: tuple[Game, ...], csv_path: Path, cache_dir: Path, encoder: Encoder) -> np.ndarray:
    manifest = {
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "model": encoder.name,
        "rows": len(games),
    }
    marker, manifest_path, vectors_path = cache_dir / "READY", cache_dir / "manifest.json", cache_dir / "vectors.npy"
    if marker.exists() and manifest_path.exists() and json.loads(manifest_path.read_text()) == manifest:
        return np.load(vectors_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    vectors = encoder.encode([_embedding_text(g) for g in games])
    np.save(vectors_path, vectors)
    manifest_path.write_text(json.dumps(manifest))
    marker.touch()
    return vectors
