# board_game_reco/gamemap.py
"""A 2D map of the catalog for the stats page, built once at image build time.

Only build_map imports umap-learn and scikit-learn, and only inside the function. The runtime
image has neither umap-learn nor numba, and reads the saved map with numpy alone.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import Game

METHODS = ("umap", "pca")


@dataclass(frozen=True)
class GameMap:
    ids: np.ndarray  # (n,) int32, Game.id in the row order of Catalog.games
    xy: np.ndarray  # (n, 2) float32
    cluster: np.ndarray  # (n,) int8
    labels: tuple[str, ...]  # one per cluster
    method: str  # "umap" or "pca"


def build_map(vectors: np.ndarray, games: tuple[Game, ...], cache_dir: Path, manifest: dict,
              *, k: int = 8, seed: int = 42, method: str = "umap") -> GameMap:
    """Projects the vectors to 2D and clusters them. Writes map.npz, manifest.json, then READY.

    method="umap" falls back to "pca" when umap-learn cannot be imported. The manifest records
    the method that ran, and the stats page names it.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    from sklearn.cluster import KMeans

    xy, used = _project(vectors, seed, method)
    # Clusters come from the full vectors, so a colour means "similar games" whichever projection drew them.
    cluster = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors).astype(np.int8)
    game_map = GameMap(np.array([g.id for g in games], dtype=np.int32), xy.astype(np.float32), cluster,
                       _cluster_labels(games, cluster, k), used)
    _save(game_map, Path(cache_dir), manifest | {"k": k, "seed": seed, "method": used})
    return game_map


def load_map(cache_dir: Path, manifest: dict) -> GameMap | None:
    """None when READY is missing or the map was built from other vectors than these."""
    cache_dir = Path(cache_dir)
    marker, manifest_path = cache_dir / "READY", cache_dir / "manifest.json"
    if not marker.exists() or not manifest_path.exists():
        return None
    saved = json.loads(manifest_path.read_text())
    if any(saved.get(key) != value for key, value in manifest.items()):
        return None
    data = np.load(cache_dir / "map.npz")
    return GameMap(data["ids"], data["xy"], data["cluster"], tuple(json.loads(saved["labels"])), saved["method"])


def _project(vectors: np.ndarray, seed: int, method: str) -> tuple[np.ndarray, str]:
    if method == "umap":
        try:
            import umap
        except ImportError:
            method = "pca"
        else:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
            return np.asarray(reducer.fit_transform(vectors)), "umap"
    centred = vectors - vectors.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    xy = centred @ components[:2].T
    # SVD signs are arbitrary. Fix them so the same vectors always draw the same picture.
    xy *= np.where(xy[np.abs(xy).argmax(axis=0), [0, 1]] < 0, -1.0, 1.0)
    return xy, method


def _cluster_labels(games: tuple[Game, ...], cluster: np.ndarray, k: int, top: int = 2) -> tuple[str, ...]:
    """The mechanics most over-represented in each cluster against the whole catalog."""
    overall = Counter(m for g in games for m in g.mechanics)
    labels = []
    for c in range(k):
        members = [g for g, label in zip(games, cluster) if label == c]
        inside = Counter(m for g in members for m in g.mechanics)
        lift = {m: inside[m] / len(members) - overall[m] / len(games) for m in inside if inside[m] >= 3}
        best = sorted(lift, key=lambda m: (-lift[m], m))[:top]
        labels.append(" / ".join(best) or f"group {c + 1}")
    return tuple(labels)


def _save(game_map: GameMap, cache_dir: Path, manifest: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "READY"
    marker.unlink(missing_ok=True)
    np.savez(cache_dir / "map.npz", ids=game_map.ids, xy=game_map.xy, cluster=game_map.cluster)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest | {"labels": json.dumps(game_map.labels)}))
    marker.touch()
