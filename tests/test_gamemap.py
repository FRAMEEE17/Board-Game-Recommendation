# tests/test_gamemap.py
import json

import numpy as np
import pytest

from board_game_reco.gamemap import GameMap, build_map, load_map
from tests.conftest import FakeEncoder

MANIFEST = {"source_sha256": "abc", "model": "fake-bag-of-words-64", "rows": 300}


@pytest.fixture(scope="module")
def sample(catalog):
    games = catalog.games[:300]
    return games, FakeEncoder().encode([f"{g.name} {' '.join(g.mechanics)}" for g in games])


def test_pca_map_is_saved_and_loads_back(sample, tmp_path):
    games, vectors = sample
    built = build_map(vectors, games, tmp_path, MANIFEST, k=4, method="pca")
    assert built.xy.shape == (300, 2) and built.xy.dtype == np.float32
    assert built.cluster.dtype == np.int8 and set(built.cluster.tolist()) == {0, 1, 2, 3}
    assert len(built.labels) == 4 and all(built.labels)
    assert built.method == "pca"
    assert list(built.ids) == [g.id for g in games]
    loaded = load_map(tmp_path, MANIFEST)
    assert isinstance(loaded, GameMap)
    assert np.array_equal(loaded.xy, built.xy) and np.array_equal(loaded.cluster, built.cluster)
    assert loaded.labels == built.labels and loaded.method == "pca"


def test_the_same_vectors_draw_the_same_map(sample, tmp_path):
    games, vectors = sample
    first = build_map(vectors, games, tmp_path / "a", MANIFEST, k=4, method="pca")
    second = build_map(vectors, games, tmp_path / "b", MANIFEST, k=4, method="pca")
    assert np.array_equal(first.xy, second.xy) and np.array_equal(first.cluster, second.cluster)


def test_a_map_built_from_other_vectors_is_not_loaded(sample, tmp_path):
    games, vectors = sample
    build_map(vectors, games, tmp_path, MANIFEST, k=4, method="pca")
    assert load_map(tmp_path, MANIFEST | {"model": "another-model"}) is None
    assert load_map(tmp_path, MANIFEST | {"source_sha256": "def"}) is None


def test_a_map_without_the_ready_marker_is_not_loaded(sample, tmp_path):
    games, vectors = sample
    build_map(vectors, games, tmp_path, MANIFEST, k=4, method="pca")
    (tmp_path / "READY").unlink()
    assert load_map(tmp_path, MANIFEST) is None
    assert load_map(tmp_path / "missing", MANIFEST) is None


def test_the_manifest_records_k_seed_and_method(sample, tmp_path):
    games, vectors = sample
    build_map(vectors, games, tmp_path, MANIFEST, k=4, seed=7, method="pca")
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["k"] == 4 and saved["seed"] == 7 and saved["method"] == "pca"
    assert saved["rows"] == 300


def test_umap_falls_back_to_pca_when_umap_is_missing(sample, tmp_path, monkeypatch):
    import builtins

    games, vectors = sample
    real_import = builtins.__import__

    def no_umap(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("no umap here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_umap)
    assert build_map(vectors, games, tmp_path, MANIFEST, k=4, method="umap").method == "pca"


def test_an_unknown_method_is_refused(sample, tmp_path):
    games, vectors = sample
    with pytest.raises(ValueError):
        build_map(vectors, games, tmp_path, MANIFEST, method="tsne")


@pytest.mark.map
def test_umap_runs_when_the_map_group_is_installed(sample, tmp_path):
    pytest.importorskip("umap")
    games, vectors = sample
    assert build_map(vectors, games, tmp_path, MANIFEST, k=4, method="umap").method == "umap"
