"""Checks that only mean something inside the Docker image. BGR_IMAGE=1 is set by the Dockerfile."""
import importlib.util
import json
import os

import pytest

from tests.conftest import CSV, ROOT

pytestmark = pytest.mark.skipif(os.environ.get("BGR_IMAGE") != "1", reason="runs inside the Docker image only")


class CountingEncoder:
    def __init__(self, inner) -> None:
        self.inner, self.name, self.texts = inner, inner.name, 0

    def encode(self, texts):
        self.texts += len(texts)
        return self.inner.encode(texts)


def test_the_baked_vectors_load_without_encoding_the_catalog():
    from board_game_reco.catalog import Catalog
    from board_game_reco.embedder import Embedder

    encoder = CountingEncoder(Embedder.default())
    catalog = Catalog.load(CSV, ROOT / ".cache" / "vectors", encoder)
    assert len(catalog.games) == 2000 and encoder.texts == 0


def test_the_baked_map_matches_the_baked_vectors():
    from board_game_reco.gamemap import load_map

    manifest = json.loads((ROOT / ".cache" / "vectors" / "manifest.json").read_text())
    game_map = load_map(ROOT / ".cache" / "map", manifest)
    assert game_map is not None and game_map.xy.shape == (2000, 2)


def test_the_image_runs_offline_as_a_non_root_user_with_no_env_file():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.getuid() == 10001
    assert not (ROOT / ".env").exists()
    assert not os.environ.get("LLM_API_KEY")


def test_the_map_builder_libraries_are_not_in_the_image():
    for name in ("umap", "numba", "llvmlite", "pynndescent"):
        assert importlib.util.find_spec(name) is None, name


def test_the_app_code_is_read_only():
    assert not os.access(ROOT / "app", os.W_OK)
    assert not os.access(ROOT / ".cache" / "vectors", os.W_OK)
