import pytest
from streamlit.testing.v1 import AppTest

STATS = "from app import stats_page\nstats_page.render()"
TIMEOUT = 60


@pytest.fixture
def fresh_map_cache():
    from app import stats_page

    stats_page.game_map.clear()
    yield stats_page
    stats_page.game_map.clear()


def test_headline_numbers_and_data_quality_line(fresh_map_cache):
    at = AppTest.from_string(STATS, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["2,000", "60 min", "2.45"]
    assert any(c.value.startswith("Data quality: 15 duplicate names · 3 unknown playtimes · 10 unknown ages")
               for c in at.caption)
    assert any(s.value == "Hidden gems (64)" for s in at.subheader)


def test_filters_narrow_the_table(fresh_map_cache):
    at = AppTest.from_string(STATS, default_timeout=TIMEOUT).run()
    at.selectbox(key="category").select("Party Game")
    at.selectbox(key="players").select(6)
    at.selectbox(key="max_minutes").select(30).run()
    assert not at.exception
    count = next(c.value for c in at.caption if c.value.endswith("games match"))
    assert 0 < int(count.split()[0].replace(",", "")) < 2000


def test_a_missing_map_says_so_instead_of_drawing(fresh_map_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_map_cache, "MAP_DIR", tmp_path / "no-map")
    at = AppTest.from_string(STATS, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any(i.value.startswith("Game map not built for this catalog") for i in at.info)


def test_a_built_map_is_drawn_with_its_method_named(fresh_map_cache, catalog, tmp_path, monkeypatch):
    import json

    import numpy as np

    from board_game_reco.gamemap import build_map
    from tests.conftest import FakeEncoder

    games = catalog.games
    vectors = FakeEncoder().encode([g.name for g in games])
    manifest = {"source_sha256": "test", "model": "fake", "rows": len(games)}
    (tmp_path / "vectors").mkdir()
    (tmp_path / "vectors" / "manifest.json").write_text(json.dumps(manifest))
    build_map(vectors, games, tmp_path / "map", manifest, k=4, method="pca")
    monkeypatch.setattr(fresh_map_cache, "MAP_DIR", tmp_path / "map")
    monkeypatch.setattr(fresh_map_cache, "VECTORS_DIR", tmp_path / "vectors")
    at = AppTest.from_string(STATS, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert not any(i.value.startswith("Game map not built") for i in at.info)
    assert any(c.value.startswith("First two principal components") for c in at.caption)
    assert np.load(tmp_path / "map" / "map.npz")["xy"].shape == (2000, 2)
