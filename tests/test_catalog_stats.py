# tests/test_catalog_stats.py
import pytest

from app import catalog_stats as cs
from app.theme import CLUSTER_DARK, PALETTE, cluster_colors
from tests.conftest import CSV


@pytest.fixture(scope="module")
def df():
    return cs.load_frame(CSV)


def test_headline_matches_the_design_numbers(df):
    assert cs.headline(df) == {"games": 2000.0, "median_minutes": 60.0, "median_weight": 2.45}


def test_data_quality_counts_the_known_gaps(df):
    quality = cs.data_quality(df)
    assert quality["duplicate_names"] == 15
    assert quality["unknown_playtime"] == 3 and quality["unknown_age"] == 10
    assert quality["price_coverage"] == pytest.approx(0.559)


def test_filters_follow_the_engine_rules(df):
    kept = cs.filtered(df, "Party Game", 6, 30)
    assert len(kept) > 0
    assert (kept["min_players"] <= 6).all() and (kept["max_players"] >= 6).all()
    assert (kept["max_playtime"] <= 30).all()
    assert all("Party Game" in value.split("; ") for value in kept["categories"])
    assert len(cs.filtered(df, None, None, None)) == 2000


def test_an_unknown_playtime_fails_a_time_limit(df):
    unknown = df[df["max_playtime"].isna()]
    assert len(unknown) == 3
    assert not set(unknown.index) & set(cs.filtered(df, None, None, 1440).index)


def test_table_has_the_columns_the_page_shows(df):
    shown = cs.table(df.head(50))
    assert list(shown.columns) == ["name", "players", "minutes", "weight", "rating", "BGG rank"]
    assert shown["BGG rank"].is_monotonic_increasing


def test_hidden_gems_and_trending(df):
    assert len(cs.hidden_gems(df)) == 64
    top = cs.trending(df, 10)
    assert len(top) == 10 and top["monthly_plays"].is_monotonic_decreasing


def test_categories_and_mechanics_are_split(df):
    assert "Party Game" in cs.categories(df)
    assert all(";" not in c for c in cs.categories(df))
    mechanics = cs.top_mechanics(df, 5)
    assert len(mechanics) == 5 and mechanics.is_monotonic_decreasing


def test_cluster_colors_run_from_the_border_colour_to_the_dark_accent():
    colours = cluster_colors(8)
    assert len(colours) == 8 and len(set(colours)) == 8
    assert colours[0] == PALETTE["border"] and colours[-1] == CLUSTER_DARK
    assert cluster_colors(1) == [PALETTE["border"]] and cluster_colors(0) == []
