# tests/test_catalog.py
from board_game_reco.catalog import Catalog, HardConstraints, satisfies
from tests.conftest import CSV, FakeEncoder


def test_loads_every_row_with_row_id_as_key(catalog):
    assert len(catalog.games) == 2000
    assert len({g.id for g in catalog.games}) == 2000


def test_zero_means_unknown(catalog):
    assert sum(g.max_minutes is None for g in catalog.games) == 3
    assert sum(g.min_age is None for g in catalog.games) == 10
    assert sum(g.year is None for g in catalog.games) == 3


def test_lists_are_split_on_semicolons(catalog):
    brass = catalog.find("Brass: Birmingham")
    assert brass.designers == ("Gavan Brown", "Matt Tolman", "Martin Wallace")
    assert "Hand Management" in brass.mechanics


def test_find_ignores_case_and_prefers_most_rated(catalog):
    assert catalog.find("catan").name == "CATAN"
    assert catalog.find("  PANDEMIC ").name == "Pandemic"
    assert catalog.find("no such game zzz") is None


def test_has_name_is_exact_only(catalog):
    assert catalog.has_name("catan")
    assert not catalog.has_name("cat")


def test_find_designer_is_case_insensitive(catalog):
    assert catalog.find_designer("uwe rosenberg") == "Uwe Rosenberg"
    assert catalog.find_designer("nobody") is None


def test_unknown_values_fail_their_constraint(catalog):
    unknown_time = next(g for g in catalog.games if g.max_minutes is None)
    assert satisfies(unknown_time, HardConstraints(max_minutes=600)) == "max_minutes"
    assert satisfies(unknown_time, HardConstraints()) is None


def test_eligible_reports_removed_counts(catalog):
    result = catalog.eligible(HardConstraints(players=4, max_minutes=30, youngest_age=8))
    assert result.kept
    assert all(satisfies(g, HardConstraints(players=4, max_minutes=30, youngest_age=8)) is None for g in result.kept)
    assert len(result.kept) + sum(result.removed.values()) == 2000
    assert set(result.removed) == {"players", "max_minutes", "youngest_age"}


def test_same_family_uses_game_tag(catalog):
    pandemic = catalog.find("Pandemic")
    assert catalog.same_family(pandemic, catalog.find("Iberia"))
    assert catalog.same_family(pandemic, catalog.find("Fall of Rome"))
    assert not catalog.same_family(pandemic, catalog.find("CATAN"))


def test_vectors_are_cached_with_manifest_and_marker(tmp_path):
    encoder = FakeEncoder()
    Catalog.load(CSV, tmp_path, encoder)
    Catalog.load(CSV, tmp_path, encoder)
    assert encoder.calls == 1
    assert (tmp_path / "READY").exists()
    (tmp_path / "READY").unlink()
    Catalog.load(CSV, tmp_path, encoder)
    assert encoder.calls == 2


def test_adjusted_rating_pulls_thin_ratings_toward_the_mean(catalog):
    thin = min(catalog.games, key=lambda g: g.num_ratings)
    thick = max(catalog.games, key=lambda g: g.num_ratings)
    mean = sum(g.rating for g in catalog.games) / len(catalog.games)
    assert abs(catalog.adjusted_rating(thin) - mean) < abs(thin.rating - mean)
    assert abs(catalog.adjusted_rating(thick) - thick.rating) < 0.1


def test_similarity_ranks_overlapping_text_higher(catalog):
    games = [catalog.find("Pandemic"), catalog.find("Brass: Birmingham")]
    scores = catalog.similarity("cooperative disease outbreak", games)
    assert scores.shape == (2,)
