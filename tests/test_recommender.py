# tests/test_recommender.py
import pytest

from board_game_reco.catalog import HardConstraints, satisfies
from board_game_reco.recommender import Recommender, Weights, grounded
from board_game_reco.intent import parse
from tests.conftest import FakeLLM, StubClassifier


def make(catalog, llm=None, intent="recommend"):
    return Recommender(catalog, StubClassifier(intent), llm)


def test_pick_satisfies_every_stated_constraint(catalog):
    result = make(catalog).recommend("a game for 4 players under an hour for my 8 year old")
    assert satisfies(result.game, HardConstraints(4, 60, 8)) is None
    assert result.relaxed == ()
    assert result.engine == "rules"


def test_reason_names_the_game_and_links_bgg(catalog):
    result = make(catalog).recommend("family game for 4, not too complicated")
    assert result.game.name in result.reason
    assert result.game.url in result.reason


def test_family_request_picks_a_family_ranked_game_and_never_adult(catalog):
    result = make(catalog).recommend("family game for 4, not too complicated")
    assert result.game.family_rank is not None
    assert "Mature / Adult" not in result.game.categories


def test_runners_up_hold_one_game_per_family(catalog):
    result = make(catalog).recommend("cooperative disease outbreak game")
    games = (result.game, *result.runners_up)
    for i, a in enumerate(games):
        for b in games[i + 1:]:
            assert not catalog.same_family(a, b)


def test_mmr_first_runner_up_is_no_closer_to_the_pick_than_plain_ranking(catalog):
    # With diversity d, the MMR choice c beats the top-scored p only if d*(sim_p - sim_c) >= (1-d)*(total_p - total_c) >= 0.
    query = "cooperative disease outbreak game"
    plain = Recommender(catalog, StubClassifier(), None, Weights(diversity=0.0)).recommend(query)
    diverse = Recommender(catalog, StubClassifier(), None, Weights(diversity=0.9)).recommend(query)
    assert plain.game == diverse.game
    closeness = lambda r: float(catalog.similarity(r.game, [r.runners_up[0]])[0])
    assert closeness(diverse) <= closeness(plain)


def test_similar_to_excludes_anchor_and_its_family(catalog):
    result = make(catalog).recommend("something similar to Pandemic")
    pandemic = catalog.find("Pandemic")
    assert not catalog.same_family(pandemic, result.game)
    assert result.trace["removed"]["same_family"] >= 3


def test_designer_filter_keeps_only_their_games(catalog):
    result = make(catalog).recommend("games by Uwe Rosenberg for 2")
    assert "Uwe Rosenberg" in result.game.designers


def test_playtime_relaxes_and_says_so(catalog):
    # No game seats 12 in 5 or 8 minutes, so this walks both relaxation steps.
    result = make(catalog).recommend("a game for 12 players in 5 minutes")
    assert result.relaxed
    assert "time limit" in result.reason
    assert result.follow_up and "Would that work?" in result.follow_up


def test_players_and_age_never_relax(catalog):
    result = make(catalog).recommend("a game for 20 players for my 2 year old")
    assert result.game is None
    assert "player count" in result.follow_up


def test_avoid_coop_never_picks_coop_when_others_fit(catalog):
    result = make(catalog).recommend("a game for 4, no co-op games")
    assert "Cooperative Game" not in result.game.mechanics


def test_off_topic_is_declined(catalog):
    result = make(catalog, intent="off_topic").recommend("what is the weather tomorrow")
    assert result.game is None and "board games" in result.reason


def test_unclear_asks_back_quoting_the_unread_part(catalog):
    result = make(catalog).recommend("a game for 4 that takes 3 turns")
    assert result.game is None and '"3"' in result.follow_up


def test_unknown_anchor_asks_back(catalog):
    llm = FakeLLM(parsed={"intent": "recommend", "players": None, "max_minutes": None, "youngest_age": None,
                          "weight": None, "coop": None, "solo": False, "first_time": False,
                          "wants_new": False, "family": False, "anchor": "Zzyzx Quest", "designer": None})
    result = make(catalog, llm).recommend("เกมคล้าย Zzyzx Quest")
    assert result.game is None and "Zzyzx Quest" in result.follow_up


def test_cloud_reason_is_used_only_when_grounded(catalog):
    reply = {"intent": "recommend", "players": 4, "max_minutes": None, "youngest_age": None, "weight": None,
             "coop": None, "solo": False, "first_time": False, "wants_new": False, "family": False,
             "anchor": None, "designer": None}
    good = make(catalog, FakeLLM(parsed=reply, written="{name} เหมาะกับ 4 คน")).recommend("เกมสำหรับ 4 คน")
    assert good.engine == "cloud"
    bad_llm = FakeLLM(parsed=reply, written="A great game rated 9.9 by 999999 people")
    bad = make(catalog, bad_llm).recommend("เกมสำหรับ 4 คน")
    assert "999999" not in bad.reason and bad.game.name in bad.reason


def test_grounded_checks_name_and_numbers(catalog):
    game = catalog.find("CATAN")
    allowed = f"{game.name} rated {game.rating:.1f}"
    assert grounded(f"{game.name} is rated {game.rating:.1f}", game, allowed)
    assert not grounded(f"{game.name} is rated 9.9", game, allowed)
    assert not grounded("A fine game", game, allowed)


def test_trace_counts_each_stage(catalog):
    result = make(catalog).recommend("a game for 4 players")
    assert result.trace["catalog"] == 2000
    assert result.trace["picked"] == 1
    assert result.trace["eligible"] <= 2000


def test_similarity_term_is_scaled_to_unit_range_and_follows_raw_similarity(catalog):
    recommender = make(catalog)
    request = parse("a relaxing nature game", catalog=catalog, classifier=StubClassifier(), llm=None)
    pool = list(catalog.games[:200])
    ranked = recommender._rank(request, None, pool)
    terms = {s.game.id: s.terms["similarity"] for s in ranked}
    assert all(0.0 <= t <= 1.0 for t in terms.values())
    raw = catalog.similarity(request.text, pool)
    best = pool[int(raw.argmax())]
    assert terms[best.id] == max(terms.values())
    whole = {s.terms["similarity"] for s in recommender._rank(request, None, list(catalog.games))}
    assert max(whole) == pytest.approx(1.0)


def test_similarity_term_does_not_depend_on_the_pool(catalog):
    recommender = make(catalog)
    request = parse("a relaxing nature game", catalog=catalog, classifier=StubClassifier(), llm=None)
    pool = list(catalog.games[:200])
    full = {s.game.id: s.terms["similarity"] for s in recommender._rank(request, None, pool)}
    half = {s.game.id: s.terms["similarity"] for s in recommender._rank(request, None, pool[:50])}
    assert all(half[i] == pytest.approx(full[i]) for i in half)


@pytest.mark.model
def test_vibe_query_is_won_on_meaning_not_rating():
    from board_game_reco.embedder import Embedder
    from board_game_reco.catalog import Catalog
    from board_game_reco.recommender import ROOT

    encoder = Embedder.default()
    catalog = Catalog.load(ROOT / "data" / "boardgames.csv", ROOT / ".cache" / "vectors", encoder)
    result = Recommender(catalog, StubClassifier(), None).recommend("a relaxing nature game")
    raw = catalog.similarity("a relaxing nature game", list(catalog.games))
    top20 = {catalog.games[i].id for i in raw.argsort()[::-1][:20]}
    assert result.game.name != "Gloomhaven"
    assert result.game.id in top20
