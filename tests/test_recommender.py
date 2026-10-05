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


def test_shorter_than_anchor_caps_playtime_below_the_anchor(catalog):
    catan = catalog.find("CATAN")
    result = make(catalog).recommend("something similar to Catan but shorter")
    assert result.game.max_minutes is not None and result.game.max_minutes < catan.max_minutes
    assert result.relaxed == ()
    assert f"shorter than {catan.name}" in result.reason


def test_shorter_keeps_a_tighter_stated_limit(catalog):
    result = make(catalog).recommend("something like Catan but shorter, under 45 minutes")
    assert result.game.max_minutes <= 45


def test_shorter_than_a_game_with_unknown_playtime_says_so(catalog):
    reply = {"intent": "recommend", "players": None, "max_minutes": None, "youngest_age": None, "weight": None,
             "coop": None, "solo": False, "first_time": False, "wants_new": False, "family": False,
             "anchor": "Chess", "designer": None, "shorter_than_anchor": True}
    result = make(catalog, FakeLLM(parsed=reply)).recommend("เกมคล้าย Chess แต่สั้นกว่า")
    assert "I do not know how long Chess plays" in result.reason


def test_default_with_no_model_never_reads_the_environment(catalog, monkeypatch):
    from board_game_reco import embedder, recommender
    from board_game_reco.llm import LLM
    from tests.conftest import FakeEncoder

    monkeypatch.setattr(embedder.Embedder, "default", classmethod(lambda cls, *a, **k: FakeEncoder()))
    monkeypatch.setattr(recommender.Catalog, "load", classmethod(lambda cls, *a: catalog))
    monkeypatch.setattr(recommender, "IntentClassifier", lambda encoder: StubClassifier())

    def no_env(cls, *a, **k):
        raise AssertionError("LLM.from_env must not run")

    monkeypatch.setattr(LLM, "from_env", classmethod(no_env))
    assert Recommender.default(llm=None).llm is None
    given = FakeLLM()
    assert Recommender.default(llm=given).llm is given


def test_default_with_no_argument_still_reads_the_environment(catalog, monkeypatch):
    from board_game_reco import embedder, recommender
    from board_game_reco.llm import LLM
    from tests.conftest import FakeEncoder

    monkeypatch.setattr(embedder.Embedder, "default", classmethod(lambda cls, *a, **k: FakeEncoder()))
    monkeypatch.setattr(recommender.Catalog, "load", classmethod(lambda cls, *a: catalog))
    monkeypatch.setattr(recommender, "IntentClassifier", lambda encoder: StubClassifier())
    from_env = FakeLLM()
    monkeypatch.setattr(LLM, "from_env", classmethod(lambda cls, *a, **k: from_env))
    assert Recommender.default().llm is from_env


def test_with_llm_shares_catalog_classifier_and_weights(catalog):
    base = Recommender(catalog, StubClassifier(), None, Weights(rating=0.8))
    llm = FakeLLM()
    session = base.with_llm(llm)
    assert session.llm is llm and base.llm is None
    assert session._catalog is base._catalog
    assert session._classifier is base._classifier
    assert session._weights is base._weights


def test_a_pick_carries_its_request_and_no_offers(catalog):
    result = make(catalog).recommend("a game for 4 players")
    assert result.request is not None and result.request.hard == HardConstraints(players=4)
    assert result.offers == ()


def test_relaxed_pick_offers_to_accept_or_keep_the_limit(catalog):
    from board_game_reco.recommender import Offer

    result = make(catalog).recommend("a game for 12 players in 5 minutes")
    assert result.offers == (
        Offer("Yes, no time limit"),
        Offer("Keep 5 min", HardConstraints(players=12, max_minutes=5), relax=False),
    )


def test_relax_offer_names_the_relaxed_limit():
    from board_game_reco.recommender import Offer, _relax_offers

    offers = _relax_offers(HardConstraints(4, 20), HardConstraints(4, 30))
    assert offers == (Offer("Yes, 30 min"), Offer("Keep 20 min", HardConstraints(4, 20), relax=False))


def test_keep_the_limit_retries_without_parsing_and_without_relaxing(catalog):
    llm = FakeLLM()
    recommender = make(catalog, llm)
    first = recommender.recommend("a game for 12 players in 5 minutes")
    keep = first.offers[1]
    again = recommender.retry(first, keep.hard, relax=keep.relax)
    assert llm.calls == []  # no guard, no parse, no reason call: the request was read by the rules
    assert again.game is None and again.relaxed == ()
    assert "player count removed" in again.follow_up
    assert [o.label for o in again.offers] == ["Ignore the player count", "Ignore the time limit"]


def test_retry_never_parses_a_cloud_request_again(catalog):
    reply = {"intent": "recommend", "players": 12, "max_minutes": 5, "youngest_age": None, "weight": None,
             "coop": None, "solo": False, "first_time": False, "wants_new": False, "family": False,
             "anchor": None, "designer": None, "shorter_than_anchor": False}
    llm = FakeLLM(parsed=reply)
    recommender = make(catalog, llm)
    first = recommender.recommend("เกมสำหรับ 12 คน 5 นาที")
    assert first.engine == "cloud" and llm.calls.count("json") == 1
    recommender.retry(first, first.offers[1].hard, relax=False)
    assert llm.calls.count("json") == 1 and llm.calls.count("guard") == 1


def test_no_match_offers_one_button_per_constraint_that_removed_games(catalog):
    result = make(catalog).recommend("a game for 20 players for my 2 year old")
    assert [(o.label, o.hard, o.relax) for o in result.offers] == [
        ("Ignore the player count", HardConstraints(None, None, 2), True),
        ("Ignore the age limit", HardConstraints(20, None, None), True),
    ]


def test_no_match_after_the_engine_dropped_time_does_not_offer_time_again(catalog):
    result = make(catalog).recommend("a game for 2 players in 5 minutes for my 3 year old")
    assert result.game is None
    assert "Ignore the time limit" not in [o.label for o in result.offers]


def test_a_dropped_constraint_is_named_in_the_reason_across_retries(catalog):
    recommender = make(catalog)
    first = recommender.recommend("a game for 2 players in 5 minutes for my 3 year old")
    second = recommender.retry(first, first.offers[0].hard, relax=True)  # ignore the player count
    third = recommender.retry(second, second.offers[0].hard, relax=True)  # ignore the age limit
    assert third.game is not None
    assert satisfies(third.game, HardConstraints(None, 5, None)) is None
    assert "I ignored the player count and the age limit" in third.reason
    assert third.request.hard == HardConstraints(2, 5, 3)


def test_cloud_reason_still_names_a_dropped_constraint(catalog):
    reply = {"intent": "recommend", "players": 20, "max_minutes": None, "youngest_age": 2, "weight": None,
             "coop": None, "solo": False, "first_time": False, "wants_new": False, "family": False,
             "anchor": None, "designer": None, "shorter_than_anchor": False}
    recommender = make(catalog, FakeLLM(parsed=reply, written="{name} เหมาะกับเด็ก"))
    first = recommender.recommend("เกมสำหรับ 20 คน ลูกอายุ 2 ขวบ")
    result = recommender.retry(first, first.offers[1].hard, relax=True)  # ignore the age limit
    assert result.engine == "cloud" and result.game.name in result.reason
    assert "I ignored the age limit" in result.reason


def test_retry_refuses_a_result_without_a_recommend_request(catalog):
    from board_game_reco.recommender import Recommendation

    with pytest.raises(ValueError):
        make(catalog).retry(Recommendation(None, "no request"), HardConstraints(4))
    declined = make(catalog, intent="off_topic").recommend("what is the weather tomorrow")
    with pytest.raises(ValueError):
        make(catalog).retry(declined, HardConstraints(4))
