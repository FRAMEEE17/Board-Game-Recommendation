"""The gold rows were drafted by a model, not labeled by a person. These checks are the guard."""
from collections import Counter

from evals.gold import load, validate_judge, validate_parse, validate_recommend, validate_relevance


def test_parse_gold_is_valid_and_complete():
    rows = load("parse.jsonl")
    assert validate_parse(rows) == []
    assert Counter(r["lang"] for r in rows) == {"en": 46, "th": 30, "other": 10}
    assert {r["split"] for r in rows} == {"dev", "test"}
    assert sum(r["intent"] == "injection" for r in rows) >= 4


def test_recommend_gold_points_at_parse_rows():
    parse = {r["id"]: r for r in load("parse.jsonl")}
    rows = load("recommend.jsonl")
    assert len(rows) == 30
    assert validate_recommend(rows, set(parse)) == []
    for r in rows:
        assert parse[r["parse_id"]]["intent"] == "recommend", r["id"]
        assert parse[r["parse_id"]]["split"] == r["split"], r["id"]


def test_gold_names_exist_in_catalog(catalog):
    for r in load("parse.jsonl"):
        if r["anchor"]:
            assert catalog.has_name(r["anchor"]), r["id"]
        if r["designer"]:
            assert catalog.find_designer(r["designer"]) == r["designer"], r["id"]


def test_relevance_gold_is_valid_balanced_and_read_by_the_rules(catalog):
    from board_game_reco.rules import parse as rule_parse

    rows = load("relevance.jsonl")
    assert validate_relevance(rows) == []
    assert len(rows) == 20
    assert Counter(r["split"] for r in rows) == {"dev": 10, "test": 10}
    for r in rows:
        rules = rule_parse(r["text"], is_game=catalog.has_name, find_designer=catalog.find_designer)
        assert rules.unread == (), r["id"]
        assert (rules.anchor is not None) == (r["kind"] == "anchor"), r["id"]


def test_judge_gold_is_half_corrupted_and_consistent_with_the_catalog(catalog):
    from board_game_reco.recommender import grounded
    from evals.judge import game_facts

    rows = load("judge.jsonl")
    assert validate_judge(rows) == []
    assert len(rows) == 30
    assert Counter(r["label"] for r in rows) == {"faithful": 15, "corrupted": 15}
    games = {g.id: g for g in catalog.games}
    caught = {"swapped_number": 0, "invented_claim": 0}
    for r in rows:
        game = games[r["game_id"]]
        passes = grounded(r["reason"], game, game_facts(game))
        if r["label"] == "faithful":
            assert passes, r["id"]
        else:
            caught[r["corruption"]] += not passes
    # Only a swapped number that appears nowhere in the facts can be caught by the number check.
    assert caught == {"swapped_number": 5, "invented_claim": 0}
