"""The gold rows were drafted by a model, not labeled by a person. These checks are the guard."""
from collections import Counter

from evals.gold import load, validate_parse, validate_recommend


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
