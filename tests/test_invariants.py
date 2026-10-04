# tests/test_invariants.py
"""Rules that must never break. They run on every gold recommend row the rules can read."""
import pytest

from board_game_reco.catalog import HardConstraints, satisfies
from board_game_reco.recommender import Recommender
from evals.gold import load
from tests.conftest import StubClassifier

PARSE = {r["id"]: r for r in load("parse.jsonl")}
ROWS = [(r, PARSE[r["parse_id"]]) for r in load("recommend.jsonl") if PARSE[r["parse_id"]]["lang"] == "en"]


@pytest.mark.parametrize("row, gold", ROWS, ids=[r["id"] for r, _ in ROWS])
def test_stated_constraints_hold_or_are_reported(catalog, row, gold):
    result = Recommender(catalog, StubClassifier(), None).recommend(gold["text"])
    if result.game is None:
        assert result.follow_up
        return
    strict = HardConstraints(gold["players"], None, gold["youngest_age"])
    assert satisfies(result.game, strict) is None
    timed = HardConstraints(max_minutes=gold["max_minutes"])
    assert satisfies(result.game, timed) is None or ("time limit" in result.reason and result.relaxed)
    if row["must_not_family"]:
        assert not catalog.same_family(catalog.find(row["must_not_family"]), result.game)
