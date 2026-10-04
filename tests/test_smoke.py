import board_game_reco
from board_game_reco import Recommender
from tests.conftest import StubClassifier


def test_package_imports():
    assert hasattr(board_game_reco, "recommend")


def test_recommend_returns_name_and_reason(catalog, monkeypatch):
    monkeypatch.setattr(board_game_reco, "_default", Recommender(catalog, StubClassifier(), None))
    name, reason = board_game_reco.recommend("family game for 4, not too complicated")
    assert name and name in reason


def test_recommend_without_a_pick_returns_empty_name_and_question(catalog, monkeypatch):
    monkeypatch.setattr(board_game_reco, "_default", Recommender(catalog, StubClassifier(), None))
    name, reason = board_game_reco.recommend("a game for 4 that takes 3 turns")
    assert name == "" and '"3"' in reason
