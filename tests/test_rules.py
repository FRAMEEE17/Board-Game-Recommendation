# tests/test_rules.py
import pytest

from board_game_reco.rules import parse

GAMES = {"catan", "ticket to ride", "splendor"}
DESIGNERS = {"uwe rosenberg": "Uwe Rosenberg"}


def rp(text):
    return parse(text, is_game=lambda s: s in GAMES, find_designer=DESIGNERS.get)


@pytest.mark.parametrize(
    "text, field, value",
    [
        ("family game for 4, not too complicated", "players", 4),
        ("family game for 4, not too complicated", "family", True),
        ("family game for 4, not too complicated", "weight", "light"),
        ("family", "family", True),
        ("a game for my kids", "family", True),
        ("a family of 4", "family", True),
        ("a family of 4", "players", 4),
        ("a strategy game under an hour", "max_minutes", 60),
        ("something for me and 3 friends in 30 minutes", "players", 4),
        ("something for me and 3 friends in 30 minutes", "max_minutes", 30),
        ("a game with my partner, about 1.5 hours", "players", 2),
        ("a game with my partner, about 1.5 hours", "max_minutes", 90),
        ("strategy <2h", "max_minutes", 120),
        ("a 30-minute game for kids aged 8", "max_minutes", 30),
        ("a 30-minute game for kids aged 8", "youngest_age", 8),
        ("something my 6 year old can play", "youngest_age", 6),
        ("2-4 players", "players", 4),
        ("by myself", "players", 1),
        ("by myself", "solo", True),
        ("no co-op games please", "coop", False),
        ("a cooperative game", "coop", True),
        ("my first board game", "first_time", True),
        ("my first board game", "wants_new", False),
        ("new releases for 2", "wants_new", True),
        ("new releases for 2", "players", 2),
        ("similar to Catan but shorter", "anchor", "catan"),
        ("games by Uwe Rosenberg for 2", "designer", "Uwe Rosenberg"),
        ("games by Uwe Rosenberg for 2", "players", 2),
        ("a heavy economic game", "weight", "heavy"),
    ],
)
def test_reads_field(text, field, value):
    assert getattr(rp(text), field) == value


@pytest.mark.parametrize(
    "text",
    ["family game for 4, not too complicated", "a strategy game under an hour", "similar to Catan but shorter"],
)
def test_fully_read_requests_leave_nothing_unread(text):
    assert rp(text).unread == ()


def test_leftover_number_is_unread():
    assert rp("a game for 4 that takes 3 turns").unread == ("3",)


def test_leftover_unit_word_is_unread():
    assert "hours" in rp("a game for a few hours").unread


def test_unknown_anchor_is_not_taken():
    assert rp("I like heavy games").anchor is None
