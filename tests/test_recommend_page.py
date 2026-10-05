# tests/test_recommend_page.py
"""The recommend page, driven by streamlit.testing.v1.AppTest on the fake catalog."""
import pytest
from streamlit.testing.v1 import AppTest

from app.recommend_page import EXAMPLES
from board_game_reco.rules import parse as rule_parse

PAGE = "from app import recommend_page\nrecommend_page.render()"
TIMEOUT = 30


def ask(at: AppTest, text: str) -> AppTest:
    at.text_area(key="request").input(text)
    return at.button(key="submit").click().run()


def page() -> AppTest:
    return AppTest.from_string(PAGE, default_timeout=TIMEOUT).run()


def test_a_request_shows_the_card_with_badge_and_reason(engine):
    at = ask(page(), "family game for 4, not too complicated")
    assert not at.exception
    result = at.session_state["last"]
    assert result.game is not None
    assert any(result.game.name in m.value for m in at.markdown)
    assert any(c.value == "Understood by: rules, no AI call" for c in at.caption)


def test_an_example_chip_fills_the_box_and_answers(engine):
    at = page()
    at.button(key="example-Family for 4").click().run()
    assert at.session_state["request"] == "family game for 4, not too complicated"
    assert at.session_state["last"].game is not None


@pytest.mark.parametrize("text", list(EXAMPLES.values()))
def test_every_example_is_read_in_full_by_the_rules(catalog, text):
    assert rule_parse(text, is_game=catalog.has_name, find_designer=catalog.find_designer).unread == ()


def test_a_rerun_renders_the_stored_result_without_asking_again(engine):
    classifier, _ = engine
    at = ask(page(), "a game for 4 players")
    assert classifier.count == 1
    at.run()
    at.run()
    assert classifier.count == 1


def test_keep_the_limit_retries_without_a_new_parse(engine):
    classifier, _ = engine
    at = ask(page(), "a game for 12 players in 5 minutes")
    assert at.session_state["last"].relaxed
    assert [b.label for b in at.button if b.key and b.key.startswith("offer-")] == ["Yes, no time limit", "Keep 5 min"]
    at.button(key="offer-1").click().run()
    result = at.session_state["last"]
    assert result.game is None and "player count removed" in result.follow_up
    assert classifier.count == 1
    assert at.button(key="offer-0").label == "Ignore the player count"


def test_accepting_the_relaxed_pick_clears_the_question_and_keeps_the_game(engine):
    classifier, _ = engine
    at = ask(page(), "a game for 12 players in 5 minutes")
    shown = at.session_state["last"].game
    at.button(key="offer-0").click().run()
    result = at.session_state["last"]
    assert result.game == shown and result.follow_up is None and result.offers == ()
    assert classifier.count == 1


def test_why_this_game_shows_the_funnel(engine):
    at = ask(page(), "a game for 4 players")
    assert any("2,000 games →" in m.value and "picked 1" in m.value for m in at.markdown)
