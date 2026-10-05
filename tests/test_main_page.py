"""The app shell: sidebar, per-session key, token count and page navigation."""
import os

from streamlit.testing.v1 import AppTest

from board_game_reco.llm import LLM
from tests.conftest import ROOT

MAIN = str(ROOT / "app" / "main.py")
TIMEOUT = 30


def main() -> AppTest:
    return AppTest.from_file(MAIN, default_timeout=TIMEOUT).run()


def test_the_shell_shows_the_key_field_and_opens_on_recommend(engine):
    at = main()
    assert not at.exception
    assert at.sidebar.text_input(key="api_key").value == ""
    assert any("Cloud AI" in m.value and "add a key" in m.value for m in at.sidebar.markdown)
    assert at.title[0].value == "What should we play this Sunday morning?"
    assert any(c.value == "Tokens this session: 0" for c in at.sidebar.caption)


def test_a_typed_key_stays_in_the_session_and_never_in_the_environment(engine, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    before = dict(os.environ)
    at = main()
    at.sidebar.text_input(key="api_key").input("gsk_session_only").run()
    llm = at.session_state["llm"]
    assert isinstance(llm, LLM) and llm._client.api_key == "gsk_session_only"
    assert dict(os.environ) == before
    assert any("on for this session" in m.value for m in at.sidebar.markdown)
    assert main().session_state["llm"] is None  # a second browser session has no model


def test_tokens_survive_a_key_change(engine):
    at = main()
    at.sidebar.text_input(key="api_key").input("gsk_first").run()
    at.session_state["llm"]._usage.update(prompt_tokens=120, completion_tokens=30)
    at.sidebar.text_input(key="api_key").input("gsk_second").run()
    assert at.session_state["tokens_before"] == 150
    assert any(c.value == "Tokens this session: 150" for c in at.sidebar.caption)
    at.sidebar.text_input(key="api_key").input("").run()
    assert at.session_state["llm"] is None
    assert any(c.value == "Tokens this session: 150" for c in at.sidebar.caption)


def test_two_sessions_share_one_engine_build(engine):
    _, builds = engine
    for text in ("a game for 4 players", "a game for 2 players"):
        at = main()
        at.text_area(key="request").input(text)
        at.button(key="submit").click().run()
        assert at.session_state["last"].game is not None
    assert len(builds) == 1
