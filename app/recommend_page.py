# app/recommend_page.py
from __future__ import annotations

from dataclasses import replace

import streamlit as st

from app import state
from board_game_reco.recommender import CONSTRAINT_LABELS, TERM_LABELS, Recommendation

EXAMPLES = {
    "Family for 4": "family game for 4, not too complicated",
    "Strategy <2h": "a strategy game under 2 hours",
    "With partner": "a game to play with my partner",
    "New releases": "a new game for 3 players",
    "Like Splendor": "something like Splendor but shorter",
}


def render() -> None:
    state.init_session()
    st.title("What should we play this Sunday morning?")
    st.caption("Any language: who's playing, how long, or a game you love.")
    with st.form("ask"):
        st.text_area("Your request", key="request", placeholder=EXAMPLES["Family for 4"],
                     label_visibility="collapsed")
        submitted = st.form_submit_button("Recommend", type="primary", key="submit")
    for column, (label, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
        column.button(label, key=f"example-{label}", on_click=_example, args=(text,), width="stretch")
    if submitted and st.session_state.request.strip():
        with st.spinner("Picking a game"):
            _ask(st.session_state.request)

    # Render from the stored result. Calling recommend here would spend tokens on every rerun.
    result: Recommendation | None = st.session_state.last
    if result is not None:
        _show(result)


def _example(text: str) -> None:
    st.session_state.request = text
    _ask(text)


def _ask(text: str) -> None:
    st.session_state.last = state.session_recommender().recommend(text)


def _take(index: int) -> None:
    result: Recommendation = st.session_state.last
    offer = result.offers[index]
    if offer.hard is None:  # accept what is already on screen: no engine call
        st.session_state.last = replace(result, follow_up=None, offers=())
    else:
        st.session_state.last = state.session_recommender().retry(result, offer.hard, relax=offer.relax)


def badge(engine: str) -> str:
    return "Understood by: " + ("rules, no AI call" if engine == "rules" else "Cloud AI")


def facts(game) -> str:
    players = (f"{game.min_players} players" if game.min_players == game.max_players
               else f"{game.min_players}-{game.max_players} players")
    minutes = f"{game.max_minutes} min" if game.max_minutes else "time unknown"
    age = f"age {game.min_age}+" if game.min_age else "age unknown"
    return " · ".join([players, minutes, age, f"weight {game.complexity:.1f}", f"rated {game.rating:.1f}"])


def _show(result: Recommendation) -> None:
    if result.game is None:
        with st.container(border=True):
            st.markdown(result.follow_up or result.reason)
            _offers(result)
            st.caption(badge(result.engine))
        return

    game = result.game
    with st.container(border=True):
        st.markdown(f"### [{game.name}]({game.url})")
        st.caption(facts(game))
        st.markdown(result.reason.replace(game.url, "").strip())
        st.caption(badge(result.engine))
    if result.follow_up:
        with st.container(border=True):
            st.markdown(result.follow_up)
            _offers(result)
    with st.expander("Why this game"):
        _why(result)


def _offers(result: Recommendation) -> None:
    if not result.offers:
        return
    for i, (column, offer) in enumerate(zip(st.columns(len(result.offers)), result.offers)):
        column.button(offer.label, key=f"offer-{i}", on_click=_take, args=(i,))


def _why(result: Recommendation) -> None:
    st.markdown("**Score per term**")
    st.table({"term": [TERM_LABELS[k] for k in result.terms], "score": [round(v, 3) for v in result.terms.values()]})
    if result.runners_up:
        st.markdown("**Runners-up**: " + ", ".join(f"[{g.name}]({g.url})" for g in result.runners_up))
    trace = result.trace
    st.markdown(f"{trace['catalog']:,} games → {trace['eligible']:,} eligible → picked 1")
    removed = [f"{CONSTRAINT_LABELS[k]} removed {v:,}" for k, v in trace.get("removed", {}).items() if v]
    if removed:
        st.caption("; ".join(removed))
