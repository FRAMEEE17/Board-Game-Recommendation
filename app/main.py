"""streamlit run app/main.py

Sunday-Morning: the recommend page and the catalog stats page, with a per-session API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/main.py` puts only app/ on sys.path. Both packages live one level up, in the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app import state
from app.recommend_page import render as recommend_page
from app.stats_page import render as stats_page
from app.theme import CSS


def _key_changed() -> None:
    state.set_key(st.session_state.api_key)


st.set_page_config(page_title="Sunday-Morning", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
state.init_session()

with st.sidebar:
    st.subheader("Engine")
    st.markdown("**Basic**: always on")
    cloud = "on for this session" if st.session_state.llm is not None else "add a key"
    st.markdown(f"**Cloud AI**: {cloud}")
    st.text_input("API key", type="password", key="api_key", on_change=_key_changed,
                  help="A Groq key. It stays in this browser session and is never saved or logged.")
    st.caption("This session only")
    tokens = st.empty()

page = st.navigation([
    st.Page(recommend_page, title="Recommend", url_path="recommend", default=True),
    st.Page(stats_page, title="Catalog stats", url_path="stats"),
])
page.run()
# Filled after the page ran, so the count includes this run's model calls.
tokens.caption(f"Tokens this session: {state.session_tokens():,}")
