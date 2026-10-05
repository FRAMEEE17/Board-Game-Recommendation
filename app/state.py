# app/state.py
"""What is shared by every session, and what belongs to one.

The encoder, catalog and classifier load once per process behind st.cache_resource. The LLM, and
with it the API key, lives only in this session's st.session_state. Nothing here reads or writes
os.environ, so a key typed in one browser never reaches another.
"""
from __future__ import annotations

import streamlit as st

from board_game_reco import LLM, Recommender


def _build() -> Recommender:
    return Recommender.default(llm=None)


@st.cache_resource(show_spinner="Loading the catalog and the model")
def shared() -> Recommender:
    return _build()


def init_session() -> None:
    st.session_state.setdefault("llm", None)
    st.session_state.setdefault("tokens_before", 0)
    st.session_state.setdefault("last", None)


def session_recommender() -> Recommender:
    """Cheap: a Recommender that points at the shared catalog and carries this session's model."""
    init_session()
    return shared().with_llm(st.session_state.llm)


def set_key(key: str) -> None:
    """Replaces this session's model. Tokens spent under the previous key stay in the total."""
    init_session()
    previous = st.session_state.llm
    if previous is not None:
        st.session_state.tokens_before += _spent(previous)
    key = key.strip()
    st.session_state.llm = LLM.from_key(key) if key else None


def session_tokens() -> int:
    init_session()
    llm = st.session_state.llm
    return st.session_state.tokens_before + (_spent(llm) if llm is not None else 0)


def _spent(llm) -> int:
    usage = llm.usage
    return usage["prompt_tokens"] + usage["completion_tokens"]
