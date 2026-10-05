from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app import catalog_stats as cs
from app.theme import PALETTE, apply_seaborn_theme, cluster_colors
from board_game_reco.gamemap import GameMap, load_map
from board_game_reco.recommender import ROOT

CSV = ROOT / "data" / "boardgames.csv"
VECTORS_DIR = ROOT / ".cache" / "vectors"
MAP_DIR = ROOT / ".cache" / "map"
METHOD_CAPTION = {
    "umap": "UMAP of the game embeddings. Colour is a k-means cluster, named by its most distinctive mechanics.",
    "pca": "First two principal components of the game embeddings (umap-learn was not available at build time). "
           "Colour is a k-means cluster, named by its most distinctive mechanics.",
}


@st.cache_data
def frame(csv_path: Path) -> pd.DataFrame:
    return cs.load_frame(csv_path)


@st.cache_resource
def game_map(map_dir: Path, vectors_dir: Path) -> GameMap | None:
    manifest_path = vectors_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return load_map(map_dir, json.loads(manifest_path.read_text()))


def render() -> None:
    apply_seaborn_theme()
    df = frame(CSV)
    st.title("Catalog stats")
    top = cs.headline(df)
    a, b, c = st.columns(3)
    a.metric("Games", f"{top['games']:,.0f}")
    b.metric("Median playtime", f"{top['median_minutes']:.0f} min")
    c.metric("Median weight", f"{top['median_weight']:.2f}")
    q = cs.data_quality(df)
    st.caption(f"Data quality: {q['duplicate_names']:.0f} duplicate names · {q['unknown_playtime']:.0f} unknown "
               f"playtimes · {q['unknown_age']:.0f} unknown ages · price known for {q['price_coverage']:.0%}")

    f1, f2, f3 = st.columns(3)
    category = f1.selectbox("Category", ["Any", *cs.categories(df)], key="category")
    players = f2.selectbox("Players", ["Any", *range(1, 13)], key="players")
    minutes = f3.selectbox("Max minutes", ["Any", 15, 30, 45, 60, 90, 120, 180], key="max_minutes")
    shown = cs.filtered(df, None if category == "Any" else category, None if players == "Any" else int(players),
                        None if minutes == "Any" else int(minutes))
    st.caption(f"{len(shown):,} games match")
    st.dataframe(cs.table(shown), hide_index=True, width="stretch")

    left, right = st.columns(2)
    _chart(left, "Weight against rating", lambda ax: sns.scatterplot(
        data=df, x="complexity", y="avg_rating", s=12, alpha=0.6, color=PALETTE["accent"], ax=ax), "weight", "rating")
    _chart(right, "Playtime spread", lambda ax: sns.histplot(
        df["max_playtime"].dropna().clip(upper=240), bins=24, color=PALETTE["accent"], ax=ax),
        "minutes (240 means 240 or more)", "games")
    _chart(left, "Games per year", lambda ax: sns.histplot(
        df["release_year"].dropna()[lambda s: s >= 1990], discrete=True, color=PALETTE["accent"], ax=ax), "year", "games")
    mechanics = cs.top_mechanics(df, 10)
    _chart(right, "Top mechanics", lambda ax: sns.barplot(
        x=mechanics.values, y=mechanics.index, color=PALETTE["accent"], ax=ax), "games", "")
    trending = cs.trending(df, 10)
    _chart(left, "Trending now", lambda ax: sns.barplot(
        data=trending, x="monthly_plays", y="boardgame", color=PALETTE["accent"], ax=ax), "plays last month", "")
    _chart(right, "Crowd-pleaser or polarizing", lambda ax: sns.scatterplot(
        data=df, x="avg_rating", y="std_deviation", s=12, alpha=0.6, color=PALETTE["accent"], ax=ax),
        "rating", "spread of ratings")

    gems = cs.hidden_gems(df)
    st.subheader(f"Hidden gems ({len(gems)})")
    st.caption("Rated 8 or higher by fewer than 2,000 people.")
    st.dataframe(gems.rename(columns={"boardgame": "name", "avg_rating": "rating", "num_ratings": "ratings"}),
                 hide_index=True, width="stretch")

    st.subheader("Game map")
    _map(df)


def _chart(column, title: str, draw, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    draw(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    column.pyplot(fig)
    plt.close(fig)  # keeps memory flat across reruns


def _map(df: pd.DataFrame) -> None:
    built = game_map(MAP_DIR, VECTORS_DIR)
    if built is None:
        st.info("Game map not built for this catalog. The Docker build makes it with "
                "`python -m board_game_reco.build`.")
        return
    names = df.set_index("row_id")["boardgame"]
    points = pd.DataFrame({"x": built.xy[:, 0], "y": built.xy[:, 1],
                           "group": [built.labels[c] for c in built.cluster], "name": names.loc[built.ids].values})
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=points, x="x", y="y", hue="group", hue_order=list(built.labels),
                    palette=cluster_colors(len(built.labels)), s=14, linewidth=0, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(METHOD_CAPTION[built.method])
