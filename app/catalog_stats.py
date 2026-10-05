# app/catalog_stats.py
"""Pure functions over the catalog CSV for the stats page. No Streamlit import, so pytest covers them."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# A 0 in these columns means unknown, as in board_game_reco.catalog.
UNKNOWN_IF_ZERO = ("max_playtime", "minimum_age", "release_year")
GEM_RATING, GEM_MAX_RATINGS = 8.0, 2000


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for column in UNKNOWN_IF_ZERO:
        df[column] = df[column].where(df[column] != 0)
    return df


def headline(df: pd.DataFrame) -> dict[str, float]:
    return {
        "games": float(len(df)),
        "median_minutes": float(df["max_playtime"].median()),
        "median_weight": float(df["complexity"].median()),
    }


def data_quality(df: pd.DataFrame) -> dict[str, float]:
    names = df["boardgame"].value_counts()
    return {
        "duplicate_names": float((names > 1).sum()),
        "unknown_playtime": float(df["max_playtime"].isna().sum()),
        "unknown_age": float(df["minimum_age"].isna().sum()),
        "price_coverage": float(df["amazon_price"].notna().mean()),
    }


def categories(df: pd.DataFrame) -> list[str]:
    return sorted({c.strip() for value in df["categories"].dropna() for c in value.split(";") if c.strip()})


def filtered(df: pd.DataFrame, category: str | None, players: int | None, max_minutes: int | None) -> pd.DataFrame:
    """Same rule as the engine: an unknown playtime fails a time limit."""
    keep = pd.Series(True, index=df.index)
    if category:
        keep &= df["categories"].fillna("").map(lambda value: category in {c.strip() for c in value.split(";")})
    if players is not None:
        keep &= (df["min_players"] <= players) & (df["max_players"] >= players)
    if max_minutes is not None:
        keep &= df["max_playtime"].notna() & (df["max_playtime"] <= max_minutes)
    return df[keep]


def table(df: pd.DataFrame) -> pd.DataFrame:
    players = df["min_players"].astype(str).where(
        df["min_players"] == df["max_players"], df["min_players"].astype(str) + "-" + df["max_players"].astype(str))
    return pd.DataFrame({
        "name": df["boardgame"],
        "players": players,
        "minutes": df["max_playtime"].astype("Int64"),
        "weight": df["complexity"].round(2),
        "rating": df["avg_rating"].round(2),
        "BGG rank": df["rank_overall"],
    }).sort_values("BGG rank").reset_index(drop=True)


def hidden_gems(df: pd.DataFrame) -> pd.DataFrame:
    """F44: rated 8 or higher by fewer than 2,000 people. Uses the raw rating, as the design says for F44."""
    gems = df[(df["avg_rating"] >= GEM_RATING) & (df["num_ratings"] < GEM_MAX_RATINGS)]
    return gems.sort_values("avg_rating", ascending=False)[["boardgame", "avg_rating", "num_ratings"]]


def trending(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nlargest(n, "monthly_plays")[["boardgame", "monthly_plays"]]


def top_mechanics(df: pd.DataFrame, n: int = 10) -> pd.Series:
    mechanics = df["mechanics"].dropna().str.split(";").explode().str.strip()
    return mechanics[mechanics != ""].value_counts().head(n)
