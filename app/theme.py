# app/theme.py
"""Sunday-Morning colours, shared by the Streamlit CSS and the seaborn charts."""
from __future__ import annotations

PALETTE = {"background": "#FFFFFF", "card": "#FFF7ED", "accent": "#FB923C", "border": "#FED7AA", "text": "#1F2937"}
# The accent darkened, so the last cluster colour still reads on white.
CLUSTER_DARK = "#9A3412"

CSS = f"""
<style>
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {PALETTE["card"]};
    border-color: {PALETTE["border"]};
}}
.stButton button, .stFormSubmitButton button {{
    border-color: {PALETTE["border"]};
}}
</style>
"""


def cluster_colors(k: int) -> list[str]:
    """k shades from the border colour to CLUSTER_DARK, in order, so neighbouring clusters stay apart."""
    if k < 1:
        return []
    start, end = _rgb(PALETTE["border"]), _rgb(CLUSTER_DARK)
    steps = [i / (k - 1) if k > 1 else 0.0 for i in range(k)]
    return ["#{:02X}{:02X}{:02X}".format(*(round(a + (b - a) * t) for a, b in zip(start, end))) for t in steps]


def apply_seaborn_theme() -> None:
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        palette=[PALETTE["accent"], CLUSTER_DARK, PALETTE["border"]],
        rc={
            "figure.facecolor": PALETTE["background"],
            "axes.facecolor": PALETTE["background"],
            "axes.edgecolor": PALETTE["border"],
            "grid.color": PALETTE["card"],
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
        },
    )


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
