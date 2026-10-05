"""python -m evals.tune  writes the similarity and rating weights chosen by the newest tune run into Weights."""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
RECOMMENDER = ROOT / "board_game_reco" / "recommender.py"


def latest_choice(results: Path = RESULTS) -> tuple[float, float]:
    """The chosen weights from the most recently written summary.csv that holds a tune run."""
    summaries = sorted(Path(results).glob("*/summary.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for summary in summaries:
        with summary.open(newline="") as f:
            metrics = {(row["suite"], row["metric"]): float(row["value"]) for row in csv.DictReader(f)}
        if ("tune", "chosen_similarity") in metrics:
            return metrics[("tune", "chosen_similarity")], metrics[("tune", "chosen_rating")]
    raise SystemExit("No tune run found. Run `uv run python -m evals.run tune` first.")


def write_defaults(path: Path, similarity: float, rating: float) -> None:
    text = Path(path).read_text(encoding="utf-8")
    text, found_similarity = re.subn(r"(?m)^    similarity: float = [0-9.]+$", f"    similarity: float = {similarity}", text)
    text, found_rating = re.subn(r"(?m)^    rating: float = [0-9.]+$", f"    rating: float = {rating}", text)
    if (found_similarity, found_rating) != (1, 1):
        raise ValueError("expected exactly one similarity and one rating default in Weights")
    Path(path).write_text(text, encoding="utf-8")


def main() -> None:
    similarity, rating = latest_choice()
    write_defaults(RECOMMENDER, similarity, rating)
    print(f"Weights.similarity = {similarity}, Weights.rating = {rating}, written to {RECOMMENDER}")


if __name__ == "__main__":
    main()
