# evals/run.py
"""python -m evals.run [parse] [route] [recommend]  (no arguments runs all three)"""
from __future__ import annotations

import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

from board_game_reco.recommender import Recommender
from evals.gold import load
from evals.suites import parse_suite, recommend_suite, route_suite

RESULTS = Path(__file__).resolve().parent / "results"


def main(names: list[str]) -> None:
    names = names or ["parse", "route", "recommend"]
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = RESULTS / f"{date.today().isoformat()}_{sha or 'nogit'}"
    out.mkdir(parents=True, exist_ok=True)

    recommender = Recommender.default()
    catalog, classifier, llm = recommender._catalog, recommender._classifier, recommender.llm
    gold_parse = load("parse.jsonl")
    results = {}
    if "parse" in names:
        results["parse"] = parse_suite(gold_parse, catalog, classifier, llm)
    if "route" in names:
        results["route"] = route_suite(gold_parse, catalog, classifier, out)
    if "recommend" in names:
        results["recommend"] = recommend_suite(load("recommend.jsonl"), {g["id"]: g for g in gold_parse}, recommender)

    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "metric", "value"])
        for suite, result in results.items():
            for metric, value in result.metrics.items():
                writer.writerow([suite, metric, round(value, 4)])
                print(f"{suite:10s} {metric:28s} {value:.4f}")
    with (out / "rows.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "row"])
        for suite, result in results.items():
            for row in result.rows:
                writer.writerow([suite, row])
    if llm is not None:
        print("tokens:", llm.usage)
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1:])
