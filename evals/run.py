# evals/run.py
"""python -m evals.run [suite ...] [--no-fp32]

No arguments runs the suites that need no key: parse, route, recommend, perf.
The suites that call the judge run only when named: judge, relevance, tune, explain, ragas.
explain without a key still reports the grounding pass rate.
--no-fp32 skips the int8 against fp32 comparison in perf, which downloads 471 MB. The Docker
image runs perf this way, with no network. BGR_RESULTS moves the output folder.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from board_game_reco.catalog import Catalog
from board_game_reco.recommender import ROOT, Recommender
from evals.gold import load
from evals.judge import DailyLimit, Judge
from evals.perf import escalated_texts, perf_suite
from evals.quality import explain_suite, judge_suite, relevance_cases, relevance_suite, tune_suite
from evals.suites import parse_suite, recommend_suite, route_suite

RESULTS = Path(os.environ.get("BGR_RESULTS") or Path(__file__).resolve().parent / "results")
OFFLINE = ("parse", "route", "recommend", "perf")
NETWORK = ("judge", "relevance", "tune", "explain", "ragas")


def split_flags(argv: list[str]) -> tuple[list[str], set[str]]:
    flags = {a for a in argv if a.startswith("--")}
    unknown = flags - {"--no-fp32"}
    if unknown:
        raise SystemExit(f"unknown flag: {', '.join(sorted(unknown))}. The only flag is --no-fp32")
    return [a for a in argv if not a.startswith("--")], flags


def git_sha() -> str:
    """Short commit hash, or "nogit" where git is missing, as in the Docker image."""
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except OSError:
        return "nogit"
    return done.stdout.strip() or "nogit"


def select(names: list[str]) -> list[str]:
    if not names:
        return list(OFFLINE)
    unknown = [n for n in names if n not in OFFLINE + NETWORK]
    if unknown:
        raise SystemExit(f"unknown suite: {', '.join(unknown)}. Choose from: {', '.join(OFFLINE + NETWORK)}")
    return names


def fp32_recommender() -> Recommender:
    """The same pipeline on the fp32 build, with its own vector cache so the int8 vectors stay put."""
    from board_game_reco.embedder import FP32_ONNX, Embedder
    from board_game_reco.intent import IntentClassifier

    encoder = Embedder.default(FP32_ONNX)
    catalog = Catalog.load(ROOT / "data" / "boardgames.csv", ROOT / ".cache" / "vectors-fp32", encoder)
    return Recommender(catalog, IntentClassifier(encoder), None)


def main(argv: list[str]) -> None:
    names, flags = split_flags(argv)
    names = select(names)
    out = RESULTS / f"{date.today().isoformat()}_{git_sha()}"
    out.mkdir(parents=True, exist_ok=True)

    recommender = Recommender.default()
    catalog, classifier, llm = recommender._catalog, recommender._classifier, recommender.llm
    rules_only = Recommender(catalog, classifier, None)
    gold_parse = load("parse.jsonl")
    by_id = {g["id"]: g for g in gold_parse}
    gold_recommend = load("recommend.jsonl")
    cases = relevance_cases(gold_recommend, by_id, load("relevance.jsonl"))

    judge = None
    if any(n in NETWORK for n in names):
        judge = Judge.from_env()
        if judge is None:
            print("LLM_API_KEY is not set: judge, relevance and tune are skipped, explain reports grounding only")

    results = {}
    if "parse" in names:
        results["parse"] = parse_suite(gold_parse, catalog, classifier, llm)
    if "route" in names:
        results["route"] = route_suite(gold_parse, catalog, classifier, out)
    if "recommend" in names:
        results["recommend"] = recommend_suite(gold_recommend, by_id, recommender)
    if "perf" in names:
        results["perf"] = perf_suite(gold_parse, [c["text"] for c in cases], rules_only,
                                     recommender if llm is not None else None,
                                     None if "--no-fp32" in flags else fp32_recommender(),
                                     escalated_texts(gold_parse, catalog))
    try:
        if judge is not None and "judge" in names:
            results["judge"] = judge_suite(load("judge.jsonl"), catalog, judge, out)
        if judge is not None and "relevance" in names:
            results["relevance"] = relevance_suite(cases, rules_only, judge)
        if judge is not None and "tune" in names:
            results["tune"] = tune_suite(cases, lambda w: Recommender(catalog, classifier, None, w), judge)
        if "explain" in names:
            results["explain"] = explain_suite(cases, rules_only, catalog, judge)
        if "ragas" in names:
            from evals.ragas_faith import RagasFaithfulness

            ragas = RagasFaithfulness.from_env()
            if ragas is not None:
                results["ragas"] = explain_suite(cases, rules_only, catalog, None, ragas)
    except DailyLimit as error:
        print(f"stopped early: {error}")

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
        print("app tokens:", llm.usage)
    if judge is not None:
        print(f"judge: {judge.calls} calls, {judge.hits} cached, {judge.tokens} tokens")
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1:])
