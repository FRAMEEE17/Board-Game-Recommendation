"""Suites that need the judge: relevance, weight tuning, judge quality and reason faithfulness."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import mean

from board_game_reco.recommender import Weights
from evals.suites import SuiteResult

GRID_SIMILARITY = (0.6, 1.0, 1.4)
GRID_RATING = (0.2, 0.4, 0.8)


def relevance_cases(gold_recommend: list[dict], gold_parse: dict[str, dict], gold_relevance: list[dict]) -> list[dict]:
    """English recommend gold rows, which the rules path can answer without a key, plus the relevance rows."""
    cases = [{"id": r["id"], "text": gold_parse[r["parse_id"]]["text"], "split": r["split"]}
             for r in gold_recommend if gold_parse[r["parse_id"]]["lang"] == "en"]
    cases += [{"id": r["id"], "text": r["text"], "split": r["split"]} for r in gold_relevance]
    return cases


def score_picks(cases: list[dict], recommender, judge) -> list[dict]:
    rows = []
    for case in cases:
        result = recommender.recommend(case["text"])
        if result.game is None:
            rows.append(case | {"got": "", "score": None, "rationale": result.follow_up or result.reason})
            continue
        verdict = judge.relevance(case["text"], result.game)
        rows.append(case | {"got": result.game.name, "score": verdict.score, "rationale": verdict.rationale})
    return rows


def _mean_score(rows: list[dict], split: str) -> float:
    scores = [r["score"] for r in rows if r["split"] == split and r["score"] is not None]
    return mean(scores) if scores else float("nan")


def relevance_suite(cases: list[dict], recommender, judge) -> SuiteResult:
    rows = score_picks(cases, recommender, judge)
    metrics = {
        "mean_relevance_dev": _mean_score(rows, "dev"),
        "mean_relevance_test": _mean_score(rows, "test"),
        "no_pick": float(sum(r["score"] is None for r in rows)),
    }
    return SuiteResult(rows, metrics)


def tune_suite(cases: list[dict], make_recommender, judge, base: Weights = Weights()) -> SuiteResult:
    """Grid over (similarity, rating) on dev only. Test is scored once, for the chosen weights.

    Ties go to the setting closest to the current defaults, so a flat grid changes nothing.
    """
    dev = [c for c in cases if c["split"] == "dev"]
    test = [c for c in cases if c["split"] == "test"]
    rows: list[dict] = []
    metrics: dict[str, float] = {}
    dev_means: dict[tuple[float, float], float] = {}
    for similarity in GRID_SIMILARITY:
        for rating in GRID_RATING:
            scored = score_picks(dev, make_recommender(replace(base, similarity=similarity, rating=rating)), judge)
            dev_means[(similarity, rating)] = _mean_score(scored, "dev")
            metrics[f"dev_s{similarity}_r{rating}"] = dev_means[(similarity, rating)]
            rows += [r | {"similarity": similarity, "rating": rating} for r in scored]

    def rank(setting: tuple[float, float]) -> tuple[float, float]:
        value = dev_means[setting]
        distance = abs(setting[0] - base.similarity) + abs(setting[1] - base.rating)
        return (value if value == value else -1.0, -distance)

    chosen = max(dev_means, key=rank)
    test_rows = score_picks(test, make_recommender(replace(base, similarity=chosen[0], rating=chosen[1])), judge)
    rows += [r | {"similarity": chosen[0], "rating": chosen[1]} for r in test_rows]
    metrics |= {
        "chosen_similarity": chosen[0],
        "chosen_rating": chosen[1],
        "dev_mean_default": dev_means.get((base.similarity, base.rating), float("nan")),
        "dev_mean_chosen": dev_means[chosen],
        "test_mean_chosen": _mean_score(test_rows, "test"),
    }
    return SuiteResult(rows, metrics)
