"""Suites that need the judge: relevance, weight tuning, judge quality and reason faithfulness."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import mean

from board_game_reco.recommender import Weights, grounded
from evals.judge import game_facts
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


def judge_suite(gold_judge: list[dict], catalog, judge, out_dir: Path, runs: int = 2) -> SuiteResult:
    """How well the judge separates faithful from corrupted reasons, against the grounded() check.

    Corrupted is the positive class. The judge's suspicion is 6 minus its faithfulness score.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    games = {g.id: g for g in catalog.games}
    rows = []
    for r in gold_judge:
        game = games[r["game_id"]]
        facts = game_facts(game)
        row = {"id": r["id"], "game": game.name, "label": r["label"], "corruption": r["corruption"],
               "corrupted": int(r["label"] == "corrupted"),
               "grounded_flags": int(not grounded(r["reason"], game, facts))}
        for run in range(runs):
            verdict = judge.faithfulness(facts, r["reason"], run=run)
            row[f"score_run{run}"] = verdict.score
            row[f"rationale_run{run}"] = verdict.rationale
        rows.append(row)

    y = [r["corrupted"] for r in rows]
    metrics: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(5, 5))
    for run in range(runs):
        suspicion = [6 - r[f"score_run{run}"] for r in rows]
        auc = float(roc_auc_score(y, suspicion))
        metrics[f"judge_auroc_run{run}"] = auc
        fpr, tpr, _ = roc_curve(y, suspicion)
        ax.plot(fpr, tpr, label=f"judge run {run + 1} (AUROC {auc:.2f})")
    positives = sum(y)
    negatives = len(y) - positives
    tpr_point = sum(r["grounded_flags"] for r in rows if r["corrupted"]) / positives
    fpr_point = sum(r["grounded_flags"] for r in rows if not r["corrupted"]) / negatives
    metrics |= {"grounded_tpr": tpr_point, "grounded_fpr": fpr_point}
    ax.scatter([fpr_point], [tpr_point], marker="s", s=60, color="black", zorder=3,
               label=f"grounded() check (TPR {tpr_point:.2f}, FPR {fpr_point:.2f})")
    if runs >= 2:
        differences = [abs(r["score_run0"] - r["score_run1"]) for r in rows]
        metrics["score_mean_abs_diff"] = mean(differences)
        metrics["score_changed_share"] = sum(d > 0 for d in differences) / len(differences)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate (faithful reason flagged)")
    ax.set_ylabel("True positive rate (corrupted reason flagged)")
    ax.legend(loc="lower right")
    path = Path(out_dir) / "judge_roc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return SuiteResult(rows, metrics, {"judge_roc": path})
