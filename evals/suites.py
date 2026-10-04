# evals/suites.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from board_game_reco.catalog import HardConstraints, satisfies
from board_game_reco.intent import parse, script_language
from board_game_reco.rules import parse as rule_parse
from evals.gold import FIELDS


@dataclass
class SuiteResult:
    rows: list[dict]
    metrics: dict[str, float]
    figures: dict[str, Path] = field(default_factory=dict)


def parse_suite(gold: list[dict], catalog, classifier, llm) -> SuiteResult:
    rows = []
    for g in gold:
        request = parse(g["text"], catalog=catalog, classifier=classifier, llm=llm)
        got = {"players": request.hard.players, "max_minutes": request.hard.max_minutes,
               "youngest_age": request.hard.youngest_age, "weight": request.weight,
               "anchor": request.anchor, "designer": request.designer}
        for f in FIELDS:
            rows.append({"query": g["text"], "lang": g["lang"], "split": g["split"], "field": f,
                         "expected": g[f], "got": got[f], "pass": got[f] == g[f], "engine": request.engine})
    metrics = {}
    for lang in ("en", "th", "other"):
        subset = [r for r in rows if r["lang"] == lang and r["split"] == "test"]
        if subset:
            metrics[f"field_accuracy_{lang}"] = sum(r["pass"] for r in subset) / len(subset)
    return SuiteResult(rows, metrics)


def route_suite(gold: list[dict], catalog, classifier, out_dir: Path) -> SuiteResult:
    """A rules miss is any gold field the rules got wrong. Signals should flag every miss."""
    from sklearn.metrics import roc_auc_score, roc_curve

    rows = []
    for g in gold:
        if g["intent"] == "injection":
            continue
        rules = rule_parse(g["text"], is_game=catalog.has_name, find_designer=catalog.find_designer)
        got = {"players": rules.players, "max_minutes": rules.max_minutes, "youngest_age": rules.youngest_age,
               "weight": rules.weight or ("light" if rules.first_time else None),
               "anchor": rules.anchor, "designer": rules.designer}
        miss = any(got[f] != g[f] for f in FIELDS) or g["intent"] == "unclear"
        _, confidence = classifier.classify(g["text"])
        rows.append({"query": g["text"], "split": g["split"], "miss": int(miss),
                     "unread": int(bool(rules.unread)),
                     "non_latin": int(script_language(g["text"]) != "latin"),
                     "low_confidence": 1.0 - confidence})
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]

    threshold = _confidence_threshold(dev)
    for r in rows:
        r["escalate"] = int(r["unread"] or r["non_latin"] or (1.0 - r["low_confidence"]) < threshold)

    metrics = {"confidence_threshold": threshold}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for signal in ("unread", "non_latin", "low_confidence"):
        y, s = [r["miss"] for r in test], [r[signal] for r in test]
        if len(set(y)) == 2:
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            metrics[f"auroc_{signal}"] = auc
            ax.plot(fpr, tpr, label=f"{signal} (AUROC {auc:.2f})")
    combined = [max(r["unread"], r["non_latin"], r["low_confidence"]) for r in test]
    y = [r["miss"] for r in test]
    if len(set(y)) == 2:
        fpr, tpr, _ = roc_curve(y, combined)
        metrics["auroc_combined"] = roc_auc_score(y, combined)
        ax.plot(fpr, tpr, label=f"combined (AUROC {metrics['auroc_combined']:.2f})", linewidth=2)
        caught = [r["escalate"] for r in test if r["miss"]]
        metrics["test_miss_recall"] = sum(caught) / len(caught)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate (escalated when rules were right)")
    ax.set_ylabel("True positive rate (escalated when rules missed)")
    ax.legend(loc="lower right")
    path = out_dir / "route_roc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return SuiteResult(rows, metrics, {"route_roc": path})


def _confidence_threshold(dev: list[dict], target_recall: float = 0.95) -> float:
    """Lowest confidence cut that still escalates at least 95% of dev misses."""
    misses = [r for r in dev if r["miss"]]
    if not misses:
        return 0.0
    for cut in np.round(np.arange(0.0, 0.5, 0.005), 3):
        caught = [r for r in misses if r["unread"] or r["non_latin"] or (1.0 - r["low_confidence"]) < cut]
        if len(caught) / len(misses) >= target_recall:
            return float(cut)
    return 0.5


def recommend_suite(gold_recommend: list[dict], gold_parse: dict[str, dict], recommender) -> SuiteResult:
    rows = []
    for row in gold_recommend:
        g = gold_parse[row["parse_id"]]
        result = recommender.recommend(g["text"])
        picked = result.game
        hard_ok = picked is None or satisfies(picked, HardConstraints(g["players"], None, g["youngest_age"])) is None
        family_ok = picked is None or not row["must_not_family"] or not recommender._catalog.same_family(
            recommender._catalog.find(row["must_not_family"]), picked)
        rows.append({"query": g["text"], "split": row["split"], "got": picked.name if picked else "",
                     "follow_up": result.follow_up or "", "engine": result.engine,
                     "pass": hard_ok and family_ok})
    metrics = {"constraint_satisfaction": sum(r["pass"] for r in rows) / len(rows)}
    return SuiteResult(rows, metrics)
