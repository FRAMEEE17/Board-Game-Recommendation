# tests/test_quality.py
import math
from types import SimpleNamespace

import pytest

from evals.gold import load
from evals.judge import Verdict
from evals.quality import explain_suite, judge_suite, relevance_cases, relevance_suite, tune_suite


class FakeJudge:
    def __init__(self, relevance=None, faithfulness=None) -> None:
        self._relevance = relevance or (lambda query, game: 3)
        self._faithfulness = faithfulness or (lambda facts, reason, run: 5)
        self.calls = self.hits = self.tokens = 0

    def relevance(self, query, game):
        self.calls += 1
        return Verdict(self._relevance(query, game), "fake", False)

    def faithfulness(self, facts, reason, run=0):
        self.calls += 1
        return Verdict(self._faithfulness(facts, reason, run), "fake", False)


class FakeRecommender:
    def __init__(self, picks) -> None:
        self._picks = picks

    def recommend(self, text):
        game = self._picks(text) if callable(self._picks) else self._picks[text]
        return SimpleNamespace(game=game, follow_up=None if game else "which one?", reason="")


def test_relevance_cases_join_english_recommend_rows_and_relevance_rows():
    parse = {r["id"]: r for r in load("parse.jsonl")}
    cases = relevance_cases(load("recommend.jsonl"), parse, load("relevance.jsonl"))
    assert len(cases) == 40
    assert len({c["id"] for c in cases}) == 40
    assert {c["split"] for c in cases} == {"dev", "test"}


def test_relevance_suite_reports_mean_per_split_and_counts_rows_without_a_pick(catalog):
    pandemic = catalog.find("Pandemic")
    picks = {"dev a": pandemic, "dev b": pandemic, "test a": None}
    cases = [{"id": text, "text": text, "split": text.split()[0]} for text in picks]
    judge = FakeJudge(relevance=lambda query, game: 4 if query == "dev a" else 2)
    metrics = relevance_suite(cases, FakeRecommender(picks), judge).metrics
    assert metrics["mean_relevance_dev"] == 3.0
    assert math.isnan(metrics["mean_relevance_test"])
    assert metrics["no_pick"] == 1.0


def test_tuning_chooses_on_dev_and_scores_test_once(catalog):
    good, bad = catalog.find("Pandemic"), catalog.find("CATAN")
    cases = [{"id": "d1", "text": "dev one", "split": "dev"}, {"id": "t1", "text": "test one", "split": "test"}]

    def make(weights):
        return FakeRecommender(lambda text: good if weights.rating >= 0.8 else bad)

    def relevance(query, game):
        fits = game is good
        return 5 if fits != query.startswith("test") else 1  # test prefers the other game

    result = tune_suite(cases, make, FakeJudge(relevance=relevance))
    assert result.metrics["chosen_rating"] == 0.8
    assert result.metrics["chosen_similarity"] == 1.0  # tie on dev goes to the current default
    assert result.metrics["test_mean_chosen"] == 1.0
    assert sum(r["split"] == "test" for r in result.rows) == 1


def test_judge_suite_reports_auroc_per_run_the_grounded_point_and_the_spread(catalog, tmp_path):
    rows = load("judge.jsonl")
    corrupted = {r["reason"] for r in rows if r["label"] == "corrupted"}
    first_faithful = rows[0]["reason"]

    def score(facts, reason, run):
        if run == 1 and reason == first_faithful:
            return 4
        return 1 if reason in corrupted else 5

    result = judge_suite(rows, catalog, FakeJudge(faithfulness=score), tmp_path)
    metrics = result.metrics
    assert metrics["judge_auroc_run0"] == 1.0 and metrics["judge_auroc_run1"] == 1.0
    assert metrics["grounded_tpr"] == pytest.approx(5 / 15)
    assert metrics["grounded_fpr"] == 0.0
    assert metrics["score_changed_share"] == pytest.approx(1 / 30)
    assert (tmp_path / "judge_roc.png").exists()


def test_explain_suite_grounds_template_reasons_and_averages_faithfulness(catalog):
    from board_game_reco.recommender import Recommender
    from tests.conftest import StubClassifier

    cases = [
        {"id": "e1", "text": "something similar to Catan but shorter", "split": "dev"},
        {"id": "e2", "text": "a game for 12 players in 5 minutes", "split": "dev"},
        {"id": "e3", "text": "family game for 4, not too complicated", "split": "test"},
    ]
    recommender = Recommender(catalog, StubClassifier(), None)
    metrics = explain_suite(cases, recommender, catalog, FakeJudge()).metrics
    assert metrics["explained"] == 3.0
    assert metrics["grounding_pass_rate"] == 1.0
    assert metrics["mean_faithfulness"] == 5.0
    without_key = explain_suite(cases, recommender, catalog, None).metrics
    assert "mean_faithfulness" not in without_key and without_key["grounding_pass_rate"] == 1.0


def test_ragas_wrapper_caches_each_score(tmp_path):
    from evals.judge import Pacer
    from evals.ragas_faith import RagasFaithfulness

    calls = []

    def score(question, response, contexts):
        calls.append(question)
        return 0.75

    ragas = RagasFaithfulness(score, tmp_path, Pacer(sleep=lambda seconds: None))
    assert ragas("q", "r", ["facts"]) == 0.75
    assert ragas("q", "r", ["facts"]) == 0.75
    assert calls == ["q"]
