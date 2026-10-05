# tests/test_perf.py
from types import SimpleNamespace

import pytest

from evals.perf import basic_texts, percentile, perf_suite, top1_agreement


def game(i):
    return SimpleNamespace(id=i, name=f"game {i}")


class Fixed:
    def __init__(self, picks=None, engines=None) -> None:
        self.picks = picks or {}
        self.engines = engines or {}

    def recommend(self, text):
        return SimpleNamespace(game=self.picks.get(text, game(0)), engine=self.engines.get(text, "rules"))


def test_percentile_uses_linear_interpolation():
    assert percentile([1, 2, 3, 4, 100], 50) == 3.0
    assert percentile([1, 2, 3, 4, 100], 95) == pytest.approx(80.8)


def test_basic_texts_cycle_the_gold_queries():
    rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert basic_texts(rows, 7) == ["a", "b", "c", "a", "b", "c", "a"]


def test_top1_agreement_lists_every_disagreement():
    agreement, rows = top1_agreement(["x", "y"], Fixed({"y": game(1)}), Fixed({"y": game(2)}))
    assert agreement == 0.5
    assert [r["agree"] for r in rows] == [True, False]
    assert rows[1]["int8"] == "game 1" and rows[1]["fp32"] == "game 2"


def test_perf_suite_reports_basic_cloud_and_quantization_metrics():
    slept = []
    cloud = Fixed(engines={"th 1": "cloud", "th 2": "rules"})
    result = perf_suite(
        gold_parse=[{"text": "q1"}, {"text": "q2"}],
        agreement_texts=["q1", "q2"],
        basic=Fixed(),
        cloud=cloud,
        fp32=Fixed({"q2": game(9)}),
        escalated=["th 1", "th 2"],
        sleep=slept.append,
    )
    metrics = result.metrics
    assert metrics["basic_n"] == 100.0
    assert metrics["basic_p95_ms"] >= metrics["basic_p50_ms"] >= 0.0
    assert metrics["cloud_n"] == 1.0 and metrics["cloud_fallbacks"] == 1.0
    assert metrics["int8_fp32_top1_agreement"] == 0.5 and metrics["disagreements"] == 1.0
    assert slept == [12.0, 12.0]


def test_perf_suite_skips_cloud_and_quantization_when_not_given():
    metrics = perf_suite([{"text": "q1"}], [], Fixed(), None, None, []).metrics
    assert "cloud_n" not in metrics and "int8_fp32_top1_agreement" not in metrics
