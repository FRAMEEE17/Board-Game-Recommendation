# evals/perf.py
"""N3 Basic latency, N4 cloud latency, N13 int8 against fp32 top-1 agreement."""
from __future__ import annotations

import time

import numpy as np

from board_game_reco.intent import script_language
from board_game_reco.rules import parse as rule_parse
from evals.suites import SuiteResult

# Seconds between escalated requests. Each makes about three app-model calls, about 1.5K tokens,
# and the app model allows 8K tokens a minute on Groq's free tier.
CLOUD_PACE = 12.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def basic_texts(gold_parse: list[dict], n: int = 100) -> list[str]:
    texts = [g["text"] for g in gold_parse]
    return [texts[i % len(texts)] for i in range(n)]


def escalated_texts(gold_parse: list[dict], catalog, limit: int = 20) -> list[str]:
    """Recommend-intent gold rows the rules cannot read in full: the ones that go to the cloud model."""
    texts = []
    for g in gold_parse:
        if g["intent"] != "recommend":
            continue
        unread = rule_parse(g["text"], is_game=catalog.has_name, find_designer=catalog.find_designer).unread
        if script_language(g["text"]) != "latin" or unread:
            texts.append(g["text"])
    return texts[:limit]


def timed(fn, texts: list[str]) -> list[float]:
    out = []
    for text in texts:
        start = time.perf_counter()
        fn(text)
        out.append((time.perf_counter() - start) * 1000)
    return out


def top1_agreement(texts: list[str], int8, fp32) -> tuple[float, list[dict]]:
    rows = []
    for text in texts:
        a, b = int8.recommend(text).game, fp32.recommend(text).game
        rows.append({"query": text, "int8": a.name if a else "", "fp32": b.name if b else "",
                     "agree": (a.id if a else None) == (b.id if b else None)})
    return (sum(r["agree"] for r in rows) / len(rows) if rows else float("nan")), rows


def perf_suite(gold_parse: list[dict], agreement_texts: list[str], basic, cloud, fp32, escalated: list[str],
               pace: float = CLOUD_PACE, sleep=time.sleep) -> SuiteResult:
    """basic and fp32 run with no app model. cloud is None without a key, fp32 is None to skip N13.

    N3 is measured on the machine this runs on. The 2-CPU Docker measurement arrives in wave 2b.
    """
    texts = basic_texts(gold_parse, 100)
    for text in texts[:5]:
        basic.recommend(text)  # warm-up, not measured
    basic_ms = timed(basic.recommend, texts)
    rows = [{"kind": "basic", "query": t, "latency_ms": round(ms, 2)} for t, ms in zip(texts, basic_ms)]
    metrics = {"basic_n": float(len(basic_ms)), "basic_p50_ms": percentile(basic_ms, 50),
               "basic_p95_ms": percentile(basic_ms, 95)}

    if cloud is not None:
        cloud_ms = []
        for text in escalated:
            start = time.perf_counter()
            result = cloud.recommend(text)
            elapsed = (time.perf_counter() - start) * 1000
            rows.append({"kind": "cloud", "query": text, "latency_ms": round(elapsed, 2), "engine": result.engine})
            if result.engine == "cloud":
                cloud_ms.append(elapsed)
            sleep(pace)
        metrics["cloud_n"] = float(len(cloud_ms))
        metrics["cloud_fallbacks"] = float(len(escalated) - len(cloud_ms))
        if cloud_ms:
            metrics["cloud_p50_ms"] = percentile(cloud_ms, 50)
            metrics["cloud_p95_ms"] = percentile(cloud_ms, 95)

    if fp32 is not None:
        agreement, agree_rows = top1_agreement(agreement_texts, basic, fp32)
        rows += [r | {"kind": "quant"} for r in agree_rows]
        metrics["int8_fp32_top1_agreement"] = agreement
        metrics["disagreements"] = float(sum(not r["agree"] for r in agree_rows))
        for r in agree_rows:
            if not r["agree"]:
                print(f'int8 and fp32 disagree on "{r["query"]}": int8 {r["int8"] or "-"}, fp32 {r["fp32"] or "-"}')
    return SuiteResult(rows, metrics)
