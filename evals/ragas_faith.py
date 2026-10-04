# evals/ragas_faith.py
"""Ragas faithfulness scored by the judge model. Optional: uv run --group ragas python -m evals.run ragas"""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path

from board_game_reco.llm import DOTENV, GROQ_URL, JUDGE_MODEL, load_dotenv
from evals.judge import CACHE, Pacer, cache_key

RAGAS_VERSION = "ragas-faithfulness-v1"
# Ragas makes about two model calls per score: one splits the answer into claims, one checks them.
TOKENS_PER_SCORE = 3000


def _ragas_score(question: str, response: str, contexts: list[str]) -> float:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    async def run() -> float:
        client = AsyncOpenAI(api_key=os.environ["LLM_API_KEY"],
                             base_url=os.environ.get("LLM_BASE_URL", GROQ_URL), max_retries=3)
        scorer = Faithfulness(llm=llm_factory(os.environ.get("JUDGE_MODEL", JUDGE_MODEL), client=client))
        result = await scorer.ascore(user_input=question, response=response, retrieved_contexts=contexts)
        return float(result.value)

    return asyncio.run(run())


class RagasFaithfulness:
    def __init__(self, score: Callable[[str, str, list[str]], float] = _ragas_score, cache_dir: Path = CACHE,
                 pacer: Pacer | None = None) -> None:
        self._score = score
        self._cache = Path(cache_dir)
        self._pacer = pacer or Pacer()
        self.calls = 0

    @classmethod
    def from_env(cls) -> RagasFaithfulness | None:
        load_dotenv(DOTENV)
        return cls() if os.environ.get("LLM_API_KEY") else None

    def __call__(self, question: str, response: str, contexts: list[str]) -> float:
        path = self._cache / f"{cache_key(RAGAS_VERSION, question, response, contexts)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        self._pacer.wait(TOKENS_PER_SCORE)
        value = self._score(question, response, contexts)
        self._pacer.record(TOKENS_PER_SCORE)
        self.calls += 1
        self._cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"value": value}), encoding="utf-8")
        return value
