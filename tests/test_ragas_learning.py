# tests/test_ragas_learning.py
"""Learning test: does Ragas faithfulness run against Groq's endpoint with the judge model?

Run by hand: uv run --group ragas pytest tests/test_ragas_learning.py -m network -s
"""
import asyncio
import os

import pytest

pytestmark = pytest.mark.network

FACTS = ("Pandemic (2008) plays 2-4 players in about 45 minutes. "
         "Mechanics: Action Points, Cooperative Game, Hand Management.")


def test_ragas_faithfulness_scores_with_the_groq_judge():
    pytest.importorskip("ragas")
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    from board_game_reco.llm import DOTENV, GROQ_URL, JUDGE_MODEL, load_dotenv

    load_dotenv(DOTENV)
    key = os.environ.get("LLM_API_KEY")
    if not key:
        pytest.skip("LLM_API_KEY not set")

    async def both() -> tuple[float, float]:
        client = AsyncOpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL", GROQ_URL), max_retries=3)
        scorer = Faithfulness(llm=llm_factory(os.environ.get("JUDGE_MODEL", JUDGE_MODEL), client=client))
        good = await scorer.ascore(user_input="a cooperative game for 4",
                                   response="Pandemic is a cooperative game for 2-4 players that takes about 45 minutes.",
                                   retrieved_contexts=[FACTS])
        bad = await scorer.ascore(user_input="a cooperative game for 4",
                                  response="Pandemic is a competitive game for 6 players that takes three hours.",
                                  retrieved_contexts=[FACTS])
        return float(good.value), float(bad.value)

    good, bad = asyncio.run(both())
    print("ragas faithfulness, faithful answer:", good, "unfaithful answer:", bad)
    assert 0.0 <= bad < good <= 1.0
