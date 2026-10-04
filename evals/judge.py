"""The evaluation judge: qwen/qwen3.8-27b on Groq in thinking mode, paced, retried and cached.

The transport lives in board_game_reco/llm.py, the only module that knows the vendor. This module
owns what only an eval run needs: the prompts, a disk cache that makes reruns free, pacing under
Groq's free tier (8K tokens and 30 requests per minute) and the retry policy.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from board_game_reco.catalog import Game
from board_game_reco.llm import Retryable, judge_client, judge_json

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "judge"
TOKENS_PER_MINUTE = 8000
REQUESTS_PER_MINUTE = 30
# A thinking call's length is unknown before it runs. This is added to the prompt estimate.
COMPLETION_ESTIMATE = 1000
MAX_ATTEMPTS = 5
# Groq asks for minutes or hours once the daily token quota is spent. Past this, stop and rerun later.
LONGEST_WAIT = 120.0
# Bump a version whenever its prompt changes, so cached scores from the old prompt are not reused.
RELEVANCE_VERSION = "relevance-v1"
FAITHFULNESS_VERSION = "faithfulness-v1"

SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "rationale"],
    "properties": {"score": {"type": "integer"}, "rationale": {"type": "string"}},
}

RELEVANCE_SYSTEM = """You judge one board game recommendation.
Read the request and the facts about the recommended game. Rate how well the game fits the request:
5: fits every part of the request and is a strong example of what was asked for.
4: fits the request with a small gap.
3: fits part of the request.
2: shares only a surface feature with the request, such as a theme word.
1: does not fit the request.
Use only the facts given. Give a one-line rationale."""

FAITHFULNESS_SYSTEM = """You check a short board game recommendation against the facts about the game.
Rate how faithful the recommendation is to the facts:
5: every claim is stated in the facts.
4: every claim is supported, with loose wording.
3: one claim is not in the facts but does not contradict them.
2: one claim contradicts the facts.
1: several claims contradict the facts or are invented.
A number that differs from the facts is a contradiction. Give a one-line rationale."""


class DailyLimit(Exception):
    """The server asked for a wait longer than LONGEST_WAIT. Rerun later: cached scores are kept."""


@dataclass(frozen=True)
class Verdict:
    score: int
    rationale: str
    cached: bool


class Pacer:
    """Sliding 60-second window over requests and tokens, kept under Groq's free-tier limits."""

    def __init__(self, tokens_per_minute: int = TOKENS_PER_MINUTE, requests_per_minute: int = REQUESTS_PER_MINUTE,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self._tpm = tokens_per_minute
        self._rpm = requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._window: deque[tuple[float, int]] = deque()

    def wait(self, estimate: int) -> None:
        while True:
            now = self._clock()
            while self._window and now - self._window[0][0] >= 60.0:
                self._window.popleft()
            used = sum(tokens for _, tokens in self._window)
            if not self._window or (len(self._window) < self._rpm and used + estimate <= self._tpm):
                return
            self._sleep(max(self._window[0][0] + 60.0 - now, 0.01))

    def record(self, tokens: int) -> None:
        self._window.append((self._clock(), tokens))


def game_facts(game: Game) -> str:
    """Everything a reason may quote about a game, in the number formats the template uses."""
    parts = [
        f"{game.name} ({game.year or 'year unknown'}) plays {game.min_players}-{game.max_players} players"
        + (f" in about {game.max_minutes} minutes." if game.max_minutes else "."),
        (f"Ages {game.min_age}+. " if game.min_age else "")
        + f"Weight {game.complexity:.1f} out of 5. Rated {game.rating:.1f} by {game.num_ratings:,} people.",
        "Categories: " + ", ".join(game.categories) + ".",
        "Mechanics: " + ", ".join(game.mechanics) + ".",
        "Designers: " + ", ".join(game.designers) + ".",
        "BGG ranks it as a family game." if game.family_rank is not None else "",
        game.url,
    ]
    return " ".join(part for part in parts if part)


def cache_key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()[:32]


class Judge:
    def __init__(self, call: Callable[[str, str, dict, str], tuple[dict, int]], cache_dir: Path = CACHE,
                 pacer: Pacer | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self._call = call
        self._cache = Path(cache_dir)
        self._pacer = pacer or Pacer()
        self._sleep = sleep
        self.calls = 0
        self.hits = 0
        self.tokens = 0

    @classmethod
    def from_env(cls, cache_dir: Path = CACHE) -> Judge | None:
        client = judge_client()
        if client is None:
            return None
        return cls(lambda system, user, schema, name: judge_json(client, system, user, schema, name), cache_dir)

    def relevance(self, query: str, game: Game) -> Verdict:
        user = f"Request: {query}\n\nRecommended game:\n{game_facts(game)}\n\nDescription: {game.description[:400]}"
        return self._score(RELEVANCE_SYSTEM, user, "relevance", cache_key(RELEVANCE_VERSION, query, game.id))

    def faithfulness(self, facts: str, reason: str, run: int = 0) -> Verdict:
        """run separates repeated scorings of the same pair, so the judge suite can measure spread."""
        user = f"Facts:\n{facts}\n\nRecommendation:\n{reason}"
        return self._score(FAITHFULNESS_SYSTEM, user, "faithfulness",
                           cache_key(FAITHFULNESS_VERSION, facts, reason, run))

    def _score(self, system: str, user: str, name: str, key: str) -> Verdict:
        path = self._cache / f"{key}.json"
        if path.exists():
            self.hits += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            return Verdict(data["score"], data["rationale"], cached=True)
        reply = self._call_with_retry(system, user, name)
        score = reply.get("score")
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError(f"judge score out of range: {score!r}")
        self._cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"score": score, "rationale": reply["rationale"]}, ensure_ascii=False),
                        encoding="utf-8")
        return Verdict(score, reply["rationale"], cached=False)

    def _call_with_retry(self, system: str, user: str, name: str) -> dict:
        estimate = (len(system) + len(user)) // 4 + COMPLETION_ESTIMATE
        for attempt in range(MAX_ATTEMPTS):
            self._pacer.wait(estimate)
            try:
                reply, tokens = self._call(system, user, SCORE_SCHEMA, name)
            except Retryable as error:
                self._pacer.record(estimate)
                wait = error.retry_after if error.retry_after is not None else 5.0 * 2**attempt
                if wait > LONGEST_WAIT:
                    raise DailyLimit(f"Groq asked to wait {wait:.0f} s. Rerun later; cached scores are kept.") from error
                self._sleep(wait)
                continue
            self._pacer.record(tokens or estimate)
            self.calls += 1
            self.tokens += tokens
            return reply
        raise RuntimeError(f"judge call failed after {MAX_ATTEMPTS} attempts")
