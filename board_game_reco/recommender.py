# board_game_reco/recommender.py
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .catalog import COOP, SOLO, Catalog, Game, HardConstraints, satisfies
from .intent import IntentClassifier, Request, parse
from .llm import LLM

ROOT = Path(__file__).resolve().parent.parent
NEW_SINCE = 2023
BANDS = {"light": (0.0, 2.0), "medium": (2.0, 3.0), "heavy": (3.0, 5.0)}
LANGUAGES = {"th": "Thai", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
TERM_LABELS = {
    "similarity": "how closely it matches what you asked for",
    "rating": "its BGG rating, weighted by how many people rated it",
    "weight_band": "its weight",
    "coop": "being co-operative",
    "solo": "its solo mode",
    "new": "being a recent release",
}
CONSTRAINT_LABELS = {
    "players": "player count",
    "max_minutes": "playtime",
    "youngest_age": "age",
    "same_family": "same family as the game you named",
    "designer": "designer",
}
DECLINES = {
    "off_topic": "I can only help with board games. Tell me who is playing, how long you have, or a game you like.",
    "injection": "I can't help with that request. Ask me for a board game instead.",
    "compare": "Comparing two games is not available yet. Tell me what you'd like to play and I'll pick one.",
    "another": "Picking another game is not available yet. Describe what you'd like and I'll pick again.",
}
REASON_SYSTEM = (
    "You write two or three sentences recommending one board game. Use only the facts given. "
    "Name the game. Do not add numbers that are not in the facts. Write in {language}."
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Weights:
    """Every scoring weight in one place (F13). Negative weights push games down."""

    similarity: float = 1.0
    rating: float = 0.4
    weight_band: float = 0.3
    coop: float = 0.3
    avoid_coop: float = -1.0
    solo: float = 0.2
    new: float = 0.3
    # MMR trade-off for runners-up: 0 ranks by score alone, 1 by difference from games already chosen.
    diversity: float = 0.3


@dataclass(frozen=True)
class Scored:
    game: Game
    total: float
    terms: dict[str, float]


@dataclass(frozen=True)
class Recommendation:
    game: Game | None
    reason: str
    relaxed: tuple[str, ...] = ()
    runners_up: tuple[Game, ...] = ()
    follow_up: str | None = None
    engine: str = "rules"
    terms: dict[str, float] = field(default_factory=dict)
    trace: dict[str, object] = field(default_factory=dict)


class Recommender:
    def __init__(self, catalog: Catalog, classifier, llm: LLM | None, weights: Weights = Weights()) -> None:
        self._catalog = catalog
        self._classifier = classifier
        self._llm = llm
        self._weights = weights

    @classmethod
    def default(cls) -> Recommender:
        from .embedder import Embedder

        encoder = Embedder.default()
        catalog = Catalog.load(ROOT / "data" / "boardgames.csv", ROOT / ".cache" / "vectors", encoder)
        return cls(catalog, IntentClassifier(encoder), LLM.from_env())

    @property
    def llm(self) -> LLM | None:
        return self._llm

    def recommend(self, text: str) -> Recommendation:
        """Relaxes only playtime (x1.5, then dropped). Players and age are never relaxed.

        The pick is re-checked against the hard constraints before returning. A model-written
        reason that does not name the pick or adds numbers falls back to the template reason.
        """
        request = parse(text, catalog=self._catalog, classifier=self._classifier, llm=self._llm)
        if request.intent in DECLINES:
            return Recommendation(None, DECLINES[request.intent], engine=request.engine)
        if request.intent == "unclear":
            return Recommendation(None, "", follow_up=_clarify(request), engine=request.engine)

        anchor = self._catalog.find(request.anchor) if request.anchor else None
        if request.anchor and anchor is None:
            question = f'I could not find "{request.anchor}" in the catalog. Could you check the name?'
            return Recommendation(None, "", follow_up=question, engine=request.engine)

        pool, removed, hard, relaxed = self._pool(request, anchor)
        trace: dict[str, object] = {
            "catalog": len(self._catalog.games), "eligible": len(pool), "removed": removed, "engine": request.engine,
        }
        if not pool:
            return Recommendation(None, "", follow_up=_no_match(removed), engine=request.engine, trace=trace)

        ranked = self._rank(request, anchor, pool)
        pick = next(s for s in ranked if satisfies(s.game, hard) is None)  # F31
        runners = self._runners_up(pick, ranked)
        reason = self._reason(request, anchor, pick, runners, relaxed, hard)
        follow_up = _relax_question(request.hard, hard, len(pool)) if relaxed else None
        trace |= {"scored": len(ranked), "picked": 1}
        return Recommendation(pick.game, reason, relaxed, tuple(r.game for r in runners), follow_up,
                              request.engine, pick.terms, trace)

    def _pool(self, request: Request, anchor: Game | None):
        first_removed: dict[str, int] | None = None
        for relaxed, minutes in _minute_steps(request.hard.max_minutes):
            hard = replace(request.hard, max_minutes=minutes)
            result = self._catalog.eligible(hard)
            kept, removed = list(result.kept), dict(result.removed)
            if anchor is not None:
                before = len(kept)
                kept = [g for g in kept if not self._catalog.same_family(anchor, g)]
                removed["same_family"] = before - len(kept)
            if request.designer:
                before = len(kept)
                kept = [g for g in kept if request.designer in g.designers]
                removed["designer"] = before - len(kept)
            first_removed = first_removed or removed
            if kept:
                return kept, removed, hard, relaxed
        return [], first_removed, request.hard, ()

    def _rank(self, request: Request, anchor: Game | None, pool: list[Game]) -> list[Scored]:
        similarities = self._catalog.similarity(anchor or request.text, pool)
        scored = []
        for game, similarity in zip(pool, similarities):
            terms = self._terms(request, game, float(similarity))
            scored.append(Scored(game, sum(terms.values()), terms))
        return sorted(scored, key=lambda s: (-s.total, s.game.id))

    def _terms(self, request: Request, game: Game, similarity: float) -> dict[str, float]:
        w = self._weights
        span = (self._catalog.rating_max - self._catalog.rating_min) or 1.0
        terms = {
            "similarity": w.similarity * similarity,
            "rating": w.rating * (self._catalog.adjusted_rating(game) - self._catalog.rating_min) / span,
        }
        if request.weight:
            low, high = BANDS[request.weight]
            distance = 0.0 if low <= game.complexity <= high else min(abs(game.complexity - low), abs(game.complexity - high))
            terms["weight_band"] = w.weight_band * (1.0 - distance)
        if request.coop is True and COOP in game.mechanics:
            terms["coop"] = w.coop
        if request.coop is False and COOP in game.mechanics:
            terms["coop"] = w.avoid_coop
        if request.solo and SOLO in game.mechanics:
            terms["solo"] = w.solo
        if request.wants_new and game.year is not None and game.year >= NEW_SINCE:
            terms["new"] = w.new
        return terms

    def _runners_up(self, pick: Scored, ranked: list[Scored], limit: int = 2, pool_size: int = 20) -> list[Scored]:
        """Maximal marginal relevance over the top candidates, at most one game per family (F21)."""
        diversity = self._weights.diversity
        chosen = [pick]
        candidates = list(ranked[1:pool_size])
        while len(chosen) <= limit:
            open_ = [c for c in candidates if all(
                not self._catalog.same_family(x.game, c.game) and not self._catalog.same_family(c.game, x.game)
                for x in chosen)]
            if not open_:
                break
            redundancy = {c.game.id: float(self._catalog.similarity(c.game, [x.game for x in chosen]).max())
                          for c in open_}
            best = max(open_, key=lambda c: (1 - diversity) * c.total - diversity * redundancy[c.game.id])
            chosen.append(best)
            candidates.remove(best)
        return chosen[1:]

    def _reason(self, request: Request, anchor: Game | None, pick: Scored, runners: list[Scored],
                relaxed: tuple[str, ...], hard: HardConstraints) -> str:
        template = _template(anchor, pick, runners, relaxed, hard)
        if self._llm is None or request.engine != "cloud":
            return template
        language = LANGUAGES.get(request.language, "the same language as this request: " + request.text)
        written = self._llm.text(REASON_SYSTEM.format(language=language), f"Game: {pick.game.name}\n{template}")
        if written and grounded(written, pick.game, template + " " + request.text):
            return f"{written} {pick.game.url}"
        return template


def grounded(reason: str, game: Game, allowed_text: str) -> bool:
    """F32 and F23: the reason names the game, and every number in it appears in the allowed text."""
    if game.name.casefold() not in reason.casefold():
        return False
    allowed = set(_NUMBER.findall(allowed_text.replace(",", "")))
    return set(_NUMBER.findall(reason.replace(",", ""))) <= allowed


def _minute_steps(minutes: int | None):
    if minutes is None:
        return [((), None)]
    return [((), minutes), (("max_minutes x1.5",), round(minutes * 1.5)), (("max_minutes dropped",), None)]


def _template(anchor: Game | None, pick: Scored, runners: list[Scored], relaxed: tuple[str, ...],
              hard: HardConstraints) -> str:
    g = pick.game
    sentences = [f"{g.name} ({g.year or 'year unknown'}) plays {g.min_players}-{g.max_players} players"
                 + (f" in about {g.max_minutes} minutes." if g.max_minutes else ".")]
    facts = [f"weight {g.complexity:.1f} out of 5", f"rated {g.rating:.1f} by {g.num_ratings:,} people"]
    if g.min_age:
        facts.insert(0, f"ages {g.min_age}+")
    sentences.append(facts[0].capitalize() + ", " + ", ".join(facts[1:]) + ".")
    if anchor is not None:
        sentences.append(f"It is the closest match to {anchor.name} outside its family.")
    elif pick.terms.get("similarity", 0) > 0:
        sentences.append("It is the closest match to what you described.")
    if "coop" in pick.terms and pick.terms["coop"] > 0:
        sentences.append("It is co-operative.")
    if "new" in pick.terms:
        sentences.append(f"It came out in {g.year}.")
    if runners:
        runner = runners[0]
        edge = max(pick.terms, key=lambda k: pick.terms[k] - runner.terms.get(k, 0.0))
        sentences.append(f"It edged out {runner.game.name} mainly on {TERM_LABELS[edge]}.")
    if "max_minutes x1.5" in relaxed:
        sentences.append(f"Nothing fit your time limit, so I allowed up to {hard.max_minutes} minutes.")
    if "max_minutes dropped" in relaxed:
        sentences.append("Nothing fit your time limit even with extra time, so I ignored the time limit.")
    sentences.append(g.url)
    return " ".join(sentences)


def _clarify(request: Request) -> str:
    quoted = " / ".join(request.unread)
    return (f'I could not read this part: "{quoted}". Could you say it another way in English, '
            "or add an API key so I can read other languages?")


def _no_match(removed: dict[str, int]) -> str:
    parts = [f"{CONSTRAINT_LABELS[k]} removed {v:,}" for k, v in removed.items() if v]
    return "No game in the catalog fits all of that (" + "; ".join(parts) + "). Which one could you loosen?"


def _relax_question(asked: HardConstraints, used: HardConstraints, fitting: int) -> str:
    who = f"{asked.players} players" if asked.players else "your group"
    if used.max_minutes is not None:
        return (f"No game fits {who} in {asked.max_minutes} min. {fitting} fit in {used.max_minutes} min. "
                "Would that work?")
    return f"No game fits {who} in {asked.max_minutes} min. {fitting} fit with no time limit. Would that work?"
