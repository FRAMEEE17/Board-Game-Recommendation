from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .catalog import Catalog, Encoder, HardConstraints
from .llm import LLM
from .rules import RuleParse, Weight
from .rules import parse as rule_parse

Intent = Literal["recommend", "compare", "another", "off_topic", "unclear", "injection"]

# Gates declining off-topic requests and the fixed compare/another replies. Below it the label is
# not trusted and the request is treated as a recommendation. It never decides whether the text was
# read: that is `unread` and script only. Tuned by `python -m evals.run route` on the dev split.
CONFIDENCE_THRESHOLD = 0.29

PROTOTYPES: dict[Intent, list[str]] = {
    "recommend": [
        "recommend a board game for four players",
        "a quick party game for a big group",
        "something like Catan but shorter",
        "a cooperative game for two",
        "แนะนำบอร์ดเกมสำหรับเล่นกับครอบครัว",
        "家族で遊べるボードゲームを教えて",
    ],
    "compare": [
        "what is the difference between Catan and Ticket to Ride",
        "compare Wingspan and Everdell",
        "Catan กับ Ticket to Ride ต่างกันยังไง",
    ],
    "another": ["give me another one", "not for me, something else", "ขออีกเกม"],
    "off_topic": [
        "what is the weather tomorrow",
        "write me a poem",
        "I like pizza",
        "พรุ่งนี้ฝนตกไหม",
    ],
}

PARSE_SYSTEM = """You read requests for a board game recommendation, in any language.
Return the fields of the schema. Use null when the request does not say.
players: how many people will play. max_minutes: the longest playtime they accept.
youngest_age: age of the youngest player. weight: light, medium or heavy.
coop: true if they want a cooperative game, false if they reject one or want competition.
first_time: they are new to board games. wants_new: they want recently released games.
family: they want a game for a family or for children.
anchor: a game they name as a reference, written as in the request. designer: a designer they name.
intent: recommend, compare (two named games), another (they want a different pick),
off_topic (not about board games), unclear (you cannot tell what they want)."""

PARSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "players", "max_minutes", "youngest_age", "weight", "coop", "solo",
                 "first_time", "wants_new", "family", "anchor", "designer"],
    "properties": {
        "intent": {"type": "string", "enum": ["recommend", "compare", "another", "off_topic", "unclear"]},
        "players": {"type": ["integer", "null"]},
        "max_minutes": {"type": ["integer", "null"]},
        "youngest_age": {"type": ["integer", "null"]},
        "weight": {"type": ["string", "null"], "enum": ["light", "medium", "heavy", None]},
        "coop": {"type": ["boolean", "null"]},
        "solo": {"type": "boolean"},
        "first_time": {"type": "boolean"},
        "wants_new": {"type": "boolean"},
        "family": {"type": "boolean"},
        "anchor": {"type": ["string", "null"]},
        "designer": {"type": ["string", "null"]},
    },
}

_BOUNDS = {"players": (1, 20), "max_minutes": (5, 1440), "youngest_age": (1, 21)}


@dataclass(frozen=True)
class Request:
    text: str
    intent: Intent = "recommend"
    hard: HardConstraints = field(default_factory=HardConstraints)
    weight: Weight | None = None
    coop: bool | None = None
    solo: bool = False
    first_time: bool = False
    wants_new: bool = False
    family: bool = False
    anchor: str | None = None
    designer: str | None = None
    unread: tuple[str, ...] = ()
    language: str = "latin"
    engine: Literal["rules", "cloud"] = "rules"


class IntentClassifier:
    """Nearest prototype phrase per intent. Confidence is the top score minus the mean of the rest."""

    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder
        self._labels: list[Intent] = [label for label, phrases in PROTOTYPES.items() for _ in phrases]
        self._vectors = encoder.encode([p for phrases in PROTOTYPES.values() for p in phrases])

    def classify(self, text: str) -> tuple[Intent, float]:
        scores = self._vectors @ self._encoder.encode([text])[0]
        best: dict[Intent, float] = {}
        for label, score in zip(self._labels, scores):
            best[label] = max(best.get(label, -1.0), float(score))
        ranked = sorted(best.values(), reverse=True)
        top = max(best, key=lambda label: best[label])
        return top, ranked[0] - sum(ranked[1:]) / len(ranked[1:])


def script_language(text: str) -> str:
    for char in text:
        code = ord(char)
        if 0x0E00 <= code <= 0x0E7F:
            return "th"
        if 0x3040 <= code <= 0x30FF:
            return "ja"
        if 0xAC00 <= code <= 0xD7AF:
            return "ko"
        if 0x4E00 <= code <= 0x9FFF:
            return "zh"
    return "latin"


def parse(text: str, *, catalog: Catalog, classifier, llm: LLM | None,
          threshold: float = CONFIDENCE_THRESHOLD) -> Request:
    intent, confidence = classifier.classify(text)
    language = script_language(text)
    rules = rule_parse(text, is_game=catalog.has_name, find_designer=catalog.find_designer)
    unread = rules.unread if language == "latin" else (text,)

    if not unread:
        return _from_rules(text, _pick_intent(intent, confidence, threshold), rules, language)

    if llm is not None:
        if llm.guard(text) is True:
            return Request(text=text, intent="injection", language=language, engine="cloud")
        reply = llm.json(PARSE_SYSTEM, text, PARSE_SCHEMA, "board_game_request")
        request = _from_model(text, reply, language) if reply is not None else None
        if request is not None:
            return request

    return Request(text=text, intent="unclear", unread=unread or (text,), language=language)


def _pick_intent(label: Intent, confidence: float, threshold: float) -> Intent:
    """Confidence decides which intent, never whether the text was read."""
    if label in ("off_topic", "compare", "another") and confidence >= threshold:
        return label
    return "recommend"


def _from_rules(text: str, intent: Intent, rules: RuleParse, language: str) -> Request:
    return Request(
        text=text,
        intent=intent,
        hard=HardConstraints(rules.players, rules.max_minutes, rules.youngest_age),
        weight=rules.weight or ("light" if rules.first_time else None),
        coop=rules.coop,
        solo=rules.solo,
        first_time=rules.first_time,
        wants_new=rules.wants_new,
        family=rules.family,
        anchor=rules.anchor,
        designer=rules.designer,
        language=language,
    )


def _from_model(text: str, reply: dict, language: str) -> Request | None:
    """F6: any value outside the allowed sets rejects the whole reply."""
    if reply.get("intent") not in PARSE_SCHEMA["properties"]["intent"]["enum"]:
        return None
    if reply.get("weight") not in ("light", "medium", "heavy", None):
        return None
    for key, (low, high) in _BOUNDS.items():
        value = reply.get(key)
        if value is not None and (type(value) is not int or not low <= value <= high):
            return None
    if reply.get("coop") not in (True, False, None):
        return None
    if any(type(reply.get(key)) is not bool for key in ("solo", "first_time", "wants_new", "family")):
        return None
    for key in ("anchor", "designer"):
        if reply.get(key) is not None and not isinstance(reply[key], str):
            return None
    return Request(
        text=text,
        intent=reply["intent"],
        hard=HardConstraints(reply["players"], reply["max_minutes"], reply["youngest_age"]),
        weight=reply["weight"] or ("light" if reply["first_time"] else None),
        coop=reply["coop"],
        solo=reply["solo"],
        first_time=reply["first_time"],
        wants_new=reply["wants_new"],
        family=reply["family"],
        anchor=reply["anchor"],
        designer=reply["designer"],
        language=language,
        engine="cloud",
    )
