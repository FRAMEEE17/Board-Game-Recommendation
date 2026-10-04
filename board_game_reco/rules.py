# board_game_reco/rules.py
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Weight = Literal["light", "medium", "heavy"]

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty-five": 45, "forty": 40, "sixty": 60, "ninety": 90,
}
_NUM = r"\d+(?:\.\d+)?|" + "|".join(sorted(_WORDS, key=len, reverse=True))
_QUAL = r"(?:\b(?:under|less than|within|at most|no more than|max(?:imum)?|up to|about|around|in)\s*|<\s*)?"
# Words that only make sense next to a number. Seeing one without a value means the rules missed something.
_UNITS = {"minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "player", "players"}

_HALF_HOUR = r"\bhalf an hour\b"
_HOURS = rf"{_QUAL}\b(an|a|{_NUM})[\s-]*(?:hours?|hrs?|h)\b"
_MINUTES = rf"{_QUAL}\b({_NUM})[\s-]*(?:minutes?|mins?)\b"
_AGE_OLD = rf"\b({_NUM})[\s-]*(?:years?|yrs?)[\s-]*olds?\b"
_AGE_AGED = rf"\b(?:ages?|aged)\s*({_NUM})\s*\+?"
_PARTNER = r"\b(?:with|for|and) (?:my |a )?(?:partner|girlfriend|boyfriend|wife|husband|spouse|date)\b"
_ALONE = r"\b(?:by myself|alone|on my own|solo|solitaire|just me)\b"
_ME_AND = rf"\bme and (?:my )?({_NUM}) (?:friends|others|people|kids|children|buddies)\b"
_GROUP = rf"\b(?:a )?(?:group|party|family|team) of ({_NUM})\b"
_RANGE = rf"\b({_NUM})\s*(?:-|to)\s*({_NUM})\s*(?:players?|people|persons?|ppl)\b"
_COUNT = rf"\b({_NUM})\s*(?:players?|people|persons?|friends|of us|ppl|kids|children)\b"
_FOR_N = rf"\bfor ({_NUM})\b"
_LIGHT = r"\b(?:not too (?:complicated|complex|heavy|hard)|light(?:weight)?|simple|easy(?: to learn)?|casual)\b"
_MEDIUM = r"\b(?:medium(?:[- ]weight)?|mid[- ]weight)\b"
_HEAVY = r"\b(?:heavy|complex|complicated|brain[- ]burner|crunchy|deep)\b"
# Light first, so "not too complicated" is consumed before "complicated" can read as heavy.
_BANDS: tuple[tuple[Weight, str], ...] = (("light", _LIGHT), ("medium", _MEDIUM), ("heavy", _HEAVY))
_AVOID_COOP = r"\b(?:no|not|without|don'?t want(?: any)?|avoid|hate|anything but)\s+(?:a |any )?co-?op(?:erative)?(?: games?)?\b"
_COMPETITIVE = r"\bcompetitive\b"
_COOP = r"\bco-?op(?:erative)?\b"
_FIRST = r"\b(?:(?:my |our )?first (?:board ?)?game|never played|new to (?:board ?games|the hobby)|beginners?)\b"
_NEW = r"\b(?:newest|newly released|new|recent|latest|just released)\b"
_FAMILY = r"\b(?:family|families|kids?|children|child-friendly|kid-friendly)\b"
_ANCHOR = r"\b(?:similar to|something like|games? like|such as|reminds? me of|fans? of|(?:i |we )?(?:really )?(?:like|love|liked|loved|enjoy|enjoyed))\s+([a-z0-9][a-z0-9:'&!.\- ]*)"
_DESIGNER = r"\b(?:designed by|by|from)\s+([a-z][a-z'.\-]+(?:\s+[a-z][a-z'.\-]+){0,3})"


@dataclass(frozen=True)
class RuleParse:
    players: int | None = None
    max_minutes: int | None = None
    youngest_age: int | None = None
    weight: Weight | None = None
    coop: bool | None = None
    solo: bool = False
    first_time: bool = False
    wants_new: bool = False
    family: bool = False
    anchor: str | None = None
    designer: str | None = None
    unread: tuple[str, ...] = ()

    @property
    def has_any_field(self) -> bool:
        return any(
            v not in (None, False)
            for v in (self.players, self.max_minutes, self.youngest_age, self.weight, self.coop,
                      self.solo, self.first_time, self.wants_new, self.family, self.anchor, self.designer)
        )


class _Work:
    """Lowercased text where matched spans are blanked, so leftovers show what was not read."""

    def __init__(self, text: str) -> None:
        self.text = text.lower()

    def take(self, pattern: str) -> list[re.Match[str]]:
        found = list(re.finditer(pattern, self.text))
        for match in found:
            self.blank(match.start(), match.end())
        return found

    def blank(self, start: int, end: int) -> None:
        self.text = self.text[:start] + " " * (end - start) + self.text[end:]


def _value(token: str) -> float:
    if token in ("a", "an"):
        return 1.0
    return float(token) if token[0].isdigit() else float(_WORDS[token])


def parse(
    text: str,
    is_game: Callable[[str], bool] = lambda name: False,
    find_designer: Callable[[str], str | None] = lambda name: None,
) -> RuleParse:
    lower_text = text.lower()
    work = _Work(text)
    designer = _longest_prefix(work, _DESIGNER, find_designer)
    anchor = _longest_prefix(work, _ANCHOR, lambda s: s if is_game(s) else None)

    minutes = [30.0 for _ in work.take(_HALF_HOUR)]
    minutes += [_value(m.group(1)) * 60 for m in work.take(_HOURS)]
    minutes += [_value(m.group(1)) for m in work.take(_MINUTES)]
    ages = [_value(m.group(1)) for m in work.take(_AGE_OLD) + work.take(_AGE_AGED)]

    players: list[float] = []
    solo = bool(work.take(_ALONE))
    if solo:
        players.append(1)
    players += [2 for _ in work.take(_PARTNER)]
    players += [_value(m.group(1)) + 1 for m in work.take(_ME_AND)]
    players += [_value(m.group(1)) for m in work.take(_GROUP)]
    players += [_value(m.group(2)) for m in work.take(_RANGE)]
    players += [_value(m.group(1)) for m in work.take(_COUNT)]
    players += [_value(m.group(1)) for m in work.take(_FOR_N)]

    weight: Weight | None = None
    for band, pattern in _BANDS:
        if work.take(pattern) and weight is None:
            weight = band

    coop = None
    if work.take(_AVOID_COOP) or work.take(_COMPETITIVE):
        coop = False
    elif work.take(_COOP):
        coop = True
    first_time = bool(work.take(_FIRST))
    wants_new = bool(work.take(_NEW))
    family = bool(re.search(_FAMILY, lower_text))

    return RuleParse(
        players=int(max(players)) if players else None,
        max_minutes=int(min(minutes)) if minutes else None,
        youngest_age=int(min(ages)) if ages else None,
        weight=weight,
        coop=coop,
        solo=solo,
        first_time=first_time,
        wants_new=wants_new,
        family=family,
        anchor=anchor,
        designer=designer,
        unread=_unread(work.text),
    )


def _longest_prefix(work: _Work, pattern: str, resolve: Callable[[str], str | None]) -> str | None:
    """Tries the words after a trigger, longest first, and keeps the first that resolves."""
    for match in re.finditer(pattern, work.text):
        words = match.group(1).strip(" .").split()
        for size in range(len(words), 0, -1):
            candidate = " ".join(words[:size])
            resolved = resolve(candidate)
            if resolved:
                end = match.start(1) + len(match.group(1)) - len(match.group(1).lstrip()) + len(candidate)
                work.blank(match.start(), end)
                return resolved
    return None


def _unread(leftover: str) -> tuple[str, ...]:
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?", leftover)
    hits = [t for t in tokens if t[0].isdigit() or t in _UNITS or t in _WORDS]
    return tuple(dict.fromkeys(hits))
