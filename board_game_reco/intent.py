from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .catalog import HardConstraints
from .llm import LLM

Weight = Literal["light", "medium", "heavy"]


@dataclass(frozen=True)
class Request:
    text: str
    hard: HardConstraints
    weight: Weight | None
    wants_new: bool
    anchor: str | None
    vibe: str


def parse(text: str, llm: LLM | None) -> Request:
    """LLM parse when available, falls back to the English rule parser on None."""
    raise NotImplementedError
