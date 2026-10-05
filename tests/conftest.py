# tests/conftest.py
from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "boardgames.csv"


class FakeEncoder:
    name = "fake-bag-of-words-64"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % 64] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


class StubClassifier:
    def __init__(self, intent: str = "recommend", confidence: float = 1.0) -> None:
        self.intent = intent
        self.confidence = confidence

    def classify(self, text: str) -> tuple[str, float]:
        return self.intent, self.confidence


class FakeLLM:
    def __init__(self, parsed: dict | None = None, written: str | None = None, flagged: bool | None = False) -> None:
        self.parsed = parsed
        self.written = written
        self.flagged = flagged
        self.calls: list[str] = []

    def json(self, system: str, user: str, schema: dict, name: str) -> dict | None:
        self.calls.append("json")
        return self.parsed

    def text(self, system: str, user: str) -> str | None:
        self.calls.append("text")
        if self.written is None:
            return None
        name = user.split("\n", 1)[0].removeprefix("Game: ")
        return self.written.replace("{name}", name)

    def guard(self, text: str) -> bool | None:
        self.calls.append("guard")
        return self.flagged

    @property
    def usage(self) -> dict[str, int]:
        return {"calls": len(self.calls), "prompt_tokens": 0, "completion_tokens": 0}


@pytest.fixture(scope="session")
def catalog(tmp_path_factory):
    from board_game_reco.catalog import Catalog

    return Catalog.load(CSV, tmp_path_factory.mktemp("vectors"), FakeEncoder())


class CountingClassifier(StubClassifier):
    """Counts parses: every recommend() classifies once, a retry never does."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def classify(self, text: str) -> tuple[str, float]:
        self.count += 1
        return super().classify(text)


@pytest.fixture
def engine(catalog, monkeypatch):
    """Points the app's shared engine at the fake catalog. Yields the classifier and the build count."""
    from app import state
    from board_game_reco.recommender import Recommender

    classifier = CountingClassifier()
    builds = []

    def build():
        builds.append(1)
        return Recommender(catalog, classifier, None)

    monkeypatch.setattr(state, "_build", build)
    state.shared.clear()
    yield classifier, builds
    state.shared.clear()
