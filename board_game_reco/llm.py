from __future__ import annotations

import json
import os
from pathlib import Path

GROQ_URL = "https://api.groq.com/openai/v1"
APP_MODEL = "openai/gpt-oss-20b"
GUARD_MODEL = "meta-llama/llama-prompt-guard-2-22m"
DOTENV = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path) -> None:
    """Reads KEY=value lines. Variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class LLM:
    """The only module that knows an LLM vendor exists. Groq through its OpenAI-compatible endpoint."""

    def __init__(self, client, model: str = APP_MODEL, guard_model: str = GUARD_MODEL) -> None:
        self._client = client
        self.model = model
        self._guard_model = guard_model
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @classmethod
    def from_env(cls, dotenv: Path = DOTENV) -> LLM | None:
        load_dotenv(dotenv)
        key = os.environ.get("LLM_API_KEY")
        if not key:
            return None
        from openai import OpenAI

        # max_retries=0: a rate-limited call falls back to the deterministic path at once (F34).
        client = OpenAI(
            api_key=key,
            base_url=os.environ.get("LLM_BASE_URL", GROQ_URL),
            max_retries=0,
            timeout=10.0,
        )
        return cls(client, os.environ.get("LLM_MODEL", APP_MODEL))

    def json(self, system: str, user: str, schema: dict, name: str) -> dict | None:
        """None on any failure, so every caller keeps its deterministic path."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                reasoning_effort="low",
                response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
            )
            self._count(response)
            return json.loads(response.choices[0].message.content)
        except Exception:
            return None

    def text(self, system: str, user: str) -> str | None:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                reasoning_effort="low",
            )
            self._count(response)
            return (response.choices[0].message.content or "").strip() or None
        except Exception:
            return None

    def guard(self, text: str) -> bool | None:
        """True when Prompt Guard flags injection or jailbreak, None when the call fails."""
        try:
            response = self._client.chat.completions.create(
                model=self._guard_model,
                messages=[{"role": "user", "content": text}],
                temperature=0,
            )
            self._count(response)
            return _flagged(response.choices[0].message.content)
        except Exception:
            return None

    @property
    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def _count(self, response) -> None:
        self._usage["calls"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._usage["prompt_tokens"] += usage.prompt_tokens or 0
            self._usage["completion_tokens"] += usage.completion_tokens or 0


def _flagged(content: str | None) -> bool | None:
    text = (content or "").strip()
    try:
        return float(text) >= 0.5
    except ValueError:
        pass
    upper = text.upper()
    if any(word in upper for word in ("MALICIOUS", "INJECTION", "JAILBREAK")):
        return True
    if "BENIGN" in upper:
        return False
    return None
