from __future__ import annotations

import json
from pathlib import Path

GOLD = Path(__file__).resolve().parent / "gold"
FIELDS = ("players", "max_minutes", "youngest_age", "weight", "anchor", "designer")
INTENTS = {"recommend", "compare", "another", "off_topic", "unclear", "injection"}


def load(name: str) -> list[dict]:
    lines = (GOLD / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def validate_parse(rows: list[dict]) -> list[str]:
    errors = []
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    for r in rows:
        missing = {"id", "text", "lang", "source", "split", "intent", *FIELDS} - set(r)
        if missing:
            errors.append(f"{r.get('id')}: missing {sorted(missing)}")
        if r.get("lang") not in {"en", "th", "other"}:
            errors.append(f"{r.get('id')}: bad lang")
        if r.get("source") not in {"brief", "adversarial", "paraphrase"}:
            errors.append(f"{r.get('id')}: bad source")
        if r.get("split") not in {"dev", "test"}:
            errors.append(f"{r.get('id')}: bad split")
        if r.get("intent") not in INTENTS:
            errors.append(f"{r.get('id')}: bad intent")
    return errors


def validate_recommend(rows: list[dict], parse_ids: set[str]) -> list[str]:
    errors = []
    for r in rows:
        if r.get("parse_id") not in parse_ids:
            errors.append(f"{r.get('id')}: unknown parse_id")
        if r.get("split") not in {"dev", "test"}:
            errors.append(f"{r.get('id')}: bad split")
    return errors


def validate_relevance(rows: list[dict]) -> list[str]:
    errors = []
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    for r in rows:
        missing = {"id", "text", "split", "source", "kind"} - set(r)
        if missing:
            errors.append(f"{r.get('id')}: missing {sorted(missing)}")
        if r.get("split") not in {"dev", "test"}:
            errors.append(f"{r.get('id')}: bad split")
        if r.get("source") not in {"brief", "adversarial", "paraphrase"}:
            errors.append(f"{r.get('id')}: bad source")
        if r.get("kind") not in {"vibe", "anchor", "quality"}:
            errors.append(f"{r.get('id')}: bad kind")
    return errors
