# Design

## Problem

Take a free-text request in any language and return exactly one board game plus a
reason, grounded in a real catalog (BoardGameGeek top ~2,000, `data/boardgames.csv`).

Constraints found while grounding:

- No LLM API key is guaranteed on the reviewer's machine. The system must return a
  real answer without one. The LLM improves parsing and wording, it is never required.
- 15 game names are duplicated, so games are keyed by `row_id`, never by name.
- `max_playtime = 0` (3 rows) and `release_year = 0` mean unknown. Left as 0, a game
  with "0 minutes" would pass every "under 2 hours" filter.
- Descriptions are short (median 78 chars). Embedding text is name + description +
  categories + mechanics, not description alone.
- "Similar to Splendor" also matches Splendor: Marvel, Splendor Duel, and the Pokémon
  edition. Recommending a reskin of the anchor is a weak answer, so the anchor's own
  family is excluded.

## Usage (caller's view)

```python
from board_game_reco import recommend

name, reason = recommend("I want a strategy game that takes no more than 2 hours")
```

```bash
python -m board_game_reco "เกมสำหรับเล่นกับแฟน 2 คน"
```

For the demo notebook, where the evidence matters as much as the answer:

```python
from board_game_reco import Recommender

rec = Recommender.default().recommend("I like games similar to Splendor")
rec.game.name, rec.reason, rec.relaxed, [g.name for g in rec.runners_up]
```

## Shape

Modules are grouped by the knowledge they own, not by execution order (per the
temporal-decomposition red flag).

| Module | Owns | Public surface |
|---|---|---|
| `catalog.py` | What a game is, how the CSV is cleaned, the embeddings and their freshness | `Game`, `HardConstraints`, `Catalog.load`, `find`, `eligible`, `similarity` |
| `intent.py` | What a request means | `Request`, `parse(text, llm)` |
| `llm.py` | The only place that knows an LLM vendor exists | `LLM.from_env`, `LLM.json` |
| `recommender.py` | The decision policy: filter, relax, score, explain | `Recommendation`, `Recommender` |
| `__init__.py` | The assignment's required signature | `recommend(text) -> (name, reason)` |

Load-bearing decisions:

- **Unknown is `None`, not 0.** `Game.max_minutes` and `Game.year` are `int | None`. The
  type forces every filter to decide what unknown means, instead of 0 silently passing.
- **Hard vs soft is a type boundary.** `HardConstraints` filters (`Catalog.eligible`).
  Everything else in `Request` only moves the score. A soft preference can never
  eliminate a game.
- **The LLM boundary returns `None` on any failure.** `parse` and the reason writer both
  have a deterministic path, so a timeout, a bad key, or no key all degrade the same way.
- **Embeddings are a materialized dataset** (Dataset Materializer) with a manifest
  (`source_sha256`, `model`, `rows`) and a readiness marker written last. A changed CSV or
  model name triggers a rebuild, never a stale cache.
- **The reason is built from scoring evidence**, the attributes that actually moved the
  score, so it cannot claim something about the game that isn't in the data.

Interface depth: callers get one call in, one immutable result out. pandas, numpy,
sentence-transformers, the cache layout, and the LLM vendor stay private to the module
that owns them.

Deliberately not done: no web server, no database, no plugin system for other datasets,
no recsys training. Each was considered and is the wrong size for a static 2,000-row
catalog.

## Synthesis decision

Base: candidate B (modules by owned knowledge, deep `Catalog` and `Recommender`).

- Candidate A (one module per pipeline stage: load, embed, parse, filter, rank, explain)
  was rejected. `Game`'s representation and the "0 means unknown" rule leaked into four
  modules, and the caller had to wire six steps together.
- Candidate C (an LLM agent calling `search_catalog` tools) was rejected. It cannot run
  without a key and cannot say why it picked a game.
- Adapted from C: the LLM as an optional upgrade for parsing and for writing the reason
  in the user's language.
- Adapted from A: the offline/online split survives as `Catalog.load` (offline,
  materialized once) versus `Recommender.recommend` (online, per request).

## Tradeoffs accepted

- We accept a small, English-only rule parser for the no-key path in exchange for a
  system that always answers. With a key, any language is parsed properly.
- We accept a ~120 MB multilingual embedding model in exchange for soft matching that
  works on Thai or Japanese requests even without an LLM.
- We accept player count and minimum age never being relaxed. A game that can't seat
  your group, or isn't suitable for your child, is wrong, not "close enough".

## Alternatives considered

- **TF-IDF instead of embeddings.** Lighter, no model download, but it matches words
  instead of meaning and fails entirely on non-English requests.
- **Send the whole catalog to the LLM and let it pick.** Simplest code, but it exposes
  the decision to the model's judgment with no scoring evidence, and needs a key.

## Open questions and risks

- Is relaxing only playtime (×1.5, then dropped) the right relaxation policy? The earlier
  diagram said time → players → complexity. Complexity is soft so it never needs
  relaxing, and relaxing players gives wrong answers.
- Which LLM endpoint should the demo use? The adapter speaks the OpenAI-compatible API, so
  OpenRouter, Groq, and OpenAI all work with `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
- Should "newly released" mean the last 3 years of the dataset (2023+), or the newest
  year present?

## Next implementation step

`Catalog.load`: parse the CSV into `Game`, then materialize embeddings with the manifest.
