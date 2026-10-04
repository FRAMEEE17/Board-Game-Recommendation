# Board Game Recommendation

A Python microservice that takes a free-text request in any language and recommends
exactly 1 board game, with a reason, grounded in a real catalog
(BoardGameGeek data, ~2,000 games with descriptions, categories, and mechanics).

## Pipeline

- **Offline** (run once): preprocess the catalog, build a combined text field per
  game, embed it with sentence-transformers.
- **Online** (per request): an LLM extracts hard constraints, soft preferences, and
  an optional anchor game from the free text; the catalog is hard-filtered, candidates
  are ranked by embedding similarity blended with rating, and an LLM explains the pick
  against the top runner-ups.
- **Guardrail**: if the hard filter empties the candidate pool, the least important
  constraint is relaxed first (time, then players, then complexity) and the relaxation
  is stated explicitly in the response, never silently.

## Status

Scaffolding only, implementation in progress.
