# Board Game Recommendation

A Python microservice that takes a free-text request in any language and recommends
exactly 1 board game, with a reason, grounded in a real catalog
(BoardGameGeek data, ~2,000 games with descriptions, categories, and mechanics).

## Pipeline

- **Offline** (run once): preprocess the catalog, build a combined text field per
  game, embed it with a local multilingual int8 ONNX model and cache the vectors.
- **Online** (per request): rules read the constraints they can (players, playtime,
  age, complexity, anchor game). Requests the rules read in full never call a model.
  When part of a request is left unread, it goes to Groq (`openai/gpt-oss-20b`) behind
  a prompt-injection guard. The catalog is hard-filtered, candidates are ranked by
  embedding similarity blended with rating, and the pick is checked against the
  constraints before a reason is written.
- **Guardrail**: if the hard filter empties the candidate pool, only playtime is
  relaxed: 1.5x first, then dropped. Player count and age are never relaxed. The
  relaxation is stated explicitly in the response, never silently.
- **No model available**: with no key, or when the call fails, the service asks one
  question back instead of guessing.

## Run

```bash
uv run python -m board_game_reco "family game for 4, not too complicated" --trace
```

`uv` installs Python 3.12 if the machine lacks it. The first run downloads the int8
embedding model (118 MB) and builds the vectors once.

For other languages, put a Groq key in `.env`:

```
LLM_API_KEY=...
```

The app reads the key from `LLM_API_KEY` and talks to
`https://api.groq.com/openai/v1`. Without a key, requests the rules cannot read in
full get a question back instead of a guess.

## Status

Wave 1 (the recommendation engine) is implemented. Docker packaging, the judge and
faithfulness suites, and the Streamlit pages come in wave 2.
