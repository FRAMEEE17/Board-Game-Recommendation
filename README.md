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

## Evaluation

```bash
uv run python -m evals.run                                 # parse, route, recommend, perf: no key needed
uv run python -m evals.run judge relevance tune explain    # calls the judge, needs LLM_API_KEY
uv run pytest                                              # invariants that must never break
```

Each run writes `evals/results/<date>_<sha>/` with `summary.csv`, `rows.csv` and the charts.

- The judge is `qwen/qwen3.8-27b` on Groq in thinking mode. Its scores vary between runs, so the judge suite scores every row twice and reports the spread. It is measured against `evals/gold/judge.jsonl` before its relevance scores are used.
- Judge scores are cached in `.cache/judge/`, keyed by the prompt version. A rerun pays only for new (query, game) pairs.
- Groq's free tier gives the judge 8K tokens a minute and 200K a day. The judge suites together can need more than one day. When the daily quota runs out, the run stops, keeps every cached score, and the same command continues later.
- Every gold row in `evals/gold/` was drafted by a model and checked by scripts against the catalog. That includes the faithful and corrupted reasons in `judge.jsonl`. No person reviewed the labels.
- `perf` measures Basic-mode latency on the machine it runs on. The 2-CPU Docker measurement comes with the Docker image. The first `perf` run downloads the fp32 model (471 MB) for the int8 comparison.

### Results

| Check | Result |
|---|---|
| Judge AUROC on the judge gold set (two runs) | 1.0 and 1.0 |
| `grounded()` check on corrupted reasons | catches 0.333 of them (TPR), flags 0.0 of the faithful ones (FPR) |
| Mean judge relevance, dev split | 3.55 |
| Mean judge relevance, test split | 3.84 |
| Reason grounding pass rate | 1.0 (39 reasons) |
| Mean judge faithfulness of reasons | 3.08, none scored 5 |
| Basic latency (N3, dev machine) | p50 2.48 ms, p95 11.04 ms |
| Cloud latency (N4, 20 escalated requests, 0 fallbacks) | p50 1453 ms, p95 1675 ms |
| int8 against fp32 top-1 agreement (N13) | 0.95 (19 of 20) |

**Judge validation.** The judge was checked on `judge.jsonl`, where half the reasons are corrupted on purpose. Both runs reached AUROC 1.0, well above the 0.75 gate, so its relevance scores were used. The plain `grounded()` string check is weaker. It catches a third of the corrupted reasons and flags none of the faithful ones.

**Weight tuning.** Nine settings of `similarity` (0.6, 1.0, 1.4) and `rating` (0.2, 0.4, 0.8) were scored on the dev split. The best was similarity 0.6 and rating 0.2 at 3.60, against 3.55 for the defaults. That gain of 0.05 is within the judge's run-to-run noise, and the test split did not move (3.84 either way). The defaults stay at similarity 1.0 and rating 0.4.

**Known limitations.** The judge flagged two weak picks. "best strategy game ever" returns Roll Player Adventures, which scored 1. "something similar to Pandemic" returns Virus!, which scored 2. Neither is fixed by reweighting.

**Quantization.** int8 and fp32 disagree on two of 20 queries. "a game about pirates" gives Rum & Bones: Second Tide on int8 and Sail on fp32. "a quick card game" gives 6 nimmt! 25 Jahre on int8 and Last Will on fp32. Agreement meets the 0.95 target exactly.

**Ragas.** `uv run --group ragas python -m evals.run ragas` is an optional faithfulness check through Ragas, outside the main install. Mean Ragas faithfulness over the 39 template reasons is 0.74 (0 to 1). The first run stopped at Groq's 200K daily token limit after about 20 scores; the rerun finished the rest from the cache. Ragas splits each reason into claims and checks each one against the game's facts, so a reason with one unsupported claim out of four scores 0.75. The 1 to 5 faithfulness number above comes from the judge's own prompt.

## Status

Wave 1 (the recommendation engine) is implemented. Docker packaging, the judge and
faithfulness suites, and the Streamlit pages come in wave 2.
