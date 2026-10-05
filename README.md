# Board Game Recommendation

Free text in any language, one board game and a reason out, from about 2,000
BoardGameGeek games. Rules and a local multilingual embedding model (int8 ONNX)
answer every request they can read in full, with no network call. Groq is used only
when part of a request is left unread. Player count and age always filter. Only
playtime is relaxed (1.5x, then dropped), and the answer says so.

## Run

```bash
uv run python -m board_game_reco "family game for 4 under an hour"   # CLI
uv run streamlit run app/main.py                                      # app at http://localhost:8501
```

`uv` installs Python 3.12. The first run downloads the model (118 MB).

For other languages and unusual phrasing, paste a Groq key in the app sidebar, or set
`LLM_API_KEY` in `.env` for the CLI. Without a key the service asks a question back
instead of guessing. Queries that need the key:

- `a game for my kids, they are 5, 12 and 7 and playing with me`
- `家族4人で30分くらいで遊べるゲーム`
- `อยากได้เกมเล่น 4 คน ไม่เกินหนึ่งชั่วโมง`

The stats page draws a game map from a one-time build:
`uv run --group map python -m board_game_reco.build`.

Docker (verified on linux/arm64):
`docker buildx build --target runtime -t board-game-reco --load . && docker run --rm -p 8501:8501 board-game-reco`

## Test

```bash
uv run pytest                                        # 248 tests, no key or network needed
uv run python -m evals.run                           # parse, route, recommend, perf
uv run python -m evals.run judge relevance explain   # needs LLM_API_KEY
docker buildx build --target test -t board-game-reco:test --load . \
  && docker run --rm --network none board-game-reco:test test
```

## Results

| Check | Result | What it means | Verdict |
|---|---|---|---|
| Unit and gold tests | 248 passed locally, and in the container under `--network none` and `--cpus=2 --memory=2g` | Runs offline on a small box and gives the same answers every time. | Very good |
| Picks that break a stated limit | 0 of 30 gold requests | The costliest mistake, a game the group cannot play, is blocked by design. | Very good |
| Intent accuracy, test split | 1.0 | Off-topic and injection requests are declined. The test set is small, so read this as a smoke test. | Good |
| Field accuracy in English, no key | 0.99 | Players, time and age are read exactly, with no cloud cost. | Very good |
| Field accuracy in Thai and other languages, no key | Thai 0.78, other 0.63 | These users need the key. Accuracy with the key is not measured yet. | Tune |
| Router (rules or model?) | AUROC 0.88, catches 94% of misread requests | Most requests never reach the paid model and almost every misread one does. Roughly 1 in 5 correct requests is sent over needlessly. | Good |
| Judge AUROC on corrupted reasons | 1.0 on both runs (30 synthetic pairs). The plain grounding check catches 0.33 | The automatic grader can gate quality. The cheap string check alone misses two bad reasons in three. | Very good |
| Mean relevance of the pick (1 to 5) | 3.55 dev, 3.84 test | Picks are usually on topic but rarely a standout. Reweighting gained only 0.05, so the fix is better features. | Tune |
| Faithfulness of reasons | Ragas 0.74, grounding check 100% (39 reasons) | No invented number or name. About 1 claim in 4 is generic wording the data cannot back. | Tune |
| Latency | Basic p95 63 ms on 2 CPUs, cloud p95 1.7 s | Instant for most requests and under 2 s when the model reads. The free Groq tier allows a few cloud requests a minute, so real traffic needs a paid plan. | Very good |
| Cold start, memory, image | 1.9 s, 512 MB peak, 909 MB on disk | Fits the smallest container tiers, so it is cheap to host. | Very good |
| int8 against fp32, same top pick | 95% (19 of 20) | The model is 4x smaller and changes the top pick in 1 request of 20. That is the limit, so do not shrink it further. | Good |
| API key in the image | none | The image can be shared or published safely. | Very good |

Next to tune, in order:

1. "Similar to X" compares descriptions only, so "like Pandemic" returns Virus!. Add a mechanics match. This is the biggest visible gap for a shop assistant.
2. Thai and other languages: measure accuracy with the key, then fix the misses.
3. Reason wording: drop or ground the generic sentences to lift faithfulness above 0.74.

## Pictures

<p>
<img src="docs/images/app-recommend.png" width="48%" alt="Recommend page">
<img src="docs/images/app-why.png" width="48%" alt="Why this game">
</p>

One card gives the game, the facts, the reason and which engine read the request. "Why this game" shows the work: 2,000 games, 6 eligible, 1 picked. The limits did the filtering, so any answer can be audited.

<p>
<img src="docs/images/app-cloud.png" width="48%" alt="Cloud AI reads a request with three ages">
<img src="docs/images/app-cloud-why.png" width="48%" alt="Limits removed per request">
</p>
<p>
<img src="docs/images/app-cloud-thai.png" width="48%" alt="A Thai request answered in Thai">
</p>

With a key the model reads what the rules cannot, including Thai, and the answer comes back in the same language. The model only reads the request. The filters and the final check still decide the pick.

<p>
<img src="docs/images/app-stats-charts.png" width="48%" alt="Catalog charts">
<img src="docs/images/app-stats-table.png" width="48%" alt="Filterable catalog table">
</p>

- Weight against rating: heavier games rate higher, so a rating-only ranker would push heavy games at casual players. The weight filter prevents that.
- Playtime: spikes at 30, 60, 90 and 120 minutes, so "under an hour" and "under two hours" are natural buckets.
- Games per year: most of the catalog is from 2010 on, so "new" (2023 or later, 187 games) is a small slice.
- Top mechanics: solo (465 games) and co-op (331) have enough choice to serve those requests.
- Trending: what people play now differs from the all-time rating (Flip 7, Bomb Busters), a signal for a popular-now mode.
- Crowd-pleaser or polarizing: most games sit at a rating spread of 1.1 to 1.5 and the few above 1.8 split opinion. A "safe pick for a group" mode can use a low spread.

<p>
<img src="docs/images/roc-router.png" width="40%" alt="Router ROC">
<img src="docs/images/roc-judge.png" width="40%" alt="Judge ROC">
</p>

Router: the rules' own "could not read this" flag alone catches 81% of misses for about 10% extra cloud calls. The combined router catches 94% for about 19%. Catching all of them would cost about half of the correct requests. The confidence signal alone (green) is weak. Judge: both runs sit in the top-left corner. The black square is the cheap check: no false alarms, but it misses two thirds of bad reasons.
