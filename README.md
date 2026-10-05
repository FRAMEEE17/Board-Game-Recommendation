# Board Game Recommendation
Demo: [https://sunday-morning.streamlit.app/]

Type what you want in any language and get one board game back, with a reason. It picks
from about 2,000 BoardGameGeek games. [https://www.kaggle.com/datasets/andrewmvd/board-games]

Most requests never leave the machine. Plain rules read the player count, time and age,
and a small local model ranks the games. A cloud model (Groq) only steps in when the
rules can't read the whole request, likely non-English text. Player count and age always
filter, because a game the group can't play is the worst answer. If nothing fits, only
playtime loosens (1.5x, then dropped), and the answer says so.

## Run

```bash
uv run python -m board_game_reco "family game for 4 under an hour"   # CLI
uv run streamlit run app/main.py                                      # app at http://localhost:8501
```

`uv` installs Python 3.12. The first run downloads the model (118 MB).

To read other languages, paste a Groq key in the app sidebar, or set `LLM_API_KEY` in
`.env` for the CLI. Without a key, a request the rules can't read gets a question back.
Try these with a key:

- `a game for my kids, they are 5, 12 and 7 and playing with me`
- `家族4人で30分くらいで遊べるゲーム`
- `อยากได้เกมเล่น 4 คน ไม่เกินหนึ่งชั่วโมง`

The stats page needs a one-time build for its game map:
`uv run --group map python -m board_game_reco.build`.

Docker, verified on arm64:
`docker buildx build --target runtime -t board-game-reco --load . && docker run --rm -p 8501:8501 board-game-reco`

## Test

```bash
uv run pytest                                        # 248 tests, no key or network
uv run python -m evals.run                           # parse, route, recommend, perf
uv run python -m evals.run judge relevance explain   # needs LLM_API_KEY
docker buildx build --target test -t board-game-reco:test --load . \
  && docker run --rm --network none board-game-reco:test test
```

## Results

| Check | Result | Why it matters | Verdict |
|---|---|---|---|
| Tests | 248 pass, also offline in the container on 2 CPUs and 2 GB | Same answers every time, runs on a small box | Very good |
| Picks that break a stated limit | 0 of 30 | The filters block the costliest mistake | Very good |
| Intent, test split | 1.0 | Off-topic and injection requests get declined. The test set is small | Good |
| English fields, no key | 0.99 | Exact limits at no cloud cost | Very good |
| Thai and other languages, no key | 0.78 and 0.63 | These users need the key, and accuracy with the key isn't measured yet | Tune |
| Router (rules or model?) | AUROC 0.88, catches 94% of misreads | Most requests skip the model. About 1 in 5 fine requests goes over anyway | Good |
| Judge on corrupted reasons | AUROC 1.0 on 30 synthetic pairs. The plain grounding check catches 0.33 | The judge can gate quality. The string check misses two in three | Very good |
| Relevance of the pick (1 to 5) | 3.55 dev, 3.84 test | Usually on topic, rarely a standout. Reweighting added 0.05, so the gap is in the features | Tune |
| Faithfulness of reasons | Ragas 0.74, grounding check 100% | No invented numbers or names, but about 1 claim in 4 is generic wording the data can't back | Tune |
| Latency | Basic p95 63 ms on 2 CPUs, cloud p95 1.7 s | Instant for most requests. The free Groq tier allows a few cloud calls a minute | Very good |
| Cold start, memory, image | 1.9 s, 512 MB, 909 MB | Fits the smallest container tier | Very good |
| int8 against fp32 | Same top pick 95% (19 of 20) | 4x smaller model for 1 changed pick in 20. Don't shrink it further | Good |
| API key in the image | none | Safe to publish | Very good |

What to tune next:

1. "Similar to X" only compares descriptions, so "like Pandemic" returns Virus!. Add a
   mechanics match. This is the gap a user sees first.
2. Measure Thai and other languages with the key, then fix the misses.
3. Reword or ground the generic sentences in the reasons to lift faithfulness above 0.74.

## Appendix

<p>
<img src="docs/images/app-recommend.png" width="48%" alt="Recommend page">
<img src="docs/images/app-why.png" width="48%" alt="Why this game">
</p>

One card shows the game, its facts, the reason and which engine read the request. "Why
this game" shows the funnel: 2,000 games, 6 eligible, 1 picked.

<p>
<img src="docs/images/app-cloud.png" width="48%" alt="Cloud AI reads a request with three ages">
<img src="docs/images/app-cloud-why.png" width="48%" alt="Limits removed per request">
</p>
<p>
<img src="docs/images/app-cloud-thai.png" width="48%" alt="A Thai request answered in Thai">
</p>

With a key the model reads what the rules can't, Thai included, and the reply comes back
in the same language. It only reads the request. The filters and the final check still
choose the game.

<p>
<img src="docs/images/app-stats-charts.png" width="48%" alt="Catalog charts">
<img src="docs/images/app-stats-table.png" width="48%" alt="Filterable catalog table">
</p>

- Weight against rating: heavier games rate higher, so ranking by rating alone would push
  heavy games at casual players. The weight filter stops that.
- Playtime piles up at 30, 60, 90 and 120 minutes, so "under an hour" is a natural bucket.
- Most games are from 2010 on. "New" (2023 or later) covers only 187 games.
- Solo (465 games) and co-op (331) have enough choice to serve those requests.
- What people play now (Flip 7, Bomb Busters) differs from the all-time rating, which
  could drive a popular-now mode.
- Most games have a rating spread of 1.1 to 1.5. The few above 1.8 split opinion, so a
  low spread could drive a safe-pick mode for groups.

<p>
<img src="docs/images/roc-router.png" width="40%" alt="Router ROC">
<img src="docs/images/roc-judge.png" width="40%" alt="Judge ROC">
</p>

### Router

The router is a binary classifier. A request is positive (y = 1) when the rule parser misreads it, meaning at least one gold field is wrong or the true intent is unclear. The prediction is "escalate to the cloud model". I evaluated it on the test split: 37 rows, 16 positive and 21 negative.

- Axes: TPR is the recall of rule failures. FPR is the share of correctly parsed requests that get escalated anyway, which is the cost of unnecessary cloud calls.
- Unread and non-Latin flags: each is a binary indicator, so its ROC is two straight segments through a single operating point. Together they sit at TPR 0.81 (13/16) and FPR 0.095 (2/21). The AUROC of 0.86 equals the balanced accuracy of that classifier.
- Deployed rule (the OR of the two flags): TPR 0.94 (15/16) at FPR 0.19 (4/21), with AUROC 0.88. Reaching 100% recall would need an FPR of about 0.52, so the extra recall is not worth it.
- Confidence signal (1 minus the margin): a continuous score with an AUROC of only 0.65. It discriminates weakly, so almost all of the router's value comes from the "could not read this" flags.

Caveat: with 16 positives, a recall of 15/16 has a wide 95% confidence interval (Clopper-Pearson, roughly 0.70 to 0.99). Treat it as a trend, not a precise estimate.

### Judge

The judge detects unfaithful explanations. Positives are the 15 reasons we corrupted on purpose, and negatives are the 15 faithful ones. The judge's 1 to 5 faithfulness rating is the ranking score.

- AUROC 1.0 on both runs: the classes separate perfectly. Every corrupted reason scores below every faithful one. The two ROC curves overlap, so sampling at temperature 1.0 did not change the ranking (mean absolute score difference between runs: 0.17).
- The black square (grounded()): a deterministic rule check, so it is a single operating point at FPR 0 and TPR 0.33 (5/15). Precision is 1.0 but recall is low, because it only checks game names and numbers. It catches swapped numbers and misses invented claims entirely.

Caveat: n is 15 per class, and the negatives are synthetic corruptions we generated ourselves. AUROC 1.0 is therefore an upper bound, not an estimate of performance on natural errors, which are harder to separate.
