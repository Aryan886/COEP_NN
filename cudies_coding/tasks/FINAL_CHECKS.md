# Final checks and submission handoff

## Submission files

Upload both files in cudies_coding/submission:

- agent.py, 21,655 bytes
- model.npz, 4,052 bytes

The active place_orders entry point uses P0 forecast-proportional allocation. The package also includes place_orders_crowd, which implements Astra P1 for later evaluation. The model loads relative to agent.py and inference does not import training code.

SHA-256:
- agent.py: ED8BEC452A47B50DA20F82947857EACFACBE9B308ED6F2B49917AE7008994658
- model.npz: 010D320C19C70985E2AD0FC1C21144A51375731CFEC63598190739A487E2A0AB

## Forecast evaluation

The final chronological block was evaluated once before full-history refitting. Target rounds 1672–2090 supplied 2,383 observed asset prices.

| Forecast | MAE | RMSE |
|---|---:|---:|
| Latest observed persistence | 4.5333 | 33.6826 |
| Causal robust persistence | 2.4953 | 23.7739 |
| Frozen Ridge | 2.5212 | 23.7696 |

After recording these values, the frozen Ridge configuration was refit on eligible observed targets from the full supplied history. The packaged artifact is therefore not the pre-holdout evaluation artifact. An offline score computed now on supplied history would be in-sample.

## Astra P1 evaluation

P1 estimates opponent flow, scores nine flow/liquidity scenarios with our candidate order included in the dominant-side calculation, then uses an exact integer budget dynamic program. It matches the harness tie rule (buys win ties), discounts both wins and losses, and retains zero orders and unspent budget. P0 remains callable as the baseline.

A paired synthetic simulation used 20 opponents plus us, matched 150-round price schedules and seeds, production-scale liquidity without the harness's extra 0.3 factor, and separate evolving volume histories. There were five seeds for each base field and one seed per liquidity stress case.

| Field | P1 minus P0 mean | Worst paired difference |
|---|---:|---:|
| Correlated forecasters | +326.0 | -301.3 |
| Mixed opponents | +285.4 | -629.1 |
| Adaptive field with 10 P1 opponents | -19.0 | -460.2 |
| Clone stress | -77.9 | -445.8 |

The evaluation gate did not show consistent P1 gains in the adaptive and clone fields, and poor-seed losses were material. The active entry point therefore remains on P0, following Astra's fallback recommendation. P1 remains implemented and tested. P1's assumed crowd response, liquidity, and shrinkage settings are synthetic design assumptions, not calibrated production facts.

## Verification and limits

- Seventeen focused tests pass, including 1,392 scoring parity cases and 30 seeded optimizer comparisons against exhaustive enumeration.
- Python syntax compilation passes for source and packaged code.
- Three fresh Python processes imported the packaged agent from another working directory, called forecasts, P0 orders, and P1 orders, and checked finite predictions, six asset keys, native integers, and budget bounds. Total process times were 940.6, 467.6, and 568.2 ms; maximum 940.6 ms.
- One packaged-agent harness smoke match scored 1,886.2 against four weak built-in opponents. This checks compatibility, not tournament competitiveness.
- Raw history.csv is unchanged. No hidden 21-team result or leaderboard rank is established.

Upload agent.py and model.npz together, then confirm the organizer received the submission.

