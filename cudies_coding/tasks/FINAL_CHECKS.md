# Final checks and submission handoff

## Frozen package

The submission directory contains only `agent.py` (10,242 bytes) and `model.npz` (4,052 bytes). The agent imports only Python standard library modules plus NumPy for artifact loading. The model is loaded from its own directory. No training, CSV, or harness file is needed at runtime. The frozen allocator is P0 forecast-proportional allocation with stable largest-remainder sizing. Astra's P1 crowd-aware policy has not been implemented; do not describe this package as implementing P1.

SHA-256:
- `agent.py`: `BDE7EBF445E0E3B3319573ED28C21ACAAA6FF937C2378E5ADB2682B8CE0453E5`
- `model.npz`: `010D320C19C70985E2AD0FC1C21144A51375731CFEC63598190739A487E2A0AB`

## One-time untouched holdout evaluation

The final chronological block was evaluated once before full-history refitting. Target rounds 1672–2090 supplied 2,383 observed asset prices. No subsequent model or policy choice was tuned on this block.

| Forecast | MAE | RMSE |
|---|---:|---:|
| Latest observed persistence | 4.5333 | 33.6826 |
| Causal robust persistence | 2.4953 | 23.7739 |
| Frozen Ridge | 2.5212 | 23.7696 |

Ridge holdout MAE by asset: quantum 1.599, byte 2.264, gold 1.742, metro 1.974, agro 5.200, solar 2.355. After recording these values, the same frozen Ridge configuration was refit on eligible observed targets from the full supplied history. Therefore the packaged artifact is not the pre-holdout evaluation artifact, and any offline score now computed on supplied history would be in-sample.

## Package checks

- Nine focused tests passed; Python syntax compilation passed for source and packaged agent.
- Three fresh Python processes launched the packaged agent from another working directory, called both public functions, and checked finite forecasts, six keys, native integer orders, and budget compliance. End-to-end process times were 542.0, 503.6, and 494.5 ms; maximum 542.0 ms.
- Five synthetic harness matches against its four weak opponents, using the full-refit packaged agent, yielded scores 3444.1, 3173.3, 1804.9, 1395.0, and 592.5; mean 2082.0, worst 592.5. These are synthetic compatibility checks, not a 21-team field evaluation.
- The raw history CSV was not changed. The package does not access files outside its own directory.

## Limits and submission

There was no reliable 20-opponent correlated/adaptive evaluation before the deadline, and P1 was not implemented. The P0 result can perform differently against actual opponents, production liquidity, and hidden forecast data. The exact normalized leaderboard metric is unknown. Upload the two files from `cudies_coding/submission/` together and confirm the submission receipt with the organizer.

