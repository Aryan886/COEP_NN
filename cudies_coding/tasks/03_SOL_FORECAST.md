# Task 03 — Sol: train and validate a compact forecast

Read `PLAN.md` and `DATA_HANDOFF.md` first. Work in `C:\Users\aryan\COEP-NN`. Time window: T+6 to T+14 minutes. If the data handoff is incomplete, use observed aligned data conservatively; do not restart the whole audit.

## Ownership

Edit `agent.py`, `train_model.py`, forecast checks in `verify_agent.py`, one model artifact, and `tasks/FORECAST_HANDOFF.md`, all under `cudies_coding`. Read `clean_data.py`; change it only after taking ownership from the previous task. Do not edit Astra's specification or organizer files.

## Implementation

1. State the approach in 3–5 lines. Use the shared chronological split and preserve the final 20% holdout.
2. Implement one shared feature-window function in `agent.py` that training imports without requiring an existing model artifact. Inputs are at most 20 observations per asset. Window cleaning must be reproducible at inference, including short histories and missing data.
3. Compare persistence, a compact regularized linear model, and at most one small sklearn boosting model if time permits. Use a small fixed parameter set. Train against next observed prices or deltas with an explicit reconstruction rule; return prices from `predict_market`.
4. Start with own recent levels/changes, short means, deviations, volatility, and partner information. Keep feature order explicit and saved with the artifact contract. Fit scalers and outlier thresholds only on fitting data.
5. Evaluate chronological MAE and RMSE overall and per asset. Report observed-target results honestly; if filtering targets, also disclose coverage and unfiltered results. Record directional/profit diagnostics without selecting on the untouched final block.
6. Prefer the simpler model unless the alternative improves validation meaningfully. Do not spend the window on per-asset hyperparameter grids or neural models. Blend only if validated.
7. Implement robust early-round predictions and a valid basic forecast-only allocator. Do not retain the starter's blanket 40-price fallback or allocate leftover budget blindly to the last asset.
8. Export a compact model loaded relative to `__file__`. Avoid unnecessary runtime imports. Time at least one fresh-process import and forecast call; switch to lightweight coefficient inference if heavy imports threaten the deadline.
9. Run tests for feature parity, empty/short/missing histories, finite predictions, and budget-valid Python integer orders. Keep diagnostics contextual and avoid silent exception swallowing.

## Handoff and time cap

Save `FORECAST_HANDOFF.md` with split boundaries, model candidates and scores, selected features/model, artifact path, fallback behavior, cold timing, tests, and unresolved concerns. Mark final holdout as unused.

By minute 14, keep a working agent even if only the linear model is ready. The next task implements Astra's frozen policy without changing the forecast interface. Do not refit on the final holdout yet.
