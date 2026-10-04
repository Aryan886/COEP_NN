# Task 01 — Sol: audit and clean asset data

Read `PLAN.md` in this folder first. Work in `C:\Users\aryan\COEP-NN`. Your time window is T+0 to T+6 minutes. This task authorizes implementation and focused tests, not model training or submission.

## Goal

Produce a reproducible, conservative data preparation function that preserves round alignment and makes missingness visible. Keep the raw CSV unchanged.

## Allowed edits

- `cudies_coding/clean_data.py`
- `cudies_coding/verify_agent.py` for data checks only
- `cudies_coding/tasks/DATA_HANDOFF.md`

Do not edit `agent.py`, `train_model.py`, the harness, or Astra's files.

## Steps

1. Inspect conventions and state the implementation approach in 3–5 lines.
2. Audit asset labels, per-round rows, known-key duplicates, missing prices, non-finite values, and extreme observations. Verify initial findings rather than blindly repeating them.
3. Recover labels only with the unique missing-asset rule in `PLAN.md`. Preserve unresolved rows in the audit; exclude them from attributed asset observations without guessing.
4. Implement `load_history(path)` returning an aligned observed-price pandas DataFrame (round index, six asset columns) and a concise audit dictionary. Missing observations stay missing. Any unexpected duplicate known key must raise an informative error unless a justified resolution is documented.
5. Separate factual recovery from optional price filtering. Investigate whether extreme values look isolated or persistent, but do not invent global correction factors or automatically clip all large moves. Leave learned filtering decisions to training and ensure targets remain distinguishable from imputed inputs.
6. Add and run small tests: one uniquely recoverable label, an ambiguous round, preservation of a missing round/asset, and unchanged raw input. Include a leakage check if implementing any imputation.
7. Write `DATA_HANDOFF.md`: interface, audit counts, recovered/unresolved labels, anomaly examples, tests run, and recommendations for training. Clearly identify unresolved assumptions.

## Done / timeout

By minute 6, hand over the loader even if anomaly analysis is unfinished. Keep uncertain prices unchanged and flag them. Report exact files changed, verification results, and the next task (`03_SOL_FORECAST.md`). Do not spend this task selecting models.
