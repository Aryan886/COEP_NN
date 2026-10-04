# Forecast handoff

## Contract

`agent.make_features(recent_history, asset, ratio_limits)` returns a reference price, 14 ordered features, and the number of valid own observations. It reads at most 20 prior slots per asset and its sector partner. Missing/non-finite/nonpositive inputs are skipped. Normally the latest valid price is the reference; when it is an isolated scale jump relative to the last-three median, the median is used. The ratio limits are fitted on the fitting segment and saved in `model.npz`. Observed targets are never imputed.

`predict_market` returns positive native floats for all six assets. An absent artifact uses the same causal reference with a documented default ratio limit and asset-specific fallback price. A corrupt artifact raises with context. The artifact is loaded relative to `agent.py`; training imports the feature function without loading it.

## Development comparison

Chronological target rounds: fitting 1–1253; validation 1254–1671; final holdout 1672–2090 was not used for model selection. The fit-only target ratio rule retained 6,969 of 7,028 available fitting targets. Validation metrics below include all 2,331 observed targets, including anomalous prices.

| Forecast | Validation MAE | Validation RMSE |
|---|---:|---:|
| Latest observed persistence | 5.747 | 50.999 |
| Causal robust persistence | 3.289 | 36.107 |
| Ridge, alpha 10 | 3.329 | 36.107 |

Ridge validation MAE by asset: quantum 2.425, byte 7.653, gold 2.891, metro 1.001, agro 1.986, solar 4.040. The large RMSE values reflect observed scale anomalies; no validation targets were hidden or corrected.

Ridge was frozen as the submission forecast because robust persistence has almost no signal for ordinary live orders. This costs about 1.2% validation MAE and is a trade-off for the 70% live-profit component. The synthetic four-opponent harness produced mixed development results before final refitting; it is weak evidence of competitive trading. No broad parameter search was run.

## Artifact and verification

The selected model is per-asset Ridge on next-price change from the causal reference, with per-asset StandardScaler means/scales and coefficient arrays stored in `cudies_coding/model.npz`. The feature names and asset order are checked on load. Training can be reproduced with `python cudies_coding/train_model.py`; the final refit command is `python cudies_coding/train_model.py --final-refit`, to be used only after recording the untouched holdout result.

Nine focused data and forecast tests passed, including feature parity, target isolation, empty/short/missing history, finite forecasts, and budget-valid native integer P0 orders. The frozen forecast interface is available for Astra's allocation policy. The current packaged allocator remains P0; P1 was not implemented or validated before the submission buffer.

