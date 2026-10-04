"""Chronological forecast comparison and compact artifact export."""

import json
import sys
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from agent import ASSETS, FEATURE_NAMES, PARTNERS, _valid_price, make_features
from clean_data import load_history


DATA_PATH = Path(__file__).with_name("history.csv")
MODEL_PATH = Path(__file__).with_name("model.npz")
RIDGE_ALPHA = 10.0


def fit_reference_limits(prices, fit_end):
    """Estimate conservative spike thresholds using fitting prices only."""
    limits = {}
    for asset in ASSETS:
        values = prices[asset].iloc[:fit_end].tolist()
        ratios = []
        for round_number in range(2, len(values)):
            window = values[max(0, round_number - 19):round_number + 1]
            valid = [_valid_price(value) for value in window]
            valid = [value for value in valid if value is not None]
            if len(valid) < 3:
                continue
            recent = float(median(valid[-3:]))
            ratios.append(max(valid[-1] / recent, recent / valid[-1]))
        limits[asset] = max(1.5, 2.0 * float(np.quantile(ratios, 0.98)))
    return limits


def build_examples(prices, asset, start_round, stop_round, ratio_limits=None):
    """Use only rounds before each observed target to create examples."""
    own_prices = prices[asset].tolist()
    partner_prices = prices[PARTNERS[asset]].tolist()
    examples = []
    for target_round in range(max(start_round, 1), stop_round):
        target = _valid_price(own_prices[target_round])
        if target is None:
            continue

        first_round = max(0, target_round - 20)
        own_history = own_prices[first_round:target_round]
        history = {
            asset: own_history,
            PARTNERS[asset]: partner_prices[first_round:target_round],
        }
        reference, features, observed_count = make_features(
            history, asset, ratio_limits
        )
        if observed_count < 3:
            continue
        last_observed = reference
        for value in reversed(own_history):
            price = _valid_price(value)
            if price is not None:
                last_observed = price
                break
        examples.append(
            {
                "round": target_round,
                "features": features,
                "reference": reference,
                "last_observed": last_observed,
                "target": target,
            }
        )
    return examples


def score_predictions(examples_by_asset, predictions_by_asset):
    absolute_errors = []
    squared_errors = []
    per_asset = {}
    directional_profit = {}
    for asset in ASSETS:
        examples = examples_by_asset[asset]
        predictions = predictions_by_asset[asset]
        targets = np.array([example["target"] for example in examples])
        errors = predictions - targets
        absolute_errors.extend(np.abs(errors).tolist())
        squared_errors.extend((errors ** 2).tolist())
        per_asset[asset] = {
            "count": len(examples),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
        }

        profits = []
        hits = []
        for example, prediction in zip(examples, predictions):
            current = example["last_observed"]
            predicted_change = prediction - current
            actual_change = example["target"] - current
            if predicted_change != 0:
                direction = 1 if predicted_change > 0 else -1
                profits.append(direction * actual_change)
                hits.append(direction * actual_change > 0)
        directional_profit[asset] = {
            "one_lot_raw_profit": float(sum(profits)),
            "direction_hit_rate": float(np.mean(hits)) if hits else None,
            "trades": len(profits),
        }

    return {
        "count": len(absolute_errors),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "per_asset": per_asset,
        "directional": directional_profit,
    }


def main():
    prices, _ = load_history(DATA_PATH)
    fit_end = int(len(prices) * 0.60)
    validation_end = int(len(prices) * 0.80)
    ratio_limits = fit_reference_limits(prices, fit_end)
    validation_examples = {}
    ridge_predictions = {}
    median_predictions = {}
    raw_predictions = {}
    coefficients = []
    intercepts = []
    centers = []
    scales = []
    training_coverage = {}

    for asset in ASSETS:
        fit = build_examples(prices, asset, 1, fit_end, ratio_limits)
        validation = build_examples(
            prices, asset, fit_end, validation_end, ratio_limits
        )
        validation_examples[asset] = validation

        # This threshold sees fitting targets only. It removes grossly
        # implausible training labels without hiding validation anomalies.
        fit_ratios = np.array([
            max(example["target"] / example["reference"],
                example["reference"] / example["target"])
            for example in fit
        ])
        ratio_limit = ratio_limits[asset]
        retained = [example for example, ratio in zip(fit, fit_ratios)
                    if ratio <= ratio_limit]
        training_coverage[asset] = {
            "available_targets": len(fit),
            "retained_targets": len(retained),
            "ratio_limit": ratio_limit,
        }

        features = np.array([example["features"] for example in retained])
        changes = np.array([
            example["target"] - example["reference"] for example in retained
        ])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(scaled_features, changes)
        coefficients.append(model.coef_)
        intercepts.append(model.intercept_)
        centers.append(scaler.mean_)
        scales.append(scaler.scale_)

        validation_features = np.array(
            [example["features"] for example in validation]
        )
        prediction = np.maximum(
            0.01,
            np.array([example["reference"] for example in validation])
            + model.predict(scaler.transform(validation_features)),
        )
        ridge_predictions[asset] = prediction
        median_predictions[asset] = np.array(
            [example["reference"] for example in validation]
        )
        raw_predictions[asset] = np.array(
            [example["last_observed"] for example in validation]
        )

    scores = {
        "raw_persistence": score_predictions(validation_examples, raw_predictions),
        "median_persistence": score_predictions(validation_examples, median_predictions),
        "ridge": score_predictions(validation_examples, ridge_predictions),
    }
    # The live score carries more weight than offline accuracy. Median
    # persistence is more accurate here but produces almost no normal trades.
    # Ridge costs 1.2% validation MAE and supplies an actual trading signal.
    selected = "ridge"

    np.savez_compressed(
        MODEL_PATH,
        kind=np.array(selected),
        assets=np.array(ASSETS),
        feature_names=np.array(FEATURE_NAMES),
        ratio_limits=np.array([ratio_limits[asset] for asset in ASSETS]),
        coefficients=np.array(coefficients),
        intercepts=np.array(intercepts),
        centers=np.array(centers),
        scales=np.array(scales),
    )
    report = {
        "fit_target_rounds": [1, fit_end - 1],
        "validation_target_rounds": [fit_end, validation_end - 1],
        "untouched_holdout_target_rounds": [validation_end, len(prices) - 1],
        "ridge_alpha": RIDGE_ALPHA,
        "training_coverage": training_coverage,
        "scores": scores,
        "selected": selected,
        "artifact": str(MODEL_PATH),
    }
    print(json.dumps(report, indent=2))


def export_final_model():
    """Refit the frozen Ridge setup after the final holdout was reported."""
    prices, _ = load_history(DATA_PATH)
    ratio_limits = fit_reference_limits(prices, len(prices))
    coefficients = []
    intercepts = []
    centers = []
    scales = []
    retained_counts = {}
    for asset in ASSETS:
        examples = build_examples(prices, asset, 1, len(prices), ratio_limits)
        retained = [
            example for example in examples
            if max(
                example["target"] / example["reference"],
                example["reference"] / example["target"],
            ) <= ratio_limits[asset]
        ]
        retained_counts[asset] = len(retained)
        features = np.array([example["features"] for example in retained])
        changes = np.array([
            example["target"] - example["reference"] for example in retained
        ])
        scaler = StandardScaler()
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(scaler.fit_transform(features), changes)
        coefficients.append(model.coef_)
        intercepts.append(model.intercept_)
        centers.append(scaler.mean_)
        scales.append(scaler.scale_)

    np.savez_compressed(
        MODEL_PATH,
        kind=np.array("ridge"),
        assets=np.array(ASSETS),
        feature_names=np.array(FEATURE_NAMES),
        ratio_limits=np.array([ratio_limits[asset] for asset in ASSETS]),
        coefficients=np.array(coefficients),
        intercepts=np.array(intercepts),
        centers=np.array(centers),
        scales=np.array(scales),
    )
    print(json.dumps({"artifact": str(MODEL_PATH), "retained_counts": retained_counts}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--final-refit"]:
        export_final_model()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: train_model.py [--final-refit]")

