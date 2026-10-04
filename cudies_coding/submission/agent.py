"""Price forecasts and a simple, budget-valid trading baseline."""

import math
import sys
from numbers import Integral
from pathlib import Path
from statistics import median


ASSETS = (
    "quantum_dynamics",
    "byte_stream",
    "gold_trust",
    "metro_rail",
    "agro_futures",
    "solar_grid",
)
PARTNERS = {
    "quantum_dynamics": "byte_stream",
    "byte_stream": "quantum_dynamics",
    "gold_trust": "solar_grid",
    "solar_grid": "gold_trust",
    "metro_rail": "agro_futures",
    "agro_futures": "metro_rail",
}
DEFAULT_PRICES = {
    "quantum_dynamics": 50.0,
    "byte_stream": 60.0,
    "gold_trust": 40.0,
    "metro_rail": 30.0,
    "agro_futures": 35.0,
    "solar_grid": 45.0,
}
FEATURE_NAMES = (
    "reference_relative_to_default",
    "recent_median_change",
    "earlier_median_change",
    "short_vs_medium_median",
    "short_vs_long_median",
    "recent_median_deviation",
    "last_price_deviation",
    "last_change",
    "recent_missing_fraction",
    "partner_recent_median_change",
    "partner_short_vs_medium_median",
    "partner_last_price_deviation",
    "partner_last_change",
    "partner_recent_missing_fraction",
)
MODEL_PATH = Path(__file__).with_name("model.npz")
DEFAULT_RATIO_LIMIT = 2.5
_model = None
_model_loaded = False


def _valid_price(value):
    if isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _window_summary(values, default_price, ratio_limit):
    if values is None:
        values = []
    try:
        window = values[-20:]
    except (TypeError, KeyError) as error:
        raise ValueError("Price history must be an ordered sequence") from error

    observed = [_valid_price(value) for value in window]
    valid = [price for price in observed if price is not None]
    if not valid:
        return default_price, [0.0] * 7, 1.0, 0

    recent = float(median(valid[-3:]))
    latest = valid[-1]
    ratio = max(latest / recent, recent / latest)
    reference = latest if ratio <= ratio_limit else recent
    previous = float(median(valid[-6:-3])) if len(valid) > 3 else recent
    earlier = float(median(valid[-9:-6])) if len(valid) > 6 else previous
    medium = float(median(valid[-5:]))
    long = float(median(valid[-10:]))
    recent_deviation = float(median(abs(price - recent) for price in valid[-5:]))
    last_deviation = math.tanh((latest - recent) / default_price)
    last_change = (
        math.tanh((latest - valid[-2]) / default_price)
        if len(valid) > 1 and ratio <= ratio_limit
        else 0.0
    )
    missing_fraction = observed[-5:].count(None) / len(observed[-5:])

    values = [
        (recent - previous) / default_price,
        (previous - earlier) / default_price,
        (recent - medium) / default_price,
        (recent - long) / default_price,
        recent_deviation / default_price,
        last_deviation,
        last_change,
    ]
    return reference, values, missing_fraction, len(valid)


def make_features(recent_history, asset, ratio_limits=None):
    """Return causal (reference_price, ordered_features, observed_count)."""
    if asset not in DEFAULT_PRICES:
        raise ValueError(f"Unknown asset: {asset}")
    if not hasattr(recent_history, "get"):
        raise ValueError("recent_history must map assets to price sequences")

    default_price = DEFAULT_PRICES[asset]
    own_limit = ratio_limits[asset] if ratio_limits is not None else DEFAULT_RATIO_LIMIT
    own_reference, own_values, own_missing, own_count = _window_summary(
        recent_history.get(asset), default_price, own_limit
    )
    partner = PARTNERS[asset]
    partner_limit = ratio_limits[partner] if ratio_limits is not None else DEFAULT_RATIO_LIMIT
    _, partner_values, partner_missing, _ = _window_summary(
        recent_history.get(partner), DEFAULT_PRICES[partner], partner_limit
    )
    features = [
        (own_reference - default_price) / default_price,
        own_values[0],
        own_values[1],
        own_values[2],
        own_values[3],
        own_values[4],
        own_values[5],
        own_values[6],
        own_missing,
        partner_values[0],
        partner_values[2],
        partner_values[5],
        partner_values[6],
        partner_missing,
    ]
    return own_reference, features, own_count


def _load_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model

    try:
        import numpy as np

        with np.load(MODEL_PATH, allow_pickle=False) as artifact:
            if tuple(artifact["assets"].tolist()) != ASSETS:
                raise ValueError("Asset order differs from agent.py")
            if tuple(artifact["feature_names"].tolist()) != FEATURE_NAMES:
                raise ValueError("Feature order differs from agent.py")
            kind = str(artifact["kind"].item())
            if kind not in {"ridge", "persistence"}:
                raise ValueError(f"Unsupported model kind: {kind}")
            ratio_limits = artifact["ratio_limits"]
            if (ratio_limits.shape != (len(ASSETS),)
                    or not np.isfinite(ratio_limits).all()
                    or np.any(ratio_limits <= 1)):
                raise ValueError("Invalid ratio limits in forecast artifact")
            model = {
                "kind": kind,
                "ratio_limits": dict(zip(ASSETS, ratio_limits.tolist())),
            }
            if kind == "ridge":
                expected_shapes = {
                    "coefficients": (len(ASSETS), len(FEATURE_NAMES)),
                    "intercepts": (len(ASSETS),),
                    "centers": (len(ASSETS), len(FEATURE_NAMES)),
                    "scales": (len(ASSETS), len(FEATURE_NAMES)),
                }
                for key in ("coefficients", "intercepts", "centers", "scales"):
                    values = artifact[key]
                    if values.shape != expected_shapes[key] or not np.isfinite(values).all():
                        raise ValueError(f"Invalid {key} in forecast artifact")
                    model[key] = values.tolist()
                if np.any(artifact["scales"] <= 0):
                    raise ValueError("Forecast feature scales must be positive")
    except FileNotFoundError:
        print(f"Forecast artifact missing at {MODEL_PATH}; using persistence", file=sys.stderr)
        model = None
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError(f"Could not load forecast artifact {MODEL_PATH}: {error}") from error

    _model = model
    _model_loaded = True
    return _model


def predict_market(recent_history):
    """Return finite next-round price forecasts for all six assets."""
    model = _load_model()
    predictions = {}
    for index, asset in enumerate(ASSETS):
        limits = model["ratio_limits"] if model is not None else None
        reference, features, observed_count = make_features(
            recent_history, asset, limits
        )
        change = 0.0
        if model is not None and model["kind"] == "ridge" and observed_count >= 3:
            change = model["intercepts"][index]
            for position, feature in enumerate(features):
                center = model["centers"][index][position]
                scale = model["scales"][index][position]
                coefficient = model["coefficients"][index][position]
                change += coefficient * (feature - center) / scale
        price = reference + change
        if not math.isfinite(price):
            raise RuntimeError(f"Non-finite forecast for {asset}")
        predictions[asset] = float(max(0.01, price))
    return predictions


def place_orders(game_state, team_history):
    """Allocate the supplied budget proportionally to forecast changes (P0)."""
    orders = {asset: 0 for asset in ASSETS}
    if not isinstance(game_state, dict):
        print("Invalid game_state: expected a dictionary", file=sys.stderr)
        return orders
    budget = game_state.get("trade_budget")
    assets = game_state.get("assets")
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
        print(f"Invalid trade budget: {budget!r}", file=sys.stderr)
        return orders
    if not isinstance(assets, dict):
        print("Invalid game_state.assets: expected a dictionary", file=sys.stderr)
        return orders
    if budget == 0:
        return orders

    recent_history = {}
    for asset in ASSETS:
        asset_state = assets.get(asset)
        if not isinstance(asset_state, dict):
            print(f"Missing asset state for {asset}", file=sys.stderr)
            return orders
        history = asset_state.get("recent_prices", [])
        if history is not None:
            try:
                history[-20:]
            except (TypeError, KeyError) as error:
                print(f"Invalid price history for {asset}: {error}", file=sys.stderr)
                return orders
        recent_history[asset] = history

    predictions = predict_market(recent_history)
    signals = {}
    for asset in ASSETS:
        history = recent_history[asset]
        if history is None or len(history) == 0:
            continue
        current_price = _valid_price(history[-1])
        if current_price is not None:
            change = predictions[asset] - current_price
            if math.isfinite(change) and change != 0:
                signals[asset] = change
    total_signal = sum(abs(change) for change in signals.values())
    if total_signal == 0:
        return orders

    fractions = []
    used = 0
    for asset in ASSETS:
        change = signals.get(asset, 0.0)
        exact_size = budget * abs(change) / total_signal
        size = math.floor(exact_size)
        orders[asset] = int(math.copysign(size, change)) if size else 0
        used += size
        if change:
            fractions.append((exact_size - size, asset))

    asset_order = {asset: index for index, asset in enumerate(ASSETS)}
    fractions.sort(key=lambda item: (-item[0], asset_order[item[1]]))
    for _, asset in fractions[: budget - used]:
        orders[asset] += 1 if signals[asset] > 0 else -1
    return orders

