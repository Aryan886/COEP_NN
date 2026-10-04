"""Price forecasts and crowd-aware integer trading allocations."""

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
OPPONENT_COUNT = 20
FLOW_DECAY = 0.75
FLOW_PRIOR_STRENGTH = 4.0
UNALIGNED_CONFIDENCE = 0.5
FLOW_WEIGHTS = (0.50, 0.35, 0.15)
SIGNAL_SIDE_SHARE = 0.80
LIQUIDITY_MULTIPLIERS = (0.7, 1.0, 1.6)
REFERENCE_LIQUIDITY = {
    "quantum_dynamics": 50.0,
    "byte_stream": 60.0,
    "gold_trust": 40.0,
    "metro_rail": 30.0,
    "agro_futures": 35.0,
    "solar_grid": 45.0,
}
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


def _order_inputs(game_state):
    """Validate public inputs and abstain when the latest price is missing."""
    if not isinstance(game_state, dict):
        print("Invalid game_state: expected a dictionary", file=sys.stderr)
        return 0, {}
    budget = game_state.get("trade_budget")
    assets = game_state.get("assets")
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
        print(f"Invalid trade budget: {budget!r}", file=sys.stderr)
        return 0, {}
    if not isinstance(assets, dict):
        print("Invalid game_state.assets: expected a dictionary", file=sys.stderr)
        return 0, {}
    if budget == 0:
        return 0, {}

    recent_history = {}
    for asset in ASSETS:
        asset_state = assets.get(asset)
        if not isinstance(asset_state, dict):
            print(f"Missing asset state for {asset}", file=sys.stderr)
            return 0, {}
        history = asset_state.get("recent_prices", [])
        if history is not None:
            try:
                history[-20:]
            except (TypeError, KeyError) as error:
                print(f"Invalid price history for {asset}: {error}", file=sys.stderr)
                return 0, {}
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
    return int(budget), signals


def proportional_orders(signals, budget):
    """P0: stable largest-remainder allocation, with zero for zero signals."""
    orders = {asset: 0 for asset in ASSETS}
    largest_signal = max((abs(change) for change in signals.values()), default=0.0)
    if budget == 0 or largest_signal == 0:
        return orders
    weights = {asset: abs(change) / largest_signal for asset, change in signals.items()}
    total_signal = sum(weights.values())
    if total_signal == 0:
        return orders

    fractions = []
    used = 0
    for asset in ASSETS:
        change = signals.get(asset, 0.0)
        exact_size = budget * weights.get(asset, 0.0) / total_signal
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


def place_orders_baseline(game_state, team_history):
    budget, signals = _order_inputs(game_state)
    return proportional_orders(signals, budget)


def _volume_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def estimate_opponent_flow(game_state, team_history, budget):
    """Shrink paired past volumes toward Astra's neutral 20-opponent prior."""
    prior = OPPONENT_COUNT * budget / (2 * len(ASSETS))
    own_records = {}
    duplicates = set()
    if team_history is None:
        team_history = []
    if not isinstance(team_history, (list, tuple)):
        print("Invalid team_history: using unaligned volume estimates", file=sys.stderr)
        team_history = []
    for record in team_history:
        if not isinstance(record, dict):
            continue
        round_number = record.get("round")
        if isinstance(round_number, bool) or not isinstance(round_number, Integral):
            continue
        if round_number in own_records:
            duplicates.add(round_number)
        own_records[round_number] = record.get("orders", {})
    for round_number in duplicates:
        del own_records[round_number]

    current_round = game_state.get("round")
    valid_round = isinstance(current_round, Integral) and not isinstance(current_round, bool)
    flows = {}
    for asset in ASSETS:
        state = game_state["assets"][asset]
        buys = state.get("recent_buy_volume", [])
        sells = state.get("recent_sell_volume", [])
        if buys is None or sells is None:
            buys, sells = [], []
        try:
            if len(buys) != len(sells):
                print(f"Unpaired volume histories for {asset}; using neutral prior", file=sys.stderr)
                buys, sells = [], []
            buys, sells = buys[-20:], sells[-20:]
        except (TypeError, KeyError) as error:
            print(f"Invalid volume history for {asset}: {error}", file=sys.stderr)
            buys, sells = [], []

        count = len(buys)
        # The harness ends both contiguous volume arrays at current_round - 1.
        # Require matching, contiguous own round IDs before using that mapping.
        aligned = valid_round and current_round >= count and all(
            number in own_records for number in range(current_round - count, current_round)
        )
        weight_sum = 0.0
        squared_weight_sum = 0.0
        weighted_buys = 0.0
        weighted_sells = 0.0
        weighted_confidence = 0.0
        for position, (buy_value, sell_value) in enumerate(zip(buys, sells)):
            buy = _volume_number(buy_value)
            sell = _volume_number(sell_value)
            if buy is None or sell is None:
                continue
            confidence = UNALIGNED_CONFIDENCE
            opponent_buy = buy * OPPONENT_COUNT / (OPPONENT_COUNT + 1)
            opponent_sell = sell * OPPONENT_COUNT / (OPPONENT_COUNT + 1)
            if aligned:
                round_number = current_round - count + position
                own_orders = own_records[round_number]
                own_order = own_orders.get(asset) if isinstance(own_orders, dict) else None
                if isinstance(own_order, Integral) and not isinstance(own_order, bool):
                    corrected_buy = buy - max(own_order, 0)
                    corrected_sell = sell - max(-own_order, 0)
                    tolerance = 1e-8 * max(1.0, buy, sell, abs(own_order))
                    if min(corrected_buy, corrected_sell) >= -tolerance:
                        opponent_buy = max(0.0, corrected_buy)
                        opponent_sell = max(0.0, corrected_sell)
                        confidence = 1.0
                    else:
                        print(
                            f"Own order exceeds aggregate volume for {asset} round {round_number}; "
                            "using unaligned estimate",
                            file=sys.stderr,
                        )
            age = count - 1 - position
            weight = FLOW_DECAY ** age
            weight_sum += weight
            squared_weight_sum += weight * weight
            weighted_buys += weight * opponent_buy
            weighted_sells += weight * opponent_sell
            weighted_confidence += weight * confidence

        if weight_sum:
            effective_count = weight_sum * weight_sum / squared_weight_sum
            confidence = weighted_confidence / weight_sum
            share = effective_count / (effective_count + FLOW_PRIOR_STRENGTH) * confidence
            buy = share * weighted_buys / weight_sum + (1 - share) * prior
            sell = share * weighted_sells / weight_sum + (1 - share) * prior
            flows[asset] = (buy, sell)
        else:
            flows[asset] = (prior, prior)

    total = sum(buy + sell for buy, sell in flows.values())
    limit = OPPONENT_COUNT * budget
    if total > limit and total > 0:
        scale = limit / total
        flows = {asset: (buy * scale, sell * scale) for asset, (buy, sell) in flows.items()}
    return flows


def build_scenarios(signals, flows, budget):
    """Return nine fixed (buy, sell, liquidity, weight) worlds per asset."""
    prior = OPPONENT_COUNT * budget / (2 * len(ASSETS))
    largest = max((abs(value) for value in signals.values()), default=0.0)
    weights = {asset: abs(value) / largest for asset, value in signals.items()} if largest else {}
    total_weight = sum(weights.values())
    scenarios = {}
    for asset in ASSETS:
        buy, sell = flows[asset]
        if total_weight:
            signal_total = OPPONENT_COUNT * budget * weights.get(asset, 0.0) / total_weight
            buy_share = SIGNAL_SIDE_SHARE if signals.get(asset, 0.0) > 0 else 1 - SIGNAL_SIDE_SHARE
            signal_buy = signal_total * buy_share
            signal_sell = signal_total * (1 - buy_share)
        else:
            signal_buy, signal_sell = prior, prior
        worlds = ((buy, sell), (signal_buy, signal_sell), (sell, buy))
        scenarios[asset] = []
        for (world_buy, world_sell), world_weight in zip(worlds, FLOW_WEIGHTS):
            for multiplier in LIQUIDITY_MULTIPLIERS:
                liquidity = 0.6 * REFERENCE_LIQUIDITY[asset] * multiplier
                scenarios[asset].append(
                    (world_buy, world_sell, liquidity, world_weight / len(LIQUIDITY_MULTIPLIERS))
                )
    return scenarios


def score_order(order, change, opponent_buy, opponent_sell, liquidity):
    """Profit without the common P&L scale; buys win dominant-side ties."""
    values = (change, opponent_buy, opponent_sell, liquidity)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Order scenario contains a non-finite value")
    if min(opponent_buy, opponent_sell) < 0:
        raise ValueError("Opponent buy and sell volumes must be nonnegative")
    if order == 0:
        return 0.0
    buy = opponent_buy + max(order, 0)
    sell = opponent_sell + max(-order, 0)
    dominant = max(buy, sell)
    # Match the organizer harness's defensive treatment of nonpositive liquidity.
    discount = min(1.0, liquidity / dominant) if liquidity > 0 else 1.0
    our_side_dominant = (order > 0 and buy >= sell) or (order < 0 and sell > buy)
    return order * change * (discount if our_side_dominant else 1.0)


def _better_utility(candidate, current):
    if current == -math.inf:
        return candidate != -math.inf
    tolerance = 1e-10 * max(1.0, abs(candidate), abs(current))
    return candidate > current + tolerance


def optimize_orders(utilities, budget):
    """Exact separable integer allocation; ties prefer less total exposure."""
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
        raise ValueError("Optimizer budget must be a nonnegative integer")
    assets = list(utilities)
    previous = [-math.inf] * (budget + 1)
    previous[0] = 0.0
    choices = []
    for asset in assets:
        table = utilities[asset]
        if table.get(0) != 0 or not all(math.isfinite(value) for value in table.values()):
            raise ValueError(f"Utilities for {asset} must be finite and include zero utility at zero")
        candidates = sorted(table, key=lambda order: (abs(order), order))
        current = [-math.inf] * (budget + 1)
        selected = [0] * (budget + 1)
        for exposure, previous_utility in enumerate(previous):
            if previous_utility == -math.inf:
                continue
            for order in candidates:
                next_exposure = exposure + abs(order)
                if next_exposure > budget:
                    continue
                utility = previous_utility + table[order]
                if _better_utility(utility, current[next_exposure]):
                    current[next_exposure] = utility
                    selected[next_exposure] = order
        previous = current
        choices.append(selected)

    best_exposure = 0
    for exposure in range(1, budget + 1):
        if _better_utility(previous[exposure], previous[best_exposure]):
            best_exposure = exposure
    orders = {}
    for position in range(len(assets) - 1, -1, -1):
        order = int(choices[position][best_exposure])
        orders[assets[position]] = order
        best_exposure -= abs(order)
    return {asset: orders[asset] for asset in assets}


def crowd_orders(game_state, team_history, budget, signals):
    if budget == 0 or not signals:
        return {asset: 0 for asset in ASSETS}
    flows = estimate_opponent_flow(game_state, team_history, budget)
    scenarios = build_scenarios(signals, flows, budget)
    utilities = {}
    for asset in ASSETS:
        table = {0: 0.0}
        if asset in signals:
            for order in range(-budget, budget + 1):
                if order:
                    table[order] = sum(
                        weight * score_order(order, signals[asset], buy, sell, liquidity)
                        for buy, sell, liquidity, weight in scenarios[asset]
                    )
        utilities[asset] = table
    return optimize_orders(utilities, budget)


def place_orders_crowd(game_state, team_history):
    """Astra P1: uncertain opponent flow, nine scenarios, exact budget DP."""
    budget, signals = _order_inputs(game_state)
    return crowd_orders(game_state, team_history, budget, signals)


def place_orders(game_state, team_history):
    return place_orders_crowd(game_state, team_history)

