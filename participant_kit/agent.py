"""
Neural Nexus -- Round 2: The Trading Pit
agent.py starter template

Fill in predict_market() and place_orders(). This file already runs
end-to-end with a dummy (non-learning) strategy so you always have
something submittable -- replace the TODO sections with your real model.

IMPORTANT: both functions are called FRESH each round, in a new process.
Nothing stored in a global variable or on `self` will survive between
calls. Anything you need must come from the arguments, or be loaded from
a file (e.g. joblib.load("my_model.pkl")) at the top of this file.
"""

import os
import joblib  # if you saved a trained model with joblib.dump(...)
import numpy as np

ASSETS = ["quantum_dynamics", "byte_stream", "gold_trust",
          "metro_rail", "agro_futures", "solar_grid"]

# TODO: load your trained model once, at import time (NOT inside the
# functions below -- this keeps per-round execution fast).
#
# IMPORTANT: use a path relative to THIS FILE, not a bare filename. The
# real engine imports your agent.py from a different working directory
# than your own folder, so a bare "my_model.pkl" will work when you test
# locally (local_test_harness.py happens to run from your folder) but will
# FAIL SILENTLY on the real engine. Always do it this way:

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
try:
    MODELS = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model.pkl. Did you run train_model.py? ({e})")
    MODELS = None


def predict_market(recent_history):
    """
    recent_history: dict[asset_id -> list of up to the last 20 REALIZED
                    prices for that asset, oldest first, most recent last]

    Returns your predicted PRICE for the UPCOMING round, per asset, e.g.:
      {"quantum_dynamics": 41.2, "byte_stream": 77.5, "gold_trust": 33.0,
       "metro_rail": 22.4, "agro_futures": 28.1, "solar_grid": 39.6}

    Scored OFFLINE against a hidden validation set you never see. It is NOT
    told what other teams are doing -- it's a pure forecasting task.
    """
    predictions = {}
    for asset in ASSETS:
        prices = recent_history.get(asset, [])
        
        # If we don't have enough history yet (or no model), guess 40.0
        if len(prices) < 20 or MODELS is None:
             predictions[asset] = 40.0
        else:
             # =================================================================
             # CRITICAL: Feature Engineering must match train_model.py!
             # If you changed the window size to 10 or added new features 
             # (like moving averages) in train_model.py, you MUST do the exact 
             # same calculations here before passing the data to your model.
             # =================================================================
             X = np.array(prices[-20:]).reshape(1, -1)
             
             # Predict! (Update this line if using PyTorch/TensorFlow)
             predictions[asset] = MODELS[asset].predict(X)[0]

    return predictions


def place_orders(game_state, team_history):
    """
    game_state: {
      "round": 47, "total_rounds": 120, "trade_budget": 14,
      "assets": {
        "quantum_dynamics": {"ticker": "QNTM",
                              "recent_prices": [...up to last 20...],
                              "recent_buy_volume":  [...total BUY lots from ALL teams, same rounds...],
                              "recent_sell_volume": [...total SELL lots from ALL teams, same rounds...]},
        ... (all 6 assets)
      },
      "your_score_so_far": 312.5,
    }
    
    team_history: list of your own past rounds:
      [{"round": 46, "orders": {...}, "profit": 41.0}, ...]

    Returns your lot allocation for THIS round -- SIGNED integers, one per
    asset (positive = buy, negative = short, zero = sit out), where the
    SUM OF ABSOLUTE VALUES is at most game_state["trade_budget"]:
      {"quantum_dynamics": 4, "byte_stream": -3, "gold_trust": 0,
       "metro_rail": 5, "agro_futures": -2, "solar_grid": 0}
      (here: total exposure = 4+3+0+5+2+0 = 14, right at budget)
    """
    budget = game_state["trade_budget"]

    # Build a recent_history dict in the same shape predict_market() expects,
    # so you can reuse your forecast to decide which way to trade.
    recent_history = {
        asset: game_state["assets"][asset]["recent_prices"]
        for asset in ASSETS
    }
    predicted = predict_market(recent_history)

    # TODO: replace this with your real allocation logic. Dummy baseline:
    # go long or short based on predicted direction, sized proportionally
    # to how big a move you're predicting (bigger predicted move = bigger
    # position, i.e. simple conviction sizing).
    
    # Calculate expected price changes (Prediction - Current Price)
    predicted_move = {}
    for asset in ASSETS:
        last_price = recent_history[asset][-1] if recent_history[asset] else predicted[asset]
        predicted_move[asset] = predicted[asset] - last_price

    # Allocate budget across ALL assets proportionally based on conviction.
    # (Bigger predicted move = bigger bet).
    total_conviction = sum(abs(v) for v in predicted_move.values()) or 1.0
    orders = {}
    remaining = budget
    for asset in ASSETS[:-1]:
        size = min(round(abs(predicted_move[asset]) / total_conviction * budget), remaining)
        direction = 1 if predicted_move[asset] > 0 else (-1 if predicted_move[asset] < 0 else 0)
        orders[asset] = direction * size
        remaining -= size
    # give whatever budget is left to the last asset, same direction logic
    last_asset = ASSETS[-1]
    direction = 1 if predicted_move[last_asset] > 0 else (-1 if predicted_move[last_asset] < 0 else 0)
    orders[last_asset] = direction * remaining

    return orders


if __name__ == "__main__":
    # Quick manual sanity check -- run this file directly to eyeball output.
    fake_history = {a: [40, 42, 39, 41, 43] for a in ASSETS}
    print("predict_market ->", predict_market(fake_history))

    fake_game_state = {
        "round": 5, "total_rounds": 120, "trade_budget": 14,
        "assets": {a: {"ticker": a[:4].upper(), "recent_prices": [40, 42, 39, 41, 43],
                        "recent_buy_volume": [20, 22, 19, 21, 23],
                        "recent_sell_volume": [18, 17, 20, 19, 16]} for a in ASSETS},
        "your_score_so_far": 0.0,
    }
    print("place_orders  ->", place_orders(fake_game_state, []))
