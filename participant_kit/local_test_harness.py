"""
Neural Nexus -- Round 2: The Trading Pit
local_test_harness.py

Run your agent.py locally against a few built-in dummy opponents, using
the SAME scoring formula as RULES.md, before you submit. Two differences
from the real engine, both flagged inline below: (1) this uses try/except
instead of real process isolation/timeouts, for faster iteration, and (2)
liquidity is scaled down further here than in production, because with
only a handful of dummy opponents instead of 20 real teams there isn't
enough order volume to ever trigger crowding otherwise -- this keeps that
part of the game testable locally. Everything else matches production
exactly.

IMPORTANT GOTCHA THIS HARNESS WILL NOT CATCH: if your agent.py loads a
model file using a bare filename (e.g. joblib.load("my_model.pkl")) it
will work here, because this harness runs from your own folder by
coincidence -- but it will FAIL on the real judging engine, which imports
your agent.py from a different working directory. See RULES.md for the
correct pattern (a path built from __file__). Test this specifically by
running this harness from a DIFFERENT directory, e.g.:
    cd .. && python3 my_submission_folder/local_test_harness.py my_submission_folder/agent.py

Usage:
    python3 local_test_harness.py                  # tests ./agent.py
    python3 local_test_harness.py path/to/agent.py  # tests a specific file
"""

import sys
import random
import importlib.util
import statistics
from collections import deque

ASSETS = ["quantum_dynamics", "byte_stream", "gold_trust",
          "metro_rail", "agro_futures", "solar_grid"]
CLUSTERS = {"tech": ["quantum_dynamics", "byte_stream"],
            "hard_assets": ["gold_trust", "solar_grid"],
            "local_econ": ["metro_rail", "agro_futures"]}
CLUSTER_OF = {a: c for c, members in CLUSTERS.items() for a in members}
CLUSTER_WEIGHT = 0.5

BASELINE_PRICE = {"quantum_dynamics": 50, "byte_stream": 60, "gold_trust": 40,
                   "metro_rail": 30, "agro_futures": 35, "solar_grid": 45}
BASE_LIQUIDITY = dict(BASELINE_PRICE)
EXEC_LIQUIDITY_SCALE = 0.6   # tuned so crowding triggers ~45% of the time at 20-team production scale
LOCAL_TEST_LIQUIDITY_SCALE = 0.3  # ADDITIONAL scale-down for local testing: with only a handful of
                                    # dummy opponents instead of 20 real teams, total order volume is
                                    # much lower, so crowding would almost never trigger (verified: 0.3%
                                    # vs 25%+ at 20 teams) without this -- this restores a locally
                                    # representative crowding rate so you can actually test how your
                                    # agent handles the short-selling / crowd-avoidance dynamic.

SESSION_MULT = {"quiet": 0.7, "regular": 1.0, "volatile": 1.6}
SESSION_ORDER = ["quiet", "regular", "volatile", "regular"]
SESSION_PERIOD, SESSION_JITTER = 20, 3

DRIFT_RHO = 0.85
MOMENTUM_STD = 0.9
STEP_NOISE_FRAC = 0.06
REVERSION_STRENGTH = 0.35   # keeps prices in a realistic band instead of drifting unboundedly
PNL_SCALE = 3.0

TRADE_BUDGET = 14
N_ROUNDS = 150         # matches the 150-round live show

# How many dummy opponents to test against locally. The real event has 20
# teams; production liquidity is already tuned for that -- this harness
# uses the same constants, so results should feel broadly representative
# even with fewer opponents (unlike the old unsigned-order design, dilution
# here depends on absolute order volume, not team count, so no extra
# scaling trick is needed).
N_DUMMY_OPPONENTS = 4


def make_price_schedule(n_rounds, seed):
    rng = random.Random(seed)
    session_idx, round_in_session = 0, 0
    next_shift = SESSION_PERIOD + rng.randint(-SESSION_JITTER, SESSION_JITTER)
    price = dict(BASELINE_PRICE)
    drift = {a: 0.0 for a in ASSETS}
    cluster_shock = {c: 0.0 for c in CLUSTERS}
    schedule = []
    for _ in range(n_rounds):
        round_in_session += 1
        if round_in_session >= next_shift:
            session_idx = (session_idx + 1) % len(SESSION_ORDER)
            round_in_session = 0
            next_shift = SESSION_PERIOD + rng.randint(-SESSION_JITTER, SESSION_JITTER)
        session = SESSION_ORDER[session_idx]
        vol_mult = SESSION_MULT[session]

        for c in CLUSTERS:
            cluster_shock[c] = DRIFT_RHO * cluster_shock[c] + rng.gauss(0, MOMENTUM_STD)

        true_move = {}
        for a in ASSETS:
            idiosync_shock = rng.gauss(0, MOMENTUM_STD)
            drift[a] = DRIFT_RHO * drift[a] + (CLUSTER_WEIGHT * cluster_shock[CLUSTER_OF[a]]
                                                + (1 - CLUSTER_WEIGHT) * idiosync_shock)
            noise = rng.gauss(0, BASELINE_PRICE[a] * STEP_NOISE_FRAC * vol_mult)
            reversion = REVERSION_STRENGTH * (BASELINE_PRICE[a] - price[a])
            move = drift[a] + noise + reversion
            price[a] = max(price[a] + move, 5.0)
            true_move[a] = move
        schedule.append({"price": dict(price), "true_move": dict(true_move), "session": session})
    return schedule


def resolve_round(orders, true_move, session):
    """orders: dict[team -> dict[asset -> signed int lots]]"""
    total_buy = {a: 0 for a in ASSETS}
    total_sell = {a: 0 for a in ASSETS}
    for team_orders in orders.values():
        for a in ASSETS:
            lots = team_orders.get(a, 0)
            if lots > 0:
                total_buy[a] += lots
            elif lots < 0:
                total_sell[a] += -lots

    exec_liquidity = {a: BASE_LIQUIDITY[a] * SESSION_MULT[session] * EXEC_LIQUIDITY_SCALE * LOCAL_TEST_LIQUIDITY_SCALE
                       for a in ASSETS}
    dominant_side, discount_dominant = {}, {}
    for a in ASSETS:
        if total_buy[a] >= total_sell[a]:
            dominant_side[a], dominant_total = "buy", total_buy[a]
        else:
            dominant_side[a], dominant_total = "sell", total_sell[a]
        imbalance = dominant_total / exec_liquidity[a] if exec_liquidity[a] > 0 else 0
        discount_dominant[a] = 1.0 if imbalance <= 1 else 1.0 / imbalance

    revenue = {team: 0.0 for team in orders}
    for team, team_orders in orders.items():
        for a in ASSETS:
            lots = team_orders.get(a, 0)
            if lots == 0:
                continue
            side = "buy" if lots > 0 else "sell"
            discount = discount_dominant[a] if side == dominant_side[a] else 1.0
            revenue[team] += lots * true_move[a] * PNL_SCALE * discount

    return revenue, total_buy, total_sell


# --- built-in dummy opponents -----------------------------------------------
def _clip_to_budget(lots, budget):
    total = sum(abs(v) for v in lots.values())
    if total <= budget or total == 0:
        return lots
    scale = budget / total
    return {a: int(v * scale) for a, v in lots.items()}


class RandomDummy:
    name = "random_dummy"
    def place_orders(self, game_state, team_history):
        lots = {a: random.randint(-2, 2) for a in ASSETS}
        return _clip_to_budget(lots, TRADE_BUDGET)


class StaticFlatDummy:
    name = "static_flat_dummy"
    def place_orders(self, game_state, team_history):
        return {a: 0 for a in ASSETS}


class NaiveMomentumDummy:
    """Chases whatever direction an asset last moved -- a common first
    instinct, and deliberately not very good. Your trained agent should
    beat this comfortably."""
    name = "naive_momentum_dummy"
    def place_orders(self, game_state, team_history):
        lots = {}
        for a in ASSETS:
            prices = game_state["assets"][a]["recent_prices"]
            last_move = (prices[-1] - prices[-2]) if len(prices) >= 2 else 0
            lots[a] = 2 if last_move > 0 else (-2 if last_move < 0 else 0)
        return _clip_to_budget(lots, TRADE_BUDGET)


class ProportionalLongDummy:
    """Always goes long a little on everything, never shorts -- represents
    a team that didn't engage with the short-selling option at all."""
    name = "proportional_long_dummy"
    def place_orders(self, game_state, team_history):
        per_asset = TRADE_BUDGET // len(ASSETS)
        lots = {a: per_asset for a in ASSETS}
        remainder = TRADE_BUDGET - per_asset * len(ASSETS)
        if remainder:
            lots[ASSETS[0]] += remainder
        return lots


DUMMY_POOL = [RandomDummy, StaticFlatDummy, NaiveMomentumDummy, ProportionalLongDummy]


def load_agent(path):
    spec = importlib.util.spec_from_file_location("agent_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_local_match(agent_mod, seed):
    schedule = make_price_schedule(N_ROUNDS, seed=seed)
    dummies = {f"dummy_{i}_{cls.name}": cls() for i, cls in enumerate(DUMMY_POOL[:N_DUMMY_OPPONENTS])}
    scores = {name: 0.0 for name in list(dummies) + ["your_agent"]}
    recent_prices = {a: deque(maxlen=20) for a in ASSETS}
    recent_buy = {a: deque(maxlen=20) for a in ASSETS}
    recent_sell = {a: deque(maxlen=20) for a in ASSETS}
    for a in ASSETS:
        recent_prices[a].append(BASELINE_PRICE[a])  # simple warm start, no history.csv dependency
    your_team_history = []

    for r in range(N_ROUNDS):
        entry = schedule[r]
        game_state = {
            "round": r, "total_rounds": N_ROUNDS, "trade_budget": TRADE_BUDGET,
            "assets": {a: {"ticker": a[:4].upper(),
                            "recent_prices": list(recent_prices[a]),
                            "recent_buy_volume": list(recent_buy[a]),
                            "recent_sell_volume": list(recent_sell[a])} for a in ASSETS},
            "your_score_so_far": scores["your_agent"],
        }

        all_orders = {}
        for name, dummy in dummies.items():
            all_orders[name] = dummy.place_orders(game_state, [])
        try:
            your_orders = agent_mod.place_orders(game_state, your_team_history)
            if (not isinstance(your_orders, dict)
                    or sum(abs(v) for v in your_orders.values()) > TRADE_BUDGET
                    or any(not isinstance(v, int) for v in your_orders.values())):
                raise ValueError("place_orders must return signed ints summing (by absolute value) to <= budget")
        except Exception as e:
            print(f"  [round {r}] your agent errored or returned invalid output ({e}) -> defaulted to zero action")
            your_orders = {a: 0 for a in ASSETS}
        all_orders["your_agent"] = your_orders

        revenue, total_buy, total_sell = resolve_round(all_orders, entry["true_move"], entry["session"])
        for name, rev in revenue.items():
            scores[name] += rev

        your_team_history.append({"round": r, "orders": your_orders, "profit": revenue["your_agent"]})
        for a in ASSETS:
            recent_prices[a].append(entry["price"][a])
            recent_buy[a].append(total_buy[a])
            recent_sell[a].append(total_sell[a])

    return scores


def check_offline_forecast_accuracy(agent_mod, seed, n_rounds=40):
    """Approximates the real offline judging: calls predict_market()
    STATELESSLY each round (only ever passing the rolling window, exactly
    like the real judge will) and compares to realized prices.
    
    Now reads from the real history file to test your model locally on 
    the same data distribution it trained on!"""
    import os
    import pandas as pd
    
    # Search for history file
    possible_paths = ["history.csv", "../history.csv", "../../history.csv", 
                      "../history/history.csv", "../history/history_training_real.csv",
                      "history_training_real.csv"]
    hist_path = None
    for p in possible_paths:
        if os.path.exists(p):
            hist_path = p
            break
            
    if not hist_path:
        print("  Could not find history.csv for local offline test. Skipping MAE check.")
        return None
        
    try:
        df = pd.read_csv(hist_path)
    except Exception as e:
        print(f"  Error reading {hist_path}: {e}")
        return None
        
    max_round = df['round'].max()
    test_start = max_round - n_rounds + 1
    
    recent_prices = {a: deque(maxlen=20) for a in ASSETS}
    
    # Pre-fill recent prices with the 20 rounds before the test period
    pre_test_df = df[(df['round'] >= test_start - 20) & (df['round'] < test_start)]
    for r in sorted(pre_test_df['round'].unique()):
        row_data = pre_test_df[pre_test_df['round'] == r]
        for _, row in row_data.iterrows():
            if row['asset'] in ASSETS:
                recent_prices[row['asset']].append(row['price'])
                
    errors = []
    
    # Run the test loop on the final n_rounds
    test_df = df[df['round'] >= test_start]
    for r in sorted(test_df['round'].unique()):
        row_data = test_df[test_df['round'] == r]
        actual = {row['asset']: row['price'] for _, row in row_data.iterrows()}
        
        recent_history = {a: list(recent_prices[a]) for a in ASSETS}
        try:
            pred = agent_mod.predict_market(recent_history)
            for a in ASSETS:
                if a in actual:
                    errors.append(abs(pred.get(a, 0) - actual[a]))
        except Exception as e:
            print(f"  [round {r}] predict_market errored ({e}) -> skipped from MAE")
            
        for a in ASSETS:
            if a in actual:
                recent_prices[a].append(actual[a])
                
    return statistics.mean(errors) if errors else None


if __name__ == "__main__":
    import os
    if len(sys.argv) > 1:
        agent_path = sys.argv[1]
    else:
        # Default to agent.py in the same directory as this script
        agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py")
    
    print(f"Loading {agent_path} ...")
    agent_mod = load_agent(agent_path)

    print(f"\nRunning {N_ROUNDS}-round local match vs {N_DUMMY_OPPONENTS} dummy opponents...")
    all_scores = []
    for seed in range(3):
        scores = run_local_match(agent_mod, seed=seed * 31 + 1)
        all_scores.append(scores)

    print("\n=== LIVE MATCH RESULTS (avg over 3 local seeds) ===")
    names = list(all_scores[0].keys())
    avg_scores = {n: statistics.mean(s[n] for s in all_scores) for n in names}
    for name, sc in sorted(avg_scores.items(), key=lambda kv: -kv[1]):
        marker = "  <-- YOU" if name == "your_agent" else ""
        print(f"  {name:<28}{sc:>10.1f}{marker}")

    print("\nChecking offline forecast accuracy (predict_market, called statelessly)...")
    mae = check_offline_forecast_accuracy(agent_mod, seed=777)
    if mae is not None:
        print(f"  Your predict_market() MAE: {mae:.2f}  (lower is better; compare against your own "
              f"iterations, there's no fixed target -- the real event scores this against the whole field)")
