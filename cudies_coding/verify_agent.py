"""Focused checks for historical data preparation."""

import hashlib
import itertools
import json
import math
import random
import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agent import ASSETS, PARTNERS, make_features, place_orders, predict_market
from clean_data import load_history
from train_model import build_examples
import agent
import local_test_harness as harness


class HistoryLoadingTests(unittest.TestCase):
    def write_history(self, rows, directory):
        path = Path(directory) / "history.csv"
        pd.DataFrame(rows, columns=["round", "asset", "price"]).to_csv(path, index=False)
        return path

    def test_recovers_only_unique_missing_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [(0, asset, 10.0) for asset in ASSETS[:-1]]
            rows.append((0, None, 42.0))
            path = self.write_history(rows, directory)

            prices, audit = load_history(path)

            self.assertEqual(prices.loc[0, ASSETS[-1]], 42.0)
            self.assertEqual(len(audit["recovered_labels"]), 1)
            self.assertEqual(audit["unresolved_rows"], [])

    def test_preserves_ambiguous_rows_without_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [(0, asset, 10.0) for asset in ASSETS[:-2]]
            rows.extend([(0, None, 41.0), (0, None, 42.0)])
            path = self.write_history(rows, directory)

            prices, audit = load_history(path)

            self.assertTrue(pd.isna(prices.loc[0, ASSETS[-2]]))
            self.assertTrue(pd.isna(prices.loc[0, ASSETS[-1]]))
            self.assertEqual(len(audit["unresolved_rows"]), 2)
            self.assertEqual({row["price"] for row in audit["unresolved_rows"]}, {41.0, 42.0})

    def test_preserves_missing_round_and_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [(0, asset, 10.0) for asset in ASSETS]
            rows.extend((2, asset, 12.0) for asset in ASSETS[:-1])
            path = self.write_history(rows, directory)

            prices, audit = load_history(path)

            self.assertEqual(prices.index.tolist(), [0, 1, 2])
            self.assertTrue(prices.loc[1].isna().all())
            self.assertTrue(pd.isna(prices.loc[2, ASSETS[-1]]))
            self.assertEqual(audit["missing_observations"], 7)

    def test_does_not_modify_raw_input_or_fill_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [(0, asset, 10.0) for asset in ASSETS]
            rows[0] = (0, ASSETS[0], None)
            path = self.write_history(rows, directory)
            before = hashlib.sha256(path.read_bytes()).hexdigest()

            prices, audit = load_history(path)

            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertTrue(pd.isna(prices.loc[0, ASSETS[0]]))
            self.assertEqual(audit["missing_prices"], 1)

    def test_rejects_duplicate_known_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_history([(0, ASSETS[0], 10), (0, ASSETS[0], 11)], directory)

            with self.assertRaisesRegex(ValueError, "duplicate known round/asset keys"):
                load_history(path)

    def test_marks_nonfinite_and_invalid_prices_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_history(
                [(0, ASSETS[0], float("inf")), (0, ASSETS[1], "bad price")],
                directory,
            )

            prices, audit = load_history(path)

            self.assertTrue(prices.loc[0, [ASSETS[0], ASSETS[1]]].isna().all())
            self.assertEqual(audit["nonfinite_prices"], 1)
            self.assertEqual(audit["invalid_prices"], 1)


class ForecastTests(unittest.TestCase):
    def test_training_features_match_inference_and_ignore_target(self):
        prices = pd.DataFrame(
            {asset: [10.0, 11.0, 12.0, 13.0, 14.0] for asset in ASSETS}
        )
        asset = ASSETS[0]
        ratio_limits = {name: 2.1 for name in ASSETS}
        example = build_examples(prices, asset, 4, 5, ratio_limits)[0]
        history = {
            asset: prices[asset].iloc[:4].tolist(),
            PARTNERS[asset]: prices[PARTNERS[asset]].iloc[:4].tolist(),
        }
        reference, features, _ = make_features(history, asset, ratio_limits)
        self.assertEqual(example["reference"], reference)
        self.assertEqual(example["features"], features)

        prices.loc[4, asset] = 999.0
        changed_target = build_examples(prices, asset, 4, 5, ratio_limits)[0]
        self.assertEqual(changed_target["features"], features)

    def test_empty_short_and_missing_histories_produce_finite_prices(self):
        histories = [
            {},
            {asset: [42.0] for asset in ASSETS},
            {asset: [None, float("nan"), 30.0, float("inf")] for asset in ASSETS},
        ]
        for history in histories:
            with self.subTest(history=history):
                predictions = predict_market(history)
                self.assertEqual(set(predictions), set(ASSETS))
                self.assertTrue(all(math.isfinite(value) and value > 0 for value in predictions.values()))
                self.assertEqual(predictions, predict_market(history))

    def test_orders_use_native_integers_and_respect_budget(self):
        assets = {asset: {"recent_prices": [10.0, 11.0, 12.0]} for asset in ASSETS}
        state = {"trade_budget": 14, "assets": assets}
        orders = place_orders(state, [])
        self.assertEqual(set(orders), set(ASSETS))
        self.assertTrue(all(type(value) is int for value in orders.values()))
        self.assertLessEqual(sum(abs(value) for value in orders.values()), 14)

        for asset in ASSETS:
            assets[asset]["recent_prices"].append(None)
        self.assertEqual(place_orders(state, []), {asset: 0 for asset in ASSETS})


class AllocationTests(unittest.TestCase):
    def make_state(self, count=2):
        return {
            "round": count,
            "trade_budget": 14,
            "assets": {
                asset: {
                    "recent_prices": [30.0, 31.0, 32.0],
                    "recent_buy_volume": [12.0] * count,
                    "recent_sell_volume": [8.0] * count,
                }
                for asset in ASSETS
            },
        }

    def test_scoring_matches_harness(self):
        asset = ASSETS[0]
        liquidity = harness.BASE_LIQUIDITY[asset] * 0.6 * 0.3
        for buy, sell, order, change in itertools.product(
            (0, 5, 12, 30), (0, 5, 12, 30), range(-14, 15), (-2, 0, 2)
        ):
            scores, _, _ = harness.resolve_round(
                {"buy": {asset: buy}, "sell": {asset: -sell}, "us": {asset: order}},
                {name: change for name in ASSETS},
                "regular",
            )
            expected = agent.score_order(order, change, buy, sell, liquidity) * harness.PNL_SCALE
            self.assertAlmostEqual(expected, scores["us"])

    def test_dominance_ties_liquidity_and_invalid_scenarios(self):
        self.assertEqual(agent.score_order(5, 1, 30, 36, 20), 5)
        self.assertAlmostEqual(agent.score_order(6, 1, 30, 36, 20), 120 / 36)
        self.assertEqual(agent.score_order(-6, -1, 36, 30, 20), 6)
        self.assertEqual(agent.score_order(4, 2, 6, 0, 10), 8)
        self.assertEqual(agent.score_order(4, -2, 6, 0, 0), -8)
        self.assertAlmostEqual(agent.score_order(14, 1, 0, 0, 7), 7)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            agent.score_order(1, 1, 0, 0, float("nan"))

    def test_optimizer_matches_brute_force(self):
        generator = random.Random(42)
        for _ in range(30):
            tables = {
                asset: {order: (0.0 if order == 0 else generator.uniform(-5, 5))
                        for order in range(-4, 5)}
                for asset in ("a", "b", "c")
            }
            orders = agent.optimize_orders(tables, 4)
            actual = sum(tables[asset][order] for asset, order in orders.items())
            best = max(
                sum(tables[asset][order] for asset, order in zip(tables, candidate))
                for candidate in itertools.product(range(-4, 5), repeat=3)
                if sum(abs(order) for order in candidate) <= 4
            )
            self.assertAlmostEqual(actual, best)
        self.assertEqual(agent.optimize_orders({"a": {0: 0, 1: -1, -1: -1}}, 3), {"a": 0})
        self.assertEqual(agent.optimize_orders({"a": {0: 0, 1: 0, -1: 0}}, 3), {"a": 0})
        self.assertEqual(agent.optimize_orders({"a": {0: 0, 1: 1}}, 0), {"a": 0})

    def test_optimizer_crosses_unattractive_increment(self):
        tables = {
            "a": {order: agent.score_order(order, 1, 30, 36, 20) for order in range(15)},
            "b": {order: order * 0.1 for order in range(15)},
        }
        self.assertEqual(agent.optimize_orders(tables, 14), {"a": 14, "b": 0})

    def test_flow_subtracts_own_order_only_with_contiguous_rounds(self):
        state = self.make_state()
        history = [{"round": number, "orders": {asset: 2 for asset in ASSETS}}
                   for number in range(2)]
        flows = agent.estimate_opponent_flow(state, history, 14)
        effective_count = 1.75 ** 2 / (1 + 0.75 ** 2)
        share = effective_count / (effective_count + 4)
        prior = 20 * 14 / 12
        self.assertAlmostEqual(flows[ASSETS[0]][0], share * 10 + (1 - share) * prior)
        unaligned = agent.estimate_opponent_flow(state, history[1:], 14)
        self.assertAlmostEqual(
            unaligned[ASSETS[0]][0], share * 0.5 * 12 * 20 / 21 + (1 - share * 0.5) * prior
        )

    def test_flow_preserves_missing_positions_and_bounds_total(self):
        state = self.make_state(3)
        for asset in ASSETS:
            state["assets"][asset]["recent_buy_volume"] = [0, None, 20]
            state["assets"][asset]["recent_sell_volume"] = [0, None, 0]
        flows = agent.estimate_opponent_flow(state, [], 14)
        weight = 0.75 ** 2
        effective_count = (1 + weight) ** 2 / (1 + weight ** 2)
        share = effective_count / (effective_count + 4) * 0.5
        expected = share * (20 * 20 / 21) / (1 + weight) + (1 - share) * (20 * 14 / 12)
        self.assertAlmostEqual(flows[ASSETS[0]][0], expected)
        for asset in ASSETS:
            state["assets"][asset]["recent_buy_volume"] = [1000]
            state["assets"][asset]["recent_sell_volume"] = [1000]
        flows = agent.estimate_opponent_flow(state, [], 14)
        self.assertLessEqual(sum(buy + sell for buy, sell in flows.values()), 280.000001)

    def test_invalid_volume_pairs_and_inconsistent_own_order(self):
        state = self.make_state(1)
        for asset in ASSETS:
            state["assets"][asset]["recent_buy_volume"] = [-1]
        flows = agent.estimate_opponent_flow(state, [], 14)
        self.assertTrue(all(pair == (280 / 12, 280 / 12) for pair in flows.values()))
        state = self.make_state(1)
        state["assets"][ASSETS[0]]["recent_sell_volume"] = []
        with patch("sys.stderr"):
            flows = agent.estimate_opponent_flow(state, [], 14)
        self.assertEqual(flows[ASSETS[0]], (280 / 12, 280 / 12))
        state = self.make_state(1)
        history = [{"round": 0, "orders": {asset: 99 for asset in ASSETS}}]
        with patch("sys.stderr"):
            actual = agent.estimate_opponent_flow(state, history, 14)
        self.assertEqual(actual, agent.estimate_opponent_flow(state, [], 14))

    def test_scenarios_and_public_edge_cases(self):
        state = self.make_state()
        flows = agent.estimate_opponent_flow(state, [], 14)
        signals = {ASSETS[0]: 2, ASSETS[1]: -1}
        scenarios = agent.build_scenarios(signals, flows, 14)
        for values in scenarios.values():
            self.assertEqual(len(values), 9)
            self.assertAlmostEqual(sum(value[3] for value in values), 1)
        for world in (0, 3, 6):
            self.assertLessEqual(sum(values[world][0] + values[world][1] for values in scenarios.values()), 280.000001)
        orders = agent.crowd_orders(state, [], 14, signals)
        self.assertGreaterEqual(orders[ASSETS[0]], 0)
        self.assertLessEqual(orders[ASSETS[1]], 0)
        self.assertEqual(orders, agent.crowd_orders(state, [], 14, signals))
        state["assets"][ASSETS[0]]["recent_prices"] = [None, float("nan")]
        self.assertEqual(agent.place_orders_crowd(state, [])[ASSETS[0]], 0)
        state["trade_budget"] = 0
        self.assertEqual(agent.place_orders(state, []), {asset: 0 for asset in ASSETS})
        self.assertEqual(agent.place_orders_crowd(state, []), {asset: 0 for asset in ASSETS})


def run_policy_match(seed, field, policy, liquidity_scale=1.0):
    """Independent 21-team replay of a synthetic schedule, with no 0.3 factor."""
    fields = {
        "correlated": (16, 2, 2, 0),
        "mixed": (8, 6, 6, 0),
        "adaptive": (6, 2, 2, 10),
        "clones": (0, 0, 0, 20),
    }
    forecast_count, momentum_count, reversion_count, crowd_count = fields[field]
    budget = 14
    generator = random.Random(seed)
    variations = [
        {asset: generator.uniform(0.85, 1.15) for asset in ASSETS}
        for _ in range(forecast_count)
    ]
    recent_prices = {asset: [harness.BASELINE_PRICE[asset]] for asset in ASSETS}
    recent_buys = {asset: [] for asset in ASSETS}
    recent_sells = {asset: [] for asset in ASSETS}
    own_history = []
    crowd_history = []
    score = 0.0
    raw_score = 0.0
    raw_wins = actual_wins = raw_losses = actual_losses = 0.0
    unused_budget = 0
    max_asset_exposure = max_sector_exposure = 0
    max_decision_ms = 0.0
    clone_gap = 0.0

    for round_number, outcome in enumerate(harness.make_price_schedule(150, seed)):
        state = {
            "round": round_number,
            "trade_budget": budget,
            "assets": {
                asset: {
                    "recent_prices": recent_prices[asset],
                    "recent_buy_volume": recent_buys[asset],
                    "recent_sell_volume": recent_sells[asset],
                }
                for asset in ASSETS
            },
        }
        started = time.perf_counter()
        predictions = agent.predict_market(recent_prices)
        signals = {asset: predictions[asset] - recent_prices[asset][-1] for asset in ASSETS}
        if policy == "P1":
            our_orders = agent.crowd_orders(state, own_history, budget, signals)
        else:
            our_orders = agent.proportional_orders(signals, budget)
        max_decision_ms = max(max_decision_ms, (time.perf_counter() - started) * 1000)
        orders = [our_orders]
        for variation in variations:
            varied = {asset: signals[asset] * variation[asset] for asset in ASSETS}
            orders.append(agent.proportional_orders(varied, budget))

        momentum = {}
        reversion = {}
        for asset in ASSETS:
            prices = recent_prices[asset]
            changes = [right - left for left, right in zip(prices[:-1], prices[1:])]
            momentum[asset] = statistics.mean(changes[-3:]) if changes else 0.0
            reversion[asset] = statistics.mean(prices) - prices[-1]
        momentum_orders = agent.proportional_orders(momentum, budget)
        reversion_orders = agent.proportional_orders(reversion, budget)
        orders.extend([momentum_orders] * momentum_count)
        orders.extend([reversion_orders] * reversion_count)
        if crowd_count:
            # These opponents share observations, forecasts and identical own
            # histories, so one calculation is exactly equivalent to ten/twenty.
            crowd_order = agent.crowd_orders(state, crowd_history, budget, signals)
            orders.extend([crowd_order] * crowd_count)
            if field == "clones" and policy == "P1":
                if our_orders != crowd_order:
                    raise AssertionError("Identical P1 agents broke clone symmetry")
        if len(orders) != 21:
            raise AssertionError("Comparison must contain 20 opponents plus us")
        for team_orders in orders:
            if sum(abs(value) for value in team_orders.values()) > budget:
                raise AssertionError("Simulation opponent exceeded its budget")

        round_profit = 0.0
        for asset in ASSETS:
            buy = sum(max(team[asset], 0) for team in orders)
            sell = sum(max(-team[asset], 0) for team in orders)
            liquidity = (
                harness.BASE_LIQUIDITY[asset] * harness.SESSION_MULT[outcome["session"]]
                * harness.EXEC_LIQUIDITY_SCALE * liquidity_scale
            )
            dominant = max(buy, sell)
            discount = min(1.0, liquidity / dominant) if dominant else 1.0
            order = our_orders[asset]
            raw_profit = order * outcome["true_move"][asset] * harness.PNL_SCALE
            dominant_side = (order > 0 and buy >= sell) or (order < 0 and sell > buy)
            actual_profit = raw_profit * (discount if dominant_side else 1.0)
            round_profit += actual_profit
            raw_score += raw_profit
            if raw_profit > 0:
                raw_wins += raw_profit
                actual_wins += actual_profit
            elif raw_profit < 0:
                raw_losses -= raw_profit
                actual_losses -= actual_profit
            if field == "clones" and policy == "P1":
                clone_raw = crowd_order[asset] * outcome["true_move"][asset] * harness.PNL_SCALE
                clone_gap += abs(actual_profit - clone_raw * (discount if dominant_side else 1.0))
            recent_prices[asset] = (recent_prices[asset] + [outcome["price"][asset]])[-20:]
            recent_buys[asset] = (recent_buys[asset] + [buy])[-20:]
            recent_sells[asset] = (recent_sells[asset] + [sell])[-20:]
        score += round_profit
        exposure = sum(abs(value) for value in our_orders.values())
        unused_budget += budget - exposure
        max_asset_exposure = max(max_asset_exposure, max(abs(value) for value in our_orders.values()))
        max_sector_exposure = max(
            max_sector_exposure,
            max(abs(our_orders[asset]) + abs(our_orders[agent.PARTNERS[asset]]) for asset in ASSETS),
        )
        own_history = (own_history + [{"round": round_number, "orders": our_orders}])[-20:]
        if crowd_count:
            crowd_history = (crowd_history + [{"round": round_number, "orders": crowd_order}])[-20:]

    return {
        "score": score,
        "raw_score": raw_score,
        "winning_dilution": 1 - actual_wins / raw_wins if raw_wins else 0.0,
        "losing_dilution": 1 - actual_losses / raw_losses if raw_losses else 0.0,
        "mean_unused_budget": unused_budget / 150,
        "max_asset_lots": max_asset_exposure,
        "max_sector_lots": max_sector_exposure,
        "max_decision_ms": max_decision_ms,
        "clone_gap": clone_gap,
    }


def compare_policies():
    """Fixed matched-seed comparison; never opens historical holdout targets."""
    report = {}
    for liquidity_scale, seeds in ((1.0, range(1, 6)), (0.5, (1,)), (2.0, (1,)), (1000000.0, (1,))):
        for field in ("correlated", "mixed", "adaptive", "clones"):
            matches = {"P0": [], "P1": []}
            for seed in seeds:
                for policy in matches:
                    matches[policy].append(run_policy_match(seed, field, policy, liquidity_scale))
            summary = {}
            for policy, results in matches.items():
                scores = [result["score"] for result in results]
                summary[policy] = {
                    "scores": [round(score, 3) for score in scores],
                    "mean": statistics.mean(scores),
                    "worst": min(scores),
                    "standard_deviation": statistics.pstdev(scores),
                    "mean_raw_profit": statistics.mean(result["raw_score"] for result in results),
                    "winning_dilution": statistics.mean(result["winning_dilution"] for result in results),
                    "losing_dilution": statistics.mean(result["losing_dilution"] for result in results),
                    "mean_unused_budget": statistics.mean(result["mean_unused_budget"] for result in results),
                    "max_asset_lots": max(result["max_asset_lots"] for result in results),
                    "max_sector_lots": max(result["max_sector_lots"] for result in results),
                    "max_decision_ms": max(result["max_decision_ms"] for result in results),
                    "clone_gap": max(result["clone_gap"] for result in results),
                }
            deltas = [right["score"] - left["score"] for left, right in zip(matches["P0"], matches["P1"])]
            summary["paired_mean_gain"] = statistics.mean(deltas)
            summary["paired_worst_gain"] = min(deltas)
            report[f"{field}_liquidity_{liquidity_scale:g}"] = summary
            print(json.dumps({"field": field, "liquidity": liquidity_scale, "results": summary}), flush=True)
    return report


if __name__ == "__main__":
    if sys.argv[1:] == ["--compare"]:
        compare_policies()
    else:
        unittest.main()
