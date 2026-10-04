"""Focused checks for historical data preparation."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from clean_data import ASSETS, load_history


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


if __name__ == "__main__":
    unittest.main()
