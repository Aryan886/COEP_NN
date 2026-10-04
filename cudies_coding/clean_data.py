"""Load the historical prices without losing round or asset alignment."""

from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = (
    "quantum_dynamics",
    "byte_stream",
    "gold_trust",
    "metro_rail",
    "agro_futures",
    "solar_grid",
)


def load_history(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return observed prices by round and an audit of uncertain source rows.

    The index is every integer round from the first through the last, and the
    columns follow ASSETS. Missing or non-finite prices remain NaN. No price
    imputation or outlier filtering is performed.
    """
    data = pd.read_csv(path)
    required_columns = {"round", "asset", "price"}
    if not required_columns.issubset(data.columns):
        missing_columns = sorted(required_columns - set(data.columns))
        raise ValueError(f"History is missing columns: {missing_columns}")
    if data.empty:
        raise ValueError("History contains no rows")

    rounds = pd.to_numeric(data["round"], errors="coerce")
    valid_rounds = rounds.notna() & np.isfinite(rounds) & (rounds % 1 == 0)
    if not valid_rounds.all():
        bad_rows = data.index[~valid_rounds].tolist()[:5]
        raise ValueError(f"History has invalid round numbers at rows {bad_rows}")
    data["round"] = rounds.astype(int)
    rows_per_round = data.groupby("round").size().value_counts().sort_index().to_dict()

    unknown_assets = data.loc[
        data["asset"].notna() & ~data["asset"].isin(ASSETS), "asset"
    ].unique()
    if len(unknown_assets):
        raise ValueError(f"History has unknown asset labels: {unknown_assets.tolist()}")

    known_rows = data[data["asset"].notna()]
    duplicates = known_rows[known_rows.duplicated(["round", "asset"], keep=False)]
    if not duplicates.empty:
        keys = duplicates[["round", "asset"]].drop_duplicates().head(5)
        raise ValueError(
            f"History has duplicate known round/asset keys: {keys.to_dict('records')}"
        )

    missing_price_count = int(data["price"].isna().sum())
    numeric_prices = pd.to_numeric(data["price"], errors="coerce")
    invalid_price_count = int((data["price"].notna() & numeric_prices.isna()).sum())
    nonfinite_price_count = int(np.isinf(numeric_prices).sum())
    data["price"] = numeric_prices.where(np.isfinite(numeric_prices))

    missing_label_count = int(data["asset"].isna().sum())
    recovered_labels = []
    unresolved_rows = []
    for round_number, rows in data.groupby("round", sort=True):
        unlabeled = rows[rows["asset"].isna()]
        if unlabeled.empty:
            continue

        known_assets = rows["asset"].dropna().tolist()
        missing_assets = set(ASSETS) - set(known_assets)
        can_recover = (
            len(rows) == len(ASSETS)
            and len(unlabeled) == 1
            and len(missing_assets) == 1
            and len(known_assets) == len(set(known_assets)) == len(ASSETS) - 1
        )
        if can_recover:
            row_index = int(unlabeled.index[0])
            asset = missing_assets.pop()
            data.at[row_index, "asset"] = asset
            recovered_labels.append(
                {"row_index": row_index, "round": int(round_number), "asset": asset}
            )
        else:
            for row_index, row in unlabeled.iterrows():
                price = row["price"]
                unresolved_rows.append(
                    {
                        "row_index": int(row_index),
                        "round": int(round_number),
                        "price": None if pd.isna(price) else float(price),
                    }
                )

    observed = data[data["asset"].notna()]
    aligned = observed.pivot(index="round", columns="asset", values="price")
    aligned = aligned.reindex(
        index=range(int(data["round"].min()), int(data["round"].max()) + 1),
        columns=ASSETS,
    ).astype(float)
    aligned.index.name = "round"
    aligned.columns.name = None

    extreme_examples = []
    extreme_count = 0
    for asset in ASSETS:
        prices = aligned[asset].dropna()
        median_price = prices.median()
        if prices.empty or median_price <= 0:
            continue
        extreme = prices[(prices > 5 * median_price) | (prices < median_price / 5)]
        extreme_count += len(extreme)
        for round_number, price in extreme.items():
            ratio = (
                max(price / median_price, median_price / price)
                if price > 0
                else float("inf")
            )
            extreme_examples.append(
                {
                    "round": int(round_number),
                    "asset": asset,
                    "price": float(price),
                    "ratio": float(ratio),
                }
            )
    extreme_examples.sort(key=lambda example: example["ratio"], reverse=True)

    audit = {
        "source_rows": len(data),
        "source_rounds": int(data["round"].nunique()),
        "rows_per_round": {
            int(row_count): int(round_count)
            for row_count, round_count in rows_per_round.items()
        },
        "rounds": len(aligned),
        "known_key_duplicates": 0,
        "missing_asset_labels": missing_label_count,
        "recovered_labels": recovered_labels,
        "unresolved_rows": unresolved_rows,
        "missing_prices": missing_price_count,
        "invalid_prices": invalid_price_count,
        "nonfinite_prices": nonfinite_price_count,
        "missing_observations": int(aligned.isna().sum().sum()),
        "rounds_with_missing_observations": int(aligned.isna().any(axis=1).sum()),
        "extreme_price_count": extreme_count,
        "extreme_price_rule": "price outside [asset median / 5, asset median * 5]",
        "extreme_examples": extreme_examples[:12],
    }
    return aligned, audit
