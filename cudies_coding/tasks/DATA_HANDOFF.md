# Data preparation handoff

## Interface

`clean_data.load_history(path)` returns `(prices, audit)`. `prices` is a float
DataFrame indexed by every integer round from the source minimum through its
maximum, with six columns in `clean_data.ASSETS` order. Each cell is the
observed price for that round and asset, or `NaN` when unavailable. The loader
does not fill, clip, or alter prices. It reads the CSV without writing to it.

`audit` includes source and missingness counts, every recovered label as
`{row_index, round, asset}`, every unresolved row as `{row_index, round, price}`,
and a small sample of extreme prices. `row_index` is the zero-based CSV data
row, excluding the header. Unknown asset labels, invalid round numbers, and
duplicate known `(round, asset)` keys raise `ValueError` with context. The
extreme-price rule is descriptive only; do not use its full-history medians
as training features or fitted thresholds.

## Verified source audit

- 12,546 source rows, rounds 0–2090; all 2,091 rounds have six source rows.
- Known labels are among the six expected assets. There are no duplicate
  known `(round, asset)` keys or missing round numbers.
- 376 labels are blank. The unique missing-asset rule recovers 316: 61
  `byte_stream`, 59 `metro_rail`, 52 each `gold_trust` and
  `quantum_dynamics`, 50 `agro_futures`, and 42 `solar_grid`.
- 60 blank-label rows remain unresolved across 30 rounds. Examples are two
  unattributed prices in round 89 (20.70, 28.44) and round 148 (26.70,
  27.99). The audit retains all 60 source row positions and prices.
- 730 source prices are missing; no additional nonnumeric or infinite prices
  were found. The aligned table has 786 missing cells across 506 rounds:
  730 source missing prices plus 56 observed prices on unresolved rows.
  Four unresolved rows also lack a price.
- Missing cells by asset: `quantum_dynamics` 158, `byte_stream` 117,
  `gold_trust` 119, `metro_rail` 139, `agro_futures` 119, `solar_grid` 134.

## Price anomalies

The loader flags 81 observed values outside one-fifth to five times their
asset's full-history median. Of these, 79 are single-round flags. The only
adjacent flagged pair is `metro_rail` rounds 1797–1798: 141.4 then 2.828,
versus 28.28 before and 27.77 after. Other examples are `byte_stream` round
1329: 110.4 → 1113.4 → 111.84; `byte_stream` round 407: 8.24 → 0.825 →
8.20; and `agro_futures` round 1626: 56.34 → 567.6 → 57.81. These patterns
look like isolated scale errors, but their cause is unverified. All observed
values remain unchanged. The median rule is only a coarse audit and may miss
local anomalies or flag legitimate regimes.

## Verification and training guidance

Ran `python cudies_coding/verify_agent.py` with the project virtual
environment: six tests passed. They cover unique recovery, ambiguous
labels, a missing round and asset, unchanged input bytes with an unfilled
price, duplicate-key rejection, and invalid/non-finite price handling.
The source CSV Git object hash remained
`748e4aa09ccf52dbc32e811622473c4291427532`.

Build training windows by round from this aligned table. Mask unavailable
targets in supervised loss; never replace a missing target with an imputed
input. If inputs need filling, use past observations only and supply a
missingness indicator. Fit any anomaly thresholds on the training portion
only, report exclusions separately, and score against observed targets.
Use chronological fit, validation, and untouched final holdout segments.

Unresolved assumptions: ambiguous labels cannot be assigned from the CSV
alone, and no correction rule for extreme prices has been validated. The
next task is `03_SOL_FORECAST.md`.
