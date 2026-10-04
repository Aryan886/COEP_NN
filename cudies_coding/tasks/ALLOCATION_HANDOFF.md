# Allocation handoff

## Implemented policy

agent.place_orders_crowd(game_state, team_history) implements Astra P1. It retains P0 as place_orders_baseline and the active place_orders entry point uses P0 because P1 did not show consistent gains in evaluation.

P1 estimates recent opponent buy/sell flow from up to 20 paired volume observations. It preserves positions when a pair is invalid, corrects our own past orders only when round IDs establish contiguous alignment, discounts confidence on unaligned observations, shrinks estimates toward 20 * budget / 12 per asset and side, and caps total estimated opponent flow at 20 * budget. The assumptions are explicit constants in agent.py.

For each asset, P1 evaluates nine equally factorized flow/liquidity scenarios: recent flow (0.50), shared-forecast crowding (0.35), and reversed recent flow (0.15), crossed with the three documented liquidity multipliers. It adds our candidate order to side totals, applies the buy-wins-ties convention from the harness, and scores both wins and losses with the same fill discount. A six-asset integer dynamic program selects orders under the given budget, allows unused budget, and prefers lower exposure on ties. A missing latest price makes that asset unavailable for trading. No fitted return/flow covariance is claimed; the utility uses forecast change times expected fill.

## Comparison and selection

A custom evaluator ran P0 and P1 against 20 synthetic opponents plus us on identical 150-round price schedules and fixed seeds. It used production-scale liquidity (BASE_LIQUIDITY * session multiplier * 0.6) without the harness's extra 0.3 factor. Forecasts and opponent randomness were paired, while volume histories evolved separately for each policy. Five seeds were used for the four base fields; one fixed seed was used per liquidity stress field.

| Field | P1 minus P0 mean, 5 seeds | Worst paired difference | P1 mean score | P0 mean score |
|---|---:|---:|---:|---:|
| Correlated forecast followers | +326.0 | -301.3 | 1,013.0 | 687.0 |
| Mixed opponents | +285.4 | -629.1 | 1,417.9 | 1,132.5 |
| Adaptive, with 10 P1 opponents | -19.0 | -460.2 | 669.2 | 688.2 |
| Clone stress | -77.9 | -445.8 | 539.5 | 617.4 |

The adaptive result is near parity on mean but has a weaker poor run. P1 also draws more crowding in clone stress: average winning/losing dilution was about 83%/83% for P1 versus 51%/44% for P0. Identical P1 copies made the same decisions (clone_gap = 0), as expected, but P1 did not improve over the P0 candidate against that field.

At 0.5x and 2x assumed liquidity, P1's one-seed paired gain was positive in correlated, mixed, and adaptive fields, but negative in clone stress at 0.5x. The no-dilution check also favored P1 because the DP concentrates orders on stronger forecast changes. These small synthetic cases do not validate the opponent-flow assumptions against real teams.

Following Astra's gate, P0 remains the production entry point because P1 did not improve consistently across adaptive and clone comparisons and had material poor-seed regressions. P1 remains callable for later evaluation without changing forecast behavior.

## Verification and limits

Seventeen focused checks pass. They verify the score formula against 1,392 harness cases, including buy/sell dominance and ties; compare the DP against brute force on 30 seeded nonmonotone utility tables; test the dominance-transition example; check alignment, own-order subtraction, invalid/missing volumes, flow caps and scenario weights; and exercise zero budget, determinism, integer output, and exposure limits.

P1 decisions took under 21 ms in the synthetic match evaluator, excluding fresh process startup and artifact import. Production-scale liquidity constants, flow shrinkage, exchangeable-team correction, and scenario weights remain uncalibrated assumptions. The known training holdout has already been consumed by the earlier forecast check and was not used for allocation selection. No 21-team hidden-field result is implied.

