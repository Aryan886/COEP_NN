# Task 04 — Sol: implement and compare allocation policies

Read `PLAN.md`, `FORECAST_HANDOFF.md`, and the latest `STRATEGY_SPEC.md` first. Work in `C:\Users\aryan\COEP-NN`. Time window: T+14 to T+23 minutes. If Astra's specification is missing, use the documented conservative default in `PLAN.md` and state that assumption.

## Ownership

Edit allocation code in `cudies_coding/agent.py`, allocation/simulation checks in `cudies_coding/verify_agent.py`, and `cudies_coding/tasks/ALLOCATION_HANDOFF.md`. Preserve forecast behavior and model artifacts unless a concrete correctness bug requires a fix. Do not modify the organizer's harness or Astra's design file.

## Implementation

1. State the data flow, files, and trade-offs in 3–5 lines. Summarize which strategy choices are frozen and which remain assumptions.
2. Retain a simple forecast-only allocator as a comparison and fallback. Avoid a broad strategy framework; small clear functions are enough.
3. Implement flow estimation using only information available before the round. Own-order subtraction requires established round alignment; otherwise use the specified conservative fallback. Bound impossible negative volumes.
4. Score candidate signed positions including zero, our effect on dominant flow, discounting, and liquidity uncertainty. Match the stated tie convention while documenting that the harness provides it.
5. Allocate the supplied integer budget across all six assets with a small dynamic program. Use native Python ints, deterministic tie-breaking, and allow unspent budget. Do not introduce portfolio coupling that invalidates additive utilities.
6. Test utilities on no crowding, dominant buys, dominant sells, equal sides, overflow, our order changing dominance, zero budget, and missing volumes. Compare optimizer results with exhaustive enumeration on a tiny example.
7. Run a bounded comparison on development data/simulations against mixed opponents and strongly correlated opponents. Evaluate a field with 20 other agents. Correct for the harness's extra local liquidity scaling in the custom evaluator; never edit the supplied harness to make a strategy look better.
8. Use matched price schedules and fixed random seeds for policy comparisons. Avoid selecting from a large number of variants. If crowd adjustment offers no consistent benefit, ship the simpler validated allocator and explain why.

## Handoff and time cap

By minute 23, stop strategy changes. Save `ALLOCATION_HANDOFF.md` with implemented assumptions, defaults, scenarios, comparative results, correctness tests, execution time, and selected policy. Distinguish synthetic-scenario evidence from real held-out history evidence. Do not promise a tournament rank.

The next task performs independent final checks and packaging. Preserve final-holdout isolation.
