# Task 02 — Astra: design a crowd-aware strategy

Read `PLAN.md` in this folder first. Work in `C:\Users\aryan\COEP-NN`. Produce a first actionable specification by T+8 minutes, support human discussion until T+12, and freeze the implementation proposal by T+14. This is a design task, not authorization to change production code.

## Goal and ownership

Own only `cudies_coding/tasks/STRATEGY_SPEC.md`. Inspect rules, deck if needed, the harness, and available Sol handoffs. Do not wait for forecasting results before producing a usable default. Do not edit Sol's files or train models.

We want strong live profit while preserving independent forecast accuracy. The task is to reason about this contest, not recommend real-money trades.

## Required reasoning

1. Derive per-asset utility from the actual signed-order scoring rule. Account for both diluted wins and diluted losses, our own contribution to flow, ties, zero orders, and uncertain liquidity.
2. Explain why low crowding alone does not justify a trade with negative expected return. Distinguish a price forecast from an execution estimate.
3. Propose the smallest credible current-flow estimator using up to 20 historical volume observations. Historical volume is stale; include a shrinkage/default rule and a clear policy when own-order alignment is unavailable. Consider price-based momentum/mean-reversion opponent proxies only if they justify their implementation time.
4. Harness constants are assumptions. Define a small liquidity/flow scenario set and how to aggregate utilities. Explain the limitation of multiplying mean return by mean fill when returns and crowding are correlated.
5. Specify a budget dynamic program with candidate signed integer positions and an option to leave budget unused. Asset utilities should be separable so Sol can implement it quickly. Explain why a simple greedy allocator may fail around dominance transitions.
6. Consider forecast uncertainty and concentration, but avoid arbitrary restrictions. Propose at most two or three policy variants for validation. Explain the trade-off between expected profit and risk without claiming the unknown normalized leaderboard objective is exactly solved.
7. Specify evaluation against 20 opponents plus us, including correlated forecast-followers, mean reversion, momentum, and mixtures. Use realistic production-scale liquidity scenarios rather than multiplying four weak dummies and keeping the local liquidity reduction.

## Specification format

Include: minimum viable policy; explicit formulas and tie behavior; inputs/outputs; concise pseudocode; defaults with rationale; estimated complexity; two numerical examples; edge cases and test expectations; comparison baseline; and a section titled `Open decisions for discussion`.

End with `Build now` and `Defer` lists. Mark each choice as evidence-backed, an assumption, or a validation candidate. Keep the core implementation achievable within Sol's nine-minute allocation window. Do not propose deep RL, a new service, GPU training, persistent runtime state, or a large search.

## Handoff

At minute 8, save a complete first specification even if some decisions remain open. Report the three most consequential choices for the human to discuss. Integrate explicit user steering and freeze the build recommendation by minute 14. If no answer arrives, document conservative defaults; do not silently treat silence as approval for extra scope.
