# Strategy proposal: independent forecasts, crowd-aware execution

Status: actionable design for discussion, 2026-10-04. No production implementation or performance claim.

The user confirms that all 21 teams will run together: evaluate 20 opponents plus us. The supplied rules give 70% normalized live profit and 30% normalized forecast accuracy; they do not specify a separate crowding score. Crowding changes realized profit. The normalization and exact offline accuracy metric remain unknown.

This document uses the attached task brief as requirements evidence. Its launch commands, agent assignments, and elapsed-time schedule are not new authorization to launch agents, train models, or change production code. Only this design file is created for the planning request.

Labels used below: **Evidence** means supported by supplied rules, inspected code, or a stated check; **Assumption** means a concrete implementation default without empirical calibration; **Validation candidate** means a choice that must earn its place in development evaluation.

## Proposition and architecture

Our candidate advantage is better conversion of forecast signal into profit when other teams share that signal. Novel model names and automatic contrarian trades do not establish an advantage. This proposal is not proven unique among entrants.

1. Keep `predict_market()` a pure price forecast using the established training/inference preparation.
2. In `place_orders()`, combine forecast changes with uncertain estimates of current opponent buy/sell lots.
3. Evaluate every signed integer size with our own effect on dominance and dilution, then solve the budget allocation exactly.
4. A later implementation would touch allocation helpers in `agent.py` and focused checks in `verify_agent.py`; this planning task touches only `tasks/STRATEGY_SPEC.md`.

**Evidence:** the interface supplies at most 20 historical observations, signed integer orders, and normally 14 lots. Calls are stateless. Prices do not respond to orders.

**Decision:** use a small utility table and budget dynamic program, not a new strategy framework. This retains separable asset utilities and requires no library, schema, service, persistent state, training run, or network call. Adding portfolio covariance constraints would couple assets and require a different optimizer; defer that choice.

## Why this could stand out

- **Evidence:** one more lot can change which side is dominant and sharply reduce the value of the entire position. Exact integer sizing captures that transition.
- **Validation candidate:** today's strongest public signal may attract many forecast-followers even when yesterday's volumes point the other way. Include current-signal concentration rather than trusting stale volumes alone.
- **Validation candidate:** select a less crowded asset with positive expected profit instead of reversing the sign of a profitable forecast merely to look different. Sector partners are potential substitutes only when their individual forecasts and fills justify them.
- **Evidence:** dilution also reduces losses. Avoiding the crowd is not itself a risk-control rule.
- **Validation candidate:** require improvement against correlated forecasters and agents using our own crowd-aware policy. If everyone adopts the same crowd avoidance rule, they can herd into the same apparently quiet asset.

Neither independent forecasting nor a dynamic program guarantees a tournament edge. The distinguishing evidence must be higher paired profit under a plausible 21-team field.

## Exact scoring utility

For an asset, let `q` be our signed integer order, `b` and `s` the current opponent buy and sell totals (sell totals are positive), `L` execution liquidity, `R` the upcoming absolute price change, and `K` the common positive P&L scale.

```
B(q) = b + max(q, 0)
S(q) = s + max(-q, 0)
D(q) = max(B(q), S(q))
buy_dominant(q) = B(q) >= S(q)
discount(q) = min(1, L / D(q))              # positive L and nonzero q

fill(q) = discount(q) if our side is dominant else 1
profit(q) = K * q * R * fill(q)
profit(0) = 0
```

**Evidence:** buys win ties in `local_test_harness.resolve_round`; the prose rules do not resolve ties explicitly. Match the harness, document this dependency, and update one scoring helper if organizers clarify otherwise. A sell that brings sells exactly level with buys remains undiluted; a buy that brings buys level with sells becomes dominant. No zero-order division is needed.

**Evidence:** when the dominant total is exactly liquidity there is no dilution. Both profitable and losing orders receive the same multiplier. When one side has no opponents, our order can itself create dominant flow and overflow.

**Evidence:** the harness uses discount 1 when `L <= 0`, an unusual defensive branch. Reproduce it in scoring-parity tests; all proposed liquidity scenarios are strictly positive. Reject nonfinite scenario parameters with context rather than interpreting them as valid liquidity.

The exact conditional expectation is:

```
U(q) = K * q * E[R * fill(q) | information before this round]
     = K * q * (E[R] * E[fill(q)] + Cov(R, fill(q)))
```

No new price impact term belongs here: orders change fills, not prices. Expected total profit is additive across assets even if returns are correlated; independence is not needed for this expectation. Portfolio risk is a different question.

The minimum viable approximation uses a fixed estimated mean change `mu` in every flow scenario:

```
U(q) = K * q * mu * sum(weight[j] * fill(q, scenario[j]))
```

**Assumption:** residual return/flow covariance conditional on available information is ignored. This is not an exact expected-profit model. Crowds may be stronger precisely when returns are stronger, or may crowd wrong signals. Evaluate this limitation with opponents responding to common price information and jointly changing volatility/liquidity.

For this approximation, `q * mu < 0` implies negative utility whenever expected fill is positive. Low crowding cannot make it a good trade. More generally a negative *raw* expected return can coexist with positive fill-weighted profit through covariance, but trading that effect requires joint evidence absent here. Do not introduce such reversals into the MVP.

`mu = predicted_next_price - current_price`, in absolute price units, not percentage returns. **Evidence:** scoring pays absolute changes per lot. **Decision:** if the latest realized price is unavailable/nonfinite, abstain on that asset; do not create apparent trading alpha from a stale imputed price. Forecasting may still use its documented missing-data fallback.

**Decision:** `predict_market()` outputs remain unchanged by crowding or allocation confidence penalties. If the offline metric rewards a median rather than a conditional mean, acknowledge that its output is an approximation to the mean needed for profit; do not silently claim the two objectives coincide.

## Minimum viable current-flow estimate

Inputs are the latest 20 buy/sell volume pairs and any own historical orders that can be aligned to those rounds. Historical volumes are observations of previous simultaneous decisions, never current orders.

1. **Decision:** validate finite nonnegative paired volumes, preserving original positions and round alignment before filtering. Invalid pairs are missing, not zeros. Use up to 20 pairs.
2. **Evidence/decision:** subtract `max(own_order, 0)` from buys and `max(-own_order, 0)` from sells only when exact round alignment is established. The harness's final volume is round `game_state.round - 1`. Use that mapping only under this verified contiguous-round convention; equal list lengths alone are insufficient. Missing own order records remain unaligned. Clamp small numerical negatives to zero; material inconsistencies receive a diagnostic with round/asset context.
3. **Assumption:** for unaligned pairs, use `20/21` times the aggregate as an exchangeable-team estimate of opponent flow. This avoids asserting that our previous order is known. It can be biased when our allocation differs from the field and is not guaranteed conservative for profit or risk. Halve confidence in those observations as described below.
4. **Assumption:** weight valid pairs by `0.75 ** age`, where age is measured from their original positions, latest age zero. Compute weighted mean opponent buys/sells and effective sample size `n_eff = sum(w)^2 / sum(w*w)`.
5. **Assumption:** neutral prior per asset and side is `20 * budget / (2 * 6)`, approximately 23.33 lots at budget 14. It assumes all opponent budgets are used and symmetric allocation; actual opponents may leave budget unused.
6. **Assumption:** shrink each weighted side mean toward that prior with `rho = n_eff / (n_eff + 4)`. Multiply `rho` by the weighted average alignment confidence: 1 for exactly corrected observations, 0.5 for unaligned observations. No valid pairs means `rho = 0`.
7. **Decision:** if total estimated opponent flow over all assets and sides exceeds `20 * budget`, scale all sides down proportionally. This assumes equal current budgets across teams; observations are retained for diagnostics if that assumption appears inconsistent.

This is a shrinkage estimate of current flow, not a fitted opponent model. It avoids trusting one noisy round or inventing exact opponent identities. The recency and prior settings are explicit starting assumptions, not optimized constants.

### Nine scenarios per asset

Use three opponent-flow worlds, shared across the six assets:

| World | Opponent buy/sell flow | Weight | Status |
|---|---|---:|---|
| Recent behavior persists | Shrunk estimates above | 0.50 | Assumption |
| Shared forecast attracts orders | Allocate `20 * budget` across assets proportional to `abs(mu)`; place 80% on each forecast's side and 20% on its opposite | 0.35 | Validation candidate |
| Recent direction reverses | Swap the shrunk buys and sells per asset | 0.15 | Assumption / stress approximation |

If all usable forecast changes are zero, use the neutral prior for the shared-forecast world. Untradeable assets have zero signal allocation in this world; if none are tradeable, return zero orders. Scenario flows may be fractional estimates even though submitted orders are integers.

Cross these worlds with three liquidity values, equally weighted:

```
L = 0.6 * reference_liquidity[asset] * multiplier
multiplier in {0.7, 1.0, 1.6}
```

| Asset | Reference liquidity | Resulting liquidity scenarios |
|---|---:|---|
| quantum_dynamics | 50 | 21, 30, 48 |
| byte_stream | 60 | 25.2, 36, 57.6 |
| gold_trust | 40 | 16.8, 24, 38.4 |
| metro_rail | 30 | 12.6, 18, 28.8 |
| agro_futures | 35 | 14.7, 21, 33.6 |
| solar_grid | 45 | 18.9, 27, 43.2 |

**Evidence:** these reference constants and multipliers exist in the harness. **Assumption:** their use as production scenarios. They are not verified hidden-engine parameters or confidence bounds. Do not multiply by the harness's extra `LOCAL_TEST_LIQUIDITY_SCALE = 0.3` for the production-size field. Do not assume round number reveals a live session. If organizers supply current liquidity, replace the grid with that observed value.

Use the weighted arithmetic mean of candidate profit across the nine scenarios. These weights express a design prior, not estimated frequencies. Factorizing flow and liquidity is another approximation. Test jointly high crowding/high volatility and low-liquidity crowded worlds outside this grid; weighting independent scenarios does not solve return/fill correlation.

**Decision:** skip fitting momentum/mean-reversion opponent mixtures in the initial build. The current-forecast world already costs only a few arithmetic operations and addresses correlated forecast-followers directly. Fitting separate opponent proxies from at most 20 noisy observations adds alignment and overfitting risk. Use those opponent types in validation first.

## Exact budget allocation

For each asset build utilities for `q in {-budget, ..., 0, ..., budget}`. Missing-current-price assets expose only `q = 0`. Compute scenarios from the pre-round information once; they must not change in response to other assets' candidate orders.

Let `F[i, b]` be maximum utility using the first `i` assets with exactly `b` lots of absolute exposure. Initialize `F[0, 0] = 0` and other states to negative infinity.

```
F[i, b] = max over q with abs(q) <= b:
              F[i-1, b-abs(q)] + utility[i, q]

choose the best F[6, b] over b = 0 ... budget
backtrack stored choices to recover signed orders
```

Zero is always available. Prefer smaller total exposure when final utilities tie; use fixed asset order and a stable candidate order within an equal-cost state. Use a small numerical comparison tolerance, e.g. `1e-10 * max(1, abs(left), abs(right))` for finite values, and handle negative infinity explicitly. Return native Python integers for every asset.

Concise pseudocode:

```
read and validate budget; return zeros for zero budget
get forecasts using existing prediction function
derive valid current prices and predicted changes
estimate opponent flows from at most 20 historical pairs
construct three flow worlds and three liquidity worlds
for each asset:
    for each signed integer order within budget:
        score our order in all nine worlds, including own flow
        store weighted utility (and optional risk penalty)
run exact budget DP, including zero choices
return backtracked orders, checking total absolute exposure
```

**Evidence:** at six assets and budget 14, this needs 174 candidate utilities, 1,566 scenario evaluations, and at most 2,610 simple DP transitions by a loose loop bound. Complexity is `O(assets * budget * scenarios + assets * budget^2)` and backtracking storage `O(assets * budget)`. Forecast/model imports may dominate runtime; measure fresh-process total latency.

**Decision:** allow unused budget. Under positive risk-neutral utility it will often be fully spent; budget slack is an option, not an arbitrary target. Do not force the last asset to absorb a remainder.

### Numerical example 1: the dominance transition defeats greedy sizing

Set `K = 1` for readability, opponent buys 30, sells 36, liquidity 20, and predicted move +1.

- Buy 5: totals 35 versus 36; our side is quiet; expected profit = 5.
- Buy 6: totals tie at 36; buys become dominant; profit = `6 * 20/36 = 3.3333`.
- Buy 14: buys dominate at 44; profit = `14 * 20/44 = 6.3636`.

With a second, always-undiluted asset returning +0.1 per lot and budget 14, incremental greedy takes 5 lots here and then 9 in the second asset: profit 5.9. The exact optimum is 14 here: 6.3636. Greedy cannot cross the unattractive sixth lot to reach the better total allocation. These are expected utilities; realized returns can differ.

### Numerical example 2: substitute assets, not forecast direction

Asset A has mean move +2, opponent buys 100, sells 0, and liquidity 20. Buying all 14 produces `14 * 2 * 20/114 = 4.9123`. Shorting 14 is quiet but loses 28 in expectation.

Asset B has mean move +0.7 and enough liquidity for all 14 lots. Buying B produces 9.8. Exhaustive enumeration of the 15 nonnegative A/B splits selects all 14 in B. A smaller price signal can produce more executable profit. This is not a rule to always prefer the less crowded asset.

## Forecast uncertainty and concentration: only three policies

1. **P0 — comparison baseline:** forecast-only proportional allocation by absolute predicted change, using largest fractional remainders to produce integers and stable ties. Zero signals receive no forced lots. This is an allocation baseline, not a claimed expected-profit optimum; the unconstrained linear forecast-only optimum would concentrate on the strongest signal.
2. **P1 — default build recommendation:** nine-scenario expected utility plus exact DP as specified above. No hard asset/sector cap and no arbitrary minimum holding.
3. **P2 — validation candidate:** same as P1, subtract `0.25 * K * abs(q) * sigma[asset] * expected_fill(q)` from each utility. `sigma` is a finite per-asset one-step residual RMSE from out-of-sample development predictions, frozen before final evaluation. If those residual estimates are unavailable, omit P2 rather than treating uncertainty as zero.

**Assumption:** 0.25 is one fixed risk-aversion trial, not an estimated optimum. This penalty is a simple additive risk heuristic, not a confidence interval for the mean, a portfolio variance calculation, or a solution for normalized rank. It can suppress weak forecasts and leave budget unused. It may reduce both profit variance and mean profit; full fills can increase downside exposure.

**Decision:** report asset and sector exposure concentration rather than adding an unsupported cap. Correlated sector losses matter to risk even though expected utility remains additive. Require evidence before replacing the DP with coupled risk optimization. Do not tune aggression from our own score alone: we do not observe the field's score distribution or final normalization.

## Evaluation designed for the actual concern

**Evidence:** the supplied local harness has only four weak opponents and extra liquidity reduction. Passing it establishes compatibility, not competitiveness. Historical prices alone contain no recorded opponent-flow ground truth, so crowd estimates cannot be validated directly from `history.csv`.

**Decision:** in the later custom evaluator, use 20 opponents plus us, each subject to the same 14-lot budget (or supplied budget). Price forecasts and orders at round t may use only information available before t. Score all teams simultaneously with the exact resolver. Keep the organizer's harness unchanged.

Use these four fixed fields, rather than a large hyperparameter search:

| Field | 20 opponents |
|---|---|
| Correlated forecasting | 16 sharing the same forecast family, 2 momentum, 2 mean reversion |
| Mixed competition | 8 correlated forecast-followers, 6 momentum, 6 mean reversion |
| Adaptive competition | 10 P1 crowd-aware agents, 6 forecast-followers, 2 momentum, 2 mean reversion |
| Clone stress | 20 agents using the same P1 policy and forecast as us |

The clone field checks symmetry and collective crowd avoidance failure, not an expectation that we beat exact copies. Deterministic agents with identical observations and histories should tie. Randomness or a fixed ticker preference does not manufacture predictive edge. Include small fixed forecast variations in the correlated field to avoid evaluating only perfect clones.

**Assumption:** momentum uses the past three observed changes; mean reversion uses the last price's deviation from its trailing up-to-20-price mean. Both allocate proportionally to signal magnitude, without future data. Forecast-followers use frozen development-trained models or a shared causal simple predictor if models are not ready. Score model-ready and simple-predictor results separately.

Start with five fixed seeds, four fields, 150 rounds and all three policies (or P0/P1 if P2 lacks residual estimates). Use identical price schedules and opponent randomness for paired comparisons; volume histories must evolve separately for each policy because our orders affect them. Use the production-scale session liquidity above with no 0.3 factor. Add a bounded stress repeat at 0.5x and 2x those liquidities, plus a no-dilution sanity check. If time permits, repeat at 19 opponents to cover the written 20-team rule.

Use both chronological development-price replay and the harness generator if feasible. Replay is conditional on simulated opponents; the generator is conditional on its chosen price process. Neither proves hidden tournament performance. In particular, do not exploit the harness's exact reversion coefficient as if verified for production.

Report paired total-profit differences versus P0, mean score, worst-seed score, dispersion, dilution on winning and losing trades separately, unused budget, maximum asset/sector exposure, and runtime. With five seeds, avoid a precise tail quantile or confident rank probability. Report per-field results; averaging can conceal failure against correlated agents.

Also measure raw counterfactual profit `sum(q * realized_move * K)` alongside actual profit. Do not optimize a fill ratio or crowd-avoidance count as a substitute for profit: reducing fills can help a losing strategy appear less bad. Forecast MAE/RMSE must be identical across allocation policies sharing the same predictor.

**Decision:** retain P1 only if paired development results improve in both correlated and mixed fields, with no material collapse in adaptive or liquidity stress cases. Inspect worst-seed deterioration explicitly. Choose P2 only if its risk reduction justifies the observed mean-profit tradeoff; otherwise keep P1. If evidence is absent or gains are inconsistent, retain P0 as the validated fallback and call the crowd advantage unproven. No leaderboard guarantee follows from this gate.

Preserve the first approximately 60% / next 20% / final 20% chronological split from the project plan. Strategy selection uses development data; final holdout is run once after freezing choices. Earlier prices may provide context for later windows, but future labels may not enter fitting or simulation decisions.

## Edge cases and verification contract

- Zero order produces zero utility; zero budget and no valid current prices produce all-zero orders.
- Every output is a finite native integer; all six asset keys are present; absolute sum respects the supplied budget.
- Empty, short, constant, missing, or nonfinite price/volume histories follow explicit fallbacks. No current price means no trade on that asset. Invalid public-input structure should carry a contextual diagnostic and safe action; unexpected programming errors should not be silently swallowed.
- Exactly corrected own orders are subtracted once; missing/misaligned records use the documented shrinkage fallback. Test differing list lengths, skipped round IDs, negative volumes, and own-order inconsistencies.
- Test no dilution, buy/sell dominance, both kinds of tie-crossing, exact liquidity, overflow, no opponents, and positive/negative realized returns.
- Check tiny DP results by exhaustive enumeration, including nonmonotone utility tables, all-negative utility, and tied optima with unused budget.
- Verify deterministic repeated calls. Import the submitted files in a fresh process from a different working directory; include model loading in the timeout measurement.
- Validate scenario totals against the assumed opponent budget and ensure candidate own flow is added exactly once.

**Evidence from this planning session:** a temporary standard-library verification compared the formula against the actual harness resolver for 1,392 combinations: opponent buy/sell volumes in `{0, 5, 12, 30}`, every order from -14 to 14, and moves in `{-2, 0, 2}`. All matched at regular-session harness liquidity. The numerical examples were also checked by calculation/enumeration. This verifies scoring arithmetic only, not the proposed estimator, optimizer implementation, or performance.

**Additional design verification:** a temporary implementation of the DP recurrence matched exhaustive enumeration on 30 seeded three-asset, four-lot signed utility tables with nonmonotone values. All-negative utilities, zero budget, minimum-exposure tie selection, and the dominance-transition counterexample also passed. This validates the recurrence; the eventual production implementation still requires its own checks. No production files or persistent test files were changed.

## Open decisions for discussion

1. **Current crowd belief:** default to shrunk past flow plus the shared-forecast scenario. The consequential question is whether the current-signal world improves decisions when historical crowd direction flips. Validation must decide; its 35% weight is not evidence about actual teams.
2. **Liquidity:** default to the stated production-scale grid until organizers disclose visibility or parameters. A claim that the liquidity is known would materially change sizing and should replace this assumption explicitly.
3. **Profit versus downside:** recommend P1 first, with P2 as the only confidence-penalty trial. The unknown normalized scoring function prevents a principled claim that either maximizes final rank. Discuss measured mean-profit cost before preferring P2.

Confirmed user steering: 21 teams total, evaluated together. No additional metric or liquidity disclosure has been supplied. The recommendation is ready for an implementation handoff; this document does not itself authorize that implementation or enact the attached multi-agent schedule.

## Build now

- **Evidence-backed:** preserve independent forecast outputs and match signed-order scoring, including buy-wins-ties and both wins/losses.
- **Assumption:** implement the explicit recency/shrinkage flow estimator and nine small scenarios as configurable constants in existing allocation code.
- **Evidence-backed:** evaluate all signed sizes and use exact separable budget DP with zero orders and unspent budget allowed.
- **Validation candidate:** compare P0 and P1 first; add P2 only with available development residual estimates.
- **Evidence-backed:** validate against 20 opponents, including correlated and adaptive agents, at production-scale liquidity; retain the simple fallback if the hypothesis fails.
- **Assumption:** fit implementation to the requested nine-minute allocation window: about 2 minutes estimator/scenarios, 3 scoring/DP, 2 correctness checks, 2 bounded comparison/handoff. If time slips, drop P2 and extra stress runs before correctness checks; report incomplete performance validation explicitly.

## Defer

- **Validation candidate:** learn opponent response to current momentum/reversion signals only after observed-volume prediction establishes added value over shrinkage.
- **Validation candidate:** estimate joint return/fill outcomes or infer liquidity from aligned own P&L; multi-asset aggregate profit makes attribution nontrivial.
- **Validation candidate:** randomized allocation among near-equal positive-profit alternatives if clone tests show synchronized crowd avoidance hurts. First quantify expected-profit cost and confirm evaluation/reproducibility requirements; no claim that random trading creates alpha.
- **Validation candidate:** covariance-aware portfolio risk, online strategic adaptation, or equilibrium calculations if measured gains justify coupling and runtime costs.
- **Decision:** no deep RL, new service, GPU training, persistent runtime state, broad strategy search, or production edits in this planning task.
