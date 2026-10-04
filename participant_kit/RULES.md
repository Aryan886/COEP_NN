# Neural Nexus — Round 2: The Trading Pit

## The game, in one paragraph

Six assets, each with a "normal" price. Each round, actual prices wobble a
bit — sometimes trending for a stretch, sometimes just noisy. You get a
budget of 14 lots per round to spend however you like across the 6 assets.
**Positive lots = you're betting it goes up (buy). Negative lots = you're
betting it goes down (short).** Guess right, you profit; guess wrong, you
lose. One twist: if everyone piles onto the same side of the same bet, that
whole side's results get watered down — smaller wins, smaller losses, as if
there wasn't enough to go around. If you're one of the few people on the
other side, you get the full result, undiluted. So: predict well, and don't
just copy the obvious move everyone else is making.

Your submission is judged on TWO things:
1. **How accurately you can forecast prices** (scored offline, on data you
   never see).
2. **How well you turn that forecast into trading decisions** (scored live,
   in the final 20-team showdown on the big screen).

## Timeline

| Time | What happens |
|---|---|
| 0:00 | `history.csv` (over 2000 rounds of past prices) + this spec released |
| 0:00 – 0:50 | You analyze history, engineer features, train a model, write your agent |
| 0:50 | Submissions close: `agent.py` (+ optional model file) |
| Judging | Engine runs your `predict_market()` against a hidden validation set (offline score) |
| Live showdown | All 20 agents' `place_orders()` run simultaneously, live, ~15 minutes, 150 rounds |

## The assets

Assets come in three publicly-known correlated pairs — knowing this, and
building features around it, is a legitimate (and rewarded) part of the
challenge:

| Sector | Assets |
|---|---|
| Tech | Quantum Dynamics (QNTM), ByteStream (BYTE) |
| Hard Assets | Gold Trust (GLDT), Solar Grid (SLR) |
| Local Economy | Metro Rail (MTRL), AgroFutures (AGRO) |

(Exact names may be re-themed if the round is sponsored — the mechanics
below never change.)

## How a round works

Every round, each asset's price moves a little. That move comes from three
ingredients:
- **A persistent trend** specific to that asset — assets can run "hot" or
  "cold" relative to normal for a stretch of rounds. This is real,
  learnable signal.
- **A shared pull** with its sector partner (see table above) — sector
  pairs tend to move somewhat together.
- **Random noise** — a smaller, genuinely unpredictable wobble on top.

Prices always stay in a realistic range around their normal level — they
don't run away to zero or to the moon. You never see the price *before* it
happens — only what already happened in past rounds, which is what your
model should be trained to predict.

## Scoring math — read this carefully, it's exactly what runs live

Every round, for every asset:
```
total_buy  = sum of all POSITIVE lots any team sent to that asset
total_sell = sum of all NEGATIVE lots (as a positive number) any team sent

dominant_side  = whichever of buy/sell has the larger total this round
dominant_total = that larger total
liquidity      = how much of that asset's flow the market can absorb before diluting

if dominant_total <= liquidity:
    discount = 1.0                                    # no crowding, nobody diluted
else:
    overflow_ratio = dominant_total / liquidity
    discount = 1.0 / overflow_ratio                     # steep -- the more crowded, the worse

for each team's order on this asset:
    raw_pnl = lots x price_move_this_round x SCALE       # standard position P&L: size x how much it moved
    # lots and price_move share a sign when you guessed the direction right
    if this team's side (buy/sell) == dominant_side:
        actual_pnl = raw_pnl x discount                  # crowded side: diluted, win or lose
    else:
        actual_pnl = raw_pnl                              # quiet side: full strength, win or lose
```

**In plain terms**: if you're buying (or shorting) the SAME side as most of
the market this round, both your wins and your losses shrink -- you only got
a partial fill, effectively. If you're on the quieter side, you get the
full, undiluted result. Being right while being contrarian pays off fully;
being right while following the crowd pays off less.

Your total score is the sum of `actual_pnl` across all 6 assets, every
round.

## Your submission: `agent.py`

Exactly two functions. **Both are called fresh, in a new process, every
round -- nothing you store in a global variable or `self.` survives between
calls.** Anything you need to remember must come from the arguments you're
given each time, or be loaded from a file you submitted.

**Loading your model file -- read this carefully.** Your model file must be
loaded using a path relative to your OWN agent.py file, never a bare
filename. The judging engine runs your agent.py from a different working
directory than your submission folder -- a bare filename will work when
you test locally (by coincidence, since local_test_harness.py happens to
run from your own folder) and then fail on the real engine. Always do it
like this:
```python
import os, joblib
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_model.pkl")
MODEL = joblib.load(MODEL_PATH)
```

```python
def predict_market(recent_history):
    """
    recent_history: dict[asset_id -> list of up to the last 20 REALIZED
                    prices for that asset, oldest first, most recent last]

    Returns your predicted PRICE for the UPCOMING round, per asset:
      {"quantum_dynamics": 41.2, "byte_stream": 77.5, "gold_trust": 33.0,
       "metro_rail": 22.4, "agro_futures": 28.1, "solar_grid": 39.6}

    Scored OFFLINE against a hidden validation set you never see. It is NOT
    told what other teams are doing -- it's a pure forecasting task.
    """


def place_orders(game_state, team_history):
    """
    game_state: {
      "round": 47, "total_rounds": 150, "trade_budget": 14,
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
```

## Data provided

- `history.csv` -- over 2000 rounds of realized prices, one row per (round,
  asset). Has a small amount of realistic messiness (a few missing values)
  -- don't blindly drop rows, think about imputation.
- `starter_template.py` -- a working skeleton with a dummy strategy, so you
  have something submittable from minute zero.
- `local_test_harness.py` -- runs your `agent.py` against a few built-in
  dummy opponents using the EXACT same scoring math as production, so you
  can sanity-check before submitting.

## Scoring formula for the final leaderboard

```
FINAL_SCORE = 0.70 x normalized(live_match_score) + 0.30 x normalized(offline_forecast_accuracy)
```
- **Live match score (70%)**: your total profit across the 150-round live
  showdown, normalized against the field.
- **Offline forecast accuracy (30%)**: how close your `predict_market()`
  predictions are (lower error = better) to a HELD-OUT validation set,
  normalized against the field. Scored before the live show, on data you
  never had access to -- there's no way to game this except building a
  genuinely good model.

## Constraints

- **Per-call timeout**: ~1.5-2 seconds per function call. If your code
  hangs, crashes, or returns something malformed, that round defaults to a
  zero/rest action -- it does not stop the match for anyone else.
- **Allowed libraries**: standard library, `numpy`, `pandas`, `scikit-learn`,
  `scipy`. Small model artifacts only (keep your saved model fast to load --
  it's reloaded fresh every round).
- **No network access, no filesystem access outside your own submitted
  files.**
- **`place_orders` must return signed integers whose absolute values sum to
  at most your trade budget.** Invalid output defaults to an all-zero round.

## FAQ

**I've never traded before -- is that a problem?** No. "Buy" just means
"bet it goes up," "short" means "bet it goes down." You don't need to know
anything about real markets -- the whole game is: predict a number, decide
how confident you are, and watch out for crowds.

**Do I need deep learning?** No. A well-tuned regression or tree-based
model (linear/ridge regression, random forest, gradient boosting) is
completely sufficient and expected for most teams. A small neural net is a
legitimate option, not a requirement.

**Does buying or shorting actually move the price?** No -- prices move due
to the market's own trend/noise process, not your orders. What your orders
DO affect is how much of your own result you get to keep: crowding into
the popular side dilutes your outcome; being on the quiet side doesn't.

**What if my forecast is great but my strategy is bad (or vice versa)?**
That's exactly why scoring is split -- we're testing both skills
separately, on purpose.

**How does `joblib` create `model.pkl` in the starter kit? Does order matter?**
By default, the starter script creates a standard Python dictionary mapping asset names to model objects (e.g., `models["quantum_dynamics"] = my_model`), and `joblib.dump` serializes that entire dictionary to a file. Order of insertion does not matter because it's a dictionary lookup by key.

**Can we use different models for different assets?**
Yes! Because `models` is just a Python dictionary, you can absolutely put an `XGBRegressor()` under one asset key and a `RandomForestRegressor()` under another. As long as your `agent.py` calls `.predict(X)` on the object it retrieves from the dictionary, it will work perfectly.
