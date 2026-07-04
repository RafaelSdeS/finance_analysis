# Stage 3: ML Agent — How It Actually Works

Source: `src/agent/` (on the `ml_agent` branch). This explains the implemented mechanics — for run commands see `CLAUDE.md`, for the phase-by-phase build plan see `ML_AGENT_ROADMAP.md`, citations reference `docs/RESEARCH_REFERENCES.md`. **Status caveat, stated up front:** per `TODO.md`, only a 12K-timestep smoke test has been run so far — the full 1M-timestep training run per window hasn't executed yet, and the agent has not yet been confirmed to beat the equal-weight baseline (Sharpe 0.71). Everything below describes what the code does when run, not a claim that a fully-trained agent currently outperforms.

## Why RL, and why this reward shape

The task is framed as sequential decision-making — deciding a portfolio allocation each day given the current state — rather than price prediction, following the dynamic portfolio theory tradition (Merton, 1969) and its RL-for-trading successors (Moody & Saffell, 2001; Jiang, Xu & Liang, 2017) — all already in `RESEARCH_REFERENCES.md`. Moody & Saffell's central point, echoed by Théate & Ernst (2020), is that reward function design is the single most failure-prone part of an RL trading system — a naive reward (e.g. raw return with no cost term) trains an agent that churns the portfolio into oblivion. The reward function below is built directly against that lesson.

## State space

The observable universe is **280 slots**: 279 real B3 tickers with at least 252 trading rows, plus one synthetic `CASH` asset. Each slot carries **23 features** (6 price + 14 fundamental + 3 macro — the exact same feature set Stage 2 produces, minus a few dropped for the agent's purposes). Flattened, that's 280×23 = 6,440 values, plus a 280-length activity mask (which tickers exist/trade on this date) and a 280-length previous-weights vector (for turnover calculation) — **7,000 dimensions total**.

`CASH` is synthesized (`feature_engineering.py`) as a price series that compounds daily at the SELIC rate: `price_t = 100 * cumprod(1 + selic/100)`. Its 14 fundamental features are all set to zero — safe because the training-only `StandardScaler` (see below) gives a zero-variance column `scale_=1.0` rather than dividing by zero.

Features are normalized with a `StandardScaler` fit **only** on data through the **first** anchored window's `train_end` — deliberately not `config.train_end`, which for the production config is the *last* window's (later) train_end. Using the earliest possible cutoff means the same scaler is guaranteed to predate every window's test span, so it can never leak later-window distribution statistics into an earlier window's out-of-sample evaluation. After scaling, any remaining NaN/inf is zeroed, and inactive (masked) cells are forced to exactly zero.

## Action space

Actions are raw logits, one per slot, bounded to `[-10, 10]` — not weights directly. Inside `env.step`, a masked softmax converts logits into weights: inactive tickers get `-inf` before the softmax (so they receive exactly zero weight), and the result is renormalized over active tickers only. No shorting — weights are always non-negative and sum to one.

## Reward function — the exact formula

```
weights           = masked_softmax(action, active_mask_t)
traded            = sum(|weights - prev_weights|)  over non-CASH tickers only
transaction_cost  = traded * (transaction_cost_bps / 10_000)     # 10 bps -> 0.001 * traded
next_returns       = active_mask_{t+1} ? log_return_{t+1} : 0     # NaN treated as 0 (flat day)
simple_return     = dot(weights, expm1(next_returns)) - transaction_cost
reward            = log1p(max(simple_return, -0.9999))
portfolio_value  *= 1 + max(simple_return, -0.9999)
```

In words: each ticker's log return is converted to a simple return, weighted, netted against a turnover-proportional transaction cost, and the resulting portfolio simple return is converted back to log space for the reward — clipped so a catastrophic day never sends `log1p` to `-inf`. Turnover is the sum of absolute weight changes **excluding the CASH leg** — moving money into or out of cash costs nothing, only rebalancing among equities does. This matches Almgren & Chriss (2001)'s linear-cost term for execution modeling and Huberman & Stanzl (2005) on realistic trading-cost inventory models (both already in `RESEARCH_REFERENCES.md`) — a simple proportional-to-turnover cost rather than a fixed per-trade fee, since the actual cost driver is how much capital moves, not how many tickers are touched.

Each episode resets to **100% CASH** (a new investor with no positions), so the very first allocation decision pays the full deployment cost of moving into equities — a deliberate choice, not an oversight, since a real investor starting from scratch faces exactly that cost.

## Algorithm and hyperparameters

PPO via Stable-Baselines3, `MlpPolicy` with `net_arch=[256, 256]`:

| Hyperparameter | Value |
|---|---|
| learning_rate | 3e-4 |
| gamma (discount) | 0.99 |
| gae_lambda | 0.95 |
| ent_coef (entropy bonus) | 0.01 |
| n_steps (rollout length/env) | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| n_envs (parallel rollout workers) | 8 |
| total_timesteps per window | 1,000,000 |

PPO (Schulman et al., 2017) and its GAE advantage estimator (Schulman et al., 2015) — both already in `RESEARCH_REFERENCES.md` — were chosen over value-based methods (DQN) because the action space here is continuous-valued allocation weights rather than a discrete "buy/sell/hold" choice; PPO's clipped surrogate objective is also comparatively forgiving of the noisy, non-stationary reward signal financial time series produce.

## Anchored rolling windows — the training strategy

There is no single fixed train/val/test split. `generate_windows(dataset_start, dataset_end, train_years=10, test_years=2)` produces a sequence of **anchored** windows: every window's training span starts at the same `dataset_start` (2000-01-03) and simply grows, while each window's 2-year test span slides forward and never overlaps the previous window's test span. Concretely, with a ~26.5-year dataset this yields 8 non-overlapping test windows (2010–2012, 2012–2014, … through 2026-06-30). The **most recent** window is the production model (`agent_best.zip`); earlier windows are trained and evaluated too, namespaced (`window_{id}_best.zip`), purely to report how robust the strategy is across different historical periods (`data/backtest/walkforward_*`).

Within a window, the tail 15% of its training span (`window_val_fraction`) is carved off chronologically as a validation set for early stopping — always the most recent portion of that window's train span, still strictly before its test span begins.

This design follows López de Prado (2015, 2018) on walk-forward optimization (already in `RESEARCH_REFERENCES.md`, flagged there as the mandatory reading on avoiding lookahead bias) more literally than a single fixed split would: because training data always starts at the true beginning of the dataset and never shrinks, and the test window for the production model is always the newest available period, the setup naturally answers "how would this strategy have performed if trained on everything available up to that point?" for several different historical points, rather than betting the whole evaluation on one arbitrary split date.

**Leakage prevention is layered three ways**, and each layer independently would be sufficient — together they leave no plausible path for future information to enter a decision:
1. **Window construction** — training always ends strictly before that window's own test span begins.
2. **Validation carving** — the val slice is the tail of that same window's training span, still before test.
3. **The global scaler** — fit only through the *first* window's train_end (the earliest cutoff across all windows), so it can never see later-window statistics, regardless of which window's test span is currently being evaluated.

## Training loop

Training runs with `SubprocVecEnv` (`n_envs=8` parallel environments) for throughput; the validation environment is always a single unparallelized instance. A `ValSharpeCallback` evaluates the policy on the validation environment every `eval_freq * n_steps = 100 * 2048 = 204,800` timesteps (the threshold is timestep-based, not a strict modulo, since a vectorized env advances in batches of `n_envs` at a time). Each evaluation appends one JSON line to a training log (`{timesteps, val_sharpe, val_max_drawdown, val_final_value, timestamp}`) and saves a checkpoint. If validation Sharpe improves, the model is saved as `{tag}_best.zip` and a degradation counter resets; after **3 consecutive** evaluations without improvement (`early_stopping_patience`), training for that window stops early.

## Metrics — exact definitions

All operate on daily log returns (except drawdown/cumulative/annualized return, which take a portfolio value series). **Risk-free rate is assumed zero** throughout — Sharpe and Sortino are not adjusted for SELIC, deliberately kept simple:

- **Sharpe** = `mean(r) / std(r) * sqrt(252)` — Sharpe (1964, 1994), already in `RESEARCH_REFERENCES.md`.
- **Sortino** = `mean(r) / std(r[r<0]) * sqrt(252)` — downside deviation only. Sortino & van der Meer (1991), *Downside Risk*, is the origin of this metric; not currently listed in `RESEARCH_REFERENCES.md`.
- **Max drawdown** = `max((cummax(v) - v) / cummax(v))`, a positive fraction.
- **Cumulative return** = `v[-1]/v[0] - 1`; **annualized return** = the same, geometrically annualized by elapsed years.
- **Win rate** = fraction of days with strictly positive log return.
- **Effective N positions** = `mean(1 / sum(w**2))` — the inverse Herfindahl-Hirschman Index, a standard concentration metric: 10 equally-weighted positions give ≈10, one position gives 1.

Note: `RESEARCH_REFERENCES.md` cites Calmar (1991) (return/max-drawdown) as a target metric, but there is currently no `calmar_ratio()` function in `metrics.py` — it's derivable from the existing cumulative-return and max-drawdown outputs but isn't computed as a named metric today. Stated here so this doc doesn't overclaim what's implemented.

## Backtesting and baselines

`evaluate.py` rolls the trained agent and three baselines through the **same** `PortfolioEnv` instance type, so every strategy's portfolio math (reward, transaction costs) goes through the identical code path — the only thing that differs is which weights are proposed each step:

- **Equal-weight**: `1/n_active` across active tickers (CASH excluded from baselines — only the agent gets the option to hold cash).
- **Market-cap weighted**: proportional to the last known market cap, forward-filled (never looking ahead) to avoid gaps.
- **Inverse-volatility**: proportional to `1/trailing_60d_volatility` — a simple risk-parity-style construction following Clarke, De Silva & Thorley (2016) on volatility-targeting (already in `RESEARCH_REFERENCES.md`).

Results are written to `data/backtest/metrics.json` (per-strategy metric dict), `data/backtest/results.parquet` (daily values + weight columns for tickers that ever exceed 0.1% weight), and four Plotly plots (cumulative value, drawdown, return distribution, top-10 weights over time).

## Walk-forward stitching

Each rolling window's environment resets its portfolio to `initial_capital` independently, so naively concatenating raw portfolio-value series across windows would show a discontinuous jump back to 100,000 at every window boundary. `stitch_walkforward` instead concatenates the **log returns** across windows and recompounds a single continuous value curve: `values = initial_capital * exp(cumsum(concatenated_log_returns))`. This produces one honest out-of-sample equity curve spanning every window's test period, rather than eight disconnected mini-backtests.

## Inference

`predict_weights(date)` reuses the same environment/observation pipeline as training, resolves `date` to the last trading day at or before the requested date, and runs the model deterministically. It falls back to equal-weight allocation on **either** of two independent triggers: any exception during load/predict, or (after a successful predict) weights that are non-finite or don't sum to 1 within 1e-6. The rationale stated in-code is direct — a daily allocation decision must never crash. `run_allocation.py` is the daily CLI wrapper: it calls `predict_weights`, attaches each position's sector for readability, and writes CSV or JSON.

## Commands

See `CLAUDE.md` → "Stage 3: Train ML Agent" for training, evaluation, and daily-inference commands.
