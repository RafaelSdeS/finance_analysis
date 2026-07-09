# CLAUDE.md

Guidance for Claude Code working in this repo.

## Overview

**Project:** Brazilian-equity ML pipeline for RL-based portfolio allocation. See `docs/specification.txt` for system design and RL objective.

**Three stages** (all scripts run from project root):
1. **Data Collection** — staged prototype→validation→full-scale pipeline (checkpointing, logging, validation).
2. **Dataset Build** — merge raw data → derived features (technical, fundamental, macro) → clean → ML-ready parquet, no lookahead bias.
3. **ML Agent** — PPO RL agent trained on the Stage 2 dataset (`ml_agent` branch).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add BOLSAI_API_KEY=sk_...  (backfill only; .env is gitignored)
```

## Run Commands

### Stage 1: Collect Raw Data

```bash
# Backfill (one-time historical via BolsAI, 2000–present); resumes from checkpoints, idempotent
python -m src.data_collection.pipeline --mode full_scale            # all ~500+ tickers
python -m src.data_collection.pipeline --mode full_scale --dry-run  # preview ticker list
python -m src.data_collection.pipeline --mode prototype --tickers PETR4 VALE3

# Quarterly incremental refresh (free yfinance, no key; >99% cost savings)
python -m src.data_collection.pipeline --mode update

# Validate (cross-check vs yfinance, 1–15% tolerance on key ratios)
python tests/data_collection/validate_vs_yfinance.py
```

### Stage 2: Build ML Dataset

Prereq: Stage 1 complete (raw data in `data/raw/`). Merges prices + fundamentals + company info via `merge_asof` backward (no lookahead).

```bash
python -m src.build_dataset.build_ml_dataset            # → data/processed/ml_dataset.parquet
python tests/build_dataset/test_final_dataset.py        # schema, shape, lookahead, NaN, returns
```

### Stage 3: Train ML Agent (ml_agent branch)

Prereq: `pip install torch stable-baselines3 gymnasium scikit-learn`. Deep-dive: `docs/STAGE3_ML_AGENT.md`, `docs/ML_AGENT_ROADMAP.md`, `src/agent/README.md`.

```bash
# One-time data prep: returns from adj_close + env tensors + train-only scaler
python src/agent/feature_engineering.py
python -m src.agent.data_pipeline
python -m src.agent.data_pipeline --universe-size 50   # optional: restrict tensors to top-50 tickers

# Train PPO — always anchored rolling windows (8 windows by default, train anchored 2000, ~2y test each);
# one PPO model per window, 1M timesteps/window by default (smoke: --timesteps 12288)
python -m src.agent.trainer
python -m src.agent.trainer --timesteps 500000 --learning-rate 1e-4 --device cuda
python -m src.agent.trainer --train-years 2 --test-years 1 --timesteps 2048  # fast smoke: many small windows
python -m src.agent.trainer --universe-size 50 --bc-pretrain  # must match data_pipeline's --universe-size, or env.py raises RuntimeError

# Sanity-check the feature set has exploitable signal before trusting the agent to find it
python -m src.agent.ranker_baseline   # supervised HistGradientBoosting, daily rank-IC on held-out tickers

# Backtest vs equal-weight / market-cap / inv-vol baselines (uses the most recent window's test split)
python -m src.agent.evaluate --model data/models/agent_best.zip

# Online retraining: continuous rollout with trailing-window fine-tuning every N days
# Retrain every 63 trading days on the last ~3 years of data, with revert-if-worse guard
python -m src.agent.rolling_eval --mode online_backtest --resume  # resume from checkpoint if exists

# Daily inference (uses production model from most recent training window)
python -m src.agent.run_allocation --date 2026-06-29 --format csv
```

**Note on recent changes (July 2026):** (a) Agent reward changed from absolute portfolio return to excess-return signal (excess of market mean). (b) Feature set upgraded 24 → 40 (raw OHLC dropped; sector-relative z-scores, momentum, quality trends, dividend signals added). (c) Actions are now temperature-scaled (`logit_scale=10` in env's masked softmax) so PPO can escape the uniform/equal-weight initialization — without it the policy provably freezes at equal-weight (trust-region math in `AgentConfig.logit_scale` comment). All models trained before any of these changes are stale; retraining required. See "Agent conviction improvements" in `STAGE3_ML_AGENT.md`.

**Outputs:** each `trainer.py` invocation trains all windows under its own scratch dir `data/models/runs/<session_id>/{window_{id},agent}_{best,final}.zip` (checkpoints deleted once a window finishes); the most recent window's model is then promoted to the stable, unmoving production path `data/models/agent_{best,final}.zip` (what `evaluate.py`/`infer.py`/`run_allocation.py` default to); scaler `data/models/feature_scaler.pkl`; online-retraining artifacts `data/models/online/agent_online_*.{zip,pkl}`; env tensors `data/processed/agent_tensors.npz` ([6565 dates × 280 tickers × 40 features] + mask); backtest `data/backtest/{metrics.json,results.parquet}` + `plots/*.html`; stitched multi-window walk-forward `data/backtest/{walkforward_results.parquet,walkforward_metrics.json}`; daily weights `data/allocations/allocation_YYYY-MM-DD.{csv,json}`.

### Utilities

```bash
python src/build_dataset/cagr_handler.py --ticker PETR4  # CAGR calculator
python src/visualizations/financial_view.py         # BBAS3 nominal vs inflation-adjusted vs SELIC (live yfinance)
jupyter notebook src/visualizations/exploration.ipynb   # full dataset validation + insights
streamlit run tools/explorer/app.py                 # interactive explorer: Data Explorer / Data Quality / Model Explorer / Training Analysis pages
```

### Tests

Plain Python scripts, no pytest. Unified test runner:

```bash
# Fast group (pure-code unit tests, no data files needed — used by CI)
python tests/run_all.py --group fast

# Data group (needs git-tracked data/raw/* and built data/processed/ml_dataset.parquet)
python tests/run_all.py --group data

# All tests
python tests/run_all.py --group all
```

**Test groups:**
- **Fast:** `test_build_dataset_features.py` (unit tests for Stage 2 feature functions)
- **Data:** `test_final_dataset.py`, `test_cagr_calculation.py`, `test_backtest_metrics.py`, `test_feature_engineering.py`, `verify_dataset_for_training.py` (gates V1–V7), `test_env_basic.py` (env invariants), `test_inference_output.py`

**Linting:**
```bash
ruff check .          # reports undefined names, unused imports/variables, bare-except
```

## Branches

- **main:** Stages 1–2 (data collection + dataset build). Latest stable.
- **build_dataset:** Stage 2 focus.
- **ml_agent:** Stage 3 RL agent. See `docs/ML_AGENT_ROADMAP.md`.

## Architecture

### Data Flow

```
BCB SGS (SELIC/CDI/IPCA) + BolsAI (OHLCV + quarterly fundamentals + dividends)
        ↓
data/raw/{prices,fundamentals,macro,dividends,company_info}/
        ↓ build_ml_dataset.py
  merge_asof(prices, fundamentals) [no lookahead] → left join company_info
  → compute_price_features()       [RSI, MA20/60, volatility, returns, drawdown]
  → compute_fundamental_features() [P/E, P/B, ROE, debt/equity, growth CAGR]
  → compute_macro_features()       [real return, excess return, rate environment]
  → fill_missing_cagr()            [backfill from earnings/revenue where API null]
  → clean (dupes, NaNs, outliers, sort)
        ↓
data/processed/ml_dataset.parquet  (one row per ticker+date)
        ↓ Stage 3
PortfolioEnv (gymnasium) → PPO trainer → evaluate vs baselines
        ↓
data/backtest/results.parquet, data/allocations/*.csv
```

### Key Modules

**Stage 1 (Data Collection)** — `src/data_collection/`, flat-function, source-agnostic:

| Module | Purpose |
|--------|---------|
| `config.py` | Shared config (tickers, keys, paths, retries, `DATA_SOURCE` per-type source switch) |
| `client.py` | BolsAI HTTP wrapper (retries, backoff): `make_client()`, `get_json()` |
| `checkpoint.py` | Resume state (JSON per collector) |
| `validate.py` | Quality gates (schemas, ranges, continuity) → `ValidationResult` |
| `collectors.py` | BolsAI: `collect_{macro,prices,fundamentals,company_info,dividends}()`; helper `_merge_save()` (idempotent append+dedup+validate+write) |
| `yf_collectors.py` | yfinance: `collect_{prices,fundamentals,dividends}_yf()` for `--mode update` |
| `pipeline.py` | Orchestration; `_collect()` dispatches to BolsAI/yfinance per `DATA_SOURCE` |

**Stage 2 (Dataset Build):**

| File | Purpose |
|------|---------|
| `src/build_dataset/build_ml_dataset.py` | Orchestration: load → merge_asof → features → clean → save. Also defines `load_prices()`, `load_fundamentals()` |
| `src/build_dataset/cagr_handler.py` | CAGR calc/fill (BolsAI first, backfill from earnings/revenue) |

**Stage 3 (ML Agent)** — `src/agent/`, all implemented:

| File | Purpose |
|------|---------|
| `config.py` | Frozen `AgentConfig`: hyperparams, 40-feature list, paths, `generate_windows()`/`window_to_config()`; `DEFAULT_CONFIG` = most recent rolling window |
| `feature_engineering.py` | Returns from `adj_close` (split-safe), corrupt-observation cleaning |
| `data_pipeline.py` | Pivot to dense tensors [dates×tickers×features] + activity mask; train-only scaler (fit on the earliest window's train_end — conservative, no lookahead into any window's test) |
| `env.py` | PortfolioEnv: masked time-varying universe, masked-softmax action, excess-return reward (net of market mean), transaction-cost term; cached load+normalization for online training |
| `metrics.py` | Sharpe, Sortino, max drawdown, win rate, max_weight, avg_daily_turnover (shared by trainer + evaluator) |
| `trainer.py` | SB3 PPO per window, val-Sharpe eval callback, JSONL logging, checkpoints, early stopping; `main()` = sole rolling-window training CLI |
| `evaluate.py` | Backtest vs 3 baselines, metrics.json, Plotly plots; supports both anchored-window and continuous-rollout backtests |
| `infer.py` | `predict_weights(date)` reusing env obs pipeline; equal-weight fallback |
| `run_allocation.py` | Daily CLI: weights + sector → CSV/JSON |
| `rolling_eval.py` | Window training/eval orchestration (`trainer.py`); online continuous-rollout backtest with trailing-window fine-tuning; walk-forward stitching; checkpoint resume support |

No `policy.py` — SB3's built-in `MlpPolicy` is used. Progress via SB3 `progress_bar=True`.

### Key Design Decisions (Stage 3)

- **Anchored rolling windows, never a fixed split:** training always partitions the dataset into anchored windows (`config.generate_windows()`; default 8 windows, train_years=10, test_years=2, train always starts at `dataset_start`, test slides forward). Each window's train span is tail-carved (`window_val_fraction`, default 15%) into train/val for early stopping; its test span is untouched. The MOST RECENT window is the production model (`agent_best.zip`) and `DEFAULT_CONFIG`'s split; earlier windows are namespaced (`window_{id}_best.zip`) and exist for robustness reporting (`data/backtest/walkforward_*`). There is no flag to opt back into a single fixed split — `python -m src.agent.trainer` always trains this way.
- **Train-only scaler:** fit StandardScaler on the FIRST window's train span only (earliest of all windows' train_end, so it can never leak future data into any window's test), apply to val/test/inference; save to `feature_scaler.pkl` (no leakage).
- **State:** all tickers' 40 normalized features (stationary/relative only — raw OHLC dropped; incl. `has_fundamentals` flag so the agent can tell "no filing yet" from "average company") + per-ticker activity mask + previous weights (for turnover calculation) concatenated → 280×40 + 280 + 280 = **11,760-dim**. Runtime NaN → 0 (mean imputation); dataset on disk keeps honest NaNs.
- **Action:** temperature-scaled masked softmax over full 280-ticker universe: `weights = softmax(action × logit_scale)`, `logit_scale=10` (279 stocks + CASH; every ticker with ≥252 rows → no survivorship bias). Inactive tickers → −∞ → weight 0; active weights sum to 1. No shorting. The temperature is essential: at scale 1 PPO's trust region (`target_kl`) limits logit drift to ~0.007/update, so the policy stays frozen at the uniform (= equal-weight) softmax init. Conversion lives in `env.action_to_weights()` (shared by `step()` and `infer.py`); `evaluate.py` baselines pre-divide their log-weights by the scale.
- **Reward (July 2026 update):** excess-return signal = (portfolio log return) − (market-mean log return) − transaction_cost. Cost = `transaction_cost_bps / 10000 × one-way-turnover` (excluding CASH leg which trades free). The excess signal amplifies per-ticker alpha (~0.1%/day) and reduces market noise (±1–2%/day), making gradient credit assignment tractable. Episodes start/reset at 100% CASH (new investor), so the first allocation pays full deployment cost.
- **Algorithm:** PPO via stable-baselines3: lr=3e-4 (3e-5 for online fine-tuning), gamma=0.997 (effective horizon ~333 trading days), gae_lambda=0.95, ent_coef=0.001, log_std_init=-2.0 (σ ≈ 0.135; with logit_scale=10 → effective logit noise ~1.35). Early stopping: patience 8 evals, improvement threshold 0.05 val Sharpe (generous — concentration initially costs turnover before alpha shows).
- **Online retraining:** Post-deployment, fine-tunes every 63 trading days on the last ~3 years of data (not anchored to 2000), with a revert-if-worse guard. Supports continuous-rollout backtest (`--mode online_backtest`) and checkpoint resume (`--resume`). See `STAGE3_ML_AGENT.md` § "Online retraining" for details.

## Dataset Verification (before Stage 3 training)

`tests/agent/verify_dataset_for_training.py` runs gates V1–V7. Fail conditions:
- **V1 Date coverage:** span < 2 years.
- **V2 Ticker coverage:** < 20 tickers with full history (≥252 rows).
- **V3 Feature completeness:** any required column missing (`ticker, trade_date, close, volume, returns, pl, pvp, roe, selic, sector` — Brazilian names: `pl`=P/E, `pvp`=P/B).
- **V4 NaN rates:** any NaN in critical cols (`ticker, trade_date, close, volume, sector`), or > 10% fundamental-feature NaN on rows at/after each ticker's first filing (NaN before the first filing is structural — CVM digital filings start ~2011 — and not gated).
- **V5 Distributions:** duplicate (ticker,trade_date) pairs, or `|return_mean|` ≥ 0.05, or ≥1% of ticker-quarters with frozen P/L (stale-valuation regression guard).
- **V6 Lookahead:** any `reference_date > trade_date`.
- **V7 Sectors:** < 3 sectors represented.

## Critical Caveats

- **CAGR backfill is ON:** `fill_missing_cagr()` (which calls `fill_cagr_columns()` per ticker) runs unconditionally in `build_ml_dataset.py`'s main pipeline → dataset has `cagr_{earnings,revenue}_5y_final` populated.
- **No lookahead (Stage 2):** `merge_asof(..., direction='backward')` — a price never sees a future fundamental. Verify `reference_date <= trade_date` after merge.
- **Valuation ratios are re-anchored daily (July 2026):** BolsAI computes `pl/pvp/market_cap/p_*/ev_*` with the price at the filing date (`close_price`) and they'd stay frozen all quarter; `recompute_valuation_daily()` rescales them by `close/close_price` (exact for price-linear ratios; EV ratios rebuilt algebraically). `close_price` is dropped from the processed dataset; a `has_fundamentals` 0/1 column is added. Known ceiling: mid-quarter splits skew ratios until the next filing (build prints a warning).
- **All feature engineering is in Stage 2**, not deferred to the agent (technicals, fundamental ratios, macro-adjusted, CAGR backfill).
- **FIIs deferred:** stocks only (prices/fundamentals/dividends). FIIs are a separate asset class; add if agent scope expands to mixed-asset.
- **BolsAI:** key in `.env`, loaded by `config.load_env()` (stdlib parser). Backfill only — paid ~€0.10/1K calls. Caps: prices `limit<=5000` (date-window paginated), fundamentals `limit<=88` (use 80).
- **yfinance:** free incremental refresh. Prices/dividends full history to 2000; fundamentals only ~4–6 quarters (enough for quarterly refresh).
- **BCB series:** selic=11 (daily), cdi=12, ipca=433 — **NOT 432** (that's the annual meta target).
- **Benchmark:** BOVA11 (IBOV proxy ETF) collected automatically; prices only.
- **Company info:** BolsAI-only (CVM metadata, rarely changes); refresh via `--mode full_scale` when new IPOs appear.
- **Checkpoints/logs** (not git-tracked): Stage 1 `data/checkpoints/{mode}/`, `data/logs/collection/collection-*.log`; Stage 3 training sessions `data/logs/agent/runs/<session_id>/{train.log,{tag}.jsonl}`; standalone `evaluate.py` runs `data/logs/agent/evaluate/evaluate_<run_id>.log`.
- **Paths:** absolute via `Path(__file__).resolve().parents[N]`; always run from project root.

## Data on Disk

- **Raw (git-tracked):** prototype tickers PETR4, VALE3, WEGE3 + benchmark BOVA11 in `data/raw/{prices,fundamentals,macro,company_info,dividends}/`. Prices/macro/dividends current to 2026-06-30; fundamentals to 2026-03-31.
- **Processed:** `data/processed/ml_dataset.parquet` (created on first build), one row per ticker+date.

## Technology Stack

- **Python 3.10+** (`list[str]`, `dict | None`).
- **Data:** pandas, numpy, pyarrow.
- **APIs:** BolsAI REST (`httpx`, backfill), BCB SGS (requests, macro), `yfinance` (incremental).
- **Config:** stdlib `.env` parser (BolsAI key only).
- **Viz:** Plotly. **ML/RL:** `torch==2.12.1` (CUDA), `stable-baselines3==2.9.0`, `gymnasium==1.3.0`, `scikit-learn==1.9.0` (older pins broke on numpy 2.x).
- **No test framework:** standalone `python script.py`.
