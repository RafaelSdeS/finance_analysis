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

# Train PPO — always anchored rolling windows (8 windows by default, train anchored 2000, ~2y test each);
# one PPO model per window, 1M timesteps/window by default (smoke: --timesteps 12288)
python -m src.agent.trainer
python -m src.agent.trainer --timesteps 500000 --learning-rate 1e-4 --device cuda
python -m src.agent.trainer --train-years 2 --test-years 1 --timesteps 2048  # fast smoke: many small windows

# Backtest vs equal-weight / market-cap / inv-vol baselines (uses the most recent window's test split)
python -m src.agent.evaluate --model data/models/agent_best.zip

# Online retraining: continuous rollout with trailing-window fine-tuning every N days
# Retrain every 63 trading days on the last ~3 years of data, with revert-if-worse guard
python -m src.agent.rolling_eval --mode online_backtest --resume  # resume from checkpoint if exists

# Daily inference (uses production model from most recent training window)
python -m src.agent.run_allocation --date 2026-06-29 --format csv
```

**Note on recent changes (July 2026):** Agent reward function changed from absolute portfolio return to excess-return signal (excess of market mean) to improve conviction. All models trained under old reward are stale; retraining required. See "Agent conviction improvements" in `STAGE3_ML_AGENT.md`.

**Outputs:** production model (most recent window) `data/models/agent_{best,final}.zip` + checkpoints; earlier windows namespaced `data/models/window_{id}_{best,final}.zip`; scaler `data/models/feature_scaler.pkl`; env tensors `data/processed/agent_tensors.npz` ([6565 dates × 279 tickers × 24 features] + mask); backtest `data/backtest/{metrics.json,results.parquet}` + `plots/*.html`; stitched multi-window walk-forward `data/backtest/{walkforward_results.parquet,walkforward_metrics.json}`; daily weights `data/allocations/allocation_YYYY-MM-DD.{csv,json}`.

### Utilities

```bash
python src/build_dataset/cagr_handler.py --ticker PETR4  # CAGR calculator
python src/visualizations/financial_view.py         # BBAS3 nominal vs inflation-adjusted vs SELIC (live yfinance)
jupyter notebook src/visualizations/exploration.ipynb   # full dataset validation + insights
streamlit run tools/explorer/app.py                 # interactive explorer: Data Explorer / Data Quality / Model Explorer / Training Analysis pages
```

### Tests

Plain Python scripts, no pytest. Run from project root.

```bash
# Stages 1–2
python tests/build_dataset/test_final_dataset.py
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY   # full 9-check suite
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY --path /dividends/PETR4 --param years=5
python tests/data_collection/{test_cagr_calculation,test_ticker_data,inspect_all_data,inspect_company_info}.py

# Stage 3
python tests/agent/verify_dataset_for_training.py   # dataset quality gates V1–V7 (see below)
python tests/agent/test_env_basic.py                # env invariants: masking, weights, determinism, speed
python tests/agent/test_backtest_metrics.py         # metric functions vs hand-computed
python tests/agent/test_inference_output.py         # inference invariants + equal-weight fallback
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
| `config.py` | Frozen `AgentConfig`: hyperparams, 24-feature list, paths, `generate_windows()`/`window_to_config()`; `DEFAULT_CONFIG` = most recent rolling window |
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
- **State:** all tickers' 24 normalized features (incl. `has_fundamentals` flag so the agent can tell "no filing yet" from "average company") + per-ticker activity mask + previous weights (for turnover calculation) concatenated → 280×24 + 280 + 280 = **7,280-dim**. Runtime NaN → 0 (mean imputation); dataset on disk keeps honest NaNs.
- **Action:** masked softmax over full 280-ticker universe (279 stocks + CASH; every ticker with ≥252 rows → no survivorship bias). Inactive tickers → −∞ → weight 0; active weights sum to 1. No shorting.
- **Reward (July 2026 update):** excess-return signal = (portfolio log return) − (market-mean log return) − transaction_cost. Cost = `transaction_cost_bps / 10000 × one-way-turnover` (excluding CASH leg which trades free). The excess signal amplifies per-ticker alpha (~0.1%/day) and reduces market noise (±1–2%/day), making gradient credit assignment tractable. Episodes start/reset at 100% CASH (new investor), so the first allocation pays full deployment cost.
- **Algorithm:** PPO via stable-baselines3 with reduced exploration: lr=3e-4 (3e-5 for online fine-tuning), gamma=0.99, gae_lambda=0.95, ent_coef=0.0 (entropy disabled), log_std_init=-2.0 (σ ≈ 0.135, down from default 1.0). Lower exploration noise allows the agent to develop conviction (concentrated positions) instead of spreading equally.
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
- **Checkpoints/logs** (not git-tracked): `data/checkpoints/{mode}/`, `data/logs/collection-*.log`.
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
