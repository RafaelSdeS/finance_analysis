# CLAUDE.md

Guidance for Claude Code working in this repo.

## Overview

**Project:** Brazilian-equity ML pipeline for RL-based portfolio allocation. See `specification.txt` for system design and RL objective.

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
python tests/raw_data/validate_vs_yfinance.py
```

### Stage 2: Build ML Dataset

Prereq: Stage 1 complete (raw data in `data/raw/`). Merges prices + fundamentals + company info via `merge_asof` backward (no lookahead).

```bash
python "src/2. build_dataset/build_ml_dataset.py"      # → data/processed/ml_dataset.parquet
python tests/processed_data/test_final_dataset.py      # schema, shape, lookahead, NaN, returns
```

See `BUILD_DATASET_ROADMAP.md` for the phase-by-phase guide.

### Stage 3: Train ML Agent (ml_agent branch)

Prereq: `pip install torch stable-baselines3 gymnasium scikit-learn`. Deep-dive: `ML_AGENT_ROADMAP.md`, `src/agent/README.md`.

```bash
# One-time data prep: returns from adj_close + env tensors + train-only scaler
python src/agent/feature_engineering.py
python -m src.agent.data_pipeline

# Train PPO (smoke: --timesteps 12288)
python -m src.agent.trainer
python -m src.agent.trainer --timesteps 500000 --learning-rate 1e-4 --device cuda

# Backtest vs equal-weight / market-cap / inv-vol baselines
python -m src.agent.evaluate --model data/models/agent_best.zip

# Daily inference
python -m src.agent.run_allocation --date 2026-06-29 --format csv

# Robustness: anchored rolling-window eval (8 windows, train anchored 2000, ~2y test each)
python -m src.agent.rolling_eval
```

**Outputs:** models `data/models/agent_{best,final}.zip` + checkpoints; scaler `data/models/feature_scaler.pkl`; env tensors `data/processed/agent_tensors.npz` ([6565 dates × 279 tickers × 23 features] + mask); backtest `data/backtest/{metrics.json,results.parquet}` + `plots/*.html`; daily weights `data/allocations/allocation_YYYY-MM-DD.{csv,json}`.

### Utilities

```bash
python src/cagr_handler.py --ticker PETR4           # CAGR calculator
python src/visualizations/financial_view.py         # BBAS3 nominal vs inflation-adjusted vs SELIC (live yfinance)
jupyter notebook src/visualizations/exploration.ipynb   # full dataset validation + insights
```

### Tests

Plain Python scripts, no pytest. Run from project root.

```bash
# Stages 1–2
python tests/processed_data/test_final_dataset.py
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY   # full 9-check suite
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY --path /dividends/PETR4 --param years=5
python tests/raw_data/{test_cagr_calculation,test_ticker_data,inspect_all_data,inspect_company_info}.py

# Stage 3
python tests/agent/verify_dataset_for_training.py   # dataset quality gates V1–V7 (see below)
python tests/agent/test_env_basic.py                # env invariants: masking, weights, determinism, speed
python tests/agent/test_backtest_metrics.py         # metric functions vs hand-computed
python tests/agent/test_inference_output.py         # inference invariants + equal-weight fallback
```

## Branches

- **main:** Stages 1–2 (data collection + dataset build). Latest stable.
- **build_dataset:** Stage 2 focus. See `BUILD_DATASET_ROADMAP.md`.
- **ml_agent:** Stage 3 RL agent. See `ML_AGENT_ROADMAP.md`.

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
  → fill_cagr_columns()            [backfill from earnings/revenue where API null — currently OFF]
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
| `src/data_loading.py` | `load_prices()`, `load_fundamentals()`, `load_macro_series()`, `inspect_dataset()` |
| `src/2. build_dataset/build_ml_dataset.py` | Orchestration: load → merge_asof → features → clean → save |
| `src/cagr_handler.py` | CAGR calc/fill (BolsAI first, backfill from earnings/revenue). Enable via line 143 in build script. |

**Stage 3 (ML Agent)** — `src/agent/`, all implemented:

| File | Purpose |
|------|---------|
| `config.py` | Frozen `AgentConfig`: hyperparams, 23-feature list, verified split dates, paths |
| `feature_engineering.py` | Returns from `adj_close` (split-safe), corrupt-observation cleaning |
| `data_pipeline.py` | Pivot to dense tensors [dates×tickers×features] + activity mask; train-only scaler |
| `env.py` | PortfolioEnv: masked time-varying universe, masked-softmax action, log-return reward |
| `metrics.py` | Sharpe, Sortino, max drawdown, win rate (shared by trainer + evaluator) |
| `trainer.py` | SB3 PPO, val-Sharpe eval callback, JSONL logging, checkpoints, early stopping |
| `evaluate.py` | Backtest vs 3 baselines, metrics.json, Plotly plots |
| `infer.py` | `predict_weights(date)` reusing env obs pipeline; equal-weight fallback |
| `run_allocation.py` | Daily CLI: weights + sector → CSV/JSON |
| `rolling_eval.py` | Anchored rolling-window backtest; reuses `evaluate.py` policies |

No `policy.py` — SB3's built-in `MlpPolicy` is used. Progress via SB3 `progress_bar=True`.

### Key Design Decisions (Stage 3)

- **Temporal splits, never random:** train 60% (oldest) / val 20% / test 20% (most recent), by date range not row count. Prevents lookahead + respects regime shifts.
- **Train-only scaler:** fit StandardScaler on train, apply to val/test; save to `feature_scaler.pkl` for inference (no leakage).
- **State:** all tickers' 23 normalized features + per-ticker activity mask concatenated → 279×23 + 279 = **6,696-dim**. Runtime NaN → 0 (mean imputation); dataset on disk keeps honest NaNs.
- **Action:** masked softmax over full 279-ticker universe (every ticker with ≥252 rows → no survivorship bias). Inactive tickers → −∞ → weight 0; active weights sum to 1. No shorting.
- **Reward:** daily log return `log(V_t / V_{t-1})`. No transaction costs / risk penalties in v1 (switch to Sharpe-based if convergence poor).
- **Algorithm:** PPO via stable-baselines3 (lr=3e-4, gamma=0.99, gae_lambda=0.95).

## Dataset Verification (before Stage 3 training)

`tests/agent/verify_dataset_for_training.py` runs gates V1–V7. Fail conditions:
- **V1 Date coverage:** span < 2 years.
- **V2 Ticker coverage:** < 20 tickers with full history (≥252 rows).
- **V3 Feature completeness:** any required column missing (`ticker, date, close, volume, returns, pe_ratio, roe, selic, sector`).
- **V4 NaN rates:** > 10% overall, or any NaN in critical cols (`ticker, date, close, volume, sector`).
- **V5 Distributions:** duplicate (ticker,date) pairs, or `|return_mean|` ≥ 0.05, or extreme outliers.
- **V6 Lookahead:** any `fundamental_date > price_date`.
- **V7 Sectors:** < 3 sectors represented.

## Critical Caveats

- **`fill_cagr_columns()` is OFF:** commented out at line 143 of `build_ml_dataset.py` → dataset lacks `cagr_{earnings,revenue}_5y_final`. Uncomment to enable.
- **No lookahead (Stage 2):** `merge_asof(..., direction='backward')` — a price never sees a future fundamental. Verify `fundamental_date <= price_date` after merge.
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
