# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Project:** Brazilian-equity ML pipeline for reinforcement-learning-based portfolio allocation.

**Goal:** Collect daily stock prices and quarterly company fundamentals, then build a machine-learning dataset ready for model training. See `specification.txt` for system design and RL objective.

**Pipeline:** Structured three-stage approach:
1. **Stage 1 (Data Collection):** Staged prototype→validation→full-scale pipeline with checkpointing, logging, validation
2. **Stage 2 (Dataset Build):** Merge raw data → add derived features (technical indicators, fundamentals, macro-adjusted) → clean → output ML-ready parquets (no lookahead bias)
3. **Stage 3 (Model):** RL agent training (future, separate branch; consumes feature-complete dataset from Stage 2)

All scripts run from project root.

## Setup

```bash
pip install -r requirements.txt
```

## Run Commands

### Stage 1: Collect Raw Data

**Prerequisites:**
```bash
# Set up environment (one-time)
pip install -r requirements.txt
# Copy .env.example to .env and add your BolsAI API key
cp .env.example .env
# Edit .env and add: BOLSAI_API_KEY=sk_...
```

**Prototype stage** (3–10 representative tickers, validates data quality):
```bash
python -m src.data_collection.pipeline --mode prototype
```
Collects: BCB macro (SELIC, CDI, IPCA), BolsAI prices + fundamentals + company info for PETR4, VALE3, WEGE3, ...

**Validation stage** (after prototype):
```bash
python tests/raw_data/validate_vs_yfinance.py
```
Cross-checks prototype data against yfinance; pass/fail determines full-scale readiness.

**Full-scale stage** (after validation passes):
```bash
python -m src.data_collection.pipeline --mode full_scale       # all ~500+ tickers
python -m src.data_collection.pipeline --mode full_scale --dry-run   # preview ticker list
python -m src.data_collection.pipeline --mode prototype --tickers PETR4 VALE3   # override
```
Resumes mid-run from checkpoints (idempotent: re-runs only fetch new data).

### Stage 2: Build ML Dataset

**Prerequisites:** Stage 1 must be complete (raw data in `data/raw/`).

Merges prices + fundamentals + company info (no lookahead bias; uses `merge_asof` backward):
```bash
python "src/2. build_dataset/build_ml_dataset.py"
```
Saves to `data/processed/ml_dataset.parquet` (one row per ticker + date).

**Validation** (after build):
```bash
python tests/processed_data/test_final_dataset.py
python tests/processed_data/test_final_dataset.py --file data/processed/ml_dataset.parquet
```
Checks schema, shape, lookahead, NaN counts, return distribution.

See `BUILD_DATASET_ROADMAP.md` for phase-by-phase implementation guide (5 phases: load, merge, feature engineering, clean, validate).

### Utilities

**CAGR calculator** (CLI and module):
```bash
python src/cagr_handler.py --ticker PETR4
```

**Visualization — Quick Snapshot** (BBAS3 nominal vs inflation-adjusted vs SELIC, live yfinance data):
```bash
python src/visualizations/financial_view.py
```

**Exploration Notebook** (full dataset validation + insights, Jupyter):
```bash
jupyter notebook src/visualizations/exploration.ipynb
```
Charts: price coverage, data completeness, liquidity, sector breakdown, inflation-adjusted returns (PETR4/VALE3/WEGE3/ITUB4), P/E/ROE/net margin by sector, market cap distribution, leverage, growth CAGR, dividend analysis.

### Tests

All tests are plain Python scripts (no pytest). Run from project root:
```bash
python tests/processed_data/test_final_dataset.py
python tests/processed_data/test_final_dataset.py --file data/processed/ml_dataset.parquet
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY
python tests/raw_data/test_cagr_calculation.py
```

## Branches

- **main:** Base branch, stable. Stages 1–2 merged here once validated.
- **build_dataset:** Stage 2 (dataset building). Focus on merging raw data → ml_dataset.parquet. See `BUILD_DATASET_ROADMAP.md`.
- **ml_agent:** Stage 3 (RL agent). Separate effort after Stage 2 complete. See `ML_AGENT_ROADMAP.md` (if exists on that branch).
**Generic API Endpoint Tester** (explore BolsAI endpoints without writing code):
```bash
# Test specific endpoint + params
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY --path /dividends/PETR4 --param years=5

# Run full validation suite (9 checks)
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY
```

## Architecture

### Data Flow

```
External APIs
├─ BCB SGS (macro: SELIC, CDI, IPCA)
└─ BolsAI (prices OHLCV + fundamentals quarterly + dividends)
        ↓
data/raw/
  ├─ prices/{TICKER}.parquet           (daily)
  ├─ fundamentals/{TICKER}.parquet     (quarterly)
  ├─ dividends/{TICKER}.parquet        (historical, ~20 years)
  ├─ macro/{selic,cdi,ipca}.parquet
  └─ company_info/company_info.parquet
        ↓
build_ml_dataset.py
  → merge_asof(prices, fundamentals)   [no lookahead]
  → left join company_info
  → compute_price_features()           [RSI, MA20/60, volatility, returns, drawdown]
  → compute_fundamental_features()     [P/E, P/B, ROE, debt/equity, growth CAGR]
  → compute_macro_features()           [real return, excess return, rate environment]
  → fill_cagr_columns()                [backfill from earnings/revenue where API null]
  → clean (drop dupes, NaNs, outliers, sort)
        ↓
data/processed/ml_dataset.parquet      (one row per ticker+date, includes dividend data)
```

### Key Modules

**Stage 1 (Data Collection):**

| File | Purpose |
|------|---------|
| `src/data_collection/pipeline.py` | Orchestration (prototype/validation/full-scale modes, checkpointing) |
| `src/data_collection/collectors.py` | All collectors: BCB macro, BolsAI prices/fundamentals/company info |

**Stage 2 (Dataset Build, build_dataset branch):**

| File | Purpose |
|------|---------|
| `src/data_loading.py` | Utilities: `load_prices()`, `load_fundamentals()`, `load_macro_series()`, `inspect_dataset()` |
| `src/2. build_dataset/build_ml_dataset.py` | Orchestration: load → merge_asof → feature engineering → clean → save ml_dataset.parquet |
| `src/cagr_handler.py` | CAGR calculation/filling: use BolsAI values first, backfill from earnings/revenue. Uncomment line 143 in build_ml_dataset.py to enable. |

**Utilities:**

| File | Purpose |
|------|---------|
| `src/visualizations/financial_view.py` | Standalone Plotly chart: BBAS3 nominal/inflation-adjusted prices + SELIC overlay (uses `yfinance`). |

## Critical Caveats

### Stage 2: No Lookahead Bias (Temporal Merge)
The merge of prices + fundamentals uses `pd.merge_asof(..., direction='backward')` to ensure no price date ever sees a *future* fundamental (e.g., a price from 2026-04-01 gets the Q1 2026 fundamental dated 2026-03-31, not a later quarter). This is critical for valid backtesting.

**Check:** After merge, verify `fundamental_date <= price_date` for all rows.

### Stage 2 Feature Engineering (Phase 3)
All feature engineering happens in Stage 2, not deferred to Stage 3 (RL Agent). This includes:
- **Technical indicators:** RSI, moving averages, volatility, momentum, drawdown
- **Fundamental ratios:** P/E, P/B, ROE, leverage, growth CAGR
- **Macro-adjusted:** real return, excess return, rate environment
- **CAGR backfill:** `fill_cagr_columns()` (currently commented out at line 143 in `build_ml_dataset.py`)

Uncomment line 143 to enable CAGR backfilling. See `BUILD_DATASET_ROADMAP.md` Phase 3 for full feature list and Phase 4a for NaN handling strategy.

### FIIs Are Deferred
Pipeline collects **stocks only** (prices, fundamentals, dividends). Real Estate Investment Trusts (FIIs) are a separate asset class with different fundamentals (NAV/P-VP vs earnings/revenue) and distributions (monthly vs irregular). Will add if Phase 3 RL agent scope expands to mixed-asset allocation.

### BolsAI API Key Handling
- Stored in `.env` (copied from `.env.example`)
- `.env` is gitignored (never commit your key)
- Loaded by `config.load_env()` (stdlib parser, no `python-dotenv` dependency)
- Required for `src/data_collection/pipeline.py` and API validator tests

### Data Collection Pipeline Structure
- All collection lives in `src/data_collection/` (old `src/1. collect_raw_data/` removed)
- **Staged approach:** Prototype with 3–10 tickers first, validate against yfinance, unlock full-scale
- **Checkpointing:** Pipeline resumes from `data/checkpoints/{mode}/` on interrupt (idempotent)
- **Logging:** All collector activity goes to `data/logs/collection-YYYYMMDD-HHMMSS.log`
- **API caps (probed):** prices `limit<=5000` (date-window paginated), fundamentals `limit<=88` (use 80)
- **BCB series:** selic=11 (daily rate), cdi=12, ipca=433. NOT 432 (that's the annual meta target)
- **Benchmark ticker:** BOVA11 (iShares Bovespa ETF, IBOV index proxy) collected automatically; prices only (no fundamentals/dividends, it's an ETF)

### Relative Paths
New pipeline uses absolute paths via `Path(__file__).resolve().parents[N]`. Run all commands from project root.

## Data on Disk

**Raw data** (tracked in git):
- Three prototype tickers: PETR4, VALE3, WEGE3
- Location: `data/raw/prices/`, `data/raw/fundamentals/`, `data/raw/macro/`, `data/raw/company_info/`
- Status: prices current (2026-06-02), fundamentals stale (2026-03-31), macro current (2026-06-02)

**Pipeline state** (NOT tracked in git):
- Checkpoints: `data/checkpoints/prototype/` and `data/checkpoints/full_scale/` (resume state per collector)
- Logs: `data/logs/collection-*.log` (timestamped collection runs)

**Processed dataset** (created on first `build_ml_dataset.py` run):
- Location: `data/processed/ml_dataset.parquet`
- One row per ticker + date (daily prices merged with quarterly fundamentals)

## Technology Stack

- **Python:** 3.10+ (uses `list[str]`, `dict | None` syntax)
- **Data:** pandas, numpy, pyarrow (parquet)
- **APIs:** BolsAI REST (direct `httpx`), BCB SGS (direct requests), `yfinance` (validation only)
- **Config:** `python-dotenv` (load `.env` for API keys)
- **Logging:** Python built-in `logging` module (file + console)
- **Viz:** Plotly (existing `financial_view.py`)
- **No test framework:** tests are standalone `python script.py` invocations (no pytest)

## Data Collection Modules (Stage 1)

Pipeline in `src/data_collection/` with flat-function architecture:

| Module | Purpose |
|--------|---------|
| `config.py` | Shared config (tickers, API keys, paths, retry limits) |
| `client.py` | HTTP wrapper (retries, backoff, logging); `make_client()`, `get_json()` |
| `checkpoint.py` | Resume state tracking (JSON per collector) |
| `validate.py` | Data quality gates (schemas, ranges, continuity); returns `ValidationResult` |
| `collectors.py` | All collectors: `collect_macro()`, `collect_prices()`, `collect_fundamentals()`, `collect_company_info()`, `collect_dividends()` |
| `pipeline.py` | Orchestration (stages in order, error handling, checkpointing) |

**Helper:** `_merge_save()` in collectors.py — idempotent append + dedup + validate + write (shared by all collectors)
