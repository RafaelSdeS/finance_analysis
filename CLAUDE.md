# CLAUDE.md

Guidance for Claude Code working in this repo.

## Overview

**Project:** Brazilian-equity dataset pipeline for ML applications.

**Two stages** (all scripts run from project root):
1. **Data Collection** — staged prototype→validation→full-scale pipeline (checkpointing, logging, validation).
2. **Dataset Build** — merge raw data → derived features (technical, fundamental, macro) → clean → ML-ready parquet, no lookahead bias.

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

### Utilities

```bash
python src/build_dataset/cagr_handler.py --ticker PETR4  # CAGR calculator
python src/visualizations/financial_view.py              # BBAS3 nominal vs inflation-adjusted vs SELIC (live yfinance)
jupyter notebook src/visualizations/exploration.ipynb    # full dataset validation + insights
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
- **Fast:** `test_build_dataset_features.py`
- **Data:** `test_final_dataset.py`, `test_cagr_calculation.py`, validate_vs_yfinance.py

**Linting:**
```bash
ruff check .          # reports undefined names, unused imports/variables, bare-except
```

## Branches

- **main:** Stages 1–2 (data collection + dataset build). Latest stable.
- **build_dataset:** Stage 2 focus.

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
```

### Key Modules

**Stage 1 (Data Collection)** — `src/data_collection/`, flat-function, source-agnostic:

| Module | Purpose |
|--------|---------|
| `config.py` | Shared config (tickers, keys, paths, retries, `DATA_SOURCE` per-type source switch) |
| `client.py` | BolsAI HTTP wrapper (retries, backoff): `make_client()`, `get_json()` |
| `checkpoint.py` | Resume state (JSON per collector) |
| `validate.py` | Quality gates (schemas, ranges, continuity) → `ValidationResult` |
| `collectors.py` | BolsAI: `collect_{macro,prices,fundamentals,company_info,dividends,corporate_events,sectors}()`; helper `_merge_save()` (idempotent append+dedup+validate+write) |
| `yf_collectors.py` | yfinance: `collect_{prices,fundamentals,dividends}_yf()` for `--mode update` |
| `pipeline.py` | Orchestration; `_collect()` dispatches to BolsAI/yfinance per `DATA_SOURCE` |

**Stage 2 (Dataset Build):**

| File | Purpose |
|------|---------|
| `src/build_dataset/build_ml_dataset.py` | Orchestration: load → merge_asof → features → clean → save. Also defines `load_prices()`, `load_fundamentals()` |
| `src/build_dataset/cagr_handler.py` | CAGR calc/fill (BolsAI first, backfill from earnings/revenue) |


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
- **Checkpoints/logs** (not git-tracked): Stage 1 `artifacts/checkpoints/{mode}/`, `artifacts/logs/collection/collection-*.log`.
- **Paths:** absolute via `Path(__file__).resolve().parents[N]`; always run from project root.

## Data on Disk

- **Raw (git-tracked):** full-scale universe, ~293 tickers + benchmark BOVA11, one parquet per ticker in `data/raw/{prices,fundamentals,dividends}/`. Prices/dividends current to 2026-06-30; fundamentals to 2026-03-31. Coverage isn't 100% uniform across types (e.g. a handful of tickers are missing a dividends file) — treat gaps as "not yet collected," not "confirmed zero," and re-run the relevant `collect_*` for that ticker to check.
- `data/raw/macro/{selic,cdi,ipca}.parquet` (one file per series) and `data/raw/company_info/company_info.parquet` (per-ticker static attributes: sector, cnpj, status, etc.) are market-wide reference tables, not per-ticker files.
- `data/raw/company_info/sectors.parquet` is a small aggregate `[sector name, ticker count]` table used to sanity-check how many companies fall in each sector — not a join key, not consumed by `build_ml_dataset.py`.
- `data/raw/corporate_events/corporate_events.parquet` is a market-wide split/inplit audit log; `company_info/sectors.parquet` and `corporate_events.parquet` are collected during `full_scale`/`prototype` runs only — skipped in `--mode update` so the free/keyless yfinance refresh path never needs a BolsAI key.
- **Processed:** `data/processed/ml_dataset.parquet` (created on first build), one row per ticker+date. Gitignored and fully regenerable from `data/raw/` via `build_ml_dataset.py` — treat anything else that shows up in `data/processed/` as generated by pipelines outside this repo's `src/` (e.g. an ML-agent branch) and not something this repo can rebuild.

## Technology Stack

- **Python 3.10+** (`list[str]`, `dict | None`).
- **Data:** pandas, numpy, pyarrow.
- **APIs:** BolsAI REST (`httpx`, backfill), BCB SGS (requests, macro), `yfinance` (incremental).
- **Config:** stdlib `.env` parser (BolsAI key only).
- **Viz:** Plotly.
- **No test framework:** standalone `python script.py`.
