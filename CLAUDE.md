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

# Fit the feature scaler (train-only, per split_config.json) — rerun after a rebuild
python -m src.build_dataset.scale_features               # → data/processed/scalers/{feature_scaler.joblib,scaler_metadata.json}
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
- **Fast:** `test_features.py`, `test_merge.py`, `test_cross_sectional.py`, `test_compute_features_chunked.py`, `test_split_config.py`, `test_dataset_versioning.py`, `test_scale_features.py`, `test_company_siblings.py`, `test_ticker_continuity.py`
- **Data:** `test_final_dataset.py`, `test_top_traded_quality.py`, `test_cagr_calculation.py`, `test_blue_chip_tickers.py`, `validate_vs_yfinance.py`, `test_collect_delisted.py`, `test_cvm_statements.py`

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
  + ml_dataset.manifest.json, split_config.json, dataset_v{N}/ snapshot
        ↓ scale_features.py (separate, deliberate step — not run every build)
data/processed/scalers/feature_scaler.joblib  (train-only fit, per split_config.json)
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
| `cvm_statements.py` + `cvm/` | CVM open-data collection (delisted fundamentals + real filing dates); thin `--step` CLI dispatching to `cvm/{http,crosswalk,statements,shares,ratios,company_info,filing_dates}.py`. `cvm/http.py` holds the one shared zip-download/retry implementation |

**Stage 2 (Dataset Build)** — `src/build_dataset/`, split by pipeline stage (each module below mirrors a section of the old monolithic `build_ml_dataset.py`):

| File | Purpose |
|------|---------|
| `build_ml_dataset.py` | Orchestration only: `main()` + the memory-bounded `compute_features_chunked()` (3-pass: per-ticker features → cross-sectional → clean+write) |
| `paths.py` | Shared path constants (avoids circular imports across the split modules) |
| `loaders.py` | `load_prices()`, `load_fundamentals()`, `load_company_info()`, `load_dividends()`, `company_siblings()` |
| `repair.py` | `repair_unadjusted_splits()` — rescales `adj_*` history where a split was left unadjusted |
| `continuity.py` | `apply_ticker_continuity()` — splices renamed/merged tickers |
| `quality_filters.py` | Coverage + filing-lag gates: `filter_tickers_with_no_fundamentals()`, `attach_filing_dates()`, `filter_excessive_filing_lag()` |
| `merge.py` | The 4 `merge_*` functions (prices+fundamentals, company_info, macro, dividends) |
| `features.py` | Per-ticker feature engineering: CAGR backfill, dividend yield, price technicals, fundamental ratios/trends, valuation re-anchoring, "advanced" contextual features |
| `cross_sectional.py` | `compute_cross_sectional_features()` — sector/market-relative features; needs the full universe at once, unlike everything in `features.py` |
| `clean.py` | `clean_dataset()` — final dedupe/inf-to-NaN pass |
| `manifest.py` | `write_manifest()`, `compute_split_dates()`, `write_split_config()`, `sync_dataset_version()` |
| `cagr_handler.py` | CAGR calc/fill (BolsAI first, backfill from earnings/revenue) |
| `scale_features.py` | Fits `ColumnTransformer` (RobustScaler on ratio columns, passthrough elsewhere) train-only, per `split_config.json`; saves `feature_scaler.joblib` + `scaler_metadata.json` |


## Critical Caveats

- **CAGR backfill is ON:** `fill_missing_cagr()` (which calls `fill_cagr_columns()` per ticker) runs unconditionally in `build_ml_dataset.py`'s main pipeline → dataset has `cagr_{earnings,revenue}_5y_final` populated. Coverage is ~60% from BolsAI; the backfill recovers an additional ~7%.
- **No lookahead (Stage 2) — ENFORCED:** `merge_asof(..., direction='backward')` on real CVM `fundamentals_available_date` (not fiscal period-end) — a price never sees a future fundamental. `volatility_*_percentile` use rolling-window rank, not global rank. Tests: `test_merge_honors_actual_filing_date`, `test_volatility_percentile_no_lookahead`. ✅ VERIFIED 2026-07-11.
- **Real filing dates (July 2026):** Fundamentals visible via CVM's `DT_RECEB` (received date), not fiscal `reference_date`. 41,530 filings from 1,223 companies, 100% coverage of 293-ticker universe; 4,657 rows (0.7%) would have violated a fixed 45/90-day buffer. Sourced from free, keyless CVM open-data portal; integrated via `src/data_collection/cvm/filing_dates.py` (`python -m src.data_collection.cvm_statements --step filing_dates`).
- **Unadjusted splits REPAIRED:** 53 corporate events in BolsAI's `adj_*` columns were never back-adjusted, causing fake returns up to −99.99%. `repair_unadjusted_splits()` detects and rescales all pre-event rows. `hl_ratio` uses `adj_high/adj_low` (not raw scales). WDCN3 quarantined (unfixable data corruption). ✅ VERIFIED 2026-07-11.
- **Valuation ratios re-anchored daily:** BolsAI computes `pl/pvp/market_cap/p_*/ev_*` at filing date; `recompute_valuation_daily()` rescales to current close (keeps `fundamentals_available_date` in output for agent state). Known ceiling: mid-quarter splits skew ratios until next filing (build warns).
- **All feature engineering is in Stage 2**, not deferred to the agent (technicals, fundamental ratios, macro-adjusted, CAGR backfill, split repair, volatility rolling rank).
- **FIIs deferred:** stocks only (prices/fundamentals/dividends). FIIs are a separate asset class; add if agent scope expands to mixed-asset.
- **BolsAI:** key in `.env`, loaded by `config.load_env()` (stdlib parser). Backfill only — paid ~€0.10/1K calls. Caps: prices `limit<=5000` (date-window paginated), fundamentals `limit<=88` (use 80).
- **yfinance:** free incremental refresh. Prices/dividends full history to 2000; fundamentals only ~4–6 quarters (enough for quarterly refresh).
- **BCB series:** selic=11 (daily), cdi=12, ipca=433 — **NOT 432** (that's the annual meta target).
- **Benchmark:** BOVA11 (IBOV proxy ETF) collected automatically; prices only.
- **Company info:** BolsAI-only (CVM metadata, rarely changes); refresh via `--mode full_scale` when new IPOs appear. Current dataset: 523 tickers total (373 ATIVO active + 85 CANCELADA delisted + 65 missing status); 3 quarantined (WDCN3 unadjusted splits unfixable, CAMB4 delisted 2019, LLIS3 delisted 2023).
- **Data quality filters (Stage 2, enforced automatically):**
  - Filing lag filter: Drop fundamentals filed >180 days after quarter-end (0.9% of rows) — prevents lookahead from unreliable late filings
  - Close-price lookup: Replace BolsAI's stale close_price with actual close from `fundamentals_available_date` — prevents false >50% valuation jumps
  - Valuation re-anchoring: Rescale P/E, P/B, etc. to current close daily (not filing-date close) — keeps ratios current
  - Split repair: Detect and rescale unrecorded splits (53 events) — prevents fake negative returns up to −99.99%
  - Sibling fill: Forward-fill missing company_info from same-CVM-company tickers (168,783 rows) — ensures all rows have sector/status metadata
  - Quarantine list: WDCN3 (raw close oscillates 6x, no repair), CAMB4 (delisted 2019, stale fundamentals), LLIS3 (delisted 2023, stale fundamentals) — eliminates data quality outliers.
- **NaN & extreme-value policy (implemented):**
  - Data quality filters (Stage 2):
    - Structural NaN (warm-up, pre-first-filing) trimmed by global start-date rule per universe.
    - Informative NaN (CAGR undefined from negative earnings/insufficient history) flagged: `cagr_earnings_defined`, `cagr_revenue_defined` (0/1); also tracks `n_quarters_available` (cumulative filing count) explaining all window-based NaNs.
    - Error NaN: prefix-shaped (no interior holes per ticker in merged fundamentals), detected via test `test_final_dataset.py::T_prefix_rule`. NaN-count regression vs previous build warned via `nan_regressions()` in `manifest.py` (logged but non-fatal, allows legitimate coverage changes).
    - Extreme ratio (144 rows |pl| > 400,000 dataset-wide, 95 in-universe, top-50): kept intact — denominators near zero are valid distress signals. No filled or clipped in the dataset; scaler's fit is robust (median/IQR) but transform is linear, so raw 400k → ~26k after scaling (still extreme, intentionally preserved). Model training handles via loss functions / clipping in the env.
  - Consumer-side (ml_agent env, not this repo): flags + neutral fills (e.g. fill CAGR NaN with 0), any NaN→0 transformation, hard `assert np.isfinite(obs).all()` before agent sees state, global start-date trim for the top-50 universe to drop pre-full-history rows.
- **Checkpoints/logs** (not git-tracked): Stage 1 `artifacts/checkpoints/{mode}/`, `artifacts/logs/collection/collection-*.log`.
- **Paths:** absolute via `Path(__file__).resolve().parents[N]`; always run from project root.
- **FutureWarnings suppressed:** `pct_change(fill_method=None)` for YoY growth; dropped all-NA columns per-file before concat.

## Data on Disk

- **Raw (git-tracked):** full-scale universe, ~293 tickers + benchmark BOVA11, one parquet per ticker in `data/raw/{prices,fundamentals,dividends}/`. Prices/dividends current to 2026-06-30; fundamentals to 2026-03-31. Coverage isn't 100% uniform across types (e.g. a handful of tickers are missing a dividends file) — treat gaps as "not yet collected," not "confirmed zero," and re-run the relevant `collect_*` for that ticker to check. In the processed dataset this ambiguity is exposed directly: `has_dividends` (0/1, set in `merge_dividends()`) marks whether a ticker was ever collected at all, so `div_yield_12m == 0` can be told apart from "never collected" rather than silently reading as a confirmed zero.
- `data/raw/macro/{selic,cdi,ipca}.parquet` (one file per series) and `data/raw/company_info/company_info.parquet` (per-ticker static attributes: sector, cnpj, status, etc.) are market-wide reference tables, not per-ticker files.
- `data/raw/company_info/sectors.parquet` is a small aggregate `[sector name, ticker count]` table used to sanity-check how many companies fall in each sector — not a join key, not consumed by `build_ml_dataset.py`.
- `data/raw/corporate_events/corporate_events.parquet` is a market-wide split/inplit audit log; `company_info/sectors.parquet` and `corporate_events.parquet` are collected during `full_scale`/`prototype` runs only — skipped in `--mode update` so the free/keyless yfinance refresh path never needs a BolsAI key.
- **Processed:** `data/processed/ml_dataset.parquet` (created on first build), one row per ticker+date, plus `ml_dataset.manifest.json` (reproducibility snapshot) and `split_config.json` (walk-forward train/val/test date cutoffs — a filter, not materialized copies). Each build that actually changes output is also snapshotted immutably to `data/processed/dataset_v{N}/` (unchanged reruns don't bump `N`); cite `dataset_v{N}` when referencing exactly which build an experiment used. `data/processed/scalers/feature_scaler.joblib` + `scaler_metadata.json` are fit train-only (per `split_config.json`) via a separate, deliberate step (`scale_features.py`), not on every build. All of the above are gitignored and fully regenerable from `data/raw/`. `data/processed/README.md` is the one tracked exception (see its content for the full list) — it documents this ownership boundary; treat anything else that shows up in `data/processed/` (e.g. `ml_dataset_training.parquet`) as generated by pipelines outside this repo's `src/` (the `ml_agent` branch) and not something this repo can rebuild.

## Technology Stack

- **Python 3.10+** (`list[str]`, `dict | None`).
- **Data:** pandas, numpy, pyarrow.
- **APIs:** BolsAI REST (`httpx`, backfill), BCB SGS (requests, macro), `yfinance` (incremental).
- **Config:** stdlib `.env` parser (BolsAI key only).
- **Preprocessing:** scikit-learn (`ColumnTransformer`/`RobustScaler` in `scale_features.py`), `joblib` (scaler serialization) — both were already installed for Stage 3 (agent), now also a Stage 2 direct dependency.
- **Viz:** Plotly.
- **No test framework:** standalone `python script.py`.
