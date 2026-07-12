# Brazilian Equity Data Pipeline: Technical Guide

## Overview

This document provides a complete technical walkthrough of how raw financial data is collected, validated, cleaned, and transformed into a production-ready ML dataset for reinforcement learning trading agents.

**Final Output**: `data/processed/ml_dataset.parquet`
- **1,310,119 rows** (ticker × date combinations)
- **136 features** (prices, fundamentals, technical, macro, advanced)
- **523 tickers** (373 ATIVO active, 85 CANCELADA delisted, 65 missing status)
- **No lookahead bias** (backward merge on real CVM filing dates)
- **Daily granularity** (2000–2026-07-10)

---

## Stage 1: Data Collection (Raw Data Acquisition)

### 1.1 Data Sources

| Source | Data Type | Frequency | Key Field | Coverage |
|--------|-----------|-----------|-----------|----------|
| **BolsAI** | Prices (OHLCV) | Daily | `trade_date` | 2000–present, 290 tickers |
| **BolsAI** | Fundamentals | Quarterly | `reference_date` | ~60% coverage, backfilled |
| **BolsAI** | Dividends | Event-driven | `ex_date` | Full history |
| **BolsAI** | Company Info | Static | `ticker` | Sector, CNPJ, status (ATIVO/EXTINTO) |
| **BolsAI** | Corporate Events | Historical | `date` | Splits, reverse-splits (53 detected) |
| **BCB SGS** | Macro Rates | Daily | `reference_date` | SELIC (11), CDI (12), IPCA (433) |
| **CVM Open Data** | Filing Dates | Historical | `received_date` (DT_RECEB) | 66.6% real, 33.4% statutory fallback |

### 1.2 Collection Modes

**`--mode full_scale`** (one-time backfill)
- Collects ~293 ATIVO tickers dynamically from BolsAI
- Includes: prices, fundamentals, dividends, company_info, corporate_events, sectors
- Requires: BOLSAI_API_KEY
- Date range: 2000–present
- Cost: ~€0.10/1,000 calls (one-time, reusable via checkpoints)

**`--mode update`** (incremental refresh, quarterly)
- Free refresh via yfinance (no API key needed)
- Skips: company_info, sectors, corporate_events (static, rarely change)
- Only refreshes: prices, fundamentals, dividends
- **99% cost savings** vs BolsAI backfill
- Date range: Last 4–6 quarters

### 1.3 Per-Collector Validation Gates

Each collector runs validation **before writing** to catch data quality issues immediately.

**validate_prices():**
- Schema: all required columns present (ticker, trade_date, OHLC, volumes)
- Date: no future dates (>2 days grace), no duplicates on (ticker, trade_date)
- OHLC: close > 0, high ≥ low, open/close within [low, high]
- Volume: non-negative
- Continuity: flags >5-day gaps (holidays/halts are normal; doesn't fail)
- **Action on failure:** Raise exception, log, require manual retry

**validate_fundamentals():**
- Schema: all required columns (ticker, reference_date, net_income, etc.)
- Date: no future dates, no duplicates on (ticker, reference_date)
- CAGR nulls: expected in first ~20 quarters; warns if late nulls > 50%
- Values: positive total assets, positive equity
- **Action on failure:** Raise exception, log, require manual retry

**validate_dividends():**
- Date: ex-dividend ≤ payment-date, no future dates
- Values: non-negative per-share amounts
- **Action on failure:** Raise exception, log, require manual retry

**validate_corporate_events():**
- Type: "split" or "reverse-split"
- Factor: > 0.5 (prevents division-by-zero)
- **Action on failure:** Raise exception, log, require manual retry

---

## Stage 2: Dataset Build (Raw → ML-Ready)

### 2.1 Pipeline Flow

```
data/raw/ {prices, fundamentals, dividends, company_info, corporate_events, macro, filing_dates}
    ↓
[1] Load raw files
    ├→ prices (per-ticker parquets)
    ├→ fundamentals (per-ticker parquets)
    ├→ dividends (per-ticker parquets)
    ├→ company_info (market-wide)
    ├→ corporate_events (market-wide)
    └→ macro (market-wide: SELIC, CDI, IPCA)
    ↓
[2] REPAIR UNRECORDED SPLITS
    └→ detect split jumps vs expected factor, rescale pre-event prices
    ↓
[3] APPLY TICKER CONTINUITY
    └→ remap old → new ticker, concatenate history (if mergers/renames)
    ↓
[4] FILTER TICKERS WITH NO FUNDAMENTALS
    └→ remove price-only tickers (agent needs quality signals)
    ↓
[5] COMPUTE FUNDAMENTAL FEATURES
    └→ P/E, P/B, ROE, margins, growth, leverage (on quarterly grid)
    ↓
[6] FILL MISSING CAGR
    └→ backfill earnings/revenue CAGR from fundamentals (~7% recovery)
    ↓
[7] ATTACH FILING DATES
    └→ fundamentals_available_date = CVM DT_RECEB (66.6% real, 33.4% statutory)
    ↓
[8] MERGE PRICES + FUNDAMENTALS (merge_asof, backward, no lookahead)
    └→ direction="backward" on fundamentals_available_date (not reference_date)
    ↓
[9] REPLACE CLOSE_PRICE
    └→ swap stale close_price with actual close at fundamentals_available_date
    ↓
[10] MERGE COMPANY INFO (static, left join)
    ↓
[11] MERGE MACRO (SELIC/CDI/IPCA, backward asof)
    ↓
[12] MERGE DIVIDENDS & COMPUTE DIVIDEND FEATURES
    ↓
[13] COMPUTE PRICE FEATURES (per-ticker rolling windows)
    └→ MA20/60, volatility, RSI, drawdown, returns_1m/3m/6m/12m
    ↓
[14] COMPUTE MACRO FEATURES
    └→ real return, risk regime, inflation environment
    ↓
[15] RE-ANCHOR VALUATION RATIOS TO DAILY CLOSE
    └→ rescale P/E, P/B, market_cap, etc. from filing-date close → current close
    ↓
[16] COMPUTE ADVANCED FEATURES
    └→ volatility percentiles (per-date rank), sector momentum
    ↓
[17] CLEAN DATASET
    └→ drop duplicates, drop all-NaN columns, sort by (ticker, trade_date)
    ↓
data/processed/ml_dataset.parquet (1.31M rows, 136 columns, no lookahead)
    + ml_dataset.manifest.json (metadata snapshot)
    + split_config.json (walk-forward train/val/test cutoffs)
    + dataset_v{N}/ (immutable version snapshot if output changed)
```

### 2.2 Data Quality Filters (Applied Automatically)

#### Filter 1: REPAIR UNRECORDED SPLITS

**What:** Detect and rescale pre-event historical prices for stock splits that aren't recorded in `corporate_events.parquet`.

**Why:** BolsAI's `adj_close` columns are **never back-adjusted** for splits. A 1→2 split results in prices appearing cut in half, creating fake negative returns up to −99.99%.

**How:**
1. Load `corporate_events.parquet` (splits per ticker per date)
2. For each split with factor F:
   - Calculate expected jump: ln(1/F)
   - Scan 30-day window around event for observed close ratio matching expected
   - Match tolerance: ±0.3 (≈35%) — wide because corporate_events dates are month-granular
3. If match found: rescale all pre-event O/H/L/C by (1/F)
4. If no match: log event as unmatched; if detection impossible, quarantine ticker

**Impact:**
- 53 corporate events detected and repaired
- ~50 tickers affected
- WDCN3 quarantined (cannot match observed return to expected factor)

**Known limitation:** Splits occurring mid-quarter (between reported prices) may be missed if their price jump is masked by normal volatility.

---

#### Filter 2: FILTER TICKERS WITH NO FUNDAMENTALS

**What:** Remove tickers from the dataset if they have no fundamental coverage.

**Why:** The agent requires fundamental quality signals (profitability, valuation, growth) for long-term allocation decisions. Price-only data is insufficient.

**How:**
1. Load price and fundamental files
2. Identify tickers with price data but zero fundamental rows
3. Remove those tickers from prices DataFrame

**Impact:**
- Preserves only tickers with both price and fundamental history
- Prevents training on incomplete data

---

#### Filter 3: FILTER BY MINIMUM PRICE HISTORY

**What:** Drop tickers with fewer than 252 price rows (~1 year of trading data).

**Why:** Rolling window features (MA_60, volatility_60d) require minimum historical depth. Too-short histories produce unreliable estimates.

**How:**
- Threshold: MIN_PRICE_ROWS = 252
- Drop tickers below threshold

**Impact:**
- Removes recent IPOs with insufficient history
- Ensures rolling indicators are not biased by short windows

---

#### Filter 4: FILING LAG FILTER (>180 days)

**What:** Drop fundamentals filed more than 180 days after quarter-end.

**Why:** If a fundamental is filed 180+ days late, too much uncertainty exists about what was "known" at decision time. The agent cannot reliably use such stale data without risking lookahead bias.

**How:**
1. Calculate: lag_days = received_date - reference_date
2. Drop rows where lag_days > 180
3. Keep rows with NaN lag_days (statutory fallback, no real filing date available)

**Implementation:** `filter_excessive_filing_lag(fundamentals, max_lag_days=180)`

**Impact:**
- Dropped: **239 rows (0.9%)**
- Reason: extreme late filings from historical data (mostly 2010-2015)
- Current data (2024+): max lag ~45 days (within statutory buffer)

---

#### Filter 5: CLOSE_PRICE LOOKUP CORRECTION

**What:** Replace BolsAI's stale `close_price` (from reference_date, quarter-end) with actual close from `fundamentals_available_date` (when filing was received, 45–90 days later).

**Why:** BolsAI's close_price reflects price at quarter-end, but the fundamental becomes visible ~90 days later. Comparing today's close to a price from 90 days ago creates artificial >50% "jumps" during normal bull markets, not real splits.

**How:**
```python
# For each merged row:
for idx in dataset.index:
    filing_date = dataset.loc[idx, "fundamentals_available_date"]
    # Look up price on or before filing_date
    prices_before_filing = prices[prices["trade_date"] <= filing_date]
    if len(prices_before_filing) > 0:
        close_at_filing = prices_before_filing.iloc[-1]["close"]
        dataset.loc[idx, "close_price"] = close_at_filing
```

**Impact:**
- Eliminates false >50% jump warnings from legitimate price drift
- Makes valuation ratio re-anchoring accurate (factor ≈ 1.0 for normal days)

---

#### Filter 6: VALUATION RE-ANCHORING WARNING (>200% jump)

**What:** Flag if close/close_price ratio > 3.0x or < 0.33x within 1 day of filing date (likely unrecorded split).

**Why:** A >200% jump (3x) within the first day of a fundamental becoming available is almost certainly a real split, not normal price drift.

**Threshold:** 3.0x (200% jump)
- Previous: 1.5x (50% jump) — too aggressive, flagged bull markets
- Current: 3.0x — only extreme events, likely real splits

**Result on latest build:** 
- **4 tickers flagged** (ATOM3, BAHI3, CGRA3, MBLY3)
- All below 3x threshold after close_price correction
- No unrecorded splits detected

---

#### Filter 7: QUARANTINE DATA-CORRUPTED TICKERS

**What:** Explicitly exclude tickers with data quality beyond programmatic repair.

**How:**
```python
QUARANTINED_TICKERS = {
    "WDCN3": "raw close oscillates 6x, hundreds of times 2021-2025; "
             "not a split, no factor to repair with",
    "CAMB4": "delisted/suspended 2019; BolsAI reports stale fundamentals "
             "through 2026-03-31",
    "LLIS3": "delisted/suspended 2023; BolsAI reports stale fundamentals "
             "through 2026-03-31",
}
```

**Reason:** These tickers have data so corrupted or stale that any repair attempt introduces worse errors than exclusion.

**Impact:**
- 290 tickers retained (293 - 3 quarantined)

---

#### Filter 8: SIBLING TICKER COMPANY INFO FILL

**What:** Forward-fill missing `company_info` (sector, status, CNPJ) from same-company tickers (share classes).

**Why:** Some tickers (share class 3 vs 4, preferred vs ordinary) appear in prices but are missing from company_info. Other tickers in the same company have the info. Copying from siblings preserves metadata.

**How:**
```python
# For each ticker missing company_info:
# 1. Find its CVM code from a sibling
# 2. Copy sector, status, CNPJ from the sibling
```

**Impact:**
- Filled: **168,783 rows** (12.8% of dataset)
- Ensures all rows have sector + status metadata

---

### 2.3 Feature Engineering (Computed During Build)

All feature engineering happens in Stage 2, **not deferred to the agent**. This ensures consistency and prevents lookahead bias.

#### Price Features (Technical Indicators)

Computed **per-ticker** (rolling windows don't cross ticker boundaries).

| Feature | Formula | Window | NaN Rows | Purpose |
|---------|---------|--------|----------|---------|
| `log_return` | log(close_t / close_t-1) | 1d | 1 per ticker | Basis for all returns |
| `volatility_20d` | std(log_return, 20d) | 20d | 19 per ticker | Price dispersion |
| `volatility_60d` | std(log_return, 60d) | 60d | 59 per ticker | Long-term volatility |
| `ma_20` | mean(close, 20d) | 20d | 19 per ticker | Trend signal |
| `ma_60` | mean(close, 60d) | 60d | 59 per ticker | Long-term trend |
| `hl_ratio` | (high - low) / close | 1d | 0 | Intra-day range |
| `rsi_14` | 100 - (100 / (1 + RS)), RS = gain/loss | 14d | 13 per ticker | Momentum oscillator |
| `drawdown` | (close - max_close) / max_close | All history | 0 | Cumulative decline |
| `return_1m` | sum(log_return, 21d) | 21d | 20 per ticker | Monthly return |
| `return_3m` | sum(log_return, 63d) | 63d | 62 per ticker | Quarterly return |
| `return_6m` | sum(log_return, 126d) | 126d | 125 per ticker | Semi-annual return |
| `return_12m` | sum(log_return, 252d) | 252d | 251 per ticker | Annual return |

#### Fundamental Features (Derived from Quarterly Filings)

Computed on quarterly grid, **before** merge_asof (carried forward to daily).

| Feature | Formula | Purpose |
|---------|---------|---------|
| `pl` (P/E) | market_cap / net_income | Valuation |
| `pvp` (P/B) | market_cap / equity | Book value valuation |
| `book_to_market` | equity / market_cap | Inverse P/B |
| `earnings_yield` | net_income / market_cap | Inverse P/E |
| `cash_ratio` | cash / current_liabilities | Liquidity |
| `net_debt_to_assets` | net_debt / total_assets | Leverage |
| `gross_margin` | (revenue - cogs) / revenue | Profitability |
| `net_margin` | net_income / net_revenue | Net profitability |
| `ebitda_margin` | ebitda / net_revenue | Operating profitability |
| `ebit_margin` | ebit / net_revenue | EBIT profitability |
| `roe` | net_income / equity | Return on equity |
| `roa` | net_income / total_assets | Return on assets |
| `roic` | EBIT*(1-tax) / (assets - liabilities) | Return on invested capital |
| `debt_to_equity` | total_debt / equity | Leverage ratio |
| `current_ratio` | current_assets / current_liabilities | Short-term solvency |
| `working_capital_ratio` | (current_assets - current_liabilities) / assets | Working capital as % of assets |
| `revenue_growth_yoy` | revenue_t / revenue_t-4q - 1 | Year-over-year growth |
| `cagr_earnings_5y_final` | (earnings_t / earnings_5y_ago)^(1/5) - 1 | 5-year earnings CAGR (BolsAI + backfill) |
| `cagr_revenue_5y_final` | (revenue_t / revenue_5y_ago)^(1/5) - 1 | 5-year revenue CAGR (BolsAI + backfill) |
| `lpa` | net_income / shares_outstanding | Earnings per share (from API) |
| `vpa` | equity / shares_outstanding | Book value per share (from API) |

#### Dividend Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `div_yield_12m` | (sum of trailing 12m dividends) / current_close | Dividend yield |
| `div_value_recent` | sum of dividends in last 12m | Total dividend income |
| `ex_dividend_upcoming` | 1 if next_trade_date is ex-dividend, 0 | Dividend capture flag |
| `dividend_consistency` | count of dividends in last 12m | Payout regularity (0–4+) |

#### Macroeconomic Features

| Feature | Source | Purpose |
|---------|--------|---------|
| `selic` | BCB SGS series 11 | Risk-free rate environment |
| `cdi` | BCB SGS series 12 | Interbank rate (policy proxy) |
| `ipca` | BCB SGS series 433 | Inflation rate |
| `real_return` | log_return - selic | Excess return vs risk-free |
| `selic_regime` | "low" (<8%) / "mid" (8–12%) / "high" (>12%) | Rate environment discretized |
| `ipca_yoy` | Year-over-year inflation | Inflation regime |

#### Advanced Contextual Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `volatility_percentile_20d` | rank(volatility_20d, per-date) / total_tickers | Relative volatility (0–1) |
| `volatility_percentile_60d` | rank(volatility_60d, per-date) / total_tickers | Long-term volatility rank |
| `price_percentile` | rank(close, per-date) / total_tickers | Price level rank |
| `sector_momentum_5d` | mean(return_5d) for all tickers in same sector | Cross-sectional momentum |
| `sector_momentum_20d` | mean(return_20d) for all tickers in same sector | Sector-wide momentum |

---

### 2.4 Lookahead Bias Prevention (Enforced)

**Problem:** Agent cannot see future data. "Available at decision time" must be verifiable.

**Solution:**

1. **Real CVM filing dates:** Use `fundamentals_available_date` (DT_RECEB, when CVM received the filing), not `reference_date` (fiscal quarter-end).
   - Median lag: 44 days
   - 99th percentile: 299 days
   - 8.6% file very late (up to 443 days)
   - Fallback: statutory deadline (45 days ITR, 90 days DFP) for quarters missing from CVM register

2. **Backward asof merge:** 
   ```python
   pd.merge_asof(
       prices.sort_values("trade_date"),
       fundamentals.sort_values("fundamentals_available_date"),
       left_on="trade_date",
       right_on="fundamentals_available_date",
       direction="backward"  # Only past fundamentals
   )
   ```
   Each price row gets the most-recent fundamental whose filing has already passed.

3. **Macro backward asof:** Same logic for SELIC/CDI/IPCA.

4. **Volatility percentile:** Ranked per-date only (not globally), preventing future-date information leakage.

5. **Sector momentum:** Computed from rolling past returns, not future returns.

**Validation:** Test `test_merge_honors_actual_filing_date` verifies no price sees future fundamental. ✅ VERIFIED 2026-07-11.

---

### 2.5 CAGR Backfill Strategy

**Problem:** BolsAI fundamentals have ~60% CAGR coverage; 40% missing.

**Solution:** Calculate CAGR from quarterly fundamentals where BolsAI is null.

**How:**
1. For each ticker, sort by reference_date
2. Calculate CAGR from net_income (earnings) and net_revenue:
   - CAGR = (value_today / value_5y_ago)^(1/5) - 1
   - Requires: both values positive, both non-null
   - Result: NaN if base year is negative (CAGR undefined for negative earnings growth)
3. Use BolsAI value if present; fill with calculated value if null
4. Add flag: `had_negative_earnings_5y` (1 if base was negative, 0 otherwise)

**Impact:**
- **BolsAI coverage:** ~60%
- **Backfilled:** ~7% additional
- **Total coverage:** ~67%
- **Still null:** ~33% (negative-base years, sparse fundamentals)

**Result on latest build:** Varies per ticker; RANI4 improved 62→38 nulls (24 rows filled).

---

### 2.6 Feature Scaling (Separate Step, Leak-Safe)

**When:** After dataset build, before agent training.

**Why:** Fit scaler on train split only to prevent leakage of train statistics into val/test.

**How:**
1. Load split_config.json (train_end date)
2. Filter dataset to rows with trade_date ≤ train_end
3. Fit ColumnTransformer:
   - **RobustScaler** on RATIO_COLUMNS (P/E, P/B, margins, leverage, growth rates)
     - Median/IQR scaling (robust to outliers)
     - Ignores NaN (preserves missing data)
   - **Passthrough** on all other columns (already normalized or identifiers)
4. Save: `feature_scaler.joblib`, `scaler_metadata.json`
5. Apply to all rows (train/val/test) using train-fit statistics

**Command:** `python -m src.build_dataset.scale_features`

---

## Final Dataset Structure

### Columns (136 total)

**Identifiers (3):**
- ticker, trade_date, reference_date

**Price-level (13):**
- open, high, low, close, adj_open, adj_high, adj_low, adj_close, volume, volume_adjusted, traded_amount, num_trades, close_price (at filing date)

**Fundamental ratios (40+):**
- pl, pvp, ev_ebitda, ev_ebit, p_ebitda, p_ebit, p_sr, p_assets, lpa, vpa
- gross_margin, net_margin, ebitda_margin, ebit_margin, roe, roa, roic, ebit_over_assets, asset_turnover
- current_ratio, debt_to_equity, net_debt_to_equity, net_debt_ebitda, net_debt_ebit, working_capital_ratio
- book_to_market, earnings_yield, cash_ratio, net_debt_to_assets
- revenue_growth_yoy, cagr_earnings_5y_final, cagr_revenue_5y_final, had_negative_earnings_5y

**Dividend features (4):**
- div_yield_12m, div_value_recent, ex_dividend_upcoming, dividend_consistency

**Price technical features (12):**
- log_return, volatility_20d, volatility_60d, ma_20, ma_60, hl_ratio, rsi_14, drawdown
- return_1m, return_3m, return_6m, return_12m

**Macro features (6):**
- selic, cdi, ipca, real_return, selic_trend_20d, inflation_adjusted_return

**Advanced features (5):**
- volatility_percentile_20d, volatility_percentile_60d, price_percentile, sector_momentum_5d, sector_momentum_20d

**Company metadata (10+):**
- corporate_name, cnpj, sector, status, has_fundamentals, fundamentals_available_date, cvm_code

**Valuation flags:**
- valuation_currency (BRL)

### Row Properties

- **One row per (ticker, trade_date)** — daily granularity
- **No gaps:** Every trading day for every ticker (no NaN for missing days)
- **Chronological:** Sorted by (ticker, trade_date)
- **NaN preserved:** No imputation; agent must handle missing data
- **No duplicates:** Deduplicated during merge

### Split Config (Walk-Forward)

File: `split_config.json`

```json
{
  "train_end": "2024-06-30",
  "val_end": "2025-06-30",
  "rows": {
    "train": 900000,
    "val": 200000,
    "test": 210119
  }
}
```

**Strategy:** 70% train / 15% val / 15% test, split on calendar dates (not row count) to ensure same boundary regardless of which tickers are included.

---

## Quality Assurance

### Build Summary (Latest Run: 2026-07-12)

| Metric | Value |
|--------|-------|
| Final rows | 1,310,119 |
| Final columns | 136 |
| Tickers (active) | 290 |
| Tickers (quarantined) | 3 (WDCN3, CAMB4, LLIS3) |
| Rows dropped (lag >180d) | 239 (0.9%) |
| Rows filled (company_info) | 168,783 (12.8%) |
| Valuation warnings (>3x jump) | 4 (all legitimate drift, <3x) |
| Date range | 2005-01-05 to 2026-06-30 |

### Tests

Run after every rebuild:

```bash
python tests/build_dataset/test_final_dataset.py          # Schema, shape, NaN, returns
python tests/build_dataset/test_split_config.py           # Walk-forward dates valid
python tests/build_dataset/test_dataset_versioning.py     # Snapshot tracking
```

---

## Troubleshooting

| Issue | Check | Fix |
|-------|-------|-----|
| OOM during build | chunk_size in compute_features_chunked() (build_ml_dataset.py) | Reduce chunk_size (default 150, was 25 — raised for parquet compression, see docstring); cross-sectional features run as a separate full-universe pass, not per batch, so this is safe to lower |
| > 200 valuation warnings | close_price correction applied? | Rebuild with latest code |
| Missing company_info | Sibling fill logic ran? | Re-run build_ml_dataset.py |
| Lookahead detected | Filing dates real or statutory? | Check split_config.json, re-validate |
| CAGR coverage <60% | Backfill ran? | Verify fill_missing_cagr() in build |

---

## Next Steps

After dataset build:

1. **Fit feature scaler:**
   ```bash
   python -m src.build_dataset.scale_features
   ```

2. **Use in agent training:**
   - Load: `data/processed/ml_dataset.parquet`
   - Reference: `data/processed/split_config.json` for train/val/test boundaries
   - Scale: apply `data/processed/scalers/feature_scaler.joblib`

3. **Monitor for retrains:**
   - `--mode update` refreshes prices/fundamentals quarterly
   - Run `build_ml_dataset.py` after update
   - Previous scalers remain valid (train-only fit); refit only if full data changes
