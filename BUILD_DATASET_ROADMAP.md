# Build Dataset Roadmap (build_dataset branch)

Stage 2: Merging raw data into ML-ready parquets. No lookahead bias.

## Phase 1: Data Loading & Inspection

### 1a. Load Raw Data Files
- [ ] Load tickers from `data/raw/prices/` (list all `{TICKER}.parquet`)
- [ ] Load price data: OHLCV, date index, schema validation
- [ ] Load fundamentals: quarterly data, date index (quarter-end)
- [ ] Load macro: SELIC, CDI, IPCA daily series
- [ ] Load company info: ticker → sector mapping, if available

### 1b. Inspect Data Quality
- [ ] Check for gaps in price data (should be daily, excluding weekends/holidays)
- [ ] Check for gaps in fundamentals (should be ~quarterly, 4 per year)
- [ ] Check macro series continuity (daily, no gaps)
- [ ] Identify date ranges: earliest price date, latest macro date
- [ ] Count rows per ticker, identify coverage (full timeline vs partial)
- [ ] Check for NaN/null frequencies per column

### 1c. Create Data Utilities (`src/data_loading.py`)
```python
def load_all_prices(data_dir) -> dict[str, pd.DataFrame]
def load_all_fundamentals(data_dir) -> dict[str, pd.DataFrame]
def load_macro_series(data_dir) -> dict[str, pd.Series]  # selic, cdi, ipca
def load_company_info(data_dir) -> pd.DataFrame
def inspect_dataset(prices, fundamentals, macro) -> dict  # gaps, coverage, dtypes
```

## Phase 2: Temporal Merge (No Lookahead)

### 2a. Merge Strategy
For each ticker, align price & fundamentals without lookahead bias:
- **Price data:** Daily, every trading day
- **Fundamentals:** Quarterly (sparse), aligned backward (no future data leaks)
- **Method:** `pd.merge_asof(prices, fundamentals, on='date', direction='backward')`
  - Price date → closest past fundamental date (no future info)
  - Forward-fill quarters until next release (e.g., Q1 2026-03-31 valid from 2026-04-01 onward)

### 2b. Macro Alignment
- Merge macro series (SELIC, CDI, IPCA) as-is (daily, forward-fill to match price dates)
- Use `merge_asof` with `direction='backward'` for consistency

### 2c. Merge Implementation (`src/build_dataset.py`)
```python
def merge_ticker_asof(ticker_prices, ticker_fundamentals, macro_series, company_info):
  """
  Merge one ticker's data without lookahead.
  Returns: one row per (ticker, date) with all features filled.
  """
  # Step 1: Merge prices + fundamentals
  merged = pd.merge_asof(
    ticker_prices.sort_index(),
    ticker_fundamentals.sort_index(),
    left_index=True, right_index=True,
    direction='backward'
  )
  
  # Step 2: Merge macro series
  for series_name, series in macro_series.items():
    merged[series_name] = pd.merge_asof(
      merged.reset_index()[['date']], series.reset_index(),
      on='date', direction='backward'
    ).set_index('date')[series.name]
  
  # Step 3: Add company info (static)
  merged['sector'] = company_info[ticker]['sector']
  
  return merged
```

- [ ] Implement merge logic for one ticker (test manually)
- [ ] Verify lookahead: no fundamentals from *future* dates
- [ ] Apply to all tickers in parallel (joblib or concurrent.futures)
- [ ] Concatenate all tickers into one DataFrame

## Phase 3: Feature Engineering

### 3a. Price-Derived Features (Technical Indicators)
For each ticker, compute from OHLCV:
- **Returns:** daily log return = `log(close_t / close_{t-1})`
- **Volatility (rolling 20d):** `std(returns_20d)`
- **Volatility (rolling 60d):** `std(returns_60d)`
- **Momentum indicators:**
  - RSI (14-period): relative strength index
  - Moving averages: MA20, MA60 (for trend identification)
  - MACD (optional): momentum divergence
- **Drawdown:** running maximum drawdown (from peak)
- **High-Low ratio:** `(high - low) / close` (intraday volatility proxy)

```python
def compute_price_features(ticker_df):
  ticker_df['return'] = np.log(ticker_df['close'] / ticker_df['close'].shift(1))
  ticker_df['volatility_20d'] = ticker_df['return'].rolling(20).std()
  ticker_df['volatility_60d'] = ticker_df['return'].rolling(60).std()
  ticker_df['rsi_14'] = compute_rsi(ticker_df['close'], 14)
  return ticker_df
```

- [ ] Implement price features
- [ ] Handle NaN from rolling windows (first 60 rows will be NaN)

### 3b. Fundamental-Derived Features
From fundamentals (quarterly, forward-filled):
- **Valuation:** P/E, P/B (price-to-earnings, price-to-book)
- **Profitability:** ROE, profit margin, ROIC
- **Leverage:** debt/equity, current ratio
- **Growth:** revenue growth YoY, earnings growth YoY
- **CAGR (5-year):** from `cagr_handler.py` (already implemented)
  - Uncomment line 143 in `build_ml_dataset.py` to enable backfill

```python
def compute_fundamental_features(ticker_df):
  # Assumes fundamentals already merged (forward-filled)
  ticker_df['pe_ratio'] = ticker_df['price'] / ticker_df['earnings_per_share']
  ticker_df['pb_ratio'] = ticker_df['price'] / ticker_df['book_value_per_share']
  ticker_df['roe'] = ticker_df['net_income'] / ticker_df['equity']
  ticker_df['debt_equity'] = ticker_df['total_debt'] / ticker_df['equity']
  return ticker_df
```

- [ ] Implement fundamental ratios (check which columns are available in BolsAI data)
- [ ] Handle division by zero (set to NaN, don't crash)

### 3c. Macro-Adjusted Features
From macro (SELIC, CDI, IPCA):
- **Real return:** return adjusted for inflation (IPCA)
- **Excess return:** return minus risk-free rate (SELIC)
- **Rate environment:** is SELIC rising/falling (lagged diffs)

```python
def compute_macro_features(ticker_df, macro_series):
  ticker_df['real_return'] = ticker_df['return'] - macro_series['ipca'].pct_change()
  ticker_df['excess_return'] = ticker_df['return'] - macro_series['selic']
  ticker_df['selic_lag_1d'] = macro_series['selic'].shift(1)
  return ticker_df
```

- [ ] Implement macro adjustments

## Phase 4: Data Cleaning & Validation

### 4a. Missing Data Handling
- **Strategy:** Drop rows with critical NaNs, forward-fill less critical ones
- **Critical columns** (drop if NaN): ticker, date, close, volume
- **Less critical** (forward-fill): fundamentals, macro (sparse updates)
- **High-NaN columns** (consider dropping): features with >30% NaN

```python
def clean_missing_data(df):
  # Drop rows with critical NaNs
  critical = ['ticker', 'date', 'close', 'volume']
  df = df.dropna(subset=critical)
  
  # Forward-fill others (within ticker)
  df = df.groupby('ticker').apply(lambda g: g.fillna(method='ffill')).reset_index(drop=True)
  
  # Drop columns >30% NaN
  nan_pct = df.isnull().sum() / len(df)
  cols_to_drop = nan_pct[nan_pct > 0.3].index
  df = df.drop(columns=cols_to_drop)
  
  return df
```

- [ ] Implement missing data strategy
- [ ] Log before/after row counts per ticker

### 4b. Outlier Detection
- **Price spikes:** daily return > 20% (may be splits, errors, or real)
- **Volume spikes:** volume > 10x rolling mean
- **Decision:** Flag (don't drop) for now; downstream can filter

```python
def flag_outliers(df):
  df['price_spike'] = df.groupby('ticker')['return'].apply(lambda x: np.abs(x) > 0.20)
  df['volume_spike'] = df.groupby('ticker')['volume'].apply(
    lambda x: x > (x.rolling(60).mean() * 10)
  )
  return df
```

- [ ] Implement flagging

### 4c. Deduplication
- Remove exact duplicates (same ticker, date, OHLCV)
- Sort by (ticker, date) ascending

```python
def deduplicate(df):
  df = df.drop_duplicates(subset=['ticker', 'date'])
  df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
  return df
```

- [ ] Apply deduplication, sort

### 4d. Data Validation
- [ ] Verify no future fundamentals leak (date check: fund_date <= price_date)
- [ ] Verify no duplicate (ticker, date) pairs
- [ ] Verify all tickers have ≥252 rows (1 year of trading days minimum)
- [ ] Verify no NaN in critical columns: ticker, date, close, volume
- [ ] Verify return distribution is reasonable (not all zeros, not all extreme)

## Phase 5: Output & Validation

### 5a. Save ML Dataset
```python
def save_ml_dataset(df, output_path):
  df.to_parquet(output_path, index=False, compression='snappy')
  print(f"Saved {len(df)} rows to {output_path}")
  
  # Log summary stats
  print(f"Tickers: {df['ticker'].nunique()}")
  print(f"Date range: {df['date'].min()} to {df['date'].max()}")
  print(f"Columns: {df.columns.tolist()}")
  print(f"Shape: {df.shape}")
  print(f"NaN count per column:\n{df.isnull().sum()}")
```

Output path: `data/processed/ml_dataset.parquet`

- [ ] Save parquet with snappy compression
- [ ] Log summary stats (tickers, date range, shape, NaN count)

### 5b. Create Test Script (`tests/processed_data/test_final_dataset.py`)
- [ ] Load ml_dataset.parquet
- [ ] Verify schema (all expected columns present)
- [ ] Verify shape (≥ X rows, ≥ N tickers)
- [ ] Verify no lookahead (fundamental dates ≤ price dates)
- [ ] Verify no NaN in critical columns
- [ ] Verify return distribution (mean, std, min, max)
- [ ] Print pass/fail with detailed report

```bash
python tests/processed_data/test_final_dataset.py
```

- [ ] Run test, fix any failures
- [ ] Accept test as "golden" for dataset quality

## File Structure (After Completion)
```
src/
├─ data_loading.py      # Utilities: load_prices(), load_fundamentals(), etc.
├─ 2. build_dataset/
│  └─ build_ml_dataset.py  # Main orchestration: load → merge → feature → clean → save
└─ ...

data/
├─ raw/
│  ├─ prices/{TICKER}.parquet
│  ├─ fundamentals/{TICKER}.parquet
│  └─ macro/{selic,cdi,ipca}.parquet
├─ processed/
│  └─ ml_dataset.parquet    (output, one row per ticker+date)
└─ ...

tests/
└─ processed_data/
   └─ test_final_dataset.py   (validation & golden test)
```

---

## Key Decisions

### Merge Direction: Backward (No Lookahead)
```
Fundamentals: |---Q1---|---Q2---|---Q3---|---Q4---|
               2026-03-31       2026-09-30       2026-12-31
Prices:        |daily|daily|daily|daily|...
               2026-01-02      2026-04-01
               
Result: Prices from 2026-01-02 → 2026-03-31 get Q1 2025 (from 2025-12-31)
        Prices from 2026-04-01 → 2026-09-30 get Q1 2026 (from 2026-03-31)
        (No price date ever sees a *future* fundamental)
```

### Rolling Window NaNs
First 60 rows (for volatility_60d) will be NaN. Decision:
- Option A: Drop first 60 rows per ticker (simplest, lose 60 days of training)
- Option B: Use alternative volatility (e.g., exponential smoothing, fills sooner)
- **Chosen:** Option A (simpler)

---

## Start Here
1. **Week 1:** Phase 1
   - Load raw data, inspect gaps
   - Write load utilities
   
2. **Week 2:** Phase 2–3
   - Implement merge_asof for temporal alignment
   - Add price & fundamental features
   
3. **Week 3:** Phase 4–5
   - Clean missing data, flag outliers
   - Save ml_dataset.parquet
   - Run validation test
   
4. **Ready for Stage 3:** ml_dataset.parquet exists, validated, no lookahead
