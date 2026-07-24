# Pipeline Forensic Audit — 2026-07-23

Read-only audit of the full pipeline (Stage 1 collection → Stage 2 build →
scaler). No code was modified. Findings verified empirically against
`data/processed/ml_dataset.parquet` (1,318,989 rows, 515 tickers) and raw
parquets where noted. Checkboxes = not yet fixed.

## 1. Data Flow Diagram

```mermaid
flowchart TD
    subgraph Stage1["Stage 1 — Collection (src/data_collection/)"]
        BCB["BCB SGS API<br/>selic=11 (daily %), cdi=12 (daily %), ipca=433 (monthly %)"]
        BOLSAI["BolsAI REST<br/>OHLCV + adj_*, fundamentals, dividends,<br/>company_info, corporate_events"]
        YF["yfinance<br/>--mode update refresh<br/>(prices: full yf-era refetch, fundamentals ~4-6q, dividends)"]
        CVM["CVM open data<br/>ITR/DFP registers → DT_RECEB filing dates"]
        BCB -->|collect_macro| RAWM["data/raw/macro/{selic,cdi,ipca}.parquet"]
        BOLSAI -->|collectors.py| RAWP["data/raw/prices/*.parquet"]
        BOLSAI --> RAWF["data/raw/fundamentals/*.parquet"]
        BOLSAI --> RAWD["data/raw/dividends/*.parquet"]
        BOLSAI --> RAWC["data/raw/company_info/*.parquet"]
        BOLSAI --> RAWE["data/raw/corporate_events/corporate_events.parquet"]
        YF -->|yf_collectors.py, _merge_save append+dedup| RAWP
        YF --> RAWF
        YF --> RAWD
        CVM -->|cvm/filing_dates.py| RAWFD["data/raw/filing_dates/filing_dates.parquet"]
    end

    subgraph Stage2["Stage 2 — Build (src/build_dataset/build_ml_dataset.py main())"]
        RAWP --> L1["load_prices()"]
        L1 --> OP["drop_orphan_prefix_rows()  (recycled-ticker garbage)"]
        OP --> SR["repair_unadjusted_splits()  (adj_* + volume rescale, event rekeying)"]
        SR --> TC["apply_ticker_continuity()  (rename/merger splice,<br/>adj-basis reconcile, volume/ratio scaling)"]
        RAWF --> TC
        TC --> NF["filter_tickers_with_no_fundamentals()  (+ QUARANTINED_TICKERS)"]
        TC --> FF["compute_fundamental_features()  (YoY, QoQ, F-score — quarterly rows)"]
        FF --> CG["fill_missing_cagr()  (December-anchored, cagr_handler.py)"]
        CG --> FD["attach_filing_dates()  (CVM DT_RECEB, statutory 45/90d fallback)"]
        RAWFD --> FD
        RAWC --> FD
        FD --> LAG["filter_excessive_filing_lag()  (>180d dropped)"]
        NF --> M1["merge_prices_and_fundamentals()<br/>merge_asof backward on fundamentals_available_date<br/>+ close_price → real close at filing"]
        LAG --> M1
        M1 --> M2["merge_company_info()  (static join + sibling fill +<br/>status inference from price recency)"]
        RAWC --> M2
        M2 --> M3["merge_macro()  (merge_asof backward on reference_date, ffill)"]
        RAWM --> M3
        M3 --> M4["merge_dividends()  (asof: last ex_date, has_dividends flag)"]
        RAWD --> M4
        M4 --> P1["PASS 1 (per ticker-batch):<br/>compute_price_features → compute_dividend_features →<br/>compute_macro_features → recompute_valuation_daily →<br/>compute_advanced_features → compute_history_relative_features"]
        RAWE --> SR
        P1 --> P2["PASS 2 (full universe, slim projection):<br/>compute_cross_sectional_features  (sector z-scores,<br/>momentum vs market/sector, beta_1y)"]
        P2 --> P3["PASS 3: join cross-sectional + clean_dataset()<br/>(dedupe, inf→NaN)"]
        P3 --> OUT["data/processed/ml_dataset.parquet"]
        OUT --> MAN["write_manifest() + write_split_config() + sync_dataset_version()"]
        OUT --> SC["scale_features.py  (RobustScaler on RATIO_COLUMNS,<br/>fit train-only per split_config FitWindow)"]
        SC --> SCOUT["data/processed/scalers/feature_scaler.joblib"]
    end
```

## 2. Audit Report

### Critical

- [ ] **Issue 1: `real_return`, `excess_return`, and `earnings_yield_vs_selic` use wrong macro units** (`src/build_dataset/features.py:309-322,546`)
  - **Severity:** Critical
  - **Why it matters for Quant Finance:** The comment in `compute_macro_features` says "selic/ipca are annual %; divide by 252," but the collected series are not annual: SGS 11/12 (selic/cdi) are **% per day** (values ~0.05) and SGS 433 (ipca) is **% per month** (values ~0.2–1.0). Verified in the built dataset: `real_return` subtracts a mean of 0.00185/day when the true daily inflation equivalent is ~0.00022/day — an 8.3x over-subtraction that biases `real_return` by roughly −34%/yr (its dataset mean is −0.165%/day, economically impossible). `excess_return` subtracts 0.00016/day vs the true ~0.00040/day (2.5x under-subtraction, ~+6pp/yr bias). `earnings_yield_vs_selic` subtracts `selic/100` = a *daily* decimal (~0.0005) from an *annualized* earnings yield, so the intended macro comparison (~0.14) is absent and the feature is just `earnings_yield` with noise. Three features are systematically wrong; any model consuming them learns distorted risk premia.
  - **Proposed Fix:** In `compute_macro_features`: `excess_return = log_return - np.log1p(selic/100)` (selic is already daily %; log1p for consistency with log returns). For inflation, convert monthly to daily: `real_return = log_return - np.log1p(ipca/100)/21`. In `compute_advanced_features`, annualize selic before comparing: `earnings_yield_vs_selic = earnings_yield - ((1 + selic/100)**252 - 1)`. Document the per-series units in `config.BCB_SERIES` so the next consumer doesn't repeat this.

### High

- [ ] **Issue 2: IPCA look-ahead — monthly inflation is visible from the 1st of the month it measures** (`src/build_dataset/merge.py:199-236`)
  - **Severity:** High
  - **Why it matters for Quant Finance:** SGS 433 stamps month M's inflation at `reference_date = M-01`, but IBGE publishes IPCA around the 10th of month M+1. `merge_macro`'s `merge_asof(..., direction="backward")` + ffill gives every trading day inside month M the full-month M print — information published ~40 days in the future. This leaks directly into the raw `ipca` feature and into `real_return` (Issue 1). A backtest conditioning on `ipca` gets an inflation-nowcast edge no live model would have.
  - **Proposed Fix:** Shift IPCA's availability date before the asof merge: either fetch the actual release calendar, or conservatively set `available_date = reference_date + MonthEnd(1) + 15 days` (always after the real release) and `merge_asof` on that. Same review should be applied to any future monthly BCB series; daily selic/cdi are same-day-known and fine.

- [ ] **Issue 3: `selic_trend_20d` bleeds across ticker boundaries and imports future data** (`src/build_dataset/features.py:320`)
  - **Severity:** High
  - **Why it matters for Quant Finance:** `df["selic"] - df["selic"].shift(20)` runs on the whole ticker-blocked batch without `groupby("ticker")`. The first 20 rows of every ticker subtract the *previous ticker's last rows'* selic — typically a 2026 value subtracted from a year-2000 row. Verified: 511/515 tickers have a non-NaN `selic_trend_20d` on their very first row (impossible for a correct 20-day trailing diff), with garbage magnitudes up to ±0.06. It is both wrong (compares unrelated dates) and a look-ahead (a ticker's earliest rows see dataset-end rate levels).
  - **Proposed Fix:** Compute per ticker: `df["selic_trend_20d"] = df.groupby("ticker")["selic"].transform(lambda s: s - s.shift(20))` — or better, compute it once on the deduplicated macro date grid (`selic_by_date.diff(20)`) and map onto rows by `trade_date`, which is also ~500x cheaper and makes the boundary problem structurally impossible.

- [ ] **Issue 4: `div_yield_12m` divides nominal dividends by dividend/split-adjusted price** (`src/build_dataset/features.py:61-113`)
  - **Severity:** High
  - **Why it matters for Quant Finance:** Trailing-12m dividends are nominal per-share amounts at their ex-dates, but the denominator is `adj_close`, which is discounted backward from "now" by every dividend and split since. The mismatch grows with distance from the present: verified on BBAS3, the yield reads 6.8% vs the true 2.9% in 2010 and 13.0% vs 5.7% in 2015, converging to correct only near dataset-end. The feature has a built-in secular downtrend that a model will read as "yields structurally compressed," and cross-sectional comparisons at a given historical date are distorted by each ticker's differing cumulative adjustment. `div_yield_sector_percentile` (cross_sectional.py) inherits this.
  - **Proposed Fix:** Compute yield on a consistent basis. Simplest correct form: sum per-event yields — for each dividend, `value_per_share / close_at_ex_date` (nominal ÷ nominal, same day, immune to later adjustments), then `div_yield_12m` = trailing sum of those event yields. Alternatively convert each dividend to the adjusted basis (`value_per_share × adj_close/close at ex-date`) before dividing by today's `adj_close`.

- [ ] **Issue 5: `status` (and to a lesser degree `sector`) are current-day snapshots joined onto all history** (`src/build_dataset/merge.py:92-192`)
  - **Severity:** High (known/documented — restated here for completeness)
  - **Why it matters for Quant Finance:** `merge_company_info` stamps today's ATIVO/CANCELADA onto every historical row; a model reading `status` at a 2012 row is told whether the company survived to 2026 — textbook feature-level survivorship leakage. The status-inference and "CANCELADA but recently trading" overrides also use `trade_date.max()` (dataset-end knowledge). `sector` is the same static join used inside `compute_cross_sectional_features`, so sector z-scores/momentum use 2026 sector classifications historically (companies do migrate sectors; lower information content, but nonzero). CLAUDE.md documents `status` and places the exclusion burden on consumers — that burden is easy to miss.
  - **Proposed Fix:** Minimum: exclude `status` from every model-facing feature list mechanically (e.g. record it in the scaler metadata / a `NON_FEATURE_COLS` constant consumed by downstream code, not just prose). Better long-term: source point-in-time listing status from the CVM register (already collected for filing dates) and join as-of.

### Medium

- [ ] **Issue 6: Same-day visibility of fundamentals (`merge_asof` exact-match on filing date)** (`src/build_dataset/merge.py:50-57`)
  - **Severity:** Medium
  - **Why it matters for Quant Finance:** `merge_asof(..., direction="backward")` includes exact matches, so a filing whose `DT_RECEB` is day T is attached to day T's row. CVM receipt timestamps are date-granular (verified: all midnight) and companies routinely file after the trading session; a strategy evaluated at T's close would often not have had those numbers intraday. This is a mild but systematic optimistic skew on the freshest quarter — exactly the rows where fundamentals moves prices.
  - **Proposed Fix:** Make fundamentals visible from T+1: `merge_asof(..., allow_exact_matches=False)`, or add one day to `fundamentals_available_date` at attach time. Update `test_merge_honors_actual_filing_date` accordingly.

- [ ] **Issue 7: Frozen BolsAI era vs re-adjusted yfinance era — a growing adj_close discontinuity at the junction** (`src/data_collection/yf_collectors.py:74-96`)
  - **Severity:** Medium (small today, grows every quarter)
  - **Why it matters for Quant Finance:** `_prices_fetch_start` correctly re-fetches the whole yfinance era each `--mode update`, keeping that era internally consistent. But yfinance's `auto_adjust` discounts rows backward from "now," while the BolsAI-era rows before the junction are frozen at their 2026 backfill basis. Every dividend a ticker pays from now on lowers the yfinance-era rows relative to the frozen BolsAI rows, opening a fake negative `log_return` exactly at the junction date — one per ticker, compounding quarterly. Today the measured basis gap is negligible (first update cycle); after a few years of quarterly updates it will look like the same class of splice bug already fixed once in `continuity.py`.
  - **Proposed Fix:** Reconcile at the junction the same way `apply_ticker_continuity` does: after each yfinance refetch, compute the ratio between the stored BolsAI adj_close at the last BolsAI date and the freshly implied yfinance adj basis at that same date, and rescale the incoming yfinance-era `adj_*` by it (anchor yfinance to the frozen BolsAI basis, not vice-versa — BolsAI rows must stay untouched per the no-reconstruction rule).

- [ ] **Issue 8: Fundamentals *values* may be restated; only availability *dates* are point-in-time** (`src/data_collection/collectors.py:260-302`, `src/build_dataset/quality_filters.py:183-235`)
  - **Severity:** Medium
  - **Why it matters for Quant Finance:** `filing_dates.py` deliberately takes the *earliest* CVM receipt (v1) as the availability date ("the market saw the numbers at v1") — but the numbers themselves come from BolsAI's current `/fundamentals/history`, which almost certainly reflects the latest restatement. Where a company restated (common after auditor review of ITRs), the dataset shows corrected figures at the original v1 date — information nobody had then. This is an as-reported vs as-restated mismatch, a classic subtle lookahead in fundamental factors.
  - **Proposed Fix:** No clean fix with BolsAI alone. The CVM open-data ZIPs already being downloaded (`cvm/statements.py`) contain every filing *version*; a point-in-time-strict build could source as-first-reported figures from CVM v1 filings for the overlap universe. Cheaper interim: document the caveat in CLAUDE.md, and quantify it once by diffing CVM v1 vs BolsAI current values on a sample to size the effect.

- [ ] **Issue 9: Residual survivorship: dead tickers with zero fundamentals are dropped, and delistings end silently** (`src/build_dataset/quality_filters.py:87-156`, `src/build_dataset/continuity.py`)
  - **Severity:** Medium
  - **Why it matters for Quant Finance:** Two mechanisms tilt the panel toward survivors: (a) `filter_tickers_with_no_fundamentals` removes tickers whose fundamentals BolsAI never covered — disproportionately old delisted names, i.e. exactly the failure cases; (b) a delisted/bankrupt ticker's series simply stops — there is no terminal event (tender cash-out, bankruptcy → ~−100%) — so any return computed over "held to the end" positions never realizes the loss. The universe work (85 CANCELADA collected, keep_separate handling) mitigates the classic form, but the built dataset still under-represents catastrophic outcomes.
  - **Proposed Fix:** (a) Log the dropped-for-no-fundamentals tickers into the manifest so universe studies can quantify the bias; where CVM statements exist for them (`cvm/statements.py` pipeline), backfill fundamentals from CVM instead of dropping. (b) Add a per-ticker terminal-event row/flag (`delist_date`, `delist_type`, terminal payoff where known — tender price from the continuity map) so downstream labels can realize the final return. The continuity map's `tender` entries already carry the intent.

### Low

- [ ] **Issue 10: Rolling-percentile features use `min_periods=1` — degenerate warm-up values instead of NaN** (`src/build_dataset/features.py:457-499`)
  - **Severity:** Low
  - **Why it matters for Quant Finance:** `volatility_*_percentile`, `price_percentile_*`, `pl_percentile_5y`, `drawdown_percentile` all rank within a rolling window with `min_periods=1`: a ticker's first row is always percentile 1.0 and early rows rank inside tiny windows. Causally safe (no future data) but statistically noise, and inconsistent with the pipeline's own warm-up policy (every other rolling feature is NaN until its window fills). Young listings get systematically extreme percentile features.
  - **Proposed Fix:** Set `min_periods` to a meaningful floor (e.g. 63 trading days, or 252 to match the zhist convention) and let the warm-up be NaN like the rest of the pipeline.

- [ ] **Issue 11: Quarter-window arithmetic is positional, not calendar** (`src/build_dataset/features.py:276-297`, `src/build_dataset/cagr_handler.py:56-104`)
  - **Severity:** Low
  - **Why it matters for Quant Finance:** YoY growth (`pct_change(4)`), F-score `shift(4)`, 4-quarter trends, and the 20-quarter CAGR lookback all assume contiguous quarterly rows. A missing vendor quarter silently stretches "1 year ago" to 15 months, mislabeling growth rates. Prefix-shaped-NaN tests cover column NaNs, not absent quarter *rows*. Coverage is mostly contiguous today, so impact is small — but it's unguarded.
  - **Proposed Fix:** Guard with dates: after computing, NaN-out rows where the shifted row's `reference_date` isn't within (say) 350–380 days (for 4q) / 4.75–5.25y (for 20q) of the current row's; or reindex each ticker's fundamentals onto a complete quarterly grid before differencing.

- [ ] **Issue 12: Market/beta reference is a self-inclusive equal-weighted universe mean; collected benchmark BOVA11 is unused** (`src/build_dataset/cross_sectional.py:93-111`, `src/build_dataset/quality_filters.py:73-75`)
  - **Severity:** Low
  - **Why it matters for Quant Finance:** `beta_1y` and `momentum_vs_market_*` benchmark against the equal-weighted mean of whatever tickers exist that day — microcap-heavy, composition-shifting (thin in early years), and including the stock itself (self-inclusion shrinks relative momentum, materially so on thin early dates). Meanwhile BOVA11 is collected precisely as the IBOV proxy but is excluded by the no-fundamentals filter before ever being used. Betas vs an equal-weight all-share index differ meaningfully from betas vs IBOV.
  - **Proposed Fix:** Route BOVA11 through as the market series for beta (exempt it from the filter for this purpose only, or load it separately in `compute_cross_sectional_features`); for momentum-vs-market, either exclude self from the mean (`(sum - x)/(n-1)`) or accept and document the EW-universe definition.

- [ ] **Issue 13: Trailing-12m dividend sum mixes pre/post-split nominal values within the window** (`src/build_dataset/features.py:93-107`)
  - **Severity:** Low
  - **Why it matters for Quant Finance:** When a split falls inside the trailing 365-day window, pre-split per-share dividends (big nominal) are summed with post-split ones and divided by one price — overstating yield for up to a year after every split. Bounded (only split-adjacent windows) but bunched exactly at corporate-event dates where other features are also stressed.
  - **Proposed Fix:** Folded into the Issue 4 fix: summing per-event yields (`value/close_at_ex_date`) makes each event self-normalizing and eliminates this window mixing too.

- [ ] **Issue 14: `amihud_illiquidity` denominator uses adjusted, not traded, currency volume** (`src/build_dataset/features.py:234`)
  - **Severity:** Low
  - **Why it matters for Quant Finance:** `volume * adj_close` understates actual traded currency in deep history (adj_close carries dividend discounts; raw `traded_amount` exists in the schema), inflating historical Amihud levels with a secular trend. The within-ticker `amihud_illiquidity_zhist_5y` largely neutralizes this; the raw column and its cross-sectional use retain the drift. Split-consistency is fine (volume was rescaled with prices).
  - **Proposed Fix:** Use `traded_amount` (or `volume * close`) as the denominator; keep the zhist variant unchanged.

- [ ] **Issue 15: CDI/SELIC daily-% and IPCA monthly-% units are undocumented at the schema level** (`src/data_collection/config.py`, dataset columns `selic`, `cdi`, `ipca`)
  - **Severity:** Low
  - **Why it matters for Quant Finance:** The raw macro columns pass through to the dataset in heterogeneous units (% per day vs % per month). Issue 1 shows even this repo's own feature code misread them; any downstream consumer is one comment away from the same bug.
  - **Proposed Fix:** Alongside the Issue 1 fix, either normalize all three to a common convention at load time (e.g. annualized decimal) with the raw units preserved under suffixed names, or at minimum record units in the manifest's column stats and CLAUDE.md.

## 3. Checked and found sound

- **Fundamentals asof-merge** (backward on real CVM `DT_RECEB`, statutory fallback), **close_price replacement**, **filing-lag filter ordering** (features computed before rows are dropped, so positional YoY windows aren't corrupted by the filter).
- **Timezone alignment:** all sources land as naive dates (BolsAI naive, BCB naive, yfinance tz-dropped via `tz_localize(None)` from America/Sao_Paulo — date-preserving for B3). No cross-source date skew found.
- **Rolling features are causal:** percentile ranks use rolling (not global) rank; zhist features are trailing-inclusive; split repair and continuity splicing verified consistent with volume scaling; dividend files under post-rename names do cover pre-rename history (verified B3SA3/VBBR3/TIMS3/YDUQ3), so no dividend loss across splices.
- **Split config / scaler:** date-based split, train-only RobustScaler fit via injected FitWindow; transform preserves column order.
- **clean_dataset** inf→NaN and `_safe_ratio` near-zero-denominator guards; loaders' implausible-dividend gate (PDGR3-class vendor error).
