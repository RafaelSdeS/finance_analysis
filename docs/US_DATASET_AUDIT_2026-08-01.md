# US dataset audit — 2026-08-01

Target: `data/processed/us_ml_dataset.parquet` (15,419,040 rows × 167 cols, 2,960
tickers, 1962-01-02 → 2026-07-29), built at commit `0b85177`, plus the raw
`data/raw/us/fundamentals/` corpus (8,143 files) behind it.

Scripts used are throwaway (scratchpad); every number below is reproducible from
the dataset with the queries described inline.

---

## What is already clean — verified, no action

- [x] **0** duplicate `(ticker, trade_date)` rows.
- [x] **0** lookahead violations — `fundamentals_available_date > trade_date` never occurs (10,680,081 fundamental-bearing rows checked).
- [x] **0** negative filing lags (`available_date` is never before `reference_date`).
- [x] **0** OHLC violations — no `high < low`, no open/close outside `[low, high]`, no non-positive `close`/`adj_close`, no negative volume. The collector-side repairs hold.
- [x] `has_fundamentals` is perfectly consistent with `reference_date` presence (0 mismatches both directions).
- [x] `days_since_fundamental` matches recomputed info-age exactly (0 mismatches) — the 2026-07-24 BR fix carried over correctly.
- [x] `sector` join is static and consistent — 0 tickers carry >1 distinct sector.
- [x] Extreme returns are plausible in count: 24 rows with `|log_return| > 2`, concentrated in genuinely broken microcaps (TSEOQ, ADAPY).
- [x] `fundamentals_tier` timeline matches the plan doc exactly: ex27 owns 1994–2000, item6 takes over 2001–2008, xbrl from 2009 (99.4% by 2012).

---

## P0 — corrupts a broad slice of training features

### 1. `f_score` and its 4 components silently treat "missing" as "bad news"

**100% of the 7,048,279 populated `f_score` values have at least one missing raw input.**
Not a sample — every single one.

`src/build_dataset/features.py` (`compute_fundamental_features`):

```python
g["f_roa_positive"]        = (g["roa"] > 0).astype(float)                                    # no guard at all
g["f_roa_improving"]       = (g["roa"] > g["roa"].shift(4)).where(yoy_ok).astype(float)
g["f_margin_improving"]    = (g["gross_margin"] > g["gross_margin"].shift(4)).where(yoy_ok).astype(float)
g["f_leverage_decreasing"] = (g["debt_equity"] < g["debt_equity"].shift(4)).where(yoy_ok).astype(float)
g["f_liquidity_improving"] = (g["current_ratio"] > g["current_ratio"].shift(4)).where(yoy_ok).astype(float)
```

Two independent defects:

- `f_roa_positive` has **no NaN guard whatsoever**. `NaN > 0` → `False` → `0.0`, i.e. "unprofitable", not "unknown". **1,706,002 rows** (15.97% of fundamental-bearing rows) have `roa` NaN but `f_roa_positive` populated.
- The other four use `.where(yoy_ok)`, but `yoy_ok` only checks **calendar spacing** (`_within_calendar_gap`: are these two rows genuinely 4 quarters apart?). It never checks whether the compared *values* exist. A NaN input on a well-spaced quarter still yields a real `0`/`1`.

Rows where the raw input is NaN but the flag is populated:

| raw input | flag | rows | % of fundamental rows |
|---|---|---:|---:|
| `roa` | `f_roa_positive` | 1,706,002 | 15.97% |
| `gross_margin` | `f_margin_improving` | 7,048,279 | 65.99% |
| `debt_equity` | `f_leverage_decreasing` | 3,002,523 | 28.11% |
| `current_ratio` | `f_liquidity_improving` | 1,716,066 | 16.07% |

Because `gross_margin` is 93.6% null dataset-wide, **`f_margin_improving` is effectively hardwired to 0** — its non-null count (7,048,279) is exactly the populated `f_score` count. So `f_score` is in practice a 4-component score with a constant-zero fifth component, and the other components degrade to 0 wherever their input is missing.

The `skipna=False` on the `f_score` sum was written specifically so an undefined
component makes the whole score undefined — that intent is correct and is being
defeated upstream, because the components never become NaN in the first place.

- [ ] Guard each component on its own raw inputs (`.notna()`) before comparing, so a missing input yields NaN and `skipna=False` propagates it.
- [ ] Decide separately whether `f_margin_improving` should exist at all given 93.6%-null `gross_margin`, or be swapped for `net_margin` (49.98% null).
- [ ] Same code path runs for BR — check and re-report BR impact before/after.

### 2. The `shares_outstanding` fix is committed but was never applied to the data

`companyfacts._reject_sequential_outliers` + the `_SHARES_UNIT_CONCEPTS` unit
guard (both in commit `820f647`) are correct, but **only 28 of 8,143 raw
fundamentals files were recollected after the fix**. 99.66% of the corpus still
holds the uncorrected values.

Evidence in the built dataset:
- `market_cap > $10T`: **3,511 rows** (BAC-PL 2,687, LTM 503, PKG 127, REGN 126, TSM 68). LTM is one of the tickers the fix was written for.
- `shares_outstanding` consecutive-filing jumps >50×: **314 filings across 175 tickers**.
- `shares_outstanding <= 0`: 286 filings; `< 1,000`: 351 filings.
- The `shares_outstanding_rejected_outlier` flag is non-null on only 96,990 of 15.4M rows — i.e. only the 28 recollected tickers.

Blast radius: `market_cap` and everything derived from it — `pl`, `pvp`, `p_sr`,
`p_assets`, `p_ebit`, `book_to_market`, `earnings_yield`, `earnings_yield_vs_selic`,
`peg_ratio`, `pvp_to_roe_ratio`, plus every `*_zhist_5y` and `*_zscore_sector`
built on those.

- [ ] Recollect `companyfacts` for the full universe (or at least every ticker whose `shares_outstanding` shows a >50× jump), then rebuild.
- [ ] Add a build-time assertion that fails loudly if the raw corpus predates a known collector fix (schema-version marker on the raw files).

---

## P1 — real, bounded, needs a decision

### 3. Fundamentals scale/parse corruption in a minority of filings

Values that are orders of magnitude wrong, concentrated in the pre-XBRL tiers but
present in all three:

- `total_assets < $100,000` for a public filer: **109 filings, 68 tickers** (item6 61, xbrl 32, ex27 16).
- `total_assets` consecutive-filing jump >50×: **327 filings, 236 tickers**. The *post*-jump value is the correct one in every case inspected, so the *pre*-jump value is the corrupt one.

Concrete:

```
CVBF  2006-12-31  total_assets =        0.0   (item6)
CVBF  2007-12-31  total_assets =       20.0   (item6)     ← should be ~$6.5B
CVBF  2011-03-31  total_assets = 6,498,352,128 (xbrl)     ← correct

BPOP  FY2006  net_income = 740          (item6, from the 2007-03-01 10-K)
BPOP  FY1990  net_income = 63,400,000   (item6, from the 2005-03-16 10-K)  ← correct raw dollars
```

Both `fds.py` (ex27) and `selected_financial_data.py` (item6) *do* implement unit
multipliers (`fds_multiplier`, `detect_unit_multiplier`), so this is inconsistent
*application*, not a missing feature. The tier-boundary medians shift by only
+0.24 / −0.13 / +0.08 orders, confirming it is **per-filing, not a uniform
per-tier offset** — so a blanket rescale would be wrong.

Note also `BPOP` shows item6 rows whose `reference_date` predates the source
filing by ~15 years (FY1990 sourced from a 2005 10-K, FY1992/93 from a 2008
10-K). Item 6 tables legitimately chain 5 years; 15 is not legitimate and
suggests year-label misparsing in `extract_years`.

Unit-invariant identity violations (these catch wrong-row as well as wrong-scale),
per unique `(ticker, reference_date)` filing:

| check | violations | of | rate |
|---|---:|---:|---:|
| `equity > total_assets` | 87 | 103,959 | 0.08% |
| `current_assets > total_assets` | 15 | 79,170 | 0.02% |
| `cash > total_assets` | 48 | 101,508 | 0.05% |
| `\|net_income\| > 10× net_revenue` | 2,146 | 87,209 | 2.46% |
| `total_debt > 10× total_assets` | 15 | 62,286 | 0.02% |
| `net_revenue < 0` | 227 | 89,158 | 0.25% |
| **any** | **2,475** | **110,121** | **2.25%** |

439 of 2,960 tickers have at least one violating filing. By tier: xbrl 2.32%,
ex27 1.25%, item6 0.60% — the `net_income > 10× net_revenue` term dominates and is
likely a `net_revenue` mapping issue for banks/BDCs/REITs rather than a scale bug.

- [ ] Add these identity checks to `validate_us_fundamentals` as warnings (they are cheap and unit-invariant).
- [ ] Quarantine or NaN the tiny-value filings (`total_assets < 1e5` etc.) — same drop-and-log convention as `loaders.load_dividends`.
- [ ] Investigate `extract_years` year-label assignment for the ~15-year-offset item6 rows.

### 4. Ticker-class contamination — 176 tickers share 59 CIKs

Preferred shares, ETNs and baby bonds are crosswalked to their issuer's CIK and
therefore inherit the **common stock's fundamentals** while carrying **their own
price series**. Every resulting valuation ratio is meaningless.

```
cik    70858  → ['BAC','BAC-PB','BAC-PK','BAC-PL','BAC-PM','BAC-PN','BAC-PO','BAC-PP','BAC-PQ','BAC-PS','MER-PK']
cik    19617  → ['AMJB','JPM','JPM-PC','JPM-PD','JPM-PJ','JPM-PK','JPM-PL','JPM-PM']   (AMJB is an ETN)
cik    92122  → ['SO','SOJD','SOJE','SOJF']                                            (baby bonds)
cik    72971  → ['WFC','WFC-PA','WFC-PC','WFC-PD','WFC-PL','WFC-PY','WFC-PZ']
```

75 tickers contain `-`. BAC-PL is the single largest contributor to the
`market_cap > $10T` count above. `BRK-A`/`BRK-B` and `BF-A`/`BF-B` are genuine
dual-class common and are defensible; preferreds and ETNs are not.

- [ ] Decide policy: exclude non-common share classes from the universe gate, or tag them with a `security_type` column so consumers can filter.
- [ ] At minimum, suppress `market_cap` and price-derived ratios where the ticker is not the issuer's common line.

### 5. Stale fundamentals are unbounded — `merge_asof` has no `tolerance`

A ticker that stops filing keeps carrying its last known fundamental forever.
The US build deliberately skips `filter_excessive_filing_lag` (the BR 180-day gate
would delete 27.7% of real US rows — correct call), but nothing replaced it.

Info-age (`trade_date − fundamentals_available_date`) over 10,680,081
fundamental-bearing rows:

| pct | days |
|---|---:|
| p50 | 73 |
| p75 | 200 |
| p90 | 569 |
| p95 | 1,016 |
| p99 | 2,302 |
| p99.9 | 4,103 |
| max | 6,926 (19 years) |

- **1,385,393 rows (12.97%)** carry fundamentals older than 400 days.
- **549,605 rows** carry fundamentals older than 1,000 days.
- 1,628 of 2,960 tickers have at least one stale row.
- Worst case `HTHIY`: last filing `reference_date` 2011-03-31, still emitting rows through 2026-07-28 — 3,498 stale rows on 15-year-old fundamentals.

This is not lookahead (the data was genuinely public), but a model reading
`roe` on a 2026 row sourced from a 2011 filing is being lied to. `days_since_fundamental`
exposes it correctly — the burden is currently entirely on the consumer.

- [ ] Add a `tolerance` to the fundamentals `merge_asof` (or a post-merge mask) so fundamentals go NaN past N days; N ≈ 400–550 keeps p90 intact.
- [ ] Alternatively keep the values and document `days_since_fundamental` as a mandatory gating feature — but pick one, do not leave it implicit.

---

## P2 — dead / degraded columns

### 6. Five columns are 100% NaN

| column | cause |
|---|---|
| `ebitda_margin` | ebitda never collected (no D&A concept mapped) — **documented**, plan §4.4 Phase E |
| `ebitda_growth_yoy` | same |
| `ebitda_margin_zhist_5y` | same |
| `dividend_coverage_ratio` | ebitda-derived (`features.py:542`) — **undocumented consequence** of the above |
| `gross_margin_qoq` | `gross_margin` is annual-cadence and 93.6% null, so the `qoq_ok` ~90-day gap guard never passes. 402 tickers have ≥2 `gross_margin` filings, yet the column is 0 non-null everywhere. |

- [ ] Drop these 5 from the build output, or add them to a documented `known_empty_columns` list in the manifest.

### 7. Line-item coverage is thinner than expected even in the XBRL era

`eps_basic` 94.2% null, `gross_margin` 93.6%, `cost_of_revenue` 93.6%,
`eps_diluted` 93.4%, `capex` 91.2%, `cashflow_ops` 86.5%, `gross_profit_reported` 78.9%,
`roic` 78.2%.

The plan doc frames the narrow line-item set as a pre-2007 characteristic. These
rates are dataset-wide and much worse than the 30.73% no-fundamentals baseline,
so the sparsity extends well into the XBRL tier — likely unmapped/alternate
us-gaap concepts rather than genuinely absent data.

- [ ] Sample 20 XBRL-tier CIKs with null `cashflow_ops`/`capex` and check whether the concept exists under a name `CONCEPT_MAP` does not cover.

### 8. 12.89% of rows predate any possible fundamentals

1,987,713 rows fall before 1994 (29 tickers in 1962 growing to 696 by 1993), with
0% `has_fundamentals` — and `momentum_vs_market_12m` is 100% null there too
(SPY's 1993 floor). These rows are price-features-only.

- [ ] Confirm this is intended. If the dataset is meant for fundamental+price modelling, a global start-date trim (as BR does for its top-50 universe) would remove 1.99M rows of structurally-incomplete history.

---

## Suggested order of work

1. **#1 f_score guards** — small, self-contained diff in `features.py`, fixes 7M rows, affects BR too.
2. **#2 recollect companyfacts** — no code change needed, just a run + rebuild; unblocks every valuation ratio.
3. **#5 staleness policy** — one-line `tolerance` or an explicit documented decision.
4. **#6 drop dead columns** — trivial.
5. **#4 share-class policy** and **#3 parser corruption** — need a design decision first.
