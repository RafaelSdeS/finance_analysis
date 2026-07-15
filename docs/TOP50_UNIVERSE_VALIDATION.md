# Top-50 Universe: Selection Design + Validation Gap Analysis

Written 2026-07-14. Scope: (1) resolve the fixed-vs-reselected universe design
question for the first ML validation phase, (2) audit `tests/` against a
broad data-integrity checklist and list only what's actually missing —
most of the checklist is already enforced today (table below).

## 1. Universe selection: fixed top-50 vs annual reselection

**Recommend: rolling, point-in-time reselection — not either option as posed.**

Both options in the original framing have a bias:

- **Fixed top-50 over the whole history** — survivorship bias. Ranking by
  total historical `traded_amount` uses the *entire* history (including
  years after the point you'd be trading at) to decide membership as far
  back as 2000. It also silently drops names that were liquid for a decade
  then delisted, because their lifetime total looks small next to a stock
  that's been trading 25 years.
- **Naive "top 50 reselected each calendar year"** — fixes survivorship but
  reintroduces lookahead at a smaller scale: ranking January 2020 by full-
  year 2020 `traded_amount` uses December's volume to decide January's
  membership. This is exactly the class of bug `CLAUDE.md` already treats
  as a hard rule for prices/fundamentals (`merge_asof(..., direction='backward')`)
  — it just hasn't been applied to universe construction yet.

**Correct construction:** at each rebalance date, rank tickers by *trailing*
`traded_amount` (e.g. trailing 252 trading days, using only data up to and
including that date), take the top 50, lock membership until the next
rebalance (quarterly or annually), then repeat walking forward. Union across
all rebalance periods for the tickers that ever qualify — this recovers
delisted names without letting any single day's membership see future volume.
Store the result as a per-date `in_universe` boolean (or a `(ticker, start,
end)` membership table), not a static list.

Note on existing code: `tests/build_dataset/test_top_traded_quality.py`'s
`build_universe()` already computes something adjacent (union of whole-period
top-50 + per-year top-50) — but it does so to pick a **generous** ticker set
worth spot-checking for data-quality bugs, which is fine for that purpose (a
superset test scope, not a leakage-sensitive backtest input). **Do not reuse
that function as the production universe for training/backtesting** — it has
both of the biases above baked in on purpose. If/when the actual point-in-time
universe-construction logic is built, it's a Stage-3/agent concern (per
`CLAUDE.md`, `ml_dataset_training.parquet` and similar are built in the
`ml_agent` branch, outside this repo's `src/`) — this repo's job is to make
sure the *raw* dataset doesn't pre-filter to currently-active tickers, so that
whichever repo builds the universe can do it correctly. That's §3.1 below.

## 2. Existing test coverage (already enforced — no action needed)

| Checklist item | Covered by |
|---|---|
| OHLC consistency (low≤open/close≤high, high≥low) | `test_top_traded_quality.py::check_ohlc_consistency`, `validate.py::validate_prices` |
| Missing trading days / calendar gaps | `test_top_traded_quality.py::check_trading_calendar_gaps` |
| Duplicate rows | `test_final_dataset.py`, `test_top_traded_quality.py` (`duplicated(["ticker","trade_date"])`) |
| Impossible prices/volume (≤0, NaN, negative) | `test_top_traded_quality.py::validate()`, `validate.py::validate_prices` |
| Stale prices (flat runs) | `test_final_dataset.py::check_stale_prices` |
| Outliers/suspicious jumps | `test_final_dataset.py::check_outliers_zscore` (signed-log1p, per-`(ticker,year)` for trend cols) |
| Corporate-action adjustment errors (splits) | `repair.py::repair_unadjusted_splits` + `test_repair.py` (5 cases incl. direction, window, threshold) |
| Fabricated/padded vendor data (flat-fill gaps) | `yf_collectors.py::_flat_run_fraction` guard (see `ANOMALY_INVESTIGATION.md`) |
| Invalid/inconsistent fundamentals | `validate.py::validate_fundamentals`, `test_ratios_no_inf.py` |
| Macro alignment, no lookahead | `test_merge.py::test_merge_macro_aligns_by_date_no_lookahead` |
| Look-ahead bias (fundamentals) | `test_merge.py::test_merge_honors_actual_filing_date`, `test_final_dataset.py` (T31, `reference_date <= trade_date`) |
| Data leakage (scaler train/val/test) | `test_scale_features.py` (train-only fit, reproducible refit) |
| NaNs / infinite values | `test_final_dataset.py` (inf check, prefix-NaN-shape rule) |
| Delisted companies / ticker renames & mergers | `test_ticker_continuity.py` (rename/merger splice), `test_collect_delisted.py` |
| Fundamentals coverage gaps explained (not silent) | `test_final_dataset.py` (CAGR NaN reason-coded), `test_quality_filters.py::test_filter_tickers_with_no_fundamentals_classifies_exclusions` |
| Weekend/fabricated trading days | `test_final_dataset.py` (`dayofweek >= 5`) |

## 3. Actual gaps — new tests to add

### 3.1 Survivorship-bias regression guard (HIGH)
- **Why:** this is the load-bearing assumption behind §1's recommendation —
  if a future refactor of the collection/filter pipeline starts dropping
  `CANCELADA`/delisted tickers, the point-in-time universe logic downstream
  silently degrades back into survivorship bias with no signal anywhere.
- **How:** in `tests/build_dataset/`, assert `ml_dataset.parquet` contains a
  non-trivial number of tickers whose `company_info.status == 'CANCELADA'`
  (currently 85 per `CLAUDE.md`), each with at least N rows of price history
  before their last trade date — not just present in `data/raw/`, but
  surviving all the way through `filter_tickers_with_no_fundamentals` /
  `filter_excessive_filing_lag` / `clean_dataset`.
- **Fails when:** delisted-ticker count in the final parquet drops below some
  floor (e.g. <60% of the raw delisted count), or any specific known-liquid
  delisted name (e.g. one already used as a fixture in
  `test_quality_filters.py`) disappears entirely.
- **Threshold:** floor ratio, e.g. 0.6 — generous enough to allow legitimate
  drops (delisted co. with zero fundamentals ever filed) without masking a
  systemic regression.
- **Severity:** hard failure — this one silently reintroduces the exact bias
  this whole analysis is about.

### 3.2 Schema/dtype contract for `ml_dataset.parquet` (MEDIUM)
- **Why:** no current test asserts a fixed set of expected columns + dtypes
  exist. The `ml_agent` branch consumes this parquet directly; a silent dtype
  drift (e.g. `trade_date` becoming `object` instead of `datetime64[ns]`, or
  a bool flag column becoming `float64` with NaNs after a merge change) would
  break the consumer without failing anything in this repo.
- **How:** a small allowlist dict `{column: expected_dtype}` for the columns
  the agent is known to depend on (`ticker`, `trade_date`, `close`,
  `adj_close`, `has_fundamentals`, `has_dividends`, the `*_defined` flags,
  macro columns) checked with `df.dtypes[col] == expected` (or
  `pd.api.types.is_datetime64_any_dtype` etc. for the date columns).
- **Fails when:** any allowlisted column is missing, or its dtype no longer
  matches (e.g. a 0/1 flag column becomes non-numeric).
- **Threshold:** exact match, no tolerance — this is a contract check, not a
  statistical one.
- **Severity:** hard failure for missing columns; warning for dtype drift
  that's still numerically compatible (e.g. `int64` → `float64` on a flag
  column is a smell but not always breaking).

### 3.3 ~~Timezone/date-dtype consistency~~ — dropped
Was a forward-looking regression guard against a hypothetical future vendor
change, not a check on data that's already on disk. Out of scope: goal here
is validating already-collected data, not hardening the pipeline against
changes that haven't happened. If a tz mismatch ever existed, it would
already show up as a symptom in the macro/fundamentals NaN checks that
exist today.

### 3.4 Cross-ticker consistency for same-company multi-class shares (LOW)
- **Why:** several companies in the universe list common+preferred pairs
  (e.g. `PETR3`/`PETR4`, `ITUB3`/`ITUB4`, `BBAS3` units). A ticker-mapping or
  CVM-crosswalk bug (wrong `cvm_code` attached to a share class) would show
  up as a sibling pair whose returns suddenly decorrelate — worth a cheap
  informational check, not a hard gate (real-world share classes do
  legitimately diverge, e.g. voting-rights premia during M&A speculation).
- **How:** for known sibling pairs (or all pairs sharing a `cvm_code` via
  `company_siblings()`, already used by `merge.py`), compute rolling 60-day
  return correlation; report pairs whose correlation drops below ~0.5 for an
  extended window.
- **Fails when:** nothing — informational only, printed for manual review.
- **Severity:** informational.

## Results — run against `data/processed/ml_dataset.parquet` (2026-07-14)

`python tests/build_dataset/test_universe_integrity.py` → **VALIDATION PASSED**
(1,327,251 rows, 517 tickers).

- **3.1 Survivorship:** 85/122 (70%) of raw `CANCELADA` tickers survive into
  the dataset with ≥10 rows — above the 60% floor. Spot-checked the "missing"
  list: `BRDT3` is a known rename already spliced into `VBBR3` via
  `ticker_continuity.json` (expected, not a leak); the rest are plausibly
  "never collected" (per `CLAUDE.md`'s standing caveat on raw-data gaps), not
  audited individually here.
- **Minor finding, not fixed:** one row in `company_info.parquet` has
  `ticker == "22853"` (a CVM code, not a real ticker symbol) for "FRIGOL
  FOODS PARTICIPAÇÕES S.A.", `status=CANCELADA`. Harmless today — it has 0
  rows in `ml_dataset.parquet` so it doesn't merge into anything — but it's a
  malformed row in the source data worth a cheap fix in the company_info
  collector if noticed again.
- **3.2 Schema/dtype contract:** first run caught 2 wrong assumptions in the
  test itself (`volume` is legitimately `int64`, `has_fundamentals` is
  legitimately `float64` per `features.py:304`) — corrected in the allowlist,
  re-ran clean.
- **3.4 Sibling correlation (informational):** 11 pairs below the 0.5
  threshold, incl. `TOYB3/TOYB4` at -0.97. Spot-checked directly — both are
  extremely illiquid micro-caps (prices 0.02–2.50, multi-month gaps between
  trades); correlation on that few actual trades is noise, not a
  ticker-mapping bug. No action needed.

## Checklist

- [x] 3.1 Survivorship-bias regression guard (hard failure) — `tests/build_dataset/test_universe_integrity.py`, not yet run against real data
- [x] 3.2 Schema/dtype contract test (hard on missing col, warn on dtype drift) — same file
- [x] ~~3.3 Timezone consistency check~~ — dropped, not applicable to validating already-collected data
- [x] 3.4 Cross-ticker sibling-correlation check (informational) — same file
