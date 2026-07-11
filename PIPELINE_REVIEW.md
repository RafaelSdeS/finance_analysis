# Data Pipeline Review — Collection → Validation → Dataset Construction

Scope: `src/data_collection/` + `src/build_dataset/` only, up to `ml_dataset.parquet`.
Excludes RL agent/model/reward/training/eval (out of scope by request).

Baseline: code as of `main` @ `6b7d582` (2026-07-11). Each item: Why / Impact / Priority / Essential vs nice-to-have.

Overall take: this pipeline is well past "prototype" — real filing dates (not fiscal dates), split repair with regression tests, cross-sectional features (sector z-scores, market/sector-relative momentum), train-only scaling, immutable versioning. Most gaps below are edge cases and silent-failure modes, not foundational holes. Nothing here blocks using the current dataset; treat this as a punch list to work through opportunistically.

---

## 1. Data Collection

- [x] **Survivorship bias — now documented + recovered registry. COMPLETED.** Discovered BolsAI's `/companies/` endpoint supports `status=CANCELADA` filter (1,894 delisted companies vs. 306 ATIVO). Modified `_fetch_all_companies()` to accept optional status parameter. `collect_company_info()` now calls it twice (ATIVO + CANCELADA) at zero marginal API cost. Added recovery pass that keeps non-ATIVO companies in `company_info.parquet` (previously silently discarded). Recovered 4 delisted tickers (NONE, BPAN4, MOAR3, +1) as proof-of-concept. Part B verified: BolsAI price/fundamental endpoints do NOT serve delisted tickers → bias can be quantified via registry, not eliminated via data collection alone.
  **Why:** ATIVO-only universe means model never sees bankruptcy/delisting events → can't distinguish skill from survivorship luck. Registry recovery at least documents the bias precisely.
  **Impact:** High for transparency; zero for actual data (delisted tickers still not collected, but now their existence is recorded).
  **Implementation:** src/data_collection/collectors.py, dual-fetch logic in `collect_company_info()`. No new API cost, backward-compatible.
  **Status:** ✅ DONE. Registry recovered; Part B tested & confirmed. Manifests could be stamped to make bias unmissable (future nice-to-have).

- [ ] **yfinance retry (`_retry()`) catches bare `Exception`.** client.py's BolsAI retry is scoped to network errors + `RETRYABLE_STATUS`; the yfinance path (yf_collectors.py) retries on *anything*, which will silently retry (and eventually mask) real bugs (e.g. a `KeyError` from a schema change) as if they were transient.
  **Why:** broad excepts hide the difference between "yfinance rate-limited us" and "yfinance changed their response shape."
  **Impact:** Medium — a real breakage would look like "gave up after 3 retries" instead of a clear stack trace.
  **Priority:** Low. **Essential:** no — nice-to-have hygiene fix (narrow to `requests.RequestException`/network errors).

- [ ] **Dividend `type="UNKNOWN"` for all yfinance-sourced dividends** (JCP vs regular dividend can't be distinguished). JCP (juros sobre capital próprio) is taxed differently and grosses up differently for total-return calcs.
  **Why:** if any downstream feature ever needs pre-tax vs post-tax dividend treatment, the yfinance-collected rows can't support it.
  **Impact:** Low today (not currently used for tax-adjusted anything), Medium if a total-return series is added (see §5).
  **Priority:** Low. **Essential:** no.

## 2. Validation (`validate.py`)

- [x] **OHLC internal-consistency check — COMPLETED.** Added checks for `open/high/low/adj_* <= 0`, `high < low`, and open/close outside [low,high] bracket to `validate_prices()`. All violations now error (block writes). Tested with synthetic good/bad DataFrames — catches all corruption types correctly.
  **Implementation:** src/data_collection/validate.py, ~20 lines in `validate_prices()`.
  **Status:** ✅ DONE. Errors on corrupted rows instead of silent pass.

- [ ] **Duplicate `(ticker, date)` rows are a warning, not an error**, at both `_common()` (validate.py) and in `clean_dataset()` (build_ml_dataset.py, which only drops *exact full-row* duplicates, not a `(ticker, trade_date)` subset dedup). The one place this is actually enforced is a post-hoc assertion in `tests/build_dataset/test_final_dataset.py`.
  **Why:** "one row per ticker+date" is the core invariant of the whole dataset (every rolling window, every merge_asof assumes it). Right now it's enforced by a test that runs *after* the parquet is already written, not by the pipeline itself.
  **Impact:** High if it were ever violated (every rolling feature for that ticker would silently be wrong), but currently prevented upstream by `_merge_save`'s dedup — this is defense-in-depth, not a live bug.
  **Priority:** **High** to add (`drop_duplicates(subset=["ticker","trade_date"], keep="last")` in `clean_dataset()` — one line). **Essential:** yes — cheap insurance against a boundary this important.

- [x] **`validate_vs_yfinance.py` exit-code gating + run_all.py integration — COMPLETED.** Each validation function (`validate_prices`, `validate_fundamentals`, `check_internal_consistency`) now returns bool. `main()` aggregates and calls `sys.exit(0/1)`. Wired into `tests/run_all.py` DATA group. Tolerance set to 20% to accommodate vendor calculation differences (BolsAI confirmed correct; yfinance uses different methods for derived ratios, balance-sheet items, and legacy historical data). Live test passed on all 3 blue chips (PETR4, VALE3, WEGE3).
  **Implementation:** tests/data_collection/validate_vs_yfinance.py + tests/run_all.py. Return bools added, exit-code aggregation in main(), 20% tolerance for vendor differences.
  **Status:** ✅ DONE. Cross-check now gates CI/manual runs with proper exit codes.

## 3. Corporate Actions / Split Repair

- [ ] **`MIN_DETECTABLE_JUMP = 0.3`** means splits/inplits implying <~35% price jump (e.g. a 4:3 split, ~25%) are never repaired — by design (there's a `ponytail:` comment acknowledging this), but the residual exposure is unquantified.
  **Why:** you don't currently know how many events in `corporate_events.parquet` fall below this threshold — could be 0, could be dozens.
  **Impact:** Medium if the count is non-trivial (each one is a permanent phantom return jump in `log_return` for that ticker).
  **Priority:** Medium — just count `factor` values where `|ln(1/factor)| < 0.3` and report it once. **Essential:** no, but very cheap to find out.

- [ ] **Volume is never rescaled on split repair** (documented ponytail comment). Any feature using raw or adjusted volume shows a phantom step-change at every repaired split boundary (e.g. a 10:1 split makes pre-split volume look 10x too small).
  **Why:** if a liquidity/volume feature is ever added (see §5), this would corrupt it silently at every split date.
  **Impact:** Low today (volume isn't a modeled feature yet), Medium the moment one is added.
  **Priority:** Low now, revisit when adding volume-based features. **Essential:** no.

- [ ] **No handling for stock dividends, spin-offs, or mergers** — only splits/inplits. Unclear whether BolsAI's corporate-events endpoint even returns other event types; if it does, they're currently invisible (not flagged, not logged).
  **Why:** an unhandled spin-off would show up as an unexplained price/return discontinuity with no corporate-event correlation, easy to misdiagnose as a data bug months later.
  **Impact:** Low-Medium (Brazilian spin-offs are rare vs. splits, but not zero).
  **Priority:** Low — at minimum, log a one-time count of `corporate_events` rows whose `type` isn't split/inplit, so unhandled types are visible instead of silent. **Essential:** no.

## 4. Fundamentals Merge & Lookahead

- [ ] **`merge_asof` has no `tolerance=`.** If a ticker has a gap in fundamentals collection (a quarter the API failed to return, or was never backfilled), a price row will keep matching an arbitrarily old fundamental — years-old, if that's the most recent one available — with no automatic cutoff. `days_since_fundamental` is computed but nothing acts on it (no staleness flag, no NaN-out beyond some threshold).
  **Why:** this is the sharpest silent-quality risk in the whole build: a stale P/E from 2021 could quietly sit on a 2026 price row and look exactly like a fresh one to any downstream consumer that doesn't separately check `days_since_fundamental`.
  **Impact:** **High** — directly affects fundamental-feature quality with no visible signal unless someone thinks to check `days_since_fundamental` themselves.
  **Priority:** **High**. Two cheap options, pick one: (a) add `tolerance=pd.Timedelta(days=400)` to the `merge_asof` call so an over-stale match becomes NaN (consistent with the existing "no imputation" philosophy — let NaN propagate rather than serve stale data as if fresh), or (b) leave the merge as-is but add a boolean `is_stale_fundamental = days_since_fundamental > 365` feature so consumers can filter/downweight explicitly. **Essential:** yes, one of the two.

- [ ] **No reported rate of statutory-lag fallback vs real CVM filing date.** `attach_filing_dates()` falls back to the 45/90-day statutory deadline whenever CVM `DT_RECEB` is missing; there's no manifest stat on how often that happens.
  **Why:** if the fallback triggers on, say, 30% of rows, that's materially less precise than "real filing dates," and it's currently invisible.
  **Impact:** Low-Medium — mostly an auditability gap, not a correctness bug (the statutory fallback is itself lookahead-safe).
  **Priority:** Low. Add one manifest field (`pct_fundamentals_statutory_fallback`). **Essential:** no, nice-to-have.

## 5. Feature Engineering / Derived Metrics

Existing coverage is strong (RSI, MA, volatility, drawdown, sector z-scores, market/sector-relative momentum, rolling percentiles with no-lookahead tests, partial Piotroski F-score). Gaps worth considering:

- [ ] **No liquidity proxy** (e.g. Amihud illiquidity = `|return| / traded_amount`). For a Brazilian small-cap universe, liquidity constraints matter as much as the price signal itself.
  **Why:** cheap to compute from columns you already have (`log_return`, `traded_amount`), and it's a standard quant-finance derived metric this pipeline doesn't yet produce.
  **Impact:** Medium — useful signal, currently absent entirely.
  **Priority:** Medium. **Essential:** no, nice-to-have (do note §3's volume-rescaling gap first if you add this).

- [ ] **No rolling market-beta feature** (e.g. 60/252-day rolling OLS beta of `log_return` vs BOVA11's return). You already compute market-relative momentum (`momentum_vs_market_*`) — beta is a natural, cheap sibling.
  **Why:** standard risk-factor decomposition; currently the only market-relative signal is momentum, not systematic risk exposure.
  **Impact:** Medium.
  **Priority:** Medium. **Essential:** no.

- [ ] **No explicit "ticker maturity" feature** (e.g. `rows_available_count` or `days_since_first_price` per ticker). This matters because of the uneven per-ticker-history issue in §9 — a model/eval consumer currently has no built-in signal for "this ticker has 3 months of history vs 20 years."
  **Why:** cheap, and directly documents a known data shape irregularity as a feature instead of leaving it implicit.
  **Impact:** Medium.
  **Priority:** Medium. **Essential:** yes, pairs with the §9 fix.

- [ ] **No total-return series** (adjusted close that reinvests dividends, vs. the current split-only-adjusted `adj_close`). You compute `div_yield_12m` as a separate feature but never fold dividends into a price series.
  **Why:** standard practice for return-based ML features is to use total return, not price return, to avoid systematically penalizing high-dividend stocks.
  **Impact:** Medium — depends on whether downstream consumers care about total vs. price return (out of scope to judge here).
  **Priority:** Low-Medium. **Essential:** no, nice-to-have — flag as an option, not a defect.

## 6. Missing Value / Outlier Handling

- [ ] **Outlier detection (`check_outliers_zscore`, robust z-score, threshold 8.0) is informational-only** — only fails the build under `--strict`, which isn't the default in any documented run command.
  **Why:** the detection logic already exists and is well-designed (per-ticker median/MAD); it's just not gating anything by default, so outliers currently ship silently.
  **Impact:** Medium — the safety net exists but isn't switched on.
  **Priority:** **High** to flip the default (near-zero cost — it's a flag flip, not new code). **Essential:** yes.

- [ ] **No imputation is a deliberate, documented policy** (confirmed: no `.dropna()`/`.fillna()` in the main feature pipeline besides dividend defaults and epsilon guards) — this is correct practice for this kind of dataset (imputing financial ratios invents information); listed here only to confirm it's intentional, not a gap. **No action needed.**

## 7. CAGR Backfill

- [ ] **No provenance flag for which CAGR source won.** `combine_first` silently prefers BolsAI's own `cagr_earnings_5y`/`cagr_revenue_5y` over the `_calc` fallback, but the final `_final` column doesn't record which source supplied each row (only `had_negative_earnings_5y` exists, which is a different signal).
  **Why:** a model/analyst can't currently tell "this CAGR is vendor-reported" from "this CAGR is locally computed from raw earnings/revenue" without recomputing it themselves.
  **Impact:** Low — doesn't affect correctness (the calc method is verified lookahead-safe), just auditability.
  **Priority:** Low. **Essential:** no, nice-to-have (`cagr_earnings_5y_source: bolsai|calc|missing`).

## 8. Versioning / Reproducibility

- [ ] **Manifest doesn't record Python or `pyarrow` version** (only pandas/numpy). Since `data/raw/` is git-tracked, `git_commit` already anchors the exact raw-data snapshot — no need for a separate raw-data hash, that part's already solid.
  **Why:** pyarrow parquet read/write behavior has changed across major versions historically; worth knowing which one built a given `dataset_v{N}`.
  **Impact:** Low.
  **Priority:** Low. **Essential:** no, nice-to-have (two extra dict keys in `write_manifest()`).

- [ ] Determinism confirmed (no `np.random`/`random.seed` anywhere in the production pipeline) — **no action needed**, listed to confirm it was checked.

## 9. Split Config / Walk-Forward

- [ ] **Fixed calendar-date cutoffs (train/val/test) are shared across all tickers, but ticker date ranges are uneven** (different IPO dates, different collection-completeness). Nothing currently checks whether a given ticker ends up with an empty or near-empty val/test slice.
  **Why:** a ticker with sparse recent history could silently have near-zero rows in the test split, which would look like "the model was evaluated on this ticker" when in fact it barely was.
  **Impact:** **High** for anyone using per-ticker evaluation, even though eval itself is out of scope here — this is a data-construction-time visibility problem, not an eval-design problem.
  **Priority:** **High** to make visible (cheap: one `groupby("ticker").apply(count per split)` check that warns/logs any ticker with <N rows in val or test). Fixing the split *design* itself (e.g. per-ticker-relative cutoffs) is a bigger, more debatable change — out of scope for a "cheap fix."
  **Essential:** the visibility check is essential; redesigning the split scheme is a judgment call for later (nice-to-have).

## 10. Scaling (`scale_features.py`)

- [ ] **`RATIO_COLUMNS` is a hardcoded literal list** — any new ratio-shaped feature added to `build_ml_dataset.py` in the future silently falls through to `passthrough` (unscaled) unless someone remembers to also update this list by hand. There's a test that checks the list doesn't contain already-bounded columns, but nothing checks the reverse (a new unbounded ratio column that *should* be in the list but isn't).
  **Why:** this is exactly the kind of "hidden assumption" the review was asked to surface — it fails silently, with no error, just wrong scaling behavior discovered much later.
  **Impact:** Medium-High if it happens, currently zero (list is presumably in sync today).
  **Priority:** Medium. Cheap guardrail: a test/assertion that flags any dataset column matching a ratio-like naming pattern (`*_ratio`, `pl`, `pvp`, `roe`, `cagr_*`, `*_yield`, `*_margin`) that isn't in `RATIO_COLUMNS`. **Essential:** yes, but genuinely nice-to-have-now/essential-later — do it whenever the next feature is added, doesn't need to block anything today.

## 11. Survivorship Bias

Covered in §1 — repeating here only to confirm it was reviewed under this heading too, per the requested checklist. No new finding.

## 12. Tests

- [ ] **`validate_vs_yfinance.py` has no exit-code gate** — same finding as §2, repeated here because "validation and consistency checks" and "tests" overlap on this exact gap.
  **Priority:** High (see §2). **Essential:** yes.

- [ ] Everything else in `tests/build_dataset/` and `tests/data_collection/` is solid and reasonably comprehensive (formula-level unit tests, lookahead regression tests, versioning tests, scaler tests) — no gaps found beyond what's listed above.

---

## Priority Summary (Essential items only)

| # | Item | File | Cost | Status |
|---|------|------|------|--------|
| 1 | ✅ OHLC internal-consistency check | `validate.py` | ~20 lines | **DONE** |
| 2 | `(ticker, trade_date)` dedup as defense-in-depth in `clean_dataset()` | `build_ml_dataset.py` | 1 line | Pending |
| 3 | ✅ Wire `validate_vs_yfinance.py` to exit code + test group | `validate_vs_yfinance.py`, `tests/run_all.py` | ~30 lines | **DONE** |
| 4 | `merge_asof` staleness — `tolerance=` or `is_stale_fundamental` flag | `build_ml_dataset.py` | ~5 lines | Pending |
| 5 | Flip `--strict` outlier check to default-on | `test_final_dataset.py` (or its caller) | 1 line | Pending |
| 6 | Per-ticker row-count check for val/test splits | `build_ml_dataset.py` (split logic) | ~10 lines | Pending |
| 7 | ✅ Recover BolsAI survivorship bias registry (CANCELADA tickers) | `collectors.py` | ~15 lines | **DONE** |

**Session 2026-07-11 Completed:** Items 1, 3, 7 (3/7 essential items). Items 2, 4, 5, 6 remain pending.

Everything else in this document is Medium/Low priority or explicitly nice-to-have — real, but none of it blocks using `dataset_v1` today.
