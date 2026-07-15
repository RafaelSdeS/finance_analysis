# Top-50 Independent ML-Readiness Audit (second pass)

Written 2026-07-14, independent of and skeptical toward
`TOP50_ML_READINESS_AUDIT.md` / `TOP50_UNIVERSE_VALIDATION.md` (both read
first, neither assumed correct). Scope: re-derive every important assumption
from the actual source and the actual on-disk `ml_dataset.parquet`
(1,321,455 rows, 515 tickers), not from the prior reports' conclusions.

## Verdict

**Not yet production-ready as currently materialized on disk.** Two real bugs
were found and fixed in source, one of them severe — but fixing the code does
not retroactively fix the parquet already on disk. **A rebuild is required
before training** (see "Action required" below, not performed by this audit
— it regenerates a committed-adjacent result file and this session doesn't
run pipeline code it just edited without your go-ahead).

Everything else checked (lookahead, corporate actions, calendar/dedup
integrity, scaling architecture, universe survivorship, schema contract) held
up under independent re-verification.

## Findings

### 1. CRITICAL — `dividend_coverage_ratio` epsilon guard produced values up to 2.3 quadrillion — FIXED

`features.py::compute_advanced_features()` computed
`ebitda / (div_value_recent * shares_outstanding + 1e-8)`. The `+1e-8` is a
classic "avoid literal division by zero" guard — but `div_value_recent == 0`
(no dividend ever paid) is not rare: **355,854 rows (27% of the whole
dataset) across 213 tickers**, confirmed by direct query. For every one of
those rows, the formula computes `ebitda / 1e-8` — a *finite* number roughly
1e8× the ticker's EBITDA. Measured on the real dataset: mean **9.1 trillion**,
p99 **2.7×10¹⁴**, max **2.3×10¹⁵** (`AXIA5`/`AXIA6`, 2016).

**Why it matters:** because the result is finite (not literal `inf`),
`clean_dataset()`'s inf→NaN pass never catches it — this silently reaches the
model as a legitimate-looking feature value. Even after `RobustScaler`
(median/IQR — robust for *centering*, but still a linear transform with no
clipping), these rows keep quadrillion-scale feature values. Fed into a
shared-weight policy network processing many tickers' feature vectors, a
column that occasionally reads 1e15 while the rest of the row is O(1) is the
kind of thing that produces gradient explosion / NaN loss with no obvious
data-side symptom — a plausible silent contributor to training instability.

**Root cause:** the epsilon-guard pattern (`x / (y + eps)`) is correct when
`y` near zero is itself a *meaningful, rare* signal (see finding 7 below for
6 other ratios that legitimately work this way, per `CLAUDE.md`'s existing
"kept intact — denominators near zero are valid distress signals" policy).
It's wrong here because "no dividend paid" is the *ordinary* state for a
large, well-defined subpopulation (growth companies with no payout policy),
not distress — "coverage" is undefined without a dividend to cover, not
infinite.

**Fix:** `src/build_dataset/features.py` — `dividend_coverage_ratio` is now
`NaN` when `div_value_recent * shares_outstanding` isn't strictly positive,
instead of dividing by `1e-8`.

**Test:** `tests/build_dataset/test_features.py::test_dividend_coverage_ratio_nan_when_no_dividend`.

### 2. WARNING — `div_yield_12m` / `div_count_12m` window was 252 *calendar* days (~8.3 months), not 12 — FIXED

`features.py::compute_dividend_features()` used
`window = np.timedelta64(252, "D")` — 252 calendar days, not a trading-day
row count. Every other "252" in this codebase (`return_12m`, `volatility_*`,
percentile windows) is a **row count** over daily trading rows, where 252
rows ≈ 1 calendar year is the correct convention. This function instead does
a calendar-date `searchsorted` over `ex_date` — a fundamentally different
mechanism where 252 calendar days is ~3.7 months short of a year. Confirmed
directly: a dividend paid 300 days ago (well within a real trailing year)
was excluded, reading `div_yield_12m = 0` / `div_count_12m = 0` for a stock
that clearly still had a trailing dividend.

**Impact:** for Brazilian issuers that pay dividends once or twice a year (a
common pattern — annual dividend + JCP), this created a spurious sawtooth:
`div_yield_12m` correctly reflects the payment for ~8 months, then reads a
false zero for the remaining ~4 months of every single year, systematically
understating trailing yield and corrupting `div_yield_sector_percentile` /
any downstream feature built on it.

**Fix:** window changed to `np.timedelta64(365, "D")`.

**Test:** `tests/build_dataset/test_features.py::test_div_yield_12m_window_is_calendar_year_not_252_days`.

### 3. Documented (not fixed — correct as designed downstream) — `status` is a current-day snapshot, not point-in-time

Confirmed empirically: `status` (ATIVO/CANCELADA) is **100% constant across
every ticker's full history** in the dataset (0/515 tickers vary). This is
because `merge_company_info()` joins company_info's *current* status onto
every historical row — by construction, not a bug in the join logic. But it
means a raw-feature consumer would see, for any row from 2012, whether that
company is *still listed today* — a feature-level lookahead/survivorship
trap distinct from (and in addition to) the universe-selection-level bias
already tracked in `TOP50_UNIVERSE_VALIDATION.md` and the `diagnosis`
branch's `DIAGNOSIS_PLAN.md` (finding F1 there).

Not fixable in this repo: point-in-time universe construction downstream
(per `TOP50_UNIVERSE_VALIDATION.md` §1) needs exactly this current-status
column to identify delisted names. The fix is documentation + a visible,
non-silent regression check:
- `CLAUDE.md` — new caveat bullet under "Company info."
- `tests/build_dataset/test_universe_integrity.py` §3.5 (new, informational)
  — measures and prints the constant-per-ticker property so it stays
  discoverable, and would flag (not fail) if a future change made status
  genuinely time-varying.

### 4. Process gap — the survivorship regression guard wasn't wired into the test runner — FIXED

`test_universe_integrity.py` (survivorship floor, schema/dtype contract,
sibling correlation) is a real, working pass/fail gate — but it wasn't in
`run_all.py`'s `FAST` or `DATA` group at all, only runnable by remembering
to invoke it by hand. Its own docstring called §3.1 "a regression guard... a
future refactor... silently reintroduces the exact bias" — but a check that
only runs when manually invoked isn't a *regression* guard, since the same
"someone forgot to run it" failure mode applies to the guard itself.

**Fix:** added `tests/build_dataset/test_universe_integrity.py` to
`run_all.py`'s `DATA` group and `CLAUDE.md`'s test-group list; updated the
file's own docstring to match. Confirmed passing standalone before wiring in
(85/122 = 70% CANCELADA survival, above the 60% floor; schema contract
clean; 11 informational low-correlation sibling pairs, all previously
spot-checked as illiquid micro-caps, not mapping bugs).

### 5. Minor — `data/processed/README.md` was accidentally deleted, `CLAUDE.md` and `.gitignore` still reference it — FIXED

Commit `851b757` ("fix: filter out tickers with filling dates > 180") was a
large doc-pruning commit that deleted several superseded docs and, it
appears unintentionally, `data/processed/README.md` along with them —
`.gitignore` still has `!data/processed/README.md` carving it out as the one
tracked exception, and `CLAUDE.md` still cites it as "the one tracked
exception" documenting the this-repo-vs-`ml_agent`-branch ownership
boundary for `data/processed/`. Restored verbatim from the pre-deletion
commit (`851b757^`).

### 6. Checked, not a bug — raw-currency fundamentals bypass `scale_features.py`'s `RobustScaler`, but are normalized downstream

`market_cap`, `net_income`, `equity`, `total_assets`, `cash`, `ebitda`,
`ebit`, `net_debt`, `total_debt`, `current_assets`, `current_liabilities`,
`shares_outstanding` are absolute BRL levels spanning many orders of
magnitude across the universe (small-caps to `PETR4`/`VALE3`), and are *not*
in `scale_features.py`'s `RATIO_COLUMNS` — they hit `remainder="passthrough"`
unscaled. This looked like a real gap (a shared-weight model seeing raw
billions next to bounded [0,1]/z-scored features) until cross-checked against
the `ml_agent` branch: `src/agent/data_pipeline.py` fits its own
`StandardScaler` over the **entire** raw feature tensor (`features: [n_dates,
n_tickers, n_features] raw (unnormalized) state features`) before the model
ever sees it. Two-stage design, not a gap — this repo's `scale_features.py`
output isn't necessarily what reaches the agent at all. (Cross-checked
`RATIO_COLUMNS` against the real 141-column dataset: all 54 present, no
column-name drift either.)

### 7. Checked, not a bug — other `+1e-8`-guarded ratios are legitimate, rare distress signals

Grepped every `+1e-8`/`+1e-12` epsilon guard in `features.py` and measured
each denominator's exact-zero rate on the real dataset: `lpa` 3.1%,
`net_income` 2.9%, `net_revenue` 1.2%, `earnings_growth_yoy` 0.02%, `roe`
2.6%, `pl` 0.4% — all an order of magnitude (or two) below
`dividend_coverage_ratio`'s 27%, and all denominators where "near zero" is
itself the economically meaningful distress event (near-zero earnings,
margin, growth), matching `CLAUDE.md`'s already-documented policy of keeping
extreme ratios intact as intentional signals (the `pl` > 400,000 example).
`payout_ratio`, `revenue_per_earning`, `ebitda_margin`, `peg_ratio`,
`pvp_to_roe_ratio`, `earnings_yield` — left as-is, consistent with that
policy.

### 8. Out of scope, tracked elsewhere — universe-selection survivorship/lookahead (F1–F3)

The `diagnosis` branch's `DIAGNOSIS_PLAN.md` already has a live, measured
investigation into exactly this class of issue at the `ml_agent`/universe-
construction layer (full-sample top-50 selection using future market cap;
signal measured on the wrong universe; statistical power). Per
`TOP50_UNIVERSE_VALIDATION.md` §1, this repo's job is only to *not
pre-filter* the raw dataset to survivors — confirmed still true (finding 4's
survivorship guard). Not duplicated here.

## Checklist against the requested audit scope

| Item | Verdict |
|---|---|
| No NaN/±Inf/invalid values reach the model | **Was false** for `dividend_coverage_ratio` (finding 1) — fixed. `clean_dataset` correctly zeroes literal inf elsewhere (`test_clean.py`, `test_final_dataset.py`). |
| Feature coverage sufficient throughout training period | Confirmed via `test_final_dataset.py`'s prefix-NaN-shape rule + CAGR coverage checks; `n_quarters_available` explains window-based NaN. |
| Prices/fundamentals/macro aligned in time, no lookahead | Re-verified `merge_asof(direction="backward")` on real CVM `fundamentals_available_date` (`test_merge.py`, `test_quality_filters.py`); re-derived by hand, not just re-read. |
| Corporate actions handled correctly | `repair_unadjusted_splits` + `apply_ticker_continuity`'s adj_close reconciliation re-verified against their own tests; no unadjusted-split leaks found live. |
| Missing data handled intentionally, consistently | `has_fundamentals`/`has_dividends`/`cagr_*_defined` flags confirmed non-ambiguous; `status`/`sector` static-join caveat now documented (finding 3). |
| Dates/calendars/ticker histories correct | No weekend rows, no fabricated trading days; ticker-continuity splice boundaries re-verified with synthetic reproductions matching `test_ticker_continuity.py`. |
| No duplicate rows / inconsistent schemas | `clean_dataset` dedup + `test_universe_integrity.py`'s dtype contract, now CI-wired (finding 4). |
| Feature engineering can't generate invalid values under edge cases | Found and fixed finding 1 (silent astronomical values evading the inf check) and finding 2 (wrong window). Audited every other epsilon guard (finding 7) — clean. |
| Normalization/scaling correct for every feature | RATIO_COLUMNS verified complete against real schema; two-stage scaling architecture clarified (finding 6). |
| Training pipeline consumes every sample without runtime failures | `test_compute_features_chunked.py` proves chunked == unchunked output + a memory/time tripwire test. |
| No per-ticker/per-period silent assumption breaks | `test_ticker_continuity.py`, `test_repair.py`, `test_quality_filters.py` all use synthetic edge-case fixtures, not just the real dataset (catches logic bugs the real data happens not to trigger yet). |
| Dataset statistics reasonable/consistent across all 50 tickers | Re-read `TOP50_ML_READINESS_AUDIT.md`'s findings, spot-checked the CCTY3/adj_close-reconciliation/MRFG3 fixes are actually present in current source (`continuity.py`, `quality_filters.QUARANTINED_TICKERS`) — confirmed, not just claimed. |
| Anomalies investigated and understood | All of the above; nothing left unexplained. |

## Tests added / changed this pass

- `tests/build_dataset/test_features.py`: `test_div_yield_12m_window_is_calendar_year_not_252_days`, `test_dividend_coverage_ratio_nan_when_no_dividend`.
- `tests/build_dataset/test_universe_integrity.py`: new §3.5 (`check_status_is_static`, informational).
- `tests/run_all.py`: `test_universe_integrity.py` added to `DATA` group.
- `CLAUDE.md`: `status` caveat added; `Data` test-group list updated.
- `data/processed/README.md`: restored.

Full suite (`python tests/run_all.py --group all`) passes: **29/29** (28
pre-existing + `test_universe_integrity.py` newly wired in).

## Action required before training (not performed by this audit)

The fixes above are in source only — `data/processed/ml_dataset.parquet` on
disk still has the old, buggy `dividend_coverage_ratio` (up to 2.3e15) and
the old `div_yield_12m` window. To get a trainable dataset:

- [ ] `python -m src.build_dataset.build_ml_dataset` (rebuild — ~20-30 min, matches the chunked-pipeline log format seen in `build_ml_dataset.py`)
- [ ] `python -m src.build_dataset.scale_features` (refit scaler on the new data)
- [ ] `python tests/run_all.py --group all` (confirm the rebuild is clean)
- [ ] `python tests/build_dataset/test_top50_ml_readiness.py` (regenerate `TOP50_ML_READINESS_AUDIT.md` against the fixed data — its dividend/CAGR-adjacent stats will change)

## Recommendation

Fix the pipeline (done), rebuild, re-run the test suite — then the Top-50
dataset is ready for the first training run. Don't train against the
currently-materialized parquet; the dividend_coverage_ratio bug alone
(quadrillion-scale values across 27% of rows) is a plausible, previously
invisible source of training instability and should be re-validated with a
clean rebuild before spending compute on it.
