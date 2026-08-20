# Data Layer Follow-Up Findings

Surfaced 2026-08-20 while verifying `DATA_LAYER_CORRECTNESS_PLAN.md` §1's currency-unit migration
against the real rebuilt `ml_dataset.parquet` (`tests/run_all.py --group data`). **None of these are
caused by §1's own scope (currency units)** — each is a different, pre-existing subsystem that
migration's verification happened to shine a light on. Not scheduled; pick up individually when
someone is already in the relevant area. Two items (§2a, §2b) already have a home in the correctness
plan and are only cross-referenced here, not duplicated.

---

## Shares outstanding / splits

- [ ] **`market_cap/shares == close` fails on 19/549 BR tickers** (worst: TIMS3, ratio ~99x,
      n=1979 — `close` ranges 0.000038 → 18.79 across TIMS3's full history, consistent with a real
      stock split/consolidation not being reflected consistently between `shares_outstanding`
      (fundamentals-sourced) and `close` (price-sourced)). `market_cap`, `shares_outstanding` and
      `close` were never touched by §1's currency-unit fix, so this predates that work. Needs the
      same per-event verification rigor as `ticker_continuity.json`, not a quick patch. Measured via
      `tests/build_dataset/test_unit_scale_invariants.py`'s `market_cap/shares == close` check.

## Fundamentals coverage

- [ ] **`cagr_revenue` NaN coverage dropped to 78.1% explained** (was implicitly higher before;
      `test_final_dataset.py`'s 80% threshold now fails, "21.9% unattributed"). Real, likely
      permanent side effect of migrating the full crosswalk to CVM: `cvm/ratios.py`'s
      `compute_ratios` always emits NaN for `cagr_revenue_5y`/`cagr_earnings_5y` (CVM never
      computes CAGR — Stage 2's `fill_missing_cagr()` backfill from earnings/revenue history is the
      only source now), whereas the 115 previously-BolsAI-sourced holdout tickers used to carry
      BolsAI's own raw CAGR figures into that coverage number before §1's migration moved them to
      CVM. A real tradeoff of that migration, not a bug — needs a decision (loosen the test's
      threshold with a documented reason, or find another CAGR source), not a silent fix.
- [ ] **4 tickers have exactly one single-day interior NaN hole** across
      `equity`/`net_income`/`total_assets` simultaneously: AZEV3 (2020-03-23), AZEV4 (2019-11-25),
      INEP3 (2018-07-02), RPMG3 (2014-06-02). `test_final_dataset.py`'s prefix-NaN rule flags this
      as a "suspicious merge bug" (all three columns, not a partial gap). Dates are unrelated to
      each other and to today's migration (2026-08-20) — looks like a narrow, pre-existing
      single-row merge/forward-fill edge case, not systemic. Worth a root-cause pass, not urgent
      (4 tickers, 1 row each).
- [ ] **`pl` frozen within-quarter on 277/19968 (1.39%)** — `test_final_dataset.py`'s regression
      guard for `recompute_valuation_daily()` allows <1%; this is just over. Not investigated
      further; likely a handful of tickers with sparse trading days per quarter rather than a
      re-anchoring regression (unrelated to §1 — `pl` is scale-invariant under §1's fix).

## Price / `adj_close` data quality

- [ ] **CAMB3 has 1 row with NaN `open/high/low/close`** (raw price data, not just `adj_close`) —
      also why `test_final_dataset.py`'s "no NaN in close" check fails (same single row). Found via
      `test_br_data_quality.py`.
- [ ] **LUXM4 has 289 rows with non-positive `adj_*`** — this is the CLAUDE.md-documented
      2-decimal-precision underflow ticker (`adj_close_precision_degraded`); flagged here only
      because `test_br_data_quality.py`'s hard validator still trips on it. Consistent with §2a's
      "flag only, no repair" decision, not a new issue.
- [ ] **`test_top50_ml_readiness.py`: 9/50 tickers NOT READY** — PETR4, ITUB4, SBSP3, BBDC4, BBAS3,
      ITSA4, GGBR4, GOAU4, VIVT3. All for `adj_close` discontinuities without a matching raw-close
      move, or large single-day `log_return` moves not matched to a recorded corporate event (e.g.
      PETR4: 2012-02-22/23 and 2020-11-20/23, `close` moves <1% but `adj_close` jumps ±70-270%).
      Price-adjustment / corporate-events domain (`repair.py`, `corporate_events`), untouched by
      §1. The generated report `TOP50_ML_READINESS_AUDIT.md` (repo root, untracked) has full detail
      per ticker.

## Already tracked elsewhere — cross-referenced, not duplicated here

- **§2a** (NaN `adj_close`, 259 rows, "flag only, no repair" already decided by owner) —
  `DATA_LAYER_CORRECTNESS_PLAN.md` §2a. `test_top_traded_quality.py`'s "2 NaN adj_close" finding
  and the LUXM4 finding above are both instances of this same, already-decided territory.
- **§2b** (98 of 202 in-panel deaths have no terminal event — research task, not an implementation
  step) — `DATA_LAYER_CORRECTNESS_PLAN.md` §2b. `test_raw_processed_reconciliation.py`'s "12
  uncovered dead tickers" (e.g. `OIBR3/4`, `RSID3`, `BDLL3/4`) is the same territory, surfaced by a
  different test.
