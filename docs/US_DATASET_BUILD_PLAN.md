# US Final Dataset — Stage 2 Build Plan

Produce `data/processed/us_ml_dataset.parquet`, the US analogue of
`data/processed/ml_dataset.parquet`, from the raw data already collected under
`data/raw/us/`. This is Phase 6's remaining two boxes in
`docs/US_EQUITIES_EXPANSION_PLAN.md`, planned out properly.

**Written 2026-07-31.** Every number below was measured against the data actually on
disk today, not estimated.

---

## 0. Headline

Stage 2 is already ~90% region-agnostic. The work is **~250 lines of new code plus a
handful of additive, default-preserving optional parameters** — the same pattern Phase 2
used to share `yf_collectors.py` between BR and US.

The hard part is not the code. It is three decisions, each of which has a measured
answer below:

| # | Decision | Measured answer |
|---|---|---|
| D1 | How big is the universe / does the build even fit in RAM? | Gate to **2,960 tickers / 15.4M rows**; the manifest read-back must be streamed regardless |
| D2 | Does BR's 180-day filing-lag gate apply? | **No — it would silently delete 27.7% of all US fundamentals rows.** Disable it |
| D3 | Where does `sector` come from? | SIC division (10 groups) derived from `sic`, keeping raw `sic_description` alongside |

---

## 1. Preconditions — collection state as of 2026-07-31 09:30

Verified live, not assumed:

- [x] **prices** — 9,593 / 10,432 tickers, 34.0M rows, finished cleanly 2026-07-30 18:13.
      The 839 missing are genuinely uncoverable (clean "no data" responses).
- [x] **fundamentals** — 8,143 tickers, 333,515 rows. The ex27-`<MULTIPLIER>` recollection
      of the 866 affected tickers **completed 2026-07-31 09:11** (PID 124528 exited; 867
      files rewritten today — confirmed by mtime, since that ad-hoc script never called
      `logging.basicConfig()` and its log stayed near-empty).
- [x] **macro** — 14 FRED series, all clean.
- [x] **company_info** — 10,432 rows (`ticker`, `cik`, `sic`, `sic_description`).
      1,336 rows have a null `sic`.
- [ ] **dividends** — **STILL RUNNING.** PID 124339, log
      `artifacts/logs/collection/collection-us_dividends_full-20260731_083307.log`.
      2,092 / 10,432 files at 09:25. Monitor task `b31st398r`.
      → *Blocks Phase D only.* Everything through Phase C can be built and tested now;
      `merge_dividends` already has a `has_dividends` 0/1 flag for exactly this
      "collected vs. confirmed zero" ambiguity, so an incomplete run degrades gracefully
      rather than corrupting anything.

All 8,143 fundamentals tickers are a strict subset of the 9,593 priced tickers — the
`filter_tickers_with_no_fundamentals` gate will cut exactly 1,450 tickers, and the
remaining panel is **29,550,180 rows** before any universe gate.

- [x] **`DTB3` collected** (3-month T-bill, daily, 1954-01-04 → 2026-07-29, 18,134 rows) as
      `risk_free_3m` in `FRED_SERIES`/`data/raw/us/macro/risk_free_3m.parquet`. See §4.2.

---

## 2. Reuse map — what already works unchanged

This table is the plan. Everything marked *works as-is* is zero code.

| Stage 2 step | US status |
|---|---|
| `load_prices` | Schema is **byte-identical** to BR (same 14 columns). Needs a `dir` param only. |
| `load_dividends` | Schema **identical**. Needs a `dir` param only. (Its BRL >1000 implausibility ceiling is currency-specific — pass it as a param or skip; a $1000+ US per-share dividend is also implausible, so leaving it is fine.) |
| `load_fundamentals` | Needs a `dir` param, `end` → `reference_date` rename, and `ticker` from the filename (US files carry `cik`, not `ticker`). |
| `load_company_info` | US table has a different shape — see D3 / §4.3. |
| `drop_orphan_prefix_rows` | Works as-is (BR ticker names in `ORPHAN_PREFIX_TICKERS` simply never match a US ticker). |
| `repair_unadjusted_splits` | **Skip.** No-ops anyway (no `corporate_events.parquet` for the US), and yfinance's `auto_adjust=True` already back-adjusts splits at the source. |
| `apply_ticker_continuity` | **Skip.** No-ops (no `ticker_continuity.json`). US renames/mergers are a known, deliberate gap for v1 — see §7. |
| `filter_tickers_with_no_fundamentals` | Works as-is. `_ticker_root()`'s trailing-digit strip is a BR share-class convention that is a near-no-op on US tickers — harmless, but worth one assertion. |
| `compute_fundamental_features` | Works as-is once `reference_date` exists. |
| `fill_missing_cagr` | Works as-is — and is **required**, not optional: `cagr_revenue_5y`/`cagr_earnings_5y` are **0% populated** in US raw fundamentals, so the backfill is the only source. |
| `attach_filing_dates` | **Skip.** BR/CVM-specific. US fundamentals already carry a real point-in-time `fundamentals_available_date` (SEC `filed`), stamped at collection. Verified: 0 rows with `end > fundamentals_available_date`. |
| `filter_excessive_filing_lag` | **Disable — see D2.** |
| `merge_prices_and_fundamentals` | Works as-is (`merge_asof(..., direction='backward')` on `fundamentals_available_date`). |
| `merge_company_info` | The `cvm_code` sibling-fill / crosswalk machinery is BR-specific → thin US variant, ~15 lines. |
| `merge_macro` | BR selic/cdi/ipca → US variant, ~30 lines. See §4.2. |
| `merge_dividends` | Works as-is. |
| `compute_price_features` | Works as-is. |
| `compute_dividend_features` | Works as-is. |
| `compute_macro_features` | Works as-is **provided the US macro merge emits columns named `selic` and `ipca`** — see D-note in §4.2. |
| `recompute_valuation_daily` | **Does not work.** Replaced — see §4.4. |
| `compute_advanced_features` | Works as-is once valuation ratios exist. |
| `compute_history_relative_features` | Works as-is. |
| `compute_cross_sectional_features` | Works as-is; needs `sector` (D3) and a benchmark series (§4.5). |
| `clean_dataset` | Works as-is. |
| `compute_features_chunked` | Works as-is (already memory-bounded, 3-pass). |
| `write_manifest` / `write_split_config` | Path-coupled **and** RAM-coupled — see §4.6. |
| `scale_features` | Path-coupled; add `columns=` to its read. |

---

## 3. The three decisions

### D1 — Universe and scale

**The problem, measured.** BR's `ml_dataset.parquet` is 1,308,104 rows × 167 columns =
488 MB. The unfiltered US panel is **29,550,180 rows** — 22.6× BR. Projected at BR's
bytes-per-row that is **~11 GB of parquet** and ~39 GB dense in RAM at float64.

This machine has **15 GB RAM (9 GB available)** and **60 GB free disk**. Two places in
`build_ml_dataset.main()` break at that size:

1. The full merged frame held resident before `compute_features_chunked()`.
2. `pd.read_parquet(OUTPUT_PATH)` at the end, to build the manifest.

`compute_features_chunked` itself is fine — it is already bounded to one 150-ticker batch,
and Pass 2's slim 11-column projection is ~2.6 GB at full scale.

**Measured universe gates** (per-ticker, computed from each ticker's own price history —
`n >= 250` rows, `median close >= $1`, plus a median dollar-volume floor):

| median dollar volume floor | tickers | rows |
|---|---|---|
| none | 6,342 | 26,668,357 |
| ≥ $100k | 4,212 | 19,755,883 |
| ≥ $500k | 3,338 | 16,890,315 |
| **≥ $1M** | **2,960** | **15,419,040** |
| ≥ $5M | 2,040 | 11,310,809 |

**Decision: the $1M gate — 2,960 tickers, 15.4M rows.** Rationale:

- It is a *quality* gate, not just a size gate. The unfiltered pool contains 927
  "Blank Checks" (SPACs), sub-dollar penny stocks, and OTC shells — noise for any ML use.
- It is computed from each ticker's **own history**, not from "is it big today", so
  formerly-liquid delisted names survive it. That matters given the survivorship-bias
  instrumentation Phases 3–4 already built.
- **Honesty caveat to record in the manifest:** a lifetime-median statistic is still a
  full-sample quantity — you could not have known in 1995 what a ticker's lifetime median
  dollar volume would be. This is a *universe-construction* gate of exactly the same class
  as BR's collection universe (`docs/TOP50_UNIVERSE_VALIDATION.md`), and point-in-time
  universe construction stays a separate downstream step, mirroring
  `build_top50_universe.py`. Do not present it as point-in-time clean.

**Even at 15.4M rows, the manifest read-back still cannot be a `pd.read_parquet`** (167
columns × 15.4M × 8B ≈ 20 GB dense). §4.6 handles this. Two cheap multipliers apply
regardless of universe choice:

- [ ] **Cast float columns to `float32` before writing.** Halves both disk and every
      transient copy. For ML features float32 is ample. One line at the end of
      `clean_dataset`'s US path — do **not** touch the BR build.

### D2 — BR's 180-day filing-lag gate must be OFF for the US

**Measured on a 400-ticker / 15,373-row sample of the real US fundamentals:**

| tier | count | p25 lag | p50 | p75 | max | % dropped by a 180d gate |
|---|---|---|---|---|---|---|
| ex27 | 122 | 83d | 88d | 90d | 241d | 0.8% |
| item6 | 584 | 72d | **425d** | **813d** | 5,190d | **54.3%** |
| xbrl | 14,667 | 38d | 47d | 246d | 4,456d | **26.8%** |
| **overall** | 15,373 | — | — | — | — | **27.7%** |

BR's gate exists because under CVM a >180-day lag signals an *unreliable late filing*. In
the US a large lag is the **normal, designed shape** of a retrospective disclosure: the
item6 tier is literally a 5-year summary table published in one 10-K, and XBRL's
2007–2008 comparatives arrive with their 2009 `filed` date. These are not late filings.

Point-in-time correctness does not depend on this gate at all — it is guaranteed by
`merge_asof` keying on `fundamentals_available_date`. What a large lag actually produces
is *stale* fundamentals, and the model already sees that directly via
`days_since_fundamental` and `fundamentals_tier`.

- [ ] Skip `filter_excessive_filing_lag` in the US build.
- [ ] **Expect, and measure, a consequence:** because merge_asof correctly refuses to show
      a fundamental before its `filed` date, the item6 tier does **not** retroactively fill
      2001–2006 for point-in-time purposes — a 2003 trade date sees nothing from a table
      published in 2006. Merged-dataset fundamentals coverage pre-2009 will therefore be
      much thinner than the raw fundamentals row counts suggest. This is correct behaviour
      (`US_EQUITIES_EXPANSION_PLAN.md` §5.2 says the same about the 2007–2008 comparatives),
      but it must be measured and stated, not discovered later as a surprise.

### D3 — `sector` from SIC

US `company_info` has `sic` + `sic_description` (**399 distinct** descriptions, 1,336 null
`sic`). BR has 51 sectors for 293 tickers. Using `sic_description` directly gives far too
many sectors-of-one, and `cross_sectional.py` NaNs those out by design — most sector
z-scores would be empty.

- [x] `sector` = **SIC division** (11 ranges incl. an unused 1800-1999 gap: Agriculture /
      Mining / Construction / Manufacturing / Transport-Communications-Electric-Gas-Sanitary
      / Wholesale / Retail / Finance-Insurance-Real Estate / Services / Public Admin), a
      12-line 2-digit range table (`SIC_DIVISIONS`/`sic_to_sector()`, `build_us_dataset.py`).
      Coarser than BR, but every group is populated.
- [x] Raw `sic_description` kept as its own column (not consumed by `sector`) so a finer
      grouping is available downstream without a rebuild.
- [x] Null/unmapped `sic` → `sector = NaN`; `cross_sectional.py` already handles this
      (groupby drops them, columns stay NaN) -- verified in `test_build_us_dataset.py`.
- [x] **Correction to the original plan text**: BR's own `LOOKAHEAD_TAINTED_COLS` does
      *not* include bare `sector` (only the 6 sector-*derived* z-score/percentile/momentum
      columns) — BR's own comment judges the raw join lower-risk than `status`. The US
      build reuses `manifest.LOOKAHEAD_TAINTED_COLS` unchanged rather than defining a
      divergent list, since the column names are identical; no new US-specific taint list
      needed.

---

## 4. What is genuinely new

Target: **one new file** (`src/build_dataset/build_us_dataset.py`) plus additive optional
params on existing modules. No new package, no abstraction layer, no region strategy class.

### 4.1 Paths and loaders — additive optional params

- [x] Added US path constants to `paths.py` (`US_PRICES_DIR`, `US_FUNDAMENTALS_DIR`,
      `US_DIVIDENDS_DIR`, `US_MACRO_DIR`, `US_COMPANY_INFO_PATH`, `US_OUTPUT_PATH`,
      `US_SPLIT_CONFIG_PATH`, `US_SCALER_DIR`).
- [x] `load_prices`/`load_fundamentals`/`load_dividends` gained an optional `dir=` param
      — zero behaviour change for BR callers (full fast suite green before/after).
      **Caveat found while wiring this up**: a bound default (`dir=PRICES_DIR`) is
      captured once at import time, so a test's `monkeypatch.setattr(module, "PRICES_DIR",
      tmp_path)` would be silently ignored — this actually broke an existing test
      (`test_loaders.py`'s dividends sanity-ceiling check) the first time through. Fixed
      by using `dir=None` + `if dir is None: dir = PRICES_DIR` inside the body instead, so
      the module global is re-read at call time. Same landmine existed for
      `write_manifest`'s `output_path`, `write_split_config`'s `path`, and
      `filter_tickers_with_no_fundamentals`'s new `known_no_fundamentals` param (all three
      have existing tests that monkeypatch the corresponding module constant) — fixed the
      same way in all three before this was caught by rerunning the full fast suite.
- [x] `load_fundamentals` normalizes the US shape in-function: `ticker` from `file.stem`
      when the column is absent (US fundamentals files carry no `ticker` at all, only
      `cik`), `end` → `reference_date` when the latter is absent. BR files already have
      both columns, so this branch never fires for BR.

### 4.2 `merge_macro_us()` — ~30 lines ✅ DONE

| BR series | US replacement | Note |
|---|---|---|
| `selic` (daily rate, %/day) | **`DTB3`** 3-month T-bill, daily, 1954→ | annual % → daily %: `((1+r/100)**(1/252)-1)*100` |
| `cdi` | *(none)* | drop the column; nothing downstream reads it (verified: only `merge_macro` mentions it) |
| `ipca` (monthly rate, %) | `cpi_sa` index level → MoM % | `pct_change()*100` |

- [ ] **`DTB3`, not `fed_funds`.** `fed_funds` is monthly (864 rows), so BR's
      `selic_trend_20d` — a literal 20-row shift on the raw daily series — would silently
      become a 20-*month* diff. `DTB3` is daily and reaches 1954, giving the trend column
      identical semantics to BR's.
- [ ] **Publication lag, same treatment as IPCA.** CPI is released ~2 weeks after month
      end; the BR code already shifts `reference_date` by `+1 month + 15 days` before the
      asof merge, precisely to stop up to ~40 days of future inflation leaking into every
      day of the reference month. Apply the identical shift to `cpi_sa`. **This is a
      no-lookahead requirement, not a nicety.** `DTB3` is same-day published — no shift.
- [ ] Emit `ipca_daily_equiv` the same way (geometric decompounding), for the same reason.

**Naming note, decided:** the US merge emits columns literally named `selic` and `ipca`.
Ugly, but it means `compute_macro_features` and `compute_advanced_features`
(`earnings_yield_vs_selic`) work with **zero edits**, and §5.5's "same column names where
the concept matches" makes a Stage 3 pretrain→finetune handoff a straight column subset.
The alternative — renaming both to `risk_free_daily`/`inflation_monthly` across BR too —
is a larger diff touching the BR dataset schema, its tests, its manifest units and its
scaler metadata, for cosmetics. Say the word if you want the rename instead; it is a
separate, mechanical change.

- [ ] *(Optional, ~2 strings)* `vix` and `term_spread_10y2y` are already collected and
      merge through the same loop for free. Left out of v1 to keep the schema BR-aligned.

### 4.3 `merge_company_info_us()` — ~15 lines ✅ DONE

Join `ticker → sic, sic_description, sector` (D3). None of the `cvm_code` sibling-fill,
CVM crosswalk fallback, or `status` inference applies. `company_siblings()` has no US
analogue (`cik` is the natural equivalent and is on the fundamentals rows, not
company_info) — out of scope for v1.

**Bug found and fixed while implementing this**: `dataset` already carries a `cik` column
by the time this runs (arrives via `merge_prices_and_fundamentals` from the per-filing
fundamentals row), and `company_info.parquet` also has its own `cik`. A naive
`df.merge(company_info, on="ticker")` would have suffixed both `cik_x`/`cik_y` instead of
erroring — silently produced a broken column pair rather than crashing. Fixed by dropping
`company_info`'s `cik` before the merge (the fundamentals-sourced one is kept, since it's
the actual CIK the filing was made under). Regression-tested in
`test_build_us_dataset.py::test_merge_company_info_us_does_not_collide_cik_columns`.

### 4.4 `compute_valuation_daily_us()` — ~30 lines, replaces `recompute_valuation_daily` ✅ DONE

**Measured on a 200-ticker sample of US raw fundamentals**, these columns are **0.0%
populated** — there was no price available at SEC-collection time:

`market_cap`, `pl`, `pvp`, `ev_ebitda`, `ev_ebit`, `p_sr`, `p_ebit`, `p_ebitda`,
`p_assets`, `net_debt_ebitda`, `ebitda`, `ebitda_margin`, `cagr_revenue_5y`,
`cagr_earnings_5y`

BR's `recompute_valuation_daily` *rescales* an existing vendor ratio by
`close / close_price`. With a NaN base and no `close_price` column at all, it produces
nothing. So the US path computes these **directly, from the daily close** — which is
strictly better than BR's re-anchoring: no vendor filing-date basis, and none of BR's
known mid-quarter-split skew.

- [x] `market_cap = close * shares_outstanding` (73.8% coverage)
- [x] `pl = market_cap / net_income` (90.2%), `pvp = market_cap / equity` (89.9%),
      `p_sr = market_cap / net_revenue` (63.8%), `p_assets = market_cap / total_assets` (88.0%),
      `p_ebit = market_cap / ebit` (58.3%),
      `ev_ebit = (market_cap + net_debt) / ebit` (net_debt 41.3%), plus `book_to_market`
      (`equity / market_cap`, needed since `compute_fundamental_features` computes it too
      early — before any price exists — so it's NaN for 100% of US rows at that stage;
      not called out in the original plan text, found while tracing the actual call order).
- [x] Own `_safe_div()` helper (not `features._safe_ratio` — same shape, but that one isn't
      exported for reuse and duplicating 4 lines was cheaper than changing its visibility).
- [x] Sets `has_fundamentals`; nothing to drop (no `close_price` on the US side).
- [x] **EBITDA family confirmed NaN** (`ebitda` still 0% raw coverage) — `ebitda_margin`/
      `ev_ebitda`/`p_ebitda`/`net_debt_ebitda` and their `*_zhist_5y` variants stay NaN.
      Deferred to Phase E as planned.
- [x] `fill_missing_cagr` reused unchanged; verified live on AAPL/GE/KO (earnings/revenue
      null counts drop substantially post-fill).
- [x] **Structural fix needed and made, not anticipated in the original plan**:
      `compute_features_chunked` (`build_ml_dataset.py`) had `recompute_valuation_daily`
      *hardcoded* inside its Pass-1 loop — reusing it unchanged for the US would have
      silently kept calling BR's re-anchoring function instead of this one. Added an
      injectable `valuation_fn=recompute_valuation_daily` parameter (default preserves BR
      behavior exactly); `build_us_dataset.main()` passes `valuation_fn=
      compute_valuation_daily_us`. Caught by actually tracing the call graph before
      wiring `main()` up, not by running it and seeing wrong output.

### 4.5 Benchmark — SPY ✅ DONE (code); accuracy at scale not yet observed

`compute_cross_sectional_features` takes a required `benchmark` series (BR: BOVA11), used
for `beta_1y` and `momentum_vs_market_*`. **SPY is already collected** (`VOO`/`^GSPC` are
not).

- [x] `SPY` threaded through identically to BOVA11 in `build_us_dataset.main()`: captured
      before the fundamentals-coverage filter drops it, run through the same
      `compute_price_features`, never a row in the output. Verified working on a 4-ticker
      (AAPL/GE/KO/SPY) mini end-to-end run — `beta_1y`/`momentum_vs_market_1m` populate
      with plausible non-null rates (§ mini-run below).
- [ ] **Known limitation, unchanged:** SPY starts 1993, US price history reaches 1962 —
      `beta_1y`/`momentum_vs_market_*` are NaN before 1993. Not fixed for v1.

### 4.6 Manifest / split config / scaler at 15M rows — ⚠️ NOT DONE, real OOM risk

**This is the one remaining blocker before Phase C can actually be run at full scale on
this machine (15GB RAM, 9GB available).** `build_us_dataset.main()` as written today ends
with `dataset = pd.read_parquet(US_OUTPUT_PATH)` (a full, dense read-back for
`write_manifest`/`write_split_config`) — at the real 2,960-ticker/15.4M-row/~190-column
scale, with columns still `float64`, that's **~23 GB dense**, comfortably over available
RAM. The 4-ticker mini-run (§ below) didn't surface this because it's far too small to hit
it. Still to do, none of it started:

- [ ] `float32` cast on the US write path (halves the 23GB estimate to ~11.5GB — likely
      still not enough alone).
- [ ] `write_manifest_from_parquet(path)` — read one column at a time
      (`pd.read_parquet(path, columns=[c])`) instead of the full frame, reusing the
      existing manifest-dict construction logic.
- [ ] `write_split_config` — needs `trade_date` only; a one-column read.
- [ ] `scale_features.main()` — pass `columns=` to its `read_parquet` when it's later
      pointed at the US dataset.
- [ ] `sync_dataset_version` — not ported at all yet (deferred; BR's version-snapshot
      shutil.copy2 logic wasn't needed to prove the pipeline works, see Phase C status).

---

## 5. Phases

Each phase ends green before the next starts, per the standing rule.

### Phase A — plumbing (no dividends dependency) ✅ DONE 2026-07-31
- [x] `DTB3` added to `FRED_SERIES` as `risk_free_3m`; `collect_macro_us()` re-run —
      18,134 rows, 1954-01-04 → 2026-07-29.
- [x] US path constants in `paths.py`.
- [x] Optional `dir=` params on the three loaders + US fundamentals normalization
      (`ticker` from filename, `end` → `reference_date`).
- [x] `tests/build_dataset/test_build_us_dataset.py` written (not a separate
      `test_us_loaders.py` — folded the loader-normalization coverage into the one new
      US test file instead, since it's small and there's no separate `build_us_loaders`
      module to mirror). Existing loader/manifest/quality-filter tests re-verified
      passing after the `dir=`/`output_path=`/`known_no_fundamentals=` param additions
      (see 4.1's bound-default bug note — this is what caught it).
- [x] `python tests/run_all.py --group fast`: **45/45 green**. `ruff check .`: clean.

### Phase B — US-specific stage functions ✅ DONE 2026-07-31
- [x] `merge_macro_us()`, `merge_company_info_us()` + SIC-division map,
      `compute_valuation_daily_us()`, `build_universe_gate()`, all in `build_us_dataset.py`.
- [x] Self-checks as real `pytest` tests (not a `_demo()` block, to match this repo's own
      convention — every other `build_dataset` module's self-checks are `test_*.py`, not
      inline demos): CPI publication-lag shift verified against hand-computed expected
      values across 3 trade dates spanning the lag boundary; `DTB3` annual→daily
      conversion round-trips to ~5% under re-annualization; `selic_trend_20d` doesn't leak
      across two tickers with disjoint calendars (same adversarial fixture shape as BR's
      own `merge_macro` test); `market_cap`/`pl`/`pvp`/`p_sr`/`p_assets`/`p_ebit`/
      `book_to_market`/`ev_ebit` all match hand-computed values; zero-denominator gives
      NaN not inf; SIC division mapping incl. null/out-of-range; universe gate's 3
      thresholds each independently tested. **9/9 pass.**
- [x] Fast tests green (45/45, see Phase A).
- [x] **Also ran a real 4-ticker (AAPL/GE/KO/SPY) mini end-to-end integration pass**
      (not in the plan originally, done because unit tests alone wouldn't catch a call-
      order/integration bug) through the actual `compute_features_chunked` with
      `valuation_fn=compute_valuation_daily_us` — full pipeline incl. dividends, cross-
      sectional, cleaning. Confirmed: 0 lookahead violations
      (`trade_date < fundamentals_available_date`), 0 inf remaining, `beta_1y` populated
      57% of rows, `pl_zscore_sector` populated (sector grouping isn't degenerate even at
      this tiny scale), 2,122 legitimate inf→NaN replacements during cleaning. This run is
      what surfaced both real bugs recorded in §4.3/§4.4 above (the cik collision and the
      hardcoded `recompute_valuation_daily`) — they did not show up in the unit tests
      above, only in the integration pass.

### Phase C — first build, universe-gated — CODE WRITTEN, NOT RUN AT FULL SCALE
- [x] `build_universe_gate()` implemented and unit-tested (thresholds verified
      independently: min_rows, min_median_close, min_median_dollar_volume).
- [x] `main()` fully wired: loaders → universe gate → SPY capture → coverage filter →
      fundamental features → CAGR fill → merges (`merge_prices_and_fundamentals`/
      `merge_company_info_us`/`merge_macro_us`/`merge_dividends`, all reused except the
      `_us` ones) → `compute_features_chunked(..., valuation_fn=compute_valuation_daily_us)`
      → manifest/split-config write. **Skipped by design, confirmed absent from the call
      graph**: split repair, ticker continuity, `attach_filing_dates`,
      `filter_excessive_filing_lag`.
- [ ] **NOT yet run at the real 2,960-ticker/15.4M-row scale** — deliberately, per explicit
      instruction (2026-07-31): the dividends collection job (PID 124339) was still running
      when this was built, and §4.6's memory-safety work (float32 cast + streaming
      manifest read-back) isn't done yet either, which on its own risks OOM on this
      machine's 15GB RAM regardless of dividends. Do not run `python -m
      src.build_dataset.build_us_dataset` for real until: (1) dividends finishes
      (`ls data/raw/us/dividends | wc -l` reaches 10,432, or close to it net of genuinely
      uncoverable tickers), AND (2) §4.6 is done.
- [ ] Expect ~15.4M rows and a build noticeably longer than BR's, once run.

### Phase D — dividends (blocked on PID 124339)
- [ ] Confirm the dividends job finished; re-run the build with `merge_dividends` +
      `compute_dividend_features` active.
- [ ] Confirm `has_dividends` correctly separates "never collected" from "confirmed zero".

### Phase E — deferred, opt-in
- [ ] EBITDA: add `DepreciationDepletionAndAmortization` to `CONCEPT_MAP`, re-collect,
      unlock the 8 NaN'd ebitda-family columns.
- [ ] `^GSPC` splice for pre-1993 benchmark coverage.
- [ ] US ticker continuity (renames/mergers).
- [ ] Point-in-time US universe builder, the analogue of `build_top50_universe.py`.

---

## 6. Gates — what must be true before this is called done

Ported from `tests/build_dataset/test_final_dataset.py`, as
`tests/build_dataset/test_us_final_dataset.py`:

- [ ] **No lookahead.** No row where `trade_date < fundamentals_available_date`. The
      already-verified `end <= fundamentals_available_date` invariant holds in raw.
- [ ] **No macro lookahead.** No row sees a CPI reading before its publication date.
- [ ] **Prefix-shaped NaN.** No interior NaN holes per ticker in merged fundamentals.
- [ ] **Rolling-window percentiles** (`volatility_*_percentile`) use rolling rank, not
      global rank — inherited unchanged, but assert it.
- [ ] **Row/ticker counts** match the universe gate: 2,960 tickers, ~15.4M rows.
- [ ] **Pre-2009 coverage measured and written down** (D2's consequence).
- [ ] **Manifest** records: the universe gate and its non-point-in-time caveat, the
      disabled filing-lag gate, the 8 NaN ebitda columns, SPY's 1993 floor, and the 7
      `LOOKAHEAD_TAINTED_COLS` (`sector` + 6 sector-derived).
- [ ] `ruff check .` clean; `python tests/run_all.py --group fast` green.

---

## 7. Explicitly not doing (and why)

- **No shared BR/US abstraction layer.** No `Region` class, no config-driven pipeline, no
  strategy objects. Two call sites and a handful of optional params is a smaller, more
  readable diff than any framework that would unify them — and the BR path stays provably
  untouched.
- **No merging US rows into `ml_dataset.parquet`.** Per §5.5: different currency, calendar,
  accounting regime, macro block.
- **No US split repair.** yfinance already adjusts splits; there is no US
  `corporate_events.parquet` to match against, and BR's own experience (three separate
  attempts at a split-matcher persistence guard, all reverted) says do not build a detector
  without evidence of a real misfire.
- **No US ticker continuity for v1.** Real gap — a renamed US ticker will appear as two
  truncated series. Deferred to Phase E rather than hand-curating a continuity map for
  2,960 tickers speculatively.
- **No point-in-time universe in this dataset.** Same split of responsibilities as BR: the
  dataset is the full gated panel, universe construction is a separate downstream filter.
