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
- [x] **dividends** — **COMPLETE** (corrected 2026-07-31, see §8.4 — the original "2,092/10,432
      at 09:25, still running" line above was superseded, and the file-count-vs-10,432 completion
      criterion it implied was wrong to begin with: a file only exists for a ticker that has ever
      *paid* a dividend, and most of the 10,432 crosswalk tickers never do). Confirmed done: the
      `us_full_scale_v2` checkpoint holds 4,209 entries with current 2026-05/06/07 `last_date`s,
      4,214 files on disk, and the last run walked all 10,432 tickers writing 0 new rows before
      exiting cleanly. Phase D is unblocked.

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

### Phase C.5 — no-lookahead audit ✅ DONE 2026-07-31, clean

Requested explicitly before any full-scale run. Covered both the raw data (independent of
any of this repo's code) and the builder code path, not just a repeat of Phase B's unit
tests.

- [x] **Full-corpus scan, all 8,143 fundamentals files / 333,515 rows** (not a sample):
      `end > fundamentals_available_date` — **0 violations**. Null
      `fundamentals_available_date` — **0**. Duplicate `end` per ticker (would corrupt
      YoY/QoQ derivation) — **0**.
- [x] **Investigated an initially alarming finding**: `fundamentals_available_date` is
      NOT monotonically increasing with fiscal `end` within a ticker — true for 6,931/8,143
      tickers (85%), 30,863 row-pairs. Traced to source (`sec/companyfacts.py:380`,
      `extract_line_items`): a row's `fundamentals_available_date` is deliberately the
      **MAX across each of its constituent line items' own `filed` date** — e.g. a balance-
      sheet item can be tagged as a prior-period comparative in a later filing before that
      quarter's income-statement figure is filed elsewhere, so the row waits for the
      *last* piece before being considered available at all. Confirmed via inline
      docstring: "the conservative (never-early) bundling date... guarantees no single
      item in the row is ever exposed before it was genuinely public." Same reasoning
      confirmed in `_derive_q4` (line 227: a derived Q4 keeps the FY total's own `filed`
      date, since the Q4 figure isn't computable before the FY total itself was filed).
      **Net effect: this can only make a row available LATER than a naive per-item view
      would suggest, never earlier — safe, and a separate (already-known) staleness
      characteristic, not a lookahead bug.**
- [x] **Macro data** (`risk_free_3m.parquet` 18,134 rows, `cpi_sa.parquet` 953 rows): 0
      nulls, 0 duplicate `reference_date`, both monotonic, plausible value ranges.
- [x] **CPI publication-lag shift verified against REAL data** (not just the synthetic
      Phase B fixture): recomputed the availability-respecting `ipca` independently and
      diffed against `merge_macro_us`'s actual output across 16,250 real (trade_date,
      ipca) pairs — **0 mismatches**.
- [x] **Broader real-ticker integration run** (AAPL, A, WMT, TXT, ZION, BOOM, GE, KO, SWK,
      SPY — deliberately including tickers that were the SUBJECT of prior real bugs:
      WMT's ex27 multiplier fix, ZION/BOOM's item6 table-selection fix, A's mixed-tier/
      off-calendar (Oct 31) fiscal year-end) through the full pipeline incl.
      `compute_features_chunked`: **0 rows with `trade_date <= fundamentals_available_date`**,
      checked both immediately post-`merge_asof` and in the final fully-featured output.
- [x] **Dtype safety check**: `fundamentals_available_date`/`reference_date` stay
      `datetime64[ns]` from raw parquet through `load_fundamentals()` — no silent
      string-comparison risk in the asof merge.
- [x] **Dividends data quality** (2,868 files collected so far): 0 null `ex_date`, 0 rows
      with `ex_date` after `payment_date`. Found 255 rows (7 tickers: ABEO, AEHL, CMCT,
      PSHG, SHIP, SUNE, SVRN — small-cap/penny-stock names) with `value_per_share` > 1000,
      up to $2.1M/share (SUNE) — clearly vendor data corruption, not real dividends.
      **Not a new problem**: `loaders.load_dividends()`'s existing >1000 sanity ceiling
      (originally written for a BRL vendor bug) already catches and drops these for the
      US too, since the loader is region-agnostic. Confirms the guard generalizes
      correctly; nothing to fix.

**The one real, already-documented caveat, restated for clarity**: `build_universe_gate()`
computes each ticker's *lifetime* median close/dollar-volume to decide inclusion — this is
NOT point-in-time (a 1995 row from a ticker that only becomes liquid decades later still
qualifies), the same category of universe-selection bias as BR's `build_top50_universe.py`.
This is conceptually distinct from a per-row feature lookahead (no future VALUE appears in
any row's own feature columns, and every check above confirms that) — it affects which
TICKERS are in the panel at all, not what any given row can see. Already called out in
§D1/§7; restated here because it's the one place a lookahead-*adjacent* concept genuinely
applies, and the audit would be incomplete without naming it explicitly.

**Conclusion: no feature-level or fundamentals-merge lookahead found**, across both the
full raw corpus and a targeted adversarial integration run. Phase C's real blockers remain
exactly the two already listed (dividends completion, §4.6 memory safety) — this audit
does not surface a new one.

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

---

## 8. Fix plan — 2026-07-31 review

Re-checked the three things standing between here and a real Phase C run, against live data.
One is a new bug, one is the known §4.6 work, one turned out to be a false alarm.

### 8.0 The build dies at line 285, not at the manifest — MEASURED, blocks everything

§4.6 calls the manifest read-back "the one remaining blocker before Phase C can actually be run."
**That is wrong, and §8.2 below inherited the error.** Measured against the real files today, the
build OOMs twice *before* it ever reaches the manifest — and the first time, it has written
nothing to disk at all.

**Failure 1 — `load_prices()` (`build_us_dataset.py:285`), the first statement in `main()`.**
The universe gate runs on line 289, *after* the load. So `load_prices` reads **all 9,593 tickers /
34,026,021 rows**, not the gated 2,960 / 15.4M. Measured at **157 B/row** (deep, incl. the object
`ticker` column) → **5.3 GB** for the result, and `load_prices` holds the 9,593-frame `dfs` list,
the `pd.concat` result, and the `sort_values` copy live simultaneously → **~16 GB peak**. Machine
has ~8 GB available. The gate cannot rescue a load that already died.

**Failure 2 — the merged frame, resident in `main()` before `compute_features_chunked`.**
D1 listed this as break point (1) and then only §4.6-fixed break point (2). Measured: US
fundamentals carry **71 columns — 49 numeric, 15 datetime, 7 object** — which once forward-filled
onto the daily panel costs **~1,002 B/row**. At the gated 15.4M rows that is **15.4 GB for the
fundamentals block alone, ~17.9 GB with prices**, before a single feature is computed.

**The saving grace: almost all of that width is collection-time provenance, not features.**

| what | cols | B/row | keep? |
|---|---|---|---|
| `*_filed` per-line-item dates | 13 | 104 | **drop** — only used at collection time to derive `fundamentals_available_date` (the MAX across items, Phase C.5); nothing in Stage 2 reads them |
| `item6_filename`, `fds_filename`, `fds_article`, `fds_form`, `item6_form`, `fds_multiplier_explicit` | 6 | 420 | **drop** — pure provenance, ≤49 unique values each, `item6_filename`/`fds_filename` average 42 chars |
| `fundamentals_tier` | 1 | 70 | **keep as `category`** (~1 B/row) — D2 relies on the model seeing it |
| numeric | 49 | 392 | **`float32`** → 196 |

- [x] **Gate before the full load — DONE 2026-07-31.** `build_universe_gate_from_files(dir)`
      (`build_us_dataset.py`) reads only `close`/`volume` per file, no concat; `load_prices` gained
      a `tickers=` filter (default `None` = load everything, zero behavior change for BR).
      `main()` now gates before loading. Measured against the real files: gate alone finds
      **3,134/9,593** qualifying tickers — **not a drift from the plan's 2,960, verified**: the
      gate now necessarily runs before `filter_tickers_with_no_fundamentals` (it has to — that
      filter needs fundamentals loaded first, which is exactly what this fix avoids doing
      upfront), so it sees all 9,593 priced tickers, not just the 8,143 with fundamentals
      coverage. Restricting the same gate to the fundamentals-covered subset reproduces **exactly
      2,960**; the other 174 are liquid price-only tickers with zero fundamentals files (mostly
      closed-end funds/ADRs — ACP, ADX, AIO, AWF, BAESY, ...) that `filter_tickers_with_no_fundamentals`
      still drops one step later, same as before this fix. Loading the gated 3,134-ticker set +
      fundamentals peaked at **5.5 GB RSS** (was ~16 GB for the old full-then-filter order) — well
      inside the ~8 GB available.
- [x] **Trim fundamentals width at load — DONE.** `load_fundamentals` drops the `*_filed`/
      `item6_*`/`fds_*` provenance columns unconditionally (confirmed absent from BR, a no-op
      there) and gained `optimize_dtypes=False` (BR keeps float64 exactly; US passes `True` →
      numeric columns to `float32`, `fundamentals_tier` to `category`, `cik` excluded — an
      identifier, not a value).
- [ ] **Skipped: `ticker` → `category` on prices.** ~0.9 GB of the projected savings, but touches
      `.str`/`.groupby`/`.merge` call sites shared with BR across several modules — real risk for a
      win that turned out to be unnecessary once the two items above landed (5.5 GB measured, not
      the ~5.9 GB projected). Add only if a real Phase C run still runs tight on memory.

**Correction to §8.2's float32 call:** I dismissed float32 there on the grounds that it doesn't
rescue the *final read-back* (11.7 GB still > 8.7 GB free). That reasoning holds for the read-back
and nowhere else — for the merged frame above, float32 is one of the three levers that make the
build possible at all. Dropping it wholesale was wrong.

**A third bug, found while verifying this fix (not in the original measurement): `market_cap` and
`ebitda` aren't just NaN in the raw US fundamentals — confirmed 0% populated across the FULL
8,143-file corpus (every single file, zero exceptions), which means `load_fundamentals`'s existing
per-file `dropna(axis=1, how="all")` (pre-existing code, unrelated to this fix) drops both columns
*entirely*, not merely NaN-fills them.** Four unguarded reads assumed the columns would at least
exist: `features.py`'s `compute_fundamental_features` (`book_to_market`, `ebitda_growth_yoy`) and
`compute_advanced_features` (`dividend_coverage_ratio`, `ebitda_margin`) — all four `KeyError`'d
the instant a real, gated US load reached them (confirmed by actually running the merge on a small
real ticker sample, not just unit fixtures). Fixed with `if "market_cap"/"ebitda" in df.columns`
guards at each site (a no-op for BR, which always has real data in both) — root-cause-sized, since
these are 4 independent feature computations with no single shared choke point to patch instead.
Regression-tested in `test_features.py`. **Not caught by Phase B/C.5's mini/adversarial runs** —
those used tiny hand-picked ticker sets and never hit this combination of columns at the real
gated scale.

**Verification note (superseded, see §8.0.1):** the full 3,134-ticker `merge_prices_and_fundamentals`
loop was NOT run to completion interactively in this session — a first attempt was backgrounded
past its timeout and the sustained memory pressure (on top of everything else already running)
crashed the user's VS Code. The load stage (5.5 GB peak) and the crash fix above were both
confirmed correct on a small real-ticker sample (AAPL/GE/KO/A/WMT/ACN/ADI/SPY) instead. Turned out
this caution was warranted: the user ran the build themselves shortly after and hit a REAL kernel
OOM-kill inside this exact function — see §8.0.1, a 4th bug, found and fixed the same day.

### 8.0.1 `merge_prices_and_fundamentals` OOM-killed for real — FOUND + FIXED 2026-07-31

The user ran `python -m src.build_dataset.build_us_dataset` directly (not backgrounded) and it
died: `... Merging ZWS / Merging ZYME / Killed`. `journalctl -k` confirmed a real kernel OOM-kill,
~10 GB anon-rss, right after the last (alphabetically) ticker's merge — i.e. inside
`merge_prices_and_fundamentals` (`merge.py`), which §8.0's analysis never covered (it measured the
*load* stage only, never ran the merge to completion).

**Root cause:** the function looped per ticker (3,134 iterations for the real US universe),
accumulating a Python list of per-ticker merged frames, then did one `pd.concat` at the end — an
unbounded "hold everything then concat" pattern, unlike `compute_features_chunked`'s Pass 1, which
deliberately streams to a temp parquet for exactly this reason. Fine at BR's ~500 tickers; not at
US's 3,134.

**Fix:** replaced the loop with a single `pd.merge_asof(..., by="ticker")` call — pandas' own
grouped asof-join, not custom logic (ladder rung 5). No accumulation, no per-ticker `.get(ticker,
empty)` fallback either (a grouped asof naturally NaNs a ticker absent from the right frame).

**Two real correctness bugs found and fixed while building this, both caught before shipping —
not found by unit tests with hand-picked fixtures, only by adversarial/real-data testing:**

1. `merge_asof(by=...)` requires its "on" column sorted **globally** across the whole frame, not
   just within each `by`-group — confirmed via a minimal pandas repro. Sorting by `[ticker, on_col]`
   raises `"keys must be sorted"` the instant two tickers' date ranges interleave (the normal case
   at real scale) — it only looked fine in a first pass because the existing hand-picked test
   fixtures' per-ticker ranges happened not to overlap. Fixed: sort by the "on" column alone;
   `by="ticker"` still correctly isolates each ticker's matches (verified directly). Added
   `test_merge_survives_interleaved_ticker_date_ranges` (5 tickers, deliberately interleaved) as a
   lasting regression guard — every other existing test in `test_merge.py` uses ≤2 tickers whose
   ranges don't overlap, which is exactly what let the first version of this bug through.
2. **105 real `(ticker, fundamentals_available_date)` duplicate pairs exist in the actual BR data**
   (e.g. AALR3: two quarters, `2016-03-31` and `2016-06-30`, both received the same day —
   `2016-10-28`, a late catch-up filing). With `direction="backward"`, a tie in the "on" key
   resolves to whichever duplicate sorts last — and pandas' default sort is not stable, so sorting
   by `fundamentals_available_date` alone left that pick effectively arbitrary. The OLD
   per-ticker-loop implementation happened to get this right by accident: it sorted by
   `reference_date` first, then re-sorted (stably) by `fundamentals_available_date`, so the later
   quarter always won ties. **Found by an old-vs-new empirical diff against the FULL real BR
   dataset** (2,128,541 rows), not a unit test — 145+ rows differed across dozens of fundamentals
   columns on an initial 60-ticker sample. Fixed: sort by
   `["fundamentals_available_date", "reference_date"]` (the later quarter wins ties, matching the
   old behavior exactly).

**Verification, this time to completion:**
- Old-vs-new implementations diffed row-for-row against the **entire real BR universe**
  (2,128,541 rows, all 77 columns, `merge_prices_and_fundamentals` run standalone on real
  `load_prices()`/`load_fundamentals()` output): **0 real mismatches** after both fixes (an
  `inf == inf` false-positive in the diff script itself was the only thing separating "0" from a
  misleading "32 columns differ" — fixed the comparison, not the code, for that one).
- `test_merge.py`: 11/11 pass, including the 3 new regression tests above.
- Real US data, 800 tickers (safely below the full 3,134 given the crash history) through load +
  merge: **4,145,648 rows, 72 columns, 5.9 GB peak RSS** — comfortably under the ~10 GB available,
  and barely above the load-only stage's own 5.5 GB, i.e. the merge itself isn't adding a second
  multiplier on top of §8.0's fix.
- Full fast suite (47/47) and data-group suite (12/12) green, `ruff` clean.

**Still not run to full 3,134-ticker completion** — deliberately, given the crash history. The
800-ticker measurement is a strong signal, not a guarantee. Launch the real attempt
`nohup ... & disown`, per [[feedback_nohup_background_jobs]].

**The 800-ticker signal turned out not to hold — real kernel OOM confirmed at full scale,
FIXED 2026-07-31 (§8.0.2 below).** The user ran the real build; it died with `Killed` right at
`MERGING PRICES + FUNDAMENTALS`'s own header, before printing even one row count.

### 8.0.2 Merge batched per-ticker-chunk — FIXED 2026-07-31

The 800/3,134-ticker extrapolation in §8.0.1 was directionally right and should have been
trusted less: 5.9 GB at 4,145,648 rows extrapolates linearly to **~21.9 GB at the full 15.4M-row
scale** — comfortably past this machine's ~9-10GB available, which is exactly what killed the
real run. `compute_features_chunked`'s Pass 1 was already ticker-batched for feature computation,
but the MERGE that fed it (`merge_prices_and_fundamentals` → `merge_company_info_us` →
`merge_macro_us` → `merge_dividends`) still ran once over the full universe first, building the
entire wide (fundamentals-width-forward-filled) frame before any batching began.

**Fix:** `compute_features_chunked` (`build_ml_dataset.py`) gained optional `tickers=`/`batch_fn=`
params — when `batch_fn` is given, Pass 1 calls it per ticker-batch instead of slicing a
pre-merged `dataset` (default `batch_fn=None` is byte-identical to the old path; BR's call in
`build_ml_dataset.main()` is untouched). `build_us_dataset.py` gained `make_merge_batch_fn()`,
which does all 4 merges scoped to one ~150-ticker batch at a time — none of the 4 are
cross-sectional (unlike Pass 2's `compute_cross_sectional_features`, which genuinely needs the
whole universe), so batching them changes nothing about correctness, only how much is resident at
once. `main()` now passes `batch_fn`/`tickers` instead of pre-building `dataset`; the raw
prices/fundamentals/company_info/dividends tables (the narrow, ~5.5GB, pre-merge form) stay
resident for the whole call — only the wide merged product is ever bounded to one batch
(measured: ~1.1GB per 150-ticker batch by the same per-row extrapolation, comfortably inside the
budget that killed the unbatched version).

Verified: `test_batch_fn_path_matches_dataset_slicing_path` (`test_compute_features_chunked.py`)
— the new plumbing produces byte-identical output to the old dataset-slicing path on the same
data. `test_make_merge_batch_fn_matches_unbatched_merge` (`test_build_us_dataset.py`) — running
the 4 merges batch-by-batch produces the exact same rows as running them once on the whole
universe. Full fast suite (47/47) green, `ruff check` clean.

**Still not run to full 3,134-ticker completion** — this fixes the measured cause of the real
crash, but hasn't itself been observed at full scale yet. Launch `nohup ... & disown`, watch RSS
across the run (Pass 2/§8.3's cross-sectional stage is the next unmeasured risk — it still holds
the full-universe slim projection at once, by design, since it needs the whole market for
sector/beta stats; that part was always meant to be full-universe-resident and is much narrower
per row than the fundamentals-merge that just got fixed).

### 8.1 CAGR is silently 100% NaN for 8.6% of US tickers — NEW BUG

**Measured, full corpus:** **701 / 8,143** fundamentals tickers have *zero* rows ending in
December — A (Agilent, FYE Oct 31), ACN (Aug), ADI (Oct), AEO, ABM, ...

`cagr_handler.py` hardcodes the fiscal-year anchor as `reference_date.dt.month == 12`, in both
`calc_annual_cagr` and `had_negative_base`. Correct for BR (CVM mandates a December FYE); wrong
for the US. Because raw `cagr_earnings_5y`/`cagr_revenue_5y` are **0% populated** for US
fundamentals (§4.4), the calculated fallback is the *only* source — so for those 701 tickers
`cagr_{earnings,revenue}_5y_final` is NaN across their entire history, and `cagr_*_defined`
dutifully reads 0 everywhere. Nothing crashes, nothing flags it.

Phase C.5's adversarial integration run *included* `A` and still missed this: that audit only
checked for lookahead violations, never CAGR fill coverage.

**Fix (~6 lines, BR output bit-identical) — DONE 2026-07-31:** threaded an `anchor_month` parameter.

- [x] `_december_periods(df)` → `_anchor_periods(df, anchor_month)`; `is_anchor` is
      `dt.month == anchor_month`, or all-True when `anchor_month is None`.
- [x] Same substitution in `calc_annual_cagr` and `had_negative_base` (both already build their
      own local `is_december`).
- [x] Threaded `anchor_month=12` through `fill_cagr_columns` → `features.fill_missing_cagr`;
      `build_us_dataset.main()` passes `anchor_month=None`.
- [x] Two tests added (`test_build_us_dataset.py`): Dec-FYE fixture output unchanged (byte-identical
      to the pre-fix formula), Oct-FYE fixture (Agilent's real cadence) goes from 100% NaN to fully
      populated at ~5%/year. Also re-ran `test_cagr_calculation.py` against 4 real BR tickers
      (PETR4/VALE3/ITUB4/WEGE3) to confirm the default path is untouched.
- [x] `python tests/run_all.py --group fast`: 47/47 green. `ruff check`: clean.

`anchor_month=None` (every quarter is an anchor → 5y CAGR that updates quarterly) rather than
per-ticker FYE detection, for two reasons: the existing `_december_periods` cumsum already
degenerates to one group per row when every row is an anchor, so the broadcast logic needs *no*
extra code; and there is **no reliable FYE signal to detect from** — `fiscal_year` is not even a
column in every fundamentals file (confirmed: the schema genuinely differs per tier;
`load_fundamentals`' `pd.concat(sort=False)` is what absorbs that today).

Cost: US CAGR updates quarterly where BR's updates annually. Say the word if matching BR's annual
cadence matters more than coverage — that's the per-ticker-FYE version, and it needs a FYE source
found first.

**Known ceiling, deliberately not fixed here:** `calc_annual_cagr` looks back 20 *rows*, not 20
real quarters — a vendor-missing quarter silently stretches the window, the same landmine
`features._within_calendar_gap` already guards YoY/QoQ against. Pre-existing for BR too. Add the
guard only if a gap audit shows it actually bites.

### 8.2 Manifest read-back will OOM at full scale — FIXED 2026-07-31

`main()` used to end with a dense `pd.read_parquet(US_OUTPUT_PATH)` → ~20–23 GB at 15.4M rows ×
~190 columns. Machine: 15 GB total, ~8 GB available.

- [x] `write_split_config`: call site now passes
      `pd.read_parquet(US_OUTPUT_PATH, columns=["trade_date"])` — the only column it ever touches.
      Zero changes inside `write_split_config` itself.
- [x] `write_manifest` gained an optional `parquet_path=` branch (`dataset=None` now, `parquet_path`
      required in its place). When given: schema/dtypes read from the parquet footer (no rows read)
      via `pf.schema_arrow.empty_table().to_pandas().select_dtypes(...)` — reusing pandas' own
      numeric-selection so "what counts as numeric" can never diverge from the in-memory branch —
      `rows` from `ParquetFile.metadata.num_rows` (free), `tickers`/`date_min`/`date_max` from
      single-column reads, and `column_stats` one column at a time. The final manifest-dict
      construction is shared between both branches (not duplicated) — only how the pieces get
      computed differs. Default (`parquet_path=None`) is byte-identical to the original in-memory
      path; BR's existing call in `build_ml_dataset.py` is untouched.
- [x] Regression test (`test_manifest.py`): `parquet_path=` produces the identical manifest to the
      in-memory path on the same data (rows/tickers/dates/columns/tainted-cols/dropped-report/
      column_stats all compared directly).
- [x] **Skipped: the float32 cast.** Turned out unnecessary once §8.0's gate-before-load +
      fundamentals-width fixes landed (measured 5.5 GB peak through the merge stage, not the
      ~11.7-17.9 GB this section was written against).
- [ ] **Skipped: porting `sync_dataset_version`.** There is no previous US build to diff against.
      Add at the second build.

### 8.3 Undocumented second OOM risk: Pass 2 — MEASURED, hit, FIXED 2026-07-31

§4.6 only ever counted the final read-back. `compute_features_chunked`'s Pass 2 holds the slim
frame for the whole universe at once, and `compute_cross_sectional_features` then keeps roughly
three live copies of it: the `df.merge(bench)` result, plus the per-ticker `result` list *and* its
`pd.concat` in the beta loop. `ticker`/`sector` are object dtype — ~15.4M × 2 Python strings ≈
1.8 GB before any copy is made.

- [x] Ran Phase C (after §8.0.2's merge-batching fix let it get this far) — it OOM'd exactly here:
      `Killed` right after printing "COMPUTING CROSS-SECTIONAL (MARKET/SECTOR) FEATURES", i.e.
      Pass 1 completed (streamed to the temp parquet successfully) and this is where it actually
      died.
- [x] **Root cause identified without needing to instrument RSS**: the beta loop was the *exact
      same* "accumulate a full-width copy per group, `pd.concat` at the end" shape that OOM-killed
      `merge_prices_and_fundamentals` (§8.0.1) — an untouched sibling of that already-fixed bug,
      not a new failure mode. It ran on the WHOLE universe at once (unlike
      `compute_price_features`'s identically-shaped per-ticker loop, which only ever sees one
      ~150-ticker Pass-1 batch — safe there, unsafe here). By the time this loop runs, `df` carries
      ~24+ columns (the original 12-column slim projection plus every zscore/percentile/momentum
      column added earlier in this same function plus the 4 merged benchmark columns) — appending
      full `g` slices into `result` held a full SECOND copy of the entire full-universe frame
      alongside the original `df` for the loop's duration.
- [x] **Fix:** accumulate only the narrow cov/var-derived `beta_1y` Series per ticker (not full
      `g`), then assign back via `df["beta_1y"] = pd.concat(beta_parts)` (aligns by each group's
      preserved, unique row index — no `ignore_index=True`/full-frame re-concat needed). Verified
      byte-identical to the old full-width-accumulation output on a synthetic multi-ticker fixture
      before touching the real file; existing `test_beta_vs_market_matches_direct_computation`/
      `test_beta_nan_before_min_periods_then_no_lookahead` (`test_cross_sectional.py`) still pass
      unchanged (independent hand-computed references, not just a self-comparison). Full fast
      suite 47/47, `ruff check` clean.
- [ ] **Not pre-emptively done**: the `ticker`/`sector` → `category` cast. Per the plan's own
      "measure, don't pre-fix" guidance — the beta-loop fix removes the one confirmed *second*
      full-universe copy; `df` itself staying resident through Pass 2 is expected/by-design (needs
      the whole market for cross-sectional stats). Revisit only if a real run still runs tight
      after this fix.

**This fix alone was NOT enough — the build still OOM'd at the exact same point (§8.0.3 below).**
The beta-loop fix was real and necessary, but a second, independent leak (§8.0.2's own `batch_fn`
plumbing) was keeping the raw prices/fundamentals/company_info tables resident through Pass 2/3
too, on top of whatever Pass 2 itself needed.

### 8.0.3 `batch_fn`'s captured tables outlived Pass 1 — FIXED 2026-07-31

After §8.3's beta-loop fix, the user re-ran the build: Pass 1 now completed all 20 batches
cleanly (confirmed by the log reaching "Batch 20/20: 110 tickers"), but it still died with `Killed`
right at Pass 2's own header — same failure point as before, meaning §8.3 helped but wasn't
sufficient by itself.

**Root cause:** §8.0.2's fix moved the 4 merges into `_MergeBatcher`/`batch_fn`, closing over
`prices`/`fundamentals`/`company_info`/`dividends` (~3-5GB) so they'd only be needed during Pass 1.
But `build_us_dataset.main()` calls `compute_features_chunked()` *synchronously* — `main()`'s own
`batch_fn` local variable stays bound in its frame for the ENTIRE nested call (all 3 passes), by
definition of how a blocked caller's stack frame works. The first attempt at fixing this
(`del batch_fn` inside `compute_features_chunked`, right after Pass 1) turned out to be a **no-op**:
confirmed via a minimal weakref repro before shipping the real fix — deleting the CALLEE's own copy
of a reference never brings an object's refcount to 0 while the CALLER still holds its own separate
reference to that same object, which it always does here. So the closed-over tables kept living
through Pass 2/3 regardless, adding ~3GB of dead weight right where §8.3's fix had just freed up
headroom.

**Fix:** replaced the plain closure with `_MergeBatcher`, a small class holding
prices/fundamentals/company_info/dividends as attributes plus an explicit `release()` that sets
them all to `None`. `compute_features_chunked` calls `batch_fn.release()` (if present — a no-op
`getattr` default for plain callables like test lambdas) right after Pass 1. This works because
`release()` **mutates the shared object's own state**, which is visible through every reference to
it — main()'s included — unlike deleting a reference, which only affects the one binding being
deleted. `main()` still separately drops its OWN `prices`/`fundamentals`/`company_info` locals
before the call (necessary but not sufficient alone: mutating `_MergeBatcher`'s attributes doesn't
touch a caller's own separate variable pointing at the same DataFrame).

Verified: `test_batch_fn_release_actually_frees_captured_state` (`test_compute_features_chunked.py`)
uses `weakref` to confirm a `release()`-based batch_fn's captured state is actually freed even
while the test keeps its own reference to the batch_fn object alive through the call (mirroring
`main()`) — this test fails against the old (ineffective) `del batch_fn` version, confirming it's a
real regression guard, not a tautology. Full fast suite 47/47, `ruff check` clean.

### 8.4 Dividends — NOT dead, actually COMPLETE. Phase D unblocked.

`collection-us_dividends_threaded2` ran 15:00→15:37, walked **all 10,432 distinct tickers**, wrote
**0 new rows**, and exited. It did not die — it *finished*. Corroborated three ways: the checkpoint
`artifacts/checkpoints/us_full_scale_v2/yf_dividends.json` holds 4,209 entries with current
`last_date`s (2026-05/06/07); 4,214 files on disk; KO/JNJ/PG/XOM/T/MMM/PEP/CVX/IBM/WMT all present.

The two `NVRM ... Out of memory` kernel messages at 14:56 are GPU VRAM, predate the 15:00 launch,
and are unrelated — not an OOM-kill of this job.

- [ ] **Correct the completion criterion** in §1 (and in the session memory note): `ls
      data/raw/us/dividends | wc -l == 10432` is **unreachable by construction** — a file is only
      written when a ticker has *ever* paid a dividend, and most of the 10,432 (warrants `*W`,
      units `*U`, shells, plain non-payers) never do. Use the checkpoint entry count or a clean log
      exit instead.
- [ ] Drop "blocked on PID 124339" from Phase D — nothing is blocking it.

### 8.5 Cosmetic

- [ ] `close_price` survives into the US output: BR's `recompute_valuation_daily` drops it at the
      end, `compute_valuation_daily_us` doesn't. One line, or just leave it — it's a real column
      (close at filing date), merely unused.

### Order

1. **8.4** — [x] done, docs/memory corrected.
2. **8.1** — [x] CAGR fix + tests, fast suite green.
3. **8.0** — [x] gate-before-load + fundamentals width trim, **including a 3rd bug found while
   verifying it** (`market_cap`/`ebitda` absent, not just NaN — 4 crash sites fixed). Measured
   5.5 GB peak through the merge stage on the real gated universe (was ~16 GB). Full 3,134-ticker
   merge not run to completion interactively (see verification note in §8.0) — confirmed correct
   on a small real-ticker sample instead.
4. **8.2** — [x] manifest/split-config streaming, regression-tested.
5. **Run Phase C** — launch `nohup`'d/`disown`'d, not interactively (see §8.0's verification
   note), watching **8.3**.
6. Phase D + the §6 gates (`tests/build_dataset/test_us_final_dataset.py`).

**Status as of 2026-07-31 (later session): §8.0/8.1/8.2 all implemented and tested** — 47/47 fast
tests, 12/12 data-group tests (incl. real BR `ml_dataset.parquet` validation and US data-quality
checks), 24 new/updated unit tests across `test_build_us_dataset.py`/`test_loaders.py`/
`test_features.py`/`test_manifest.py`, `ruff check` clean. **Phase C itself has NOT been run at
full scale** — the remaining risk is real but unmeasured beyond the merge stage (Pass 2, §8.3;
`compute_features_chunked`'s per-batch Pass 1 arithmetic upcasting float32 back to float64, not
separately measured either). Next session should launch the real build backgrounded
(`nohup ... & disown`) and watch, not assume clean based on this session's smaller-scale checks.

**Status as of 2026-07-31 (yet later session): real OOM confirmed at full scale, fixed (§8.0.2).**
The user ran the real build and it died exactly where §8.0.1's extrapolation warned it might —
kernel-killed right at `merge_prices_and_fundamentals`, before printing a row count. Root cause:
the merge that fed Pass 1's already-batched feature loop was itself still whole-universe. Fixed
by pushing the merge into the same per-batch loop (`tickers=`/`batch_fn=` on
`compute_features_chunked`, `make_merge_batch_fn()` in `build_us_dataset.py`). 47/47 fast tests
green (2 new regression tests), `ruff check` clean. **Still not run to full 3,134-ticker
completion** — the fix is measured-consistent (≈1.1GB/batch vs. the ≈22GB the unbatched merge
would have needed), not yet observed end-to-end. Launch `nohup ... & disown`, watch RSS through
Pass 2 (§8.3, still genuinely unmeasured — the one remaining full-universe-resident stage, by
design).

## 9. Memory budgeting — the build stops assuming it owns the machine (2026-08-23)

§8's fixes all shared a blind spot: each one bounded a *stage*, but nothing bounded the *build*,
and nothing ever asked how much RAM was actually free. `chunk_size=150` was a constant. The
15.35M-row build did complete (`us_ml_dataset.parquet`, 2026-08-23 12:07, 2,903 tickers) — but
only by using essentially everything available, which is why the kernel OOM killer had already
taken an unrelated VS Code session once (§8.0 verification note).

### 9.1 Where the memory actually went (measured, not estimated)

Column-level sizing off the built parquet's own schema + row count, plus the §8.0 RSS measurements:

| Stage | Peak | Composition |
|---|---|---|
| **Pass 1** | **~8 GB** | `batch_fn` pinned the whole gated universe — prices + fundamentals, **5.5 GB measured** (§8.0) — while each batch added ~2.5 GB on top |
| Pass 2 | ~4.5 GB | slim input 1.23 GB + `out` 1.72 GB (12 float64 cols x 15.35M rows) + the beta loop's per-ticker Series list and its `pd.concat` |
| Pass 3 | ~5.2 GB | resident `slim` 1.7 GB + `clean_dataset` making **four** full copies of a ~1 GB row group |

Pass 1's 5.5 GB was a **floor**, not a peak: every §8 fix bought headroom on top of a number that
never moved. That's the thing worth fixing.

### 9.2 Fixes

- [x] **`memory.py` — budget + hard ceiling.** `budget_gb()` reads real `MemAvailable` and
      subtracts a reserve (default 4 GB) left for the rest of the machine; `chunk_size_for()`
      turns that into a batch size, clamped to `[MIN_CHUNK, MAX_CHUNK]` so row groups still
      compress. `apply_limit()` sets `RLIMIT_DATA` at 1.25x the budget, so an overrun raises
      `MemoryError` **inside this process** instead of handing the kernel a choice of victims —
      and can't disappear into the 3 GB swap partition and thrash. Both `main()`s call
      `memory.report()` first. Overrides: `BUILD_MEM_BUDGET_GB`, `BUILD_MEM_RESERVE_GB`,
      `BUILD_MEM_NO_RLIMIT`.
- [x] **Pass 1 loads its own batch from disk** (`_load_batch_from_disk`, `load_prices(tickers=)`
      already existed, `load_fundamentals(tickers=)` added). Kills the 5.5 GB floor outright.
      **Zero extra I/O** — raw data is one parquet per ticker, so each file is read exactly once
      either way; the batches just partition reads that already happened. **Byte-identical**,
      because `compute_fundamental_features` and `fill_missing_cagr` are strictly
      `groupby("ticker")` passes — asserted directly in
      `test_per_batch_fundamentals_stages_match_whole_universe`, which fails if anyone later adds
      a cross-ticker statistic to either (that failure would otherwise be silent).
      `_MergeBatcher` now takes an injected `load_batch` so tests can still drive it from
      in-memory fixtures.
- [x] **Coverage filter runs on a narrow projection.** `filter_tickers_with_no_fundamentals` reads
      only ticker sets, last-trade dates and row counts from `prices`, and only the ticker set
      from `fundamentals` — so `load_prices(columns=["ticker","trade_date"])` (~0.36 GB vs ~5 GB
      dense) plus `fundamentals_ticker_index()` (parquet footers, no column data) answer it
      exactly. The dense universe frame is now never built at any point in the US build.
- [x] **`cross_sectional.OUT_DTYPE = float32`.** All 12 outputs are z-scores, percentile ranks,
      return differences and rolling betas — two or three significant digits of real signal.
      1.47 GB -> 0.74 GB in Pass 2, and Pass 3 holds that same frame resident throughout.
      `test_beta_vs_market_matches_direct_computation`'s `rtol` moved 1e-9 -> 1e-6 to match the
      declared dtype, and now asserts the dtype explicitly so it can't be loosened by accident.
- [x] **Beta loop preallocates.** One `np.full(len(df))` written by position via
      `index.get_indexer`, instead of ~2,900 per-ticker Series (each carrying its own index) held
      for a final `pd.concat` that needed parts and result live at once.
- [x] **`clean_dataset` stops copying the frame four times.** `drop_duplicates(ignore_index=True)`
      replaces `.drop_duplicates().copy()` (and folds in the trailing `reset_index`); inf->NaN
      goes column-by-column over float columns only, instead of `df[numeric_cols].replace(...)`
      materialising ~150 columns, replacing into a second copy, then assigning back. This runs
      once per Pass-3 row group, so every copy here was paid ~20x per build.

### 9.3 Status

`ruff check` clean; **59/59 fast tests green**, including 8 new ones in
`tests/build_dataset/test_memory.py`. On this machine (8.5 GB available) the build now sizes
itself to a 4.5 GB budget with a 5.6 GB hard cap and 241 tickers/batch, leaving 4 GB for
everything else.

**Not yet observed end-to-end at full scale** — same honest caveat as §8.0.2/§8.0.3 carried:
the reasoning is measured-consistent and the equivalence properties are regression-tested, but a
real full-universe run under the new budget hasn't been executed. Run it `nohup`'d and watch RSS.
