# True Quarterly US Fundamentals, 1995–2006

**Status:** In progress. Supersedes `US_EQUITIES_EXPANSION_PLAN.md`'s framing that
2001–2006 is "annual Item 6 chaining, and that's the ceiling" — it isn't; see below.

## Context

US fundamentals (`data/raw/us/fundamentals/{TICKER}.parquet`) are quarterly only from
2007 (the `xbrl` tier). Everything older is **annual**: `ex27` (1995–2000) and `item6`
(2001–2006). Goal: make all eras quarterly, correctly.

Two problems, one of which is a **pre-existing bug this work must fix, not just avoid**:

1. **No quarterly data before 2007.** ~12 years of history at 1/4 the resolution.
2. **The schema cannot tell the two apart.** There is no period-length column anywhere.
   Flow columns (`net_revenue`, `net_income`, `ebit`, `gross_profit_reported`,
   `cashflow_ops`, `capex` — `companyfacts.py:290 _FLOW_ITEMS`) silently mix quarterly
   magnitudes (xbrl) with annual magnitudes (ex27/item6) **in the same column**,
   distinguishable only by `fundamentals_tier`. Every flow-derived ratio (margins,
   `p_sr`, `ev_ebit`, `roe`, `roa`) inherits the mix. Verified on the real on-disk
   `AAPL.parquet`: 75 xbrl rows at a 91-day median `end`-gap sitting in the same columns
   as 7 ex27 + 5 item6 rows at ~365 days.

Downstream, this is already costing more than resolution. `features.py:298`
`QOQ_GAP_DAYS=(60,120)` / `YOY_GAP_DAYS=(300,400)` reject essentially every pre-2007 row,
so QoQ/YoY features produce nothing for a decade of history; and `cagr_handler`'s 5-year
lookback crosses the 2006/2007 annual→quarterly boundary and produces a bogus ~4x level
break. **Fixing the mixing is a bigger downstream win than the added coverage.**

## What is already verified (empirically — do not re-derive)

- **10-Qs are already in the cached index.** `universe.py:45` matches `^10-K|^10-Q`, so
  `data/raw/us/sec/edgar_10k10q_filings.parquet` already holds 760,595 10-Q rows.
  No index rebuild. URL = `https://www.sec.gov/Archives/{filename}` (`fds.py:65`).
- **1995–2000 quarterly is nearly free.** 22 of 24 sampled real 10-Qs (1996–2001) carry an
  EX-27 exhibit with `PERIOD-TYPE ∈ {3-MOS, 6-MOS, 9-MOS}`, `ARTICLE 5` — the format
  `fds.py` already parses. Two lines block it: the `10-K` form filter (`fds.py:216`) and
  the `PERIOD-TYPE != "YEAR"` early-return (`fds.py:126`).
- **Flows are cumulative YTD; instants are not.** CIK 1000366 FY1999, raw tags:
  revenue `121,701 → 275,712 → 428,558 → 576,997` (3/6/9-MOS + 10-K YEAR) while
  `TOTAL-ASSETS` moves `1,061,164 → 1,131,387 → 850,394 → 971,809`. Differencing yields
  **Q4 net income = 32,563 − 36,739 = −4,176**, a loss quarter invisible in annual data.
- **2001–2006 needs HTML parsing, and it works.** `pandas.read_html` parses these filings.
  AAPL Q3 FY2004 table 4 (shape 40×14): header row 0 = `Three Months Ended`×5 (cols 2–6)
  then `Nine Months Ended`×5 (cols 8–12); row 1 = dates; `Net sales` → `2014` (3mo current),
  `1545` (3mo prior-year), `5929` (9mo). **The discrete 3-month column is printed**, so
  differencing is only a fallback there. Reg S-X captions are real anchors:
  `CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS` appears exactly 2× (TOC + statement).
- **Scale.** Restricted to the 8,017 crosswalk CIKs: 26,071 10-Qs in 1995–2001 (1,523 CIKs)
  + 29,321 in 2001–2006 (1,765 CIKs) = **55,392 fetches ≈ 1.8h** at the shared 10 req/s
  throttle floor (`http._throttle` is one global lock; the 8 workers do not raise it).

## Decisions (confirmed with user, 2026-08-01)

- **Uncertain quarter → NaN the flows, never guess — plus a companion 0/1 flag**, following
  the BR collector's informative-NaN convention: `features.py:708 cagr_earnings_defined`,
  `merge.py:367 has_dividends`, `features.py:206 adj_close_precision_degraded`. The value
  is NaN'd; a 0/1 column records *why*; the count is logged so the coverage cost is
  measurable. Never fill, never clip.
- **Unrecoverable fiscal years keep their annual row**, labeled `period_months=12`. Loses
  no existing coverage; the label is what makes it safe (`df[df.period_months==3]` gives a
  clean quarterly panel, impossible today).
- **All 4 phases**, with a ~50-filing dry run gating the expensive Phase 3 fetch.

## Schema change

Three columns, set by **every** tier:

| column | dtype | meaning |
|---|---|---|
| `period_months` | `Int8` | period length the row's **flow** columns describe: 3, 6, 9, 12 |
| `flows_derived` | `int8` 0/1 | ≥1 flow value came from a subtraction, not a printed figure |
| `flows_defined` | `int8` 0/1 | 0 = flows NaN'd because reconstruction was unsafe (mirrors `cagr_*_defined`) |

Per tier: `xbrl` derives from `round((end-start)/30.44)` — **must not be hardcoded to 3**,
because `_quarterly_only`'s `ifrs-full` exemption (`companyfacts.py:171-177`) puts genuine
12-month foreign-filer rows inside the "quarterly" tier. `ex27` from `<PERIOD-TYPE>`.
`tenq` from the parsed header. `item6` always 12.

**Checked, not a real problem:** `loaders.py:137`'s `optimize_dtypes` path downcasts
`Int8`/`int8` to `float32`. Verified empirically (2026-08-01): `pd.NA` converts cleanly to
`NaN`, and equality checks (`period_months == 3`) still resolve correctly post-downcast —
no precision loss possible for values 0–12. No loaders.py change made; would have been
unrequested complexity for a non-problem.

## Phases

Each phase is independently shippable and verifiable.

### Phase 0 — persist the plan
- [x] Write this plan to `docs/US_QUARTERLY_BACKFILL_PLAN.md` with checkboxes, and link it
      from `docs/US_EQUITIES_EXPANSION_PLAN.md`.

### Phase 1 — schema + the shared differencer (code only, no collection run) — ✅ DONE 2026-08-01
- [x] Add `ytd_to_discrete(df, flow_cols=None)` to `sec/companyfacts.py`.
- [x] Grouping robust to FYE change: start a new group when `period_months <= prev` **or**
      `|implied_start − group_start| > 20d`.
- [x] Difference **raw YTD values only, never chain already-differenced ones**.
- [x] Guards → NaN flows + `flows_defined=0` + logged reason: step ≠ 3 months; group's
      first row not 3-MOS; FYE-change guard tripped; **differenced `net_revenue < 0`**.
      `net_income` is *not* sign-checked.
- [x] Add the three columns across all four tier builders (xbrl real, item6/ex27 constant
      12-month placeholders — ex27 goes real in Phase 2); `_derive_q4` sets `flows_derived`.
      `xbrl`'s `period_months` derives from `(end-start)` snapped to {3,6,9,12}, confirmed NOT
      hardcoded via an ifrs-full annual-fact test.
- [x] `validate.py` warn-only additions (period_months presence/validity/mixing, negative
      net_revenue, flows_defined=0 count). `loaders.py` — checked empirically, not needed
      (see above).
- [x] Tests in `test_sec_companyfacts.py` (5 new) + `test_sec_fundamentals.py` (1 new).

**Verify:** ✅ all 47 fast tests pass incl. the real CIK 1000366 FY1999 reconciliation
(Q4 NI = −4,176 exactly) and the AAPL/HSBC-style ifrs-full period_months=12 case.

### Phase 2 — EX-27 quarterly, 1995–2000 — ✅ DONE 2026-08-01
- [x] `fds.py`: accept `PERIOD-TYPE ∈ {3-MOS,6-MOS,9-MOS,YEAR}` → `period_months`;
      form filter widened to `("10-K", "10-Q")` prefixes.
- [x] **Period end from `<PERIOD-END>` for quarterly exhibits, `<FISCAL-YEAR-END>` only for
      `YEAR`** (new `_fds_period_end` helper) — never a fallback, guards the exact ADP bug
      (`fds.py`'s own docstring) at 26,071-filing scale instead of one.
- [x] Ordering: as-first-reported dedup (moved before differencing — it needs exactly one
      row per period) → `_fill_missing_multipliers` (unchanged position) → `ytd_to_discrete`
      → **recompute ratios**.
- [x] Q4 needed no separate code path — the 10-K's `YEAR` exhibit joins the same frame and
      is just the 4th YTD link, transformed in place by `ytd_to_discrete`. No annual/Q4
      collision in this tier (confirmed: EX27_ERA_END's existing 1-year buffer already
      covers the "early-2001 stragglers" case, no change needed there).
- [x] Tests in `tests/data_collection/test_sec_fds.py` (4 new, 1 rewritten to match the new
      "quarterly IS mapped" behavior).

**Verify:** ✅ CIK 1000366 FY1999 reproduces all four quarters end-to-end through the real
fetch→parse→multiplier-fill→dedup→differencing→ratio-recompute pipeline incl.
**Q4 NI = −4,176** and Q4's `net_margin` computed on the discrete (not annual YTD) figures.
All 47 fast tests pass.

### Phase 3 — HTML 10-Q, 2001–2006 — ✅ DONE 2026-08-01 (scope: income statement only)
- [x] New `sec/tenq.py`, house `build_cik_history(cik, filings)` signature. New file
      justified: Item 6's locator is a single year-header row; a 10-Q uses a two-row
      period header mapping to column ranges — incompatible table models.
- [x] Reuse `_row_values`, `_normalize_label`, `_row_text`, `detect_unit_multiplier`,
      `_UNITS_RE` from `selected_financial_data.py`. **Own `ROW_ALIASES`** (income-statement
      labels only: `net_revenue`, `net_income`, `cost_of_revenue`).
- [x] **Scope reduced from the original plan**: cash-flow statement (`cashflow_ops`/`capex`)
      NOT implemented this pass — real added complexity (a second table-location problem +
      per-column period-length reconciliation against income-statement rows, since income
      items are already-discrete but cash-flow items are still YTD) not yet verified against
      live data. Flagged in `tenq.py`'s module docstring so it isn't mistaken for existing
      coverage. Income-statement items need **no differencing at all** in this era — every
      sampled filing prints the discrete 3-month figure directly, confirmed live — so
      `ytd_to_discrete` isn't called by this tier.
- [x] `_TIER_PRIORITY` addition deferred to Phase 4 (needs `xbrl`/`ex27`/`item6` reconciled
      together with `tenq` in one place).
- [x] Tests in `tests/data_collection/test_sec_tenq.py` (7 new), registered in
      `tests/run_all.py`'s FAST list (was missing — new test files aren't auto-discovered).

**Two real bugs found via live-data testing, fixed at their root in
`selected_financial_data.py` (shared with Item 6, which inherits the fixes for free):**
1. `_normalize_label` didn't collapse internal whitespace — AAPL's real "Cost of  sales"
   (embedded double space, an HTML-entity artifact) failed every alias match.
2. `_row_values`'s paren-merge only handled a 2-cell split (`['(25', ')']`). AAPL's real
   "Net income (loss)" row renders its prior-year loss as a **3-cell** split
   (`['(8', '(8', ')']` — the negative value colspan-duplicated same as every other figure,
   THEN its closing paren alone) — silently parsed as a $8M profit instead of an $8M loss.

**Verify:** ✅ AAPL Q1 FY2004 (fetched live, CIK 320193, accession 0001104659-04-003080)
reconciles exactly end-to-end: `net_sales=$2,006M/$1,472M` (current/prior), `cost_of_sales=
$1,470M/$1,066M`, `net_income=$63M/-$8M` (the real prior-year loss, correctly negative).
All 48 fast tests pass.

### Phase 4 — Q4 for 2001–2006, tier priority, validation — ✅ DONE 2026-08-01 (code only)
- [x] `fundamentals._derive_annual_q4(quarters, annual)`: guards — exactly 3 quarters nest
      in `(fy_end-370d, fy_end-20d]`, spaced 60–120d, Q3 within 60–120d of FY end, derived
      Q4 revenue ∈ `[0, 0.60 × FY revenue]`. Any guard failing → no Q4, annual row untouched.
- [x] Item6 row **consumed explicitly** (removed from `annual`, not left for the 10-day
      `cluster_period_ends` dedup to resolve) — its `end` is only a Dec-31-ish guess, so a
      >10-day miss could have silently shipped both the 12mo and 3mo row for one real period.
- [x] Derived row keeps every non-flow item6 column as-is (`total_assets`, `equity`,
      `eps_basic`, `item6_form/filename`...) and inherits `fundamentals_tier="item6"` (most of
      its data still comes from there) — ratios recomputed on the adjusted flows.
- [x] `_TIER_PRIORITY = {"xbrl":0, "ex27":1, "tenq":2, "item6":3}` wired into
      `build_company_fundamentals`; a real bug caught by testing: the naive `quarters[
      "fundamentals_tier"]="tenq"` blanket assignment had to move BEFORE calling
      `_derive_annual_q4`, else it would clobber the derived rows' intended `"item6"` tag.
- [x] `validate_us_fundamentals` additions already landed in Phase 1 (period_months presence/
      validity/mixing, negative net_revenue, flows_defined=0 counts) — nothing further needed.
- [x] `loaders.py`'s `FUNDAMENTALS_PROVENANCE_COLS` extended with `tenq_filename`/`tenq_form`.
- [x] Tests in `test_sec_fundamentals.py` (4 new: Q4 derivation success/incomplete/rejected-
      share, tier priority) — all 8 existing `build_company_fundamentals` tests needed a new
      `tenq.build_cik_history` mock added (new call site in the function under test).

**Deliberately NOT done this session — needs separate explicit go-ahead:**
- [ ] Re-measure `_FLOORS` against real quarterly-magnitude data before touching thresholds
      (quarterly figures are ~4x smaller than the annual ones `_FLOORS` was tuned against).
- [ ] Full rebuild of all 8,143 ticker files (`python -m src.data_collection.sec.fundamentals`
      / `run_us_full_scale.py fundamentals`, ~1.8h wall-clock at the 10 req/s SEC floor for the
      ~3,300 affected CIKs) — this is a live external-facing collection run overwriting real
      collected data, out of scope for an unattended code session per this repo's working rules.

**Verify (code-level, no live run):** ✅ all 48 fast tests pass; whole-repo `ruff check` clean;
`_derive_annual_q4` reconciles the CIK-1000366-shaped fixture exactly (Q4 revenue=FY−ΣQ,
ratios recomputed, non-flow columns pass through); tier priority confirmed end-to-end through
`build_company_fundamentals` on a synthetic tenq/item6 overlap.

## Known limits (state, don't fix)

- **Survivorship.** Tier-1 crosswalk is current tickers only; dead pre-2007 companies with
  no surviving ticker are absent. Pre-existing, unchanged by this work.
- **~8% of 1995–2000 10-Qs carry no EX-27**, plain `.txt`, pre-HTML — not recoverable by
  `tenq.py`. Degrades gracefully: one missing exhibit costs one quarter, not the year.
- **As-first-reported vs same-basis differencing cannot both hold.** We keep
  as-first-reported and *detect* breakage via the negative-revenue guard rather than
  prevent it.
