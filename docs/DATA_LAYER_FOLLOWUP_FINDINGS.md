# Data Layer Follow-Up Findings

Surfaced 2026-08-20 while verifying `DATA_LAYER_CORRECTNESS_PLAN.md` §1's currency-unit migration
against the real rebuilt `ml_dataset.parquet` (`tests/run_all.py --group data`). **None of these are
caused by §1's own scope (currency units)** — each is a different, pre-existing subsystem that
migration's verification happened to shine a light on. Two items (§2a, §2b) already have a home in
the correctness plan and are only cross-referenced here, not duplicated.

**Update 2026-08-21:** all six items investigated and addressed in code (see each item below for
what changed and why). Two of six turned out not to be bugs at all — the test heuristics were
wrong, not the data (NaN-hole width, pl-frozen illiquidity) — and are fixed test-only. `cagr_revenue`
is accepted as a permanent tradeoff (owner decision), threshold lowered accordingly. CAMB3, the
top-50 `adj_close` glitches, and the shares/splits mitigation are real code changes, confirmed
against a real rebuild (owner ran it): top-50 fully fixed (9→0 NOT READY), NaN-hole and pl-frozen
checks pass clean, cagr_revenue passes at the new threshold; shares/splits unchanged (19/544,
TIMS3 still worst) — see that item for why. One new finding surfaced during this rebuild's
verification (AFLT3, below) — turned out to be a real bug in this session's own top-50 fix, not a
pre-existing issue; fixed same day.

## New finding (surfaced 2026-08-21 verifying the above, not part of the original six)

- [x] **AFLT3: 2 corporate-events "leaking" into `log_return`** (`test_final_dataset.py`'s
  `no unadjusted split jumps` check). Traced against raw `data/raw/br/prices/AFLT3.parquet`: a real
  1000:1 reverse split (2007-12, recorded twice in `corporate_events.parquet` at inverse-equal
  ratios, same duplicate-representation pattern as TIMS3). Root cause, found by actually running
  `repair_unadjusted_splits()` then `repair_isolated_adj_close_glitches()` on this ticker in
  isolation (not a full rebuild): **this session's new glitch-repair, not a pre-existing bug.**
  `repair_unadjusted_splits()` correctly makes `adj_close` continuous across the real split. But
  raw `close` has its own, unrelated one-day error right next to it (`18.0 → 0.018 → 18.0` on
  2007-12-07, a decimal-placement glitch, not the split) — and *after* `adj_close` is already
  fixed, that bad `close` value alone makes the ratio (`adj_close/close`) pulse and revert, fooling
  `repair_isolated_adj_close_glitches()` into "fixing" the one value that was already correct,
  reintroducing the exact discontinuity `repair_unadjusted_splits()` had just removed. My first
  version assumed a ratio pulse-and-revert could only ever mean adj_close was the broken side; it
  can also mean close is.
  **Fixed 2026-08-21:** added a second condition — `close` itself must stay flat across the same
  window (`abs(close's own log-return) < MIN_DETECTABLE_JUMP` on both the in and revert transitions)
  — a real glitch only ever shows up on one side of the ratio, never both. Regression-tested in
  `tests/build_dataset/test_repair.py::test_glitch_repair_ignores_ratio_pulse_caused_by_bad_close`
  (replicates AFLT3's exact shape). Re-verified against real data: AFLT3's leak is gone (0/2), and
  a full corpus re-scan with both repairs chained shows 77 rows/49 tickers still fixed (down from
  the original 92/58 — the difference is exactly this class of false positive being excluded), 71
  of those 77 still on the two systemic dates (37 on 2012-02-22, 34 on 2020-11-20) — the core
  top-50 fix (PETR4/ITSA4/GGBR4/GOAU4/SBSP3/VIVT3/etc.) is unaffected.

---

## Shares outstanding / splits

- [x] **`market_cap/shares == close` fails on 19/549 BR tickers** (worst: TIMS3, ratio ~99x,
      n=1979 — `close` ranges 0.000038 → 18.79 across TIMS3's full history, consistent with a real
      stock split/consolidation not being reflected consistently between `shares_outstanding`
      (fundamentals-sourced) and `close` (price-sourced)). `market_cap`, `shares_outstanding` and
      `close` were never touched by §1's currency-unit fix, so this predates that work. Needs the
      same per-event verification rigor as `ticker_continuity.json`, not a quick patch. Measured via
      `tests/build_dataset/test_unit_scale_invariants.py`'s `market_cap/shares == close` check.
      **Investigated 2026-08-21 (TIMS3 traced end to end):** `data/raw/br/cvm/shares.parquet` (FRE
      capital_social) has **zero rows** for TIM S.A.'s current CNPJ (`02421421000111`) — CVM's own
      shares timeline is simply absent for this entity, so `cvm/ratios.py`'s native (post-2018)
      `shares_outstanding` is 100% NaN for TIMS3. The non-null values actually reaching
      `ml_dataset.parquet` (~2.42B, plausible for real TIM) come from `TIMP3→TIMS3` in
      `ticker_continuity.json` (rename, ratio 1.0) splicing TIMP3's own (real, CVM-populated)
      2010–2020ish shares history forward — which then never updates again, including across
      TIMS3's own real 2025-07 100:1 reverse split (`corporate_events.parquet` confirms it; raw
      `close` correctly jumps ~103x on 2025-07-03), because there's no later FRE row to pick up the
      new count. Net effect: `shares_outstanding` is frozen at a decade-plus-old TIMP3 value forever.
      **Not a quick patch** — confirmed genuinely per-ticker: some of the 19 are likely this same
      "spliced-then-frozen-across-a-later-split" shape, others may be pure FRE-cnpj gaps with no
      continuity splice to borrow from at all. A **partial, generic mitigation** does exist though:
      after `_shares_asof()` in `cvm/ratios.py`, forward-adjust the last known share count by any
      `corporate_events` split/inplit factor dated after it and before `reference_date` — this would
      fix the "real split happened, share count never caught up" failure mode (plausibly most of the
      19) without needing per-ticker hand-verification. Doesn't fix a bare FRE-cnpj gap with nothing
      to splice from. Worth scoping as its own small task if picked up.
      **Implemented 2026-08-21:** `cvm/ratios.py`'s `_shares_asof()` now also returns each matched
      FRE record's own `effective_date`; a new `_apply_share_events()` forward-adjusts the share
      count by every `corporate_events` split/inplit dated after that `effective_date` and at-or-
      before `reference_date`, via a new `_share_events()` that collapses same-event duplicate rows
      (some real events are recorded twice within days at an inverse-but-equal ratio — e.g. TIMS3's
      2007 reverse split as both "1000:1" and "1:0.001" — collapsing avoids double-applying one
      event). Wired into `build_fundamentals()`. Covers the "real split happened, FRE never caught
      up" shape (plausibly most of the 19, definitely TIMS3); a bare FRE-cnpj gap with nothing to
      splice from is unchanged (still NaN/stale) — that half of the 19 still needs the
      `ticker_continuity.json`-style per-ticker pass this item always said it would. Unit-tested
      (synthetic dedup + forward-adjustment, replicating TIMS3's exact shape, plus a direct
      `_ticker_family()` check against the real `ticker_continuity.json`) in
      `tests/data_collection/test_cvm_statements.py`.
      **Confirmed against a real rebuild 2026-08-21: still 19/544 tickers outside band, TIMS3 still
      worst (ratio 99.24, unchanged).** Root cause, found by actually inspecting the rebuilt data:
      the mitigation needs BOTH (a) a continuity relative with real FRE shares data, AND (b) that
      relative's OWN reference_date series to still extend past the real split date. TIMP3 (TIMS3's
      donor) stopped filing in 2020, before TIMS3's real 2025 split — there's no row left in
      TIMP3's own file for the event to attach to, so the frozen value still propagates through
      Stage 2's forward-fill untouched. Fixing this fully needs the adjustment applied where the
      forward-fill actually happens (Stage 2's daily merge), not at the raw per-ticker CVM-build
      stage — a bigger change, not attempted here. The mitigation as shipped only helps a ticker
      that keeps filing under its own name past a real split; whether any of the other 18 fit that
      shape (vs. TIMS3's shape) isn't yet confirmed — the "worst 5" list changed identity (AZUL3,
      ADMF3, RVEE3, CELP6 now visible) but the total count didn't move, so no regression is evident,
      just an unconfirmed reshuffle of which of the 19 rank worst.
      **TIMS3 itself resolved 2026-08-22, and it was neither a shares-freezing nor a splice
      problem after all**: TIMS3's raw (non-`adj_*`) `open`/`high`/`low`/`close` is itself on a
      wrong ~100x-too-small scale for its whole pre-2025-07-03 history — a vendor OHLC defect,
      confirmed by cross-checking against `TIMP3` (its real pre-2020 entity, spliced in via
      continuity), whose close is exactly 100.000000x TIMS3's native close on all 2,273
      overlapping trading days 2011–2020. `repair_unadjusted_splits()` already detected and fixed
      this for `adj_close` (the two-year-old "TIMS3's /10000 arrives as two /100 jumps" comment)
      but never rescaled the plain OHLC columns — which is exactly what `_price_asof()` and
      `merge.py`'s close-price lookup both read. Fixed via a new `RAW_OHLC_ALSO_UNADJUSTED = {"TIMS3"}`
      set in `repair.py` that also rescales raw OHLC for flagged tickers. Not yet run against a
      full rebuild to confirm `test_unit_scale_invariants.py` goes green.

      **AZUL3/ADMF3/RVEE3/CELP6 investigated independently 2026-08-22 (per the note above — do
      NOT assume they share TIMS3's mechanism, and they don't):** none has a `ticker_continuity.json`
      splice, and only RVEE3 has any `corporate_events.parquet` row at all, so the TIMS3-style
      "spliced donor stopped filing before a real split" shape doesn't apply to any of the four.
      Two different, more general mechanisms are responsible instead:

      1. **A universal `reference_date`-vs-`fundamentals_available_date` price-anchor mismatch —
         present dataset-wide, just usually small enough to pass.** `cvm/ratios.py:312` computes
         `market_cap` (and every price-linear ratio: `pl`, `pvp`, `p_sr`, `p_ebit`, `p_ebitda`,
         `p_assets`, `ev_*`) off `close_price = _price_asof(reference_date)` — the fiscal
         quarter-end price. `merge.py`'s later close-price lookup (its own comment: "BolsAI's
         close_price is from reference_date, 45-90 days earlier") deliberately **overwrites**
         `close_price` with the price as of `fundamentals_available_date` instead, to kill stale-
         price valuation jumps — but does **not** rescale `market_cap`/the ratios already built
         off the old anchor to match. `recompute_valuation_daily()` then re-anchors those ratios
         to the *daily* close using `factor = close/close_price`, where `close_price` is now the
         availability-date price while `market_cap`'s embedded basis is still the reference-date
         price. Net effect: `(market_cap/shares_outstanding)/close` collapses to exactly
         `close_price@reference_date / close_price@fundamentals_available_date` — a **constant per
         fundamentals quarter**, confirmed empirically (std ≈ 0 within every `fundamentals_available_date`
         group, for all 4 tickers **and** for PETR4/VALE3 too — 62/61 quarters, ratio wobbling
         0.69–1.26 around 1.0). It's invisible for stable large caps because price rarely moves
         much in a 45–90 day gap; it dominates for anything volatile within that window:
         - **AZUL3**: ratio is a *perfect* constant (std ~0) across all 43 non-null rows — one
           quarter (`reference_date` 2026-03-31, `close_price` 228.83) whose `fundamentals_available_date`
           (2026-05-07ish) fell right after AZUL3's own real ~90% single-day crash
           (2026-04-20, confirmed in its raw price file, no corporate_events/continuity entry —
           this is the still-unresolved AZUL4→AZUL54→AZUL53→AZUL3 judicial-recovery restructuring
           documented above). Both close-price readings are individually correct; they're just
           6 weeks apart across a real crash, which is exactly what this mechanism can't survive.
         - **ADMF3**, **CELP6**: same mechanism, recurring per-quarter with a different ratio each
           time (ADMF3: 2.01/3.68/0.52/1.25 across its 4 fundamentals quarters; CELP6: 1.0/1.0/
           0.057/2.34/1.26/0.36/1.01 across 7) — ordinary small/micro-cap volatility inside each
           filing-lag window, nothing split- or vendor-related.
      2. **RVEE3 additionally has its own, genuine bug — a real split double-counted.**
         `corporate_events.parquet` records a real 1:10 split for RVEE3 on 2025-08-07 (confirmed:
         raw close drops from ~70 to ~6.88 that day, matching the recorded factor). But
         `data/raw/br/cvm/shares.parquet`'s own FRE record for RVEE3's CNPJ *already* jumps
         1,017,115 → 10,171,150 (exactly 10x) at `effective_date` 2025-08-06 — CVM's own filing
         already reflects the post-split count, one day before the trading-adjustment date.
         `_apply_share_events()` (added for TIMS3, `cvm/ratios.py`) can't tell "FRE already caught
         up" from "FRE is frozen": it reapplies the corporate_events factor whenever the event
         date falls after the matched FRE row's `effective_date`, so it multiplies the
         *already-adjusted* 10,171,150 by another 10x, landing on 101,711,500 — confirmed exactly
         in the raw fundamentals file (`shares_outstanding` reads 101,711,500 from 2025-09-30
         onward, should be 10,171,150). This is a real regression surfaced by the TIMS3 fix, not
         present before it — TIMS3 needed the forward-adjustment because its donor (TIMP3) never
         filed again after the split; RVEE3 never needed it at all, but the code has no way to
         distinguish the two shapes. Not fixed here (investigation only, per owner's standing
         "don't fix without being asked" rule) — the fix would need `_apply_share_events()` to
         skip an event whenever the matched FRE row's `effective_date` is already on-or-after
         that event's date (i.e., only forward-adjust for events *after* the FRE record, never
         events the FRE record already absorbed).

      **Net assessment:** none of the 4 need TIMS3-style raw-OHLC repair. Mechanism 1 (anchor
      mismatch) is the dominant driver for all four and is likely the real explanation for most of
      the remaining 18 tickers outside `test_unit_scale_invariants.py`'s band, not just these —
      worth its own scoped fix (rescale `market_cap` by the same reference→availability price
      ratio at the point `merge.py` swaps `close_price`, before `recompute_valuation_daily()` runs)
      rather than another per-ticker patch. Mechanism 2 (RVEE3's double-count) is narrow and
      specific to `_apply_share_events()`.

## Fundamentals coverage

- [x] **`cagr_revenue` NaN coverage dropped to 78.1% explained** (was implicitly higher before;
      `test_final_dataset.py`'s 80% threshold now fails, "21.9% unattributed"). Real, likely
      permanent side effect of migrating the full crosswalk to CVM: `cvm/ratios.py`'s
      `compute_ratios` always emits NaN for `cagr_revenue_5y`/`cagr_earnings_5y` (CVM never
      computes CAGR — Stage 2's `fill_missing_cagr()` backfill from earnings/revenue history is the
      only source now), whereas the 115 previously-BolsAI-sourced holdout tickers used to carry
      BolsAI's own raw CAGR figures into that coverage number before §1's migration moved them to
      CVM. A real tradeoff of that migration, not a bug — needs a decision (loosen the test's
      threshold with a documented reason, or find another CAGR source), not a silent fix.
      **Investigated 2026-08-21:** confirmed no additional backfill source exists —
      `calc_annual_cagr()` needs 20 contiguous quarters at a fixed anchor month; CVM's own
      per-ticker history is the only input. Not solvable in code, only by lowering the threshold
      (e.g. to ~75-78% with this migration cited as the reason) or accepting the coverage drop.
      **Decided 2026-08-21 (owner): accepted.** `test_final_dataset.py`'s `cagr_revenue` check
      (`cagr_earnings`'s own threshold is untouched) now requires >75% explained instead of >80%,
      with this migration cited inline as the reason.
- [x] **4 tickers have exactly one single-day interior NaN hole** across
      `equity`/`net_income`/`total_assets` simultaneously: AZEV3 (2020-03-23), AZEV4 (2019-11-25),
      INEP3 (2018-07-02), RPMG3 (2014-06-02). `test_final_dataset.py`'s prefix-NaN rule flags this
      as a "suspicious merge bug" (all three columns, not a partial gap). Dates are unrelated to
      each other and to today's migration (2026-08-20) — looks like a narrow, pre-existing
      single-row merge/forward-fill edge case, not systemic. Worth a root-cause pass, not urgent
      (4 tickers, 1 row each).
      **Root-caused 2026-08-21 — not a bug, and not single-day either.** The "hole" is the *start*
      of a real gap that's actually one to several **quarters** wide (RPMG3: 2012-09-30 →
      2015-09-30, three years of missing `reference_date`s in the final dataset), caused by the
      already-intentional filing-lag filter (`quality_filters.MAX_ACCEPTABLE_FILING_LAG_DAYS=180`)
      dropping quarters CVM shows as filed >180 days late (confirmed for RPMG3 via
      `filing_lag_days` on the surrounding quarters — a real multi-quarter late-filing streak,
      plausibly a distress/judicial-recovery period for a refinery that's been in trouble for
      years). It hits `equity`/`net_income`/`total_assets` "simultaneously" only because all three
      come from the one dropped quarterly row — not a merge bug at all. The test's heuristic (flag
      any ticker with a NaN transition in all three tracked columns) can't tell a legitimate
      filing-lag gap from a real defect, and undersells the gap width by only counting transition
      *points*, not span. Fix is test-only: either cross-check flagged tickers against
      `filing_lag_days` to auto-exonerate this cause, or just downgrade the message from
      "suspicious merge bug" to informational.
      **Fixed 2026-08-21:** `test_final_dataset.py`'s prefix-NaN rule now measures each hole's
      *width* (contiguous NaN run length) instead of just counting transition points, and only
      flags a ticker as "suspicious merge bug" if its narrowest hole in all three columns is under
      `MIN_SUSPICIOUS_GAP_ROWS = 15` trading days — comfortably below a real quarter-scale
      filing-lag gap (60+ days) and comfortably above a genuine single-row artifact.
- [x] **`pl` frozen within-quarter on 277/19968 (1.39%)** — `test_final_dataset.py`'s regression
      guard for `recompute_valuation_daily()` allows <1%; this is just over. Not investigated
      further; likely a handful of tickers with sparse trading days per quarter rather than a
      re-anchoring regression (unrelated to §1 — `pl` is scale-invariant under §1's fix).
      **Root-caused 2026-08-21 — confirmed illiquidity, not a bug.** Sampled several of the 277
      groups directly (AFLT3, AHEB3): `volume == 0` for essentially every trading day in the frozen
      quarter, so `close` itself never moves — `recompute_valuation_daily()`'s
      `factor = close/close_price` is correctly constant because its input is constant.
      `pl` *does* track `close`; `close` just has nothing to track on a stock that didn't trade.
      Fix is test-only: either loosen the <1% threshold with this documented reason, or (cleaner)
      restrict the "eligible" population to days with `volume > 0` so the check measures what it
      actually intends.
      **Fixed 2026-08-21:** `test_final_dataset.py`'s check now groups only rows with `volume > 0`,
      so a stock that genuinely didn't trade no longer counts against the guard.

## Price / `adj_close` data quality

- [x] **CAMB3 has 1 row with NaN `open/high/low/close`** (raw price data, not just `adj_close`) —
      also why `test_final_dataset.py`'s "no NaN in close" check fails (same single row). Found via
      `test_br_data_quality.py`.
      **Root-caused 2026-08-21 — trivial.** `data/raw/br/prices/CAMB3.parquet`, 2019-08-15: every
      OHLC + `adj_*` field is NaN and `volume == 0` — a phantom non-trading day from the raw
      source, not a merge artifact. Generic fix: drop any row where the full OHLC vector is NaN
      with zero volume (a one-line filter in the price collector/validator — would guard every
      ticker, not just this one, and is a narrower, more targeted sibling of the existing
      `_drop_incomplete_today()` guard, which only covers a live still-forming "today" bar, not a
      NaN row buried in history).
      **Fixed 2026-08-21:** `loaders.load_prices()` now drops any row where the full OHLC vector is
      NaN, dataset-wide — fixes the built `ml_dataset.parquet` for every ticker going forward. Also
      added `src/data_collection/one_off/fix_camb3_phantom_row.py` to clean the already-collected
      raw file itself (so `test_br_data_quality.py`'s raw-file sweep is clean at the source too) —
      written but not run yet; needs an explicit go-ahead since it mutates a stored raw file.
- [x] **LUXM4 has 289 rows with non-positive `adj_*`** — this is the CLAUDE.md-documented
      2-decimal-precision underflow ticker (`adj_close_precision_degraded`); flagged here only
      because `test_br_data_quality.py`'s hard validator still trips on it. Consistent with §2a's
      "flag only, no repair" decision, not a new issue.
      **Confirmed 2026-08-21 — no new work.** Matches the already-decided §2a territory exactly;
      the fix (if any) is teaching `test_br_data_quality.py`'s hard validator to accept
      `adj_close_precision_degraded == 1` as sufficient, same as `test_final_dataset.py` already
      does — not a data fix.
- [x] **`test_top50_ml_readiness.py`: 9/50 tickers NOT READY** — PETR4, ITUB4, SBSP3, BBDC4, BBAS3,
      ITSA4, GGBR4, GOAU4, VIVT3. All for `adj_close` discontinuities without a matching raw-close
      move, or large single-day `log_return` moves not matched to a recorded corporate event (e.g.
      PETR4: 2012-02-22/23 and 2020-11-20/23, `close` moves <1% but `adj_close` jumps ±70-270%).
      Price-adjustment / corporate-events domain (`repair.py`, `corporate_events`), untouched by
      §1. The generated report `TOP50_ML_READINESS_AUDIT.md` (repo root, untracked) has full detail
      per ticker.
      **Investigated 2026-08-21 — root cause found, high-confidence fix path exists.** All 9
      tickers show the *identical* pair of discontinuity dates (2012-02-22/23 and 2020-11-20/23),
      confirmed by directly reading `TOP50_ML_READINESS_AUDIT.md`'s §6 CRITICAL list and PETR4's
      raw price file. These 9 span totally unrelated sectors (oil, banks, utilities, steel,
      telecom) with no matching `corporate_events` entry and no real-world split/reorg on either
      date — a single company having a real corporate action doesn't explain dozens of unrelated
      tickers glitching on the exact same two calendar dates. This is a vendor-side (BolsAI)
      batch data defect on those two dates specifically, the same *class* of bug this audit
      already found and fixed twice elsewhere in this repo (MRFG3's 2-decimal `adj_close` floor,
      and the ticker-continuity splice reconciliation in `continuity.py`, both §4 of the audit
      doc). Same fix shape applies: detect a single-day pulse (`adj_close` swings hard away from
      *both* neighboring days while raw `close` doesn't, no nearby `corporate_events` row — a
      cleaner signal than a real split, which never reverts the next day), and repair by
      recomputing that day's `adj_close` from the previous day's `adj_close` × the day's raw-close
      return — exactly the technique `fix_mrfg3_adj_close.py` already used. Would plausibly also
      clear a chunk of the 21 "READY WITH CAVEATS" tickers' single-day `log_return` warnings for
      free, since those look like the same pattern at lower amplitude. Medium effort, precedented,
      low risk — worth scoping as a generalized version of `fix_mrfg3_adj_close.py` rather than a
      one-off.
      **Implemented 2026-08-21:** `repair.repair_isolated_adj_close_glitches()` (detects the ratio
      pulse-and-revert shape described above; repairs by holding the prior day's ratio through the
      bad row), wired into `build_ml_dataset.main()` right after `repair_unadjusted_splits()`. A
      dry-run of the detector alone against every raw BR price file (read-only, no repair applied)
      found 92 such rows across 58 tickers, 82 of them on exactly the two dates above — confirming
      the systemic-vendor-defect read, not a per-ticker coincidence. Unit-tested in
      `tests/build_dataset/test_repair.py` (isolated pulse repaired; a real permanent split left
      alone; a real volatile-but-ratio-flat price move left alone; short series don't crash).
      **Confirmed against a real rebuild 2026-08-21: `test_top50_ml_readiness.py` now reports
      29 READY + 21 READY WITH CAVEATS + 0 NOT READY** (was 9 NOT READY) — full fix, all 9
      previously-flagged tickers cleared.

## Already tracked elsewhere — cross-referenced, not duplicated here

- **§2a** (NaN `adj_close`, 259 rows, "flag only, no repair" already decided by owner) —
  `DATA_LAYER_CORRECTNESS_PLAN.md` §2a. `test_top_traded_quality.py`'s "2 NaN adj_close" finding
  and the LUXM4 finding above are both instances of this same, already-decided territory.
- **§2b** (98 of 202 in-panel deaths have no terminal event — research task, not an implementation
  step) — `DATA_LAYER_CORRECTNESS_PLAN.md` §2b. `test_raw_processed_reconciliation.py`'s "12
  uncovered dead tickers" (e.g. `OIBR3/4`, `RSID3`, `BDLL3/4`) is the same territory, surfaced by a
  different test.
