# Data Layer Correctness Plan

**Split out of the former `DATA_LAYER_HARDENING_PLAN.md` on 2026-08-20 (that file has been removed —
it was never committed, and its entire contents live across these two plans).** It mixed bug-fixing
with refactoring; the two have different risk profiles and different sequencing. Pure reorganization
now lives in **`DATA_LAYER_ORGANIZATION_PLAN.md`** and runs *behind* this one.

**Goals, in priority order:**
1. **All monetary values in real currency units** — full BRL / full USD everywhere, never "thousands".
2. **Every derived column means one thing** — no column that is a percent in one row and a fraction in another.
3. **Proof they work** — invariants that fail loudly, not conventions held in comments.

**Status:** §1 is DONE end to end as of 2026-08-20 — normalization code (steps 1/1b/2) committed
(`6ddc959`), migration (step 3) run over the full 695-ticker crosswalk (612 written, 83 skipped for
no price file — matches the crosswalk exactly, confirmed NOT ATIVO-scoped), `ml_dataset.parquet`
rebuilt and `scale_features.py` re-fit (step 5). §3's Edits 1–5, §2c's fix, and §4/§5/§6 slices also
applied (⚠️ APPLIED / ✅ APPLIED). `tests/run_all.py --group fast`: **55/55 pass**.

**Invariant test results (`tests/build_dataset/test_unit_scale_invariants.py`), post-rebuild:**
BR 4/5, US 5/5. All three headline identities (`vpa*shares==equity`, `lpa*shares==net_income`,
`book_to_market*pvp==1`) pass 100% on both markets — confirms the migration worked (PETR4 spot
check: `equity` 481,854,000 → 481,854,000,000, exactly ×1000). Margin-scale check (redesigned
pooled, not per-ticker — see the test's own docstring) also passes both markets, confirming §2c.
**One open, pre-existing finding, out of §1's scope:** `market_cap/shares == close` fails on
**19/549 BR tickers** (worst: TIMS3, ratio ~99x, n=1979) — `market_cap`/`shares_outstanding`/`close`
were never touched by §1's currency-unit fix, so this predates this work; looks like stale
shares-outstanding not tracking a real split (TIMS3's `close` ranges 0.000038→18.79 across its
history). Not investigated further here — needs the same per-event verification rigor as the
continuity map, not a quick patch.

**Also fixed in this pass, unrelated to §1's currency-unit scope but directly exposed by running
real data through the rebuilt pipeline:**
- `scale_features.py`'s `RATIO_COLUMNS` and `src/portfolio/features.py`'s `GROWTH` list both
  referenced `cagr_revenue_5y`/`cagr_earnings_5y` (intermediate, pre-fill columns dropped before
  the final dataset is written) instead of the real `..._final` columns. Both always broken —
  masked by synthetic-fixture tests (`test_scale_features.py`, `test_features.py`) that manufacture
  a column for every listed name regardless of whether it's real. Fixed both, plus
  `test_features.py`'s hardcoded 120/121 count assertions → 118/119. **This one was live-breaking**:
  `alpha.py` derives its design matrix straight from `feature_columns()`, so Stage 3 would have
  hard-crashed (`KeyError`) on the next run, not just failed a test.
- `validate_vs_yfinance.py` hardcoded its own local `K = 1000` in two places
  (`_print_fund_rows`, `check_internal_consistency`) — a third, independent copy of the
  thousands-convention assumption §1 was written to eliminate. Post-migration this double-scaled
  every recomputed ratio and fundamentals comparison by 1000x (`lpa` read 1589.67 vs BolsAI's 1.59,
  a fake +99900% "mismatch"). Fixed; re-run: `OVERALL: PASS`, all real vendor diffs back to a sane
  0–8% range.
- `terminal_events.parquet` needed a re-run after the `ml_dataset.parquet` rebuild (documented in
  CLAUDE.md's Run Commands as a required post-`build_ml_dataset.py` step) — not a bug, just an
  ordering step that hadn't happened yet. Re-ran: same 104 rows as before (§2b's territory,
  untouched by §1).

**Real pre-existing findings surfaced by running the DATA group post-rebuild, reported but NOT
fixed here** (each is a different subsystem than §1's currency-unit scope — `market_cap/shares==
close` on 19 BR tickers, `cagr_revenue` coverage, 4 tickers' single-day NaN holes, `pl` freeze rate,
CAMB3/LUXM4 price quality, 9 top50 NOT READY tickers, 12 uncovered dead tickers): moved to their own
tracked file, **`DATA_LAYER_FOLLOWUP_FINDINGS.md`**, so this doc stays scoped to §1's own history.

Revert with `git checkout -- . && rm tests/data_collection/test_yf_collectors.py tests/build_dataset/test_unit_scale_invariants.py src/data_collection/ratios.py`
— note this does **not** revert the migration's data changes (`data/raw/br/fundamentals/*.parquet`,
`ml_dataset.parquet`, the scaler); those need `git checkout -- data/raw/br/fundamentals/` separately
if ever needed (processed/ outputs are gitignored and just need a re-run of the old code).

All findings measured read-only against the real tree: 1,328 BR price files, 612 BR fundamentals,
8,283 US fundamentals, `ml_dataset.parquet` at 1,706,604 rows / 567 tickers / 22,832 ticker-quarters.

**Re-verified live 2026-08-20** (every number re-measured, not carried over):
`book_to_market * pvp` median **0.001**, 0.00% of rows within 1% of 1.0 · `vpa * shares / equity`
median **1000.0** · **259** NaN `adj_close` rows, **0** flagged `degraded=1` · `terminal_events`
**104** rows vs **202** in-panel deaths · **716** price files with no `company_info` row.
Corrections made in that pass are marked **[rev 2026-08-20]**.

> ### ⚠️ Tolerance discipline — read before writing any check in this document
> Every identity here is asserted at a **10% band**, because different vendors legitimately disagree
> by a few percent on the same line item. A tighter tolerance measures **provenance** ("was this
> computed from exactly these stored numbers"), not **correctness**, and on a multi-vendor panel it
> produces hundreds of thousands of false alarms. Measured directly: at 0.1%, seven columns "fail"
> on 500K+ rows; at 10%, all but two are clean and the offenders sit within 0.6–4.5%. The real bugs
> in this document are off by **100×** and **1000×** — they do not need a tight tolerance to find.

---

## §0.5 — Which market is this plan about?

**~85% BR.** A full invariant sweep against both built datasets — 23,220 BR ticker-quarters,
123,804 US — at the 10% band:

| identity | BR | US |
|---|---|---|
| `net_debt == total_debt - cash` | 100.0% | **100.0%** |
| `roe`, `roa`, `net_margin`, `current_ratio` | 88–99.9% | **100.0%** |
| `debt_equity`, `asset_turnover`, `ebit_margin` | 97.8–100% | **100.0%** |
| `lpa * shares`, `vpa * shares`, `pl` | 98.7–99.4% | **100.0%** |
| `book_to_market * pvp == 1` | **0.0%** 🔴 | **100.0%** |
| `ebitda_margin` scale (§2c) | **0.0%** 🔴 | n/a — 0 rows |

**US passes every single check at 100%.** Strong independent confirmation of §1's core claim that
US is the reference implementation and needs no change. It also means every 🔴 here is BR-only.

Per-section applicability, so nobody re-derives it:

| § | BR | US | note |
|---|---|---|---|
| §1 units | **all of it** | verify only | US already `unit_scale=1`; now empirically confirmed |
| §2a NaN `adj_close` | yes | **unmeasured** | same shared `yf_collectors` price path — **check before assuming clean** |
| §2b terminal events | yes | no | US survivorship is a separate, accepted decision (`US_COLLECTOR_FIX_PLAN` §4) |
| §2c `ebitda_margin` | yes | no | `ebitda` never collected for US |
| §3 periodicity | yes | **yes** | Edit 3's guard exists *because* of a US-only crash |
| §5 proof | shared | shared | |
| §6 coverage gaps | yes | no | all items are BR paths |

- [ ] **Open gap: §2a was never measured on US.** `collect_prices_yf` is the *same shared function*
      for both markets, so the NaN-`adj_close` hole is structurally possible there too. The BR
      number (259 rows) is measured; the US number is unknown. Measure before closing §2a.

---

## §0 — Verified healthy (bounds where bugs can hide)

- **No lookahead.** 0 violations on all four checks: `fundamentals_available_date <= trade_date`,
  `reference_date <= fundamentals_available_date`, `days_since_fundamental >= 0` and equal to the
  real date difference, `filing_lag_days <= 180` (max observed 179).
- **TTM standardization works.** `lpa * shares == net_income * 1000` on **100%** of 9,274 sampled
  BR ticker-quarters; flows are genuinely TTM, not YTD. BOLSAI_EXIT_PLAN Task 0/1 succeeded.
  **[rev 2026-08-20] Re-measured over all 21,429 ticker-quarters (not a sample): 98.8% hold within
  a 10% band** — claim stands. Note for whoever re-runs it: at 0.1% only 56.8% hold, and the
  residual is **era-split within individual tickers** (AALR3: 0% exact for 2016–2025, 60% for
  2026), i.e. a BolsAI-era vs CVM-era provenance mix, not an arithmetic error. §1's full-panel
  rebuild collapses that mix as a side effect. **Do not re-tighten below ~10%** — it would measure
  which vendor produced the row, not whether the row is right.
- **Feature identities hold**: `overnight_gap + intraday_return == log_return` (max dev 1.8e-15),
  `f_score == sum(f_*)` exactly.
- **0 duplicate `trade_date`** across all 1,328 price files.
- **123 thin files (<10 rows) in `data/raw/`; 0 reach `ml_dataset`.** Quality filters work.
- **Continuity map well-maintained** (31 verified events); MRFG3→MBRF3 correctly deduped.
- All `_merge_save` call sites pass a real validator (count corrected to 12 in §5).

---

## §1 — Normalize all money to real currency units (**the headline change**)

### The problem

The same column means different things in different markets, in a **shared** code path.

| | `equity` for the last quarter | `equity / shares` vs `vpa` | convention |
|---|---|---|---|
| **BR** (PETR4) | `481,854,000` | 0.0374 vs 37.386 — **1000× apart** | BRL **thousands** |
| **US** (AAPL) | `1.0752e11` | 7.359865310083953 vs 7.359865310083953 — **exact** | USD **units** |

Set in code at `cvm/ratios.py:65` (`k = 1000.0`) versus `sec/fundamentals.py:102`
(`compute_ratios(..., unit_scale=1)`). `build_us_dataset.py` reuses BR's feature stages unchanged,
so `features.py` runs over both conventions with no idea which it's holding.

### The two bugs this already caused

Every ratio that stays *within* one scale is fine (`cash_ratio`, `revenue_per_earning`,
`net_debt_to_assets`, `working_capital_ratio`). Every ratio that **crosses** thousands↔units is
wrong. Exactly two crossings exist, and **both are broken** — a 2/2 failure rate.

> ⚠️ **[rev 2026-08-20] `ebitda_margin` was originally listed here as "fine". It is not — see §2c.**
> Different defect (percent-vs-fraction, not thousands-vs-units), different layer (Stage 2, not the
> collector), which is why the crossing analysis missed it. The crossing count stays 2; the
> *unit-convention* defect count is **3**, and §1's fix-at-the-source strategy does **not** catch
> the third one.

**1a. `book_to_market` (`features.py:344` — `equity / market_cap`)**

| | median |
|---|---|
| stored | **0.00057** |
| correct | **0.56701** |

`book_to_market * pvp` = 0.001 where it must be 1.0; **0%** of rows within 1% of correct. The 1000×
gap is uniform across every year 2011–2026. Correct for US, wrong for BR — same line of code.

**1b. `dividend_coverage_ratio` (`features.py:556` — `ebitda / (div_value_12m * shares)`)**

| | median |
|---|---|
| stored | **0.00** |
| correct | **3.10** |

Ratio exactly 1000.0. (`payout_ratio = div_value_12m / lpa` is **correct** — both per-share BRL.
Don't "fix" it.)

### The fix: normalize at the source, so `features.py` never sees two scales

`compute_ratios(..., unit_scale=...)` is already the designed seam. Set it to 1 everywhere and scale
the inputs once, at ingest. **US already does this — it becomes the reference implementation and
needs no change.**

- [x] ✅ **APPLIED 2026-08-20 — `cvm/ratios.py`** — scale the level columns ×1000 once after `_ttm()`, then set `k = 1.0`
      (or delete `k`). Every ratio it computes stays **numerically identical**: `pl`/`pvp`/`p_*`/
      `ev_*`/`lpa`/`vpa` already multiply by `k`, and `roe`/`roa`/margins/`asset_turnover`/
      `current_ratio`/`debt_equity`/`net_debt_*`/`roic` are scale-invariant. Only stored levels move.

      ⚠️ **[rev 2026-08-20] Scale the RAW inputs, not the output names.** The "11 level columns"
      listed below are *stored output* columns, and 4 of them do not exist yet at the scaling
      point (right after `_ttm()`, line 48): `cash`, `total_debt`, `net_debt` and `ebitda` are
      **derived at lines 53–62**, from raw columns with different names. Scale this set instead —
      the derived four then inherit the scale for free:
      `net_income`, `equity`, `net_revenue`, `ebit`, `total_assets`, `current_assets`,
      `current_liabilities`, `gross_profit`, `depr_amort`, `cash_caixa`, `cash_aplic`,
      `debt_st`, `debt_lt`.
      `gross_profit`/`depr_amort` are not stored but **must** be scaled anyway — they feed
      `gross_margin` (scale-invariant, unaffected) and `ebitda` (a stored level, affected).
      Scaling the output names instead would *miss* `ebitda`/`net_debt`, leaving `ev_*`,
      `p_ebitda`, `net_debt_*` silently wrong — the exact bug class this fixes.
      `market_cap` is computed at line 64 from `close_price * shares_outstanding`, both already
      units: leave it, and leave the scaling point **above** it.
- [x] ✅ **APPLIED 2026-08-20 — `yf_collectors.py::collect_fundamentals_yf`** — dropped the two `/ K`
      divisions (they used to convert yfinance's already-full-BRL figures *down* into thousands) and
      `compute_ratios(base)` now passes `unit_scale=1` explicitly. The module-level `K` is retired
      from `yf_collectors.py` (no longer imported there — it still lives in the moved `ratios.py`
      as `compute_ratios`'s own default for the BolsAI/CVM callers).
- [x] ✅ **APPLIED 2026-08-20 — `br/collectors.py::collect_fundamentals`** (BolsAI) — the response's
      11 level columns are now scaled ×1000 on ingest, so the paid path obeys the same convention if
      `DATA_SOURCE` is ever flipped back. Cold path, but leaving it is how the convention silently
      breaks later.
- [x] ✅ **VERIFIED 2026-08-20 — `features.py:344` and `:556`** — both become correct with **no edit**,
      confirmed unchanged. No interim ×1000 added; that would double-count after normalization.
- [x] ✅ **VERIFIED 2026-08-20 — `sec/*`** — no change, all six import sites already pass
      `unit_scale=1` explicitly: `sec/fundamentals.py:102`, `sec/fds.py:200` + `:219` + `:501`,
      `sec/tenq.py:311`, `sec/companyfacts.py:659`, `sec/selected_financial_data.py:645`.

      **Cross-plan note:** `DATA_LAYER_ORGANIZATION_PLAN.md` §O1 (`compute_ratios` + `FUND_FULL_COLS`
      → `src/data_collection/ratios.py`) is done — see that plan's status. All six `sec/*` imports
      plus `br/collectors.py`'s `FUND_FULL_COLS` import now point at `..ratios`, not `..yf_collectors`.

**The 11 level columns that change (BR only):** `net_income`, `equity`, `net_revenue`, `total_debt`,
`ebitda`, `ebit`, `net_debt`, `cash`, `total_assets`, `current_assets`, `current_liabilities`.
`market_cap`, `shares_outstanding`, `close_price` are already units.

**Validators need no changes** — checked: `validate_fundamentals` only tests a CAGR null rate, and
`validate_us_fundamentals` uses purely relational tests (`equity > total_assets`,
`cash > total_assets`). All scale-invariant. This materially de-risks the migration.

### Migration

- [x] ✅ **[rev 2026-08-20] VERIFIED — in-place replacement works.** The claim was that
      `cvm/ratios.py` rebuilds the full history and `_merge_save` dedups `keep="last"`, so new rows
      replace old ones with no migration script. Exercised against the **real**
      `storage._merge_save` + `validate_fundamentals` with a synthetic BolsAI-era file (63 quarters,
      2010-12-31..2026-06-30, thousands) overwritten by a CVM-scale rebuild (units). 6/6:

      | check | result |
      |---|---|
      | validation passed, write happened | PASS |
      | row count unchanged, no duplicate append (63→63) | PASS |
      | values replaced in place (`net_income` 1.0 → 1000.0) | PASS |
      | zero duplicate ticker-quarters | PASS |
      | sorted by `reference_date` | PASS |
      | stale BolsAI-only column blanked to NaN (63/63) | PASS |

      Mechanism confirmed at `storage.py:102`: `concat([df_old, df_new])` then
      `drop_duplicates(subset=["ticker","reference_date"], keep="last")` — `df_new` is concatenated
      last, so it wins every collision. **No migration script needed.**

      ⚠️ **Two things this run also established:**
      1. **`_merge_save` returns `None` and writes NOTHING when validation fails.** A migration
         driver that ignores the return value would report success while changing nothing. Check it
         — this is the same hazard §5's audit item covers, and it silently no-op'd this very
         verification twice before it was noticed.
      2. **Quarters the rebuild doesn't emit survive at the old scale.** Confirmed with a file
         holding pre-2010 quarters: 8 rows stayed in thousands beside migrated rows — a genuine
         1000× discontinuity *inside* one ticker. **Measured risk in practice: nil.** A 40-file
         sample of `data/raw/br/fundamentals/` has earliest `reference_date` = **2010-12-31 in
         every file, 0 files starting earlier** (median 63 rows = the full CVM span). CVM's floor
         and the data's floor are the same date, so there is nothing to orphan. Re-check if any
         pre-2010 fundamentals are ever added.

- [x] ✅ **DONE 2026-08-20 — migrated correctly, NOT via `pipeline.py`.** Ran
      `build_fundamentals(tickers=None, rebuild=True)` directly (no CLI exposes `rebuild=True` over
      the full crosswalk — `cvm_statements.py --step fundamentals` defaults `rebuild=False`).
      Result: `612 written, 83 skipped (existing/no prices)`, and **83 + 612 = 695 = the exact
      crosswalk size** — confirmed read-only afterward, proving every crosswalk ticker was
      attempted, not just the 565 ATIVO ones. Spot-checked the previously-stranded sample
      (`AELP3`, `ALSC3`, `APER3`, `BFRE11`, `BIDI3/4/11`, `BLUT3`) — all rewritten at the migration's
      timestamp. The 115-file trap below did **not** recur.

      🔴 **Do NOT migrate via `pipeline.py`. It would silently leave 115 files in thousands.**
      `collect_fundamentals_cvm(tickers, mode)` forwards the caller's already-scoped list to
      `build_fundamentals(tickers=..., rebuild=True)`, and `pipeline.py` scopes that list to
      **`_active_tickers()` (status == ATIVO)**. Measured: 612 fundamentals files on disk vs **565
      ATIVO tickers — 115 files belong to non-ATIVO tickers** (`AELP3`, `ALSC3`, `APER3`, `BFRE11`,
      `BIDI3/4/11`, `BLUT3`, …), written by the separate `br/cvm_statements.py` delisted path.
      A pipeline-driven migration rebuilds the 565 and never touches the 115.

      This is **worse than the leftover-rows risk it replaces**: it doesn't produce a mid-history
      discontinuity inside one ticker, it produces a **clean 1000× split across the panel** — 497
      tickers in units, 115 in thousands, every one of them internally consistent. Cross-sectional
      z-scores (`cross_sectional.py`) and the single panel-wide `alpha.py` model both silently
      ingest two currencies. And per §2b these 115 are disproportionately the delisted names whose
      terminal payoffs the label already under-covers.

      Migrate with `build_fundamentals(tickers=None, rebuild=True)` over the **full crosswalk**, not
      through `pipeline.py`. Then re-run the invariant test **per ticker, not pooled** — a pooled
      median of a 497/115 mix still reads ≈1.0 and hides it.
- [ ] Watch for BolsAI-era leftover rows in tickers CVM doesn't cover — those would stay in
      thousands and produce a mid-history 1000× discontinuity **within a single ticker**. Distinct
      from the whole-file gap above; the per-ticker invariant test catches both. **Not explicitly
      re-checked** — the invariant test's per-ticker median would only flag this if it dominates a
      given ticker's history; a short thousands-scale prefix inside an otherwise-fixed ticker could
      still slip past a median-based check. Low measured risk (§1's Migration section: all sampled
      fundamentals files start at CVM's own floor, 2010-12-31, so there's no pre-2010 tail to orphan).
- [x] ✅ **DONE 2026-08-20 — rebuilt `ml_dataset.parquet` (auto-snapshotted `dataset_v{N}`, manifest +
      split_config written) and re-fit the scaler.** `scale_features.py` hit an unrelated pre-existing
      bug on the real rebuild (`RATIO_COLUMNS` listed `cagr_revenue_5y`/`cagr_earnings_5y`, which
      don't exist in the output — only the `..._final` columns do; fixed, see this doc's Status
      section). Owner ran the rebuild + migration; the scaler fix and re-run were done in this pass
      once real data exposed the bug.

### The guard that makes it stick

- [x] ✅ **APPLIED + RUN 2026-08-20 — one invariant test, run for BOTH markets**:
      `tests/build_dataset/test_unit_scale_invariants.py` (new, DATA group). Asserts the scale
      identities that already hold in the vendor layer — `vpa * shares == equity` ·
      `lpa * shares == net_income` · `market_cap / shares == close` · `book_to_market * pvp == 1` —
      per ticker (min 5 valid rows), failing on the worst offender at the 10% band, not a pooled
      median. **Post-rebuild result: 4/5 BR, 5/5 US.** The three headline identities pass 100% on
      both markets — see this doc's Status section for the PETR4 spot check and the one remaining
      finding (`market_cap/shares==close`, 19 BR tickers, pre-existing, out of §1's scope).
- [x] ✅ **APPLIED + RUN 2026-08-20 — margin-scale consistency**, same file, **redesigned mid-flight**:
      the original per-ticker spread design (largest pairwise ratio among the 4 `*_margin` columns'
      medians, >10x fails) produced false positives on genuine near-zero-`net_revenue` distress
      quarters (one BR ticker's `ebit_margin` legitimately reads -31,023,867% — CLAUDE.md documents
      these as intentionally kept unclipped). Replaced with a **pooled** median ratio of each margin
      against `gross_margin` (the anchor), computed only over rows where both are in a plausible
      [0.5, 1000] range — this mirrors how the real §2c bug was actually found (a uniform panel-wide
      factor, not a per-ticker one) and is immune to individual distress-row blowups. Passes both
      markets post-rebuild: BR `ebitda_margin/gross_margin` pooled median **0.607** (pre-fix this
      would have read ~0.004) confirms §2c landed.
- [ ] Record the convention in CLAUDE.md: **all monetary values are full currency units (BRL/USD);
      no column is ever denominated in thousands.** Today it lives only as `K = 1000` in a module
      BR fundamentals no longer route through.

---

## §2 — Other data-correctness bugs

### 🔴 2a. 259 NaN `adj_close` rows reach the dataset unflagged

Three reasonable decisions compose into a hole:

1. `validate_prices` excludes `adj_*` from its NaN check, justified as *"already flagged downstream
   via `adj_close_precision_degraded`"*.
2. `features.py:206` — `(adj_close > 0) & (adj_close < 0.05) & quantized`. **`NaN > 0` is False, so
   the flag can never fire on NaN**; `0.0 > 0` is False, so never on an exact zero either.
3. `test_final_dataset.py`'s prefix-NaN rule covers only `equity`/`net_income`/`total_assets` —
   never a price column.

The validator waves through exactly what the flag cannot catch, and no test looks. All 259 NaN rows
carry `degraded == 0`; LUXM4 has 288 raw `adj_close == 0` rows with only 24 flagged.

| ticker | NaN | note |
|---|---|---|
| EPAR3 | 253 | 2006, `close` = 0.002 — genuine microcap underflow |
| BPAC11 | 2 | 2026-07-20 / 07-31, `close` ≈ R$56, volume ≈ 4.8M |
| MAPT4 / CAMB3 / HBRE3 | 2 / 1 / 1 | |

- [ ] Widen the flag to what consumers assume: `isna() | (<= 0) | (near_floor & quantized)`.
      Order matters: `np.isclose(NaN, NaN)` is False, so `isna()` must be the **first** OR term,
      not folded into `quantized`.
- [ ] Correct the false justification comment in `validate.py:74-76`.
- [ ] Extend the prefix-NaN test to `adj_close`.

- [x] ✅ **[rev 2026-08-20] DECIDED by owner: flag only, no repair.** A BPAC11 refetch was considered
      and dropped — 6 rows in one small ticker, not worth touching the price path during the
      migration. The three bullets above are the whole of §2a. For the record this leaves those 6
      rows NaN-but-flagged, which is the intended end state, not an oversight. EPAR3's 253 rows are
      unrecoverable regardless (2006 microcap underflow at `close` = 0.002).

### 🔴 2b. 98 of 202 in-panel deaths have no terminal event (48%)

`terminal_events.parquet` has 104 rows; 202 tickers stop trading inside the panel. The rest just
end, so `forward_excess_return()` goes NaN instead of a realized payoff. **BRFS3 is one**, still
marked `ATIVO`, despite BRF being absorbed into Marfrig/MBRF3 in 2025. Per CLAUDE.md's own
measurement — in-panel deaths mostly die *rising* — leaving these NaN systematically drops
acquisition-premium outcomes from the label.

> ⚠️ **This is a research task, not an implementation step.** "Triage the 98" is per-ticker
> verification against B3/CVM, and the continuity map's bar is "each entry individually verified."
> Do not schedule it alongside two-line fixes. Nothing else in this plan depends on it.

- [ ] Run `terminal_events.find_rename_candidates()`; triage the 98 into real delistings, unspliced
      renames needing a `ticker_continuity.json` entry (BRFS3 likely `merger`/`keep_separate`), and
      names still trading that the ATIVO gate wrongly dropped.
- [ ] Verify a sample against B3/CVM before hand-adding — the map's entries are each individually
      verified and that bar is worth holding.

### 🔴 2c. `ebitda_margin` is a fraction; every sibling margin is a percentage — **FIXED**

Found by a fresh invariant sweep 2026-08-20, not present in any earlier audit. **100% of BR rows**,
ratio exactly **0.01** at every quantile from p01 to p99 — a uniformly missing `× 100`, not vendor
noise (a clean constant across the whole distribution is the tell).

| column | median | convention |
|---|---|---|
| `ebitda_margin` | **0.1295** | **fraction** ⚠️ |
| `ebit_margin` | 11.13 | percent |
| `net_margin` | 5.77 | percent |
| `gross_margin` | 29.93 | percent |

EBITDA margin is normally *higher* than EBIT margin (it adds back D&A); here it read 100× smaller.
Sample (AALR3 2016-06-30): `ebitda` 142,503 / `net_revenue` 942,696 → stored **0.151**, correct
**15.12**.

**Both vendor layers were correct.** `cvm/ratios.py:83` and `yf_collectors.py:792` each compute
`ebitda / net_revenue * 100`. The defect was that **Stage 2 recomputed the column and dropped the
`× 100`** at `features.py:586`. Every sibling margin passes through from the collector untouched;
this one alone was recomputed, wrongly. **This is why §1's "normalize at the source" strategy would
not have caught it** — the source was already right.

**A test locked the bug in.** `test_features.py:977` asserted
`("ebitda_margin", "net_revenue", 100.0 / 500.0)` — expected **0.2**, a fraction. The case was
written to regression-test `_safe_ratio`'s NaN handling and took the then-current output as
"expected".

**Why the vendor validators never caught it:** `validate_vs_yfinance.py:160` and
`validate_us_vs_vendor.py:163` both check `ebitda_margin == ebitda/net_revenue*100` — but against
**raw collector output**, which is correct. The defect was introduced after they run. Nothing
validated the built dataset's margin scale.

**Blast radius** — a declared model input, not an unused column: `scale_features.py:51`
(RobustScaler) · `build_top50_universe.py:90` · `src/portfolio/features.py:22` (Stage 3 alpha
input) · `features.py:754` (feeds `ebitda_margin_zhist_5y`, also a Stage 3 input at
`portfolio/features.py:56`). LightGBM splits are monotone-invariant and the `*_zhist_5y` robust
z-score is scale-invariant, so **measured backtest impact is likely nil** — but any linear model,
any cross-margin comparison, and any human reading the column was wrong.

- [x] ✅ **APPLIED 2026-08-20** — `* 100` added at `features.py:586` (with a comment recording why
      it is load-bearing), and the expected value at `test_features.py:977` corrected to
      `100.0/500.0*100`. Both in the same edit, as required. `tests/run_all.py --group fast`:
      **55/55 pass.** Not yet reflected in `ml_dataset.parquet` — needs the rebuild.
      Verified unaffected: `test_history_relative.py`'s `_FUND_LOC_SCALE` synthesizes
      `ebitda_margin` directly and never calls `compute_advanced_features`; the `*_zhist_5y`
      z-score is scale-invariant either way.
- [ ] The margin-scale guard is tracked in §1's "guard that makes it stick" — one test, both
      concerns.
- [ ] US: no action. `ebitda` is never collected (0 rows), so `ebitda_margin` is 100% NaN there —
      already recorded in `manifest.empty_columns` and `DATA_INTEGRITY_TEST_PLAN.md` D5.

---

## §3 — Periodicity consistency

Mostly consistent on a **4-quarter basis**: `*_growth_yoy` (`pct_change(4)`), `f_*_improving`
(`shift(4)`), `*_trend_4q` (`diff(4)`), `cagr_*_5y`, `div_*_12m`, `payout_ratio`. TTM flows over
point-in-time stocks for `roe`/`roa`/`asset_turnover`/margins — standard practice. ✓

### 🟡 The fundamentals-delta grid is ragged, and 3 columns are misnamed

**Keep both horizons — they are complementary, not redundant.** For a TTM flow,
`diff(1)` = newest quarter vs the same quarter a year ago (fast, catches a turn immediately) and
`diff(4)` = trailing year vs the prior year (slow, confirms the trend). A fast/slow pair, like two
moving averages. The `*_trend_4q` family already *is* the `diff(4)` half.

The real problem is that which horizons exist depends on which metric you look at:

| metric | basis | `diff(1)` | `diff(4)` |
|---|---|---|---|
| `gross_margin` | TTM / TTM | `gross_margin_qoq` | **was missing** |
| `net_margin` | TTM / TTM | `net_margin_qoq` | `margin_trend_4q` |
| `roe` | TTM / point-in-time | `roe_qoq` | `roe_trend_4q` |
| `roa` | TTM / point-in-time | **was missing** | `roa_trend_4q` |
| `debt_equity` | point-in-time / point-in-time | `debt_equity_qoq` | `debt_trend_4q` |
| `current_ratio` | point-in-time / point-in-time | `current_ratio_qoq` | **was missing** |

6 metrics × 2 horizons = 12 slots; 9 were filled.

**Naming is only wrong where the inputs are TTM.** `_TTM_COLS` (`cvm/ratios.py:16`) is flows only —
`net_revenue`, `gross_profit`, `ebit`, `net_income`, `depr_amort`. Balance-sheet items are never
TTM'd, so:

- `debt_equity_qoq`, `current_ratio_qoq` — built from point-in-time balance-sheet items. `diff(1)`
  here is a **genuine quarter-over-quarter change, correctly named. Leave them alone.**
- `gross_margin_qoq`, `net_margin_qoq` — TTM/TTM. `diff(1)` is a single-quarter *year-over-year*
  delta, because subtracting two 4-term rolling sums cancels the three shared quarters.
- `roe_qoq` — mixed: TTM numerator, point-in-time denominator, so it blends both readings.

Measured on the mixed case: `roe_qoq` lag-1 autocorrelation **−0.356** vs **+0.293** for `roe`
`diff(4)`, median abs 2.31 vs 4.19.

- [x] ⚠️ **APPLIED (code only) — grid completed to 12**: `gross_margin` `diff(4)`, `roa` `diff(1)`,
      `current_ratio` `diff(4)`. **[rev 2026-08-20] All 5 edits are in the working tree** (this
      section previously read "Not yet implemented" — that was stale). Verified present:
      `features.py` `roa_qoq` at :378, `trend_cols` gains `gross_margin` + `current_ratio`,
      presence guard `if col in q.columns else np.nan` in the loop body; `scale_features.py`
      `RATIO_COLUMNS` +3; `build_top50_universe.py` `fundamental_cols` +1.

      🔴 **The rebuild has NOT run.** `ml_dataset.parquet` contains **neither** `roa_qoq` nor
      `gross_margin_trend_4q` — confirmed against the live parquet schema. The three columns exist
      in code and nowhere else, so **none of this is verified against real data yet**, and the
      US-build KeyError that Edit 3's guard exists to prevent has never been exercised.

      Edit 1's no-guard justification **re-verified**: `roa` is read unguarded at `features.py:397`
      (`f_roa_positive`) and `:398`, same function scope, same unconditional path. The new
      `roa_qoq` at :378 runs *earlier*, so an absent `roa` would now crash sooner — but it would
      already have crashed. No new failure mode. ✓

      Exact scope, as scouted:

      **Edit 1 — `features.py`, the `*_qoq` block.** `g["roa_qoq"] = g["roa"].diff(1).where(qoq_ok)`.

      **Edit 2 — `features.py`, `trend_cols`.** Add `"gross_margin": "gross_margin_trend_4q"` and
      `"current_ratio": "current_ratio_trend_4q"`. New keys use the metric's full name; the four
      existing entries keep their historical (inconsistent) output names — `margin_trend_4q`,
      `debt_trend_4q` — because three registries reference those strings.

      **Edit 3 — `features.py`, the `trend_cols` loop body: a presence guard is REQUIRED.** The
      loop does `q[col].diff(4)` unguarded. `gross_margin` is 93.6% null in the US build, and
      `load_fundamentals`'s per-file `dropna(how="all")` drops an all-NaN column *outright* — so it
      is **absent, not NaN-valued**, and the unguarded read KeyErrors. Same hazard already handled
      for `book_to_market`/`ebitda` at `features.py:344`/`:354`. **This is a US-build crash, not a
      BR one — it would not show up in a BR-only test run.**

      **Edit 4 — `scale_features.py::RATIO_COLUMNS`.** Unlisted columns never reach the RobustScaler.

      **Edit 5 — `build_top50_universe.py::fundamental_cols`.** This list zero-fills
      fundamental-derived columns on `has_fundamentals == 0` rows. Its own comment records that the
      original list *missed* sibling columns and produced "an inconsistent missing-data signal"
      (`TOP50_UNIVERSE_ML_READINESS_AUDIT.md` §1.3) — omitting the three new columns here recreates
      exactly that bug. Guarded by `if col in df.columns`, so safe.

      **NOT edited — `src/portfolio/features.py`.** ✅ **DECIDED by owner 2026-08-20: Stage 2 only.**
      The requirement is that the three columns are **present in `ml_dataset.parquet` and correct**
      — nothing more. Rationale, for whoever reopens it: `alpha.py:96` uses the keep-list as the
      literal design matrix, and `alpha.py:98` sets no `feature_fraction`/`bagging_fraction`, so
      LightGBM evaluates every feature at every node. Three extra columns change the greedy split
      path, hence every tree, hence alpha → weights → turnover → net return, at every walk-forward
      rebalance. It also spends a `trials.csv` entry, raising the deflated-Sharpe multiple-
      comparisons bar (see `PORTFOLIO_IMPROVEMENT_PLAN.md`'s STOP banner). Not worth spending on
      three features with no prior reason to expect signal.

      **Acceptance:** after the rebuild, `roa_qoq`, `gross_margin_trend_4q` and
      `current_ratio_trend_4q` exist in `ml_dataset.parquet`, are non-null outside warm-up, and
      match `diff(1)`/`diff(4)` of their source metric. Nothing in Stage 3 changes.

- [ ] 🟡 **DEFERRED — rename the three TTM-based `diff(1)` columns** (`gross_margin_qoq` →
      `gross_margin_yoy_1q`, and likewise `net_margin_qoq`, `roe_qoq`) to say what they measure.
      Do **not** rename `debt_equity_qoq`/`current_ratio_qoq` — those names are already accurate.

      **[rev 2026-08-20] Why deferred:** this is cosmetic, and it touches **`src/portfolio/features.py:32-33`**
      — the Stage 3 keep-list the owner scoped out. It is *safe* (a rename doesn't change values, so
      LightGBM sees an identical matrix and backtest results would not move), but it is 4 files plus
      tests for zero data-correctness gain. Blast radius, verified: `portfolio/features.py:32,33` ·
      `build_top50_universe.py:98` · `scale_features.py:59` · `features.py:375-377` ·
      `test_features.py:613,619,626`. The names are misleading; the numbers are right. Pick this up
      only when someone is already in those files.
- [ ] Record the convention in CLAUDE.md beside the units convention: **`_qoq` means adjacent-quarter
      and is only valid on point-in-time inputs; on TTM inputs a 1-lag diff is a single-quarter YoY.**
- [ ] Note for later: none of this yields a *true* one-quarter change on TTM-based metrics. That
      needs single-quarter (non-TTM) flows, which the pipeline deliberately doesn't retain. Separate
      decision if it's ever wanted — `_ttm()` would have to keep the pre-sum column.

---

## §5 — Proof it works

- [x] ⚠️ **Roster drift guard.** `run_all.py` hand-lists 71 scripts; two things had already fallen
      off — `test_top50_ml_readiness.py` (478 lines) and `tests/api/*` — **never run in any group**.
      Added `roster_drift()` + `EXCLUDED`, failing the run if a test file on disk is in no group.
      Lists stay explicit (FAST-vs-DATA isn't derivable from a filename); only omission becomes
      impossible. **[rev 2026-08-20] Executed and green** — the guard ran clean across 55 scripts.
      (`test_top50_ml_readiness.py` went into **DATA**, not FAST.)
- [ ] **Second drift trap, still open.** **32** files hand-list their own test functions in
      `__main__`; a test added there silently never runs. **Re-audited 2026-08-20 by AST: 0
      currently uncalled** — confirmed a trap, not a wound. The proposed ~15-line AST check was
      **prototyped and works**: parse each test file, collect `FunctionDef`s named `test_*`,
      collect every `Name`/`Attribute` id referenced inside the `if __name__ == "__main__"` block,
      fail on the set difference. Skip files containing `pytest.main`.
      Skipped: rewriting all 32 to `pytest.main([__file__])` — a 32-file diff for what one guard
      catches. (Naming a convention for *new* tests is an organization concern — see
      `DATA_LAYER_ORGANIZATION_PLAN.md` §O5.)
- [ ] **The §1 cross-market invariant test** is the single highest-value item in this document.
- [ ] **Audit `_merge_save` return handling at all 12 call sites** (**[rev 2026-08-20]**, was "11";
      grep returns 14 hits, of which `cvm/ratios.py:149` and `validate.py:130` are docstring/comment
      mentions). It returns `None` on validation failure; a caller that checkpoints anyway turns
      that into permanent silent data loss on resume. Spot-checked `yf_collectors.py:649` — correct.
      The 12: `br/collectors.py` :131 :197 :251 :392 · `yf_collectors.py` :633 :649 :741 :911 :1014
      :1075 · `cvm/ratios.py` :177 · `us/fred_collectors.py` :46.
      **Not hypothetical** — this exact silent no-op defeated the §1 migration verification twice.
- [ ] **Checkpoint skip semantics.** Verify `SKIP_REPROBE_INTERVAL` can't strand a ticker forever
      and that `mark_skip`/`clear_skip` are symmetric.
- [ ] **[rev 2026-08-20] One unexplained test failure, unreproduced.** A single run reported
      54 passed / 1 failed; five subsequent runs were 55/55 and the failing script name was not
      captured. Probably load contention, but unproven. If it recurs, **capture the script name** —
      that is the whole ask.

**Not doing:** replacing `run_all.py` with bare pytest — its subprocess-per-script isolation may be
load-bearing for the `main()`-style files, and it has a `NON_BLOCKING` concept for vendor tests.

---

## §6 — Collection coverage gaps

- [x] ⚠️ **463 lines of tests moved out of `src/`.** `yf_collectors._demo()` was a third of the
      module with a 27-line `tests/` shim that only called it. Moved verbatim to
      `tests/data_collection/test_yf_collectors.py`. **1,548 → 1,078 lines.**
      **[rev 2026-08-20] Now executed and passing** (1.33s) — previously only "parses, not executed".
- [x] ⚠️ **Dead code deleted.** `pipeline._data_tickers()` + `_tickers_with_company_info()` (never
      called — ruff misses both); `collectors.collect_corporate_events()` + `collect_sectors()`
      (zero call sites, unreachable even by flipping `DATA_SOURCE`).
- [x] ✅ **[rev 2026-08-20] 716 missing `company_info` rows — MEASURED, NO ACTION NEEDED.** Closed
      by measurement rather than backfill; the decision this item asked for is "don't".

      **714 of the 716 never reach `ml_dataset` at all**, and the reason is decisive: **all 714 have
      zero fundamentals files.** The quality gate excludes them on that basis alone, so adding a
      registry row would change nothing — they would still be dropped. (Sample: `ABNB3`, `ABRE3`,
      `ACES3/4`, `ADHM3`, `AEDU3`, `AESL3/4`, `AFLU3/5` — delisted names recovered by
      `collect_delisted.py` for price history only, exactly as intended.)

      **The remaining 2 (`AXIA5`, `AXIA6`) are correct, not leaks.** They are also the *only* two
      dataset tickers lacking a fundamentals file. Both carry the right `sector`
      ("Emp. Adm. Part. - Energia Elétrica"), `status` ATIVO, and non-null fundamentals — sourced
      via **sibling fill** from `AXIA3`/`AXIA7`, which share CNPJ `00.001.180/0001-26`
      (AXIA ENERGIA S.A.). Share classes of one company legitimately share financials, the same way
      `PETR3`/`PETR4` do. Documented mechanism, working as designed.

      **Backfilling 716 registry rows would be pure churn.** Reopen only if a ticker without a
      fundamentals file ever needs to reach the dataset on its own.
- [ ] **`refresh.py` never refreshes `company_info`/`sectors`/`corporate_events`** — only
      `pipeline.py --mode update` does. Follow CLAUDE.md's "one-command top-up" advice and a status
      change or new split never lands. Fold the three free stages in (CVM CAD + yfinance, cheap).
- [ ] **Small follow-on, not a bug:** the count "716 of 1,328 price files have no `company_info`
      row" reads alarming and is now permanently misleading in isolation. Wherever it survives as a
      metric, pair it with "714 of which are correctly gated out of the dataset."
- [ ] **105 tickers marked ATIVO have stale price files** (VVAR11 last traded 2018-11-23, PCAR4
      2020-02-28, FJTA4 2019-11-11). Overlaps the 98 from §2b — **same research-task caveat applies,
      don't schedule it as an implementation step.**

---

## §7 — Documentation drift (correctness-of-facts only)

Structural docs work — the `docs/` index, the module tables, the graph rebuild — lives in
`DATA_LAYER_ORGANIZATION_PLAN.md` §O6.

- [ ] CLAUDE.md says "~293 tickers" (actual: 1,328 raw price files, 567 in the dataset) and
      "fundamentals to 2026-03-31" (actual: 2026-06-30).
- [ ] Add the **units convention** (§1) and **periodicity convention** (§3) — the two facts that
      would have prevented three of the bugs in this document.
- [ ] **CLAUDE.md's `DATA_SOURCE` note needs the migration caveat from §1**: the ATIVO-scoped ticker
      list means `pipeline.py` is not a whole-panel rebuild tool. That property is invisible at the
      call site and caused the 115-file gap.

---

## Sequencing

```
0. tests/run_all.py --group fast                     ← DONE 2026-08-20, 55/55 green
1. §1 invariant test, PER TICKER, 10% band (written FIRST, expected to FAIL on BR)  ← DONE 2026-08-20
1b. [Organization plan §O1] compute_ratios -> data_collection/ratios.py             ← DONE 2026-08-20
      ← the one organization item that runs early: a pure move, no behavior
        change, and step 2 edits that function's signature
2. §1 normalization: cvm/ratios.py (scale RAW inputs), yf_collectors, br/collectors  ← DONE 2026-08-20
      fast suite re-verified green after 1b and after 2 (55/55 each time)
--- NOT YET RUN: steps 3+ mutate real data (600+ git-tracked files / rebuild). Owner's call. ---
3. §1 migration: build_fundamentals(tickers=None, rebuild=True) over the FULL crosswalk
      -- NOT via pipeline.py (ATIVO-scoped, would strand 115 files)
      -- CHECK the return value; _merge_save no-ops silently on validation failure
      -> invariant test passes per ticker, both markets
4. §2a NaN flag (flag-only, decided)
4b. Measure §2a on US (same shared collector, never checked)
5. REBUILD ml_dataset -> snapshot dataset_v{N} -> re-fit scaler        ← OWNER RUNS THIS
      -> materializes §3's 3 new columns and §2c's ebitda_margin fix
      -> exercises Edit 3's US guard for the first time
6. §5 AST trap, _merge_save audit (12 sites), checkpoint semantics
7. §6 refresh.py gap                                 ← pull forward if a run is imminent
8. §7 docs (units + periodicity conventions, CLAUDE.md drift)
--- research tasks, not implementation steps, no dependencies on the above ---
9. §2b 98 terminal events  +  §6 105 stale-ATIVO tickers
--- then, behind a green suite ---
10. DATA_LAYER_ORGANIZATION_PLAN.md
```

Steps 2–5 all change `ml_dataset.parquet`, so they want **one** rebuild, not four.

**Rebuilds are the owner's to run.** Nothing in this plan should trigger one unattended.
