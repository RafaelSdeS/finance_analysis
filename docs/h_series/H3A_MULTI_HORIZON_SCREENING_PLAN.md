# H3a — Multi-Horizon Characteristic Screening (Phase 2: Add Complexity, Only If Warranted)

**Status:** implementation-ready design, not yet coded. Depends on H3 PASS.

## Objective

Test whether characteristics measured at k=5 trading days (short) or k=252 trading days (long,
~12 calendar months) carry additional, independent predictive signal beyond the existing k∈{21,63}
set H1 already screened, to determine whether that medium-horizon-only scope was a real finding
or just where the project happened to look first.

## Rationale

H1 tested only k ∈ {21, 63} days. Nothing has been tested at a genuinely short horizon (closer to
the daily-frequency space M1-M4 already found no alpha in) or a genuinely long horizon (12m+,
closer to the project's stated "hold for years" objective). Given the actual goal is long-term
investing, it's a real gap that no long-horizon screen exists yet. This must come **after** H3,
not before or alongside it — introducing new characteristics at the same time as fixing H3's
anchor/blend would make it impossible to tell whether any resulting pass/fail came from the
construction fix or the new signal. Basic experimental design: isolate one variable at a time.

## Design

**Step 0 (prerequisite, gates whether k=252 runs at all):** before computing anything, run H0's
exact power-floor methodology (`milestone_h0.py`) at k=252 to get its min-detectable mean IC.
Compare against the actual IC magnitudes H1's real survivors landed at (0.035–0.088 — e.g.
`pl`=0.0522, `pvp`=0.0593, `momentum_vs_market_12m`=0.0876, `volatility_60d`=−0.0739, per
`H1_FINDINGS.md`'s gate table). If k=252's power floor exceeds that range, a "no survivors" result
at k=252 would mean *underpowered*, not *no signal* — report that directly as the k=252 outcome
and do not run the full characteristic screen at k=252 at all; it would be spending implementation
effort on a test that can't distinguish its own two possible conclusions. k=5 does not have this
problem (shorter horizon → more, not fewer, effective independent observations) and proceeds
regardless.

**Fixed characteristic list (k=5, short horizon), exact formulas — no others added without
returning to this doc first:**
- `volume_shock_5d = volume_5d_avg / volume_20d_avg − 1` (short-window liquidity/attention shock)
- `turnover_shock_5d = turnover_ratio_5d_avg / turnover_ratio_20d_avg − 1` (short-horizon version
  of `turnover_ratio`, mirroring `volume_shock_5d`'s own formula)

**Deliberately not a price-return/technical-indicator list.** An earlier draft included
`reversal_5d = −return_5d` and `rsi_5` — dropped. Both are pure price technicals, exactly the
feature family M1-M4 already tested at daily/short horizons and found carries zero cross-sectional
signal using a full conv-trunk model with far more power than a rank-IC screen would have here;
re-testing that same family with a weaker tool spends FDR budget re-asking a question this project
already has stronger evidence answers "no." Both replacement characteristics instead extend
`turnover_ratio` — one of H1's cleanest survivors (significant at **both** k=21 and k=63,
sector-neutral, per `H1_FINDINGS.md`) — to a short horizon. This is testing whether a family H1
already validated *elsewhere* also carries signal at k=5, a genuinely different and better-motivated
question than re-probing the family M1-M4 closed. Capped at 2 to keep the new multiple-testing
family small and deliberate, not an open-ended technical-indicator search.

**k=252 (long horizon), if Step 0 clears it:** reuse the *same* 16 characteristics H1 already
screened at k∈{21,63}, recomputed at k=252 — no new characteristic definitions needed, only a new
horizon for existing ones. **k=5 (2 tests) and k=252 (16 tests) are corrected as two separate FDR
families, not pooled into one 18-test family** — the same logic this document already uses to
justify running its own family separate from H1 (a large, low-powered family shouldn't loosen the
rejection threshold for a better-motivated one) applies again one level down: k=252 is the more
novel, better-motivated half; pooling it with k=5's low-expected-power tests would only dilute it
for no benefit, and splitting costs nothing.

**No-peeking discipline:** this k-grid (5, 252) and this exact characteristic list were fixed
before examining any post-2024 data, the same discipline H1/H2/H3 already follow via
`CONFIRMATION_START`. Nothing here is chosen post-hoc from having looked at what "seemed to work"
in the recent segment.

## Implementation

- **Targets need no Stage 2 work — corrected from an earlier draft.** `spine.py::build_forward_targets()`
  already takes `k` as a free parameter, computing forward relative returns directly from the daily
  `adj_close` panel `features.py::_load_daily_prices()` loads (see H3's Step 0); `build_monthly_panel()`
  already loops `k_horizons` and calls `build_forward_targets()` once per k. Widening
  `k_horizons=(21, 63)` to include 5 and 252 needs **zero new Stage 2 columns** for the target
  side — it's an argument change, not a pipeline change.
- Only the 2 new k=5 **characteristic** (predictor) columns (`volume_shock_5d`,
  `turnover_shock_5d`) genuinely need new computation, since `CHARACTERISTIC_COLUMNS` currently
  assumes every characteristic already exists as a precomputed `ml_dataset.parquet` column. Both
  are derivable from data `h_series` can already reach (`volume`, `turnover_ratio` are already
  Stage 2 columns; the 5d/20d rolling averages are the only new arithmetic) — whether that's
  computed as a genuine new Stage 2 column or directly inside `h_series` from an added daily-volume
  read is an implementation-time choice, not a blocker either way. The 16 k=252 characteristics
  need no new computation at all, just the existing `CHARACTERISTIC_COLUMNS` resampled at a longer
  target horizon.
- **FDR correction — hierarchical, not joint-with-H1, and split by horizon (see Design):** H1's
  PASS is treated as a closed, frozen decision (it's referenced elsewhere in this project, e.g.
  `composite.load_survivors()`, as a stable artifact — reopening it every time a new horizon is
  tested would mean no decision in this project ever actually closes). H3a runs **two separate**
  BH-FDR corrections: one over the k=5 family (2 tests), one over the k=252 family (16 tests, only
  if Step 0 clears it). H1's original characteristics keep their original H1 verdict at k∈{21,63}
  unchanged regardless of what H3a finds at the new horizons — standard hierarchical/stage-wise
  multiple-testing practice (correct within each pre-registered stage, don't retroactively re-pool
  across stages or across horizons within this stage).
- **Redundancy check before counting a "new" survivor as incremental:** for each new survivor,
  compute its partial Spearman IC controlling for the nearest-correlated existing H1 survivor
  (e.g. `turnover_shock_5d` against `turnover_ratio`). If the partial
  IC is not itself significant by the same NW-HAC/FDR standard, the "new" characteristic is
  redundant with an existing one, not incremental — exclude it from the enlarged survivor set
  used in the next step, regardless of its marginal-IC survival.
- If new, non-redundant survivors remain, rerun H3's combination layer on the enlarged survivor
  set as a separate, labeled comparison run — never silently overwriting the original H3 result.

## Expected Outcome

Either (a) new long/short-horizon characteristics survive the same rigorous gate H1 already
used, expanding H3's usable signal set, or (b) nothing new survives, and H1's original
medium-horizon scope is confirmed as the actual signal-bearing zone in this data rather than an
arbitrary starting point.

## Validation

Same FDR/NW-HAC/sign-consistency gate as H1, applied separately per horizon family (k=5, k=252 —
see Design/Implementation), not jointly. **Sign-consistency at k=252 is diagnostic, not gating:**
H1's ≥60%-sign-consistency-across-sub-windows check was implicitly sized for k∈{21,63}; at k=252
the same sub-window logic likely leaves too few independent sub-windows to mean anything (Risks).
Concretely, if the number of independent sub-windows available at k=252 falls below 3, report
sign-consistency at k=252 as "inconclusive" and do not let it fail a characteristic on its own —
the NW-HAC t-stat + FDR + redundancy check remain the operative gates for k=252 survivors. k=5 is
unaffected (shorter horizon gives more, not fewer, sub-windows) and uses the unmodified H1 gate.
Any new survivors get compared via H3's own permutation-null and quintile-monotonicity tests, run
twice — once on H3's original survivor set, once on the enlarged set — so the delta is directly
attributable.

## Success Criteria

At least one new characteristic × horizon combination (a) survives its own horizon-specific
hierarchical FDR gate (k=5's 2-test family or k=252's 16-test family, corrected separately — see
Design/Implementation), (b) survives the redundancy check (significant partial IC against its
nearest existing H1 survivor), **and** (c) its inclusion moves H3's post-2024 bootstrap CI on the
IR delta (enlarged-set vs. original-set) to exclude 0 — the identical quantitative bar H3 itself
uses, not a separate, looser one.

## Failure Criteria

- No survivors under H3a's own FDR correction (either horizon family).
- Survivors exist but all fail the redundancy check (duplicate existing signal, not new).
- Survivors clear both gates but the resulting IR-delta bootstrap CI still includes 0.
- k=252 specifically: Step 0's power floor already exceeds H1's observed survivor IC range —
  reported as "long-horizon screening is underpowered at this sample size," a distinct, honest
  conclusion from "no long-horizon signal exists," and not something further screening at k=252
  can resolve without more data.

In any of these cases, H3's medium-horizon-only scope stands as final, not provisional.

## Risks & Assumptions

- Expanding the multiple-testing family increases the FDR correction's severity — a real
  characteristic at a new horizon might fail to survive purely because the family got larger, not
  because the effect isn't real. This is a known cost of rigorous multi-hypothesis correction,
  not a flaw in the test, but worth stating so a non-survival isn't over-read.
- Sign-consistency sub-windows (H1's gate requires ≥60% sign-consistency across sub-windows) were
  implicitly sized around k∈{21,63}. At k=5, more/shorter sub-windows are natural and should
  give the check *more* power, not less. At k=252, the same sub-window logic likely leaves too
  few independent sub-windows to assess sign-consistency meaningfully at all — another symptom of
  the same power problem Step 0 checks for. **Resolved, not left open (Validation): below 3
  independent sub-windows at k=252, sign-consistency is reported as inconclusive and doesn't gate
  that characteristic** — the other three criteria (NW-HAC, FDR, redundancy) carry the k=252 gate
  instead.
- **Deliberately not re-testing the price/technical family M1-M4 already found empty** (no daily
  cross-sectional alpha from price/technical features) — the k=5 list (Design) was corrected to
  drop `reversal_5d`/`rsi_5` for exactly this reason and instead extends `turnover_ratio`, one of
  H1's actual survivors, to a short horizon. A failure to find k=5 survivors here would still be
  informative on its own terms (does a *validated* family's signal persist at a much shorter
  horizon), not merely re-confirm M1-M4 by construction the way the dropped candidates would have.

## Next Decision Gate

**New survivors + measurable improvement** → fold into H3's active feature set going forward,
rerun H4 on the updated score series.
**No improvement** → H3's original medium-horizon scope is final; proceed to Phase 3 (H4)
unchanged, using H3's original score series.
