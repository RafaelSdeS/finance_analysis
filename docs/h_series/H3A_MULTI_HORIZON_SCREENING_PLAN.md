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
- `reversal_5d = −return_5d` (5-trading-day price return, sign-flipped — short-term reversal)
- `rsi_5` (5-day RSI, already computable the same way `rsi_14` is in Stage 2's `features.py`)
- `volume_shock_5d = volume_5d_avg / volume_20d_avg − 1` (short-window liquidity/attention shock)

Capped at 3 to keep the new multiple-testing family small and deliberate, not an open-ended
technical-indicator search.

**k=252 (long horizon), if Step 0 clears it:** reuse the *same* 16 characteristics H1 already
screened at k∈{21,63}, recomputed at k=252 — no new characteristic definitions needed, only a new
horizon for existing ones. This keeps the new family size exactly 16 (k=252) + 3 (k=5) = 19 tests,
a fixed, known number going into the FDR correction (see Implementation).

**No-peeking discipline:** this k-grid (5, 252) and this exact characteristic list were fixed
before examining any post-2024 data, the same discipline H1/H2/H3 already follow via
`CONFIRMATION_START`. Nothing here is chosen post-hoc from having looked at what "seemed to work"
in the recent segment.

## Implementation

- Extend `features.py::build_monthly_panel()`'s target/characteristic construction to compute
  forward returns at k=5 (new Stage 2 column, doesn't exist yet — `return_1m/3m/6m` cover
  21/63/126d, nothing at 5d) and k=252 (also new — nothing currently covers 12m). The 3 new k=5
  characteristics (`reversal_5d`, `rsi_5`, `volume_shock_5d`) are computed once at Stage 2 level,
  same as existing technicals; the 16 k=252 characteristics reuse Stage 2's existing fundamental
  computations, just resampled at the longer horizon.
- **FDR correction — hierarchical, not joint-with-H1:** H1's PASS is treated as a closed, frozen
  decision (it's referenced elsewhere in this project, e.g. `composite.load_survivors()`, as a
  stable artifact — reopening it every time a new horizon is tested would mean no decision in
  this project ever actually closes). H3a runs its **own, separate** BH-FDR correction across
  exactly its own new family: 19 tests (16 characteristics × k=252, plus 3 characteristics × k=5)
  if Step 0 clears k=252, or 3 tests (k=5 only) if it doesn't. H1's original characteristics keep
  their original H1 verdict at k∈{21,63} unchanged regardless of what H3a finds at the new
  horizons — this is standard hierarchical/stage-wise multiple-testing practice (correct within
  each pre-registered stage, don't retroactively re-pool across stages).
- **Redundancy check before counting a "new" survivor as incremental:** for each new survivor,
  compute its partial Spearman IC controlling for the nearest-correlated existing H1 survivor
  (e.g. `reversal_5d` against `momentum_vs_market_12m`/`momentum_vs_sector_12m`). If the partial
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

Same FDR/NW-HAC/sign-consistency gate as H1, applied jointly across the expanded
horizon × characteristic family. Any new survivors get compared via H3's own permutation-null and
quintile-monotonicity tests, run twice — once on H3's original survivor set, once on the enlarged
set — so the delta is directly attributable.

## Success Criteria

At least one new characteristic × horizon combination (a) survives H3a's own hierarchical FDR
gate, (b) survives the redundancy check (significant partial IC against its nearest existing H1
survivor), **and** (c) its inclusion moves H3's post-2024 bootstrap CI on the IR delta
(enlarged-set vs. original-set) to exclude 0 — the identical quantitative bar H3 itself uses, not
a separate, looser one.

## Failure Criteria

- No survivors under H3a's own FDR correction.
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
  the same power problem Step 0 checks for, not a separate issue to solve independently.
- Short-horizon characteristics risk re-entering the problem space M1-M4 already found empty
  (no daily cross-sectional alpha using price/technical features). A failure to find short-horizon
  survivors here would *corroborate* that earlier finding rather than being a new, surprising
  negative result — it's testing genuinely different (potentially non-price) short-horizon
  characteristics, not repeating M1-M4's exact test.

## Next Decision Gate

**New survivors + measurable improvement** → fold into H3's active feature set going forward,
rerun H4 on the updated score series.
**No improvement** → H3's original medium-horizon scope is final; proceed to Phase 3 (H4)
unchanged, using H3's original score series.
