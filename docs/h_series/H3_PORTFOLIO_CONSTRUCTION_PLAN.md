# H3 — Fitted Portfolio Construction (Phase 1: Validate the Premise)

**Status:** implementation-ready design, not yet coded.

## Step 0 (prerequisite, not templated below)

**Answered, not open:** `features.py::_load_daily_prices()` already reads exactly this — daily
`adj_close` per ticker (+ BOVA11) straight from `ml_dataset.parquet`, pivoted wide by
`trade_date` — for `build_forward_targets()`'s own use. The anchor's trailing-covariance lookback
reuses this same function/source, not a new one. What's still genuinely open (not answered by
this) is the second Risks bullet below: that the eligible-ticker universe aligns exactly between
`_load_daily_prices()`'s wide frame and the monthly decision panel's active-universe membership —
that's a real check, not a five-minute one, and stays open until verified.

**Leakage guardrail (non-negotiable part of Step 0, not optional polish):** the daily return
series feeding `estimate_cov()` must strictly terminate at *T−1* relative to the monthly
rebalance's execution date T. Because the anchor uses a *daily* covariance to set weights for a
*monthly* execution, an off-by-one here — including T's own return in the lookback window used to
decide a trade executed at T's close — is a lookahead leak: it means the covariance estimate used
to size the trade partly depends on the same day's price the trade executes at. `T`'s return must
be programmatically masked/excluded when building the lookback slice, not just conventionally
omitted. Note this is a distinct question from whether `risk_portfolios.py::trailing_returns()`'s
existing "`ending at t (inclusive)`" convention is safe to copy — that convention was reasoned
through for the R-series' own daily-indexed `PricePanel`/`run_backtest` loop, where day `t`'s
close *is* the same close the backtest trades at, by that loop's own construction. Whatever daily
source H3 ends up using may not share that same timing convention with h_series' `decision_date`,
so the T−1 cutoff must be verified against the *actual* source chosen here, not assumed safe by
analogy.

---

## Objective

Replace H2's two hand-picked pieces — the cap-weight anchor and the ad hoc γ-tilt — with fitted
equivalents, to test whether the 10 characteristics that survived H1's rigorous screen carry real
incremental value once that specific confound is removed.

## Rationale

H2 already did the hard part correctly in one place: its ridge regularization strength (λ) was
selected via walk-forward CV (`composite.py::select_lambda`), not guessed. It still failed its
own decision gate, and the diagnostics say exactly why: the untilted cap-weight anchor *alone*
already had IR = 0.9544 pre-2024 (se 0.00276, t = 2.187, n = 62); the full tilted composite only
reached IR ≈ 1.12; and the date-permutation null (shuffling scores across dates, rerunning) put
the *real* signal's IR at roughly the 22nd percentile of a null distribution centered at 1.27
(p = 0.78). In plain terms: a portfolio built from randomly shuffled scores did as well as the
real ones. The anchor was asserted, not fit — and it was doing virtually all the work.

This must happen before Phase 2 (new characteristics, H3a) or Phase 3 (behavior layer, H4)
because both would silently inherit the same confound if left in place — there's no point
teaching a stateful hold/sell layer to trust a score whose portfolio-construction step is already
known to be broken. It also must run on the **same H1 survivor set H2 used**, not an expanded
one, so that a pass or fail here is attributable to the anchor/blend fix alone.

## Design

Three components, replacing H2's two hand-picked pieces:

**(a) Combination layer** (extends `composite.py`) — generalize the existing ridge-only
`walk_forward_scores`/`select_lambda` pair into a `select_model()` that also competes ElasticNet
and a shallow GBM (max_depth ≤ 3) against ridge, selected by the same pre-2024-only mean OOS
Spearman IC criterion `select_lambda` already uses. `composite.py`'s docstring currently states
ridge is deliberately the *only* model considered ("linear-on-ranks is the max complexity this
dataset has earned"). Widening this is justified specifically because letting walk-forward CV
choose the model class is *more* data-driven than fixing it by assertion, not less; it's kept to
2 extra candidates, not an open-ended search, for the same reason the original ladder was chosen.

*GBM regularization constraint:* `max_depth ≤ 3` alone is not enough. On ~90 monthly rows with
highly correlated fundamental/technical characteristics, a shallow-but-unconstrained tree still
greedily isolates small, noisy clusters — where ridge instead gracefully shrinks the same
collinear features. The GBM candidate must additionally enforce `max_features='sqrt'` (feature
subsampling per split) and a high `min_samples_leaf` (floor to be set relative to the ~62-row
pre-2024 training window, not a fixed absolute count copied from a larger-data default). The
train-IC-vs-OOS-IC gap must be logged per model class. **Disqualification rule, mechanical, not a
manual call:** if GBM wins the raw CV-IC selection but its train/OOS IC gap exceeds 2× ridge's own
gap on the same folds, GBM is automatically disqualified and `select_model()` falls back to the
best of ridge/ElasticNet instead — a fixed, pre-registered rule, not a judgment call left for
later, since this document's whole premise is that no stage gets to hand-pick an exception.

**(b) Anchor layer** (new `anchor.py`) — per decision date, build trailing daily returns for the
month's eligible tickers, feed through `risk_portfolios.py`'s already-implemented
`estimate_cov()` (Ledoit-Wolf shrinkage, handles the L~126-250, N≤50 near-singular regime with no
tuning) into `min_variance_weights()` (long-only simplex QP) or `risk_parity_weights()` (Spinu's
convex reformulation, cyclical coordinate descent). Both solvers are reused as library calls,
verbatim — they were built and validated for the R-series and need no changes.

**Anchor-type selection is fit, not asserted — the same discipline as everything else in this
document.** Picking min-variance vs. risk-parity by hand would be the identical "asserted, not
fit" mistake H2 was killed for, just moved one layer over. Selection criterion: on pre-2024 data
only, re-estimate both anchor types per expanding fold exactly as they'll run live (no fitted
parameter beyond the trailing covariance itself, so this isn't circular the way tuning a model
hyperparameter against the reported metric would be), and pick whichever anchor-alone series has
the higher realized Sharpe/IR over that segment. That choice is then frozen and confirmed once on
the untouched post-2024 segment, same as `select_lambda`'s λ and component (a)'s model class.

**(c) Blend layer** (new `blend.py`) — `w_posterior = w_anchor + κ·(w_view − w_anchor)`, where
`w_view` is stage-(a)'s score renormalized onto the simplex and κ ∈ [0,1] is *derived* from the
winning model's own pre-2024 out-of-fold predictive skill — not grid-searched against the
reported IR the way H2's γ was (that grid-search-against-the-reported-metric is precisely the
circularity being fixed here). **κ is anchored to out-of-fold mean Spearman IC, not R².** OOF R²
against realized cross-sectional equity returns is notoriously noisy and routinely negative (a
model can easily predict worse than the historical cross-sectional mean out of fold, which is a
normal, uninformative outcome in this setting, not evidence of zero skill) — constraining a
negative R² into [0,1] either breaks the formula, forces a hard floor at 0 with a discontinuous
weight jump right at that floor, or manufactures false confidence. IC is what this project
already scores every stage by (`select_lambda`'s own selection criterion, H1's gate, H0's power
floors), so anchoring κ to it is also the more internally consistent choice, not just the more
stable one. Concretely: `κ = sigmoid(k · (IC_OOF − IC_min))`, where `IC_min` is **H0's own
pre-registered minimum-detectable mean IC** (0.0300 at k=21, `H0_FINDINGS.md`) — reusing an
already-established, non-arbitrary threshold from this project rather than inventing a new one —
and `k` is a fixed slope constant, **derived from two already-established project numbers, not
swept against the reported backtest IR** (that sweep is exactly the circularity κ itself is
designed to avoid). Two calibration points, both reused rather than invented: κ=0.5 at
`IC_OOF = IC_min` (true by construction of the sigmoid's center) and κ≈0.9 at `IC_OOF = 0.088`
(H1's single strongest real survivor IC, `momentum_vs_market_12m`, per `H1_FINDINGS.md` — i.e. "if
the model's OOF skill matches the best signal this project has actually measured, trust the view
strongly"). Solving `sigmoid(k·(0.088−0.0300)) = 0.9` gives `k = ln(9)/0.058 ≈ 37.9`. This is fixed
before any backtest is run, exactly like `IC_min` itself — not a free knob left to taste.

## Implementation

- **Data flow:** `H1_FINDINGS.json` survivors (`composite.load_survivors()`, unchanged, still
  raises if H1 isn't a fresh PASS) → `composite.build_feature_matrix()` (unchanged) → generalized
  `walk_forward_scores(panel, X, feature_cols, target_col, folds, model_factory)`, refit per
  expanding fold exactly as today, just with `model_factory` swapping in `Ridge(alpha)`,
  `ElasticNet(alpha, l1_ratio)`, or `GradientBoostingRegressor(max_depth<=3)` → `select_model()`
  picks `(model_class, hyperparams)` maximizing pre-2024 mean IC, same `CONFIRMATION_START` guard
  H2 already enforces (2024-03 → 2026-07 untouched by any hyperparameter choice) → score series
  stitched across all folds including the confirmation segment, scored once.
- **Anchor:** `estimate_cov()` (Ledoit-Wolf + trace-relative jitter for PD-ness) →
  `min_variance_weights()` (SLSQP, bounds `[0, max_weight]`, equality constraint sum=1) or
  `risk_parity_weights()` (closed-form per-coordinate update, no bounds needed by construction) —
  **type selected** by pre-2024 anchor-alone Sharpe/IR (Design §b), not hand-picked. Both already
  handle the fallback chain (`_solve_with_fallback` — jitter retry, then an analytic
  inverse-variance/inverse-vol proxy) built for the R-series; reused as-is.
- **Blend:** normalize the view score to the eligible-universe simplex; κ from the winning
  model's stitched pre-2024 OOF mean Spearman IC via `κ = sigmoid(k · (IC_OOF − 0.0300))`, with
  `k = ln(9)/0.058 ≈ 37.9` (Design §c derivation — fixed from `IC_min` and H1's best survivor IC,
  not tuned against the resulting backtest IR).
- **Constraints:** monthly rebalance (unchanged cadence); `max_weight` passed straight into
  `min_variance_weights()`'s existing parameter; turnover cost `c_sell=c_buy=0.03%`
  (`src/rl_agent/config.py::CostConfig`, reused so cost assumptions stay consistent project-wide).
- **Validation:** rerun `milestone_h2.py`'s existing `permutation_null()` (cross-date) and
  quintile-monotonicity check verbatim against H3's blended weights and score series; add a new
  `within_date_permutation_null()` (same module) for the complementary cross-sectional test —
  same signature/return shape as `permutation_null()` so both slot into the same downstream
  reporting code.

## Expected Outcome

If the H1 survivors carry real, learnable incremental information beyond a risk-based prior, the
blended portfolio's IR should exceed the anchor-alone IR by a margin that survives the
permutation-null test — on **both** the pre-2024 selection window and the previously-untouched
post-2024 confirmation segment (H2's own breakdown only ever examined this pre-2024).

## Validation

**Two permutation-null tests, not one — they isolate different null hypotheses, and passing only
one wouldn't rule out the other's failure mode:**

1. **Cross-date test** (H2's existing `permutation_null()`, reused verbatim, not modified): each
   date's *intact* real score cross-section — full inter-ticker structure preserved — gets
   reassigned to a different, randomly-drawn date's real universe/anchor/forward-returns.
   Compares the real (correctly-dated) IR's percentile against the null distribution this
   produces. Tests whether *this specific date's* score cross-section maps to *its own* outcomes
   better than to a randomly-matched other date's — i.e., whether the score-to-return mapping is
   meaningfully date-specific, not an artifact of which dates happened to be volatile.
2. **Cross-sectional (within-date) test** (new, complementary): for each date independently,
   shuffle scores *across the eligible tickers on that same date* before passing them to the
   blend layer, preserving that date's exact score distribution and market conditions while
   breaking the specific ticker-to-score pairing. This isolates pure within-date ranking skill —
   closer to a standard Fama-MacBeth-style rank test — separately from any date-level effect the
   cross-date test alone could conflate with real skill.
3. OOS score-quintile monotonicity: mean realized forward return should be monotonic (or
   near-monotonic) across score quintiles.
4. Bootstrap CI on the IR delta (blended vs. anchor-alone), checking it excludes 0.
5. Repeat all four, unmodified, on the post-2024 confirmation split.

## Success Criteria

- Blended IR exceeds anchor-alone IR with **both** permutation-null tests (cross-date and
  within-date) at **p < 0.10, on both splits** (H2 only had the cross-date version break down
  pre-2024; requiring both tests on the untouched post-2024 segment too is a deliberately harder
  bar than H2 cleared).
- Quintile monotonicity holds to the same standard H2 used (adjacent-quintile ordering, not
  necessarily perfectly monotonic across all 5).
- Bootstrap CI on the IR delta excludes 0 on both splits.

## Failure Criteria

- Permutation-null p ≥ 0.10 on **either** test (cross-date or within-date) on either split —
  passing only one would leave open exactly the failure mode the other test exists to catch
  (date-level artifact vs. pure ranking-skill artifact), so either one failing is a real failure,
  not a partial pass. Failing the cross-date test specifically repeats H2's exact failure mode
  with the confound removed, meaning the confound wasn't the whole story.
- Quintile monotonicity fails even with a fitted anchor and blend.
- The anchor *alone* (before any blend) underperforms BOVA11/CDI — a more fundamental finding
  than H3 was designed to test, echoing R1-R3's result that structural, no-alpha policies don't
  clear the bar either; would mean the anchor choice (min-variance vs. risk-parity) itself needs
  re-examination before any scoring model can be evaluated on top of it.

## Risks & Assumptions

- Daily price source for the anchor's covariance is resolved (Step 0) — `_load_daily_prices()`,
  already reused as-is, not a new source.
- Assumes the eligible-ticker universe can be aligned exactly between the monthly h_series panel
  and whatever daily source is used — misalignment would silently bias the covariance estimate.
  This is the item Step 0 refers to as still genuinely open.
- ~90 monthly observations across an estimated 2-3 independent multi-year cycles limits how much
  can be trusted even from a properly-fit procedure (H0's own power-floor analysis already
  established this ceiling) — a PASS here is evidence, not proof.
- **This directly interacts with the Success Criteria's bar, and the two should be read together.**
  Passing requires 4 roughly-independent conjunctive tests (cross-date permutation, within-date
  permutation, monotonicity, bootstrap CI) on each of 2 splits. Even a real, moderate effect has a
  non-trivial chance of missing at least one of the 4 by chance alone at ~62 pre-2024
  observations — that is the cost of raising the bar over H2's single-test standard, not a flaw in
  the tests themselves, but it means a **narrow miss on exactly one test (e.g. 3 of 4 clearing
  comfortably, one at p≈0.11-0.13) should be reported and investigated as borderline**, not
  written up identically to a clean, uniform failure across all four.
- GBM, despite CV selection, may still overfit more than ridge on this sample size; the
  train-IC-vs-OOS-IC gap should be reported per model class as a diagnostic, not just the winner
  picked silently.

## Next Decision Gate

**PASS** → proceed to Phase 2 (H3a: multi-horizon screening).
**FAIL** → do not proceed to H3a/H4 on this foundation. Report as a substantive negative result —
H1's survivors carry nothing beyond a risk-based prior at this sample size — and fall back to the
risk-based anchor alone (structurally the same conclusion the R-series already reached for
no-alpha policies) as the practical baseline, rather than continuing to build behavior/reporting
layers on top of an unproven scoring signal.
