# H3 — Fully Data-Driven Portfolio Construction (detailed)

**Status:** draft — design-level detail, not yet implemented.

## Objective

Fix the one diagnosed defect in H2: the anchor and the tilt-blend were asserted, not fit, and
that's where nearly all of H2's reported IR came from (untilted cap-weight anchor alone: IR
0.95 pre-2024, t=2.19; full tilted composite: IR 1.12; permutation-null IR distribution centered
at 1.27, p=0.78 — the tilt's contribution over the anchor was indistinguishable from noise).
H3 replaces both hand-picked pieces with fitted/optimized equivalents and reruns H2's own
validation tools against the result.

## Reused unchanged

- `src/h_series/features.py::build_monthly_panel()` — characteristics + raw targets.
- `src/h_series/spine.py::iter_expanding_folds()` — same expanding-window spine as H0/H1/H2.
- `src/h_series/milestone_h1.py` — the FDR/NW-HAC survivor gate. H3 never runs without a fresh
  H1 PASS (`composite.py::load_survivors()` already enforces this for H2; same guard applies).
- `src/h_series/composite.py::build_feature_matrix()` — rank-normalized survivor features.
- `src/rl_agent/risk_portfolios.py::estimate_cov()`, `min_variance_weights()`,
  `risk_parity_weights()` — the pure numeric solvers only (NOT `make_risk_weight_fn`, which is
  coupled to `PricePanel`/`environment.run_backtest`'s daily-index loop; H3 runs over the
  h_series monthly `decision_date` panel instead, so it needs its own thin adapter loop, not
  the existing wrapper).
- `src/h_series/milestone_h2.py`'s `permutation_null()` and quintile-monotonicity check —
  rerun as-is against H3's output.

## Stage 2 — Combination layer (extends `composite.py`)

`composite.py`'s docstring currently states ridge is deliberately the *only* model
("linear-on-ranks is the max complexity this dataset has earned"). Revising that stance is
justified here specifically because letting the data choose the model class is *more*
data-driven, not less — but keep it to 2 extra candidates, not an AutoML sweep:

- [ ] Generalize `walk_forward_scores(panel, X, feature_cols, target_col, folds, alpha)` into
      `walk_forward_scores(panel, X, feature_cols, target_col, folds, model_factory)` — same
      per-fold refit loop, `model_factory` swaps in `Ridge(alpha)`, `ElasticNet(alpha, l1_ratio)`,
      or a shallow `GradientBoostingRegressor`/`LGBMRegressor` (max_depth<=3, few hundred trees —
      this dataset has ~90 monthly decision dates, a deep GBM would just overfit faster than ridge did).
- [ ] Generalize `select_lambda()` into `select_model()`: same pre-2024-only mean-IC criterion
      (`CONFIRMATION_START` untouched), grid now spans model class × its own hyperparameter grid.
      Selection stays walk-forward and out-of-sample — this is the one part of H2 that was
      already correctly data-driven; H3 keeps the discipline, just widens what's being selected.
- [ ] Output a comparison table (per target: `k21_raw`, `k21_sector_neutral`, `k63`) of each
      model class's selected hyperparameters + pre-2024 mean IC, so the model choice itself is
      auditable, not silent.

## Stage 3 — Anchor layer (new: `src/h_series/anchor.py`)

- [ ] Per decision date, build trailing daily returns for the panel's active tickers over a
      lookback window (mirrors `risk_portfolios.trailing_returns`, but sourced from the h_series
      monthly panel's own price history rather than `PricePanel` — confirm exact source against
      `features.py`/`loaders.py` before writing this; not yet verified which module already has
      daily closes joined in at this stage).
- [ ] `estimate_cov()` (Ledoit-Wolf shrinkage, already handles the L~126-250, N<=50 regime) →
      `min_variance_weights()` and `risk_parity_weights()` (both already built, reused verbatim).
- [ ] Produce one anchor weight vector per decision date per policy — this replaces the
      hand-picked cap-weight/equal-weight choice entirely; the anchor is now a solver output.
- [ ] Report both policies' standalone IR (same diagnostic H2 ran for cap-weight) before any
      blending — need to confirm the *learned* anchor alone still beats or matches the old
      hand-picked anchor's 0.95 IR; if it doesn't, that's a finding in itself.

## Stage 4 — Blending layer (new: `src/h_series/blend.py`)

Ladder call: start with the cheapest thing that could work, escalate only if it's insufficient.

- [ ] **Primary (rung 1):** shrinkage-style interpolation —
      `w_posterior = w_anchor + κ · (w_view − w_anchor)`, where `w_view` is the H3-stage-2 score
      renormalized to a simplex and κ ∈ [0,1] is *derived* from the winning model's own pre-2024
      out-of-fold IC/R² (higher OOF skill → higher κ), not grid-searched against the reported IR
      the way H2's γ was. This directly closes the exact circularity that made H2's blending
      indefensible.
- [ ] **Stretch (rung 2, only if rung 1 underperforms):** full Black-Litterman — reverse-optimize
      implied prior returns `π = δ·Σ·w_anchor`, set view uncertainty `Ω` from the model's OOF
      residual variance, solve the standard BL posterior. More principled, more moving parts
      (needs a risk-aversion δ) — don't build this unless rung 1 demonstrably needs it.

## Stage 5 — Constraints

- [ ] Monthly rebalance (unchanged cadence throughout H0-H3).
- [ ] Max-weight cap — reuse `min_variance_weights()`'s existing `max_weight` param.
- [ ] Turnover penalty at the blend/execution step — reuse the EIIE cost rates
      (`c_sell=c_buy=0.03%`, `src/rl_agent/config.py::CostConfig`) so cost assumptions stay
      consistent across the whole project rather than inventing a second number.

## Stage 6 — Validation

- [ ] Rerun `milestone_h2.py`'s `permutation_null()` against the H3 blended portfolio.
- [ ] Rerun the quintile-monotonicity check.
- [ ] Same pre-2024 (selection) / post-2024 (confirmation, untouched) split H2 already used.
- [ ] Compare against: (a) H3's own anchor-alone (no blend), (b) H2's old cap-weight-anchor
      result, (c) BOVA11, (d) CDI.

## Stage 7 — Decision gate (define before running stage 6)

- [ ] PASS requires: blended portfolio's IR exceeds its own anchor-alone IR by a margin that
      survives the permutation-null test at p < 0.10, on **both** pre-2024 and post-2024 splits
      (H2 only had this break down on the pre-2024 selection window — post-2024 confirmation is
      the harder, honest bar).
- [ ] If it fails: the conclusion is that the H1-surviving characteristics carry nothing beyond
      a risk-based prior at this sample size — a real, useful negative result, not a bug to
      chase.

## Open implementation questions (resolve before coding stage 3)

- Where does h_series currently get daily close prices from, if at all — `features.py` builds a
  *monthly* panel; stage 3 needs daily trailing returns for the covariance estimate. May need a
  new loader call, not yet identified.
- Universe alignment: H1/H2 run over the same eligible-ticker set every month; stage 3's
  `eligible_mask`-equivalent needs to match that set exactly, or the anchor and the view will be
  scored over different universes.
