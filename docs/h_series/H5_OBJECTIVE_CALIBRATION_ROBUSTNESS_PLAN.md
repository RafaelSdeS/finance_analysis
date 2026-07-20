# H5 — Objective Calibration + Robustness (Phase 3: Optimize & Prepare to Deploy)

**Status:** draft design — less grounded than H3/H4 (drafted ahead of seeing their results).
Depends on H4 PASS.

## Objective

Make the implicit risk-aversion assumption behind H3's anchor an explicit, deliberately chosen
(or evaluated-across-a-range) decision, and stress-test whether the H3+H4 result is robust or a
single-path artifact.

## Rationale

H3's min-variance/risk-parity anchor implicitly encodes a risk-aversion stance — per
`risk_portfolios.py`'s own docstring, it's the degenerate case of Merton's portfolio problem
(`w* = (1/γ)Σ⁻¹(μ−r)`) when μ is treated as homogeneous — without that stance ever being
deliberately chosen; it's whatever falls out of the solver's default formulation. Separately, H2's
failure was specifically a single-path artifact that only became visible once tested against a
permutation null — the standing lesson from that failure is that *any* positive result in this
project should be assumed fragile until proven otherwise by an explicit robustness pass, not
trusted on the strength of one successful backtest.

## Design

**(a) Risk-aversion:** either fix a specific stance (e.g. via the already-available
`vol_target_overlay()` in `risk_portfolios.py`) or evaluate H3+H4's result across a small swept
range and report sensitivity rather than silently picking one.

**(b) Robustness — two distinct questions, not one, and the cheap one runs first:**

1. **Fixed-decision sensitivity (primary, cheap):** freeze H3+H4's already-selected model,
   anchor, blend, and thresholds exactly as chosen. Replay the *identical* historical backtest
   under perturbed realized costs and a perturbed eligible universe, without refitting or
   reselecting anything. This answers "is the strategy I already have robust to my assumptions
   being slightly wrong" — the actual question this stage exists to ask, and it's cheap because
   nothing is refit.
2. **Reselection robustness (stretch, expensive, only if (1) raises concerns):** rerun H3+H4's
   full fitting/selection pipeline under the same perturbations, to check whether the *selection
   process itself* is stable. This requires full walk-forward refits per grid cell and should not
   be run by default — flag it as a follow-up, not a default part of this stage.

Block-bootstrap resampling (reusing `rl_agent/metrics.py::block_bootstrap_ci`) of the H3+H4
backtest's return series tests the same fixed-decision question from a different angle (return
sequence uncertainty rather than assumption uncertainty).

## Implementation

- `block_bootstrap_ci`: block length = 12 months (matches the shortest plausible "cycle" length
  referenced elsewhere in this project's own diagnostics, e.g. H0/R-series' multi-year-cycle
  framing — not an arbitrary pick), N = 2,000 draws, percentile method (report the 5th/95th
  percentile of the resampled net-of-cost IR; success = 5th percentile > 0). This replaces the
  ambiguous "excludes 0 in ≥90% of resamples" phrasing with one precise, standard definition.
- Risk-aversion sweep: if using `vol_target_overlay`, sweep `vol_target_ann ∈ {8%, 10%, 12%, 15%}`
  (a range spanning conservative to the anchor's natural unconstrained vol); report H3+H4's
  net-of-cost IR at each, not just the single silently-defaulted value.
- Cost perturbation (fixed-decision test, §1 above): replay at cost multipliers
  `{0.5×, 1×, 1.5×, 2×}` of the standing `c_sell=c_buy=0.03%`, weights/thresholds unchanged.
- Universe perturbation (fixed-decision test, §1 above): 20 random draws, each dropping the
  bottom 2 tickers by eligibility score from that month's active universe (a concrete, small,
  reproducible perturbation — not "a few," and small enough that it plausibly reflects real
  eligibility-cutoff noise rather than a stress test of a different kind); rerun the frozen
  decision on each draw, tabulate pass/fail against H3's/H4's original gate criteria.

## Expected Outcome

Either the H3+H4 verdict holds up across a reasonable neighborhood of assumptions — meaning it's
a real, robust finding — or it's fragile to small perturbations, meaning the earlier PASS was
closer to lucky than earned.

## Validation

Report bootstrap CI width and whether it excludes 0 across resamples; report the fraction of the
perturbation grid that still passes H3's and H4's original gate criteria unchanged.

## Success Criteria

The block-bootstrap's 5th-percentile net-of-cost IR (N=2,000, 12-month blocks) is > 0; the
fixed-decision strategy still passes H3's/H4's original gate criteria at cost multiplier ≤ 1.5×
(realistic — real slippage/spread beyond the flat 3bp fee is plausible, per H4's own Risks
section) and in ≥ 75% of the 20 universe-perturbation draws.

## Failure Criteria

5th-percentile bootstrap IR ≤ 0, or the pass/fail verdict flips at a cost multiplier ≤ 1.5× or in
a clear majority of universe-perturbation draws — indicates the earlier PASS was fragile, not
robust, and should be treated as inconclusive rather than a green light for H6.

## Risks & Assumptions

Block bootstrap on ~90 monthly points with real autocorrelation (multi-year cycles) may itself
have limited power to detect fragility — a "robust" verdict here is reassuring but not airtight,
given the same small-sample ceiling that has constrained every stage in this project (H0's power
floors apply here too, not just to the original characteristic screen).

## Next Decision Gate

**Robust** → proceed to H6 with a specific, justified risk-aversion setting (not the silent
default).
**Fragile** → treat H3+H4 as inconclusive, not validated; do not proceed to deployment planning
(H7) on this basis. **Concrete attribution procedure, not a vague heuristic:** rerun the same
fixed-decision perturbation grid on H3's anchor-alone (no H4 layer) and H3+H4 separately — if
H3-alone degrades similarly, the fragility is upstream (scoring/anchor/blend); if only H3+H4
degrades and H3-alone is stable, it's H4's thresholds specifically. Revisit only the implicated
stage narrowly, rather than restarting the whole program broadly.
