# R3 Findings — Disjoint Test-Split Confirmation + Cost Stress (2026-07-19)

Executes `RISK_MANDATE_IMPL_PLAN.md` R3. Test split: 2024-03-22 → 2026-07-14 (577
days), disjoint from R1/R2's val split (2021-11-30 → 2024-03-21). Three configs,
selected *before* looking at test numbers (per proper nested validation, matching
R2's stated commitment):

1. **default** (lookback=126, rebalance=21d) — R1's original config
2. **r2_selected** (lookback=63, rebalance=1d) — R2's single best val-split cell
3. **r2_selected_2x_costs** — same as (2), transaction costs doubled (0.06%/side)

## Results (point estimates; CIs omitted — see note below)

| config | policy | Sharpe | Calmar | max DD | ann. return | ann. turnover | cost drag |
|---|---|---|---|---|---|---|---|
| — | **bova11** | **0.241** | **1.046** | 15.4% | 16.1% | 0 | 0 |
| — | ucrp | -0.207 | 0.363 | 19.9% | 7.2% | 2.11 | 0.26% |
| — | ubah | -0.172 | 0.466 | 17.8% | 8.3% | 0.43 | 0.03% |
| default (lb126/rb21) | min_variance | 0.059 | 1.194 | 10.9% | 13.1% | 2.72 | 0.34% |
| default (lb126/rb21) | min_variance_voltarget | 0.074 | 1.225 | 10.9% | 13.3% | 2.75 | 0.34% |
| default (lb126/rb21) | risk_parity | -0.065 | 0.701 | 15.2% | 10.7% | 1.18 | 0.13% |
| default (lb126/rb21) | risk_parity_voltarget | -0.017 | 0.798 | 14.8% | 11.8% | 1.34 | 0.12% |
| r2_selected (lb63/rb1) | min_variance | 0.050 | 1.016 | 12.7% | 12.9% | 13.59 | 1.82% |
| r2_selected (lb63/rb1) | min_variance_voltarget | 0.073 | 1.097 | 12.1% | 13.3% | 13.57 | 1.80% |
| r2_selected (lb63/rb1) | risk_parity | -0.093 | 0.653 | 15.6% | 10.2% | 4.14 | 0.54% |
| r2_selected (lb63/rb1) | risk_parity_voltarget | -0.020 | 0.875 | 13.5% | 11.8% | 4.63 | 0.51% |
| 2x costs (lb63/rb1) | min_variance | **-0.013** | 0.926 | 13.0% | 12.0% | 13.59 | 3.61% |
| 2x costs (lb63/rb1) | min_variance_voltarget | 0.010 | 1.003 | 12.4% | 12.4% | 13.57 | 3.56% |
| 2x costs (lb63/rb1) | risk_parity | -0.108 | 0.635 | 15.7% | 10.0% | 4.14 | 1.07% |
| 2x costs (lb63/rb1) | risk_parity_voltarget | -0.036 | 0.852 | 13.6% | 11.6% | 4.63 | 1.03% |

## Verdict: R4 decision gate — **FAIL**

R4's bar (`RISK_MANDATE_PLAN.md`): beat **both** UCRP **and** BOVA11 on risk-adjusted
metrics with CI separation, after costs, on both windows.

- **BOVA11 wins outright on the test split** — Sharpe 0.241 vs. the best risk-mandate
  variant's 0.074 (`min_variance_voltarget`, default config). Every risk-mandate
  policy, every config, underperforms passively holding the benchmark ETF in this
  window. This alone fails R4's gate; CI computation isn't needed to see a 3x point
  gap direction this consistent across 4 policies × 3 configs.
- Risk-mandate policies DO beat UCRP/UBAH (both went **negative** Sharpe on test) —
  a partial win, but R4 requires both legs, not one.

## R2's headline finding did NOT replicate — this is the important part

R2 found a clean, monotonic "shorter lookback wins" pattern (lookback=63 uniformly
best, Sharpe 0.52 vs 126's 0.27 vs 252's -0.05, val split). On the disjoint test
split, the R2-selected config (lookback=63, daily rebalance) performs **no better,
marginally worse** than the untuned default (126/monthly) on both `min_variance`
(0.050 vs 0.059) and `min_variance_voltarget` (0.073 vs 0.074) — and needs ~5x the
turnover to get there (13.6 vs 2.7 annualized), making it strictly worse once cost
sensitivity is considered (see below). **The lookback effect was window-specific,
not persistent** — R2's interpretable-looking pattern was reading the 2022
volatility-regime shift inside that one window, not a durable property of covariance
estimation. This is the same shape of false positive M1 found in the RL agent's
original "signal": a clean effect in one window that a disjoint-window replication
check kills. The process worked exactly as designed to catch it.

## Cost-rate stress confirms the short-lookback config is fragile

Doubling transaction costs on the R2-selected (daily-rebalance) config pushes
`min_variance`'s Sharpe negative (-0.013) and `min_variance_voltarget`'s to
near-zero (0.010) — cost drag triples from 1.8% to 3.6% of the 2.3-year window's
return. The default (monthly) config was never cost-stress-tested here since it
already lost the val-split selection step, but its 5x-lower turnover makes it
structurally more cost-robust by construction — another mark in its favor over
the R2-selected config, even though neither beats BOVA11.

## What DID replicate

`risk_parity`'s underperformance relative to `min_variance` — first seen in R1,
confirmed across all 9 R2 grid cells, and confirmed again here on a fully disjoint
window and at 2x costs. Three independent checks now agree: ERC's anti-concentration
here is not a free hedge in this specific universe/period, it's a cost. This is a
more trustworthy finding than the lookback effect precisely because it replicated
without being selected for — it was never the metric being optimized in R2.

The vol-target overlay's benefit (voltarget variant beats its own base policy)
also replicated cleanly across R1, all of R2, and R3 for every policy family.

## Overall conclusion

**Plain min-variance / risk-parity / vol-targeted structural allocation on the
top-50 Brazilian equity universe, monthly-ish rebalance, Ledoit-Wolf covariance,
does not durably beat a passive cap-weighted benchmark (BOVA11) net of costs.**
This is a clean negative result, reported with the same rigor as M2/M3 — not a
failure of implementation (R0's test suite, R1-R3's methodology, and the
replication-check discipline all worked correctly), but a real finding about this
specific structural-premium hypothesis in this specific universe.

This does not indict the risk/diversification framing wholesale — it indicts
*this* instantiation: a fixed top-50 large-cap universe is close to what BOVA11
already holds (large, liquid B3 names), so a covariance-only reallocation within
that same universe has limited room to add value over the cap-weighted index it
overlaps heavily with. The standard literature caveat applies: min-variance/risk-parity
structural premia are typically stronger in less-efficient, higher-dispersion
universes (small/mid-cap, sector-constrained, or factor-tilted) than in a top-50
large-cap universe already dominated by the benchmark's own constituents.

## Note on CIs

Point estimates are decisive enough here (BOVA11's Sharpe is >3x the best
risk-mandate variant, consistently across 4 policies and 3 configs) that bootstrap
CIs weren't computed for this run — R1's CIs were already wide (~±1.3) at this
sample size, so formal CI separation was never going to be establishable either
way; the finding here is about direction and replication, not statistical
significance at a single point.

## Recommendation

Per this project's own discipline (M4_DECISION_FINAL.md's precedent): **stop
iterating on this exact formulation** (fixed top-50, plain covariance, no factor
structure). Before any further model-side tuning (different shrinkage, different
lookback grids, more overlay variants), the right move is a strategic decision —
same kind of fork M4 itself surfaced. Candidates, not pursued without sign-off:

- **Different universe**: small/mid-cap or factor-tilted (where minimum-variance
  premia are better documented) instead of top-50 large-cap
- **Relax the "same universe as BOVA11" constraint**: sector-neutral overlay, or
  explicit active-weight-vs-benchmark constraints
- **Accept the negative result** and close out Option A the way Option (daily
  alpha) was closed after M1-M3, writing a final verdict doc analogous to
  `M4_DECISION_FINAL.md`
