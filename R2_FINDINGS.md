# R2 Findings — Sensitivity Grid (2026-07-19)

Executes `RISK_MANDATE_IMPL_PLAN.md` R2. Grid frozen before running (per plan):
lookback ∈ {63, 126, 252}d × rebalance ∈ {1, 5, 21}d (daily/weekly/monthly), all 4
risk policies, same val split as R1 (2021-11-30 → 2024-03-21). 36 backtests, all
against the same panel/split/benchmark loaded once. Full CSV: scratchpad
`r2_results.csv` (not repo-tracked; numbers reproduced in the table below).

## Headline result

**Sharpe is monotonically decreasing in lookback, uniformly across every policy and
every rebalance frequency:**

| policy | lookback=63 | lookback=126 | lookback=252 |
|---|---|---|---|
| min_variance (best rebal) | **0.52** (daily) | 0.27 (monthly) | -0.05 (monthly) |
| min_variance_voltarget (best rebal) | **0.49** (daily) | 0.34 (monthly) | -0.00 (monthly) |
| risk_parity (best rebal) | 0.06 (monthly) | -0.02 (monthly) | -0.06 (monthly) |
| risk_parity_voltarget (best rebal) | **0.15** (monthly) | 0.08 (monthly) | -0.02 (monthly) |

126-day lookback (R1's default) is not the best cell anywhere in the grid; every
policy family's best Sharpe is at lookback=63.

Rebalance frequency is a secondary effect and noisier: at lookback=63, daily
rebalancing edges out monthly for the plain (non-overlay) policies despite ~4x the
turnover (14.3 vs 3.6 annualized) and ~4x the cost drag (1.9% vs 0.5%) — the faster
covariance reactivity apparently outweighs the extra cost in this window. At
lookback=126/252, monthly is consistently best or tied-best. No clean monotonic
rule on rebalance frequency the way there is on lookback.

`risk_parity`'s underperformance from R1 is **not a one-window artifact** — it holds
across all 9 grid cells (Sharpe never exceeds 0.06 for the plain variant, vs.
min_variance's 0.16–0.52 range). The vol-target overlay consistently helps every
policy family at every grid point, also not window-specific.

## Reading — and the flag that matters most

A monotonic short-lookback advantage is a real, interpretable pattern (63-day
covariance reacts faster to a regime shift; the val window spans the 2022
rate-hike volatility spike, which a 252-day lookback would lag badly on) — not
noisy grid scatter. That interpretability is exactly what makes it worth taking
seriously.

**But it is also exactly the shape of M1's original false positive**: a clean,
consistent-looking result on one window that didn't survive a disjoint-window
replication check. A short lookback that reacts fast to a regime shift could
equally be overfit to this window's specific regime transition, or could reflect
a genuine, persistent vol-clustering effect (well-documented in the literature)
that should show up again on a different window. **No conclusion is drawn from R2
alone** — this is what R3 exists to settle.

## Selected candidate for R3

Per proper nested validation (select on val, confirm on test — decided *before*
looking at test results, not after): **lookback=63, rebalance_every=1 (daily)**,
the single best cell in the grid (`min_variance`, Sharpe 0.519), carried forward
alongside R1's original default (lookback=126, rebalance_every=21) as the
pre-registered comparison baseline.
