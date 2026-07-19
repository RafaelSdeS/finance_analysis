# R1 Findings — Risk Mandate, Val-Split Backtest (2026-07-19)

Executes `RISK_MANDATE_IMPL_PLAN.md` R1.2. Run: `experiments/risk_mandate_20260719T104432181562_2257576/`
(commit `9967c9b`, dataset commit `6a2807a`, config `configs/risk_mandate.json` defaults —
lookback=126d, rebalance=21d, vol_target=12%/yr, Ledoit-Wolf, `max_weight` uncapped).

**Window**: val split, 2021-11-30 → 2024-03-21 (~2.3 years, 28 rebalances, all 50
top-50 members eligible at every rebalance — no entrant exclusions in this window).
Note: this window overlaps the exact period M1 (`M4_DECISION_FINAL.md`) found the
RL agent's apparent "signal" was actually one dominant oil-shock trend, not skill —
worth keeping in mind when reading any strategy's outperformance here.

## Results

| policy | ann. return | Sharpe | Sortino | Calmar | max DD | vol | ann. turnover | cost drag | eff. N | cash | total-return 95% CI | Sharpe 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **min_variance_voltarget** | 16.5% | **0.339** | 0.597 | **1.192** | **13.9%** | 12.7% | 2.43 | 0.29% | 10.0 | 12.9% | [-0.07, 1.01] | [-1.13, 1.53] |
| min_variance | 15.9% | 0.272 | 0.472 | 0.928 | 17.1% | 14.8% | 2.65 | 0.33% | 10.1 | 0.0% | [-0.15, 1.07] | [-1.21, 1.41] |
| risk_parity_voltarget | 12.8% | 0.078 | 0.136 | 1.066 | 12.0% | 12.4% | 1.01 | 0.10% | 9.8 | 34.8% | [-0.10, 0.87] | [-1.34, 1.32] |
| risk_parity | 9.9% | -0.022 | -0.037 | 0.460 | 21.6% | 19.6% | 1.27 | 0.14% | 40.4 | 0.0% | [-0.34, 1.18] | [-1.45, 1.22] |
| ucrp | 5.6% | -0.165 | -0.275 | 0.214 | 26.0% | 22.9% | 2.58 | 0.32% | 51.0 | 2.0% | [-0.45, 1.21] | [-1.55, 1.15] |
| bova11 | 10.5% | 0.004 | 0.007 | 0.499 | 21.0% | 19.5% | 0 | 0 | 0 | 0.0% | [-0.29, 1.15] | [-1.28, 1.23] |
| ubah | 14.8% | 0.199 | 0.318 | 0.701 | 21.1% | 20.0% | 0.43 | 0.03% | 37.2 | 2.0% | [-0.26, 1.50] | [-1.17, 1.53] |
| random_portfolio | 16.2% | 0.277 | 0.427 | 1.154 | 14.1% | 16.8% | 0.41 | 0.03% | 23.5 | 5.6% | [-0.13, 1.31] | [-1.01, 1.59] |
| random_rebalancing | 0.9% | -0.352 | -0.566 | 0.031 | 28.0% | 23.3% | 125.85 | 15.54% | 26.2 | 2.1% | [-0.52, 1.03] | [-1.75, 0.94] |
| constant_cash | 12.5% | nan | nan | nan | 0.0% | 0.08% | 0 | 0 | 1.0 | 100% | [0.30, 0.32] | — |
| best_stock (hindsight) | 140.0% | 1.849 | 3.239 | 4.564 | 30.7% | 44.7% | 0.44 | 0.03% | 1.0 | 0.0% | [1.01, 26.1] | [0.64, 2.95] |

## Reading

1. **Point estimates favor `min_variance_voltarget` on every risk-adjusted metric**:
   best Sharpe (0.34), best Sortino (0.60), best Calmar (1.19), lowest max drawdown
   (13.9%) among all non-degenerate strategies, and it beats both UCRP (Sharpe −0.17)
   and BOVA11 (Sharpe 0.00) outright. `min_variance` (no overlay) is second-best on
   most metrics — the diversification/vol-drag story from `RISK_MANDATE_PLAN.md` §1.1–1.3
   is directionally supported.

2. **But CIs are wide and overlapping across every strategy** (Sharpe 95% CIs all span
   roughly [-1.1, +1.5], total-return CIs all cross zero except `min_variance_voltarget`'s,
   whose lower bound at −0.07 is still not separated from UCRP's or BOVA11's own CIs).
   28 rebalances / ~588 trading days is a short, high-variance window for a block
   bootstrap to resolve — **R4's "CI separation" bar is not met yet**. This result
   is suggestive, not confirmatory.

3. **Surprise, reported honestly per project discipline**: `risk_parity` underperforms
   `min_variance` here (Sharpe −0.02 vs 0.27, Calmar 0.46 vs 0.93), the opposite of
   the naive expectation that ERC's anti-concentration (effective N 40.4 vs 10.1)
   is a free hedge. In this window, spreading risk that evenly diluted into the
   period's laggards cost more than min-variance's concentration risk paid for. Not
   yet enough evidence to reject ERC — one window, no disjoint-window check yet (R3).

4. **Vol-targeting overlay helps both base policies** (`min_variance_voltarget` beats
   `min_variance` on Sharpe/Calmar/drawdown/turnover-cost; `risk_parity_voltarget`
   beats `risk_parity` on every metric) — consistent with §1.2's CDI-hurdle argument:
   sizing the equity sleeve to a fixed vol target rather than staying always-100%-invested
   pays off, even over a period where equities broadly outperformed cash (12.9%/34.8%
   average cash weight, well below CDI's own ~8.65%/yr making cash a real drag here,
   yet the overlay variants still won on risk-adjusted terms).

5. **`min_variance`'s `mean_cash_weight = 0.0%`** is expected and correct: the
   non-overlay policies are the "optimal risky sleeve" from §1.1, always fully invested
   in the risky sleeve by design — cash sizing is exclusively the `*_voltarget` variants' job.

6. Cost drag is low across the board for monthly-rebalanced policies (0.10%–0.33%
   over 2.3 years) — confirms §3.5's turnover-control design (warm-started QP, drift
   on non-rebalance days) is working as intended. `random_rebalancing`'s 15.5% cost
   drag is the useful negative control: daily reshuffling without a signal is
   ruinously expensive, as expected.

## Verdict against R4's decision gate

**Not yet a pass or fail** — promising point estimates, insufficient statistical
power on a single 2.3-year window. Per `RISK_MANDATE_IMPL_PLAN.md`'s own discipline
(R2 grid frozen before further runs, R3 disjoint-window check before any success claim):

- Do **not** conclude the risk mandate works from this run alone.
- R2 (lookback × rebalance-frequency grid) and R3 (test-split + cost-stress
  robustness) are the load-bearing next steps before a real verdict — same standard
  M1 applied when the RL agent's first-window "signal" didn't replicate.
- `risk_parity`'s underperformance here should specifically be re-checked on the
  test split before treating it as a real finding rather than one window's noise.

## Files

- `experiments/risk_mandate_20260719T104432181562_2257576/` — this run's full artifacts
  (`report.html`, `metrics_summary.json`, `run_manifest.json`, `config.json`)
- `experiments/risk_mandate_20260719T104421511002_2256954/` — the preceding `--dry-run`
  (eligibility report only)
