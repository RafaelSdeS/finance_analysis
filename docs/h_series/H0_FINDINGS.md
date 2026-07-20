# H0 Findings — Walk-Forward Spine, Baselines, Power Analysis

Stitched OOS: 91 monthly observations (2018-12-31 onward through the dataset's last date).

## Power floors (frozen BEFORE any H1 characteristic is examined)

- Min detectable mean IC (t=2, k=21): 0.0300 (NW lag 0 months)
- Min detectable mean IC (t=2, k=63): 0.0519 (NW lag 2 months)
- Min detectable annualized IR (t=2): 0.726

## Baseline summary (monthly, net of 3bps/side costs; block-bootstrap CI, block=4 months)

### ucrp
- Total return (stitched OOS): 0.678 [-0.545, 4.296]
- Monthly active return vs BOVA11: mean=-0.00180, NW-t=-0.90

### bova11
- Total return (stitched OOS): 1.069 [-0.412, 4.786]
- Monthly active return vs BOVA11: mean=0.00000, NW-t=0.00

### min_variance
- Total return (stitched OOS): 0.925 [-0.299, 3.426]
- Monthly active return vs BOVA11: mean=-0.00172, NW-t=-0.49

### classical_mv
- Total return (stitched OOS): 9.134 [1.136, 44.814]
- Monthly active return vs BOVA11: mean=0.02056, NW-t=2.32

## Interpretation: classical_mv's point estimate is not a robust bar

classical_mv shows the strongest point estimate here (total return 9.13, active-return NW-t=2.32) but also by far the widest bootstrap CI (44.8 / 1.1 = 39.4x spread). This is the textbook instability of naive sample-mean Markowitz optimization (Michaud 1989's "optimization enigma"): with no return VIEW, only a noisy trailing-mean ESTIMATE, and no per-name weight cap, the optimizer concentrates hard in whatever name had the best noisy trailing mu -- a few lucky/unlucky realizations dominate the whole stitched path. It is exactly why risk_portfolios.py's own policies (min_variance, risk_parity) deliberately carry no mu estimate at all (RISK_MANDATE_PLAN.md). **H2's bar is NOT "beat classical_mv's point estimate" -- a lucky concentrated bet can do that by construction. The bar is beating it with a MATERIALLY NARROWER bootstrap CI, i.e. genuine breadth-of-evidence outperformance, not concentrated luck.**
