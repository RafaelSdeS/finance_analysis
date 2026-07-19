# Risk/Diversification Mandate — Research Plan (Option A pivot, 2026-07-19)

Successor to the M-series (see `M4_DECISION_FINAL.md`): M1–M3 rejected the daily
cross-sectional alpha hypothesis on the top-50 universe. This plan drops forecasting
entirely and pursues structural portfolio optimization — Kelly/Merton with no return
view, degenerating to minimum-variance / risk-parity / volatility-targeted portfolios.
The null alpha result is an *input* here, not an obstacle.

## Objective

Maximize long-run risk-adjusted compounding (geometric growth) on the existing top-50
dynamic universe with CDI-accruing cash, using only estimated risk structure (covariance),
never predicted returns. Success = better Sharpe-vs-CDI / Calmar / max-drawdown than the
naive structural baselines (UCRP, UBAH, BOVA11, constant-cash mixes) after costs, with
bootstrap CIs — not raw return.

## Conceptual framing

- Merton: `w* = (1/γ) Σ⁻¹(μ − r)`. With no alpha view, set equal expected excess returns
  → weights driven purely by Σ. Variants along that degeneracy:
  1. **Minimum variance** (long-only, simplex-constrained)
  2. **Risk parity / equal risk contribution**
  3. **Volatility targeting** (scale equity sleeve vs CDI cash to hold constant ex-ante vol)
- Edges being harvested: volatility-drag reduction, diversification, rebalancing premium.
  All exist without any predictive claim.

## Architecture (reuses Stage 3 harness)

No new training loop, no PVM, no networks, no RL. Architecturally these are new
entries beside `src/rl_agent/baselines.py`:

- **New module `src/rl_agent/risk_portfolios.py`**:
  - trailing covariance estimator on `PricePanel` return windows — `sklearn.covariance.LedoitWolf`
    (mandatory: 50 assets × ~250-day windows, sample covariance near-singular)
  - `min_variance_weights(Σ)` — long-only simplex QP via `scipy.optimize.minimize`
  - `risk_parity_weights(Σ)` — standard fixed-point/Newton iteration
  - `vol_target_weights(w_equity, σ_target)` — cash/equity blend using CDI cash slot 0
  - each exposed as a weight-policy callable fed through the existing `run_backtest()`
- **Config**: `configs/risk_mandate.json` — lookback window, rebalance frequency,
  vol target, γ; all through the existing `ExperimentConfig` pattern (extend minimally,
  nothing hardcoded downstream)
- **Reused unchanged**: `PricePanel`/`GlobalAssetIndex`, `environment.run_backtest`
  (cost model, μ fixed-point, drift), `metrics.py` (+ bootstrap CIs), `plots.write_report`,
  split handling. Universe membership churn is already handled by the global-space
  mask/gather machinery.

## Data requirements

None new. Prices, CDI, and the top-50 membership file already on disk cover everything.
(USD/BRL, term structure etc. are Option-B concerns — out of scope.)

## Milestones

R0/R1 detailed implementation spec: `RISK_MANDATE_IMPL_PLAN.md`.

- [ ] **R0 — Skeleton + sanity**: `risk_portfolios.py` with min-variance only; synthetic-data
      test (known 2-asset Σ → analytic weights); wire into `run_backtest` on the train split.
- [ ] **R1 — Baseline comparison**: min-variance + risk-parity + vol-target vs the existing
      7 baselines on the existing split; full report + bootstrap CIs.
- [ ] **R2 — Sensitivity**: lookback ∈ {63, 126, 252}d × rebalance ∈ {daily, weekly, monthly}.
      Expect turnover cost to dominate daily rebalancing; monthly likely wins after costs.
- [ ] **R3 — Robustness**: disjoint time window (same discipline as M1's replication check);
      cost-rate stress (2× transaction cost).
- [ ] **R4 — Decision gate**: does any variant beat UCRP *and* BOVA11 on risk-adjusted
      metrics with CI separation, after costs, on both windows? If yes → candidate for the
      long-term investment agent core; if no → the structural-premium hypothesis is also
      rejected for this universe, document and reassess.

## Validation discipline (carried over from M-series)

- Point-in-time universe membership only; no `status` feature; no lookahead.
- Bootstrap CIs on every comparison; no single-window, single-seed claims.
- Negative results get written up (`R*_FINDINGS.md`) with the same rigor as M2/M3.
- No hyperparameter fishing: the R2 grid is fixed up front; anything outside it needs a
  written rationale before running.

## Known risks / ceilings

- Min-variance in a 50-name equity universe concentrates in low-vol names (utilities/banks
  in B3) — risk parity is the hedge against that degenerate corner.
- Covariance nonstationarity across Brazilian rate cycles; Ledoit-Wolf shrinkage + R3's
  disjoint-window check are the guards.
- CDI is a high hurdle (~8.65%/yr log-space): a pure-equity min-variance sleeve can be
  structurally beaten by cash, same attractor economics as the RL agent faced. Vol-targeting
  (explicit cash blend) is the variant that engages with this honestly rather than fighting it.
