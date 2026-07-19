# Risk Mandate — R0/R1 Implementation Plan (2026-07-19)

Executes milestones R0–R1 of `RISK_MANDATE_PLAN.md`. Scope: covariance estimation,
min-variance, equal-risk-contribution (ERC), and volatility-targeting policies, wired
through the existing `run_backtest` harness and compared against the existing 7 baselines.
No new dependencies, no new data, no training loop.

---

## 1. WHY — mathematical rationale (architectural alignment)

### 1.1 Alpha degeneracy: Merton without μ

Merton: `w* = (1/γ) Σ⁻¹(μ − r·1)`. M1–M3's verdict is not "μ is unknown" — it is that the
**cross-sectional dispersion of μ is unmeasurable** with our features: the best admissible
estimate is `μᵢ − r = c` (one unknown constant, identical across assets). Substituting:

```
w* = (c/γ) Σ⁻¹1
```

Direction and scale separate cleanly:
- **Direction** `Σ⁻¹1 / (1ᵀΣ⁻¹1)` is exactly the **minimum-variance portfolio**. Dropping μ
  doesn't break Merton; homogeneous μ makes min-variance the *optimal* risky sleeve.
- **Scale** `(c/γ)` — how much of the sleeve to hold vs cash — is one scalar we cannot
  estimate from c but CAN control via an explicit volatility target (§1.2).
- **ERC** is the robust sibling: `Σ⁻¹` amplifies estimation error in Σ's smallest
  eigenvalues (the classic error-maximization corner). Equalizing risk contributions
  `RCᵢ = wᵢ(Σw)ᵢ = σ_p²/N` uses only Σ's better-estimated structure; it interpolates
  between min-variance and equal-weight and is the hedge against min-variance
  concentrating in 3 low-vol utilities/banks.

### 1.2 The CDI hurdle: why vol targeting is mandatory, not optional

Measured on our panel (log-space, 2011–2026): CDI ≈ **8.65%/yr**, equal-weight equity
≈ **8.22%/yr**. Long-run (geometric) growth of a portfolio is `g ≈ μ_arith − σ²/2`.
With the equity sleeve barely matching cash arithmetically, **any uncompensated variance
makes cash dominate compounding** — this is precisely the cash attractor that swallowed
the RL agent (`EIIE_DIAGNOSIS_PLAN.md`), rediscovered as arithmetic rather than as a
softmax pathology.

Kelly/Merton sizing of the cash/equity split: `f* = (μ_p − r)/(γ σ_p²)`. We can't
estimate `μ_p − r`, but holding **ex-ante portfolio vol constant** (`f = σ_target/σ̂_p`,
capped at 1, remainder to CDI slot 0) is fractional Kelly under an assumed-positive,
unknown premium — and variance, unlike returns, IS forecastable (vol clustering). The
overlay makes cash an explicitly sized decision instead of a degenerate all-or-nothing
attractor. Estimation error argues for fractional (half-)Kelly ⇒ conservative
`vol_target_ann` defaults.

### 1.3 The structural premium being harvested (no forecast required)

1. **Volatility-drag reduction / diversification return**: `g_portfolio − Σwᵢgᵢ ≈
   ½(Σwᵢσᵢ² − σ_p²) > 0` whenever correlations < 1. Free geometric return from
   variance reduction alone.
2. **Rebalancing premium**: any constant-mix policy systematically sells relative
   winners / buys relative losers; positive expectation under mean-reverting *relative*
   prices, bounded below by −(turnover × cost). Measurable in our harness: `run_backtest`
   already reports turnover and cost drag per policy.
3. **Sharpe improvement from vol predictability**: targeting constant ex-ante vol
   raises realized Sharpe if σ is persistent (it is) even when μ is unpredictable (it
   is, per M3). A variance forecast is not an alpha claim.

Success criterion (unchanged from `RISK_MANDATE_PLAN.md` R4): beat UCRP **and** BOVA11
on Sharpe-vs-CDI / Calmar / max-drawdown after costs, with bootstrap-CI separation.

---

## 2. WHAT — modules, config, integration

### 2.1 New file: `src/rl_agent/risk_portfolios.py`

Four components, all pure numpy/sklearn/scipy, no torch:

| Function | Contract |
|---|---|
| `trailing_returns(panel, t, lookback, gidx)` | `[L, n_active]` simple-return slice from `panel.price_relative_batch(np.arange(t-lookback+1, t+1)) − 1`, columns `gidx`. Same information set as day-t drift (relatives through t's close), so **zero lookahead by construction**. |
| `estimate_cov(returns, cfg)` | `Σ [n_active, n_active]`, annualized (×252). `cov_estimator="ledoit_wolf"` (default) or `"ewma"` (§3.1). Always PD after jitter. |
| `min_variance_weights(cov, max_weight, warm_start)` | Long-only simplex QP (§3.2). Returns `(w, converged: bool)`. |
| `risk_parity_weights(cov)` | ERC via cyclical coordinate descent (§3.3). Returns `(w, converged: bool)`. |
| `vol_target_overlay(w_eq, cov, vol_target_ann)` | `f = clip(vol_target_ann / sqrt(w_eqᵀΣw_eq), 0, 1)`; equity sleeve ×f, `1−f` to `CASH_GIDX`. |
| `make_risk_weight_fn(policy, cfg, panel)` | Closure factory following the `make_*_weight_fn` pattern in `baselines.py`; wires the above into a `weight_fn(t, w_prev, w_drift, panel)`. |

Policy names (registered, §2.3): `min_variance`, `risk_parity`, and each
`*_voltarget` variant.

**Inside `make_risk_weight_fn` — the per-day logic:**

```
if (t − start_idx) % cfg.risk.rebalance_every != 0:
    return w_drift                      # non-rebalance day: zero turnover, zero cost (UBAH pattern)
active = panel.slot_gidx[t][panel.valid[t]]          # same helper pattern as baselines._active_gidx
eligible = active gidx with ≥ min_history_frac·lookback finite returns in the slice
    # entrants with short history are excluded this rebalance; they enter once seasoned.
    # departures need no code: weight_fn simply never writes a non-active gidx (global-space scatter).
R = trailing_returns(panel, t, lookback, eligible)
Σ = estimate_cov(R, cfg)
w_eq, ok = optimizer(Σ)                 # min_variance_weights or risk_parity_weights
if not ok: fallback chain (§3.4)
w_global = zeros(n_global); w_global[eligible] = w_eq
if voltarget: w_global = vol_target_overlay(...)
cache w_global until next rebalance     # also the QP warm start
return w_global
```

### 2.2 Config: `RiskConfig` dataclass + `configs/risk_mandate.json`

Add to `src/rl_agent/config.py` (same frozen-dataclass pattern; `from_dict` must
default a missing `"risk"` section so every existing EIIE config parses unchanged):

```python
@dataclass(frozen=True)
class RiskConfig:
    policies: tuple = ("min_variance", "risk_parity",
                       "min_variance_voltarget", "risk_parity_voltarget")
    lookback: int = 126            # trading days; R2 grid {63, 126, 252}
    min_history_frac: float = 0.8  # eligibility: finite-return coverage within lookback
    rebalance_every: int = 21      # trading days; R2 grid {1, 5, 21}
    cov_estimator: str = "ledoit_wolf"   # or "ewma"
    ewma_halflife: int = 63        # trading days (λ = exp(−ln2/halflife))
    vol_target_ann: float = 0.12   # ex-ante annualized σ for *_voltarget policies
    max_weight: float = 1.0        # per-name cap; 1.0 = off (R2 may probe 0.10)
    solver_tol: float = 1e-9
    warm_start: bool = True        # reuse previous rebalance's solution as QP x0
```

`ExperimentConfig` gains `risk: RiskConfig = field(default_factory=RiskConfig)`.

`configs/risk_mandate.json`: `data`/`costs` copied from `eiie_baseline.json` verbatim
(same window, same 3 bp costs), `risk` as above, `eval.baselines` = all 7 existing names.
No `model`/`train` sections needed (defaults ignored — nothing trains).

### 2.3 Integration points (verified against current code)

- **The one seam**: `environment.run_backtest(panel, weight_fn, c_sell, c_buy, start_idx,
  end_idx)` with `weight_fn(t, w_prev, w_drift, panel) → w_target` (global-space
  `[n_global]`, sums to 1, cash at `CASH_GIDX`). Risk policies are ordinary weight_fns —
  identical cost model (`solve_mu`), drift (eq. 7), and bookkeeping as the agent and all
  baselines. **Zero changes to `environment.py`.**
- **Registration**: new tuple `RISK_POLICY_NAMES` in `risk_portfolios.py`; extend
  `baselines.run_baseline`'s dispatch to route those names (import from
  `risk_portfolios`). `BASELINE_NAMES` stays untouched → every existing test and config
  keeps passing.
- **Runner**: new thin CLI `src/rl_agent/risk_experiment.py`
  (`python -m src.rl_agent.risk_experiment --config configs/risk_mandate.json
  [--eval-split val|test] [--dry-run]`) — a clone of `experiment.py`'s post-training
  half: seed → `load_price_panel` → `compute_window_split` → run every `risk.policies`
  entry + every `eval.baselines` entry through `run_baseline` → `summarize` each vs CDI
  and BOVA11 → `write_report` (focal strategy slot = first policy in `risk.policies`;
  the report machinery needs no changes) → `metrics_summary.json` + `run_manifest.json`.
  No pretrain, no PVM, no sanity-gate dependency on torch.
- **Metrics additions** (small, `metrics.py`): ex-ante **diversification ratio**
  `DR = (wᵀσ)/√(wᵀΣw)` and **effective N** `1/Σwᵢ²` logged per rebalance into the
  summary — these are the mandate's own KPIs (M2 already used effective_n).

---

## 3. HOW — algorithm specifications

### 3.1 Covariance estimation

**Primary — Ledoit-Wolf** (`sklearn.covariance.LedoitWolf`):
- Input: demeaned daily simple returns `X [L, N]`, L ∈ {63,126,252}, N ≤ 50. At
  L=63, N=50 the sample covariance is near-singular; LW's shrinkage toward
  `(tr(S)/N)·I` with analytically optimal intensity guarantees well-conditioned PD
  output with no tuning. Use `assume_centered=False` (let it demean), annualize ×252.
- **Jitter regardless of estimator**: `Σ += ε·(tr(Σ)/N)·I`, `ε = 1e-8` — scale-relative,
  so it survives the unit change if lookback/annualization conventions ever move.

**Alternative — EWMA** (nonstationarity across BR rate cycles; `cov_estimator="ewma"`):
- RiskMetrics recursion equivalent, implemented vectorized: weights
  `π_k ∝ λ^k`, `λ = exp(−ln 2 / halflife)`, k = age in days, normalized to sum 1 over
  the lookback window; `Σ = (√π ⊙ X̃)ᵀ(√π ⊙ X̃)` on demeaned rows.
- EWMA's effective sample size (≈ `(1+λ)/(1−λ)` ≈ 2·halflife/ln2 days) is *smaller* than
  the window, so it needs shrinkage MORE, not less: feed the √π-weighted rows into
  LedoitWolf rather than using the raw weighted product — one code path handles both
  estimators' conditioning.

### 3.2 Minimum variance — long-only simplex QP

```
min_w  wᵀΣw     s.t.  Σwᵢ = 1,  0 ≤ wᵢ ≤ max_weight
```

- `scipy.optimize.minimize(method="SLSQP", jac=lambda w: 2Σw)` with the analytic
  Jacobian (SLSQP without jac finite-differences 50 dims — slow and noisy).
- **Conditioning**: solve on rescaled `Σ̂ = Σ/mean(diag(Σ))` (daily annualized variances
  ~1e-2–1e-1; rescaling puts the objective near unit scale). Argmin is invariant to
  positive scaling of Σ.
- **Warm start**: `x0` = previous rebalance's solution (restricted/renormalized to
  today's eligible set; new entrants start at 0⁺). Fewer iterations AND less
  solution flip-flop between near-degenerate optima ⇒ lower turnover.
  First rebalance: `x0` = inverse-variance weights (already near the answer).
- **Post-solve hygiene**: `w = clip(w, 0, None); w /= w.sum()` — SLSQP returns
  −1e-12-grade violations; never let them reach `run_backtest`'s simplex expectation.
- **Convergence failure** (`res.success is False` or non-finite w): fallback chain §3.4.

### 3.3 Equal risk contribution — cyclical coordinate descent

Spinu's convex formulation: minimize `½xᵀΣx − c·Σᵢ ln xᵢ` over `x > 0` (any c > 0),
then `w = x/Σxᵢ`. Strictly convex on the positive orthant ⇒ unique solution; ERC is
interior (all wᵢ > 0) so no bound handling needed. Per-coordinate closed-form update
(Griveau-Billion et al.):

```
bᵢ = (Σx)ᵢ − Σᵢᵢxᵢ                    # off-diagonal contribution
xᵢ ← (−bᵢ + √(bᵢ² + 4Σᵢᵢc)) / (2Σᵢᵢ)   # positive root of the per-coordinate quadratic
```

Sweep cyclically; converged when `max_i |RCᵢ/σ_p² − 1/N| < 1e-8`. Floor `xᵢ ≥ 1e-12`
against log-domain underflow. Init: inverse-vol. Cap `max_iter = 500` sweeps (typical
convergence: < 50). On non-convergence: fall back to **inverse-vol weights** (naive
risk parity — the ρ-blind ERC) and log it; §3.4 chain applies beyond that.
Note: `max_weight` does not apply to ERC in R1 (ERC is anti-concentrated by
construction; a cap would break the ERC property silently).

### 3.4 Solver failure handling (both optimizers)

Ordered fallback chain, each step logged with `t`, policy, and reason (the logger
`experiment.py` already configures):

1. Retry once with 10× jitter (`ε = 1e-7`).
2. Fall back to the estimator-appropriate analytic proxy: inverse-variance
   (min-var) / inverse-vol (ERC) on `diag(Σ)` — always finite, always simplex.
3. Hold previous rebalance's weights (i.e., return `w_drift`).

A backtest never dies on a solver hiccup, and the manifest records the count of
fallbacks per policy (a run with >1% fallback rebalances is flagged in the checklist).

### 3.5 Transaction-cost posture

Costs are already priced correctly by `solve_mu` (3 bp/side) — nothing to build. The
plan controls turnover at the source: monthly default rebalance, warm-started QP,
and drift (not re-solve) on non-rebalance days. Expected orderings to verify in R1:
daily min-var strictly dominated by monthly after costs; ERC turnover < min-var
turnover (smoother weights). A no-trade band (skip rebalance if ‖w_drift − w_new‖₁ <
threshold) is deferred to R2 — `# ponytail:` comment marks the hook point.

### 3.6 R0 sanity suite — `tests/rl_agent/test_risk_portfolios.py`

Synthetic, deterministic, CPU-only, fast group (joins `tests/run_all.py --group fast`).
All closed forms below are exact — assert to 1e-6 unless noted.

| Check | Setup | Analytic truth |
|---|---|---|
| Min-var 2-asset | σ₁=0.10, σ₂=0.20, ρ=0.3, Σ exact (bypass estimator) | `w₁ = (σ₂²−ρσ₁σ₂)/(σ₁²+σ₂²−2ρσ₁σ₂) = 0.034/0.038 = 0.89473…` |
| ERC 2-asset | same Σ | `wᵢ ∝ 1/σᵢ` exactly, ∀ρ: `w₁ = 2/3` |
| Min-var N-asset diagonal | σ = (0.1, 0.2, 0.4), ρ=0 | `wᵢ ∝ 1/σᵢ²` → (16/21, 4/21, 1/21) |
| ERC N-asset diagonal | same | `wᵢ ∝ 1/σᵢ` → (4/7, 2/7, 1/7) |
| ERC property | random PD Σ (fixed seed, N=50) | all `RCᵢ` equal to 1e-8 rel.; all wᵢ > 0 |
| Vol overlay | Σ exact, `vol_target = ½·σ_p` | `f = 0.5`, cash slot = 0.5 exactly; `vol_target ≥ σ_p` ⇒ f = 1, zero cash |
| Degenerate Σ | duplicate column (ρ=1 pair), N=5 | jittered solve finite, simplex, no exception |
| Estimator recovery | 2,000 iid samples from known diagonal Σ (seeded) | LW min-var weights within 0.05 of analytic (loose — sampling error) |
| Max-weight cap | 2-asset min-var, `max_weight=0.6` | `w = (0.6, 0.4)` (cap binds, remainder to other asset) |
| Harness smoke | synthetic `PricePanel` (reuse `tests/rl_agent/test_baselines.py` fabricated-market pattern) | `run_backtest` completes; every day's weights simplex to 1e-9; turnover == 0 on all non-rebalance days; PV finite; entrant with < min-history correctly held at 0 until seasoned |
| Determinism | same synthetic panel, two full runs | bit-identical `BacktestResult` (no RNG anywhere in these policies) |

R0 exits when this suite passes and `ruff` is clean. **No real-data run in R0.**

### 3.7 R1 execution (after R0 green — runs require explicit go-ahead)

1. `python -m src.rl_agent.risk_experiment --config configs/risk_mandate.json --dry-run`
   (panel load + split + eligibility stats only).
2. Full run on default config (val split): 4 policies + 7 baselines, one report.
3. Read out: Sharpe-vs-CDI, Calmar, maxDD, turnover, cost drag, DR, effective-N, with
   block-bootstrap CIs vs UCRP and BOVA11 — the existing `metrics.py`/`plots.py` output.
4. Findings → `R1_FINDINGS.md` (same rigor as M2/M3; negative result gets written up).

---

## Deliverables checklist

- [x] **R0.1** `src/rl_agent/risk_portfolios.py` (estimator, both optimizers, overlay, weight_fn factory, fallback chain)
- [x] **R0.2** `RiskConfig` in `config.py` (+ tolerant `from_dict`); `configs/risk_mandate.json`
- [x] **R0.3** `run_baseline` dispatch extension (`RISK_POLICY_NAMES`), `BASELINE_NAMES` untouched
- [x] **R0.4** `tests/rl_agent/test_risk_portfolios.py` — full §3.6 table, wired into `run_all.py --group fast` — **written, not yet run** (project working rule: code isn't executed until explicitly requested)
- [x] **R0.5** `src/rl_agent/risk_experiment.py` thin runner (+ `--dry-run`)
- [ ] **R0.6** DR + effective-N additions in `metrics.py` — **descoped**: `effective_n_holdings` already existed in `metrics.py` and covers the concentration KPI; a true ex-ante diversification ratio needs per-rebalance Sigma plumbed out of the weight_fn closure, which isn't exposed by `BacktestResult` today — deferred to R1 alongside deciding how to surface it (`# ponytail:` marks the hook in `risk_portfolios.py`'s `make_risk_weight_fn`)
- [ ] **R0.7 (new)** run `python tests/rl_agent/test_risk_portfolios.py` (or `run_all.py --group fast`) and fix anything that fails — **blocked on user go-ahead to execute**
- [ ] **R1.1** dry-run clean on real panel (eligibility stats sane at every rebalance date)
- [ ] **R1.2** full val-split run: 4 policies vs 7 baselines, report + CIs
- [ ] **R1.3** `R1_FINDINGS.md` written; R2 grid frozen before any further runs

### Deviation from the original spec (worth flagging)

`eligible_mask`'s original design (§2.1) assumed a NaN-based finite-return coverage
check. Tracing `data.py`'s `load_price_panel` showed `close/high/low` are always
`ffill().bfill()`'d dense (paper Sec. 3.3's flat-decay convention) — a recent
IPO/entrant's pre-listing days are backfilled to its first real price, not NaN, so
`isfinite` would never catch it. Implemented instead as: find the leading run of
closes identical to the trailing window's first value (exactly what a backfilled
prefix looks like) and require real (non-flat) coverage >= `min_history_frac`. Covered
by `test_eligible_mask_entrant_seasoning` and the entrant assertions inside
`test_harness_smoke`.

## Explicit non-goals (R0/R1)

Leverage / shorting (long-only simplex is a hard constraint of `run_backtest`'s cost
model); no-trade bands (R2); EWMA as default (R2 comparison, LW is primary); per-name
caps as default (`max_weight=1.0`, probed in R2); any μ estimate anywhere.
