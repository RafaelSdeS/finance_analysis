# Portfolio V1 — Implementation Plan

Companion to `PORTFOLIO_ARCHITECTURE_PROPOSAL.md`. That doc argues *what* to build
(predict-then-optimize: LightGBM α → cost-aware convex optimizer). This doc is *how
to build it*, grounded in what's actually in the repo as of **2026-07-24** (every
file/function/column below was verified against the tree, not assumed).

**Status:** pre-implementation. No modeling code exists in `src/` yet.

---

## 0. Corrections to the proposal's roadmap (potholes found while grounding it)

The proposal's §11 roadmap has several references that don't match the repo. Fixed here;
the build order in §2 below supersedes §11.

| # | Pothole in proposal | Reality | Fix |
|---|---------------------|---------|-----|
| P1 | Phase 1 says "write this once as a `(df, date, threshold) → tickers` helper." | **Already written and tested**: `src/build_dataset/build_top50_universe.py::build_top50_membership` + `filter_to_top50_universe` do exactly this (trailing 252d `traded_amount`, quarterly rebalance, locked membership, union recovers delisted names, no lookahead). Test `tests/build_dataset/test_top50_universe.py` passes (11 checks). | **Reuse it.** Do not rewrite. Parametrized by `top_n` and `rebalance_freq` (default `"Q"`). Phase 1 collapses to a thin call + a real-data smoke test. |
| P2 | §7 / Phase 1 cite `TOP50_UNIVERSE_VALIDATION.md` and `test_blue_chip_tickers.py`. | **Correction (this doc had it wrong too — verified 2026-07-24):** `test_blue_chip_tickers.py` **does exist**, just at `tests/data_collection/test_blue_chip_tickers.py`, not `tests/build_dataset/` — it passed in the `data` test group. It's a static-list raw-data spot-check on fixed blue chips (PETR4, VALE3, ...), not the point-in-time universe logic, but it's real. `TOP50_UNIVERSE_VALIDATION.md` and the doc cited in code comments, `docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md`, are **confirmed absent** (repo-wide `find` and directory listing both empty). | Real scaffolding to lean on: `test_top_traded_quality.py`, `test_top50_universe.py`, `test_top50_ml_readiness.py`, `test_blue_chip_tickers.py`. Ignore the two phantom docs only. |
| P3 | Phase 0 (restatement jitter check) is listed **first** but says "rerun the full walk-forward backtest, check alpha survives." | The backtest doesn't exist until Phases 2b–5. Phase 0 literally can't run before the thing it perturbs. | **Move Phase 0 to run after the first full backtest exists** (it's a wrapper around the harness). It's cheap *once the harness exists*, not first. Becomes step §2.6 below — a gate before trusting results / before expanding the universe. |
| P4 | §3.2 frets about fitting scaling train-only via `iter_fit_windows()`. | **LightGBM is scale-invariant.** Stage A needs no scaler. `scale_features.py` / `iter_fit_windows` / `feature_scaler.joblib` are irrelevant to a GBM. | **Drop the scaler from V1 entirely.** The `iter_fit_windows` seam only matters if a NN α-head is tried (deferred, Phase 7). One less moving part. |
| P5 | §3.2 "refit at each walk-forward step" implies reusing the existing split seam. | `iter_fit_windows` returns a *single* window; `split_config.json` is one fixed train/val/test split. | The walk-forward **retrain loop is new code owned by the backtest module**, with its own quarterly retrain schedule. Do **not** conflate it with `iter_fit_windows` (scaler-only, and we're not using the scaler — see P4). |
| P6 | §5 objective defines `Δw = w − w_{t-1}` but the universe **churns every quarter** (§7). | When a ticker drops out of the top-N, `w_{t-1}` still holds it → forced liquidation → turnover that isn't an optimizer "decision." Unhandled in the proposal. | The optimizer runs over `index = union(held names, current universe)`. Names that left the universe are pinned `w=0` (forced sale); their forced-sale turnover **is counted** in cost + turnover metrics. Spelled out in §2.4. |
| P7 | Label ("6–12m forward excess return over CDI") never made concrete. | `excess_return` feature exists but is *daily/trailing*, not the forward label. `cdi` is %/trading-day. | Concrete formula in §2.2, incl. the `adj_close_precision_degraded` mask and dropping the last H rows per ticker. |
| P8 | Deps assumed available. | `scikit-learn`+`joblib` present; **`lightgbm` and `cvxpy` are not** in `requirements.txt`. | Add `lightgbm` only at Phase 2b, `cvxpy` only at Phase 4 — not upfront. **Phase 3 needs no new dep**: `sklearn.covariance.LedoitWolf` already ships with the installed scikit-learn. |
| P9 | — | `src/visualizations/{agent_performance,agent_vs_benchmarks,rolling_eval_results}.ipynb` reference the **deleted** `src.agent.*` / `rolling_eval.py` / `artifacts/backtest` (the 2026-07-23 RL reset). | Orphaned, non-blocking. Don't resurrect (RL was deliberately reset). Flag for later cleanup only. |

---

## 2. Build order (ponytail: harness-first, thinnest end-to-end skeleton before any model)

The proposal's phases are right but mis-ordered for *de-risking*. The smallest thing that
produces a **real, trustworthy out-of-sample number** is the evaluation harness + a no-model
baseline on the already-built point-in-time universe. Every later phase (α, Σ, optimizer, and
the Phase-0 jitter test) is a swap into that same harness. Build it first and thinnest.

New code lives in a new package **`src/portfolio/`** (Stage 3). One shared helper module,
no premature abstraction.

### 2.1 — Point-in-time liquid universe  *(proposal Phase 1 — mostly done)*
- [x] **Pre-existing pipeline validated end-to-end on real data (2026-07-24), before building on it:** `python -m src.build_dataset.build_top50_universe` ran clean on the real `ml_dataset.parquet` in 3s — 171 tickers ever qualified for top-50, 190,951 rows after the earliest-fundamental trim (14.6% of the full dataset, a sane share). Membership churn confirmed on real dates: 103 quarterly periods from 2001-03-30 to 2026-07-14; comparing period 4 (2001) vs period 100 (2025), overlap was 11 of ~50 members — real, expected churn, not a static list. No errors, no latent bugs found in this dependency.
- [ ] `src/portfolio/universe.py`: thin wrapper over the **existing** `build_top50_membership` / `filter_to_top50_universe`. Expose `liquid_universe(df, top_n=50) -> membership` and the rebalance-date list (the membership `start` dates *are* the quarterly rebalance calendar — reuse them, don't recompute).
- [ ] Smoke test on **real** `ml_dataset.parquet` (not just the synthetic unit test): assert membership churns over time (2015 members ≠ 2020 members), every member has ≥252 trailing days as of its qualifying date, and no member's first qualifying date precedes its first trade.
- Done when: universe membership table materializes for the real dataset and the churn/no-lookahead asserts pass.

### 2.2 — Forward label + feature matrix  *(prereq for 2b; fixes P7)*
- [ ] `src/portfolio/labels.py::forward_excess_return(prices, horizon_td)`:
  - Per ticker, date-sorted: `fwd_ret = adj_close.shift(-H) / adj_close - 1`.
  - `fwd_cdi = rolling forward product of (1 + cdi/100) over the next H trading days − 1` (cdi is %/day).
  - `label = fwd_ret − fwd_cdi`.
  - `H`: 252 td (12m) as default, 126 td (6m) as an ablation switch.
  - Mask `label = NaN` where `adj_close_precision_degraded == 1` on the base row.
  - Last H rows per ticker are NaN (no forward window) → excluded from training. **This is the leakage boundary; nothing downstream may impute them.**
- [ ] `src/portfolio/features.py::feature_columns()`: return the **§4.4-E keep list (121 cols)** as a literal, minus `sector` unless one-hot is explicitly wanted. Import `manifest.LOOKAHEAD_TAINTED_COLS` and `assert` none of them are in the keep list (a live guard, so a future dataset change can't silently leak a tainted column in).
- Done when: label column builds, the tainted-column `assert` passes, and a spot check confirms the last-H-rows-per-ticker are NaN.

### 2.3 — Backtest harness + equal-weight baseline  *(proposal Phase 2a — the real foundation)*
- [ ] `src/portfolio/backtest.py`: given a `weights_fn(date, universe, state) -> weights`, walk quarterly rebalance dates, apply weights, accrue daily portfolio returns between rebalances (equities via `adj_close`, cash via `cdi`), charge cost on `Δw` at each rebalance (incl. forced-sale turnover from P6). Returns a per-day equity curve + per-rebalance weight/turnover log.
- [ ] `src/portfolio/metrics.py`: annualized return, Sharpe, **deflated Sharpe** (López de Prado — small, self-contained), max drawdown, annual turnover, avg holding period (yrs), no-trade fraction, regime-sliced returns (SELIC median split + explicit crisis windows).
- [ ] First `weights_fn`: **equal-weight** the current liquid universe. This is the floor everything must beat.
- [ ] Baselines in the same harness: buy-and-hold BOVA11, 100% CDI.
- [ ] Test: on a 2-ticker synthetic panel, assert equity curve compounding, turnover accounting (incl. a forced exit), and that a zero-cost / no-rebalance run reproduces buy-and-hold exactly.
- Done when: equal-weight, BOVA11, and 100%-CDI curves + the §8 metric table print out-of-sample on the real universe.

### 2.4 — Cost-aware convex optimizer  *(proposal Phase 4 — needs Σ from 2.5 and α from 2b, but the program is standalone)*
- [ ] Add `cvxpy` to `requirements.txt`.
- [ ] `src/portfolio/optimizer.py::solve(alpha, Sigma, w_prev, c1, c2, lam, w_max)` implementing §5 exactly:
  `max αᵀw − (λ/2)wᵀΣw − c₁‖Δw‖₁ − (c₂/2)‖Δw‖₂²`, s.t. long-only, `w_i ≤ w_max`, `Σw + w_cash = 1`, `w_cash ≥ 0`.
  - **Index = `union(w_prev.index, current_universe)`** (P6). Names not in the current universe pinned `w=0`. Cash column: `α_cash = forward CDI carry`, `Σ` row/col ≈ 0.
  - `c₁` = **one-way** cost (§5 note: not the round-trip 0.06% — that double-counts), per-asset liquidity-scaled via `amihud_illiquidity`/`turnover_ratio` (§6). `c₂` starts at 0.
- [ ] Test: a 3-asset toy where the analytic no-trade band is known — assert a below-band α improvement produces **exactly** zero trade, and above-band produces a trade.
- Done when: the solver returns feasible long-only weights summing to 1 (incl. cash) and the no-trade-band test passes.

### 2.5 — Risk model Σ  *(proposal Phase 3 — no new dep, P8)*
- [ ] `src/portfolio/risk.py::shrinkage_cov(returns_window)` = `sklearn.covariance.LedoitWolf` on trailing daily equity returns of the current universe. Cash row/col ≈ 0.
- [ ] Validate conditioning (condition number, PSD) vs raw sample cov on a real window.
- Done when: Σ is PSD and well-conditioned on a real trailing window; wired into 2.4.

### 2.6 — Forecaster (Stage A)  *(proposal Phase 2b)*
- [ ] Add `lightgbm` to `requirements.txt`.
- [ ] `src/portfolio/alpha.py`: LightGBM regression on the 2.2 forward-excess-return label, features from 2.2, **native NaN handling** (no imputation — the flags in §4.4-E explain the NaNs). `monotone_constraints` on `earnings_yield_vs_selic` per §4.1.
- [ ] **Walk-forward retrain loop (new, owns its schedule — P5):** at each quarterly rebalance date `t`, train on all samples whose label window `[d, d+H]` ends `≤ t` minus an embargo (**purged + embargoed**, §3.2); predict α for the current cross-section. Expanding window.
- [ ] Diagnostics: out-of-sample rank-IC of α vs realized forward return; SHAP interaction on (valuation × `f_score`) per §4.2; feature-importance watch for raw-SELIC-level splits per §4.1.
- Done when: α ranks forward returns out-of-sample (rank-IC > 0 on held-out rebalances) **and** an α-weighted (still equal-ish, pre-optimizer) portfolio beats the 2.3 equal-weight floor.

### 2.7 — Full walk-forward backtest + evaluation  *(proposal Phase 5)*
- [ ] Wire α (2.6) → Σ (2.5) → optimizer (2.4) → harness (2.3). Run purged/embargoed walk-forward end to end.
- [ ] Report the full §8 panel: deflated Sharpe, turnover/holding-period distributions, regime slices, **cost-sensitivity curve** (one-way `c` = 0.03% → 0.15% → 0.3%, round-trip shown alongside), vs all three baselines.
- Done when: the panel prints out-of-sample and the strategy's relationship to the CDI floor is stated honestly (beating CDI net of cost is the bar; not beating it is a valid answer, not a bug).

### 2.6′ — Restatement sensitivity / jitter check  *(proposal Phase 0, relocated per P3)*
- [ ] `src/portfolio/jitter.py`: perturb the α-driving fundamental ratios (§4.2 value+quality axes) with noise at a **stated** magnitude (±10% leverage/margin, ±15% earnings-derived — write it in the output), rerun 2.7, check whether measured alpha survives.
- Runs **after** 2.7 exists (it wraps the harness). Gate before Phase 6 universe expansion. A pass = "not yet falsified," not "clean" (§9.2).

### 2.8 — Expand universe  *(proposal Phase 6)*
- [ ] Raise `top_n` (same point-in-time helper), re-run 2.7. Data-filter change only — no architecture change.

**Deferred (proposal Phase 7+, do not build now — YAGNI):** quantile/probabilistic α head (Stage A′), CVaR risk term, multi-period Boyd/G-P optimizer, sequence-model α, offline-RL allocator. Each is gated on beating the V1 baseline on the deflated metric first.

---

## 3. Dependencies to add (only when its phase starts)
- Phase 2.4: `cvxpy`
- Phase 2.6: `lightgbm`
- Phase 3 (2.5): **none** — `LedoitWolf` is in the installed scikit-learn.
- Optional/deferred: `shap` (diagnostic only).

## 4. Invariants every phase must uphold (from CLAUDE.md + the proposal)
- No lookahead: features as-of `t` only; labels are forward and their last-H-per-ticker rows are NaN and never imputed.
- Never feed `manifest.LOOKAHEAD_TAINTED_COLS` (9 cols) to the model — enforced by the `assert` in 2.2.
- Universe is point-in-time (§7 mandate) — the existing helper already guarantees this; don't add a second, drifted "liquid universe" code path.
- Cost is one-way in the objective (§5), liquidity-scaled per asset (§6).
- Tests are plain `python tests/…py` scripts (no pytest); add each phase's test to `tests/run_all.py` groups.
