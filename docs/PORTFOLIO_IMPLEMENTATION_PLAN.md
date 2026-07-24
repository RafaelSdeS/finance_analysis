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
- [x] `src/portfolio/universe.py`: thin wrapper over the **existing** `build_top50_membership` / `filter_to_top50_universe`. Exposes `liquid_universe()`, `rebalance_dates()`, `universe_at()`.
- [x] Smoke test on **real** `ml_dataset.parquet`: `tests/portfolio/test_universe.py` (data group) — 103 rebalance periods 2001-03-30→2026-07-14, churn confirmed (2001 vs 2025 overlap 11/~50), every member has ≥252 trailing days as of its qualifying date (min observed exactly 252), no qualifying period starts before that ticker's first row. All pass.
- **Done — 2026-07-24.**

### 2.2 — Forward label + feature matrix  *(prereq for 2b; fixes P7)*
- [x] `src/portfolio/labels.py::forward_excess_return(df, horizon_td)`: implemented exactly as specced (shift-based fwd_ret, log1p-cumsum-based fwd_cdi over the same H-row window, `adj_close_precision_degraded` mask, non-positive-price mask). `tests/portfolio/test_labels.py` (fast) checks every value against an independently hand-computed reference (not the implementation's own logic), confirms the last-H-rows-per-ticker are NaN, and confirms output realigns correctly regardless of input row order. All pass.
- [x] `src/portfolio/features.py::feature_columns()`: the literal 120-numeric (+1 `sector`) keep-list, with a live `assert` against `manifest.LOOKAHEAD_TAINTED_COLS`. `tests/portfolio/test_features.py` (data group) additionally verifies against the real dataset: exact counts (120/121), zero tainted-column overlap, and every listed column actually exists in `ml_dataset.parquet`. All pass.
- **Done — 2026-07-24.**

### 2.3 — Backtest harness + equal-weight baseline  *(proposal Phase 2a — the real foundation)*
- [x] `src/portfolio/backtest.py`: `run_backtest()` implemented as specced — genuine buy-and-hold-with-share-drift between rebalances (not daily renormalization, which would erase the "winners run" behavior §4.3 depends on), universe-exit force-liquidation counted in turnover (P6), cost applied one-way to equities only (not the cash leg). Plus `equal_weight_fn`, `buy_and_hold_curve`, `cdi_curve`.
- [x] `src/portfolio/metrics.py`: `annualized_return`, `sharpe_ratio`, `deflated_sharpe_ratio` (Bailey & López de Prado, `scipy.stats.norm`), `max_drawdown`, `turnover_stats` (annual turnover, avg holding period, no-trade fraction), `regime_slice` (SELIC median split + `CRISIS_WINDOWS`), `full_report()`.
- [x] `equal_weight_fn` is the first/reference `weights_fn`.
- [x] Baselines: `buy_and_hold_curve` (BOVA11) and `cdi_curve` (100% CDI), same harness/metrics.
- [x] `tests/portfolio/test_backtest.py` (fast): a 2-ticker synthetic panel checked against an independent, non-vectorized reference simulator — equity curve, per-rebalance turnover (incl. the forced-exit case, verified nonzero even though `equal_weight_fn` never mentions the exiting ticker), and a zero-cost no-op run reproducing a static buy-and-hold exactly. `tests/portfolio/test_metrics.py` (fast) checks each stat against direct formulas, plus DSR sanity (bounded in [0,1], ranks a clear edge above noise, exactly 0.5 for a deterministic zero-Sharpe series, monotonic in `n_trials`). All pass.
- [x] `src/portfolio/run_baseline.py` — real end-to-end run (`python -m src.portfolio.run_baseline`, 1.6s): equal-weight liquid universe (2001–2026, 6279 daily obs) returned 14.1% annualized / 0.65 Sharpe / -62% max drawdown / 0.90 annual turnover / 2.23y avg holding period; BOVA11 buy-and-hold (rebased to its 2008 inception) 9.4% / 0.49 Sharpe / -50% max drawdown; 100% CDI 12.0% / near-riskless (max drawdown exactly 0, as it must be for a monotonically-compounding cash accrual). All directionally sane, no NaN/crashes. Regime slices correctly show both equity strategies deeply negative through GFC/COVID while CDI stays positive throughout.
- **Done — 2026-07-24.** `python tests/run_all.py --group all`: 38/38 passed (25 fast + 13 data, up from 25/8 before this phase).

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
