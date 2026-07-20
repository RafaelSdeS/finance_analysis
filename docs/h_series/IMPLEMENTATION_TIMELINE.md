# Implementation Timeline — What, How, Why

Same 3-phase shape as a standard "validate the premise → add complexity only if warranted →
optimize and prepare to deploy" plan — gated by evidence at each phase boundary, not calendar
time. See `H_SERIES_ROADMAP.md` for the checklist form and the full rationale for each
divergence from the naive version of this plan; this doc is the one-glance version.

## Phase 1 — Validate the Premise

| # | Stage | What | How | Why |
|---|-------|------|-----|-----|
| 0 | Data question | Confirm the daily-price source h_series will use for the anchor's covariance estimate | Investigate `features.py`/`loaders.py` — h_series' panel is monthly, stage-1's anchor needs daily returns | Blocks H3 stage 3; five-minute investigation, not a milestone |
| 1 | **H3** | Build a factor model + risk-based portfolio construction, walk-forward backtest, evaluate vs. BOVA11/CDI — then gate on the result | Fitted characteristic combination (`composite.py`: ridge/elastic-net/GBM, walk-forward-selected — not hand-picked weights); learned anchor (`anchor.py`: min-variance/risk-parity via `risk_portfolios.py`'s solvers — not a cap-weight/equal-weight guess, and not mean-variance fed a hand-built expected-return formula, since M1-M4 already found no reliable cross-sectional mu); fitted blend (`blend.py`, κ from OOF skill) | H2's own ablation showed a hand-picked anchor alone explained ~all of its reported IR (0.95 vs. 1.12 tilted, permutation-null p=0.78) — the anchor and blend were asserted, not fit. This is "validate the premise" done right: the gate is a permutation-null + quintile-monotonicity test on both pre-2024 and post-2024 splits, not a single raw Sharpe cutoff (H2 already showed a Sharpe-like metric can look fine and still fail that stricter test) |

**Gate:** PASS → Phase 2. FAIL → stop and report that H1's survivors carry nothing beyond a
risk-based prior at this sample size — a real result, not a bug to chase.

## Phase 2 — Add Complexity, Only If Warranted

| # | Stage | What | How | Why |
|---|-------|------|-----|-----|
| 2 | **H3a** (`H3A_MULTI_HORIZON_SCREENING_PLAN.md`) | Extend the characteristic screen to short-horizon (~1-5d) and long-horizon (12m+) signals, beyond the current 21/63d set | Same FDR/NW-HAC gate as H1, new horizon columns | "Try a fancier ranker, adopt only if it wins" — already partly built into H3 stage 2 (ridge vs. elastic net vs. GBM, picked by walk-forward OOS IC), extended here to new horizons. Sequenced *after* H3 specifically so a result isn't confounded between the anchor fix and new characteristics |

**Excluded, not deferred:** macro regime classification. Bucketing into "high rate / inflation /
growth / stress" is manual classification regardless of whether the boundary values are fit —
rejected by standing rule (`H_SERIES_ROADMAP.md`'s "Rejected alternative" section), not a gap.

## Phase 3 — Optimize & Prepare to Deploy

| # | Stage | What | How | Why |
|---|-------|------|-----|-----|
| 3 | **H4** | The actual long-term-investor behavior: hold while thesis valid, sell on deterioration/better opportunity, cash when opportunities are poor | New `position_tracker.py` — hysteresis hold/sell bands (θ_buy/θ_sell) + cost-aware trade trigger + cash-preference rule, **all thresholds walk-forward selected** | The fitted version of "rebalance quarterly, or when drift exceeds X%, with position limits" — same goal (control turnover, avoid concentration), but every number is selected from data instead of asserted. H3 alone only produces a fresh monthly reweight, not this behavior |
| 4 | H5 (`H5_OBJECTIVE_CALIBRATION_ROBUSTNESS_PLAN.md`) | Calibrate the risk-aversion parameter H3's anchor currently leaves at solver default; stress-test result stability | Explicit γ selection (or a small evaluated range) instead of a silent default; bootstrap resampling (`block_bootstrap_ci`) under perturbed costs/universe | The "measure real costs and slippage before trusting the backtest" step, done with tools this project already built (H2's permutation-null machinery, `rl_agent/metrics.py`) rather than a manual spot-check |
| 5 | H6 (`H6_FINAL_REPORT_PLAN.md`) | Final comprehensive report | Adapt (not rebuild) the existing EIIE reporting infra (`rl_agent/plots.py`, `metrics.py`) into one report for the finished pipeline | Current outputs are text-only `*_FINDINGS.md`/json; a full side-by-side against every baseline (UBAH, UCRP, Best-Stock, BOVA11, CDI, old EIIE agent, old H2 composite) needs the richer format |
| 6 | (Contingent) Deployment plan (`H7_DEPLOYMENT_PLAN.md`) | Monthly decision-date scheduling, real-time filing-date latency handling, a runnable "generate this month's target portfolio" script | Reuse the CVM filing-date infra already built for Stage 2 | Only worth designing if H1-H6 collectively hold up net of realistic costs — not scheduled by default |
| — | RL-for-execution (deferred, unscheduled) (`RL_EXECUTION_LAYER_PLAN.md`) | Cost-aware order scheduling given H4's already-decided target weights — *not* stock selection | Would reuse RL machinery narrowly, only for execution | Only revisited if H4's simpler rule-based trigger proves insufficient — ladder discipline, don't reach for it first |

## Set aside for now (not forgotten)

**Tax-aware cost model** — Brazilian capital-gains tax, dividend withholding. Explicitly dropped
from the active plan per 2026-07-20 decision. Current costs stay brokerage-only (3bp,
`c_sell=c_buy=0.03%`). Revisit after Phase 3 if still relevant.

## Standing rule across every stage

No hand-picked weights, no hand-picked formulas standing in for a fitted quantity, and no
manually-defined categories/classifications anywhere. Every threshold, weight, or blend parameter
is selected by the same walk-forward CV discipline H1/H2 already established, or it doesn't go in.

## Explicitly excluded, not deferred

Deep architectures (Transformers, LSTMs/GRUs, DQN, PPO/DDPG/SAC as the core agent, MoE/
hierarchical nets) — ruled out by the earlier architecture comparison and by M1-M4's evidence,
not omitted for lack of time.
