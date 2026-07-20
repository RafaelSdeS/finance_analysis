# H-Series Roadmap (post-H2)

Structured as 3 phases — validate the premise, add complexity only if warranted, optimize and
prepare to deploy — the same shape as any sound quant-research plan. Each phase only starts once
the previous one is either done or has produced a clear, reportable result (pass or fail).

## Phase 1 — Validate the Premise

- [ ] **0. Resolve H3's open data question.** Confirm where daily close prices come from for the
      anchor's trailing-covariance estimate (h_series' panel is monthly; the anchor needs daily
      returns). Quick investigation, not a milestone — blocks H3 stage 3 specifically.

- [ ] **1. H3 — fix the diagnosed H2 defect** (`H3_PORTFOLIO_CONSTRUCTION_PLAN.md`). Runs on the
      **same H1 survivor set H2 used** — deliberately not expanded yet, so a pass/fail here is
      attributable to the anchor + blending fix alone, not confounded with new characteristics.
      - Stage 2: model-class selection (ridge/elastic net/shallow GBM via walk-forward OOS IC).
      - Stage 3: learned anchor (min-variance/risk-parity, reusing `risk_portfolios.py` solvers).
      - Stage 4: blend (shrinkage interpolation first; full Black-Litterman only if needed).
      - Stages 5-7: constraints, rerun `permutation_null()`/quintile-monotonicity, decision gate
        on both pre-2024 and post-2024 splits.
      - **Decision point:** PASS → Phase 2. FAIL → real result (H1 survivors carry nothing
        beyond a risk-based prior at this sample size) — worth stopping and reporting, not
        automatically proceeding to Phase 2 on a broken foundation. This is the same "prove it
        beats the benchmark before building further" gate as any factor-model validation phase,
        just enforced with a permutation-null + monotonicity test instead of a single Sharpe
        cutoff — a raw "Sharpe > 0.5" threshold on ~90 monthly observations is itself a coin flip
        away from noise; H2 already showed why (its composite's IR looked fine until the
        permutation test revealed it was indistinguishable from a shuffled score).

## Phase 2 — Add Complexity, Only If Warranted

- [ ] **2. H3a — multi-horizon characteristic screening** (`H3A_MULTI_HORIZON_SCREENING_PLAN.md`).
      Extends H1's screen to short-horizon (~1-5d) and genuinely long-horizon (12m+)
      characteristics, on top of the existing 21/63-day set. Deliberately sequenced *after* H3,
      not before or alongside it — doing it first would confound whether any H3 result came from
      the anchor fix or from new characteristics. If H3a surfaces new survivors, rerun H3's
      stage 2 combination on the enlarged set as a separate, cleanly attributable comparison.
      This is the "train a fancier ranker, adopt it only if it actually beats the simpler one"
      idea — already built into H3 stage 2's model-class competition (ridge vs. elastic net vs.
      GBM, picked by walk-forward OOS IC, not by preference) rather than treated as a later
      add-on.

- [ ] **Not on this roadmap: macro regime classification.** Explicitly excluded per the standing
      rule below — defining discrete regime buckets is itself manual classification, independent
      of whether the boundary values are fit. Nothing here adjusts allocation by a human-defined
      regime label.

## Phase 3 — Optimize & Prepare to Deploy

- [ ] **3. H4 — long-term investor behavior layer** (`H4_LONG_TERM_BEHAVIOR_PLAN.md`). Consumes
      whichever H3(/H3a) score series is current. Position tracker, hysteresis hold/sell bands,
      cost-aware trade trigger, cash-preference rule — all thresholds walk-forward selected, none
      hand-picked. This is the fitted version of "rebalance quarterly, or when drift exceeds a
      fixed %, with position limits" — same concern (control turnover, avoid concentration), but
      every number in it is selected from data, not asserted. Doesn't require H3 to have formally
      PASSED its gate to be worth testing (H4 answers a different question — does stateful
      hold/sell beat naive reweighting — but it does require H3 to have been *run*, since it's
      the input).
      - **Decision gate:** must beat H3's naive monthly reweight net of costs, not just BOVA11/CDI.

- [ ] **4. H5 — objective calibration + robustness** (`H5_OBJECTIVE_CALIBRATION_ROBUSTNESS_PLAN.md`).
      Two things H3/H4 currently leave implicit:
      (a) the risk-aversion parameter behind the anchor is whatever the solver defaults to, never
      deliberately chosen (flagged as an open item in H3 stage 4) — decide it explicitly, or
      evaluate across a small γ range instead of one silent default; (b) stability testing —
      rerun H3+H4's pass/fail under perturbed cost rates, bootstrap resampling (reuse
      `block_bootstrap_ci` from `src/rl_agent/metrics.py`), and small universe changes, to check
      the result isn't a single-path artifact the way H2's was. This is the "measure real
      transaction costs and slippage before trusting the backtest" step, done with the tools this
      project already built rather than a manual sensitivity check.

- [ ] **5. H6 — final comprehensive report** (`H6_FINAL_REPORT_PLAN.md`). Adapt (not rebuild) the existing EIIE reporting
      infra (`src/rl_agent/plots.py`, `metrics.py` — Sharpe/Sortino/Calmar/VaR/CVaR, turnover,
      cost drag, IR vs. BOVA11, bootstrap CIs) into one report for the final pipeline, replacing
      the current text-only `*_FINDINGS.md`/json outputs. This is where a full side-by-side
      against every baseline (UBAH, UCRP, Best-Stock hindsight, BOVA11, CDI, the old EIIE agent,
      the old H2 composite) would live.

- [ ] **6. (Contingent, not scheduled) Deployment plan** (`H7_DEPLOYMENT_PLAN.md`). Monthly
      decision-date scheduling, real-time filing-date latency handling, a runnable "generate this
      month's target portfolio" script. Only worth planning if H1-H6 collectively hold up — no
      reason to design deployment mechanics for a strategy that hasn't yet cleared its own gates
      net of realistic costs.

- [ ] **(Deferred, unscheduled, contingent) Narrow RL for execution only** (`RL_EXECUTION_LAYER_PLAN.md`).
      Not stock selection — cost-aware order scheduling given H4's already-decided target weights.
      Only revisited if H4's simple rule-based trigger proves insufficient in practice. Not a
      committed step; listed here so it isn't forgotten, not because it's next.

## Deliberately out of scope right now

**Tax-aware cost model** — considered (Brazilian capital-gains tax, dividend withholding), set
aside for now by explicit decision, not forgotten. Current cost figures stay brokerage-only
(`c_sell=c_buy=0.03%`, i.e. 3bp — matches standard transaction-cost assumptions). Revisit once
Phase 3 is otherwise complete, if still relevant.

## Where this already matches a standard "validate → add complexity → optimize/deploy" plan

- Walk-forward backtesting across the full available history (H0's spine starts 2018-12-31,
  driven by data coverage, not a deliberate cutoff choice) — same principle as any such plan.
- 3bp transaction costs — already the project's standing assumption, not new.
- Risk-based portfolio construction (mean-variance-family) over a black-box model — same
  conclusion, reached independently via M1-M4/R1-R3's evidence.
- A "prove it before adding complexity" gate at each stage.
- **Monthly**, not quarterly, rebalancing — a deliberate difference, not a gap: already decided
  earlier in this project ("monthly rebalancing acceptable for implementation simplicity");
  H4's hysteresis/cost-trigger layer means monthly is just the *evaluation* cadence, not the
  *trade* cadence — trades only fire when the fitted trigger says so, same spirit as "quarterly
  or on >20% drift" but the threshold is fit, not asserted.

## Rejected alternative: hand-picked factor + expected-return model + manual classification

A "factor scorer (40% value / 40% quality / 20% growth) + mean-variance optimizer using
dividend-yield-plus-earnings-growth as expected returns + a macro regime classifier" design was
considered and rejected. Three distinct instances of the same underlying problem, not one:

1. **Hand-picked factor weights** (40/40/20) — already flagged as a no-go.
2. **Hand-picked expected-return formula** (dividend+growth) — a fitted estimate replaced by an
   asserted one. Also contradicts the R-series' own finding (`risk_portfolios.py` docstring):
   M1-M3 found no measurable cross-sectional mu — why H3's anchor is a **variance-only** solver
   (min-variance/risk-parity), never mean-variance fed a hand-built return proxy.
3. **Manual classification** (a macro regime classifier bucketing into "high rate / inflation /
   growth / stress") — rejected outright, not just the threshold values inside it. **Defining
   the categories themselves is the hardcoding**, independent of whether the boundary numbers are
   walk-forward fit. Nothing on this roadmap classifies data into human-defined buckets anywhere
   — not regimes, not factor tiers, not score bands. Every H4 threshold (θ_buy/θ_sell,
   cash-preference cutoff) is a **continuous** decision-rule parameter selected by walk-forward
   grid search, not a category boundary. If regime-like structure is ever wanted, the only
   acceptable form is unsupervised structure the data itself produces (e.g. a data-selected
   number of clusters via BIC) — and even that should be treated skeptically at this sample size,
   not reached for by default.

**Standing rule for every future H-stage:** no hand-picked weights, no hand-picked formulas
standing in for a fitted quantity, no manually-defined categories/classifications anywhere in the
pipeline — fit or select from data via the existing walk-forward CV discipline, every time.

## Explicitly not on this roadmap

Deep architectures ruled out by the earlier comparison (Transformers, LSTMs/GRUs, DQN, PPO/DDPG/SAC
as the core agent, MoE/hierarchical nets) and hand-picked signal weights — both already rejected
by evidence gathered earlier in this project, not omissions.
