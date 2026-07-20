# H-Series Roadmap (post-H2)

Order of work and why, tying together `H3_PORTFOLIO_CONSTRUCTION_PLAN.md` and
`H4_LONG_TERM_BEHAVIOR_PLAN.md`. Each step only starts once the previous one is either done or
has produced a clear result (pass or a real, reportable fail).

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
      - **Decision point:** PASS → proceed. FAIL → real result (H1 survivors carry nothing
        beyond a risk-based prior at this sample size) — worth stopping and reporting, not
        automatically proceeding to H4 on a broken foundation.

- [ ] **2. H3a — multi-horizon characteristic screening** (not yet detailed as its own doc).
      Extends H1's screen to short-horizon (~1-5d) and genuinely long-horizon (12m+)
      characteristics, on top of the existing 21/63-day set. Deliberately sequenced *after* H3,
      not before or alongside it — doing it first would confound whether any H3 result came from
      the anchor fix or from new characteristics. If H3a surfaces new survivors, rerun H3's
      stage 2 combination on the enlarged set as a separate, cleanly attributable comparison.

- [ ] **3. H4 — long-term investor behavior layer** (`H4_LONG_TERM_BEHAVIOR_PLAN.md`). Consumes
      whichever H3(/H3a) score series is current. Position tracker, hysteresis hold/sell bands,
      cost-aware trade trigger, cash-preference rule — all thresholds walk-forward selected, none
      hand-picked. Doesn't require H3 to have formally PASSED its gate to be worth testing (H4
      answers a different question — does stateful hold/sell beat naive reweighting — but it
      does require H3 to have been *run*, since it's the input).
      - **Decision gate:** must beat H3's naive monthly reweight net of costs, not just BOVA11/CDI.

- [ ] **4. (Deferred, unscheduled, contingent) Narrow RL for execution only.** Not stock
      selection — cost-aware order scheduling given H4's already-decided target weights. Only
      revisited if H4's simple rule-based trigger proves insufficient in practice. Not a
      committed step; listed here so it isn't forgotten, not because it's next.

## Beyond H4 (draft — speculative, not yet discussed in as much depth as H3/H4)

These weren't derived from a diagnosed failure the way H3 was from H2 — they're gaps I can see
from what H0-H4 do and don't cover. Treat as a starting draft to react to, not a committed plan.

- [ ] **5. H5 — objective calibration + robustness.** Two things H3/H4 currently leave implicit:
      (a) the risk-aversion parameter behind the mean-variance anchor is whatever the solver
      defaults to, never deliberately chosen (flagged as an open item in H3 stage 4) — decide it
      explicitly, or evaluate across a small γ range instead of one silent default; (b) stability
      testing — rerun H3+H4's pass/fail under perturbed cost rates, bootstrap resampling (reuse
      `block_bootstrap_ci` from `src/rl_agent/metrics.py`), and small universe changes, to check
      the result isn't a single-path artifact the way H2's was.

- [ ] **6. H6 — tax-aware cost model.** Every cost figure so far (`c_sell=c_buy=0.03%`) is a
      brokerage-fee proxy — no Brazilian capital-gains tax, no monthly R$20k stock-sale
      exemption, no dividend withholding. For a strategy whose main selling point is minimizing
      turnover, that's a real gap: H4's cost-aware trade trigger was tuned against an incomplete
      cost signal, and after-tax returns could look meaningfully different from what's reported
      so far. Rebuild the trigger against realistic after-tax expected return.

- [ ] **7. H7 — final comprehensive report.** Adapt (not rebuild) the existing EIIE reporting
      infra (`src/rl_agent/plots.py`, `metrics.py` — Sharpe/Sortino/Calmar/VaR/CVaR, turnover,
      cost drag, IR vs. BOVA11, bootstrap CIs) into one report for the final pipeline, replacing
      the current text-only `*_FINDINGS.md`/json outputs. This is where a full side-by-side
      against every baseline (UBAH, UCRP, Best-Stock hindsight, BOVA11, CDI, the old EIIE agent,
      the old H2 composite) would live.

- [ ] **8. (Contingent, not scheduled) Deployment plan.** Monthly decision-date scheduling,
      real-time filing-date latency handling, a runnable "generate this month's target
      portfolio" script. Only worth planning if H1-H7 collectively hold up — no reason to design
      deployment mechanics for a strategy that hasn't yet cleared its own gates net of realistic
      costs.

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
