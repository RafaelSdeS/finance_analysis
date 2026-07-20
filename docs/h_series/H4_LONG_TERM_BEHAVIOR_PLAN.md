# H4 — Long-Term Investor Behavior Layer (detailed)

**Status:** draft — design-level detail, not yet implemented. Highest design risk in the
H-series so far: unlike H3 (~80% reused code), almost nothing here exists yet.

## Objective

H3 (once built) produces a validated monthly weight vector — but a fresh cross-sectional
reweight every month is not what "hold while the thesis is valid, sell only on deterioration or
a better opportunity, hold cash when opportunities are poor, avoid unnecessary trading" means.
That behavior needs a stateful, position-level decision layer on top of H3's scores. Nothing in
H0-H3 tracks a *held position* over time — H4 is that layer.

## Why this, not something else

The from-scratch design (see prior conversation) explicitly wanted holding period to *emerge*
from the model rather than being fixed. A naive monthly full reweight can't produce that — every
name is re-decided from scratch every period regardless of whether anything about it changed.
H4 turns "which stocks to hold this month" into "does this held position still deserve to be
held, and does a candidate outside the portfolio clear the bar to replace it net of cost."

## Design

- [ ] **Input:** H3's per-ticker blended score, per decision date (stage 4 output).
- [ ] **State:** a position tracker per ticker — currently held (bool), entry score, entry date,
      current score, score trajectory since entry.
- [ ] **Hold/trim/exit rule (hysteresis bands, not a single threshold):** a held position exits
      only when its score falls through a *lower* band than the one a new candidate must clear
      to enter — the standard hysteresis trick to prevent thrashing between two names with
      near-identical scores. Concretely: enter above `θ_buy`, exit below `θ_sell` < `θ_buy`.
- [ ] **Cost-aware trade trigger:** even inside the hysteresis bands, only execute a rotation
      (sell A, buy B) if the expected score improvement, translated into expected return via the
      same OOF IC used in H3 stage 2, exceeds the modeled round-trip cost (reuse
      `CostConfig.c_sell/c_buy = 0.03%` from `src/rl_agent/config.py`, same rate H3 uses). This
      is the literal implementation of "trade only when compelling."
- [ ] **Cash-preference rule:** define opportunity quality per period as the cross-sectional
      dispersion of H3 scores (or top-decile absolute score level) — when it's low, nothing
      clearly clears the bar, and unallocated capital should sit in cash (CDI-accruing, same
      convention as the rl_agent cash asset) rather than being forced into a full reweight.
- [ ] **No hand-picked thresholds:** `θ_buy`, `θ_sell`, and the cash-preference dispersion
      threshold are exactly the kind of constant the user has already flagged as a no-go once
      (the 40/35/15/10 factor scorer). Select all three via the same nested walk-forward grid
      search H2/H3 already use — grid over band widths, pick by pre-2024 net-of-cost OOS
      Sharpe/IR, confirm on the untouched post-2024 split. Same discipline, not a new principle.
- [ ] **Holding period is a measured output, not a target.** Report the resulting distribution
      of holding periods (median, IQR) as a diagnostic — if H4 is working, it should come out in
      months, not days, without ever having been told to.

## New modules

- [ ] `src/h_series/position_tracker.py` — stateful hold/trim/exit logic + cash-preference rule,
      operating on H3's monthly score series.
- [ ] `src/h_series/milestone_h4.py` — orchestrator: threshold grid search (walk-forward,
      pre-2024 selection only), final backtest, validation, findings doc — mirrors
      `milestone_h2.py`/`milestone_h3.py`'s structure.

## Validation

- [ ] Same pre-2024/post-2024 split, same `permutation_null()`-style check adapted to a
      stateful strategy (shuffle score trajectories, not just cross-sections — needs more care
      than H2/H3's date-shuffle, since H4's decisions depend on score *history*, not just the
      current cross-section).
- [ ] Turnover and holding-period distribution reported alongside IR — a strategy that "passes"
      on IR but trades as often as H3's naive reweight didn't actually add the behavior it was
      built for.

## Decision gate

- [ ] PASS requires H4 to beat H3's naive monthly-reweight portfolio **net of costs**, not just
      beat BOVA11/CDI. If H4 doesn't beat H3 net of costs, the honest conclusion is that the
      behavioral layer was aesthetic — it made the strategy *look* more like a long-term
      investor without making it a *better* one — and that's a legitimate, reportable outcome.

## Open risk

This stage has no prior code to lean on (H3 reuses ~80% of H1/H2/risk_portfolios.py; H4 is
closer to 20% reuse — mostly cost rates and the walk-forward/permutation-test infrastructure).
Most likely failure mode: with ~90 monthly decision dates, a 2-parameter threshold grid search
(θ_buy, θ_sell) plus a 3rd (cash threshold) may not have enough independent cycles to select
robustly — watch for this explicitly rather than discovering it post-hoc the way H2 did with its
anchor confound.
