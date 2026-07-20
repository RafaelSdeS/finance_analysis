# H4 — Long-Term Investor Behavior Layer (Phase 3: Optimize & Prepare to Deploy)

**Status:** implementation-ready design, not yet coded. Highest design risk in the H-series so
far: unlike H3 (~80% reused code), almost nothing here exists yet. Depends on H3(/H3a) having run.

## Objective

Implement the literal behavior the project's stated investment philosophy calls for — hold while
the thesis is valid, sell only on deterioration or a clearly better opportunity, hold cash when
nothing is attractive — none of which exists anywhere in H0-H3, which only ever produce a fresh
monthly cross-sectional reweight.

## Rationale

A monthly full reweight, however well-constructed, re-decides every position from scratch every
period, which cannot produce holding periods measured in years or "trade only when compelling"
behavior by construction — it's a structural gap, not a tuning problem, so no amount of improving
H3's scoring model would ever produce this behavior on its own. A new, stateful decision layer is
required. This comes after H3(/H3a) specifically because it consumes their score series as its
only input — there is nothing to make hold/sell decisions about without a score to evaluate
positions against.

## Design

- **Input:** H3(/H3a)'s per-ticker blended score, per decision date.
- **State:** a position tracker per ticker — currently held (bool), entry score, entry date,
  score trajectory since entry.
- **Hold/trim/exit rule (hysteresis bands, not a single threshold):** a held position exits only
  when its score falls through a *lower* band than the one a new candidate must clear to enter —
  the standard hysteresis trick to prevent thrashing between two names with near-identical
  scores. Concretely: enter above `θ_buy`, exit below `θ_sell` < `θ_buy`. **Both operate on the
  score's cross-sectional percentile rank within that date's eligible universe (e.g. `θ_buy=0.7`
  = top 30%), not the raw score level.** H3's score is refit per expanding fold, so its raw
  distribution isn't guaranteed stable across folds/time; a percentile-rank threshold is
  comparable across the whole backtest by construction, a fixed raw-level threshold isn't.
- **Forced exit on universe departure (not a `θ_sell` event):** if a held ticker exits the
  eligible universe entirely (delisted, drops out of the dynamic top-50 membership), it is
  liquidated regardless of its current score — a structural exit, not a threshold-triggered one.
  Reuses the same departing-ticker handling already solved for the PVM (`src/rl_agent/pvm.py`)
  rather than a new, separate mechanism.
- **Cost-aware trade trigger:** even inside the hysteresis bands, only execute a rotation (sell A,
  buy B) if the expected score improvement — translated into expected-return units via the same
  OOF fit used in H3's combination layer — exceeds the modeled round-trip cost (reuse
  `CostConfig.c_sell=c_buy=0.03%` from `src/rl_agent/config.py`, the same rate H3 uses). This is
  the literal implementation of "trade only when compelling."
- **Cash-preference rule:** define opportunity quality per period as the cross-sectional
  dispersion of H3 scores (or top-decile absolute score level) — when it's low, nothing clearly
  clears the bar, and unallocated capital sits in cash (CDI-accruing, same convention as the
  `rl_agent` cash asset) rather than being forced into a full reweight. **Precedence, explicit:**
  cash-preference only governs *unallocated* capital (freed by a `θ_sell` exit, or never
  deployed) — it never force-liquidates an existing held position on its own; only `θ_sell` or
  universe departure can exit a held position. The two rules act on disjoint capital, not in
  conflict, by construction.
- **No hand-picked thresholds:** `θ_buy`, `θ_sell`, and the cash-preference dispersion threshold
  are exactly the category of constant already ruled out once in this project (the rejected
  40/35/15/10 factor scorer). All three are selected via the same walk-forward grid search
  H2/H3 already use — grid over band widths, picked by pre-2024 net-of-cost OOS Sharpe/IR,
  confirmed once on the untouched post-2024 split. Same discipline, not a new principle.
- **Holding period is a measured output, not a target.** Report the resulting distribution of
  holding periods (median, IQR) as a diagnostic — if H4 is working, it should come out in months,
  not days, without ever having been told to.

## Implementation

- New module `src/h_series/position_tracker.py`: `PositionState` per ticker (held, entry_score,
  entry_date, score_history); `decide(t, current_scores, held_state, thresholds) -> target_weights`
  — enter if score > θ_buy and not held; exit if score < θ_sell and held; otherwise hold at
  drifted weight (no forced reweight).
- Cost-aware trigger: `OOF_IC_slope` is the univariate OLS slope of realized forward return
  regressed on OOF score, fit once per pre-2024 stitched fold output (`slope = cov(score_OOF,
  realized_return) / var(score_OOF)`) — a single fixed number carried forward, not refit per
  trigger evaluation. `expected_gain = (candidate_score − current_score) × OOF_IC_slope`; rotate
  only if `expected_gain > c_sell + c_buy` (written as the sum of both legs, not `2×c_sell`, so it
  stays correct if the two rates ever diverge).
- Cash-preference: at each rebalance date, if cross-sectional score dispersion (e.g. IQR of
  active scores) falls below a floor δ (grid-selected), shift uninvested capital to CDI-accruing
  cash rather than forcing full allocation.
- **Threshold selection, bounded grid + significance check on the winner (not a raw argmax):**
  grid search over `θ_buy ∈ {0.60, 0.70, 0.80}` (top 40/30/20% by percentile rank),
  `θ_sell ∈ {0.30, 0.40, 0.50}` (bottom 30/40/50%, always `< θ_buy`), `δ` swept over 3 values
  spanning observed pre-2024 score-dispersion deciles — 27 cells, not an unbounded search, on
  pre-2024 data only, selection criterion = net-of-cost OOS Sharpe/IR (same walk-forward
  discipline as H2's λ and H3's κ). **The selected cell must then beat a fixed, un-tuned
  reference policy (`θ_buy=0.70, θ_sell=0.30`, no cash rule) by a margin whose bootstrap CI
  excludes 0** — mirroring H3's own bar — before being trusted, precisely because a 27-cell grid
  against ~62 training rows will likely produce *some* cell that looks good by chance alone (the
  same mechanism that produced H2's γ problem). Confirmed once on post-2024 without further
  tuning.
- New module `src/h_series/milestone_h4.py`: orchestrator — threshold grid search, final
  backtest, validation, findings doc — mirrors `milestone_h2.py`'s structure.

## Expected Outcome

A portfolio that trades measurably less often than H3's naive monthly reweight, with a
holding-period distribution (median months held) that looks like a long-term investor's, and —
critically — net-of-cost performance at least as good, ideally better once turnover savings are
accounted for.

## Validation

- Backtest H4 against H3's naive reweight on the identical score series and identical walk-forward
  folds, same pre-2024/post-2024 split — the only difference between the two runs is the decision
  rule layered on top, not the underlying signal; report turnover, holding-period distribution,
  and net-of-cost IR side by side.
- **Holding-period measurement must handle right-censoring explicitly.** Positions still open at
  the end of the backtest window are not "short holds" — they're incomplete observations. Report
  the median/IQR over *closed* positions only, and separately report the count and fraction of
  positions still open at window end (a Kaplan-Meier-style censored estimate is a stretch goal,
  not required for a first pass, but the closed-only number must never be presented as if it were
  the full-sample median without that caveat attached).
- Permutation-null-style check adapted to a stateful strategy: shuffle score *trajectories*, not
  just single-date cross-sections, since H4's decisions depend on score history — needs more care
  than H2/H3's simple date-shuffle.
- Turnover and holding-period distribution reported alongside IR — a strategy that "passes" on IR
  but trades as often as H3's naive reweight didn't actually add the behavior it was built for.

## Success Criteria

H4's net-of-cost IR ≥ H3's net-of-cost IR on both splits, **and** the grid-selected threshold
combination beats the fixed reference policy with bootstrap CI on the IR delta excluding 0
(Implementation §Threshold selection), **and** median holding period over closed positions is
≥ 3 months (the naive reweight's implicit holding period is 1 month; 3 months is a deliberately
non-trivial bar, not just "more than the minimum possible"), **and** turnover is measurably lower
than H3's.

## Failure Criteria

H4 underperforms H3 net of costs — the behavioral layer cost more in missed opportunities than it
saved in turnover. A legitimate, reportable outcome: the strategy looked more like a long-term
investor's without actually being a better one — distinguishing aesthetic alignment from
substantive value, a distinction already made explicit earlier in this project's own reasoning.

## Risks & Assumptions

- Three simultaneously grid-searched thresholds (`θ_buy`, `θ_sell`, `δ`) against ~90 monthly
  observations risks the same single-path overfitting H2 had with just one parameter (γ) — the
  primary risk carried forward from the original H4 draft.
- Assumes the OOF-IC-slope translation from score units to expected-return units is stable enough
  to use as a cost-comparison yardstick — untested, needs its own sanity check before the
  cost-aware trigger can be trusted.
- The cost-aware trigger is only as good as the cost estimate it compares against. `c_sell+c_buy`
  is brokerage fees only — no bid-ask spread, no slippage. If real trading costs are higher, the
  trigger will fire *more* often than actually optimal, since it's comparing expected gain against
  an understated bar. Worth a sensitivity check (does the selected threshold combination still win
  under a higher assumed cost) before trusting the trigger's calibration, not just its existence.
- No prior code to lean on here (H3 reuses ~80% of H1/H2/`risk_portfolios.py`; H4 is closer to
  20% reuse — mostly cost rates and the walk-forward/permutation-test infrastructure).

## Next Decision Gate

**PASS** → proceed to H5 (calibration/robustness) using H4 as the standing pipeline.
**FAIL** → revert to H3's naive monthly reweight as the standing pipeline; report H4's behavioral
layer as a validated negative (not simply omitted), and reconsider whether coarser/fewer
thresholds (e.g. only `θ_sell`, no separate cash rule) would be more robust at this sample size
before retrying — narrow the design, don't abandon the goal outright on the first attempt.
