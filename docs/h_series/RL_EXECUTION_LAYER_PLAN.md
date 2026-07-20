# RL-for-Execution (Deferred, Unscheduled)

**Status:** deliberately undesigned. Listed so it isn't forgotten, not because it's planned.

## Objective

If ever pursued: reduce execution-level cost (order-timing/slippage-like effects) via smarter
scheduling of trades given already-decided target weights — narrowly scoped to *execution*, not
stock selection.

## Rationale

Ladder discipline: don't reach for a learned component until the simpler one (H4's rule-based
cost-aware trigger) is shown, in practice, to be insufficient. Revisited only if H4 demonstrably
underperforms specifically due to execution-timing effects, not general model weakness — a
narrow, falsifiable trigger condition, not a vague "if we want to try RL again."

## Design / Implementation

Intentionally undesigned. If pursued, would reuse RL machinery (e.g. the existing
`src/rl_agent/environment.py` cost-solver machinery) scoped only to scheduling execution of
already-decided target weights over a short window — never to decide *what* to hold.

## Risks & Assumptions

Re-approaches the exact failure mode that broke the original EIIE agent — giving a learner more
freedom than the data can responsibly support — unless kept very narrowly scoped to execution
timing alone. This is the primary reason it's deferred rather than attempted alongside H4.

## Next Decision Gate

**Critique of the previous version of this section: "revisit if H4 demonstrably underperforms due
to execution-timing effects specifically" had no way to actually tell the two apart — an
unfalsifiable trigger condition dressed as a specific one.** Concrete diagnostic instead: compare
H4's backtest return series under its current idealized fill assumption (trades execute exactly
at the modeled close, per the standing convention throughout this project) against the same
series with a slippage perturbation applied (e.g. trades fill at a fixed number of basis points
worse than the modeled close, calibrated to a plausible bid-ask estimate for top-50 B3 liquidity).
If the gap between the two is large **and concentrated specifically around H4's own trade-trigger
dates** (not a general, diffuse return shortfall spread evenly across the whole backtest), that's
the actual signature of an execution-timing problem — as opposed to the scoring model simply
picking weaker positions, which would show up as a diffuse gap, not a trigger-date-concentrated
one. Only *that* specific pattern justifies opening this file for real design work; a diffuse gap
means the problem is upstream (H3's scoring), not execution, and this stage stays closed.
