# H6 — Final Comprehensive Report (Phase 3: Optimize & Prepare to Deploy)

**Status:** draft design. Depends on H5.

## Objective

Produce one clear, complete comparison of the final pipeline (H3+H4, calibrated per H5) against
every relevant baseline, replacing the current scattered text-only findings docs.

## Rationale

Every prior stage (H0-H5) writes its own narrow `*_FINDINGS.md`; there is no single artifact
showing the whole picture side by side. Before considering deployment, a decision needs one
report answering "does this actually beat doing nothing, or doing something simpler," which the
current fragmented outputs don't provide directly.

## Design

Adapt `src/rl_agent/plots.py::write_report()` (self-contained HTML, plotly embedded once) to
consume H3+H4's backtest output instead of an EIIE `BacktestResult`, alongside every established
baseline, split into two groups shown separately, not pooled into one table:

- **Same-window group** (directly comparable): UBAH, UCRP, Best-Stock (hindsight, N=30 seeds for
  Random — reported as a range/band, not a single point), Constant-Cash, BOVA11, CDI, and the old
  H2 composite, all re-evaluated over H3+H4's own walk-forward window (H2's original run already
  covers this span, so no rerun is needed — just the same window's numbers, pulled together).
- **Different-window reference group** (explicitly labeled as such, never placed in the same
  ranked table): the old EIIE agent's result, evaluated over its own 2021-11→2024-03 val split —
  a genuinely different period than H3+H4's. Shown for historical context only, with an explicit
  caption stating the windows differ and the numbers are not directly comparable.

## Implementation

Wrap H3+H4's final backtest results in the same `BacktestResult`-shaped structure `plots.py`
expects (or extend `write_report()` to accept a slightly more general input); reuse `metrics.py`'s
full suite (Sharpe/Sortino/Calmar/VaR/CVaR/turnover/cost drag/IR vs. BOVA11) and
`block_bootstrap_ci` for every series in the comparison, not just the primary one.

## Expected Outcome

One HTML report a reader unfamiliar with the whole project's history could open and immediately
see whether this research produced something that beats simply buying and holding the index or
sitting in cash.

## Validation

**Programmatic consistency check, not manual review as the primary gate:** a script parses every
numeric claim embedded in the report and asserts exact equality against the corresponding value
in its source `*_FINDINGS.json` (H0-H5) — a mismatch fails the build, the same way a broken test
fails CI. Manual review is a secondary pass (does the narrative read clearly, is anything
misleadingly framed), not the mechanism catching numeric drift. Each embedded number is also
tagged with the git commit / dataset version it was sourced from (same convention as
`run_manifest.json` elsewhere in this project), so a later upstream rerun can't silently make the
report stale without it being detectable.

## Success Criteria

Report renders correctly; includes every baseline from both groups (§Design), correctly
partitioned; the programmatic consistency check passes with zero mismatches; the baseline list
itself was fixed *before* H3+H4's final numbers were seen (no post-hoc dropping of a baseline
that happens to make the result look worse).

## Failure Criteria

N/A in the pass/fail sense this program has otherwise used — this stage doesn't generate a new
verdict, it presents one. Its failure mode is being misleading (cherry-picked baselines,
inconsistent numbers), not the underlying strategy failing.

## Risks & Assumptions

Assumes `plots.py`'s report format generalizes cleanly to a non-EIIE backtest shape — likely needs
real adaptation work, not just a data swap, since the current report assumes EIIE-specific fields
(e.g. allocation evolution driven by the CNN's output) that don't exist for H3+H4's simpler
weight-vector output.

## Next Decision Gate

**Report shows a clear net-of-cost edge over every baseline** → proceed to H7 (deployment
planning).
**Report shows no edge, or an edge that doesn't survive H5's robustness bar** → do not proceed to
deployment; the report itself becomes the final, honest deliverable of this research program.
