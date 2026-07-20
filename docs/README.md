# docs/ index

Chronological order of the research: **M-series → risk_mandate (R-series) → h_series (H-series)**,
plus **eiie_agent/** which spans the middle of that timeline (the RL agent whose failure the
M-series diagnoses). Each series' own docs cross-reference each other by filename.

## eiie_agent/ — Stage 3 RL agent (`src/rl_agent/`), iteration 1
- `EIIE_AGENT_PLAN.md` — design & implementation plan, faithful reproduction of Jiang/Xu/Liang (2017)
- `EIIE_DIAGNOSIS_PLAN.md` — phase-by-phase working log diagnosing the cash-attractor failure
- `EIIE_IMPROVEMENT_PLAN.md` — hypothesis-driven experiment sequence following the diagnosis
- `EIIE_INVESTIGATION_ASSESSMENT.md` — overall assessment across the investigation

## m_series/ — does daily cross-sectional alpha exist? (closed: **no**)
- `M2_FINDINGS.md`, `M2M3_STATUS.md` — entropy-floor calibration + supervised ranking probe
- `M4_DECISION_FINAL.md` — final verdict: no daily cross-sectional alpha in the top-50 universe
  using price/technical features. Successor: `risk_mandate/`.

## risk_mandate/ — R-series: structural (no-alpha) portfolio policies (closed: **FAIL vs BOVA11**)
- `RISK_MANDATE_PLAN.md`, `RISK_MANDATE_IMPL_PLAN.md` — plan + R0/R1 implementation
- `R1_FINDINGS.md` — val-split backtest
- `R2_FINDINGS.md` — lookback × rebalance sensitivity grid
- `R3_FINDINGS.md` — disjoint test-split confirmation + cost stress

## h_series/ — medium-horizon factor research (current work)
- `MEDIUM_HORIZON_RESEARCH_PLAN.md` — the greenfield program plan (H0 onward)
- `H0_FINDINGS.md`/`.json` — walk-forward spine, baselines, power analysis
- `H1_FINDINGS.md`/`.json` — sector-neutral alpha screen (**PASS** — 10/16 characteristics survive)
- `H2_FINDINGS.md`/`.json` — ridge composite + benchmark-relative construction (**FAIL** —
  anchor confound: untilted anchor alone already beats BOVA11, tilt's contribution is
  statistically indistinguishable from noise)
- `H3_PORTFOLIO_CONSTRUCTION_PLAN.md` — fixes H2's diagnosed anchor/blending defect
- `H4_LONG_TERM_BEHAVIOR_PLAN.md` — position-level hold/sell behavior layer on top of H3
- `H_SERIES_ROADMAP.md` — **start here** for what's next and in what order (also lists rejected
  designs, e.g. hand-picked factor weights, and the standing no-hardcoding rule)

## Dataset / general
- `FEATURE_SCALING_AUDIT.md` — Stage 2 (`ml_dataset.parquet`) feature/scaling design audit
- `RESEARCH_REFERENCES.md` — papers behind the project's design choices
- `TODO.md` — local, gitignored work notes (not shared/committed)
- `notes.md`, `specification.txt` — original pre-M-series design notes

## papers/
Reference PDFs (e.g. Jiang, Xu & Liang 2017).

---
**Known stale references (pre-existing, not fixed by this reorg):** `CLAUDE.md` and
`FEATURE_SCALING_AUDIT.md` still mention a few docs that were deleted outright in an earlier
cleanup commit (`PER_TICKER_SCALING_PLAN.md`, `TRAINING_SPEEDUP_PLAN.md`,
`TOP50_UNIVERSE_VALIDATION.md`, `TOP50_UNIVERSE_ML_READINESS_AUDIT.md`) — their content is gone
from the working tree (recoverable via `git log --diff-filter=D`), not just moved, so the
references weren't rewritten. Likewise `README.md`'s links to `STAGE1_DATA_COLLECTION.md` /
`STAGE2_DATASET_BUILD.md` / `STAGE3_ML_AGENT.md` / `ML_AGENT_ROADMAP.md` don't resolve to
anything in this branch.
