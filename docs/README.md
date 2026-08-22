# docs/ index

One row per doc, with a status — verified against each file's own content, not guessed from
its filename (a first-pass version of this table existed only as a draft inside
`DATA_LAYER_ORGANIZATION_PLAN.md` §O6 and got 2 statuses wrong; corrected here).

**live** — being actively worked, has genuinely open items. **done** — implemented, kept for
rationale. **historical record** — a dated snapshot (audit/proposal), not meant to be re-opened.
**backlog** — parked ideas, not scheduled.

| doc | status | what it's for |
|---|---|---|
| `DATA_LAYER_ORGANIZATION_PLAN.md` | done | Stage 1 file/module reorganization; closed 2026-08-21, all items done or cut |
| `DATA_LAYER_CORRECTNESS_PLAN.md` | done | Stage 1/2 bug-fix pass; closed out, 2 non-blocking notes left |
| `DATA_LAYER_FOLLOWUP_FINDINGS.md` | done | findings surfaced while verifying the correctness plan's §1 migration |
| `DATA_INTEGRITY_TEST_PLAN.md` | live | data-quality test coverage; `--market us` golden gate still open |
| `US_EQUITIES_EXPANSION_PLAN.md` | live | US collection buildout; Phase 6 (full-universe scale-up) in progress |
| `US_DATASET_BUILD_PLAN.md` | live | Stage 2 US dataset build; full 3,134-ticker run not yet completed end-to-end |
| `US_COLLECTOR_FIX_PLAN.md` | live | consolidated US collector bug list; most items closed, a few (universe-scope, staleness policy) still open |
| `PORTFOLIO_IMPROVEMENT_PLAN.md` | live | active Stage 3 research log — **read its STOP banner first** |
| `BOLSAI_EXIT_PLAN.md` | done | moved BR collection off the paid BolsAI default onto free sources |
| `DATA_COLLECTION_REORGANIZATION_PLAN.md` | done | `src/data_collection/` market-namespacing (`br/`/`us/`/`cvm/`/`sec/`) |
| `PORTFOLIO_ARCHITECTURE_PROPOSAL.md` | historical record | Stage 3 "what to build" — pre-implementation design; superseded by the actual code + the improvement plan's live findings |
| `PORTFOLIO_IMPLEMENTATION_PLAN.md` | historical record | Stage 3 "how it was built," grounded in the tree as of 2026-07-24 |
| `US_DATASET_AUDIT_2026-08-01.md` | historical record | dated data-quality audit of the US dataset as it stood 2026-08-01 |
| `SURVIVORSHIP_BIAS_AUDIT_2026-08-15.md` | historical record | dated survivorship-bias audit + remediation record |
| `FEATURE_IDEAS.md` | backlog | candidate Stage 2 features, parked until Stage 3 modeling resumes |

CLAUDE.md cites all of these inline at the point they're relevant; this table is only for
"what exists and is it still live" at a glance. Re-verify status here whenever a doc's own
checkboxes change materially — this is a snapshot, not a generated report.
