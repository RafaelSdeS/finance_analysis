# Data Layer Organization Plan

**Split out of the former `DATA_LAYER_HARDENING_PLAN.md` on 2026-08-20** (that file has been
removed — never committed, contents fully carried across), alongside
**`DATA_LAYER_CORRECTNESS_PLAN.md`**. It mixed bug-fixing with refactoring. They have different
risk profiles and belong in different passes.

**Goal:** the collectors are maintainable and navigable — BR and US readable the same way, and
nothing importing vendor-neutral code from a vendor-named module.

**This plan changes no data.** Every item is a move, a rename, a deletion, or a doc. That is exactly
why it runs **behind a green correctness suite**, not alongside correctness fixes — a refactor and a
data fix failing at the same time is two debugging problems wearing one trenchcoat.

> **Scope rule, stated because it was previously an unstated omission:** this covers
> `src/data_collection/` only. Stage 2 and Stage 3 are deliberately untouched — see §O4 for the
> measurement behind that.

**One exception to the "runs last" rule:** §O1 runs early, before the correctness plan's §1
normalization. Reason given in that section.

---

## §O1 — 🔴 Lift the vendor-neutral code out of `yf_collectors.py` (**do this first**)

The highest value-per-line item in this plan, and the only one that isn't cosmetic.

Six modules import from `..yf_collectors`, and **none of them want yfinance**:

| importer | imports | why it is not yfinance code |
|---|---|---|
| `sec/companyfacts.py:29`, `sec/fds.py:46`, `sec/fundamentals.py:18`, `sec/selected_financial_data.py:34`, `sec/tenq.py:50` | `compute_ratios` | pure algebra over a dict of line items — that is *why* SEC can pass `unit_scale=1` and reuse it |
| `br/collectors.py:27` | `FUND_FULL_COLS` | the on-disk fundamentals schema, shared by every source |

So the **US SEC pipeline** and the **BolsAI BR collector** both depend on a module named after a
yfinance collector.

**The original plan made this worse.** It proposed splitting into `yf/*` while keeping
`yf_collectors.py` as a re-export shim "so no import site changes" — which preserves the wrong
dependency *and* adds a layer on top of it. After that split, `sec/companyfacts.py` would import a
yfinance shim, which forwards into a yfinance package, to reach vendor-neutral algebra.

- [x] ✅ **DONE 2026-08-20 — moved `compute_ratios` + `FUND_FULL_COLS` to `src/data_collection/ratios.py`.**
      `yf_collectors.py` now imports both from `.ratios` (still uses them internally — the module
      keeping the name isn't the same as the module keeping the code). All 6 downstream import
      sites repointed to `..ratios`: `sec/fundamentals.py`, `sec/fds.py`, `sec/tenq.py`,
      `sec/companyfacts.py`, `sec/selected_financial_data.py`, `br/collectors.py`
      (`FUND_FULL_COLS`). Two test files also fixed (`tests/data_collection/test_yf_collectors.py`,
      `test_ratios_no_inf.py`, `test_skip_existing.py`) rather than left on the old path — a
      re-export shim was explicitly rejected above, so nothing should still import vendor-neutral
      code through `yf_collectors`. `tests/run_all.py --group fast`: 55/55 after the move.

**Sequence it before the correctness plan's §1**, which edits `compute_ratios`' signature
(`unit_scale`). Moving a function is a cleaner diff than moving one that just changed, and the six
importers are then already pointing at their final home. This is the only item here that runs early.

---

## §O2 — Make BR and US readable the same way

| concern | BR | US | |
|---|---|---|---|
| vendor fundamentals pkg | `cvm/` | `sec/` | ✓ symmetric |
| prices/dividends | `yf_collectors.py` (shared) | same | ✓ shared, correct |
| orchestration CLI | `br/pipeline.py` (224 ln, `--mode`/`--tickers`/`--dry-run`) | `us/run_us_full_scale.py` (103 ln, **no argparse**) | ✗ asymmetric |
| vendor-package CLI | `br/cvm_statements.py` (`--step`) | none for `sec/` | ✗ asymmetric |
| post-collection audit | `br/stats.py` (69 ln) | none | ✗ asymmetric |

**Table re-verified 2026-08-20, all rows hold.** argparse appears in **four** `br/` files
(`pipeline.py`, `cvm_statements.py`, `collect_delisted.py`, `stats.py`) and **zero** `us/` files.

- [x] ✅ **DONE 2026-08-21 — `git mv run_us_full_scale.py us/pipeline.py`, added argparse.**
      `--mode` (checkpoint namespace, default `us_full_scale_v2`), `--tickers` (override the
      universe for prices/dividends/fundamentals/company_info — threaded into all four `run_*`
      signatures, which previously always called `_all_tickers()`/globbed the priced-tickers
      set internally with no override), `--steps` (replaces the old bare `sys.argv[1:]`), and
      `--dry-run`. Reused the existing six `run_*` functions and `STEPS` dict verbatim — did not
      replicate BR's `DATA_SOURCE` multi-vendor dispatch or ATIVO-ticker filtering, neither of
      which has a US equivalent (US prices/dividends are pure yfinance, fundamentals pure SEC).
      Smoke-tested with `--dry-run --tickers AAPL MSFT`; `tests/run_all.py --group fast`: 55/55.

- [ ] **~~Consider a US `stats.py` analogue.~~ CUT [rev 2026-08-20].** Speculative work for
      symmetry's sake — nobody has asked for a US audit report, and building one because BR has one
      is how a codebase grows a second thing to maintain for free. Add it when someone actually
      wants the numbers, not before.

---

## §O3 — Split and dedupe inside `src/data_collection/`

- [x] ✅ **DONE 2026-08-21 — split `yf_collectors.py` into a `yf/` package.** `yf/_common.py`
      (290 ln) · `yf/prices.py` (455 ln) · `yf/fundamentals.py` (130 ln) · `yf/dividends.py`
      (173 ln, includes `collect_splits_yf` — one ~50-line function, not enough of a distinct
      concern for a 5th file). One correction to the plan's own file map: `_extract_dividends`
      and `_last_completed_trading_day` physically sat under the old module's "prices" banner
      but are called from **both** `collect_prices_yf` and `collect_dividends_yf` — moved to
      `_common.py` instead of `prices.py`, or `dividends.py` would import across its sibling for
      no reason. Deleted `yf_collectors.py` outright (`git rm -f`) — no shim, per the plan's own
      call below; repointed every import site: `br/pipeline.py`, `refresh.py`, `us/pipeline.py`,
      `one_off/backfill_known_gaps.py`, and 7 test files (`test_pipeline_dispatch.py`'s
      `mock.patch.object(pipeline.yf_collectors, ...)` calls split three ways to
      `yf_prices`/`yf_fundamentals`/`yf_dividends`; `test_yf_collectors.py` kept its filename
      but its import block now pulls from `yf._common`/`yf.prices`/`yf.dividends`). Prose
      references to the old filename fixed in 5 more files (`checkpoint.py`, `cvm/ratios.py`,
      `br/collectors.py`, `features.py`, `validate_us_vs_vendor.py`) — one of them
      (`features.py`'s `yf_collectors.FLAT_RUN_PADDING`) was already wrong before this move;
      `FLAT_RUN_PADDING` lives in `one_off/backfill_known_gaps.py`, never did in `yf_collectors.py`.
      `ruff check src/ tests/`: clean. `tests/run_all.py --group fast`: 55/55.

- [x] 🟡 **CUT 2026-08-21 — `for_each_ticker` dedupe.** Re-read `collectors.py`'s own docstring
      before touching it: *"No abstract base class: the four sources have genuinely different
      fetch logic and share nothing worth abstracting."* This item was already the plan's own
      lowest-confidence entry ("cut this first if trimming") for exactly that reason — ~90 lines
      saved in exchange for one new abstraction, on a hot path, contradicting the module's
      explicit design stance. Not done. Revisit only if a real second/third instance of the
      shared shape shows up and the abstraction pays for itself twice over, not once.

- [x] ✅ **DONE 2026-08-21 — moved `collect_macro` → `br/macro.py`.** Verbatim move (imports,
      docstring, body unchanged). Repointed 3 call sites: `br/pipeline.py` (`collectors.collect_macro`
      → `macro.collect_macro`), `refresh.py` (`br_collectors.collect_macro` → `br_macro.collect_macro`),
      and `tests/data_collection/test_macro_bare_object.py` (imports + all `collectors.*` mock
      targets repointed to `macro.*`). `collectors.py` lost its now-unused `httpx`/`datetime`... no
      — `datetime` still used by `collect_prices`'s dedup window; only `httpx` (macro-only) dropped.
      `ruff check`: clean. `tests/run_all.py --group fast`: 55/55.

- [x] ✅ **DONE 2026-08-21 — wired `validate_sectors` into `build_sectors()`.** 4 lines:
      `cvm/sectors.py` now imports `validate` and calls `validate.validate_sectors(df)` before
      writing `sectors.parquet`, logging and skipping the write on failure (same
      `if not vr.passed: log.error(...); return` shape used everywhere else in this codebase —
      `build_sectors()` writes directly rather than through `_merge_save`, since `sectors.parquet`
      is a `(name, count)` reference table with no date column to dedupe on, so `_merge_save`
      doesn't fit here).

**Not splitting:** `features.py` (822 after the applied §3 edits), `build_us_dataset.py` (540),
`sec/*` (largest 667). Large but single-concern and already well-separated.

---

## §O4 — Why Stage 2 and Stage 3 are deliberately untouched

Measured across all 79 files / 14,712 lines in `src/`:

- **Stage 2 (`build_dataset/`, 12 files)** is already organized the way §O2/§O3 are *trying to get*
  `data_collection` to be: one file per pipeline stage, each named for its concern
  (`loaders`/`merge`/`repair`/`continuity`/`quality_filters`/`clean`/`manifest`), with `paths.py`
  as the single place that reaches across into `data_collection`. Largest files are `features.py`
  (822) and `cagr_handler.py` (440) — big but single-concern, same reason `sec/*` isn't split.
- **Stage 3 (`portfolio/`)** is already one file per concern with thin `run_*.py` drivers.
- **The cross-stage import surface is clean**: Stage 2 reaches into Stage 1 through `paths.py` plus
  three explicit path-constant imports (`build_us_dataset.py`, `terminal_events.py`); Stage 3
  reaches into Stage 2 only through `build_dataset.paths`. **No layering violations found.**

**Conclusion: no reorganization work is warranted outside `data_collection/`.** The asymmetry this
plan exists to fix is real and specific to Stage 1 — the only stage that grew two market
implementations at different times.

---

## §O5 — Test conventions

Three conventions coexist. Recounted 2026-08-20 by AST over the 50 files that actually define a
`test_*` function: **15 `pytest.main` / 32 hand-listed in `__main__` / 3 other**. (An earlier
"15 / 26 / 36" summed to 77 against 50 real files — the third bucket had counted every file with a
`main()`, including non-test helpers.)

- [x] ✅ **DONE 2026-08-21 — named the convention in CLAUDE.md's Tests section, corrected the
      "No test framework" claim.** New tests: `pytest.main([__file__])` with real `test_*`
      functions + bare `assert` — already the largest single convention (15/50 files) with a real
      runner behind it. Did not migrate the other 32 hand-listed `__main__` files — the AST
      guard (`main_block_drift()` in `tests/run_all.py`) already delivers the consistency a
      32-file diff would buy, at 15 lines.

---

## §O6 — Documentation structure

Fact-level doc drift (wrong ticker counts, missing conventions) lives in the correctness plan §7.
This is about *finding* things.

- [x] ✅ **DONE 2026-08-21 — added `docs/README.md`.** One row per doc, status column, verified
      against each file's actual content (not filenames) rather than copying the draft table
      above verbatim — that draft had 2 statuses wrong: `US_DATASET_BUILD_PLAN` and
      `US_COLLECTOR_FIX_PLAN` both still have real open items (confirmed by reading their tails:
      the former's Phase C full-scale run "still NOT been run to full 3,134-ticker completion,"
      matching CLAUDE.md's own "full-universe scale-up in progress"; the latter has an open
      appendix item plus one box pending a run-confirmation), so both moved from **done** to
      **live**. Also added 2 files the draft predated: `DATA_INTEGRITY_TEST_PLAN.md` (**live** —
      `--market us` golden gate still open) and `FEATURE_IDEAS.md` (new **backlog** status — parked
      ideas, not scheduled, doesn't fit live/done/historical). `DATA_LAYER_CORRECTNESS_PLAN.md`
      and `DATA_LAYER_FOLLOWUP_FINDINGS.md` moved **live → done**, both closed out by commit since
      the draft was written. `PORTFOLIO_ARCHITECTURE_PROPOSAL.md` added as **historical record**
      (the draft omitted it) — superseded by the actual Stage 3 code plus the live improvement-plan
      research log.

- [x] ✅ **DONE 2026-08-21 — added the missing `sec/tenq.py` row to CLAUDE.md's `sec/` module
      table**, described accurately (10-Q inline-HTML tier, real Q1–Q3 resolution for the
      2001–2006 window, Q4 derived cross-tier against item6's annual total) rather than guessed
      from the filename.

- [x] ✅ **DONE 2026-08-21 — updated CLAUDE.md's `data_collection` module table**: added
      `ratios.py`, replaced the `yf_collectors.py` row with `yf/`, added `br/macro.py`, replaced
      `us/run_us_full_scale.py` with `us/pipeline.py` (new CLI description). No shim exists to
      describe — `yf_collectors.py` is gone outright, per §O3 above.

- [x] ✅ **DONE 2026-08-21 — ran `/graphify . --update`.** 173 changed files / 58 deletions
      caught up (most of that backlog predates this session — the graph hadn't been rebuilt since
      2026-07-16), including this session's moves. AST (code) extraction ran to completion; the
      docs/README.md-triggered semantic-concept pass over the 21 changed doc files was interrupted
      (subagent killed mid-run) and skipped rather than retried blind — code nodes/edges (the part
      that actually matters for "not pointing at dead paths") are current, doc-level concept nodes
      for those 21 files are not freshly extracted this run. Re-run `/graphify . --update` to pick
      that up whenever wanted; nothing is broken by leaving it.

---

## Sequencing

```
EARLY -- before the correctness plan's S1 normalization:
  O1. compute_ratios + FUND_FULL_COLS -> data_collection/ratios.py           [DONE 2026-08-20]

LATER -- only once DATA_LAYER_CORRECTNESS_PLAN.md is through its rebuild
         and the suite is green:
  O2. us/pipeline.py CLI                                                    [DONE 2026-08-21]
  O3. yf/ package split (imports updated, no shim needed)                   [DONE 2026-08-21]
      br/macro.py move                                                      [DONE 2026-08-21]
      validate_sectors wiring                                               [DONE 2026-08-21]
      for_each_ticker dedupe                                                [CUT 2026-08-21]
  O5. name the test convention                                              [DONE 2026-08-21]
  O6. docs/README.md, CLAUDE.md tables, graphify rebuild                    [DONE 2026-08-21]
```

**Why the ordering.** §O1 has to precede the signature change it would otherwise collide with.
Everything else is pure refactoring with no data effect, so it goes behind a green suite: if
something breaks after this plan runs, the cause is unambiguously a move, not a data fix.

**Status: closed 2026-08-21.** Every item is done or explicitly cut with rationale (see §O3);
nothing left open. `tests/run_all.py --group fast`: 55/55. `ruff check src/ tests/`: clean.
