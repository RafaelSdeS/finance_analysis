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

- [ ] **Give US the same entry-point shape as BR.** `us/pipeline.py` with `--mode`/`--tickers`/
      `--dry-run` over the existing `run_*` stage functions. It already has the stages; it's missing
      the CLI. Biggest maintainability win on the US side and mostly wiring.

- [ ] **~~Consider a US `stats.py` analogue.~~ CUT [rev 2026-08-20].** Speculative work for
      symmetry's sake — nobody has asked for a US audit report, and building one because BR has one
      is how a codebase grows a second thing to maintain for free. Add it when someone actually
      wants the numbers, not before.

---

## §O3 — Split and dedupe inside `src/data_collection/`

- [ ] **Split `yf_collectors.py` (1,078 lines after the test extraction) into a `yf/` package**:
      `yf/_common.py` (~330) · `yf/prices.py` (~415) · `yf/fundamentals.py` (~180) ·
      `yf/dividends.py` (~150). Already separated by comment banners, so it is close to a pure move.

      ⚠️ **[rev 2026-08-20] Justify this on concern count, not line count.** The original text
      argued from "1,078 lines" three paragraphs after declaring *"Not splitting: `features.py`
      (822), `build_us_dataset.py` (540), `sec/*` … Size alone isn't a reason."* Those cannot both
      be the principle, and the line-count framing invites the next person to split `features.py`
      too. The defensible argument: **`yf_collectors.py` holds four distinct collectors** (common,
      prices, fundamentals, dividends) while `features.py` does *one thing* at 822 lines. Concern
      count justifies the split; size does not.

      ⚠️ **Reconsider the re-export shim.** Once §O1 has moved `compute_ratios` and
      `FUND_FULL_COLS`, nothing outside `yf/` needs the old module names — so a shim would mostly
      exist to avoid touching import sites that *should* be touched. Prefer updating the imports and
      deleting `yf_collectors.py` outright. Keep a shim only if a real external caller turns up.

- [ ] 🟡 **Dedupe the per-ticker collector skeleton.** `collect_prices`, `collect_fundamentals`,
      `collect_dividends` in `br/collectors.py` share one ~50-line body (checkpoint → skip-list →
      loop → `should_skip` → `is_complete` → fetch → empty-check → `_merge_save` → checkpoint →
      `clear_skip`; `except` → `mark_skip`; `finally` → sleep), differing only in directory,
      columns, fetch call, date column, validator, and checkpoint key. One `for_each_ticker()`
      helper: ~150 → ~60 lines of the module's current 407.

      **Scoped to those three.** The yfinance three share the shape but add threading,
      `_prices_fetch_start`, junction reconciliation, and the failure-abort — one helper for all six
      is the over-abstraction the module docstring already warns about, on the hot path.

      **[rev 2026-08-20] Lowest-confidence item in this plan.** ~90 lines saved in exchange for one
      new abstraction, on a hot path, in a module whose own docstring cautions against exactly this.
      The 3-of-6 scoping is right. **Cut this first if the plan needs trimming.**

- [ ] **Move `collect_macro` → `br/macro.py`.** It's BCB SGS: free, keyless, runs in every mode —
      the only live-by-default function in a module whose docstring says BolsAI and whose every
      other function needs a paid key. Confirmed at `br/collectors.py:87`.

- [ ] **Wire the orphaned `validate_sectors` to the live path.** It was used only by the deleted
      `collect_sectors()`; meanwhile `cvm/sectors.py::build_sectors()` writes `sectors.parquet`
      **unvalidated** and emits the exact `(name, count)` schema it checks. Confirmed orphaned:
      `validate.py:246`, zero call sites. **4 lines**, and it closes a real gap rather than a
      tidiness preference.

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

- [ ] **Name the convention for new tests, and correct the docs.** CLAUDE.md says "No test
      framework", but `pytest==8.3.4` is pinned and is the runner in 15 files. Pick one shape for
      *new* tests and write it down.

      **Recommendation: `pytest.main([__file__])`.** It's already the largest single convention with
      a real runner behind it, and it makes the AST drift-guard unnecessary for any file adopting
      it.

      **Do not migrate the existing 32.** A 32-file diff buys consistency that the AST guard in
      `DATA_LAYER_CORRECTNESS_PLAN.md` §5 already delivers as a 15-line check. Convention for new
      files, guard for old ones.

---

## §O6 — Documentation structure

Fact-level doc drift (wrong ticker counts, missing conventions) lives in the correctness plan §7.
This is about *finding* things.

- [ ] 🟡 **`docs/` has no index — 13 files, 5,413 lines, and no way to tell live from finished
      without opening each.** The biggest *navigability* gap in the repo, and the best
      return-per-line in this plan. CLAUDE.md cites these docs inline, scattered across five
      different sections, so nothing answers "what's in `docs/` and which of it is still true?"

      Add a ~25-line `docs/README.md`: one row per doc, with a **status** — the load-bearing
      column, since the set is a mix of live and historical with nothing marking which:

      | status | docs |
      |---|---|
      | **live** | `DATA_LAYER_CORRECTNESS_PLAN` · `DATA_LAYER_ORGANIZATION_PLAN` · `PORTFOLIO_IMPROVEMENT_PLAN` (has a STOP banner) · `US_EQUITIES_EXPANSION_PLAN` (Phase 6 in progress) |
      | **done** | `BOLSAI_EXIT_PLAN` · `DATA_COLLECTION_REORGANIZATION_PLAN` · `US_DATASET_BUILD_PLAN` · `US_COLLECTOR_FIX_PLAN` |
      | **historical record** | `US_DATASET_AUDIT_2026-08-01` · `SURVIVORSHIP_BIAS_AUDIT_2026-08-15` · `PORTFOLIO_IMPLEMENTATION_PLAN` |

      **Verify each status before writing it** — this table is a first pass from filenames plus
      CLAUDE.md's own references, not an audit of the contents.

- [ ] **`sec/tenq.py` is missing from CLAUDE.md's `sec/` module table.** The table lists
      `http`/`universe`/`crosswalk`/`companyfacts`/`fds`/`selected_financial_data`/`fundamentals`
      but not `tenq.py`, which is a live `compute_ratios` consumer (`:311`).

- [ ] **Update CLAUDE.md's `data_collection` module table** after §O1/§O3 land — new `ratios.py`,
      new `yf/` package, `br/macro.py`, and `yf_collectors.py` either gone or reduced to a shim.

- [ ] `/graphify . --update` after the moves, so the knowledge graph doesn't point at dead paths.

---

## Sequencing

```
EARLY -- before the correctness plan's S1 normalization:
  O1. compute_ratios + FUND_FULL_COLS -> data_collection/ratios.py

LATER -- only once DATA_LAYER_CORRECTNESS_PLAN.md is through its rebuild
         and the suite is green:
  O2. us/pipeline.py CLI
  O3. yf/ package split (imports updated, shim only if needed)
      br/macro.py move
      validate_sectors wiring          <- 4 lines, do it whenever
      for_each_ticker dedupe           <- lowest confidence, cut first if trimming
  O5. name the test convention
  O6. docs/README.md, CLAUDE.md tables, graphify rebuild
```

**Why the ordering.** §O1 has to precede the signature change it would otherwise collide with.
Everything else is pure refactoring with no data effect, so it goes behind a green suite: if
something breaks after this plan runs, the cause is unambiguously a move, not a data fix.

**Ranked by value if you only do some of it:** §O1 (real dependency fix) → §O6's `docs/README.md`
(navigability, ~25 lines) → `validate_sectors` (4 lines, closes a gap) → `br/macro.py` (fixes a
miscategorization) → §O2's US CLI → §O3's split → `for_each_ticker`.
