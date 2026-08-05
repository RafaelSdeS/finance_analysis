# src/data_collection/ + data/ reorganization plan (2026-08-05)

**Status: implemented 2026-08-05.** All checkboxes below were applied in one pass
(§15 first, then §6-§14, then §1-§5, per the recommended order) and verified: all
61 tests green (`tests/run_all.py --group all`), `ruff check .` clean. Left as a
file per house style so the rationale survives across sessions; see closing
checklist for the short version.

## 0. What's actually disorganized (and what isn't)

`src/data_collection/` has three already-good subpackages (`cvm/` for CVM/Brazil,
`sec/` for SEC/US, both cleanly namespaced) but the 15 files one level up are flat and
unlabeled: BR collectors, US collectors, shared infra, and one-off incident-fix scripts
all sit side by side with no grouping. Concretely:

| Category | Files |
|---|---|
| Shared infra (market-agnostic) | `config.py`, `client.py`, `checkpoint.py`, `validate.py`, `yf_collectors.py` |
| BR-specific | `collectors.py` (BolsAI), `pipeline.py` (BR CLI), `collect_delisted.py`, `cvm_statements.py` |
| US-specific | `fred_collectors.py`, `run_us_full_scale.py` |
| One-off incident scripts (never imported elsewhere — confirmed via grep) | `backfill_known_gaps.py`, `fix_mrfg3_adj_close.py` |
| BR-only despite generic name | `stats.py` (reads only `MACRO_DIR`/`COMPANY_DIR`/`PRICES_DIR`/`FUND_DIR`, no `US_*` dirs — a real coverage gap, out of scope here) |

**A real bug, not just a style issue:** `_merge_save`/`_chunk_dates` (the shared
idempotent-write/date-windowing helpers every collector uses) are defined *inside*
`collectors.py` — a BR/BolsAI-specific file. `fred_collectors.py` (US) and
`cvm/ratios.py` (BR) both import `_merge_save` by reaching into `collectors.py`, so a
US module currently has a hard dependency on a Brazil-specific one. That's the
structure causing the coupling, not a naming accident — worth fixing regardless of
whether the rest of this plan happens.

`data/` (see §6) is **not** actually disorganized — it's asymmetric (BR flat,
US nested under `us/`) but that's a documented, consistent convention already used
independently by both `data_collection/config.py` and `build_dataset/paths.py`, and
spelled out in `CLAUDE.md`'s "Data on Disk" section. Recommend leaving it alone.

---

## 1. Extract `storage.py` (fixes the real coupling bug)

Move `_merge_save` and `_chunk_dates` out of `collectors.py` into a new
`src/data_collection/storage.py` (parquet append/dedup/validate/write + date-window
chunking — generic to any collector, BolsAI/FRED/CVM/yfinance alike).

- [x] Create `storage.py` with `_merge_save`, `_chunk_dates` (moved, not copied)
- [x] `collectors.py`: import both from `storage` instead of defining them
- [x] `fred_collectors.py`: `from .collectors import _merge_save` → `from .storage import _merge_save` (drops the BR dependency entirely)
- [x] `cvm/ratios.py`: `from .. import collectors, config, validate` → `from .. import config, storage, validate`; call site `collectors._merge_save` → `storage._merge_save`
- [x] `yf_collectors.py`: same swap (`from .collectors import _merge_save` → `from .storage import _merge_save`)
- [x] Test files importing `_merge_save`/`_chunk_dates` off `collectors`: `test_merge_save_new_rows_only.py`, `test_chunk_dates_leap_year.py` — repoint to `storage`

## 2. Group loose top-level files into `br/`, `us/`, `one_off/`

`cvm/` and `sec/` stay exactly where they are — both names are already
self-evidently market-specific (CVM = Brazilian regulator, SEC = US regulator), and
leaving them in place means zero changes to any file *inside* those two directories
(only the one or two files that import *from* them gain a `..`). Moving them too
would touch every file inside both subpackages for no real clarity gain — skipped.

Target:

```
src/data_collection/
  __init__.py
  config.py            # shared
  client.py            # shared
  checkpoint.py        # shared
  validate.py           # shared
  storage.py            # shared (new, see §1)
  yf_collectors.py      # shared (parametrized: BR update mode + US backfill both use it)
  cvm/                   # unchanged, unmoved
  sec/                   # unchanged, unmoved
  br/
    __init__.py
    collectors.py        # was top-level collectors.py
    pipeline.py           # was top-level pipeline.py
    collect_delisted.py
    cvm_statements.py
    stats.py              # was top-level stats.py (content is BR-only today)
  us/
    __init__.py
    fred_collectors.py
    run_us_full_scale.py
  one_off/
    __init__.py
    backfill_known_gaps.py
    fix_mrfg3_adj_close.py
```

- [x] `git mv collectors.py br/collectors.py`
- [x] `git mv pipeline.py br/pipeline.py`
- [x] `git mv collect_delisted.py br/collect_delisted.py`
- [x] `git mv cvm_statements.py br/cvm_statements.py`
- [x] `git mv stats.py br/stats.py`
- [x] `git mv fred_collectors.py us/fred_collectors.py`
- [x] `git mv run_us_full_scale.py us/run_us_full_scale.py`
- [x] `git mv backfill_known_gaps.py one_off/backfill_known_gaps.py`
- [x] `git mv fix_mrfg3_adj_close.py one_off/fix_mrfg3_adj_close.py`
- [x] Add `br/__init__.py`, `us/__init__.py`, `one_off/__init__.py` (empty, matches `cvm/`/`sec/` convention)

## 3. Fix internal relative imports (one level deeper for moved files)

Every moved file's imports of shared modules (`config`, `client`, `checkpoint`,
`validate`, `storage`, `yf_collectors`) go from `from . import X` to `from .. import X`.
Cross-references into the still-top-level `cvm`/`sec` packages gain one `..` too.

- [x] `br/collectors.py`: `from . import checkpoint, client, config, validate` → `from .. import checkpoint, client, config, storage, validate`
- [x] `br/pipeline.py`: `from . import collectors, config, yf_collectors` → `from . import collectors` + `from .. import config, yf_collectors`
- [x] `br/collect_delisted.py`: `from . import collectors, config` → `from . import collectors` + `from .. import config`
- [x] `br/cvm_statements.py`: `from .cvm.X import Y` (×6) → `from ..cvm.X import Y`
- [x] `br/stats.py`: `from . import config` → `from .. import config`
- [x] `us/fred_collectors.py`: `from . import checkpoint, client, config, validate` → `from .. import checkpoint, client, config, storage, validate`
- [x] `us/run_us_full_scale.py`: `from . import config` → `from .. import config`; `from .sec import ...` → `from ..sec import ...`; `from .yf_collectors import ...` → `from ..yf_collectors import ...`
- [x] `one_off/backfill_known_gaps.py`: `from .yf_collectors import backfill_price_gap` → `from ..yf_collectors import backfill_price_gap`
- [x] `one_off/fix_mrfg3_adj_close.py`: `from . import config` → `from .. import config`

## 4. Update call sites outside `src/data_collection/`

Confirmed via grep — nothing else in the repo does dynamic/string-based imports
(`importlib`, `__import__`) of these modules, and no CI/packaging config lists
explicit module paths, so this is a plain mechanical find-and-replace.

**Tests** (module path in the `from src.data_collection... import` line changes;
test logic is untouched):
- [x] `tests/data_collection/test_skip_existing.py` → `from src.data_collection.br import collectors, config` (config path unchanged, only collectors moves)
- [x] `tests/data_collection/test_macro_bare_object.py` → `collectors` from `.br`
- [x] `tests/data_collection/test_collect_delisted.py` → `collect_delisted`, `collectors` from `.br`
- [x] `tests/data_collection/test_pipeline_dispatch.py` → `pipeline` from `.br`
- [x] `tests/data_collection/test_fred_collectors.py` → `fred_collectors` from `.us`
- [x] `tests/data_collection/test_chunk_dates_leap_year.py` → `_chunk_dates` from `storage` (see §1)
- [x] `tests/data_collection/test_merge_save_new_rows_only.py` → `_merge_save` from `storage` (see §1)

All `test_sec_*.py`, `test_cvm_*.py`, `test_client_fail_fast.py`,
`test_prices_yf_skip_existing.py`, `test_prices_consecutive_failures.py`,
`test_yf_collectors_demo.py`, `test_ratios_no_inf.py`, `test_us_data_quality.py`
need **no changes** — they import `sec`, `cvm`, `client`, `config`, `yf_collectors`,
all of which don't move.

**Docs / CLI invocations** — every `python -m src.data_collection.X` where X moved:
- [x] `README.md`: `pipeline` → `br.pipeline` (2 occurrences)
- [x] `CLAUDE.md`: `pipeline` → `br.pipeline` (Run Commands section + module table); `cvm_statements` → `br.cvm_statements`; add `br.`/`us.`/`one_off.` prefixes to the Key Modules table; note `run_us_full_scale` → `us.run_us_full_scale`
- [x] Each moved file's own module docstring (`Usage: python -m src.data_collection.X`) — update to new dotted path
- [x] `docs/US_EQUITIES_EXPANSION_PLAN.md`, `docs/US_COLLECTOR_BUG_AUDIT.md`, `docs/US_QUARTERLY_BACKFILL_PLAN.md`, `docs/PIPELINE_FORENSIC_AUDIT_2026-07-23.md`, `docs/PORTFOLIO_*.md`: historical/narrative references, lower priority — a search-and-replace pass, not correctness-critical (these are audit logs, not runnable instructions)

**Not affected** (verified): `artifacts/checkpoints/*.json` (keyed by collector-name
strings like `"prices"`, not module paths), `.github/workflows/ci.yml` (only invokes
`src.build_dataset.build_ml_dataset`), `pyproject.toml` (no explicit path list).

## 5. Ruff / lint

- [x] `ruff check .` after the move — pure import-path fixups, should be clean, but
      confirm no `F401`/`F821` from a missed relative-import fixup.

## 6. `data/` — proceeding despite the cost (per go-ahead 2026-08-05)

Original recommendation (above, superseded) was to skip this: it's a cosmetic
symmetry gain, not a coupling-bug fix like §1. Rafael wants it done anyway, "risks"
acknowledged. §7–§12 below are the complete, verified plan — every file that
touches a BR raw-data path, confirmed by grepping the whole repo (not just
`src/`/`tests/`), so nothing is discovered mid-migration.

**One fact that changes the risk profile a lot: only the BR side is git-tracked.**
`data/raw/us/` is fully covered by a `.gitignore` rule (line 60) — `git ls-files
data/raw/us | wc -l` returns **0**. It is **not moving anyway** (see target layout
below), so there is zero git or filesystem operation on the US side at all. The
only real data movement is `git mv` on the BR side: **2,139 git-tracked files**
across 9 subdirectories, which git will record as renames (content-identical,
`git log --follow` keeps working).

## 7. Target layout

```
data/raw/
  br/                  # NEW — everything currently directly under data/raw/ moves here, unchanged internally
    prices/
    fundamentals/
    macro/
    company_info/
    dividends/
    corporate_events/
    cvm/
    filing_dates/
    reference/
  us/                  # UNCHANGED — already correctly namespaced, not touched
    prices/
    fundamentals/
    macro/
    dividends/
    sec/
```

Confirmed via `ls data/raw`: those 9 names are the *entire* current contents of
`data/raw/` outside `us/` — nothing is left behind, nothing is ambiguous.

## 8. Git mechanics (BR side only)

```bash
git mv data/raw/prices           data/raw/br/prices
git mv data/raw/fundamentals     data/raw/br/fundamentals
git mv data/raw/macro            data/raw/br/macro
git mv data/raw/company_info     data/raw/br/company_info
git mv data/raw/dividends        data/raw/br/dividends
git mv data/raw/corporate_events data/raw/br/corporate_events
git mv data/raw/cvm              data/raw/br/cvm
git mv data/raw/filing_dates     data/raw/br/filing_dates
git mv data/raw/reference        data/raw/br/reference
```

`git mv` creates `data/raw/br/` implicitly on the first call. No `.gitignore`
change needed — the existing `data/raw/us/` line is untouched since that path
doesn't move.

- [x] Run the 9 `git mv` commands above
- [x] `git status` — confirm exactly 2,139 renames, nothing added/deleted

## 9. Path-constant definitions (the only 2 files with real logic)

Both `src/data_collection/config.py` and `src/build_dataset/paths.py` are already
"single source of truth" modules — every other consumer in the repo (`loaders.py`,
`merge.py`, `repair.py`, `collectors.py`, `cvm/*.py`, `src/portfolio/*.py`, most of
`tests/`) imports the *name* (`PRICES_DIR`, `FUND_DIR`, etc.), not a literal string.
Confirmed by grepping every one of those names repo-wide: every consumer outside
these two files references the constant by name — fixing the two definitions
fixes every downstream consumer automatically, no separate edits needed there.

**`src/data_collection/config.py`** (lines 122–137) — insert one new
`BR_RAW_DIR` constant, rebase the 7 BR dir constants onto it, leave every `US_*`
line untouched:

```python
# --- Paths ---
RAW_DIR = PROJECT / "data/raw"
BR_RAW_DIR = RAW_DIR / "br"         # NEW
PRICES_DIR = BR_RAW_DIR / "prices"          # was RAW_DIR / "prices"
FUND_DIR = BR_RAW_DIR / "fundamentals"      # was RAW_DIR / "fundamentals"
MACRO_DIR = BR_RAW_DIR / "macro"            # was RAW_DIR / "macro"
COMPANY_DIR = BR_RAW_DIR / "company_info"   # was RAW_DIR / "company_info"
DIVIDENDS_DIR = BR_RAW_DIR / "dividends"    # was RAW_DIR / "dividends"
CORP_EVENTS_DIR = BR_RAW_DIR / "corporate_events"  # was RAW_DIR / "corporate_events"
CVM_DIR = BR_RAW_DIR / "cvm"                # was RAW_DIR / "cvm"
US_RAW_DIR = RAW_DIR / "us"         # unchanged
US_MACRO_DIR = US_RAW_DIR / "macro"                 # unchanged
US_PRICES_DIR = US_RAW_DIR / "prices"               # unchanged
US_SEC_DIR = US_RAW_DIR / "sec"                     # unchanged
US_COMPANY_INFO_PATH = US_SEC_DIR / "company_info.parquet"  # unchanged
US_FUNDAMENTALS_DIR = US_RAW_DIR / "fundamentals"   # unchanged
US_DIVIDENDS_DIR = US_RAW_DIR / "dividends"         # unchanged
```

**`src/data_collection/cvm/filing_dates.py:29`** — the one BR path that does
*not* go through a named `config.*_DIR` constant, so it needs its own fix or it
would silently keep writing to the old (now-nonexistent-post-move) location:
```python
OUTPUT_PATH = config.RAW_DIR / "filing_dates/filing_dates.parquet"
# ->
OUTPUT_PATH = config.BR_RAW_DIR / "filing_dates/filing_dates.parquet"
```

**`src/build_dataset/paths.py`** (lines 13–21) — same treatment, insert `/br` in
the 9 BR literal strings, leave lines 29–36 (`US_*`) untouched:
```python
PRICES_DIR = ROOT / "data/raw/br/prices"
FUNDAMENTALS_DIR = ROOT / "data/raw/br/fundamentals"
COMPANY_INFO_PATH = ROOT / "data/raw/br/company_info/company_info.parquet"
CVM_CROSSWALK_PATH = ROOT / "data/raw/br/cvm/fca_crosswalk.parquet"
MACRO_DIR = ROOT / "data/raw/br/macro"
DIVIDENDS_DIR = ROOT / "data/raw/br/dividends"
CORPORATE_EVENTS_PATH = ROOT / "data/raw/br/corporate_events/corporate_events.parquet"
FILING_DATES_PATH = ROOT / "data/raw/br/filing_dates/filing_dates.parquet"
CONTINUITY_PATH = ROOT / "data/raw/br/reference/ticker_continuity.json"
# US_PRICES_DIR etc. (lines 29-36): unchanged, still "data/raw/us/..."
```

- [x] Edit `config.py` paths block as above
- [x] Edit `cvm/filing_dates.py:29`
- [x] Edit `build_dataset/paths.py` paths block as above

## 10. Standalone literal-path fixes (functional, not just docstrings)

These files build a `data/raw/...` path themselves instead of importing the
centralized constant — confirmed each one is a real runtime path, not a comment,
by reading the surrounding code:

- [x] `src/build_dataset/cagr_handler.py:311` — `default="../data/raw/fundamentals"` → `"../data/raw/br/fundamentals"` (standalone CLI script, run from `src/build_dataset/`)
- [x] `scripts/inspect/inspect_all_data.py:43` — `RAW_DIR = Path("../data/raw")` → `Path("../data/raw/br")` (run from `scripts/inspect/`; the `FOLDERS` list's `"financials"` entry is a pre-existing dead reference — no such dir ever existed — leave it, out of scope)
- [x] `scripts/inspect/inspect_company_info.py:22` — `PARQUET_PATH = "../data/raw/company_info/company_info.parquet"` → add `/br`
- [x] `tests/build_dataset/test_final_dataset.py:28-29` — `CORPORATE_EVENTS_FILE`, `FUNDAMENTALS_DIR` (test-local redefinitions, not imported from `paths.py`) → add `/br`
- [x] `tests/data_collection/test_collect_delisted.py:55` — `path = ROOT / f"data/raw/prices/{ticker}.parquet"` → add `/br`
- [x] `tests/data_collection/validate_vs_yfinance.py:26-27` — `PRICE_DIR`, `FUND_DIR` → add `/br`
- [x] `tests/data_collection/test_cagr_calculation.py:34` — `FUND_DIR = "data/raw/fundamentals"` → add `/br`

`tests/data_collection/test_us_data_quality.py` needs **no change** — every path
in it already reads `data/raw/us/...`.

## 11. Notebooks (git-tracked, real literal paths in cell source)

- [x] `src/visualizations/exploration.ipynb:88` — `data_root = Path("data/raw")` → `Path("data/raw/br")` (single edit; lines 91/95/99/102-105 all derive from `data_root`, fixed automatically). Line 47's `Path('data/raw').exists()` check still returns `True` either way (the parent dir still exists) — harmless, optional to touch.
- [x] `src/visualizations/agent_vs_benchmarks.ipynb:68,74` — 2 literal `data/raw/prices`, `data/raw/macro/{name}` cells → add `/br`
- [x] `src/visualizations/rolling_eval_results.ipynb:30900,59800,59806` — 3 literal `data/raw/prices...`, `data/raw/macro/{name}` cells → add `/br`

The latter two notebooks predate the 2026-07-23 Stage 3+ reset (last touched
2026-06-30/07-09, before the reset) and may be stale/orphaned — flagging, not
resolving that here; fixed anyway since they're real tracked files and the edit
is cheap.

## 12. Docs — cosmetic pass, not correctness-critical

Occurrence counts (bare `data/raw/...` mentions, confirmed each file also
contains legitimate `data/raw/us/...` mentions that must NOT be touched):

| File | `data/raw` mentions |
|---|---|
| `CLAUDE.md` | 12 |
| `docs/PIPELINE_FORENSIC_AUDIT_2026-07-23.md` | 7 |
| `docs/US_DATASET_BUILD_PLAN.md` | 4 |
| `docs/US_COLLECTOR_BUG_AUDIT.md` | 3 |
| `docs/US_EQUITIES_EXPANSION_PLAN.md` | 3 |
| `docs/FEATURE_IDEAS.md`, `docs/DATA_INTEGRITY_AUDIT_2026-07-24.md`, `docs/US_QUARTERLY_BACKFILL_PLAN.md` | 2 each |
| `README.md`, `docs/US_DATASET_AUDIT_2026-08-01.md`, `docs/PORTFOLIO_ARCHITECTURE_PROPOSAL.md` | 1 each |
| `tests/run_all.py` (comments only, not functional) | 2 |

**Do not bulk-replace `data/raw/` → `data/raw/br/` blindly** — that would corrupt
every existing `data/raw/us/...` mention into `data/raw/br/us/...`. Either edit
file-by-file, or use a negative-lookahead pass and spot-check the diff:
```bash
perl -pi -e 's{data/raw/(?!us/)}{data/raw/br/}g' CLAUDE.md README.md docs/*.md tests/run_all.py
```
- [x] Run the pass above (or edit manually), then `git diff` and confirm no
      `data/raw/br/us/` artifacts were introduced
- [x] `CLAUDE.md`'s "Data on Disk" section specifically needs a rewrite (not just
      a path substitution) — it currently frames BR as the implicit/flat default
      and US as "nested under `us/`"; after this move both are symmetric and that
      framing sentence should change, not just the paths inside it

## 13. What does NOT need to change (verified)

- `artifacts/checkpoints/*.json` — keyed by collector-name strings (`"prices"`,
  `"fundamentals"`, ticker symbols), never a filesystem path. Confirmed by reading
  a live checkpoint file.
- `.env.example` — no `data/raw` reference.
- `.github/workflows/ci.yml` — only invokes `src.build_dataset.build_ml_dataset`.
- `pyproject.toml` / ruff config — no explicit path list.
- Every consumer that imports `PRICES_DIR`/`FUND_DIR`/etc. *by name* rather than
  redefining a literal string: `src/build_dataset/{loaders,merge,repair,
  quality_filters,continuity,manifest,scale_features,build_top50_universe,
  build_us_dataset,build_ml_dataset}.py`, `src/data_collection/{collectors,
  collect_delisted,stats,yf_collectors}.py`, `src/data_collection/cvm/{crosswalk,
  shares,ratios}.py`, `src/portfolio/*.py`, `src/visualizations/sample_us_dataset.py`,
  and the majority of `tests/build_dataset/*.py` — confirmed via repo-wide grep on
  every BR constant name (`PRICES_DIR`, `FUNDAMENTALS_DIR`, `COMPANY_INFO_PATH`,
  `CVM_CROSSWALK_PATH`, `MACRO_DIR`, `DIVIDENDS_DIR`, `CORPORATE_EVENTS_PATH`,
  `FILING_DATES_PATH`, `CONTINUITY_PATH`) that none of these redefine the literal
  string themselves.
- `graphify-out/` — contains cached path references but is gitignored and fully
  regenerable; just rerun `/graphify . --update` after the move instead of
  hand-editing the cache.
- `data/processed/` — untouched entirely (this plan only moves `data/raw/`).

## 14. Execution order

1. §8 — `git mv` the 9 BR subdirectories (data layer first, so a mid-edit crash
   leaves data in the new location, not split)
2. §9 — fix the 3 path-constant definition points (`config.py`,
   `cvm/filing_dates.py`, `build_dataset/paths.py`)
3. §10 — fix the 7 standalone literal-path files
4. §11 — fix the 3 notebooks
5. Run `python tests/run_all.py --group data` — this is the real correctness
   check: it needs the git-tracked `data/raw/*` on disk and will fail loudly on
   any missed path
6. §12 — docs cosmetic pass, `git diff` sanity check for `data/raw/br/us/` typos
7. `/graphify . --update` to refresh the stale `graphify-out/` cache

---

## 15. Why this reorg needed a dozen files, and how to make the next one need one

`build_dataset/paths.py`'s own docstring already claims to be a "single source of
truth" — and it is, but only *within* `build_dataset`. It doesn't import from
`data_collection/config.py`; it **independently retypes the same 9 physical BR raw
paths** as separate literal strings. That's not a coincidence causing this
particular pain — it's the direct mechanism: two independent definitions of the
same physical location can't help but drift apart the moment the physical
location changes, and every consumer that skips *both* canonical modules and
types its own literal (§10/§12's 7+ files) makes it worse. This pattern repeats
smaller-scale in two more places, confirmed by grep:

- `"company_info.parquet"` joined onto `config.COMPANY_DIR` independently in
  **4 places**: `collectors.py:353`, `pipeline.py:61`, `pipeline.py:69`,
  `cvm/company_info.py:56` (`paths.py`'s `COMPANY_INFO_PATH` makes it a 5th).
- `"corporate_events.parquet"` joined onto `config.CORP_EVENTS_DIR` inline in
  `collectors.py:458`, independently retyped again in `paths.py`'s
  `CORPORATE_EVENTS_PATH`.
- `data/processed/...` has the same drift, smaller scale: `sample_us_dataset.py:17`,
  `tests/build_dataset/test_top_traded_quality.py:33`, and
  `tests/build_dataset/test_final_dataset.py:27` each hardcode their own literal
  instead of importing `paths.OUTPUT_PATH`/`paths.US_OUTPUT_PATH`.

**Confirmed safe to fix:** grepped `src/data_collection/` for any import of
`build_dataset` — none exists (only comments mention the package name). The
dependency only ever runs one way, matching the pipeline's actual data flow
(collection → build): `build_dataset` importing from `data_collection` doesn't
create or risk a cycle.

### 15.1 Consolidate `config.py` — add the 2 missing file-level constants

```python
# after COMPANY_DIR / CORP_EVENTS_DIR are defined:
COMPANY_INFO_PATH = COMPANY_DIR / "company_info.parquet"
CORP_EVENTS_PATH = CORP_EVENTS_DIR / "corporate_events.parquet"
```
- [x] `collectors.py:353`, `pipeline.py:61`, `pipeline.py:69`,
      `cvm/company_info.py:56` → replace `config.COMPANY_DIR / "company_info.parquet"`
      with `config.COMPANY_INFO_PATH`
- [x] `collectors.py:458` → replace `config.CORP_EVENTS_DIR / "corporate_events.parquet"`
      with `config.CORP_EVENTS_PATH`

### 15.2 Make `build_dataset/paths.py` a re-exporter for RAW paths, not a redefiner

Only the **PROCESSED**-side constants (`OUTPUT_PATH`, `SPLIT_CONFIG_PATH`,
`SCALER_DIR`, `TOP50_*`, `US_OUTPUT_PATH`, `US_SPLIT_CONFIG_PATH`,
`US_SCALER_DIR`) are genuinely `build_dataset`'s own artifacts — those stay
defined here. Everything RAW-side gets imported instead of retyped:

```python
"""
paths.py — shared filesystem paths for the dataset build.

RAW-side paths are owned by data_collection/config.py (Stage 1 decides where
collected data lives) and re-exported here so Stage-2 submodules keep a single
import to reach for. This module defines only its own PROCESSED-side outputs
-- the one thing genuinely specific to the dataset build.
"""

from pathlib import Path

from src.data_collection.config import (
    PRICES_DIR,
    FUND_DIR as FUNDAMENTALS_DIR,
    COMPANY_INFO_PATH,
    MACRO_DIR,
    DIVIDENDS_DIR,
    CORP_EVENTS_PATH as CORPORATE_EVENTS_PATH,
    BR_RAW_DIR,
    US_PRICES_DIR, US_FUNDAMENTALS_DIR, US_DIVIDENDS_DIR, US_MACRO_DIR,
    US_COMPANY_INFO_PATH,
)
from src.data_collection.cvm.crosswalk import CROSSWALK_PATH as CVM_CROSSWALK_PATH
from src.data_collection.cvm.filing_dates import OUTPUT_PATH as FILING_DATES_PATH

ROOT = Path(__file__).resolve().parents[2]

# Hand-maintained (docs/... ticker renames); no producer in data_collection to
# import from, so this one stays a locally-built path, just off the shared root.
CONTINUITY_PATH = BR_RAW_DIR / "reference/ticker_continuity.json"

# --- This module's own concern: build output ---
OUTPUT_PATH = ROOT / "data/processed/ml_dataset.parquet"
SPLIT_CONFIG_PATH = ROOT / "data/processed/split_config.json"
SCALER_DIR = ROOT / "data/processed/scalers"
TOP50_UNIVERSE_PATH = ROOT / "data/processed/ml_dataset_top50_universe.parquet"
TOP50_MEMBERSHIP_PATH = ROOT / "data/processed/top50_universe_membership.parquet"
US_OUTPUT_PATH = ROOT / "data/processed/us_ml_dataset.parquet"
US_SPLIT_CONFIG_PATH = ROOT / "data/processed/us_split_config.json"
US_SCALER_DIR = ROOT / "data/processed/us_scalers"
```

Every existing consumer (`loaders.py`, `merge.py`, `repair.py`, `continuity.py`,
`quality_filters.py`, `manifest.py`, `src/portfolio/*.py`,
`src/visualizations/sample_us_dataset.py`, `tests/build_dataset/*.py`) imports
these names *from `paths.py`*, unchanged — this is a pure internal refactor of
one file, zero ripple to any of its ~20 consumers. Confirmed no name changes
(`FUNDAMENTALS_DIR`, `CORPORATE_EVENTS_PATH`, etc. keep their existing
`paths.py`-side names via the `as` aliases above even though `config.py` spells
them differently internally).

- [x] Rewrite `build_dataset/paths.py` as above
- [x] Run `python tests/run_all.py --group fast` — pure import-graph change,
      should be silent

### 15.3 Do this *before* §6–§14, not after

With 15.1/15.2 applied, §9's `build_dataset/paths.py` edit disappears entirely —
moving `data/raw/` to `data/raw/br/` then only requires touching **one**
definition point (`data_collection/config.py`'s `BR_RAW_DIR`, plus the
already-noted `cvm/filing_dates.py:29` exception) instead of two. Recommended
order: 15.1 → 15.2 → verify fast tests green → then §6–§14 (the physical move).

### 15.4 Regression guard — one runnable check

Nothing above stops the *next* new file from hardcoding a literal again. Add a
grep-based test (fits the existing "fast" group — pure code, no data files):

```python
# tests/build_dataset/test_no_hardcoded_data_paths.py
"""Guards the path-consolidation invariant (docs/DATA_COLLECTION_REORGANIZATION_PLAN.md
§15): raw/processed data locations must be imported from config.py/paths.py, never
retyped -- that drift is what made the 2026-08 data/raw/br/ move touch 10 files
instead of 2."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {
    ROOT / "src/data_collection/config.py",
    ROOT / "src/build_dataset/paths.py",
    ROOT / "src/data_collection/cvm/crosswalk.py",
    ROOT / "src/data_collection/cvm/filing_dates.py",
}
PATTERN = re.compile(r'''["']\.{0,2}/?data/(raw|processed)/''')


def find_violations() -> list[str]:
    violations = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path in ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if PATTERN.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    return violations


def test_no_hardcoded_data_paths():
    violations = find_violations()
    assert not violations, "hardcoded data/raw or data/processed path(s) found " \
        "outside config.py/paths.py -- import the constant instead:\n" + "\n".join(violations)


if __name__ == "__main__":
    test_no_hardcoded_data_paths()
    print("OK")
```

Scoped to `src/**/*.py` only — the pipeline code that actually runs regularly.
`scripts/inspect/*.py` (standalone dev utilities, no `src.*` imports today) and
`tests/` (some legitimately need their own fixture-style literals) are lower
stakes; converting `scripts/inspect/*.py` to import `config.py` too is a nice-to-have,
not required, since they're rarely touched and not part of the pipeline.

- [x] Add `tests/build_dataset/test_no_hardcoded_data_paths.py` as above
- [x] Add it to `tests/run_all.py`'s fast group
- [x] Confirm it passes once 15.1/15.2 are in place (it will fail before, on the
      3 `data/processed` hardcodes listed above — those aren't part of §6-§14's
      scope but are cheap 1-line fixes to pick up while touching this)

---

## Summary checklist

- [x] §1 — extract `storage.py`, repoint `fred_collectors.py`/`cvm/ratios.py`/`yf_collectors.py` (fixes real US→BR coupling bug)
- [x] §2 — `git mv` 9 files into new `br/`, `us/`, `one_off/` packages (code reorg)
- [x] §3 — fix relative imports in the 9 moved files
- [x] §4 — fix 7 test files' import lines, update README/CLAUDE.md/docstrings/docs
- [x] §5 — `ruff check .` clean
- [x] §15.1/§15.2 — consolidate duplicate path definitions (`config.py` gains 2
      constants, `paths.py` becomes a re-exporter) — **do this first**, it shrinks
      §6-§14 from 2 definition files to 1
- [x] §15.4 — add the hardcoded-path regression-guard test; fix the 3
      `data/processed` hardcodes it will catch
- [x] §6-§14 — `data/raw/` BR→`br/` migration: git-mv 2,139 tracked files, fix the
      now-single path-constant definition point, fix 7 standalone scripts/tests,
      fix 3 notebooks, cosmetic docs pass, verify with `tests/run_all.py --group data`

Explicitly **not** doing: moving `cvm/`/`sec/` under `br/`/`us/` in the code layer
(already self-labeled, would touch every file inside both packages for no real
gain); extending `stats.py` to cover US (`stats.py` moves as-is under §2; covering
US is a feature gap, separate ask); moving `data/raw/us/` (already correctly
placed, not git-tracked, zero reason to touch it); converting
`scripts/inspect/*.py` to import `config.py` (nice-to-have, not required);
env-var-based/runtime-relocatable data roots (speculative — nothing in this repo
needs the data location to vary at runtime, a single centralized constant module
is enough).
