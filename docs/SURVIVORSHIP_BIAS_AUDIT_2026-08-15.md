# Survivorship-Bias Audit & Remediation — 2026-08-15

Scope: both markets' data-collection and Stage-2/3 pipelines, re-verified live against CVM's
and SEC's own sources (not just this repo's cached copies or prior docs) before any code
changed. Two changes landed from this: BR terminal-event labels (`src/build_dataset/
terminal_events.py`), and a US survivorship-coverage manifest field. One investigated fix
(BR crosswalk repair) turned out to be a dead end and is recorded here so it isn't re-attempted.

---

## 1. BR: the FCA crosswalk cannot be repaired (dead end, verified)

**Starting hypothesis:** 189 BR tickers with real price history, delisted 2010–2023, were
missing from the CVM fundamentals crosswalk (`cvm/crosswalk.py`) despite CVM's open-data floor
being `START_YEAR=2010` — looked like a fixable gap.

**What live verification against CVM's own FCA zips found (not this repo's cache):**

1. `Codigo_Negociacao` (the trading-code field itself) is **100% blank in every FCA filing,
   every company, 2010–2017** — checked full-year samples (494/658/628 rows for 2012/2015/2017,
   zero populated). Populated only from 2018 on (402–751 rows/year). The crosswalk's real
   recoverable floor is **2018**, four years past `START_YEAR`.
2. FCA reports the code *as of filing* — survivor-style, the same failure mode as SEC's
   `company_tickers.json` (`sec/universe.py`'s own docstring already names this pattern).
   Confirmed: `KROT3` (delisted via rename to `COGN3` in 2019) appears in **zero** FCA years
   2018–2026; only `COGN3` does. No amount of re-scanning years recovers a renamed/delisted
   code — the source doesn't carry it, period.
3. A hypothesized zero-padding regex fix (`BSLI04` → `BSLI4`, `PRPT3B` → `PRPT3`) was tested
   against all 782 distinct trading codes in FCA 2018–2026: 104 rejected by the current regex,
   31 normalize to a standard ticker shape — of those, **0** are tickers with prices on disk and
   no crosswalk entry. **Zero recovery. Not built.**
4. Every other CVM open-data module was checked for a ticker field: FCA's other 9 sub-modules
   (`geral`, `auditor`, `endereco`, `escriturador`, `dri`, `canal_divulgacao`,
   `departamento_acionistas`, `pais_estrangeiro_negociacao`), FRE's `volume_valor_mobiliario`
   module, and the static `cad_cia_aberta.csv` master registry (2,677 rows, live-fetched) — none
   carry a trading-code field. `valor_mobiliario` is CVM's only source for one, and its blank
   pre-2018 field is a hard ceiling.

**Conclusion:** not a bug, not a wrong-module problem. Documented in `cvm/crosswalk.py`'s
docstring and `cvm/http.py`'s `START_YEAR` comment so this isn't re-investigated from scratch.
The one remaining avenue, not pursued: B3's free COTAHIST daily files carry ticker+company-name+
ISIN back to ~1986 and could bridge ticker→name→CNPJ via fuzzy matching (same shape as the US
side's Alpha Vantage name-matching, see §3) — outside CVM's own portal, untouched here.

---

## 2. BR: terminal-event labels (implemented)

**The audit this replaces assumed the wrong failure mode.** `docs/PIPELINE_FORENSIC_AUDIT_
2026-07-23.md` Issue 9(b) framed the gap as "a delisted/bankrupt ticker's series simply stops
... any return computed over 'held to the end' positions never realizes the loss" — implying a
blanket −100% fix. Measuring the real panel first:

| | |
|---|---|
| Tickers whose series ends >365d before the panel's own last date | 125 |
| ...of those with enough history to measure a 60-day trend | 114 |
| Died at <10% of their own peak (wipeout already in the price) | 22 |
| Died at 10–50% of peak | 46 |
| **Died at >50% of peak (loss NOT in the price)** | 46 |
| Median final-60-trading-day return | **+2.9%** |
| **Died while still RISING over the final 60 days** | **64 / 114** |

Acquisition-at-a-premium is the dominant BR terminal event, not bankruptcy. A blanket −100%
would have made the label *worse*, not better.

**What was built** (`src/data_collection/cvm/delistings.py` + `src/build_dataset/
terminal_events.py`):

- `cvm/delistings.py` fetches CVM's own `cad_cia_aberta.csv` (cancellation date + reason +
  current registry status), joined to tickers via the existing FCA crosswalk. New CLI step:
  `python -m src.data_collection.br.cvm_statements --step delistings`.
  - **Real bug caught testing against live data, not synthetic fixtures:** `cad_cia_aberta.csv`
    carries one row per registration *episode*, not one per company — 140/2,530 CNPJs had
    duplicate rows, 34 with genuinely different `SIT`/`DT_CANCEL` (e.g. Vibra Energia: one stale
    2003 `CANCELADA` row from a pre-1978-rule registration, plus a current `ATIVO` row). A naive
    `drop_duplicates` would have silently read some still-active companies as delisted. Fixed:
    keep the row reflecting the company's actual current state — prefer any `ATIVO` row, else
    the latest `DT_CANCEL`.
- `terminal_events.py` turns the resolved reason into a payoff: bankruptcy/liquidation reasons
  (`LIQUIDAÇÃO EXTRAJUDICIAL`, `ELISÃO POR EXTINÇÃO DA CIA`, `CANCELAMENTO DE OFÍCIO`) pay `0.0`;
  every other resolved reason (voluntary cancellation, incorporation/merger) pays the ticker's
  own last observed `adj_close`. A ticker whose registry status is still `ATIVO` gets **no**
  terminal payoff — that's an unspliced rename, not a delisting; `find_rename_candidates()`
  reports those separately (report only, never auto-applied to `ticker_continuity.json`).
- `src/portfolio/labels.py`'s `forward_excess_return()` gained an optional `terminal_events`
  param: only rows that are *already* NaN because the forward window runs past the ticker's own
  last row get filled — a live ticker, a delisted-but-unresolved ticker, or a mid-history NaN
  (e.g. precision-degraded) is untouched by construction. The CDI leg compounds only to the
  ticker's own last date, not the full horizon — the position is realized there, not at `t+H`.
  Wired into all 4 real callers (`run_full_backtest.py`, `run_alpha_diagnostic.py`,
  `visualize_portfolio.py`, `plot_tree.py`) via `terminal_events.load_terminal_events()`.

**Measured effect on the real panel** (`data/processed/ml_dataset.parquet`, horizon=252):

| | |
|---|---|
| Delist events resolved from CVM's registry | 677 tickers (118 with a cancellation reason) |
| Terminal events applied (dead-inside-panel ∩ resolved reason) | 78 tickers (15 failure, 63 acquired) |
| Rename candidates surfaced (report only) | 31, incl. `KROT3`→`COGN3`, `VVAR4`/`VVAR11`→`BHIA3` |
| Label rows recovered NaN → real value | **17,889 (1.37% of 1,308,104)** |
| Pre-existing valid labels changed | **0** |
| Labels regressed to NaN | **0** |
| Tickers touched at all | exactly the 78 with a resolved terminal event |

Run order: `build_ml_dataset.py` → `cvm_statements.py --step delistings` → `terminal_events.py`
(a separate, deliberate step, same shape as `scale_features.py` — not run every build).

---

## 3. US: survivorship coverage is now a manifest field (measurement only)

`sec/universe.py` already built a genuinely survivorship-bias-free roster (every CIK that filed
a 10-K/10-Q since 1994 — verified to drop Enron/Lehman/WorldCom/pre-buyout Twitter in the right
quarter) and a `compute_coverage()` function to measure the gap against it — but nothing called
it. `build_us_dataset.py` now calls it before `write_manifest()`, recorded as
`survivorship_coverage` (per-year `roster_ciks`/`priced_ciks`/`coverage`).

Measured just now against the real roster/crosswalk on disk: coverage rises from ~41% (2017) to
~72% (2026) as the tier-1 (current-listings-only) crosswalk's own collection matured — the
median across all years pulled down hard by the pre-roster-maturity era, consistent with
`compute_coverage`'s own documented caveat that this is a lower bound (a tier-1-only crosswalk
can't distinguish "not yet collected" from "dead company, unrecoverable at this tier").

**Deliberately not built:** a CIK-keyed crawl of EDGAR fundamentals for the ~2,503 orphan CIKs
already name-matched against Alpha Vantage's delisted-equity roster (`docs/US_COLLECTOR_FIX_
PLAN.md` §4). `sec/fundamentals.py`'s `build_company_fundamentals(cik, filings)` already accepts
a bare CIK — the only reason this doesn't happen today is that the driver is ticker-keyed. Not
built because there is currently no consumer for fundamentals without matching prices (no
`market_cap`/valuation ratios without a close price, and price recovery for dead US companies
needs paid data — Alpha Vantage's free tier caps at 25 req/day, ~376 days serially for 9,390
symbols). Revisit only if a fundamentals-only research question comes up, or if paid price data
is ever purchased.

---

## 4. Not doing (dead ends and deliberate scope cuts)

- **FCA crosswalk regex fix** — measured 0 recovery, see §1.
- **US orphan-CIK fundamentals crawl** — no consumer today, see §3.
- **Blanket −100% on every BR delisting** — contradicted by measurement (§2); implemented
  taxonomy-based payoff instead.
- **B3 COTAHIST ticker bridge for pre-2018 BR delistings** — the one route to the 173 (of 189
  originally investigated) pre-2018-delisted tickers, but needs fuzzy name→CNPJ matching for a
  cohort of mostly small failed companies. Not attempted.
- **Auto-applying rename candidates to `ticker_continuity.json`** — `find_rename_candidates()`
  reports only; `continuity.py`'s rename/merger/keep_separate distinction is a judgement call a
  CNPJ-sharing join cannot make on its own.

## Verification

```bash
python tests/run_all.py --group fast              # 55/55, incl. new tests/portfolio/test_terminal_events.py
ruff check src/ tests/                             # clean

python -m src.data_collection.br.cvm_statements --step delistings
python -m src.build_dataset.terminal_events
python -m src.build_dataset.build_us_dataset       # writes survivorship_coverage into the manifest
```
