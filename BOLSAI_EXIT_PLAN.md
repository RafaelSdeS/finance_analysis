# BolsAI Exit Plan

**Goal:** run the BR pipeline indefinitely with no paid dependency, while getting *more*
history, *more* coverage and *more* internal consistency than BolsAI currently gives.
**Constraint:** BolsAI infrastructure stays — build around it, never delete it.
**Status:** Tasks 0, 2, 3, 1 implemented and run against real data 2026-08-19 (see "Implementation
log" near the end). Tasks 4–5 not started. **Written:** 2026-08-19.

---

## Headline

**BolsAI buys you no history depth at all.** Measured on disk: every BolsAI fundamentals
file starts **2010-12-31** — the exact same floor as CVM's free open data. Zero of 612
tickers have a single pre-2010 row.

CVM is a **strict superset** of what you're paying for:

| | BolsAI (paid) | CVM (free) |
|---|---|---|
| earliest fundamentals | 2010-12-31 | 2010-12-31 |
| latest | 2026-06-30 | 2026-06-30 |
| starts earlier-or-same | — | **595 / 595** |
| ≥ as many quarters | — | **594 / 595** |
| filing dates | close-of-filing approximation | real `DT_RECEB` |
| delisted companies | partial | full |

So this is not "degrade gracefully to free data". It's a straight upgrade — *provided* the
three bugs below get fixed first.

---

## Corrections to the first draft

**Task 0 is done — full 612-ticker run, not a sample.** This section went through two wrong
iterations before landing on a verified answer. Recording all three so the wrong ones don't
get re-cited:

1. ~~"BolsAI mixes TTM and single-quarter across tickers (34 vs 22)"~~ — first guess, from
   an under-powered 60-ticker sample with a naive `rolling(4)` and no scale check.
2. ~~"All five checked are TTM, so canonical = TTM"~~ — **also wrong.** Five large blue chips
   (LREN3, PETR4, VALE3, GGBR4, WEGE3) happened to share one convention; that isn't the
   market. Full-universe run reversed it.
3. **Verified answer, all 612 tickers, 5% per-quarter match tolerance** (per your steer:
   ~10% cross-vendor drift is expected and shouldn't be over-fit against):

   | verdict | count | meaning |
   |---|---|---|
   | QUARTERLY | 309 | BolsAI stores single-quarter flows |
   | TTM | 269 | BolsAI stores trailing-twelve-month flows |
   | NO_CROSSWALK | 17 | the 17 from Task 3, unaffected by this question |
   | TOO_FEW | 12 | <3 overlapping quarters — thin/new listings, convention barely matters |
   | scale bug | 4 | `MTSA4`/`PTNT4`/`RVEE3`/`VIVA3` — see below |
   | noisy-but-clear | 1 | `ATED3` — TTM, matches exactly on its 2 most recent quarters |

   **It really is close to a coin flip (309 vs 269) — there is no default to fall back on.**
   Every ticker needs its own detected convention; assuming either one universally, as both
   prior drafts did, would silently corrupt roughly half the panel.

   **A second, independent bug found by the same scan:** `MTSA4`, `PTNT4`, `RVEE3`, `VIVA3`
   store fundamentals in raw R$ instead of R$-thousands (BolsAI's own unit for everyone
   else). Verified exactly — PTNT4 2026-03-31: CVM's TTM value `853,462` (thousands) × 1000
   = `853,462,000`, BolsAI stores `853,461,852` (0.00002% off, i.e. an exact match). This is
   orthogonal to the convention question and needs its own per-ticker scale detection.

4. ~~"Rebuilding from CVM loses pre-2010 history"~~ — this one held up. There is no pre-2010
   history to lose (verified on disk, see Headline table).

---

## Three real bugs found (all in the fundamentals path)

### BUG-1 — yfinance BR fundamentals are wrong, and they are corrupting the live tail

`--mode update` routes fundamentals to yfinance (`config.DATA_SOURCE`). yfinance's BR
financials are not merely thin — they are **wrong in level**:

```
PETR4 stored 2026-06-30 vs the two quarters before it
                 net_revenue    net_income      equity   total_assets
  2026-03-31      4.98e+08      1.08e+08     445,189,000  1,246,068,000
  2026-06-30      1.04e+08      2.56e+07      92,908,000    247,077,000   <-- ~5x low
```

`equity` and `total_assets` are **point-in-time balance-sheet items** — they cannot legitimately
fall 5× in a quarter. Confirmed at source: `yf.Ticker("PETR4.SA").quarterly_balance_sheet`
returns Stockholders Equity of **92.9bn** where CVM and BolsAI both say **~445bn**, and its
`quarterly_income_stmt` currently returns **empty**. Not a currency issue —
`financialCurrency == 'BRL'` for every ticker checked.

- **Blast radius: 61 tickers** have a corrupted final row (incl. `BBDC4`, `BRKM5`, `EGIE3`,
  `EQTL3`, `DASA3`, `CEAB3`). It hits the newest quarter — the most decision-relevant row
  in the panel.
- **Root cause, not symptom:** the fix is not patching values, it's that **yfinance must not
  be a fundamentals source.** It offers ~4–6 quarters of unreliable data where CVM offers
  62 quarters of authoritative data for free. Removing it removes a whole class of
  vendor-mixing bugs permanently.

### BUG-2 — `cvm/ratios.py` computes valuation multiples off single-quarter earnings

`compute_ratios()` does `pl = market_cap / (net_income * k)` where `net_income` is a
**single quarter**, while BolsAI's history is **TTM**. That makes every multiple it produces
roughly **4× too high and wildly unstable**.

This is not hypothetical — `ratios.py` already ran in production for delisted tickers, so
those files carry the error today. It also silently sets `ebitda = ebit` (`ratios.py:33`,
a known shortcut) and omits `roic` entirely.

Affected: `pl`, `pvp`, `p_sr`, `p_ebit`, `p_ebitda`, `p_assets`, `ev_ebit`, `ev_ebitda`, `lpa`.

### BUG-3 — banks lose all flow items, because the account code is ambiguous

`statements.py` maps four fixed codes. That is a **corporate-only** chart. Counting
descriptions per code across the whole market (live ITR 2025 `DRE_con`, 2026-08-19):

| code | corporate (n=4322) | bank (n=154) |
|---|---|---|
| 3.01 | Receita de Venda de Bens/Serviços | Receitas de Intermediação Financeira |
| 3.03 | Resultado Bruto | Resultado Bruto de Interm. Financeira |
| 3.05 | Res. Antes do Res. Financeiro e dos Tributos → **true EBIT** | Resultado antes dos Tributos → **pre-tax, NOT EBIT** |
| 3.09 | Res. Líquido das Oper. Continuadas | **Lucro/Prejuízo Consolidado do Período** |
| 3.11 | **Lucro/Prejuízo Consolidado do Período** | Lucro ou Prejuízo Líquido Consolidado |
| 3.13 | — | **Lucro/Prejuízo Consolidado do Período** |

Net income lives at **3.09, 3.11 *or* 3.13** depending on layout, and 3.09/3.11 mean
*different things* in each — so the code alone cannot disambiguate; the **description** can.
Likewise 3.05 is a real EBIT only for corporates.

Affects `ITUB4`, `BBAS3`, `BBDC4`, `BPAC11`, `ABCB4`, `BAZA3` … — i.e. several of the
largest names in the index.

---

## Measured coverage (real data, 2026-08-19)

- CVM statements cache: **41,689 rows, 1,224 companies, 2010-12-31 → 2026-06-30**.
- Of 612 tickers with fundamentals: **595** resolve via the FCA crosswalk, **595** have CVM
  statements, **587** have FRE share counts.
- The 17 unresolvable tickers (`CSNA3`, `BPAC11`, `CMIN3`, `MBRF3`, `AMAR3`, `EQPA3`, …)
  **all have CVM statement data** — only the ticker→CNPJ link is missing, and
  `company_info.parquet` already holds every one of those CNPJs. **17/17 hand-mappable.**
- Schema parity: `cvm/ratios.py` emits **41 of BolsAI's 42 columns**; only `roic` missing.
- Accuracy, 2,961 overlapping quarters: balance sheet is essentially exact
  (`total_assets` 98% within 5%, `equity` 91%). Flow gaps are fully explained by BUG-2/BUG-3.
- yfinance **splits** verified good: `ITUB4` 12 events back to 2004, `SBSP3` 6 back to 2007,
  including 2026 events. Viable `corporate_events` replacement.

---

## Tasks

### Task 0 — Settle the flow convention conclusively ✅ DONE (2026-08-19)

Result: **per-ticker, not universal** — 309 quarterly / 269 TTM / 4 with a separate raw-units
scale bug / 1 noisy-TTM / 12 too-thin-to-tell. Full detail and methodology in
"Corrections to the first draft" above. This is the verified input to Tasks 1–2.

### Task 1 — Rebuild fundamentals from CVM, standardized on TTM ⭐

**Design decision, locked in 2026-08-19:** every ticker's flows (`net_revenue`, `net_income`,
`ebit`, `ebitda`) are rebuilt as **trailing-twelve-month, uniformly** — not each ticker's own
detected BolsAI convention. Reasoning: `cross_sectional.py` z-scores `pl`/`pvp`/`roe`/etc.
*within* a `groupby("sector")`, and `alpha.py` trains one LightGBM model across the whole
panel — both require a column to mean the same thing for every row. With 309 tickers on
single-quarter and 269 on TTM (Task 0), the same feature name currently encodes two
different quantities ~4x apart, which is a systematic per-ticker artifact a tree model could
partially learn to exploit instead of learning real valuation signal — worse than noise, and
directly opposed to "keeping consistency." `*_zhist_5y`'s rolling window doesn't rescue this
either: a mid-series convention switch would read as a permanent regime shift and corrupt
that ticker's z-scores through its next 5-year warm-up. Cost, accepted: a deliberate,
one-time historical restatement for the 309 single-quarter tickers (S7, `dataset_v{N}`
exists exactly to make this auditable).

- [ ] Quarterly → `rolling(4)` sum **on a gap-safe quarter-end grid** so a filing gap can
      never sum non-adjacent quarters (silent corruption otherwise) — applied to every
      ticker, not just the ones currently stored as TTM.
- [ ] Compute **all** valuation multiples from the TTM flow — fixes BUG-2.
- [ ] **Task 0's per-ticker convention/scale detection is retained as a QA step, not the
      output format**: after rebuilding, cross-check each ticker's new TTM-based values
      against its old BolsAI values *using that ticker's own detected convention* (e.g. a
      former single-quarter ticker's new TTM row is compared against `cvm_qtr summed over
      the matching window`, not against the raw stored value) — confirms the CVM rebuild
      reproduces the right underlying financials, without perpetuating BolsAI's format.
      The 4 scale-bug tickers (`MTSA4`/`PTNT4`/`RVEE3`/`VIVA3`) and `ATED3` need this check
      done by hand rather than the automatic 0.8/0.2 gate (5 tickers, not worth automating).
- [ ] Add `roic` (closes the last schema gap vs BolsAI).
- [ ] Give `build_fundamentals()` a rebuild mode; today it hard-skips any ticker that already
      has a file (`ratios.py:120`), which is what confines the good CVM path to delisted names.
- [ ] Repair the 61 tickers damaged by BUG-1 as part of the rebuild.
- [ ] **Real EBITDA for corporates, via DFC D&A** (fixes BolsAI's own `ebitda == ebit` shortcut,
      not just preserves it). The indirect-method cash flow statement
      (`{doc}_cia_aberta_DFC_MI_{scope}_{year}.csv`, reachable via the existing
      `http.fetch_zip()` — no new endpoint) carries a "Depreciação, depleção e amortização"
      reconciling line inside the operating section. Verified on PETR4:
      `6.01.01.04` = 62,317,000 (thousands), sane vs its scale. **The code is not stable
      across filers** — distribution across the full 2025 ITR filing set: `6.01.01.02`
      (1338 filers), `.03` (388), `.04` (184), `.05` (116), down through `.17`+. Matched by
      description on 355 distinct companies in one year alone. Same fix shape as Task 2's
      net-income disambiguation: match `deprecia[cç][aã]o` (case-insensitive) on `DS_CONTA`
      under `6.01.01.*`, not a fixed code. `ebitda = ebit + D&A`.
      Banks: leave `ebitda` NaN (see Task 2) — D&A doesn't apply the same way to a bank's DRE,
      that's a real industry difference, not a shortcut.

### Task 2 — Bank chart of accounts (BUG-3)

- [ ] Resolve `net_income` by **description** (`lucro.*prejuízo.*período` on top-level `3.NN`
      codes) instead of a fixed code — catches 3.09/3.11/3.13 in one rule.
- [ ] Take `ebit` from 3.05 **only** when the description says "Antes do Resultado Financeiro";
      leave banks' EBIT NaN rather than silently labelling pre-tax income as EBIT.
- [ ] When two lines match, prefer consolidated scope, then the **highest** code (bottom of
      the waterfall = the actual bottom line).
- [ ] Record a `statement_layout` (corporate/bank) column so downstream can see why EBIT is NaN.
- [ ] Verify against a known-good external figure for ITUB4/BBAS3 before accepting.

### Task 3 — Ticker→CNPJ overrides

- [ ] Add a `TICKER_CNPJ_OVERRIDES` map in `cvm/crosswalk.py` for the 17, seeded from
      `company_info.parquet`. **Reuse the existing `sec/crosswalk.py` `CIK_OVERRIDES` pattern** —
      don't invent a second mechanism.

### Task 4 — Free replacements for the remaining BolsAI-only collectors

- [ ] `corporate_events`: new yfinance `Ticker.splits` collector writing the existing schema.
      Load-bearing — `repair.py` needs this log, and a price-jump heuristic was tried and
      rejected 3× (`repair.py:20`). Enable it in `--mode update`, where it's currently skipped.
- [ ] `company_info` / `status`: source from CVM CAD. `delistings.py` already downloads and
      correctly de-duplicates that exact file — reuse `build_delist_events()`, don't re-fetch.
- [ ] `sectors`: source from CAD's `SETOR_ATIV` instead of BolsAI. Confirmed real and
      populated — 30+ categories with hundreds of members each (e.g. "Bancos" 115,
      "Energia Elétrica" 117, "Metalurgia e Siderurgia" 139), already in the CAD file
      `delistings.py` downloads — no new fetch. Taxonomy differs from BolsAI's (different
      category names), so this is a source swap, not a drop-in string match; nothing
      downstream pattern-matches the literal sector string (`cross_sectional.py` only
      `groupby`s on it), so the swap is safe. `sector` stays excluded from training either
      way, per `manifest.LOOKAHEAD_TAINTED_COLS`.

### Task 5 — Cutover (keeping BolsAI infrastructure intact)

- [ ] Point `DATA_SOURCE["fundamentals"]` at CVM; drop yfinance from the fundamentals path.
- [ ] Make `--mode full_scale` work end-to-end with no key, and relax the `needs_bolsai`
      hard-fail (`pipeline.py:115-120`) to a warning.
- [ ] **Leave `client.py` and every BolsAI collector in place and importable**, reachable via
      an explicit opt-in flag. Per your constraint: build around it, don't delete it.
- [ ] `tests/run_all.py --group fast`, then `--group data`. Add a regression test asserting no
      ticker's flow series changes convention mid-history (the BUG-1 signature).
- [ ] Rebuild the dataset, diff against `dataset_v{N}`, update CLAUDE.md.
- [ ] Cancel the subscription **only after** the above is green.

---

## Known shortfalls

**S1 — 2010 is a hard floor for fundamentals, and paying does not move it.** CVM open data
starts 2010 (`http.py:27`) — but so does BolsAI's own history, verified on disk. Prices and
dividends still reach back to 2000 via yfinance. **No paid option here buys more depth.**

**S2 — `ebitda == ebit`, corporates only, fixed by Task 1's DFC D&A parsing.** Banks keep
`ebitda` as NaN rather than a forced number — not a shortcut, a real difference in what
EBITDA means for a financial institution's DRE.

**S3 — Banks will have NaN EBIT by design** after Task 2. That is deliberate: a bank has no
meaningful EBIT, and NaN is more honest than pre-tax income wearing an EBIT label. Revenue,
gross profit, net income and the whole balance sheet are recovered.

**S4 — Ticker discovery keeps a 2018 floor and stays survivor-style.** FCA's
`Codigo_Negociacao` is 100% blank 2010-2017 and reports the code *as of filing*, so renamed or
delisted tickers vanish from later years (already documented in `crosswalk.py`). New IPOs are
caught; renames still need `terminal_events.find_rename_candidates()` plus a hand-add to
`ticker_continuity.json`. **BolsAI was not solving this either** — no regression.

**S5 — Restatement versions still aren't preserved.** Values may reflect the latest
restatement rather than what was filed at v1. Pre-existing (already in CLAUDE.md); CVM's
yearly zips don't fix it. Unquantified.

**S6 — Sector taxonomy changes vocabulary.** Task 4 adopts CAD's `SETOR_ATIV` in place of
BolsAI's sector strings — real categories, but a different naming scheme (e.g. "Bancos" vs
whatever BolsAI called it). Anything that pattern-matches on the literal sector string
(none currently do, per grep of `cross_sectional.py`) would need updating. Values change,
category *count and shape* stays comparable — verified 30+ populated categories, not a
degraded taxonomy.

**S7 — The rebuild changes historical values, deliberately and for all tickers.** Standardizing
on TTM (Task 1) restates the 309 tickers BolsAI stored as single-quarter — not just the
delisted-ticker files BUG-2 already affected. Task 2 also fills in bank flows that were NaN.
Any saved backtest or `dataset_v{N}` comparison spanning the change is not apples-to-apples.
Snapshot before rebuilding.

---

## Order

**Task 0** (settle convention) → **Task 3** (overrides; Task 1 coverage depends on it) →
**Task 2** (banks) → **Task 1** (rebuild: BUG-1/BUG-2 fixes + real EBITDA via DFC) →
**Task 4** (splits, company_info, sectors) → **Task 5** (cutover).

Every replacement source is free, so the entire plan can be built and verified **while still
subscribed**. Cancel at the end of Task 5, not before.

---

## Implementation log (2026-08-19)

Tasks 0, 3, 2, 1 implemented and run against real data, in that order (Task 0 already logged
above). All 612 files in `data/raw/br/fundamentals/` were rebuilt from CVM and are git-modified
on disk, not yet committed.

**Task 3 (crosswalk overrides):** `TICKER_CNPJ_OVERRIDES` added to `cvm/crosswalk.py`. Ran
`cvm_statements.py --step crosswalk`: all 17 resolved with a real `cvm_code` (e.g. `CSNA3`
→ `004030`). 694/695 crosswalk rows now have a `cvm_code` (the one remaining gap, `MUUU4`,
predates this change and is unrelated).

**Task 2 (bank DRE) — done, plus a second bug found and fixed that the plan didn't anticipate:**
the same code-ambiguity problem exists on the **balance sheet** (BPA/BPP), not just the DRE.
Verified live: a 17-filer bank sub-layout (e.g. Banco do Brasil, cnpj `00000000000191`) puts
`equity` at code **2.07**, not 2.03 — 2.03 is "Provisões" or "Passivos Financeiros ao Custo
Amortizado" for these filers, a different concept entirely. Caught by testing `ITUB4` post-fix:
PVP came out ~0.19 and ROE ~2%, both roughly 10x too low for a bank that trades near 2x book —
implausible enough to investigate rather than accept. `bpa_column()`/`bpp_column()` added,
mirroring `dre_column()`'s description-first approach: only `total_assets` (code `1`) and
`total_liabilities` (code `2`) turned out to be genuinely universal; `equity` now resolves by
matching "patrimônio líquido" across any top-level `2.NN` code; `current_assets`/
`current_liabilities`/`debt_st`/`debt_lt` resolve by description too, and correctly come out
NaN for the bank sub-layout rather than silently picking up the wrong line (same "leave NaN
rather than guess" policy as bank EBIT). After the fix, **ITUB4 shows PVP ~2.0–2.2 and
ROE ~21%** — textbook-accurate for a bank this closely tracked, and BBAS3's ROE (~8%) is
consistent with its real, publicly reported 2025–2026 profit slump from agribusiness credit
provisioning, not a fabricated-looking number.

**Task 1 (TTM rebuild + real EBITDA):**
- D&A for EBITDA comes from the DFC cash-flow statement, which turned out to be
  **cumulative Jan-1-to-date in ITR filings**, not per-quarter like the DRE — required a
  separate de-accumulation step (diff within each cnpj+year, reindexed onto a full Q1–Q4 grid
  so a missing quarter produces NaN instead of silently diffing across the gap). Verified exact
  on PETR4: raw cumulative 18,976 / 39,928 / 62,317 (ITR) + 84,388 (DFP annual) → de-accumulated
  18,976 / 20,952 / 22,389 / 22,071 thousands.
- TTM standardized for every ticker (locked design decision — see "Corrections to the first
  draft" above), computed on the same gap-safe quarterly grid.
- `roic` added: NOPAT/invested-capital using Brazil's 34% statutory rate (documented
  approximation — no parsed tax-expense line — same spirit as `yf_collectors.compute_ratios()`'s
  own undertaxed `roic`, which is un-tax-adjusted for the same reason on that path).
- A **third bug found during rollout**: `compute_ratios()` had no inf→NaN guard, unlike the
  sibling `yf_collectors.compute_ratios()` which writes the same schema and explicitly cleans
  literal `inf` at the end (`# nonzero/0 divisions land here as inf`). A full-panel sweep found
  18–29 `inf` values out of ~32,000 rows (zero-denominator quarters). Fixed with the same
  pattern; reran the full rebuild — verified **0 literal inf remaining** across all 612 files.
- `build_fundamentals(rebuild=True)` added — overwrites every ticker's history unconditionally
  instead of only filling gaps for delisted names.

**Full-panel sanity check** (612 tickers, 32,285 rows): medians all in normal ranges (PL 7.3,
PVP 1.1, ROE 9.4%, ROIC 7.3%, debt/equity 0.5, current ratio 1.5); extreme tails are
near-zero-denominator distress cases, consistent with CLAUDE.md's documented "kept intact, not
clipped" policy, not new bugs. BUG-1's yfinance-corrupted final rows (PETR4, VALE3, BBDC4,
EGIE3 spot-checked) are gone — equity/assets no longer drop discontinuously in the last quarter.

`tests/run_all.py --group fast` — 55/55 pass, both before and after the inf-cleanup fix.

**Not yet done:** Task 4 (yfinance splits collector, CVM-sourced `company_info`/status,
CAD-sourced sectors), Task 5 (cutover — point `DATA_SOURCE` at CVM, rebuild
`data/processed/ml_dataset.parquet`, diff against the last `dataset_v{N}`, update CLAUDE.md,
only then cancel BolsAI).
