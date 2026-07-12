# Delisted-ticker universe via BolsAI — probe results (2026-07-11)

Addresses F1 (survivorship bias) in DIAGNOSIS_PLAN.md (`diagnosis` branch). The
"zero delisted companies — structural" conclusion in CLAUDE.md is **wrong for prices**:
BolsAI serves full price history for delisted tickers; it just doesn't link them in
company metadata.

## What the API actually has (verified with live calls)

| Endpoint | Delisted coverage |
|---|---|
| `/companies/?status=CANCELADA` | 1,894 companies, but only **3** have `ticker_primary` (BPAN4, MOAR3, PETZ3 — all delisted Jan 2026). No usable ticker link for the rest. |
| `/stocks/` (paginated) | **5,377 tickers including delisted.** ~1,060 BDRs; 1,539 stock-like (`[A-Z]{4}` + 3/4/5/6/11); **1,247 not in our current 294-ticker raw universe.** |
| `/stocks/{t}/history` | **Works for delisted.** History ends at the true delisting date: SMLS3 2021-06-04, HGTX3 2021-09-17, TIET11 2021-03-26, GNDI3 2022-02-11, LAME4 2022-01-21, SULA11 2022-12-23. |
| `/fundamentals/{t}/history` | **404 for delisted** (only the Jan-2026 trio still resolves: BPAN4 54q, MOAR3 60q). BolsAI cannot supply delisted fundamentals. |
| `/companies/{t}` | Mostly 404 for delisted; a few return stale `ATIVO` (BTOW3, CESP6, ELPL3, IGTA3). No reliable sector/status by ticker. |
| BDRs (~1,060) | Prices only; `/fundamentals/{t}/history` 404s for all sampled BDRs — structural (foreign issuers don't file with CVM), not a bug. |
| API bugs | HTTP 500 on `/stocks|/fundamentals` history for BTOW3, CESP6, ELPL3, IGTA3 — reproducible, server-side. Worth reporting to BolsAI. |

**CVM open-data spikes (2026-07-11, both PASS):**
- FCA `valor_mobiliario` register carries `Codigo_Negociacao` (ticker) + CNPJ per filer —
  resolved 3/3 delisted anchors (SMLS3, LAME4, HGTX3). This is the ticker↔cvm_code crosswalk
  BolsAI can't provide.
- FRE `capital_social` carries `Quantidade_Total_Acoes` per company/date → shares outstanding
  for market_cap/pl/pvp on delisted names. ITR/DFP zips confirmed to contain
  `DRE/BPA/BPP_{con,ind}` statement CSVs.

## Implications

- **Prices-side survivorship fix is available now and cheap.** ~1,000–1,250 extra
  tickers × 1–2 paginated calls (PRICE_LIMIT=5000 ≈ 20y/call) ≈ ~2,500 calls ≈ **€0.25**.
- **Fundamentals for delisted must come from CVM open data** (DFP/ITR, free/keyless).
  We already consume that portal: `filing_dates.py` covers 1,223 companies — far more
  than the 293 ATIVO — so delisted filings are already flowing through that pipeline.
  Computing ratios from raw statements is a real project (Stage 1.5), not a flag flip.
- **Sector metadata for delisted:** the CANCELADA list has sector + cvm_code + cnpj;
  the missing piece is ticker↔cvm_code mapping, recoverable from CVM open data.
- **The 1,247 "new" tickers are not all delisted companies.** The list includes
  second share-classes of companies we already hold (ALPA3 vs ALPA4, ALUP3/4/11),
  renames/continuations (VVAR3→VIIA3, BRDT3→VBBR3, DTEX3→DXCO3), and mergers
  (LAME4/BTOW3→AMER3, GNDI3→HAPV3). Renames are NOT survivorship cases — including
  both legs double-counts; a rename/merger map is required before backfill.

Regenerate the raw lists anytime: paginate `/stocks/` (`limit=500`) and
`/companies/?status=CANCELADA` (`limit=50`); this probe cost ~50 calls.

## Plan (approved 2026-07-11; every task has a Goal and a runnable Test)

### 1a. Delisted prices — CODE DONE, collection pending
**Goal:** every stock-like `/stocks/` ticker not in `data/raw/prices/` gets a parquet ending
at its true last-trade date.
**Test:** `tests/data_collection/test_collect_delisted.py` — pure candidate-filter check +
delisting-date anchors (SMLS3 2021-06-04, LAME4 2022-01-21, HGTX3 2021-09-17, ±7 days).
- [x] `src/data_collection/collect_delisted.py` — enumerates candidates, calls
  `collect_prices()` directly (bypasses pipeline's ATIVO gate, which was the root cause of
  delisted being unreachable). Suffix-11 units gated on FCA crosswalk to exclude FIIs/ETFs.
- [ ] Run it: `python -m src.data_collection.collect_delisted` (~1,250 tickers ≈ €0.25)
- [ ] Anchors in `test_collect_delisted.py` go from SKIP to PASS

### 1b. Delisted fundamentals from CVM — CODE DONE, collection pending
**Goal:** every delisted ticker with prices gets `data/raw/fundamentals/{t}.parquet` in the
exact BolsAI schema (conventions verified live on BPAN4: single-quarter flows, thousands),
so `filter_tickers_with_no_fundamentals()` stops dropping it. BolsAI files never overwritten.
**Test:** `tests/data_collection/test_cvm_statements.py` — synthetic ratio-math check
(always runs) + CVM-vs-BolsAI cross-source check on overlapping tickers (15% tolerance,
SKIPs until caches exist).
- [x] `src/data_collection/cvm_statements.py`: FCA crosswalk (spike PASS 3/3), DFP/ITR
  DRE+BPA+BPP parser (con>ind, ITR quarterly-only rows, Q4 = DFP annual − interim sum),
  FRE share counts, BolsAI-schema ratios, per-ticker parquet via `_merge_save`.
- [ ] Run steps: `python -m src.data_collection.cvm_statements` (downloads ~2×16 years of
  CVM zips — slow but free; cached under `data/raw/cvm/`, idempotent)
- [ ] Cross-source test goes from SKIP to PASS
- Known ceilings (ponytail-commented in code): ebitda==ebit (no DFC parsing); bank DRE
  layout → NaN flow columns (same gap BolsAI has); shares = total across classes.

### 1c. Delisted company_info (sector/cvm_code) — CODE DONE, run pending
**Goal:** every crosswalk-resolved delisted ticker has a `company_info.parquet` row,
`status="CANCELADA"`, non-null sector — automatically excluded from `--mode update`.
**Test:** zero null sector/cvm_code among backfilled delisted; CANCELADA set disjoint from
`_active_tickers()`.
- [x] `cvm_statements.synthesize_company_info()` — CANCELADA registry (sector, cvm_code;
  ticker-less on BolsAI) joined to tickers via crosswalk on CNPJ.
- [ ] Run: `python -m src.data_collection.cvm_statements --step company_info`

### P2. Ticker renames & mergers — DONE (map grows by hand)
**Goal:** expanded universe never double-counts a company across a rename, never loses
pre-rename history of a surviving entity.
**Test:** `tests/build_dataset/test_ticker_continuity.py` (fast group) — rename splice,
merger ratio splice, duplicate-date guard, missing-map no-op.
- [x] `data/raw/reference/ticker_continuity.json` — 6 verified 1:1 renames seeded, incl.
  VVAR3→VIIA3→BHIA3 chain. Policy: only verified ratios enter; absent = two clean series.
- [x] `apply_ticker_continuity()` in `build_ml_dataset.py`, called after split repair
  (so ratios don't look like unrepaired splits). Renames splice prices+fundamentals;
  mergers splice prices only. Boundary = new ticker's actual first trade date.
- [ ] Verify LAME3/LAME4→AMER3 exchange ratios, then add as `merger` entries

### P3. Multi-share-class awareness — DONE
**Goal:** callers can group tickers by company (PETR3/PETR4 → one firm).
**Test:** `tests/build_dataset/test_company_siblings.py` (fast group).
- [x] `company_siblings()` in `build_ml_dataset.py` — `dict[cvm_code, [tickers]]`.

### 1d. Terminal events — deferred (needs P2 event types + 1a-1c data)
- [ ] Final return leg per delisting: tender price / merger ratio / zero for bankruptcy.
  Until then delisted series just stop — bias shrinks but doesn't vanish.

### Full-loop regression (after collection runs)
- [ ] `python -m src.build_dataset.build_ml_dataset` → rebuild
- [ ] `python tests/run_all.py --group all` → all green, `dataset_v{N}` bumps,
  manifest reflects the larger universe
- [ ] Re-run F1 analysis (`diagnosis` branch) on the survivorship-free universe
- [ ] Report the HTTP 500 tickers (BTOW3, CESP6, ELPL3, IGTA3) to BolsAI
- [ ] Fix the stale CLAUDE.md claim ("zero delisted companies — structural")
