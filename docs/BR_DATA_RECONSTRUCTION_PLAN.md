# BR Data Reconstruction Plan

Spec-driven rebuild of `data/raw/br/` after the 2026-08-23 recollection (`3f58753`)
regressed the universe from 1,328 to 383 price files.

Status: **P0–P1b done. P2 (live collection, free) next — needs your go-ahead.**
Decisions locked 2026-08-25 (see §5).

---

## 1. How the pipeline actually works

### 1.1 The one command

```bash
python -m src.data_collection.br.pipeline --mode full_scale
```

Executes in this order (`src/data_collection/br/pipeline.py`):

| # | Step | Source | Writes | Cost |
|---|------|--------|--------|------|
| 0 | `tickers = collectors.get_all_tickers()` | **live BolsAI `/stocks/`** when a key is set (else `_active_tickers()`) | — | ~11 calls |
| 1 | stage `macro` | BCB SGS (`selic=11`, `cdi=12`, `ipca=433`) | `macro/{selic,cdi,ipca}.parquet` | free |
| 2 | stage `company_info` | CVM CAD + FCA crosswalk | **overwrites** `company_info/company_info.parquet` | free |
| 3 | stage `sectors` | CVM | `company_info/sectors.parquet` | free |
| 4 | stage `corporate_events` | yfinance `Ticker.splits` | `corporate_events/corporate_events.parquet` | free |
| 5 | `active = tickers ∩ _active_tickers()` | ← **DEFECT-1** | — | — |
| 6 | stage `prices` | yfinance (`DATA_SOURCE["prices"]`) | `prices/*.parquet` | free |
| 7 | stage `fundamentals` | CVM (`DATA_SOURCE["fundamentals"]`) | `fundamentals/*.parquet` | free |
| 8 | stage `dividends` | yfinance (`DATA_SOURCE["dividends"]`) | `dividends/*.parquet` | free |

Step 0 enumerates BolsAI's whole universe; step 5 intersects it with CVM's ATIVO set and keeps only
the overlap. Measured live 2026-08-25 — the numbers reconcile exactly with the run log:

```
/stocks/ raw universe                    5,377
get_all_tickers()  (regex-filtered)      1,449   ← step 0, matches the log's "1449 tickers"
CVM ATIVO (company_info)                   564
  ∩  intersection                          490   ← matches the log's "filtered to 490/1449"
  BolsAI-only, discarded                   959   ← the tier-1/2 delisted universe, thrown away
  CVM-ATIVO-only, never requested           74   ← 13 units + recent IPOs BolsAI's registry lacks
```

### 1.5 What builds `company_info.parquet`

`cvm/company_info.py`'s `synthesize_company_info()`, wired **hardcoded** at `pipeline.py:144` — it
is *not* routed through `DATA_SOURCE` (which only has `prices`/`fundamentals`/`dividends` keys), so
BolsAI's `collectors.collect_company_info()` is dead code on this path. It composes three CVM
sources and is **incremental, never a rebuild**:

1. `build_crosswalk()` → FCA: `ticker, cnpj, corporate_name, cvm_code` (2018+ only, survivor-style)
2. `build_delist_events()` → CAD registry: `ticker, cnpj, delist_date, motivo_cancel, sit`
3. `sector_by_ticker()` → CVM `SETOR_ATIV`
4. reads the **existing** parquet, refreshes `status` + `sector` in place, appends unseen tickers
   from (2), then `drop_duplicates("ticker", keep="last")`

Two consequences that matter here:

- **It never drops a ticker.** Rows only accumulate. So the file's 691 rows (vs. 694 before the
  recollection) means the file itself was wiped alongside the price files — had it survived, every
  old ticker *and its CNPJ* would still be there. That CNPJ column is exactly what §7 needs.
- **Reactivation guard.** CVM's `SIT` is CNPJ-level, so a retired code whose company still trades
  reads ATIVO. A non-ATIVO → ATIVO flip is only trusted when that ticker's own price file has traded
  within `_REACTIVATION_STALE_DAYS = 120`. This is why CIEL3/AZUL4/ELET3 still read ATIVO — with no
  price file on disk, `_recently_traded()` returns False, but the guard only blocks *reactivations*,
  not rows that were already ATIVO.

### 1.2 What this command does NOT do

Three entrypoints exist that `pipeline.py` never calls. Skipping them is what makes a
routine run silently survivor-only:

```bash
python -m src.data_collection.br.collect_delisted   # BolsAI prices for dead tickers  [PAID]
python -m src.data_collection.br.cvm_statements     # CVM fundamentals for dead tickers [free]
python -m src.build_dataset.terminal_events         # Stage 2: realized payoff at death [free]
```

### 1.3 Why "dead" tickers vanish — three distinct tiers

CVM's registry status is **company-level (CNPJ), not ticker-level**, so it cannot reliably
say a ticker *code* is retired. Of the 949 price files deleted in `3f58753`:

| Tier | CVM says | Pipeline does | yfinance does | Count | Example |
|------|----------|---------------|---------------|-------|---------|
| 1 | ticker absent from registry | never in the list | — | **715** | KROT3, ABYA3 |
| 2 | `CANCELADA` | ATIVO gate skips it | would 404 anyway | 104 | GOLL4, LINX3, SULA11 |
| 2b | `SUSPENSO` | ATIVO gate skips it | would 404 anyway | 8 | — |
| 3 | `ATIVO` (company alive, code retired) | **tries it** | **404 →** `empty_runs: 1` | 122 | CIEL3, AZUL4, ELET3, STBP3 |

Tier 3 splits again: **renames** (ELET3→AXIA3, VVAR3→BHIA3, BTOW3→AMER3) keep full history
under the new symbol on yfinance and only need a `ticker_continuity.json` entry.
**Genuine delistings** (CIEL3, AZUL4, STBP3, ODPV3) do not.

CVM's FCA `Codigo_Negociacao` is only populated from 2018 on and is survivor-style, so
**tier 1 is unreachable by any free source.** Only BolsAI's `/stocks/` registry can name them.

### 1.4 Full flow, end to end

```
   BolsAI /stocks/          CVM FCA+CAD           BCB SGS         yfinance
   (tier-1 discovery)       (roster/status)       (macro)         (prices/divs/splits)
          │                       │                   │                  │
          └────────┬──────────────┘                   │                  │
                   ▼                                  ▼                  ▼
            company_info.parquet ─────────────► data/raw/br/{macro,prices,dividends,
                   │                                          corporate_events}/
                   ▼                                             │
            CVM statements+shares ──► fundamentals/*.parquet ─────┤
                                                                  ▼
                                              build_ml_dataset.py (Stage 2)
                                       repair splits → splice continuity → merge_asof
                                       → features → clean → ml_dataset.parquet
                                                                  ▼
                                              terminal_events.py (realized death payoff)
                                                                  ▼
                                              scale_features.py (train-only scaler)
```

---

## 2. The four defects

**DEFECT-1a — the ATIVO intersection discards BolsAI's own discovery.** `pipeline.py:~159` keeps
only `get_all_tickers() ∩ CVM-ATIVO`. The 959 BolsAI-known tickers dropped there *are* the tier-1/2
delisted universe — the exact set P4 pays BolsAI to rediscover. The pipeline already enumerates
them live and throws them away one line later, handing them to nothing.

**DEFECT-1b — `get_all_tickers()`'s regex drops stock units.** `^[A-Z0-9]{4}[3-8]$`
(`br/collectors.py:75`) excludes every suffix-11 ticker because FIIs/ETFs share that suffix, and
only a hand-maintained `KNOWN_UNIT_TICKERS = {"BOVA11", "BPAC11"}` is added back. 117 suffix-11
names exist in the raw universe; **13 ATIVO operating-company units are silently dropped**:
ALUP11, BMGB11, BRBI11, ENGI11, IGTI11, JSLG11, KLBN11, RNEW11, SANB11, SAPR11, STBP11, TAEE11,
VVAR11 — the *liquid* instrument for those companies, all served fine by yfinance (verified live:
TAEE11 = 4,661 rows back to 2007). This is why BPAC11 and BOVA11 survived and the rest did not.

**DEFECT-2 — fundamentals scoped to ATIVO.** `pipeline.py:~172` passes `active` to the
fundamentals stage, so delisted names are stranded (612 → 382 files). CVM statements are keyed
by CNPJ and *do* contain delisted companies — the source is fine, the gate is the bug.

**DEFECT-3 — no delisted collection was run.** §1.2's three commands were never invoked after
the recollection, so tiers 1/2/2b are simply absent from disk. This is the survivorship bias.

**DEFECT-4 — no coverage floor anywhere in `tests/`.** A 949-file drop passed silently; grep for
file-count or ticker-count assertions returns nothing. `test_br_data_quality.py` caught the
*price-jump rate* regression (14.36% vs a 12% ceiling) but nothing caught the universe collapse.

Not a defect — resolved by decision D1: 77 unrepairable >10× price artifacts across 28 tickers
(SUZB3, CPFE3, BRKM5, AZEV3 …), all in 2000–2010 deep history that yfinance newly supplies and
`corporate_events` has no matching entry for. Flooring the panel at 2010 removes the entire class.

---

## 3. The specs (write these first — all fail today)

Two new files. Each spec pins one defect and is fixed one at a time.

`tests/data_collection/test_universe_derivation.py` — **FAST** group, synthetic:

- [x] **S1** `test_full_scale_prices_include_already_collected_non_active_tickers` +
  `test_get_all_tickers_keeps_crosswalk_confirmed_units` — DEFECT-1a: a full_scale run must extend
  `_recover_stale_company_info_tickers` beyond `mode == "update"`. DEFECT-1b: `get_all_tickers()`
  must resolve suffix-11 units via the FCA crosswalk, not the hand-maintained `KNOWN_UNIT_TICKERS`.
  *Confirmed failing 2026-08-25* — AMER3 dropped from the prices call; TAEE11 dropped from
  `get_all_tickers()`'s result.

`tests/data_collection/test_universe_coverage.py` — **DATA** group, real `data/raw/br/`:

- [x] **S2** `test_every_active_ticker_has_prices` — `status == ATIVO` ⇒ a price file exists, or the
  ticker is in an explicit `NO_YF_COVERAGE` set with a reason.
  *Confirmed failing 2026-08-25:* 182 missing.
- [x] **S3** `test_units_are_collected` — every ATIVO `*11` unit has a price file.
  *Confirmed failing 2026-08-25:* 23 missing (not 11 — recount at write time).
- [x] **S4** `test_panel_contains_dead_tickers` — at least 700 tickers whose last price date is
  more than 90 days stale (i.e. the panel is not survivor-only).
  *Confirmed failing 2026-08-25:* 0 — every file is current.
- [x] **S5** `test_crosswalk_tickers_have_fundamentals` — every FCA crosswalk ticker has a
  fundamentals file (not gated on having a price file too — DEFECT-2 is about the fundamentals
  stage's own scoping, independent of what the prices stage did).
  *Confirmed failing 2026-08-25:* 309/691 (44.7%) missing, ceiling is 10%.
- [x] **S6** `test_no_silent_universe_collapse` — hard floors: prices ≥ 1200, fundamentals ≥ 550,
  dividends ≥ 300. Bumped deliberately, never automatically.
  *Confirmed failing 2026-08-25:* 383 / 382 (dividends 335 already clears its floor).

Both files use the repo's newer convention: real `test_*` functions with bare `assert`, ending in
`raise SystemExit(pytest.main([__file__]))`. Register in `tests/run_all.py`'s `FAST`/`DATA` lists —
`roster_drift()` fails the suite otherwise.

*Skipped:* a baseline-JSON coverage tracker. S6's three constants do the same job in three lines.

---

## 4. Execution order

Each phase ends green on its spec before the next starts.

- [x] **P0 — Write S1–S6, then clear interfering state.** Done 2026-08-25. Six specs across two
      files (`tests/data_collection/test_universe_derivation.py` FAST,
      `tests/data_collection/test_universe_coverage.py` DATA), registered in `tests/run_all.py`,
      confirmed to fail for the stated reasons (see §3). Then the hygiene pass:
      ```bash
      rm data/processed/ml_dataset.tmp.parquet          # 466 MB, interrupted-rebuild leftover — done
      rm -rf artifacts/checkpoints/full_scale           # BR: negative cache + skip state — done
      rm -rf artifacts/checkpoints/speed_test           # orphaned mode, never re-run — done
      # KEPT: artifacts/checkpoints/{us_*,prototype,update}, data/raw/br/cvm/* — see §6
      ```
      `prototype`/`update` BR checkpoints were left in place — out of scope (checkpoints are
      keyed per-mode, so they cannot affect a `full_scale` run) and not listed in §6.

- [x] **P1a — Fix DEFECT-1b (units) → S1 unit test green.** Done 2026-08-25.
      `br/collectors.py`: deleted `KNOWN_UNIT_TICKERS`; `get_all_tickers()` now resolves suffix-11
      via the FCA crosswalk, the same test `collect_delisted.py:36` already used. `import re` moved
      to module level (was function-local). BOVA11 needed no special case — `pipeline.run()` already
      unions `config.BENCHMARK_TICKERS` in unconditionally, and it isn't a real crosswalk-resolvable
      company anyway (verified: absent from `fca_crosswalk.parquet`). BPAC11 needed no special case
      either — it *is* crosswalk-resolvable, so the new logic picks it up on its own merits, not by name.

- [x] **P1b — Fix DEFECT-1a (discarded universe) → S1 unit test green.** Done 2026-08-25.
      `br/pipeline.py`: `_recover_stale_company_info_tickers` now runs in every mode, not just
      `update` — deleted the `if mode == "update":` guard. Existing behavior for `update` and for
      brand-new tickers is unchanged (regression-checked: both pre-existing
      `test_recover_stale_company_info_tickers_*` cases in `test_pipeline_dispatch.py` still pass).
      Fundamentals remain ungated, unaffected by this change — that gap is DEFECT-2 / P3.
      Full FAST suite run: 60/61 pass; the one failure (`test_sec_unit_currency.py`, US/SEC
      JPY-vs-USD handling) is unrelated to this plan and pre-existing.

- [ ] **P2 — Re-run collection → S2, S3 green.**
      `python -m src.data_collection.br.pipeline --mode full_scale`
      Recovers the 80 never-attempted tickers incl. all units. Free. Tier-3 dead codes will still
      404 — those go in `NO_YF_COVERAGE` with a reason, they are not failures.

- [ ] **P3 — Fix DEFECT-2 → S5 green.** Make `--mode full_scale` rebuild the full crosswalk
      (`build_fundamentals(tickers=None, rebuild=True)`) rather than scoping to `active`.
      `full_scale` already means "everything". Free, ~1 line.

- [ ] **P4 — Fix DEFECT-3 → S4 green.** The tools already exist; run them:
      ```bash
      python -m src.data_collection.br.cvm_statements --step crosswalk   # units need this first
      python -m src.data_collection.br.collect_delisted --dry-run        # review the list
      python -m src.data_collection.br.collect_delisted                  # PAID — tier 1+2 prices
      python -m src.data_collection.br.cvm_statements                    # delisted fundamentals
      ```
      *No new restore script.* `collect_delisted.py` already enumerates BolsAI's `/stocks/`
      universe, filters to tickers with no price file on disk, and bypasses the ATIVO gate —
      exactly the tier-1 discovery path nothing else can provide.

      **Also run `collect_company_info` (BolsAI) to harvest CNPJs** — §7 explains why (715 of the
      949 deleted tickers have no CNPJ anywhere, which blocks CVM fundamentals *and* CVM dividends,
      not just prices).

      ⚠ **Guard against BolsAI's rename phantom (§9).** `candidate_tickers()` selects on "no price
      file on disk", so retired codes like KROT3 and ELET3 *are* candidates — and BolsAI serves
      them live data through today under the dead symbol. Collecting them unguarded double-counts
      a company already in the panel under its successor. After collection, truncate each restored
      ticker at its real last trading date (cross-check against the successor via
      `terminal_events.find_rename_candidates()`), or exclude any candidate whose returned series
      extends past its known rename date.

      ⚠ **The harvested CNPJs must be unioned into `fca_crosswalk.parquet`, not left in
      `company_info`.** `cvm/ratios.py:311`'s `build_fundamentals` iterates `CROSSWALK_PATH` and
      never reads `company_info` — and `build_crosswalk()` rebuilds that file from FCA alone
      (2018+, survivor-style). Without the union, the paid CNPJ harvest buys nothing: tier-1 names
      get prices, then get dropped by `filter_tickers_with_no_fundamentals`. Make the union
      additive and idempotent — FCA wins on conflict, BolsAI fills only what FCA cannot know.

- [ ] **P4b — CVM dividends/JCP for delisted names (free).** New `cvm/dividends.py`, ~80 lines,
      reusing `http.fetch_zip` + the crosswalk + `shares.py`'s pattern. Source and limits in §7.
      Write into `dividends/` in the same schema `collect_dividends_yf` produces, so Stage 2 needs
      no change. yfinance stays authoritative where it has coverage; CVM fills dead names and
      pre-2010 gaps only.
      *Optional, low value:* `fre_cia_aberta_capital_social_desdobramento` → `corporate_events`.
      Same file shape already (`ticker, date, ratio_from, ratio_to, factor`), ~20 lines, but
      coverage is thin (127–207 CNPJs/yr) and it dies after 2022. Skip unless a split gap bites.

- [ ] **P5 — Fix DEFECT-4 → S6 green.** Floors should now pass on their own. Commit them.

- [ ] **P6 — Floor the panel at 2010.** Add `min_date=None` to `loaders.load_prices()` and pass
      `2010-01-01` from `build_ml_dataset.py:363` (the single BR call site; US always passes
      `dir=US_PRICES_DIR`, so it is unaffected). Deletes the 77-artifact problem outright —
      no quarantine list, no jump-driven repair.

- [ ] **P7 — Rebuild and close the loop.**
      ```bash
      python -m src.build_dataset.build_ml_dataset
      python -m src.build_dataset.terminal_events
      ```
      Then run `terminal_events.find_rename_candidates()` — it already reports tier-3 renames
      (ELET3→AXIA3, VVAR3→BHIA3, BTOW3→AMER3 …). Hand-add each to `ticker_continuity.json`
      (never auto-applied, by design), rebuild once more, then:
      ```bash
      python -m src.build_dataset.scale_features
      python tests/run_all.py --group all
      ```

- [ ] **P8 — Update `CLAUDE.md`.** Its "Data on Disk" counts (1,328 / 612) and the BolsAI
      `adj_close` caveats are stale: yfinance shows 0 non-positive `adj_close` (was 288) and
      4.4% two-decimal rows (was 33%).

---

## 5. Decisions locked (2026-08-25)

**D1 — Panel floors at 2010.** Matches the CVM fundamentals floor (2010-12-31), so pre-2010 rows
carry no fundamentals anyway. Removes 77 unrepairable >10× artifacts without writing repair code.
Cost: loses price-only 2000–2010 history (technical features and long beta windows only).
Reversible — it is one argument at one call site.

**D2 — BolsAI stays, scoped to tier-1 discovery only.** Key is live and spend is approved.
yfinance/CVM remain primary for everything else; BolsAI is called *only* by `collect_delisted.py`
for names no free source can produce. Justified by measurement — on the 379 overlapping tickers
yfinance beats BolsAI on every axis:

| | BolsAI | yfinance |
|---|---|---|
| Rows | 1,391,762 | **1,732,070** (1.24×) |
| Big jumps (>50%), same window | 2,529 | **1,062** |
| Non-positive `adj_close` | 288 | **0** |
| Rows pinned to 2 decimals | 33% | **4.4%** |

---

## 6. State and caches that interfere with a re-run

### Delete before P2

| Path | Size | Why it must go |
|------|------|----------------|
| `artifacts/checkpoints/full_scale/yf_prices.json` | 29 KB | **The dangerous one.** Holds the negative cache: 102 tickers at `empty_runs: 1`. `EMPTY_RUNS_SKIP_THRESHOLD = 3`, so two more runs and each goes dark for 10 runs — no request, no retry. Includes tier-3 renames we specifically want to re-probe. |
| `artifacts/checkpoints/full_scale/yf_dividends.json` | 30 KB | Per-ticker `checked_through` — a ground-up run must re-walk full dividend history, not resume. |
| `artifacts/checkpoints/full_scale/macro.json` | 249 B | "macro trusts stale checkpoint after wipe" was fixed in `6c937df`, so this is belt-and-braces. Refetching BCB is seconds. |
| `artifacts/checkpoints/speed_test/` | 21 KB | Orphaned mode, never re-run. |
| `data/processed/ml_dataset.tmp.parquet` | 488 MB | Interrupted-rebuild leftover. |

### Keep — deleting these is expensive and buys nothing

| Path | Size | Why |
|------|------|-----|
| `artifacts/checkpoints/us_*` | 1.5 MB | US collection state, untouched by a BR rebuild. Deleting forces a full ~10k-ticker US re-collect. |
| `data/raw/br/cvm/stmt_{dfp,itr}_*.parquet` | 35 files | Parsed CVM statement years, CNPJ-keyed and immutable. Not ticker-scoped, so P3's scoping fix does **not** invalidate them. |
| `data/raw/br/cvm/{fca,fre}_*.parquet` | 26 files | Parsed FCA/FRE years. Re-downloading is ~50 yearly zips at ~10 MB each. |

Delete a `cvm/` cache **only** when its parser changes. P4b adds a *new* FRE read
(`distribuicao_dividendos_classe_acao`), which is a different member of the same zip — the cached
`fre_{year}.parquet` files are `shares.py`'s already-parsed output, not the raw zip, so P4b
re-downloads regardless. No deletion needed.

### Why the price checkpoint is safe to delete

`_prices_fetch_start` (`yf/_common.py:105`) derives the fetch window from the **parquet file**, not
the checkpoint: it finds rows where `num_trades` is NaN (the yfinance-sourced marker) and refetches
from the *earliest* such row so the whole yfinance era stays internally consistent. The checkpoint's
`last_date` is only consulted via `_seed_last_date` when no yfinance rows exist. So the file is the
source of truth and the checkpoint is a cache — except for `empty_runs`, which lives *only* in the
checkpoint. Deleting it loses nothing but the negative cache, which is exactly the intent.

### One thing this buys for free

`_bolsai_junction_date` + `_reconcile_yfinance_junction` (`yf/prices.py:210`) already detect a
BolsAI→yfinance boundary in a mixed file and rescale the yfinance segment to match. So P4's
restore-then-top-up is a **supported path, not a hack** — a restored BolsAI file that yfinance can
still serve gets a reconciled continuation rather than a discontinuity. For tier-1/tier-2 names
yfinance 404s anyway, so it is moot there.

---

## 7. CVM dividends/JCP and corporate events (measured live 2026-08-25)

CVM **does** publish dividends and splits, and they **do** cover delisted companies. Three sources,
all free, all keyless, all through the existing `http.fetch_zip`:

| Source | Window | Coverage | Gives |
|--------|--------|----------|-------|
| FRE `distribuicao_dividendos_classe_acao` | **2010–2022** (retired in FRE 2024+) | 736 CNPJs, 38,669 rows | share class (`Especie_Acao`/`Classe_Acao`), type (Dividendo Obrigatório / JCP / Outros), `Montante` in BRL, **payment date** |
| DMPL (`DFP`/`ITR`) | **2010–2025**, continuous | 342–474 CNPJs/yr, ~14k dividend+JCP rows/yr | total charged to equity — no class, no date, no per-share |
| FRE `capital_social_desdobramento` | 2010–2022 | 127–207 CNPJs/yr (thin) | approval date + share counts before/after → ratio derivable |

**Delisted coverage verified.** 20 of 23 dead/renamed tickers tested were present in FRE dividends:
LINX3, HGTX3, SULA11, BIDI11, TIET11, GNDI3, SMLS3, LAME4, BTOW3, ESTC3, CESP6, BRML3, IGTA3,
CCRO3, ELET3, VVAR3, GOLL4, CIEL3, ODPV3, MRFG3. The three misses (AZUL4, STBP3, BRFS3) are
non- or low-dividend payers, so zero rows is likely correct rather than a gap.

**Why it is worth doing.** Dead tickers currently get `has_dividends = 0`, which reads as "paid
nothing" when it means "not collected" — a feature-level survivorship artifact that makes dead
companies look systematically like non-payers in `div_yield` / `payout_ratio`.

**The binding constraint is the ticker↔CNPJ map, not the dividend data.** Of the 949 deleted price
tickers:

- 234 have a CNPJ (from BolsAI's old `/companies/` registry, cached in the pre-wipe `company_info`)
- 192 of those 234 are present in FRE dividends
- **715 have no CNPJ anywhere** — unmappable, so CVM cannot attach anything to them

That is the *same* blocker as tier-1 prices, and it has the same fix: BolsAI's registry is the only
source of those CNPJs. **P4 should therefore also run `collect_company_info` to harvest CNPJs**, not
just prices — every CNPJ recovered unlocks that ticker's CVM dividends *and* its CVM fundamentals
for free.

### Limits to respect

- **Payment date, not ex-date.** Fine for `div_yield` / `payout_ratio` / `dividend_coverage_ratio`,
  which are trailing-12m sums. Typically 30–90 days later than the ex-date, so **not** suitable for
  an event-study signal keyed to the price drop.
- **Total BRL, not per-share.** `Montante` is the amount for the whole class; per-share needs
  ÷ class share count from `fre_cia_aberta_capital_social_classe_acao` (which does survive into
  2025, unlike the dividend table). Note `shares.py` currently reads only `Quantidade_Total_Acoes`
  from the non-class table — the per-class file is a separate read.
- **2023 is a hole.** FRE's dividend table collapses to 9 CNPJs in 2023 and is absent from 2024+.
  DMPL covers 2023–2025 but without class or per-share, so it can only sanity-check a total.
  yfinance already covers 2023+ for anything still listed, and anything delisted before 2023 is
  fully inside FRE's good window — so the hole mostly affects names that died in 2023–2025.
- **Returns are unaffected either way.** yfinance's `adj_close` already bakes in reinvestment, so
  the dividends table only ever feeds features, never the return series.

---

## 8. Known limits after this plan

- **Tier-1 fundamentals and dividends may have no ticker to attach to.** CVM is CNPJ-keyed
  throughout; the free ticker↔CNPJ map (FCA crosswalk) only starts 2018. A tier-1 name recovered
  from BolsAI prices without a CNPJ gets neither fundamentals nor dividends, and will be dropped by
  `quality_filters.filter_tickers_with_no_fundamentals`. Harvesting CNPJs in P4 is what shrinks this.
- **Splits before 2004 stay thin** — `corporate_events` has 3 recorded events in 2001 and 3 in 2002,
  market-wide. Moot under D1 (panel floors at 2010).
- **New IPOs still need a BolsAI key to discover** (`get_all_tickers()`), unchanged from before.

---

## 9. "Why not just use BolsAI for everything?" (probed live 2026-08-25)

Asked and tested directly against the API rather than inferred from docs.

### What BolsAI genuinely does better — the reason it stays in the plan

Delisted price and fundamentals history, with correct terminal dates:

| Ticker | Prices | Span | Fundamentals |
|--------|-------:|------|-------------:|
| HGTX3 | 3,997 | 1999-07-29 → 2021-09-17 | — |
| SULA11 | 3,764 | 2007-10-05 → 2022-12-23 | 48 q → 2022-09-30 |
| BTOW3 | 3,823 | 2007-08-08 → 2023-01-19 | — |
| CIEL3 | 3,638 | 2009-12-18 → 2024-08-26 | 63 q |
| LINX3 | 2,072 | 2013-02-08 → 2021-06-25 | 38 q → 2021-03-31 |
| AZUL4 | 2,016 | 2017-04-11 → 2025-05-28 | 39 q |
| TIET11 | 1,294 | 2016-01-04 → 2021-03-26 | — |
| BIDI11 | 732 | 2018-04-30 → 2022-06-17 | — |

No free source produces any of this. That is the whole case, and it is a good one.

### Why "all BolsAI" is not simpler in the way that matters

**Three of the four defects are not vendor problems.** DEFECT-1 (stale universe intersection) and
DEFECT-2 (fundamentals scoped to ATIVO) are code bugs in `pipeline.py`; DEFECT-4 is a missing test.
Switching vendors leaves all three exactly as they are — the same P1/P3/P5 work is still required.

**Price quality regresses, measured.** On the 379 tickers present in both trees: 2,529 big jumps vs
yfinance's 1,062 in the same window, 288 non-positive `adj_close` vs 0, 33% of rows pinned to two
decimals vs 4.4%, and 24% fewer rows.

**Two fixed fundamentals bugs come back.** BUG-1 (equity/total_assets dropping ~5× with no event)
and the TTM-vs-point-in-time per-ticker coin flip that corrupts cross-sectional comparison. Both
regression-tested in `tests/data_collection/test_cvm_statements.py`; both are why CVM replaced it.

**A new bug arrives: the rename phantom.** BolsAI resolves a retired code to the *live* entity and
serves current data under the dead symbol, unflagged:

- `/fundamentals/KROT3/history` and `/fundamentals/COGN3/history` return **byte-identical** rows
  through 2026-06-30 (`lpa=0.32, vpa=6.65, shares=2064266831` on both). KROT3 was renamed in 2019.
- `/stocks/KROT3/history` returns prices through **2026-08-24**; `/stocks/ELET3/history` likewise,
  despite the AXIA3 rename.

Collecting both codes double-counts one company — inflating the universe, corrupting every
sector- and market-relative feature, and duplicating a return stream in the optimizer. This is the
opposite failure from survivorship bias and just as damaging. Hence P4's truncation guard.

**Dividends are annual summaries, not a payment series.** `/dividends/{ticker}` returns
`{dividend_yield_ttm, ttm_per_share, current_price, total_payments, annual_summary:[{year,
total_per_share, payments}]}` — no ex-date, no pay-date, no per-payment rows, and `current_price`
is a stale snapshot (37.4 for LINX3, delisted 2021). yfinance gives real per-payment records; CVM
FRE gives per-class amounts with payment dates. BolsAI gives neither.

**Macro is out of scope for it regardless** — SELIC/CDI/IPCA come from BCB either way.

### Verdict

All-BolsAI is fewer moving parts and more wrong data, and it still leaves P1/P3/P5 to write. The
plan's split is not complexity for its own sake: it is one extra call path
(`collect_delisted.py`, which already exists) used only where BolsAI is the sole option.
