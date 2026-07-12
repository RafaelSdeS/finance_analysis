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
| API bugs | HTTP 500 on `/stocks|/fundamentals` history for BTOW3, CESP6, ELPL3, IGTA3 — reproducible, server-side. Worth reporting to BolsAI. |

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

## Plan

- [ ] Build ticker classification for the 1,247: share-class of existing | rename/merger continuation | true delisting | other asset type (cross-ref CVM registry + corporate events)
- [ ] Decide rename policy (splice VVAR3+VIIA3 into one series vs keep the surviving leg only)
- [ ] Backfill prices for true delistings via existing `collect_prices()` (works as-is — it takes a ticker list; est. ~€0.25)
- [ ] Delisted fundamentals from CVM DFP/ITR raw statements (extend the `filing_dates.py` CVM pipeline) — separate milestone, needed before delisted names get fundamental features
- [ ] Sector for delisted via CANCELADA list join on cvm_code
- [ ] Handle delisting terminal event in the dataset (final return leg: tender price / merger ratio / worthless) — without it, delisted series just stop and the bias only shrinks, doesn't vanish
- [ ] Rebuild dataset; re-run IC + agent eval on the survivorship-free universe (F1 test)
- [ ] Report the HTTP 500 tickers (BTOW3, CESP6, ELPL3, IGTA3) to BolsAI
- [ ] Fix the stale CLAUDE.md claim ("zero delisted companies — structural")
