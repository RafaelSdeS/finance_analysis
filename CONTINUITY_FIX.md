# Ticker-Continuity Splice Defects Fix — Implementation Checklist

**Date:** 2026-07-15 | **Status:** In Progress | **Implemented by:** Claude  
**Issue:** Four categories of data corruption in ticker-continuity splicing; adj_close reconciliation factors triggered audit.

## Summary

- ✅ **Updated ticker_continuity.json:**
  - Added `keep_separate` type (parallel-trading acquirers: HGTX3→SOMA3, SULA11→RDOR3, SOMA3→AZZA3, BRML3→ALOS3, ENAT3→BRAV3)
  - Added vendor-alias rename entries: ARZZ3→AZZA3, RRRP3→BRAV3
  - Updated `_doc` to explain new type
- ✅ **Reordered build_ml_dataset.py:** repair *before* continuity (fixes event rekeying, honest factors)
- ✅ **Updated continuity.py:** skip `keep_separate` type; add factor sanity check [1/50, 50]
- ✅ **Updated repair.py:** rekey events through continuity map (splits under old names now match new-name rows)
- ✅ **Added test cases:** vendor-alias consolidation + keep_separate no-op
- ☐ **Run fast tests:** `python tests/run_all.py --group fast` (waiting for user terminal)
- ☐ **Run build:** `python -m src.build_dataset.build_ml_dataset` (verify no crash, inspect logs)
- ☐ **Run data tests:** `python tests/run_all.py --group data` (duplicate-detector, splice-continuity guards)

## Expected Dataset Changes

**Deduplicates (removes old alias):**
- ARZZ3 → consolidated into AZZA3 (3,827 rows)
- RRRP3 → consolidated into BRAV3 (1,407 rows)

**Keeps separate (both tickers stay, old treated as delisted):**
- HGTX3 (1998–2021) stays separate from SOMA3 (real Soma pre-acq history)
- SULA11 (2007–2022-12) stays separate from RDOR3 (real RDOR IPO 2020-12, not SulAmérica)
- SOMA3 (pre-merger) stays separate from AZZA3 (real Arezzo post-2011 IPO)
- BRML3 (pre-merger) stays separate from ALOS3
- ENAT3 (pre-merger) stays separate from BRAV3

**Series Boundaries (corrected):**
- AZZA3 now starts 2011-02-02 (real Arezzo IPO), not 2000 (Hering history)
- RDOR3 now starts 2020-12-10 (real IPO), not 2007 (SulAmérica rebased at IPO)

**Numerically unchanged (post-repair order):**
- TIMS3, B3SA3, BHIA3, NATU3, MBRF3, SAUD3, EMBJ3, AXIA3/5/6, RIAA3 (correctly spliced and repaired)
- ~0.696 factor for TIMP3→TIMS3 (post-repair, down from 6963 pre-repair)

## Verification Checklist

After build completes and user confirms no crashes:

- [ ] Fast group tests pass (`test_vendor_alias_rename_drops_duplicate`, `test_keep_separate_ignores_merger`)
- [ ] Build log shows:
  - Explicit "vendor alias: dropping N duplicate rows" lines for ARZZ3/RRRP3
  - NO graft events (keep_separate entries skipped with no output)
  - Factor sanity: all factors in printout are [1/50, 50]
  - Split repair messages show ODPV3-era splits matching SAUD3 rows post-rename
- [ ] Data tests pass:
  - Duplicate-series detector finds no 90%+ close-match pairs
  - Splice-continuity guard finds no |1-day returns| >25% at event boundaries
  - NaN-regression report reviewed (any delisted-return tickers is expected)
- [ ] Spot-check dataset:
  - ARZZ3, RRRP3 absent (deduped)
  - AZZA3 ticker count unchanged, AZZA3 starts 2011-02-02 only
  - RDOR3 starts 2020-12-10 (not 2007)
  - HGTX3, SULA11, SOMA3, BRML3, ENAT3 present as separate delisted series
  - TIMS3 boundary row (2020-10-13) has 0% return, no artificial jump

## Notes

- **repair.py event rekeying:** Rebuilds ticker-descendant chains from the map (resolves VVAR3→VIIA3→BHIA3 to add split events under all three names). Only happens if CONTINUITY_PATH exists and has matching entries.
- **Lazy approach:** No synthetic "source of truth" recomputation; just fixed the broken pieces (map, order, type handling, event matching). Smaller diff, preserves existing correctness.
- **Future:** Once delisted-universe restoration (DIAGNOSIS_PLAN F1) is underway, the kept-separate tickers become valuable for survivorship-bias research.
