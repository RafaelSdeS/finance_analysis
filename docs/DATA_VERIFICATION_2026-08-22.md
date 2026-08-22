# Data Verification — 2026-08-22

Ran the full test suite (`tests/run_all.py --group all`) plus targeted checks against
the real, on-disk data. Fast group: 55/55 pass. Data group: 14/19 pass — the 5
failures are broken down below with root cause. Also ran an independent raw-price
scan (not part of the existing suite) looking for scale artifacts.

## Root cause: `ml_dataset.parquet` is 3 commits stale

`data/processed/ml_dataset.manifest.json` records `git_commit: 25cdd84`. HEAD is
`b2907a2`, and one of the 3 commits in between — `35532d9 fix: close out
DATA_LAYER_FOLLOWUP_FINDINGS.md` — patches `src/build_dataset/repair.py` with a
close-stability guard. **The shipped parquet was built before that fix landed.**

Confirmed directly: ran `repair_unadjusted_splits` + `repair_isolated_adj_close_glitches`
from the *current* code against AFLT3's raw price file — output is correct (glitch
guard fires, 0 misfires). But the on-disk `ml_dataset.parquet` still shows the
pre-fix bug: `adj_close` on 2007-12-07 is `0.006353` where it should be `6.352723`
(an extra, erroneous ÷1000 that the old `repair_isolated_adj_close_glitches` applied
before the guard existed). This is what `test_final_dataset.py`'s "2 events leaking"
failure is — both leak entries are the same single AFLT3 incident.

- [ ] Rebuild: `python -m src.build_dataset.build_ml_dataset`
- [ ] Rerun `python -m src.build_dataset.terminal_events` (currently older than
      `ml_dataset.parquet` — `test_artifact_coherence.py`'s failure, and
      `forward_excess_return()` silently no-ops without it)
- [ ] Refit scaler: `python -m src.build_dataset.scale_features` — its mtime
      (08:56) is *older* than `ml_dataset.parquet`'s (10:26) on the same day, i.e.
      the checked-in scaler was fit before the last rebuild, not after
- [ ] Re-run `tests/run_all.py --group data` after, to confirm the AFLT3 leak and
      terminal_events checks go green

## Already-known, already-accepted (no new action)

Verified these against `docs/DATA_LAYER_FOLLOWUP_FINDINGS.md` /
`DATA_LAYER_CORRECTNESS_PLAN.md` — all three are deliberate, documented, owner-
accepted residuals, not new bugs, and a rebuild will **not** change them:

- `test_unit_scale_invariants.py`: 19/544 tickers outside the 10% `market_cap/shares
  == close` band — documented limitation of the shares-outstanding forward-adjust
  fix (a continuity donor that stopped filing before its real split, e.g. TIMS3).
- `test_top_traded_quality.py`: LUXM4's 289 non-positive `adj_*` rows — the
  CLAUDE.md-documented 2-decimal precision underflow ticker; "flag only, no
  repair" was already decided. The validator just isn't taught to exempt it.
- `test_raw_processed_reconciliation.py`: 12 uncovered dead tickers (BDLL3/4,
  CTSA3/4, JFEN3, LIQO3, MEND5/6, OIBR3/4, RSID3, YBRA4) — same bucket as the
  already-tracked "98 of 202 in-panel deaths have no terminal event" research
  item (§2b), explicitly deprioritized, not an implementation gap.

## New observation (informational, zero current impact)

Independent raw-price scan (all 1,328 BR ticker files, not gated on any
corporate_events match) for a same-day close ratio near an exact power of ten
(10/100/1000) with zero volume around the jump — the signature of a scale
artifact, distinct from what `repair_unadjusted_splits` already covers via the
corporate_events log:

- 9 of 11 hits (BALM4, BSLI3, CALI3, MNPR3, NORD3, NUTR3, RSUL4, UGPA3, VULC3)
  are exactly the tickers already listed in
  `src/data_collection/one_off/backfill_known_gaps.py`'s `FLAT_RUN_PADDING` —
  confirmed vendor-side (BolsAI/yfinance both) gap-padding artifacts, "no fix
  available from this vendor for this window," already triaged.
- GEPA4 looked new but is **correctly handled**: it has a matching
  `corporate_events.parquet` entry (INPLIT, factor 0.001, 2007-11-01) and
  `adj_close`/`log_return` are already continuous across it in the built
  dataset — false positive from the scan, not a bug.
- HAGA4 is a single-day round-trip at ~0.00001–0.00010 (2000-06-01, immediately
  reverts) — economically negligible, already at the documented precision floor.
- UGPA3 specifically: raw `close` jumps exactly ×1000 for 431 rows
  (2005-08-23 → 2007-05-04), volume=0 throughout. Confirmed **zero live effect**
  on the ML dataset — `market_cap`/`pl` are NaN for every row in that window (no
  fundamentals coincide), so no downstream feature is corrupted today. Flagging
  only because raw `close`/`open`/`high`/`low` ship as columns in
  `ml_dataset.parquet` and this window would corrupt any future feature that
  reads raw price directly (not `adj_close`) — no action needed unless that
  changes.

## yfinance fundamentals: USD-denominated data mislabeled as BRL (ADR-linked tickers)

Investigated a user-recalled concern that some tickers' yfinance data might be in USD
instead of BRL. **Prices are fine** (`Ticker.history`, `info['currency']` both correctly
report BRL for every ticker checked, on-disk `close`/`adj_close` match live BRL quotes).
The bug is real but confined to yfinance's *fundamentals* endpoints
(`quarterly_balance_sheet`, `quarterly_financials`) for tickers that are also dual-listed
as a US ADR — Yahoo's backend appears to serve the ADR's USD-denominated financials under
the BR ticker's own `.SA` symbol, while `info['financialCurrency']` still (wrongly) reports
`"BRL"`, so that field can't be used to auto-detect it.

Method: compare live yfinance `quarterly_balance_sheet` Stockholders Equity against
on-disk CVM `equity` for the latest overlapping quarter-end, across every BR ticker in
`data/raw/br/fundamentals/` that also has a US-listed ADR.

- **Confirmed affected: VALE3 (ADR `VALE`) and PETR4/PETR3 (ADR `PBR`/`PBR.A`).** Ratio
  CVM/yfinance ≈ 5.19–5.3 for both — matches the BRL/USD FX rate, not a segment or
  reporting-method difference. VALE3 e.g. 2026-06-30 equity: CVM R$201.47B vs. yfinance
  R$38.01B (labeled); PETR4 same date: CVM R$481.85B vs. yfinance R$92.91B (labeled).
- **Checked clean** (ratio 1.00–1.24, ordinary vendor/definition noise): ITUB4, BBDC4,
  BBDC3, ABEV3, SBSP3, CMIG4, BRKM5, GGBR4, CSNA3, TIMS3, UGPA3, VIVT3, PCAR3, MBRF3,
  SUZB3 — i.e. every other dual-listed BR/ADR pair checked does NOT reproduce the bug.
- **Separate, unrelated gap found in passing:** EMBR3 (ADR `ERJ`) — yfinance's
  `quarterly_balance_sheet` returns a completely empty index for this ticker. Missing
  data, not a currency mismatch.

**Zero impact on the shipped dataset.** `DATA_SOURCE["fundamentals"] = "cvm"`
(`config.py`) means `collect_fundamentals_yf` is never called for BR by default — the
on-disk `equity` values above for both VALE3 and PETR4 are the correct CVM ones, verified
against a live CVM pull in the same session. The only place this surfaces at all is
`tests/data_collection/validate_vs_yfinance.py`'s cross-check (`TICKERS = ["PETR4",
"VALE3", "WEGE3"]` — PETR4 was already in the default list), where it already produces
the intended "likely currency mismatch" note rather than a false FAIL, because that
script's `>200%`-no-fail threshold (`_print_fund_rows`) already exists for exactly this
shape of vendor divergence.

**Standing caveat, not a bug to fix:** this is a property of yfinance's own data, not
something in this repo to patch. The condition that keeps it harmless is
`DATA_SOURCE["fundamentals"]` staying `"cvm"` for BR — if that switch is ever flipped to
`"yfinance"`, VALE3/PETR4 (and presumably any other BR/ADR pair, e.g. a future addition)
would need this resolved first. See also CLAUDE.md's yfinance caveat under "Data sources
& limits".

## Bottom line

The data layer's own test suite is doing its job — it caught a real regression
(AFLT3) that a stale build artifact reintroduced, and correctly separates it from
three unrelated, already-accepted residuals. No new correctness bugs found beyond
the stale-build issue above; recommend the rebuild + rerun steps checked off above.
The yfinance-fundamentals-in-USD finding (previous section) is a confirmed vendor
quirk, already harmless by construction (CVM is the real BR fundamentals source) —
informational, not on the rebuild punch list.
