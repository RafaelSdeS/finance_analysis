# Data Integrity Test Plan

## Implementation log — 2026-08-16

The six "ready" items (§5b) are implemented, verified against real data, and registered in
`tests/run_all.py`'s `DATA` group. The two "not ready" items were investigated further during
implementation and turned out to have clean answers, so they're now implemented too — see
below. `--market us` golden gate (§4 D6) is not done; still needs the check-by-check triage
§5b flagged.

**Files changed:**
- `src/data_collection/validate.py` (P0-c) — `validate_prices` gained a raw-OHLC NaN check
  (root cause of D2/D3: every prior check was a comparison, and NaN compares False in pandas).
  `_common`'s duplicate-date check promoted warn→error, scoped to `(ticker, date_col)` — NOT
  bare `date_col` as originally planned: `corporate_events.parquet` is a shared multi-ticker
  file with 255 real same-date-different-ticker rows, which a bare-date check would have
  wrongly failed. `adj_*` columns deliberately excluded from the new NaN check (known,
  accepted precision-underflow class, see below) — only raw `open/high/low/close`.
- `tests/build_dataset/test_artifact_coherence.py` (P0-b, new).
- `tests/build_dataset/test_manifest_drift.py` (P0-a, new) — median-drift + row/column/date
  checks for BR, survivorship-floor checks for US. Whole-panel `nan_regressions()` NOT
  included, per the calibration finding. `check_market_drift()` is written market-agnostic and
  called for both BR and US as planned — it correctly no-ops for US today (no snapshot exists).
- `tests/build_dataset/test_raw_processed_reconciliation.py` (P1-b, new).
- `tests/data_collection/test_br_data_quality.py` (P1-a, new) — also closes D11 (US dividends
  sweep) via one market-agnostic function called for both roots.
- `tests/data_collection/test_us_data_quality.py` — extended with the same NaN-OHLC rate
  ceiling P0-c introduces, since it's real on the US side too (measured, see below).
- `tests/run_all.py` — all 4 new files registered in `DATA`.

**A bug found while implementing, before it shipped:** the original plan's reconciliation
identity (`raw = kept + dropped + spliced`) was written and initially coded as a literal sum
— `567 + 735 + 31 = 1333 ≠ 1328`. Caught before commit by testing against real data; fixed to
a set difference (`kept ∩ spliced = 5`, the `keep_separate` continuity entries).

**New findings from running these tests against real data (none were known before today):**

| # | Finding | Severity | Where |
|---|---|---|---|
| — | `dataset_v*` snapshot directories are **not market-namespaced** — `OUTPUT_PATH.parent == US_OUTPUT_PATH.parent`. A naive snapshot-glob for the drift check picked up a **BR** snapshot and diffed it against the **US** manifest (wrong columns, wrong dollar scale, 15 spurious "median drift" violations). Fixed by matching on the snapshot's actual output filename before use. | Caught before shipping | `test_manifest_drift.py` |
| D2/D3 | BOVA11 + CAMB3 raw NaN OHLC — as before. | Real, unfixed | `data/raw/br/prices/{BOVA11,CAMB3}.parquet` |
| NEW | **LUXM4**: 289/5032 rows (2000–2005) where `adj_close` underflows to a literal `0.00` — a 5th, previously-unnamed instance of CLAUDE.md's documented precision-underflow class (WDCN3/CAMB4/LLIS3/CCTY3 were the only named ones). Rate-ceilinged, not a new bug — already masked/flagged downstream. | Known class, newly identified instance | `data/raw/br/prices/LUXM4.parquet` |
| NEW | Same NaN-OHLC defect class is real on the **US side** too: 111/9,700 price files (1.14%), 209 rows. Rate-ceilinged in `test_us_data_quality.py` pending a recollection pass. | Real, unfixed | `data/raw/us/prices/*.parquet` |
| **NEW, largest** | **48 tickers with confirmed `status=CANCELADA`** — including **AMER3 (Americanas)**, whose panel ends 2023-01-19 after a ~68% one-week crash (the real accounting-fraud collapse) — fall outside every resolution path `terminal_events.py` has (not spliced, no terminal payoff, not even flagged by `find_rename_candidates()`). Root cause: CVM's own registry still shows these entities as `sit=ATIVO` (40) or `SUSPENSO(A) - DECISÃO ADM` (8) at the company level even though the ticker stopped trading — both `build_terminal_events()` and `find_rename_candidates()` are working exactly as designed, neither was built for "company delisted with no CVM-resolved cancellation reason and no live sibling ticker to redirect to." Their forward-return label is silently NaN instead of reflecting the real (often large, negative) outcome. **This is the single most significant finding of this whole investigation** — a real, previously-unknown survivorship-bias hole, exactly the class of problem this exercise was for. Fixing the classification logic is an economic judgment call (does `sit=ATIVO`-but-dead default to failure? acquired? a new bucket?) that belongs to whoever owns `terminal_events.py`'s payoff rules — not something this test-writing pass did on its own. | **Real, unfixed, needs a decision** | `src/build_dataset/terminal_events.py` |
| PRE-EXISTING, unrelated | Running the full `DATA` group surfaced 2 existing tests (untouched this session) also currently red against the committed build: `test_final_dataset.py` (NaN in `close` — same CAMB3 row as D3; **plus** 339/18,843 P/L rows not varying daily within quarter; **plus** an interior-NaN "merge bug" shape in 5 tickers: AZEV4, BGIP4, BNBR3, BSLI3, INEP3) and `test_top_traded_quality.py` (2 NaN `adj_close` rows in the top-50 universe — same CAMB3 + BPAC11/EPAR3/HBRE3/MAPT4 pattern). None of these were caused by anything changed this session (only `validate.py`, new test files, and `run_all.py` were touched — the processed parquet is byte-identical to the session start). Most likely predate the `7116232 data: BR collection refresh 2026-08-16` commit landing without a `--group data` run to confirm it was clean. Not investigated further — separate scope from implementing this test plan. | Pre-existing, not investigated | `data/processed/ml_dataset.parquet` |

**Final `--group data` tally:** 13 passed, 4 failed (`test_final_dataset.py`,
`test_top_traded_quality.py` pre-existing and unrelated to this session;
`test_br_data_quality.py`, `test_raw_processed_reconciliation.py` are this session's new tests
correctly catching the real findings above). `--group fast`: 55/55 passed, no regressions.

---

Investigation date: 2026-08-16. Branch `refactor`, dataset build `2026-08-15T20:29:53+00:00`
(git 750e199), BR `dataset_v2`.

**Status: recommendation only. No code written yet.**

---

## 1. What already exists (don't rebuild these)

The processed BR dataset is already well covered. `tests/build_dataset/test_final_dataset.py`
(667 lines, DATA group) is the golden gate and already asserts: no lookahead
(`reference_date <= trade_date`), no duplicate `(ticker, trade_date)`, no inf in numerics,
no weekend rows, `has_fundamentals=0` rows carry NaN fundamentals, NaN shapes are prefix-only
per ticker, asof merge picks the most recent *filed* quarter, no unadjusted split jumps leaking
into `log_return`, P/L varies daily within a quarter, filing-date ordering, CAGR flag domains.

Alongside it: `test_universe_integrity.py` (schema contract, survivorship floor on CANCELADA
tickers, `status` staticness, duplicate price series / round-trip oscillation detection),
`test_top_traded_quality.py` (OHLC consistency, calendar gaps, fundamentals coverage on the
top-50), `validate_vs_yfinance.py` (sampled vendor cross-check), `test_us_data_quality.py`
(raw US sweep).

The gaps below are the things *nothing* currently checks.

---

## 2. Defects this investigation actually found

Evidence for why each recommendation earns its place. All found without reading the large
parquets end-to-end — manifests plus one metadata-light raw sweep.

| # | Finding | Where | Currently caught by |
|---|---------|-------|---------------------|
| ~~D1~~ | ~~`data/processed/scalers/` does not exist~~ — **withdrawn 2026-08-16, not a defect.** Nothing consumes the scaler. See §2c. | — | n/a |
| D2 | **BOVA11 has a NaN `close`/`adj_close` on its latest bar (2026-08-14).** BOVA11 is the market series behind `beta_1y` and `momentum_vs_market_*`. | `data/raw/br/prices/BOVA11.parquet` | nothing |
| D3 | CAMB3 has an interior NaN `close` at 2019-08-15. | `data/raw/br/prices/CAMB3.parquet` | nothing |
| D4 | **103 of 612 raw fundamentals files contain `inf` values.** `clean.py` silently converts inf→NaN downstream, so the processed "no inf" check passes while the underlying vendor defect goes unmeasured. | `data/raw/br/fundamentals/` | nothing (`test_ratios_no_inf.py` is synthetic) |
| D5 | US dataset ships **4 columns that are 100% NaN** (`ebitda_margin`, `ebitda_margin_zhist_5y`, `ebitda_growth_yoy`, `dividend_coverage_ratio`). The manifest records them in `empty_columns` — nothing fails on it. | `us_ml_dataset.manifest.json` | nothing |
| D6 | **`us_ml_dataset.parquet` (5.5 GB, 2903 tickers, 1962–2026) has zero validation.** `test_build_us_dataset.py` is synthetic-only; no DATA-group test opens the built US file. | processed US | nothing |
| D7 | **Corrected 2026-08-16 — the original framing here was wrong twice over.** Conditioned properly (`has_fundamentals=1`, `reference_date >= 2011-01-01`), the real-`DT_RECEB` share is **80.8%**, not the 56.6% the raw "43.4% NaN `filing_lag_days`" figure implied — that headline counted `has_fundamentals=0` rows. And "all pre-2011 fundamentals are on the statutory fallback by construction" was **false**: pre-2011 rows are 82.7% real-dated (the `filing_dates` floor is 2010-12-31, which covers the Q4-2010 filings), and there are only 9,934 such rows anyway. The real gap is narrower than stated but still unguarded: **no floor is asserted**, so a CVM collection regression toward 0% real dates would pass silently and quietly degrade the no-lookahead claim into a statutory estimate. | `quality_filters.attach_filing_dates` | nothing |
| D8 | `nan_regressions()` exists and is called from `sync_dataset_version()` — but only **prints a warning**. No test fails on cross-build drift. `column_stats` (nan_pct/mean/std/p1/p50/p99, 157 BR + 162 US columns) is captured every build and **never compared to anything**. | `manifest.py:271,318` | nothing |
| D9 | BR `survivorship_coverage` is `"not tracked"` in the manifest; US records per-year coverage. | `ml_dataset.manifest.json` | partially — `test_universe_integrity.check_survivorship` asserts a floor at test time, but nothing is recorded per build |
| **D10** | **`validate.py` has no NaN check on prices at all — the root cause of D2/D3.** Every gate in `validate_prices` is a comparison (`close <= 0`, `high < low`, bracket checks); **NaN compares False in pandas**, so a NaN OHLC row passes every one of them. Duplicate `trade_date` is only a `warn()`, not an `error()`. This is the write-time gate for **both markets**, so all 9,700 US price files share the blind spot. | `src/data_collection/validate.py:40-86` | nothing |
| **D11** | **US dividends (4,243 files) are swept by nothing.** `test_us_data_quality.py` covers prices/fundamentals/macro only. | `data/raw/us/dividends/` | nothing |
| ~~D12~~ | ~~`US_SCALER_DIR` declared but no code writes it~~ — **withdrawn 2026-08-16.** Correct as an observation, but not a gap: US scaling is deliberately out of scope. See §2c. | — | n/a |

**Good news, and the reason #4 below is worth locking in:** the BR raw→processed ticker
reconciliation closes *exactly* today —

```
raw − kept − dropped_no_fundamentals − continuity.old  =  ∅        (residual 0)
|raw|=1328  |kept|=567  |dropped|=735  |spliced|=31
kept-but-no-raw-file: 0    dropped-not-in-raw: 0    spliced-not-in-raw: 0
```

⚠️ **This is a set identity, not a sum** — `567 + 735 + 31 = 1333 ≠ 1328`. The gap is real and
meaningful: `kept ∩ spliced = 5`. Those five are the continuity map's `keep_separate` entries
(parallel-trading acquirer, both legs deliberately stay independent), so they appear in the
map *and* legitimately remain in the panel. `kept ∩ dropped = 0` and `dropped ∩ spliced = 0`.
Write the test as a set difference; anyone who codes it as an equality of counts will get a
failing test and conclude, wrongly, that the data is broken.

Structural sweeps of raw BR came back clean otherwise: 0 duplicate dates, 0 non-positive
prices, 0 OHLC bracket violations, 0 negative volume, 0 unsorted files, 0 weekend rows,
0 future dates across 1328 price files / 2.29M rows; 0 duplicate `reference_date`,
0 `available < reference`, 0 negative assets/shares across 612 fundamentals files;
dividends and macro (selic/cdi/ipca) clean.

That reconciliation identity is the single strongest anti-survivorship invariant available,
and it holds right now — which is exactly when to freeze it into a test.

---

## 2c. Feature scaling is out of scope — confirmed 2026-08-16

Checked on request. **The model genuinely does not need it, and D1/D12 are withdrawn.**

Evidence:

- `src/portfolio/` contains **zero** references to `scaler`, `joblib`, `scale_features`, or
  `SCALER_DIR` (repo-wide grep, no matches). Nothing loads `feature_scaler.joblib`.
- The only learner in the stack is `lgb.LGBMRegressor` (`alpha.py:102`). The two other
  numerical components aren't feature-scale-sensitive: `LedoitWolf` (`risk.py:12`) estimates
  covariance over **returns**, already on a common scale, and `cvxpy` (`optimizer.py:13`)
  optimizes over alpha predictions and Σ, not raw features.
- **This was already decided.** `docs/PORTFOLIO_IMPLEMENTATION_PLAN.md:22`, pothole P4:
  "*LightGBM is scale-invariant. Stage A needs no scaler. `scale_features.py` /
  `iter_fit_windows` / `feature_scaler.joblib` are irrelevant to a GBM.* → **Drop the scaler
  from V1 entirely.**"

Why the invariance is exact, not approximate: a decision tree splits on one feature at a
time via a threshold test. `RobustScaler` is a strictly monotone affine map per column,
`(x - median) / IQR`, which preserves the ordering of values — so the set of reachable
partitions is identical and the optimal split gain is unchanged. LightGBM's histogram
binning derives bin edges from the data distribution, and an affine transform maps those
edges identically. Predictions are the same modulo float noise. `lambda_l1`/`lambda_l2`
don't change this: they regularize **leaf output values** (target space), not feature space.

Consequence for this plan: the missing `data/processed/scalers/` is **not** a data-integrity
defect and needs no test, no fix, and no mention in the artifact-coherence gate.

The one condition that would reverse this: a neural-net alpha head (deferred to Phase 7 per
the same doc). NNs are not scale-invariant — gradient descent on unscaled features with wildly
different ranges converges badly, and this dataset deliberately keeps extreme ratios unclipped
(`|pl| > 400,000` distress cases per CLAUDE.md), which would dominate a NN's first layer. If
that head is ever built, scaling comes back and `iter_fit_windows()` is the seam for it.

Housekeeping note, not a recommendation to act now: `scale_features.py`,
`tests/build_dataset/test_scale_features.py`, `SCALER_DIR`/`US_SCALER_DIR`, and
`manifest.py:329`'s `copytree` are now unreachable from any consumer. They're harmless and
already tested, so deleting them buys little and costs a docs pass — but CLAUDE.md's Run
Commands currently present `scale_features.py` as a normal build step ("rerun after a
rebuild"), which reads as required. Worth softening that line to "optional, unused by the
current LightGBM pipeline" so the next person doesn't file the same false bug I did.

---

## 2b. BR / US coverage matrix

The two markets are **not** symmetric, and some of the asymmetry is deliberate (US raw is
gitignored and survivor-only by construction via the SEC crosswalk; there is no US continuity
map and no US terminal-events concept). What follows marks where the plan applies to both,
where it's legitimately BR-only, and where the US side is simply missing something.

| Concern | BR | US | Plan item |
|---|---|---|---|
| Raw price/fundamentals sweep | none today | `test_us_data_quality.py` | P1-a (BR), §4 (US extensions) |
| Raw dividends sweep | none today (314 files) | **none today (4,243 files)** — D11 | P1-a covers both |
| Write-time NaN gate | **blind** — D10 | **blind** — D10 (same function) | P0-c, new |
| Processed golden gate | `test_final_dataset.py` | **none** — D6 | §4 `--market us` |
| Manifest drift | 2 snapshots exist (v1, v2) | ⚠️ **no snapshots — not implementable yet** | P0-a (BR only, see below) |
| Artifact coherence | split/manifest/snapshot | split/manifest only (no US snapshot) | P0-b (scaler dropped, §2c) |
| Raw→processed reconciliation | closes at 0 today | not applicable as-is | P1-b (BR), §4 (US variant) |
| Survivorship measured per build | not tracked — D9 | `survivorship_coverage` recorded, 33 years | §4 (BR), P0-a asserts floor (US) |
| Filing-date provenance | 80.8% real-dated — D7 | SEC point-in-time date, unmeasured | §4, both |

⚠️ **Correction found in final review:** `data/processed/` contains only `dataset_v1/` and
`dataset_v2/`, and both are **BR** (`ml_dataset.manifest.json`). There is **no US snapshot
directory at all**, so the US half of P0-a's drift check has nothing to diff against — drift
needs two builds and US has exactly one manifest. Either `sync_dataset_version()` needs a US
path (it hardcodes the BR `SCALER_DIR`/`OUTPUT_PATH` — check it), or the US drift gate waits
until a second US build exists. Don't write it as "same function, two paths" and expect it
to run.

Cost note: US file counts are **7–13×** BR's, not a flat 7× (prices 9,700 vs 1,328 = 7.3×;
fundamentals 8,283 vs 612 = 13.5×; dividends 4,243 vs 314 = 13.5×). A per-file US sweep is
minutes, not seconds — keep it in `DATA`, never in `FAST`.

---

## 3. Recommended tests

**Four new test files (P0-a, P0-b, P1-a, P1-b) plus one source fix (P0-c).** Ranked by
(bugs caught) ÷ (lines + runtime).

### P0-a — `tests/build_dataset/test_manifest_drift.py` (group: DATA)

Manifest-vs-manifest only. Reads two small JSON files, no parquet. Runtime ~0s.
Catches D5, D8, and the entire class of "the rebuild silently got worse".

Compare `data/processed/ml_dataset.manifest.json` against the newest
`data/processed/dataset_v{N}/ml_dataset.manifest.json`:

- [ ] No column present in the previous build is missing from the current one
- [ ] `rows` did not drop by more than a threshold (say 2%). **Do not assert `tickers` never
      drops** — this project actively quarantines tickers (WDCN3, CAMB4/LLIS3, CCTY3, and the
      13 in `dropped_no_fundamentals.quarantined`), so a legitimate build can lose names.
      Assert a bounded drop instead, or the gate fires on your own cleanup work
- [ ] `date_max` is >= the previous build's (never goes backwards)
- [ ] `empty_columns` is empty — **fails on US today (D5)**, so land it with an explicit,
      dated allowlist of the 4 known-empty US columns rather than a blanket skip
- [ ] ⚠️ **`nan_regressions()` as a whole-panel assert does NOT work — needs redesign, see
      the calibration box below.** Do not land it as "returns empty"
- [x] `column_stats` p50 drift: for each numeric column present in both, fail if the median
      moved more than N× the previous IQR proxy `(p99-p1)`. Catches a unit change, a
      scale bug, or a currency/percent regression that leaves NaN counts untouched.
      **Calibrated 2026-08-16 on v1→v2: max observed drift 0.071, and 0 of 156 columns
      exceed 0.25. Threshold 0.25 is safe; 0.15 would still pass and catch more.**
- [ ] ⚠️ **BR only for now.** "Run it for both markets" was wrong — there is no US snapshot
      directory to diff against (see §2b). Write the function market-agnostic, wire BR now,
      wire US when a second US build exists
- [ ] Assert the US `survivorship_coverage` per-year ratio against a floor and against
      year-over-year regression (the US half of P1-b, see below). **This one needs no
      snapshot** — it reads the current US manifest alone, so it works today. Verified
      populated: 33 year-records, e.g. 1994 coverage 0.171 (403 priced / 2,351 roster CIKs),
      1996 coverage 0.107. Set the floor per-era, not globally — early-90s coverage is
      ~10–17% by construction and a single global floor is either vacuous or always red

### ⚠️ Calibration result — P0-a's NaN gate is not landable as written

Measured `dataset_v1 → dataset_v2` (both legitimate builds) on 2026-08-16:

```
v1: 1,308,104 rows / 510 tickers        v2: 1,706,604 rows / 567 tickers
row delta +30.5%, ticker delta +57, columns added 0, dropped 0

nan_regressions(threshold=2.0)  -> 107 regressions
nan_regressions(threshold=5.0)  ->  91 regressions
nan_regressions(threshold=10.0) ->   4 regressions
```

**None of those 107 are data corruption.** v2 added 57 tickers — mostly thin/delisted names
recovered by the survivorship work — and thin tickers carry more NaN, so `nan_pct` rises
across nearly every column at once. **`nan_pct` is simply not comparable across builds when
universe composition changes**, and this project's whole direction is to keep widening the
universe with exactly the kind of sparse names that move this metric.

Raising the threshold doesn't fix it, it just blinds the check: at 10.0 only 4 survive, and
you'd miss any genuine 5–9pp regression.

**Redesign:** compute the NaN comparison on the **intersection of tickers present in both
builds**, not the whole panel. A cohort that exists in v1 and v2 has a stable composition,
so any nan_pct rise within it is a real regression. Then a 2pp threshold becomes meaningful.
Cost: one extra `groupby` per manifest — but `column_stats` is whole-panel only, so this
needs the per-cohort stats recorded **at build time** in `write_manifest()`. That's a
`manifest.py` change, not just a test.

Lazier interim option if that's too much: assert only on the columns that *don't* depend on
ticker sparsity (macro, calendar, price-technical columns present for every row), and leave
the fundamental-coverage columns to the median-drift check, which is already validated clean.

**Genuine signal found in the same run, worth investigating independently of any test:**
`num_trades` NaN went 1.79% → 64.48% between v1 and v2, and `amihud_illiquidity`
0.53% → 14.03%. Those are far outside the cohort effect the other 105 share. Either the new
tickers systematically lack `num_trades`, or the vendor stopped supplying it — worth one look.

### P0-b — `tests/build_dataset/test_artifact_coherence.py` (group: DATA)

JSON/filesystem only, no parquet. Runtime ~0s. ~20 lines.

The failure mode: training against a `split_config.json` computed from a *different* dataset
build, so the train/val/test cutoffs don't match the panel they're filtering. Nothing today
notices. (Scaler checks removed — see §2c.)

BR:

- [ ] `split_config.json.built_at` == `ml_dataset.manifest.json.built_at` (the split wasn't
      recomputed against a stale panel, or vice versa)
- [ ] The highest `dataset_v{N}/ml_dataset.manifest.json` matches the current top-level
      manifest (`sync_dataset_version` actually ran)
- [ ] `terminal_events.parquet` exists and its **mtime** is not older than the dataset build.
      Note: its schema is `[ticker, delist_date, event_type, terminal_payoff]` — there is
      **no `built_at` column**, so mtime is the only handle. `forward_excess_return()`
      silently no-ops when the file is missing, which is the worst kind of failure: a label
      that quietly changes meaning. BR-only; no US analogue

US — **asymmetric, don't just copy the BR block**:

- [ ] `us_split_config.json.built_at` == `us_ml_dataset.manifest.json.built_at`. This one
      transfers directly. **The snapshot check does not** — there is no `us_dataset_v{N}/`
      directory (see §2b); check whether `sync_dataset_version()` even has a US path before
      writing an assert against one
- [ ] No US terminal-events check — the concept doesn't exist on that side (see §4)
- [ ] No scaler check on either side (§2c)

### P0-c — fix the NaN blind spot in `src/data_collection/validate.py` (both markets)

**Do this before P1-a.** Not a test — a ~4-line fix to the shared write-time gate, and the
root cause of D2 and D3.

`validate_prices` expresses every check as a comparison, and NaN compares False in pandas, so
a NaN OHLC row passes `close <= 0`, `high < low`, and all the bracket checks silently. That's
how BOVA11's NaN got written to disk in the first place. The same function gates all 9,700 US
price files, so the hole is market-wide.

- [ ] `validate_prices`: `r.error()` if any of the OHLC/`adj_*` columns contain NaN
- [ ] `_common`: promote duplicate `date_col` from `warn()` to `error()` — a duplicated
      trading day is never legitimate in either market
- [ ] Add an explicit **latest-bar completeness** check: the final row must have complete
      OHLC. This is D2's exact shape and the most likely to recur, since `--mode update`
      appends to the tail every quarter
- [ ] `validate_us_fundamentals` / `validate_fundamentals`: same NaN audit on their key columns

One guard in the shared function beats the same guard duplicated in two test files — and it
fixes collection-time too, so bad rows stop reaching disk instead of merely being reported
after the fact. Both `test_us_data_quality.py` and the new BR test then inherit it for free.

### P1-a — `tests/data_collection/test_br_data_quality.py` (group: DATA)

The BR analogue of `test_us_data_quality.py`. **Delegate to `validate.validate_prices` /
`validate_fundamentals` exactly the way the US test does** rather than hand-rolling the
predicates — with P0-c landed, that single call covers NaN, dupes, OHLC, non-positive prices,
and latest-bar completeness for both markets at once. The BR test then only adds what's
genuinely BR-specific. ~100 lines, not 150. Runtime ~1–2 min over 1328+612+314 files.
Catches D2, D3, D4.

Per-file sweep of `data/raw/br/{prices,fundamentals,dividends,macro}`.

*Inherited from `validate.py` once P0-c lands — one call per file, no new predicates:*
NaN OHLC (**fails on BOVA11 (D2) and CAMB3 (D3)**), duplicate `trade_date`, future dates,
non-positive prices, negative volume, OHLC brackets, latest-bar completeness.

Treat BOVA11 as a hard failure rather than folding it into a rate ceiling — it feeds
`beta_1y` and `momentum_vs_market_*` for the entire panel, so one NaN there is
disproportionately expensive compared to a NaN on one thin ticker.

*BR-specific additions:*

- [ ] Prices: monotone dates and no weekend rows (all pass today — freeze them).
      `validate.py` doesn't check either, and weekends are market-calendar-specific
- [ ] Fundamentals: no `inf` in numeric columns — **fails on 103/612 files (D4)**. Land as
      a rate ceiling against today's measured 16.8% with a dated comment, so it catches a
      *new* systemic regression without blocking on the existing backlog. Note the US test
      already asserts inf at **zero tolerance** — so this is a genuine BR-only backlog, not
      a shared blind spot
- [ ] Fundamentals: no duplicate `reference_date`, no NaN/future `reference_date`,
      no negative `total_assets`/`shares_outstanding` (all pass today)
- [ ] Macro: `selic`/`cdi`/`ipca` have no NaN, no duplicate `reference_date`, and sit in
      plausible ranges — and specifically that `ipca` is series 433, not 432 (a monthly
      series has ~319 rows since 2000; the annual meta target would have ~26). Cheap guard
      against a caveat the repo already documents as having bitten once
- [ ] Dividends: no negative `value_per_share`, no future `ex_date`, `ex_date <= payment_date`
- [ ] **Write the dividends sweep as a market-agnostic function taking a root path**, then
      call it for `data/raw/us/dividends` too — 4,243 US files currently swept by nothing
      (D11). Same three predicates apply to both markets unchanged

### P1-b — `tests/build_dataset/test_raw_processed_reconciliation.py` (group: DATA)

The anti-survivorship gate. **BR-only, deliberately** — see the US note at the end of this
section. Reads one column (`ticker`) from the processed parquet plus two small JSONs —
cheap. ~60 lines.

- [ ] **Every raw price ticker is accounted for**: `raw == kept ∪ dropped_no_fundamentals ∪
      continuity.old`, residual exactly 0, and no kept ticker lacks a raw file.
      Holds today — freeze it. A future silent disappearance becomes a test failure instead
      of a quiet universe shrink
- [ ] Every `dropped_no_fundamentals` bucket key is one of the six known reasons; fail on a
      new unlabeled bucket, and fail if `gap_unexplained` grows past its current size (1)
- [ ] ⚠️ **Terminal-events coverage — does NOT hold today, needs investigation before it can
      be an assert.** Measured 2026-08-16, tickers ending early and not in the continuity map:

      | cutoff before `date_max` | end early | have terminal event | **uncovered** |
      |---|---|---|---|
      | 30d  | 268 | 81 | **187** |
      | 90d  | 194 | 81 | **113** |
      | 180d | 188 | 81 | **107** |

      This is the exact seam where survivorship bias re-enters — a ticker that just stops,
      carrying a NaN forward return instead of a realized payoff — so it's the most important
      item in this plan and also the least ready. `terminal_events.parquet` has 85 rows and
      covers 81 of the early-enders; the other ~107 are unexplained at any cutoff.
      **Find out what those 107 are before writing the test.** Plausible benign categories:
      tickers whose CVM registry lookup found nothing, names already excluded from the label
      by other means, or thin tickers that stopped trading without a formal cancellation.
      If most turn out benign, this lands as a rate ceiling; if not, it's a real data gap and
      the test is the least of the work.
- [ ] `terminal_events.find_rename_candidates()` returns nothing new — an unspliced rename
      (CVM status still ATIVO but no longer trading) is a survivorship hole disguised as a
      delisting. Report-only; the map is hand-maintained on purpose

**Why there is no US version of this test.** The BR identity works because BR raw is
git-tracked, includes delisted names recovered via `collect_delisted.py`, and has both a
continuity map and a cancellation registry to reconcile against. US has none of that: the
collection universe is gated by `sec/crosswalk.py` (SEC's `company_tickers.json`, current
listings only), so it is **survivor-only by construction** — a decision consciously accepted
2026-07-29 per `docs/US_COLLECTOR_FIX_PLAN.md` §4. A reconciliation test there would pass
trivially and prove nothing, because the raw tree can't contain what was never collected.

The honest US equivalent is to *measure the gap rather than reconcile it*, which the pipeline
already does: `sec/universe.py` builds a genuinely bias-free roster (every CIK that filed a
10-K/10-Q since 1994) purely to compute the manifest's `survivorship_coverage` per-year ratio.
So the US check belongs in P0-a as a drift assert on that field (floor + no year-over-year
regression), not as a separate reconciliation file. Cheap, and it's the only number that
actually tracks US survivorship bias.

---

## 4. Smaller additions to existing files (no new file needed)

- [x] **D7, filing-date provenance floor (BR)** → **measured 2026-08-16, ready to land.**
      Among rows with `has_fundamentals=1` and `reference_date >= 2011-01-01` (1,186,503 rows),
      the real-`DT_RECEB` share is **80.8%** → set the floor at **71%**. Add as one check in
      `test_final_dataset.py` and record the share in the manifest so P0-a watches it.
      Note the headline "43.4% NaN `filing_lag_days`" from §2 D7 is a whole-panel figure that
      includes `has_fundamentals=0` rows; conditioned properly the picture is much better.
      Also: only 9,934 fundamental rows predate 2011 (the `filing_dates` floor is
      2010-12-31), so the pre-2011 statutory-fallback exposure is negligible, not systemic.
      This check keeps "no lookahead" from quietly degrading into "statutory estimate".
- [ ] **D7 US analogue** → `sec/fundamentals.py` sets a point-in-time
      `fundamentals_available_date` from the real filing index, but nothing measures how
      often it's genuinely sourced vs. defaulted, and `build_us_dataset.py` skips
      `attach_filing_dates` entirely (no `filing_lag_days` column exists on the US side).
      Worth one measurement pass before deciding whether it needs a gate — if the US date is
      always real, this is a no-op; if it silently falls back the way BR's does for ~19% of
      post-2011 fundamental rows, it's the same lookahead risk with no instrument on it.
- [ ] **D6, US processed dataset** → don't write a second 667-line validator. Add a
      `--market {br,us}` flag to `test_final_dataset.py`; it already takes `--file`, so most
      of the work is separating the market-agnostic checks from the BR-specific ones.
      Then add it to the `DATA` list. **The BR-only list below is an educated guess, not a
      verified enumeration** — split-jump/`log_return` (US skips split repair), continuity,
      and filing-lag ordering (US has no `filing_lag_days` column) are near-certain, but the
      other ~20 checks haven't been triaged. Budget for that triage; see §5b.
- [ ] **D9, BR survivorship coverage** → have `build_ml_dataset.py` pass a
      `survivorship_coverage` frame for BR the way `build_us_dataset.py` does (per-year
      ratio of CANCELADA-inclusive universe to surviving universe). Cheap, and it makes the
      bias *measurable per build* rather than only asserted once in
      `test_universe_integrity.py`.
- [ ] Register the four new files in `tests/run_all.py`'s `DATA` list. None of them hit a
      live vendor, so none belong in `NON_BLOCKING` — they should be able to fail CI.

---

## 5. Deliberately not recommended

- **No `pandera` / `great_expectations` dependency.** The manifest's `column_stats` already
  is the schema+distribution snapshot those libraries would build, and `nan_regressions()`
  is already written. Adding a framework buys a DSL, not a check.
- **No per-row statistical validation of the 5.5 GB US parquet on every run.** Read the
  manifest; touch the parquet only for the handful of checks that genuinely need row-level
  data, and read single columns when so.
- **No new outlier/anomaly detector.** `test_final_dataset.check_outliers_zscore` and
  `test_universe_integrity`'s round-trip oscillation check already cover this, and both
  carry hard-won threshold calibration in their comments. Adding a third would mostly
  generate noise to triage.
- **No re-attempt at a "is this a real split" persistence guard.** Rejected three times
  already per CLAUDE.md; don't reopen without new evidence of an actual misfire.

---

## 5b. Readiness — checked 2026-08-16

**Ready to implement as written** (verified or calibrated against real data):

| Item | Status |
|---|---|
| P0-c `validate.py` NaN guard | Root cause confirmed at `validate.py:40-86`. ~4 lines |
| P0-b artifact coherence | ~20 lines. BR full; US drops the snapshot check |
| P1-b reconciliation identity | Verified: residual **exactly 0** today — as a *set* difference, not a sum |
| P0-a median-drift check | Calibrated: max observed 0.071, 0/156 cols exceed 0.25. **BR only** |
| P0-a US survivorship floor | Verified populated (33 year-records); needs per-era floors |
| §4 D7 provenance floor | Measured: 80.8% → floor 71% |
| P1-a raw sweep | Predicates verified against the real tree. Sequenced **after** P0-c, since it inherits those predicates; only the inf ceiling (16.8%) is a judgement call |

**Not ready — blocked on a decision or an investigation, not on typing:**

| Item | Blocker |
|---|---|
| P0-a NaN-regression gate | **Design change.** 107 false positives on a legitimate build. Needs per-cohort stats recorded in `write_manifest()` — see the calibration box in §3 |
| P0-a US drift half | **Missing prerequisite.** No `us_dataset_v{N}/` snapshot exists, so there is nothing to diff. Needs a US path in `sync_dataset_version()` or a second US build first |
| P1-b terminal-events coverage | **Investigation.** 107 tickers uncovered at a 180d cutoff; unknown whether benign. Determines assert vs. rate ceiling |
| §4 `--market us` golden gate | **Sizing unknown.** Requires triaging which of `test_final_dataset.py`'s ~25 checks are BR-specific; 3 identified, ~20 untriaged |

The blocked items include the ones that matter most for survivorship bias, which is the thing
you opened with. That's not a reason to delay the ready items — they're independent — but it
is a reason not to call the plan "done" once those are green.

### Review log — corrections made 2026-08-16

Final self-review of this document found six errors in my own recommendations. Recorded so
the reasoning is auditable rather than silently patched:

1. **Reconciliation shown as a sum.** `567 + 735 + 31 = 1333 ≠ 1328`. It's a set identity;
   `kept ∩ spliced = 5` (the `keep_separate` entries). Coded as a count equality, the test
   fails and implies corruption that isn't there.
2. **D7 overstated the problem twice.** "43.4% on fallback" counted `has_fundamentals=0`
   rows (true figure 80.8% real-dated), and "all pre-2011 on fallback by construction" was
   simply false (82.7% real-dated).
3. **"Run P0-a for both markets" is not possible** — no US snapshot directory exists.
4. **P0-b asserted a US snapshot pair** that likewise doesn't exist.
5. **`terminal_events.parquet` has no `built_at` column** — mtime is the only handle.
6. **"`tickers` did not drop at all"** contradicts the project's own quarantine practice.

Also corrected: US/BR file-count ratio (7–13×, not 7×), and the §4 US-golden-gate BR-only
check list is now labelled a guess rather than an enumeration.

---

## 6. Suggested order

1. **P0-c (`validate.py` NaN guard)** — ~4 lines, fixes the root cause of D2/D3 at
   collection time for **both markets**, and every other test inherits it. Do this first.
2. P0-b (`test_artifact_coherence.py`) — smallest file, ~20 lines now that the scaler checks
   are gone (§2c). Lower value than originally rated; keep it, it's nearly free.
3. P0-a (`test_manifest_drift.py`) — pure JSON, guards every future rebuild. Land the
   **BR** median-drift + column/row/date checks and the **US survivorship floor** (which
   needs no snapshot); leave the NaN gate and the US drift half for later.
4. P1-b (`test_raw_processed_reconciliation.py`) — locks in an invariant that holds *today*.
5. P1-a (`test_br_data_quality.py`) — largest, and will fail on landing (D2/D3/D4), so it
   needs the allowlist/rate-ceiling calibration pass. Do it when you can spend time on
   the thresholds.
6. §4 items as follow-ups. The `--market us` golden gate (D6) is the biggest single
   coverage win left on the US side.

Fix D2 regardless of test timing — a NaN on the benchmark's latest bar feeds `beta_1y` and
`momentum_vs_market_*` for the whole panel. (D1 withdrawn, §2c.)
