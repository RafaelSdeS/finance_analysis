# Data Integrity Audit — 2026-07-24

Scope: `data/processed/ml_dataset.parquet` (1,319,349 rows · 515 tickers · 167 cols,
`dataset_v1`, git `9ea26f1`) + the Stage-2 build code that produced it. Verified against
the actual built dataset and `data/raw/`, not just the CLAUDE.md caveats.

Legend: 🔴 Critical · 🟠 Medium · 🟡 Low · ✅ Passed

---

## 1. Passed checks

- ✅ **Manifest ↔ dataset shape.** `manifest.rows/tickers` == parquet (1,319,349 / 515).
  Column set identical. `dataset_v1` content fingerprint (rows/tickers/dates/cols) matches
  current build. Zero duplicate `(ticker, trade_date)` index rows.
- ✅ **`dropped_no_fundamentals` logged in manifest** (survivorship-relevant drops are
  queryable, not just stdout). Delisted names ARE retained in the panel: 115 `CANCELADA`
  vs 400 `ATIVO` tickers — universe-level survivorship is materially mitigated.
- ✅ **Benchmark de-biased (Issue 2 fix holds).** `beta_1y` / `momentum_vs_market_*` use the
  BOVA11 series, not an equal-weighted panel mean. No `_mkt_*` scratch columns leak into the
  output. `beta_1y` core distribution sane (p1 −0.37 / p50 0.63 / p99 2.16), no `inf`.
- ✅ **No `inf` anywhere** in the final dataset (clean pass works). `adj_close` has no interior
  NaN holes. `f_score` strictly in [0, 5] with `skipna=False` (undefined component → NaN, not a
  silently-partial score).
- ✅ **Lookahead guards intact.** `merge_asof(direction="backward")` on real `fundamentals_available_date`;
  rolling (not global) percentiles; YoY/QoQ/F-Score gated by `_within_calendar_gap` so a
  vendor-missing quarter can't masquerade as a 1-quarter/1-year window.

---

## 2. Critical & medium risks

### 🔴 C1 — Near-identical price series across *different* companies
Structural dedup on `(ticker, trade_date)` cannot catch this; it's cross-*ticker* duplication.
The initial exact-hash pass under-*and*-over-counted — refined into a tolerance-based,
CNPJ-aware check (`test_universe_integrity.py` §3.6). Two categories fell out:

**False positives — same legal entity, not corruption.** `ALOS3`/`ALSO3` and `MEGA3`/`SRNA3`
share an identical CNPJ in the CVM crosswalk (Aliansce Sonae/Allos; Omega Energia/Serena Energia)
— same company, two ticker mnemonics, same class as the already-documented `ELET5→AXIA5` rename.
Not yet in `ticker_continuity.json` (no verified rename date on hand to add one without guessing),
but excluded from the corruption guard via CNPJ match instead.

**Real corruption — confirmed distinct CNPJs:**

| Group | rows | shared span | In final dataset? | Status |
|-------|-----:|-------------|:---:|--------|
| `BAHI3` ≡ `CGRA3` | 4319 | 2000-01-10 … 2026-07-10 | yes | **Quarantined** (`BAHI3`; `CGRA3` kept) |
| `ATOM3` ≡ `MBLY3` ≡ `LVTC3` | 2307 | 2017-03-23 … 2026-07-10 | `ATOM3`/`MBLY3` yes, `LVTC3` no | **Quarantined** (all 3) |
| `ARND3` ≡ `PORT3` | 1156 | 2021-10-25 … 2026-07-10 | yes | **Quarantined** (both) |
| `GFTT3` ≡ `GFTT4` | 2 | 2001-01-19 … 2004-06-29 | no (`< MIN_PRICE_ROWS`) | Not corruption — vendor stub

- **Only the price files match.** Each group's *fundamentals* differ, so the impostor side carries
  its own real fundamentals bolted onto a copied price series.
- **`BAHI3`/`CGRA3` resolved and quarantined.** BolsAI's own `market_cap` can't cross-validate
  either side (it's tautologically `shares_outstanding × close` off the *same* shared price, so it
  always reads a perfect 1.000 ratio for both). But `CGRA3` has 32 real dividend events on file,
  and every one prices at a plausible 4–18% yield against the shared series — independent
  corroboration the series is genuinely `CGRA3`'s (Grazziotin). `BAHI3` (Bahema Educação) has no
  dividends file to check the other way. `BAHI3` added to `QUARANTINED_TICKERS`.
- **`ATOM3`/`MBLY3`/`LVTC3` and `ARND3`/`PORT3` — resolved, quarantine all 5 (no winner
  identifiable).** No internal signal available: neither side of either group has a dividends
  file, and market_cap is tautological (derived from the same shared price on both sides). External
  checks came up empty too: yfinance has zero data for `ATOM3.SA`/`MBLY3.SA`/`LVTC3.SA`/`PORT3.SA`
  (delisted/no data), and its `ARND3.SA` series only weakly correlates with ours (0.27 daily
  log-return correlation) — not strong enough to positively vouch for `ARND3` either. Decisively,
  BolsAI's own **live** API (queried directly via MCP, same verification method that resolved
  `CCTY3` previously) independently reproduces the exact same confusion right now:
  `get_price_history("ATOM3")`, `("MBLY3")`, and `("LVTC3")` all return **`WDCN3`**'s data (a third,
  unrelated, already-quarantined ticker); `get_price_history("PORT3")` returns **`ARND3`**'s data.
  `get_price_history("PETR4")` returned correct data in the same session, ruling out a general tool
  malfunction. This is a **live, reproducible vendor-side ticker-resolution bug**, not a stale
  artifact of our original collection — re-running the collector today would fetch the identical
  wrong data. `PORT3`'s `company_info` corporate_name is "WILSON SONS S.A." (a materially larger,
  since-delisted logistics company) vs `ARND3`'s small holding-company fundamentals, which raised
  the scale mismatch but wasn't sufficient on its own. With no side of either group positively
  identifiable and the vendor's live backend independently confirming it can't disambiguate them
  either, all 5 (`ATOM3`, `MBLY3`, `LVTC3`, `ARND3`, `PORT3`) were quarantined together rather than
  guessing a winner — same "no reliable source found" policy as `WDCN3`/`CAMB4`/`LLIS3`/`CCTY3`.
- **`GFTT3`/`GFTT4` — not corruption, vendor stub data.** Both are a single identical 2-row
  placeholder (open=high=low=close=5.00 exactly, round volumes) from 2001–2004. Never reaches the
  final dataset (`< MIN_PRICE_ROWS`), so the regression guard's row-count floor now matches
  `MIN_PRICE_ROWS` and no longer flags it.
- **This is the same failure class as WDCN3/CCTY3** — nothing in the pipeline compares one
  ticker's series against another's — now caught by the regression guard on every future build.

### 🟠 C2 — Non-positive `adj_close` pollutes raw-adj price technicals
`compute_price_features` masks `adj_close<=0 → NaN` **only for the log path** (`adj = ...where(>0)`,
used by `log_return`/`overnight_gap`/`intraday_return`). Every *other* technical reads raw
`g["adj_close"]`, so the 360 rows where `adj_close == 0.00` (BIOM3, LUXM4, NUTR3, UNIP6; 2000–2015,
the 2-dp precision-floor underflow) emit **real-looking garbage**:

- `drawdown == −1.0` (fake −100% drawdown) on 62 rows.
- `ma_20`/`ma_60` dragged toward 0 (min 0.0), and their **rolling windows stay contaminated for the
  next 19/59 rows** after each zero.
- `rsi_14 → 0`, `price_vs_ma20 → 0`, `price_percentile_{1y,5y}` rank a 0 as the window minimum (and the
  5y window keeps it for 5 years).

Confined to 4 deep-history microcaps but silent and unflagged. (`hl_ratio`/`log_return` correctly NaN.)
Note: the zeros are **interior, not a leading prefix** (BIOM3/NUTR3/UNIP6), so the fix must route
derived features through the masked series — masking the *shipped* `adj_close` column would create
interior NaN holes and break the prefix-NaN invariant (`test_final_dataset::T_prefix_rule`).

---

## 3. Low-severity risks / observations

- 🟡 **L1 — Precision-floor quantization inflates deep-history `log_return`.** 2,583 rows with
  `|log_return|>1.0` (max +5.60 / min −9.13); most are the C1 duplicates, the rest are microcaps whose
  `adj_close` is pinned at 0.01→0.02→0.03 steps (each step is a >40–70% "return"). Only 6 of these are
  caught by `adj_close_precision_degraded`, because the flag marks the *pinned* row but not the
  *transition* row that carries the fake return. Once C1 is quarantined the tail shrinks ~80%.
- 🟡 **L2 — `beta_1y` outliers.** Core distribution is fine, but min −9.72 / max 13.07 from
  thin/low-overlap tickers (near-degenerate market-variance windows). No `inf`, but economically
  implausible values reach the model. Optional: NaN/flag beta where the window's market-return
  variance is degenerate, or clip to a sane band.
- 🟡 **L3 — Split-repair has no persistence guard (by design).** The single-day jump/tolerance matcher
  can't distinguish a permanent split from a coincidental large in-window move; both guard designs were
  reverted (see `repair.py` comment). Data shows the matcher isn't over-firing — 2,703 transient
  `|lr|>0.35` spikes that reverse next day were correctly left un-repaired — but there's no regression
  test asserting a repaired jump actually *persisted*. Residual risk, not an observed defect.
- 🟡 **L4 — Interior NaNs in gap-guarded derived features are expected, not error-NaN** —
  `revenue_growth_yoy` (6,956), `roe_qoq` (8,139), `cagr_earnings_5y_final` (66,054). These come from
  the calendar-gap guards and CAGR-undefined (negative-earnings) cases, all flagged
  (`cagr_*_defined`, `n_quarters_available`). Confirm `T_prefix_rule` scopes to merged raw fundamentals
  only and does **not** assert prefix-shape over these columns (would be a false failure).
- 🟡 **L5 — Extreme ratios preserved by policy** (`pl` up to 1.6e6, `earnings_yield` 9.2e4,
  `payout_ratio` 4.4e3). Documented as intentional (near-zero denominators = distress signal); scaler is
  robust. Noted for completeness, no action.

---

## 4. Fixes & pytest assertions

### C1 — detect duplicate price series + quarantine impostors
Landed as `check_no_duplicate_price_series()` / `_cnpj_alias_pairs()` in
`test_universe_integrity.py` (§3.6, wired into `run_all.py`'s DATA group). Tolerance-based
(`np.allclose`, not exact hash — `ARND3`/`PORT3` only match to ~5e-9, not bit-for-bit), bucketed
by `(row count, first date, last date)` to stay O(n), excludes same-CNPJ pairs (real aliases) and
`QUARANTINED_TICKERS` (already handled).

All 6 tickers (`BAHI3`, `ATOM3`, `MBLY3`, `LVTC3`, `ARND3`, `PORT3`) resolved and added to
`quality_filters.QUARANTINED_TICKERS` — `BAHI3` via the dividend-corroboration evidence above;
the other 5 via yfinance (no data for 4 of 5, weak 0.27 correlation for `ARND3`) plus a live
BolsAI API cross-check (`get_price_history` misresolves `ATOM3`/`MBLY3`/`LVTC3`→`WDCN3` and
`PORT3`→`ARND3` right now, confirming a live vendor bug rather than a stale collection artifact).
The guard now passes: `VALIDATION PASSED` on `test_universe_integrity.py`.

### C2 — route price technicals through the masked series
In `features.py::compute_price_features`, `adj` is already the masked series — reuse it (do **not**
touch the shipped `adj_close` column):

```python
g["ma_20"]         = adj.rolling(20).mean()
g["ma_60"]         = adj.rolling(60).mean()
g["price_vs_ma20"] = adj / g["ma_20"]
g["price_vs_ma60"] = adj / g["ma_60"]
g["hl_ratio"]      = (g["adj_high"] - g["adj_low"]) / adj
g["drawdown"]      = (adj - adj.cummax()) / adj.cummax()
g["rsi_14"]        = _rsi(adj, 14)
```
and in `compute_advanced_features`, rank on a masked copy for the percentile rolls:
```python
adj = g["adj_close"].where(g["adj_close"] > 0)   # add at top of the per-ticker loop
# ...then rank `adj` (not g["adj_close"]) in price_percentile_1y / _5y
```
Assertion:
```python
def test_nonpositive_adjclose_yields_nan_technicals(df):
    z = df[df["adj_close"] <= 0]
    for c in ["ma_20", "drawdown", "rsi_14", "price_vs_ma20",
              "price_percentile_1y", "price_percentile_5y"]:
        assert z[c].isna().all(), f"{c} computed off a non-positive adj_close"
```

### L2 — beta sanity (optional)
```python
var = g["_mkt_log_return"].rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).var()
g["beta_1y"] = (cov / var).where(var > 1e-8)   # degenerate-variance window → NaN, not a wild beta
```

### L3 — persistence regression test (optional, guards future repairs)
For each event `repair_unadjusted_splits` fixes, assert the post-jump price level holds
(median of next ~20 rows within, say, 25% of the jump-day close) so a coincidental one-day move
can't be silently "repaired".

---

## Action checklist
- [x] C1: land duplicate-price-series regression guard, CNPJ-aware + tolerance-based (`test_universe_integrity.py` §3.6).
- [x] C1: resolve + quarantine `BAHI3` (dividend-corroboration evidence; `CGRA3`'s series is genuine).
- [x] C1: resolve `ATOM3`/`MBLY3`/`LVTC3` and `ARND3`/`PORT3` (yfinance + live BolsAI API cross-check, no winner identifiable either group) and quarantine all 5.
- [x] C1: `GFTT3`/`GFTT4` — confirmed vendor stub data, not corruption; test row-count floor now matches `MIN_PRICE_ROWS` so it's no longer flagged.
- [x] C1: regression guard passes (`VALIDATION PASSED`).
- [ ] C1 (optional): once a verified rename date is known, move `ALOS3`/`ALSO3` and `MEGA3`/`SRNA3` into `ticker_continuity.json` proper instead of relying on the CNPJ-match exclusion.
- [x] C2: route price technicals + percentiles through the masked `adj`; regression tests added, fast-group green.
- [x] C2: rebuild landed externally (manifest `git_commit: 4b865ce`) — `drawdown < -0.999` dropped by exactly 62 rows, matching the original audit's zero-`adj_close`-caused count; `BAHI3` confirmed absent (0 rows). `ATOM3`/`MBLY3`/`LVTC3`/`ARND3`/`PORT3` quarantine postdates that rebuild — **dataset needs one more rebuild** to drop these 5.
- [ ] L2: optional beta degenerate-variance guard.
- [ ] L3: optional split-repair persistence test.
- [ ] L4: verify `T_prefix_rule` excludes gap-guarded derived columns.
```
