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

### 🔴 C1 — Four byte-identical price series across *different* companies
Structural dedup on `(ticker, trade_date)` cannot catch this; it's cross-*ticker* duplication.

| Pair | rows | shared span | close(last) | Notes |
|------|-----:|-------------|------------:|-------|
| `BAHI3` ≡ `CGRA3` | 4319 | 2000-01-10 … 2026-07-10 | 25.00 | Bahema vs Grazziotin — unrelated |
| `ATOM3` ≡ `MBLY3` | 2307 | 2017-03-23 … 2026-07-10 | 2.37 | Atom vs Mobly |
| `MEGA3` ≡ `SRNA3` | 973 | 2021-12-27 … 2025-11-13 | 12.62 | — vs Serena |
| `ARND3` ≡ `PORT3` | 1156 | 2021-10-25 … 2026-07-10 | 0.63 | — vs Portobello (CANCELADA) |

- **Only the price files are identical.** Each pair's *fundamentals* differ (shapes 62/62 but
  values differ; 61 vs 26; 20 vs 18; 18 vs 20). So the impostor ticker carries its **own real
  fundamentals bolted onto a copied price series** — i.e. one raw `prices/*.parquet` is a copy
  of the other company's.
- **Impact is concentrated in the extreme tail.** `BAHI3`+`CGRA3` alone are **79%** of all 2,583
  `|log_return|>1.0` rows; these two drive `volatility_20d` up to 1.48 and `return_12m` up to 3.4.
  Every derived price/return/vol/beta feature for all 8 tickers is fabricated for at least one of
  each pair.
- **Impostor is identifiable by listing date vs series start.** e.g. Mobly (`MBLY3`) IPO'd Feb 2021
  and its own fundamentals start 2019-12-31, yet its price series starts 2017-03-23 = Atom's listing
  → `MBLY3`'s price file is the copy. Same logic (series-start vs company IPO/first-filing) resolves
  the others; finalize each with a yfinance cross-check before quarantining.

**This is the same failure class as WDCN3/CCTY3 but undetected** — nothing in the pipeline compares
one ticker's series against another's.

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
Durable guard (fails the build/test if any recur), in `tests/data_collection/` or `test_universe_integrity.py`:

```python
def test_no_duplicate_price_series():
    """No two distinct tickers may share a byte-identical OHLCV price series."""
    import hashlib, pandas as pd
    sigs = {}
    for f in (RAW / "prices").glob("*.parquet"):
        g = pd.read_parquet(f).sort_values("trade_date")
        cols = ["trade_date", "open", "high", "low", "close", "volume"]
        h = hashlib.md5(
            pd.util.hash_pandas_object(g[cols], index=False).values
        ).hexdigest()
        sigs.setdefault(h, []).append(f.stem)
    dupes = [v for v in sigs.values() if len(v) > 1]
    assert not dupes, f"identical price series across tickers: {dupes}"
```

Then, after confirming each impostor via yfinance (series-start vs company IPO/first-filing), add the
copied-price tickers to `quality_filters.QUARANTINED_TICKERS` with the reason
(e.g. `"MBLY3": "raw price file is a byte-identical copy of ATOM3; Mobly IPO'd 2021 but series starts 2017-03-23 = Atom's listing"`).
`MBLY3` is already positively identified; `CGRA3/BAHI3`, `MEGA3/SRNA3`, `ARND3/PORT3` need the
cross-check to pick which side is the copy.

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
- [x] C1: land duplicate-price-series regression guard (`test_universe_integrity.py` §3.6, commit `476585c`).
- [ ] C1: yfinance-confirm impostor in each pair; add copied-price tickers to `QUARANTINED_TICKERS`; rebuild.
- [x] C2: route price technicals + percentiles through the masked `adj`; regression tests added (commit `f00c6f1`).
- [ ] C2: rebuild `ml_dataset.parquet` so the fix reaches the shipped dataset (code fix alone doesn't touch `dataset_v1`).
- [ ] L2: optional beta degenerate-variance guard.
- [ ] L3: optional split-repair persistence test.
- [ ] L4: verify `T_prefix_rule` excludes gap-guarded derived columns.
```
