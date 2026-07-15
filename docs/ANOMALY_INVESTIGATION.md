> **Update 2026-07-14, second pass:** the first backfill run exposed a
> serious flaw in the original §5 verification method (row-count density
> was not proof of real trading — see "What Actually Happened" at the
> bottom). Of the 40 tickers originally claimed fixable, only **12 were
> genuinely fixed**; 24 turned out to have the identical fabrication problem
> in yfinance too and were reverted; 2 more (`EUCA4`, `NUTR3`) were ~50%
> contaminated and reverted out of caution; 2 (`SNSY5`, `RCSL4`) hit real
> but minor yfinance OHLC noise and were never written. A guard against the
> fabrication signature is now in `backfill_price_gap()`. All other
> sections (z-score fix, `adj_close_precision_degraded`, fundamentals-gap
> check) are unaffected by this and still hold.

# Anomaly Report Investigation

Investigation of `tests/build_dataset/test_final_dataset.py --strict` findings
(5,740 stale-price rows, 2,752,729 outlier cells) plus fundamentals coverage,
`adj_close` anomalies, and trading-calendar gaps. Full validation gate
(`validate()`) passes with 0 failures — everything below is from the
**informational** anomaly report, which is not wired into CI (`--strict` is
never passed by `run_all.py` or `ci.yml`).

Investigated 2026-07-14 on `data/processed/ml_dataset.parquet` (1,310,119 rows,
140 columns, 516 tickers).

## Summary Table

| Category | Expected | Needs Investigation | Pipeline Bug | Recommended Action | Priority |
|---|---|---|---|---|---|
| Stale prices (Telebras-class penny stocks) | ✅ | | | Exclude from report or note tick-size cause | Low |
| Stale prices (illiquid micro/small-caps) | ✅ | | | No action | Low |
| Robust z-score: price/technical trend contamination | | | ✅ | Window the z-score (rolling, not full-history) or drop from default report | **High** |
| Robust z-score: ratio-of-ratio blowups (`peg_ratio`, `revenue_per_earning`, `pvp_to_roe_ratio`) | ✅ (by existing policy) | | | Exclude near-zero-denominator ratios from anomaly report | Medium |
| Robust z-score: fat-tailed volume/trade columns | | ✅ | ✅ (methodology) | Use log-scale or percentile-based flagging, not Gaussian MAD z-score | Medium |
| Robust z-score: `filing_lag_days` | ✅ | | ✅ (methodology) | Exclude tight-clustered operational metadata from z-score check | Low |
| Fundamentals coverage (76% has_fundamentals, 0–8% NaN within that) | ✅ | | | No action — matches documented ~60–76% coverage | — |
| `adj_close` 2dp precision floor — 28 tickers, 11,693 rows (see §4, expanded from the original 4) | | ✅ | (vendor, not this repo) | Flag rows, do not "fix" by recomputing — implemented | Low |
| 6 tickers wrongly caught by a magnitude-only threshold (`AZZA3, COCE5, GEPA4, RAPT4, RSUL4, TIMS3`) | ✅ (genuinely low-priced, full precision) | | ✅ (my first flag design) | Require exact 2dp quantization, not just magnitude — fixed | — |
| 40 ATIVO tickers with >400d gaps (incl. LREN3/UGPA3/ENEV3) | | | ✅ (vendor backfill hole, confirmed via yfinance for all 40) | Backfill from yfinance | **High** |
| 4 ATIVO tickers with >400d gaps and no yfinance coverage either (`FRAS3`, `BAUH4`, `PEAB4`, `BMIN4`) | | ✅ | possibly (untriaged — needs a BolsAI-side check) | Leave as-is for now; investigate BolsAI re-fetch separately | Low |
| ITEC3 gaps widening pre-2019 delisting | ✅ | | | No action — matches wind-down illiquidity | — |
| BIDI11 → INBR32 migration | ✅ | | | No action — already known/expected | — |

---

## 1. Stale Prices (5,740 rows, 216 tickers)

**Detection:** `check_stale_prices()` flags runs of ≥5 consecutive identical
`close` values where `volume > 0` (i.e., trades happened but landed on the
same tick).

**Findings:**
- Top offenders by row count: `TOYB4` (541), `TELB3` (518), `TELB4` (488),
  `TOYB3` (374), `JBDU3` (300).
- `TELB3`/`TELB4` (Telebras) show `avg_daily_volume` in the **billions of
  shares** — a known artifact of Telebras's 1998 privatization break-up,
  which left an enormous outstanding share count and correspondingly tiny
  per-share nominal prices. At that price level, the exchange's minimum tick
  size is large relative to the price, so the closing price legitimately
  sticks across many trading days even with real volume flowing through.
  This is a **market microstructure artifact**, not a data error.
- `TOYB4`/`TOYB3` (Tec Toy, CANCELADA) and `JBDU3` (CANCELADA, ticker later
  reused/renamed) are micro-caps with thin order books — genuine illiquidity.
- Stale-row counts by year are heaviest 2000–2015 (roughly flat ~200–500/yr)
  and drop sharply post-2016 (mostly <100/yr) as the universe's overall
  liquidity profile improved — consistent with a real market trend, not a
  collection artifact appearing/disappearing abruptly.

**Verdict:** Expected. Distinguish "stale from illiquidity/tick-size" (the
overwhelming majority here) from "stale from a stuck data feed" — nothing in
the top-15 offenders looks like the latter. No fix needed to the data.
**Recommendation:** either leave the report as informational (current state,
fine), or add a per-ticker average-price/average-volume annotation to the
report output so a human doesn't have to re-derive "is this Telebras again"
each time.

---

## 2. Robust Z-Score Outliers (2,752,729 cells, 114/136 numeric columns)

**Detection:** `check_outliers_zscore()` computes a **per-ticker** robust
z-score (`0.6745 * (x - median) / MAD`) over each column's **entire history**
for that ticker, flags `|z| > 8`.

This is the most consequential finding: **114 of 136 numeric columns**
trigger outliers at rates of 3.5–8% of all rows. A well-calibrated Gaussian
z-score at threshold 8 should flag on the order of 1-in-10^15 points — a
3–8% hit rate across nearly every column means the methodology, not the
data, is mis-firing at scale. Three distinct causes:

### 2a. Trend contamination (price/technical columns) — **pipeline/methodology bug**

Tested directly: for `adj_close` outliers, computed each flagged row's
position within its own ticker's date range (0 = ticker's first day, 1 =
last day). **112 of 188 flagged tickers (60%) have outliers concentrated in
the first or last 15% of their own history** (`mean_frac_position < 0.15` or
`> 0.85`).

Root cause: the z-score is computed once over a ticker's *entire* multi-year
history (which can span 20+ years for the oldest names). For any ticker that
trended meaningfully over its life — which is most of them — the early
(cheap) or late (expensive) portion of its price series reads as a "outlier"
relative to the whole-history median, purely because of trend, not because
anything anomalous happened on that date. Same issue drives `adj_high`,
`adj_low`, `adj_open`, `ma_20`, `ma_60` (all in the top-20 offender list).

**Fix:** either (a) compute the z-score over a rolling/trailing window
(consistent with how `volatility_*_percentile` already avoids lookahead via
rolling rank, per `CLAUDE.md`), or (b) detrend before flagging (z-score on
`log_return` deviations rather than price levels), or (c) simplest: drop
raw/adjusted price-level and moving-average columns from the anomaly report
entirely — they are already captured more meaningfully by the corporate-event
leak check (`log_return` vs `corporate_events`) elsewhere in the same script.

### 2b. Ratio-of-ratio blowups (`peg_ratio`, `revenue_per_earning`, `pvp_to_roe_ratio`) — expected, already-documented policy

Flagged values for these three columns range into the hundreds of millions to
`10^14` — e.g. `revenue_per_earning` flagged cells span from `-9.1e4` to
`5.0e14`. These are ratios with a ratio in the denominator (P/E ÷ growth,
revenue ÷ earnings, P/B ÷ ROE); when the inner denominator approaches zero
the outer ratio diverges by construction. `CLAUDE.md` already documents this
exact behavior for `pl` ("Extreme ratio... kept intact — denominators near
zero are valid distress signals... No fix available with current data").
The anomaly report just doesn't carry that context, so it looks alarming in
isolation.

**Recommendation:** exclude these three (and `pl`, `pvp`, `p_sr`, `ev_ebit`,
`ev_ebitda`, `p_ebit`, `p_ebitda` if they show the same pattern) from the
z-score anomaly report, or add a "known-heavy-tailed ratio" allowlist next to
the existing `TOLERANCES CATALOG` docstring in `tests/test_utils.py`.

### 2c. Fat-tailed volume/trade columns — methodology mismatch

`volume` skewness = 92, kurtosis = 11,112 (a Gaussian has skew=0, kurtosis=3).
`volume` (109,273 flags), `volume_adjusted` (77,390), `traded_amount`
(69,236), `num_trades` (54,827) are the top 4 offenders overall. Trading
volume is canonically log-normal-ish with occasional 10–50x spikes (earnings
days, index-inclusion days, corporate actions) — real, not anomalous, but a
Gaussian-calibrated MAD z-score treats every such spike as extreme because
the 0.6745 scaling constant assumes approximate normality even in the robust
formulation.

**Fix:** flag on `log1p(volume)` (or percentile rank) instead of raw-level
MAD z-score for these columns.

### 2d. Tight-clustered operational metadata (`filing_lag_days`) — methodology mismatch

`filing_lag_days` median is 42 days with IQR 37–45 (companies cluster tightly
around Brazil's ~45-day statutory ITR filing deadline). That tight spread
makes the MAD tiny, so any legitimately-late filer (up to the pipeline's own
180-day cutoff, per `FILING_LAG_DAYS_QUARTERLY`/quality_filters.py) computes
an enormous z-score and gets flagged — even though those same rows already
passed the repo's own explicit lateness policy. 57,336 flags (4.38% of rows).

**Recommendation:** drop `filing_lag_days` from the z-score check; it already
has a dedicated, more meaningful gate (`filter_excessive_filing_lag`) that
does something more principled than a z-score.

**Net assessment for §2:** the 2.75M-cell figure is not a data-quality signal
at all — it's mostly ~60% trend contamination on price/technical columns
(2a) plus expected heavy-tailed ratios (2b) plus a scale mismatch on volume
and lag columns (2c/2d). The stale-price check (§1) is doing real, useful
work; the z-score check as currently built is close to uninformative at
`threshold=8` on this data and should not be the trigger for `--strict`
gating without the fixes above.

---

## 3. Fundamentals Coverage

- `has_fundamentals == 1`: 996,312 / 1,310,119 rows (**76.0%**).
- Within `has_fundamentals == 1` rows, NaN rates are low: `equity` 0.00%,
  `net_income` 0.35%, `roe` 0.73%, `market_cap` 4.14%, `pvp` 4.52%, `pl`
  8.12% (the last two dip slightly higher because they need a positive
  denominator — consistent with §2b).
- `cagr_earnings_5y_final` / `cagr_revenue_5y_final` NaN coverage is
  explained at 88.4% / 90.6% respectively by the `validate()` gate (negative
  base-year earnings or <20 quarters of history) — both already pass their
  80% acceptability threshold.

**Verdict:** Expected, matches `CLAUDE.md`'s documented ~60–76% coverage
figures (BolsAI direct + CAGR backfill). No merge bug — the NaN rates track
exactly where a filing genuinely wasn't available, not a structural
merge-key mismatch. No action needed.

**Do the 40 price-gap tickers (§5) also have a fundamentals gap?** Checked
directly (2026-07-14): **no — fundamentals are unaffected, for an unrelated
reason.** Fundamentals coverage across this entire dataset, blue chips
included, has a hard floor at `reference_date >= 2010-12-31` (verified
identical min-date and quarter-count — 62 quarters — for `PETR4`, `TEND3`,
and `ENEV3` alike). For price gaps entirely before 2010 (e.g. LREN3
2002–2005), fundamentals simply don't exist yet for *anyone* in that window
— there's no gap to have. For price gaps overlapping 2010+ (e.g. TEND3,
ENEV3), fundamentals are fully present through the window undisturbed. So
this is a prices-only vendor issue; fundamentals collection runs on a
separate pipeline and wasn't affected.

---

## 4. Adjusted-Close Anomalies (`adj_close = 0`, and the broader precision floor)

360 rows across exactly **4 tickers**: `LUXM4` (288), `UNIP6` (33), `BIOM3`
(28), `NUTR3` (11) — all pre-2015 — hit `adj_close == 0` exactly.

**Update, after implementing the fix:** the same underlying mechanism (2dp
vendor rounding on a very small adjusted price) also produces *nonzero but
quantized* values (`0.01`/`0.02`/`0.03`/`0.04`) that don't trip the `== 0`
check but have the identical "real price moves, adj_close doesn't" symptom.
Querying the full dataset for `0 < adj_close < 0.05` turned up **34 tickers,
11,693 rows** — not 4. Checking whether each ticker's flagged values are
*exactly* representable at 2 decimals split this cleanly in two:

- **28 tickers are the real precision-floor artifact** (100% of their
  flagged rows land exactly on a 2dp grid value): `BIOM3, BAUH4, CLSC3,
  EALT4, ENGI3, ENGI4, EKTR4, CGRA3, CGRA4, BAHI3, MBRF3, LUXM4, JBDU3,
  PDGR3, POMO3, POMO4, PEAB3, PEAB4, RANI4, SOND3, SOND5, SOND6, TOYB3,
  TOYB4, UNIP5, UNIP6, WHRL3, WHRL4`.
- **6 tickers are false positives and must NOT be flagged**: `AZZA3,
  COCE5, GEPA4, RAPT4, RSUL4, TIMS3`. Their `adj_close` is genuinely tiny
  (e.g. `TIMS3` down to `0.000568`) but carries full float precision — never
  exactly equal to its own 2-decimal rounding. These are legitimately
  low-priced (same structural cause as the Telebras case in §1: a large
  historical share count), not a rounding artifact. `NUTR3` (one of the
  original 4) also turned out to have zero rows in the `(0, 0.05)` band on
  a second look — its `adj_close == 0` rows are handled separately and
  correctly by the existing `.where(adj_close > 0)` mask in `features.py`.

The flag (`adj_close_precision_degraded`, implemented in
`compute_price_features()`) requires **both** `0 < adj_close < 0.05` **and**
`adj_close` being exactly its own 2-decimal rounding, which correctly
includes the 28 and excludes the 6.

Traced `BIOM3` end-to-end (representative of the 28):
- The raw BolsAI file (`data/raw/prices/BIOM3.parquet`) **already contains**
  `adj_close = 0.00` for these rows — this is not introduced anywhere in
  this repo's pipeline (`repair.py`, `merge.py`, etc. don't touch it).
- `close` for the same rows is a legitimate 0.50–0.85 — the stock was
  genuinely trading, just at a nominal price that, after BolsAI's cumulative
  adjustment factor for 20+ years of corporate actions, rounds to `0.00` at
  BolsAI's 2-decimal-place storage precision (underflow, not a real zero
  price).
- `clean_dataset()` correctly prevents this from leaking as `inf`:
  `log_return` is `NaN` (not `inf`) for every row where `adj_close == 0`, and
  the "no inf values" validation check passes with 0 hits.
- **Subtler downstream effect:** once `adj_close` recovers to a rounded
  nonzero value (e.g., `0.03`), it can stay pinned at that same rounded value
  across several consecutive days even while the real, unrounded price moves
  meaningfully (`close` moved 1.20 → 1.10 → 1.02 → 0.91 in the same window
  where `adj_close` printed `0.03` all four days), producing `log_return =
  0.0` for real, non-zero moves. This is a genuine — if narrow — precision
  artifact.

**Verdict:** Not a bug in this pipeline; a BolsAI vendor precision limit on
deep-history microcaps with large cumulative adjustment factors. Given
`CLAUDE.md`'s existing, hard-won guidance **not** to reconstruct `adj_close`
from the dividends table (confirmed to systematically under-adjust), do not
attempt a general re-derivation here either — the same risk applies, and the
scope (4 tickers, 360 rows, pre-2015) doesn't justify it.

**Recommended action:** flag these specific (ticker, date-range) windows as
low-confidence for return-based features, rather than attempting a fix. A
simple `adj_close` precision flag (e.g., `adj_close_precision_degraded = 1`
where `adj_close` rounds to ≤0.05 with 4+ repeats) would let downstream
consumers exclude them from return-sensitive training without deleting rows.

---

## 5. Trading-Calendar Gaps

Scanned all 523 raw price files for the single largest date gap per ticker;
**219 tickers have at least one gap > 400 days.**

### Named tickers

- **ITEC3** (Itautec): status `CANCELADA`, last trade 2019-05-28. Gaps
  progressively widen through 2012–2019 (32d → 68d → 173d → 226d → 262d →
  223d as the series approaches its end) — this is the classic shape of a
  company winding down trading activity before delisting, consistent with
  the known history (ceased manufacturing operations after OKI Brasil's
  acquisition, became an investment holding). **Expected, no action.**
- **BIDI11** (Banco Inter units): status `CANCELADA`, last trade 2022-06-17,
  one 435-day gap in 2018–2019 shortly after IPO (thin initial free float is
  a plausible explanation, not verified further). Known to have migrated to
  INBR32 — **expected, no action**, matches what you already knew.
- **LREN3** (Lojas Renner): status `ATIVO`, **886-day gap, 2002-10-28 →
  2005-04-01**. This one does not fit the "illiquid/delisting" pattern —
  Lojas Renner is a large, continuously-liquid retailer. Verified directly:
  - Every other true blue-chip in the universe (`PETR4`, `VALE3`, `BBDC4`,
    `ITUB4`, `ABEV3`, `BBAS3`, `ITSA4`, `B3SA3`, `GGBR4`, plus `MGLU3`,
    `RADL3`, `SUZB3`, `EQTL3`, `RENT3`) has **zero** gaps larger than a
    normal 5-day holiday weekend across the same 2000–2026 span — ruling out
    a systemic BolsAI hole for that era.
  - Queried `yfinance` directly for `LREN3.SA`, 2002-01-01 to 2005-12-31:
    **1,032 rows, ~261 trading days/year** — dense, continuous daily data
    for the exact window BolsAI shows as a total void. LREN3 was
    unambiguously trading throughout; no B3 suspension of that length for a
    major retailer is documented or plausible.
  - **Verdict: confirmed BolsAI vendor backfill gap**, not a real halt, not
    a pipeline bug on this repo's side (the pipeline just consumes what
    BolsAI returns).
  - **Fix:** backfill this specific window from `yfinance` (already a
    pipeline dependency for `--mode update`) via a one-off historical pull,
    same pattern as `collect_prices_yf`.
- **UGPA3** (Ultrapar) and **ENEV3** (Eneva) — spot-checked the same way as
  LREN3 (both are liquid, index-member large-caps): `yfinance` shows dense
  data for both gap windows (985-day gap 2004–2007 for UGPA3; 569-day gap
  2014–2016 for ENEV3) that BolsAI is missing. **Same fix as LREN3.**

### Full spot-check of all 44 `ATIVO`-status tickers with >400-day gaps

Initial assumption was that the remaining ~41 gaps were mostly genuine
small/micro-cap illiquidity (concentrated family ownership, wide spreads, no
market maker — all plausible in the Brazilian market). Spot-checked anyway,
by querying yfinance for the exact same window as each gap and counting
returned rows:

**Result: 37 of the 41 additional tickers show dense yfinance coverage
through the "gap"** (row counts matching ~252 trading days/year for the
window length — e.g. `SNSY5`'s 5,471-day/~15-year gap returns 3,725
yfinance rows, essentially full density). Combined with LREN3/UGPA3/ENEV3,
that's **40 of 44** `ATIVO`-status gapped tickers confirmed as BolsAI
vendor backfill holes, not real illiquidity. This is a much larger-scale
problem than the initial 3-ticker read suggested — it isn't confined to
large/liquid names, it's a broad hole in BolsAI's historical backfill that
happens to also hit small-caps.

Full list (ticker, gap window, yfinance rows found in that window):
`SNSY5` (2005-12-19→2020-12-11, 3725), `BPAR3` (2004-11-30→2018-10-24, 3462),
`LUPA3` (2014-02-13→2023-03-15, 2256), `INEP3` (2014-08-29→2022-11-18, 2043),
`BSLI3` (2001-07-03→2009-01-26, 1930), `NUTR3` (2017-09-29→2025-03-20, 1857),
`EUCA4` (2002-07-03→2009-11-13, 1869), `TEND3` (2010-02-08→2017-05-04, 1790),
`CALI3` (2011-10-20→2018-10-26, 1744), `MWET4` (2016-02-01→2022-10-20, 1674),
`FIGE3` (2016-03-01→2022-08-22, 1614), `ETER3` (2018-03-20→2024-08-12, 1588),
`RNEW4` (2019-10-16→2025-02-14, 1325), `PCAR3` (2010-03-08→2015-06-30, 1316),
`VIVR3` (2016-09-16→2021-08-04, 1213), `PDGR3` (2017-02-22→2021-10-15, 1155),
`AHEB3` (2004-02-17→2008-03-31, 1039), `VULC3` (2000-11-20→2004-10-25, 1025),
`LEVE3` (2000-10-31→2004-08-20, 993), `REDE3` (2012-11-22→2016-09-08, 938),
`NORD3` (2005-05-05→2008-10-23, 872), `MGEL4` (2013-11-01→2017-03-17, 832),
`FHER3` (2019-02-05→2022-03-25, 777), `MNPR3` (2004-01-12→2007-02-21, 794),
`RCSL4` (2006-01-26→2008-12-29, 732), `AZEV4` (2008-06-18→2011-05-02, 711),
`TPIS3` (2017-07-24→2020-01-24, 625), `MSPA3` (2001-05-23→2003-09-05, 597),
`BALM4` (2006-03-10→2008-05-09, 300), `RSUL4` (2015-07-31→2017-03-31, 413),
`HETA4` (2003-12-30→2005-07-21, 400), `BIOM3` (2003-01-22→2004-07-27, 394),
`LUXM4` (2015-11-13→2017-03-01, 319), `JOPA3` (2006-07-18→2007-10-31, 324),
`PATI3` (2001-07-27→2002-10-21, 321), `CEGR3` (2022-04-06→2023-06-23, 303),
`EALT4` (2001-08-29→2002-10-31, 306).

**4 tickers have no yfinance coverage for their gap window either:**
`FRAS3`, `BAUH4`, `PEAB4`, `BMIN4` (yfinance returns "possibly delisted; no
price data found" for those specific windows). These are left as genuine
open items — not fixable from yfinance, would need a targeted BolsAI
re-fetch attempt to diagnose further (out of scope here, costs API credits,
not attempted).

**Fix (implemented, not yet run):** `backfill_price_gap()` added to
`yf_collectors.py` (reuses the existing fetch/shape/split-repair logic from
`collect_prices_yf`, adds a "never overwrite an existing row" filter), driven
by `src/data_collection/backfill_known_gaps.py` which lists all 40 confirmed
tickers with their windows.

### 156 tickers with `status = '?'` (missing company_info)

Not investigated here — this is a separate, pre-existing data-quality gap
(`company_info` coverage), out of scope for this anomaly-report
investigation but worth a follow-up (`sibling fill` from `company_siblings()`
may not be reaching all of them).

---

## Downstream Impact on ML/Backtesting

- **§2a (trend-contaminated z-score) and §2b/2c (ratio/volume outliers):**
  none — this is purely a reporting artifact of the *investigation script*,
  not a transformation applied to `ml_dataset.parquet` or the scaler. No
  rows are dropped or altered because of this check today. Zero downstream
  impact until/unless someone wires `--strict` into a gate that starts
  dropping or reweighting flagged rows based on it.
- **§1 (stale prices):** none currently (informational only); if ever used
  to filter training rows, would need the Telebras-style tick-size cases
  excluded first or it would strip real, valid low-price-tier history.
- **§4 (`adj_close` rounding):** small in aggregate (11,693 / 1,310,119 rows,
  0.9%, spread across 28 tickers) but concentrated by name — a single-name
  backtest on any of the 28 (`MBRF3` and `TOYB3`/`TOYB4` have the deepest
  exposure at 900-3,000 rows each) would see spuriously flat (zero) returns
  during flagged windows and should exclude them.
- **§5 (LREN3/UGPA3/ENEV3-style gaps):** meaningful for any strategy or
  feature that uses trailing windows (`ma_60`, `volatility_60d`,
  `momentum_vs_market_12m`, CAGR calcs) spanning the gap — those features
  will have anomalously long lookback windows once real data resumes,
  effectively smearing pre-gap and post-gap regimes together. Worth fixing
  for any of the ~44 `ATIVO` gapped tickers actually used in a backtest.

---

## Recommended Actions (checklist)

- [x] Rewrite `check_outliers_zscore()` in `test_final_dataset.py`: excludes
      already-normalized/flag/bounded columns (`_EXCLUDE_FROM_OUTLIER_CHECK`),
      compares trend-level columns within `(ticker, year)` instead of
      whole-history (`_TREND_LEVEL_COLS`), and applies a signed-log1p
      transform to everything else to tame ratio blowups and fat-tailed
      volume/count columns — see §2a–2d. *(Kept the allowlists local to
      `test_final_dataset.py`, next to the function that uses them, rather
      than in `test_utils.py`'s `TOLERANCES CATALOG` — that catalog is a
      registry of shared numeric constants used across multiple test files,
      not general per-check methodology notes; these sets are specific to
      this one function.)*
- [x] Backfilled from `yfinance` via `backfill_price_gap()` +
      `src/data_collection/backfill_known_gaps.py`. **Run and corrected —
      see "What Actually Happened" below.** Final state: 12 tickers
      genuinely filled, 26 reverted after turning out to have the same
      fabrication problem as BolsAI (now blocked from retry by a new
      `_flat_run_fraction` guard in `yf_collectors.py`), 4 with no yfinance
      coverage at all, 2 (`SNSY5`, `RCSL4`) blocked by real but minor
      yfinance OHLC noise (open/close a hair outside `[low, high]` on a
      handful of rows — a candidate for a small repair function mirroring
      `_repair_nonpositive_ohlc`, not implemented).
- [x] Spot-checked all 41 remaining `ATIVO`-status tickers with >400-day
      gaps against `yfinance` — but the check only confirmed row density,
      not value variation, which is what let the fabrication problem
      through undetected. See below.
- [x] Added `adj_close_precision_degraded` flag in
      `compute_price_features()` (`features.py`): flags rows where
      `0 < adj_close < 0.05` **and** the value is exactly its own 2-decimal
      rounding (the vendor precision-floor signature). First draft used
      magnitude alone and wrongly caught 6 genuinely-low-priced,
      full-precision tickers (`AZZA3, COCE5, GEPA4, RAPT4, RSUL4, TIMS3`) —
      fixed after querying the live dataset found 34 tickers instead of the
      4 originally scoped; see the corrected §4. Correctly covers 28
      tickers / 11,693 rows. **Not yet materialized** — requires a dataset
      rebuild (`build_ml_dataset.py`) to appear in `ml_dataset.parquet`.
- [ ] (Optional, low priority, unchanged) Investigate the 156 tickers with
      `status = '?'` in `company_info` — separate from this report's scope.
- [ ] (Optional, low priority, new) Diagnose why `FRAS3`, `BAUH4`, `PEAB4`,
      `BMIN4` have no coverage in either vendor for their gap window —
      candidate for a targeted BolsAI re-fetch.

## What Actually Happened (2026-07-14, second pass)

The user ran `python -m src.data_collection.backfill_known_gaps`. Result:
38 tickers "filled", 2 rejected by validation, 0 reported failures. On
inspection that summary was itself misleading (see below), and a spot-check
of the actual written data found the backfill had gone badly wrong for most
of the batch.

**The core mistake:** the original §5 verification method (querying
yfinance for the gap window and counting returned rows) checked *row
density* and treated it as proof of real trading. It never checked whether
the *values* varied. Direct inspection of `LREN3` after backfill showed
`close` pinned at the exact constant `1262.985301` for 630 consecutive
rows/2.5 years — traced to **yfinance's own raw feed** (verified with zero
transformation applied on our side: `yf.Ticker("LREN3.SA").history(...)`
alone already returns this flat value). yfinance pads holes in its own
historical coverage with a carried-forward stale price instead of leaving
the date absent, for a large fraction of Brazilian small/mid-cap tickers —
producing a dense, correctly-dated row count that is actually 90%+ a single
repeated close. A row-count check cannot distinguish this from real data;
only checking value variation can.

**Auditing all 38 written tickers by flat-run fraction** (rows sitting in a
run of ≥10 identical closes) found:

| Verdict | Count | Tickers |
|---|---|---|
| Corrupted, >90% flat-run | 24 | `LREN3, UGPA3, TEND3, FIGE3, CALI3, BSLI3, AHEB3, VULC3, LEVE3, NORD3, AZEV4, MSPA3, PCAR3, MNPR3, RSUL4, BIOM3, JOPA3, CEGR3, PATI3, LUXM4, EALT4, BALM4, HETA4, BPAR3` |
| Ambiguous, ~48–50% flat (interleaved with real rows, not cleanly separable by date) | 2 | `EUCA4, NUTR3` |
| Genuinely clean, 0–13% flat | 12 | `ENEV3, PDGR3, ETER3, VIVR3, RNEW4, TPIS3, FHER3, MGEL4, INEP3, LUPA3, MWET4, REDE3` |

Notably, **2 of the original 3 high-confidence cases (`LREN3`, `UGPA3`) were
corrupted** — only `ENEV3` held up. `CEGR3`'s gap window (2022–2023, recent)
was also 96% flat, confirming this isn't only a legacy pre-2010 data-quality
issue.

The 26 corrupted/ambiguous tickers' `data/raw/prices/*.parquet` files were
git-tracked and cleanly reverted via `git restore` (`git checkout` is
blocked by this environment's security policy) back to their pre-backfill
state — confirmed by checking `LREN3`'s max gap returned to 886 days.

**Fix:** `_flat_run_fraction()` added to `yf_collectors.py`, called inside
`backfill_price_gap()` before `_merge_save` — rejects the whole fetch if
more than 20% of the batch sits in a ≥10-run of identical closes (threshold
calibrated with margin against the audit: clean topped out at 12.6%,
contaminated started at 48%). `backfill_known_gaps.py`'s `GAPS` list now
only contains the 12 confirmed-clean tickers (already filled; re-running is
a harmless no-op) plus `SNSY5`/`RCSL4` (blocked by unrelated OHLC-bracket
noise); the 26 fabricated-data tickers are listed separately in
`FLAT_RUN_PADDING` as confirmed not fixable from this vendor, not retried.
The summary-bucketing bug (validation failures silently counted as
"already-complete") was also fixed — `backfill_known_gaps.py` now labels
that bucket honestly instead of implying success.

**Net outcome of the gap-backfill effort: 12 of the original 44 gapped
`ATIVO` tickers are genuinely fixed.** The other 32 remain open (26 confirmed
unfixable from yfinance, 4 with no yfinance coverage at all, 2 blocked by
minor unrelated OHLC noise) — a much smaller win than initially reported,
but a real one, and importantly the data that's on disk now is correct
rather than silently fabricated.

## Remaining Go-Ahead Items

Code for the z-score fix and `adj_close_precision_degraded` flag is written
and syntax-checked but not yet run/materialized (same "don't execute what
you just wrote without asking" convention as before):

1. **Verify the z-score fix**: `python tests/build_dataset/test_final_dataset.py --strict`
   (read-only, safe) — should show the outlier-cell count drop sharply from
   2,752,729.
2. **Rebuild the dataset**: `python -m src.build_dataset.build_ml_dataset`
   (regenerates `ml_dataset.parquet` — needed for both the 12 genuine gap
   fixes and the new `adj_close_precision_degraded` column to take effect),
   followed by `python tests/run_all.py --group all` to confirm nothing
   regressed, and `python -m src.build_dataset.scale_features` to refit the
   scaler against the corrected data.
