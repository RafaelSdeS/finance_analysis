# Feature Set & Scaling Audit

Scope: this repo's Stage 2 output (`ml_dataset.parquet`) only — no model exists yet, so this
evaluates the feature/scaling design in the abstract (what a cross-sectional ML model *would*
see), not against a trained model's actual behavior. Companion to `TOP50_ML_READINESS_AUDIT.md`
(data quality) and `DIAGNOSIS_PLAN.md` (no-alpha investigation) — this doc is about feature
design, not data bugs or the alpha diagnosis itself.

Two normalization mechanisms already exist and are evaluated per-feature against them:
- **`scale_features.py`**: `RobustScaler` (median/IQR) on `RATIO_COLUMNS`, train-split-only fit.
  Fixes fat tails on already-unitless ratios. Does not fix cross-sector comparability.
- **`cross_sectional.py`**: sector z-score/percentile + market/sector momentum, full-universe,
  per-date. Fixes cross-sector comparability for the 5 columns it covers.

Grouped by category where the verdict is identical across the group; singled out where it isn't.

**2026-07-15 update:** user decision — raw/absolute columns (§2, and `lpa`/`vpa` in §3) stay as
model inputs, not just parquet reference columns. They're useful for visual validation and may be
used directly. This retracts this doc's original "exclude from model feature list" recommendation
for those columns (§2, §3's `lpa`/`vpa` note, and Summary list §4 below) — nothing was ever removed
from the pipeline, this only changes the *recommendation* about the model's input list. The
`close/ma_20`-style ratio additions (§ Missing features, now implemented) stand regardless: they're
additions, not replacements, and the raw `ma_20`/`ma_60` levels stay in the dataset either way.

---

## 1. Price technicals (`features.py::compute_price_features`)

| Feature | Verdict |
|---|---|
| `log_return`, `return_1m/3m/6m/12m` | **Keep as-is.** Already scale-free (log-space, cross-ticker comparable), `MAX_RETURN_GAP_DAYS` guard already kills fake cross-gap returns. Correctly passthrough. |
| `hl_ratio`, `drawdown`, `rsi_14` | **Keep as-is.** Already ratio/bounded, already scale-free (`hl_ratio` and `drawdown` divide by the ticker's own price; `rsi_14` bounded [0,100]). Correctly passthrough. |
| `volatility_20d`, `volatility_60d` | **Keep, low-priority optional tweak.** Std of log-returns — already in return-space so reasonably comparable across tickers (unlike a price level), right-skewed but not pathologically fat-tailed. Could log-transform for a marginal skew fix; not urgent, `volatility_*_percentile` (self-relative) already covers the more important normalization need. |
| `ma_20`, `ma_60` | **Kept as model inputs (2026-07-15 decision), plus fixed.** Raw level is still incomparable across tickers in isolation — `price_vs_ma20`/`price_vs_ma60` (`adj_close/ma_20`, `adj_close/ma_60`) were added alongside them (implemented, see Missing features §1) so the comparable signal now exists too; the raw levels themselves stay in the dataset and the model's input list per the user's decision above. |
| `adj_close_precision_degraded` | **Keep.** Binary flag, correctly untouched. |

## 2. Raw OHLCV & absolute size levels

`open/high/low/close`, `adj_open/adj_high/adj_low/adj_close`, `market_cap`, `shares_outstanding`,
`volume`, `num_trades`, and the raw fundamental-statement absolutes (`equity`, `total_assets`,
`cash`, `current_assets`, `current_liabilities`, `total_debt`, `net_debt`, `ebit`, `ebitda`,
`net_income`, `net_revenue`).

**Kept as model inputs (2026-07-15 decision) — not excluded.** None of these are normalized by
either mechanism; `test_final_dataset.py`'s `_TREND_LEVEL_COLS` set (lines 57-63) already
*documents* that these trend with company size/growth. This doc originally recommended excluding
them from the model's feature list — the user overruled that: they stay, useful for visual
validation and possibly as direct inputs too. What *did* ship: every one of the fundamental-
statement absolutes already had a properly normalized derivative living alongside it in the
dataset (`equity`/`market_cap` → `book_to_market`; `net_income`/`market_cap` → `earnings_yield`;
`net_revenue`/`net_income` → margins; etc.) — those derivatives are additional signal on top of the
raw absolutes, not a replacement for them.

`volume`/`num_trades` were the one sub-case with no normalized derivative anywhere in the pipeline.
`repair.py:30-31` said so explicitly: *"volume is not rescaled... no cross-scale volume features
exist yet"*. **Fixed 2026-07-15:** `volume_ratio_20d` (`volume / volume.rolling(20).mean()`) added
in `compute_price_features` — see Missing features §2 (implemented). Raw `volume`/`num_trades`
still stay in the dataset and the model's input list, same as everything else in this section.

## 3. Fundamental ratios (`RATIO_COLUMNS`, `RobustScaler`)

General verdict: **right scaler family.** These are fat-tailed but genuinely unitless
(`pl`, `pvp`, margins, `roe`, `debt_equity`, growth rates, `*_qoq`/`*_trend_4q`), and
`RobustScaler`'s median/IQR fit is the documented, deliberate choice for the extreme-denominator
blowups already characterized in `CLAUDE.md` (144 rows `|pl| > 400,000`, kept intact by design).
No change to the scaler choice for this group as a whole. Four specific exceptions:

- **`lpa`, `vpa` — kept as model inputs (2026-07-15 decision).** These are per-share currency
  absolutes (EPS, book value/share) that inherit the same "arbitrary per-share scale" problem as
  raw price, softened by `RobustScaler`'s outlier-robustness rather than fixed. This doc originally
  recommended dropping them as redundant with `pl`/`pvp`/`earnings_yield` — the user overruled
  that, same as the raw-absolutes decision in §2. No change made; still in `RATIO_COLUMNS`
  (`RobustScaler`), still model inputs, still also used as an intermediate by `payout_ratio`.

- **FIXED 2026-07-15 — `earnings_yield` was computed twice with two different formulas, the
  second silently winning.** `features.py` (`compute_fundamental_features`, pre-merge) used to
  compute `net_income / market_cap`; `compute_advanced_features` (post-`recompute_valuation_daily`)
  computes `1.0 / (pl + 1e-8)`. Confirmed via `build_ml_dataset.py`'s call order: price features →
  macro → **`recompute_valuation_daily`** (re-anchors `pl`/`market_cap` to daily close) →
  **advanced features** (recomputes `earnings_yield`). The first definition's `market_cap` was
  stale (computed before re-anchoring) and got silently overwritten by the second, correctly
  re-anchored one every time — net effect was always correct by accident of ordering, but it was
  two competing definitions of one column name with no test pinning which should win. Removed the
  dead `compute_fundamental_features` definition; only the re-anchored `compute_advanced_features`
  one remains. Regression coverage: `test_earnings_yield_recomputed_from_reanchored_pl` (new) and
  `test_fundamental_features_ratios` (updated to assert the column no longer appears from that
  function alone) in `test_features.py`.

- **Same epsilon-overflow bug as `dividend_coverage_ratio`, confirmed in four more places —
  investigated 2026-07-15, fix recommended for all four.** `667ed8c` fixed `dividend_coverage_ratio`'s
  `+1e-8` epsilon pattern by switching to `.where(denom > 0)` → NaN. The same `+1e-8` pattern is
  still present in `payout_ratio` (line 354), `revenue_per_earning` (372), `peg_ratio` (462), and
  `pvp_to_roe_ratio` (465) — plus the line-468 `earnings_yield` recompute.

  Queried `ml_dataset.parquet` directly for how often each denominator is exactly zero:
  `lpa` (`payout_ratio`) 4.05% (40,600 rows), `net_income` (`revenue_per_earning`) 3.79% (38,266),
  `earnings_growth_yoy` (`peg_ratio`) 0.02% (220), `roe` (`pvp_to_roe_ratio`) 3.44% (34,613), `pl`
  (`earnings_yield`) 0.62% (5,719). None of these approach `dividend_coverage_ratio`'s 27%
  "ordinary case" rate — but frequency turns out not to be the deciding test. The actual
  mechanism: without the epsilon, `x/0` produces `inf`, which `clean_dataset`'s existing
  `inf→NaN` pass (`clean.py:22`) already catches correctly. The epsilon's only effect is turning
  that `inf` into a large-but-*finite* number that escapes inf-cleanup entirely (e.g.
  `net_revenue / 1e-8`) — identical to what made `dividend_coverage_ratio` a bug, independent of
  row count. This is distinct from organic extremes like raw `pl` reaching 400k from a genuinely
  tiny nonzero earnings denominator (`CLAUDE.md`'s accepted "real distress signal, kept intact"
  policy) — no epsilon involved there, so that policy is unaffected.

  **Recommend the same `.where(denom != 0)` → NaN fix for all five.** Impact ranges from 220 rows
  (`peg_ratio`) to ~41K rows (`payout_ratio`) — same bug pattern regardless of scale.

- **CAGR columns (`cagr_earnings_5y_final`, `cagr_revenue_5y_final`)** — keep, `RobustScaler` is
  appropriate; already paired with `cagr_*_defined` flags distinguishing "computed" from
  "undefined" (correctly excluded from the ratio itself, so the flags don't need scaling).

## 4. Cross-sectional (`cross_sectional.py`)

`pl_zscore_sector`, `pvp_zscore_sector`, `roe_zscore_sector`, `debt_equity_zscore_sector`,
`div_yield_sector_percentile`, `momentum_vs_{market,sector}_{1m,3m,12m}` — **keep as-is**, this is
the mechanism that actually solves cross-ticker comparability for the columns it covers, already
covered at length this session (sector z-score, not global normalization, and rolling-safe where
it needs to be). One gap: **valuation z-scores exist only at sector granularity, momentum only at
market+sector.** A `pl_zscore_market`/`roe_zscore_market` (all-universe, not sector-restricted)
doesn't exist — low priority, and possibly redundant once sector granularity is verified (open
question from earlier in this session — how many tickers actually land per sector).

## 5. Percentile/self-relative (`compute_advanced_features`)

`volatility_20d_percentile`, `volatility_60d_percentile`, `price_percentile_5y`,
`pl_percentile_5y`, `drawdown_percentile` — **keep as-is.** Correctly rolling (not global) rank,
so no lookahead; bounded (0,1] by construction, correctly passthrough. One real gap: **all of
these use a 5-year window (`window_252 = 252*5`) except `drawdown_percentile` (1-year,
`window=252`) — inconsistent, and `price_percentile_5y` in particular is missing the standard
1-year ("52-week high/low") version**, which is a distinct, commonly-used signal from the 5-year
one, not a duplicate. See §7.

## 6. Fundamental trend/quality (`f_score` family, `*_trend_4q`, `n_quarters_available`,
`days_since_fundamental`)

**Keep as-is.** Binary flags correctly unscaled; `*_trend_4q` are already diffs (comparable
across tickers, correctly in `RATIO_COLUMNS`); `n_quarters_available`/`days_since_fundamental` are
correctly excluded from scaling (they're metadata about data availability, not a market signal —
already in `test_final_dataset.py`'s `_EXCLUDE_FROM_OUTLIER_CHECK`).

## 7. Macro (`selic`, `cdi`, `ipca`, `excess_return`, `real_return`, `selic_trend_20d`)

**Keep as-is.** Market-wide (identical across all tickers on a date) — cross-sectional
comparability is a non-issue by construction. Already small-magnitude, no scaling needed.

## 8. Static/identifier columns (`sector`, `status`, `cnpj`, `cvm_code`, `ticker`)

**Keep in the dataset, exclude from the model's raw feature list** — already `CLAUDE.md`'s
documented position for `status` (current-day snapshot joined onto every historical row = feature-
level lookahead if used directly) and implicitly for the rest (join keys/metadata, not signals).
No scaling question applies; this is a feature-list-scoping decision for whoever builds the model,
already flagged, not something this audit needs to re-litigate.

---

## Missing features

Ranked by expected impact vs. cost, using the ratio-not-level principle established this session
(`hl_ratio`, `price_percentile_5y` as the existing precedents) rather than introducing a new
normalization mechanism.

**2026-07-15: items 1–6 below all implemented** (plus two not originally on this list, added along
the way — see the notes after item 6). Not implemented: item 7 (market-wide z-scores).

### High priority

1. **`close / ma_20`, `close / ma_60`** (price-vs-trend ratio) — **IMPLEMENTED** as `price_vs_ma20`/`price_vs_ma60`.
   - **Why:** `ma_20`/`ma_60` currently carry zero usable signal in raw form (§1). This is the
     direct fix, and it's the standard technical-analysis form of a moving-average feature.
   - **How:** `adj_close / ma_20`, `adj_close / ma_60` — one line each in
     `compute_price_features`, right after `ma_20`/`ma_60` are computed.
   - **Scaling:** none needed — R$/R$ cancels, dimensionless by construction, hovers near 1.0 for
     every ticker regardless of price level. Passthrough.
   - **Trade-off:** none significant. Optionally keep raw `ma_20`/`ma_60` for other derived calcs,
     just exclude them from the model's feature list (§1/§2 pattern).

2. **Volume relative to its own trailing average** (`volume / volume.rolling(20).mean()`) — **IMPLEMENTED** as `volume_ratio_20d`.
   - **Why:** the dataset's only stated gap of its own accord (`repair.py:30-31`). Raw `volume`
     spans orders of magnitude across the universe (blue chip vs. micro-cap) and is currently
     unusable as a cross-ticker input; "unusual volume relative to this ticker's own recent norm"
     is a standard, well-established signal (volume spikes precede/confirm price moves) that's
     completely absent right now.
   - **How:** per-ticker rolling mean of `volume` over 20 trading days, then the ratio — same
     `groupby("ticker")` pass as `ma_20`, in `compute_price_features`.
   - **Scaling:** ratio is already ~centered near 1.0 and comparable across tickers; still
     right-tailed (spike days), so pair with `RobustScaler` (add to `RATIO_COLUMNS`) rather than
     leaving raw.
   - **Trade-off:** none significant — cheap, uses data already collected.

3. **Turnover ratio** (`volume / shares_outstanding`) — **IMPLEMENTED** 2026-07-15, in `compute_advanced_features` (not `compute_price_features`, since `shares_outstanding` is a fundamentals column merged in earlier).
   - **Why:** a second, complementary volume normalization — "% of the float traded today,"
     comparable across tickers of very different sizes in a way raw volume never can be, and
     distinct information from the trailing-average ratio above (level of liquidity vs. change in
     liquidity).
   - **How:** `volume / shares_outstanding`, in `compute_price_features` (or wherever
     `shares_outstanding` is already merged in).
   - **Scaling:** unitless already; `RobustScaler` for outlier days (thin-float microcaps can spike
     to unusual multiples).
   - **Trade-off:** `shares_outstanding` comes from fundamentals (quarterly cadence) so this
     inherits the same filing-lag characteristics as other fundamental-derived ratios — acceptable,
     same as every other ratio already mixing daily price with quarterly fundamentals.

### Medium priority

4. **`price_percentile_1y`** (52-week high/low position) — **IMPLEMENTED**, same name, in `compute_advanced_features`.
   - **Why:** `price_percentile_5y` exists but the 1-year "distance from 52-week high" framing is
     the more standard, widely-used version and is a genuinely different signal for younger/newer
     listings where 5 years of history doesn't exist yet or dilutes a recent regime change.
   - **How:** same `rolling(window=252, min_periods=1).rank(pct=True)` pattern already used for
     `drawdown_percentile`, applied to `adj_close` — this is inconsistency-fix + gap-fill in one
     (§5 already flags `drawdown_percentile`'s window as the odd one out).
   - **Scaling:** none — bounded (0,1] by construction, same as its 5y sibling.
   - **Trade-off:** correlated with `price_percentile_5y` for tickers with a full 5y history; adds
     value mainly for shorter-history tickers and regime-change detection. Not redundant enough to
     skip, but don't add a 2y/3y version too — diminishing returns.

5. **Volatility regime ratio** (`volatility_20d / volatility_60d`) — **IMPLEMENTED** as `volatility_ratio_20_60`, added to `RATIO_COLUMNS`.
   - **Why:** short-vs-long vol ratio is a standard, cheap "is volatility expanding or
     contracting" regime signal. Currently only the self-relative percentile versions exist, which
     answer a different question ("high vs. my own history") than this ("accelerating vs. my own
     recent trend").
   - **How:** direct ratio of the two already-computed columns — one line, no new rolling window.
   - **Scaling:** ratio, roughly centered near 1.0, right-tailed during vol spikes → pair with
     `RobustScaler`.
   - **Trade-off:** minor collinearity with the two inputs it's built from; standard and accepted
     in technical-analysis feature sets, not a concern here.

6. **Rolling beta vs. market** — **IMPLEMENTED** 2026-07-15 as `beta_1y`, in `cross_sectional.py`
   (252-day window, `min_periods=60`). This was the first rolling window inside the full-universe
   cross-sectional pass (everything else there is a single-date `groupby.transform()`), so it
   needed its own `groupby("ticker")` loop, its own `CROSS_SECTIONAL_INPUT_COLS`/`OUTPUT_COLS`
   wiring (missing the `OUTPUT_COLS` entry would have silently dropped the column in the chunked
   production path only — caught before it shipped), and its own no-lookahead + chunking tests in
   `test_cross_sectional.py` (`test_beta_vs_market_matches_direct_computation`,
   `test_beta_nan_before_min_periods_then_no_lookahead`). Deliberately excluded from
   `test_compute_features_chunked.py`'s chunked-vs-unchunked equality list: that fixture's prices
   are constant daily drift with no noise, so the market series there has exactly zero variance and
   `beta_1y` is `0/0 = NaN` everywhere — correctly caught by that test's own "not trivially all-NaN"
   sanity guard, not a bug.
   - **Why:** the one genuinely missing standard equity-factor feature — `momentum_vs_market`
     (cross_sectional.py) captures relative *return*, but nothing currently captures relative
     *risk exposure* to market moves, which is a distinct and standard factor.
   - **How:** needs the full-universe market return series (already computed as the mean used in
     `momentum_vs_market_*`) — rolling covariance of each ticker's `log_return` against that market
     series, divided by the market series' rolling variance. Belongs in `cross_sectional.py` (needs
     the same full-universe-at-once treatment as momentum) or as a two-pass addition to
     `compute_price_features` if the market series is passed in.
   - **Scaling:** unitless by construction (ratio of covariance to variance), typically 0.5-1.5 for
     most equities — passthrough, no scaler needed.
   - **Trade-off:** the most implementation-effort item on this list (needs market-return series
     threaded through, a rolling window choice, and a sector-of-one-style guard for degenerate
     variance). Worth it, but not a one-line change like the rest — flagging as medium priority on
     cost, not on value.

**Also implemented 2026-07-15, not originally on this list** — raised in a follow-up discussion
about `open`/`high`/`low` specifically (grep confirmed `open`/`adj_open` were never used to derive
anything before this): `overnight_gap` + `intraday_return` (decompose `log_return` into
prior-close→open and open→close components, `overnight_gap` gap-guarded the same way as
`log_return`) and `true_range_ratio` (`max(high-low, |high-prev_close|, |low-prev_close|) /
close`, catches gap days `hl_ratio` misses). All in `compute_price_features`, passthrough (no
scaler — same reasoning as `hl_ratio`/`log_return`).

Also implemented 2026-07-15, second follow-up round: **Amihud illiquidity**
(`|log_return| / (volume * adj_close)`, `compute_price_features`) — price impact per unit of
currency traded, distinct from `volume_ratio_20d` (which only flags unusual volume level, not
price sensitivity to it). Added to `RATIO_COLUMNS` (fat-tailed).

### Lower priority / optional

7. **`pl_zscore_market`, `roe_zscore_market`** (all-universe z-score, not sector-restricted) — only
   worth adding after checking sector granularity (open question from earlier this session); if
   sectors already have healthy peer counts, this is likely redundant with the sector version.

8. **Log-transform on `volatility_20d`/`volatility_60d`** — marginal skew reduction, not required
   given the percentile versions already handle the "is this high/low" framing.

Explicitly **not** recommending: momentum-of-momentum / acceleration features, OBV-style
cumulative volume-price indicators, or additional macro spreads (`cdi - selic` is ~constant in
Brazil, near-zero signal) — speculative complexity without a concrete gap driving them, unlike
everything above.

---

## Summary lists

### 1. Unchanged

- `log_return`, `return_1m/3m/6m/12m`, `hl_ratio`, `drawdown`, `rsi_14`, `volatility_20d/60d`,
  `adj_close_precision_degraded`
- All of `RATIO_COLUMNS` (including `lpa`/`vpa`, per the 2026-07-15 keep-as-inputs decision) —
  `RobustScaler` stays the right choice
- `pl_zscore_sector`, `pvp_zscore_sector`, `roe_zscore_sector`, `debt_equity_zscore_sector`,
  `div_yield_sector_percentile`, `momentum_vs_{market,sector}_*`
- `volatility_20d_percentile`, `volatility_60d_percentile`, `pl_percentile_5y`
- `f_score` family, `*_trend_4q`, `n_quarters_available`, `days_since_fundamental`,
  `cagr_*_defined`
- `selic`, `cdi`, `ipca`, `excess_return`, `real_return`, `selic_trend_20d`
- Raw OHLCV/size absolutes (`open/high/low/close`, `adj_open/adj_high/adj_low/adj_close`,
  `market_cap`, `shares_outstanding`, `volume`, `num_trades`, raw statement lines) — kept as model
  inputs per the 2026-07-15 decision, not excluded
- `sector`, `status`, `cnpj`, `cvm_code` — keep in dataset, exclude from model feature list
  (already `CLAUDE.md`'s documented position for `status`; the only exclusion item that stands —
  it's a lookahead concern, not a redundancy one, so the 2026-07-15 decision doesn't touch it)

### 2. Modified

- [x] `ma_20`, `ma_60`: added `price_vs_ma20`, `price_vs_ma60` (`adj_close/ma_20`,
      `adj_close/ma_60`) alongside the raw levels — both stay as model inputs
- [ ] `drawdown_percentile`: reconcile its 1-year window against `price_percentile_5y`'s 5-year
      window — either both windows are intentional (document why) or one is a bug (not addressed
      this pass — `price_percentile_1y` was added alongside instead of resolving this)
- [x] `earnings_yield`: deleted the stale `compute_fundamental_features` definition; only the
      re-anchored `compute_advanced_features` one remains (fixed 2026-07-15)
- [ ] `payout_ratio`, `revenue_per_earning`, `peg_ratio`, `pvp_to_roe_ratio`, `earnings_yield`:
      confirmed same epsilon-overflow bug as `dividend_coverage_ratio` (investigated 2026-07-15,
      see §3) — replace `+1e-8` with `.where(denom != 0)` → NaN in all five — **not yet fixed**
- [x] ~~`lpa`, `vpa`: exclude from the model feature list~~ — retracted, user decision 2026-07-15:
      keep as model inputs

### 3. New (prioritized)

- [x] `price_vs_ma20`, `price_vs_ma60` — implemented
- [x] `volume_ratio_20d` — implemented, fixes the pipeline's self-documented volume gap
- [x] `turnover_ratio` (`volume/shares_outstanding`) — implemented, in `compute_advanced_features`
- [x] `price_percentile_1y` — implemented
- [x] `volatility_ratio_20_60` — implemented
- [x] `overnight_gap`, `intraday_return`, `true_range_ratio` — implemented (added outside this
      list's original scope, see Missing features note above item 7)
- [x] `amihud_illiquidity` — implemented (added outside this list's original scope, second
      follow-up round)
- [x] `beta_1y` — implemented (`cross_sectional.py`), highest implementation cost on this list
- [ ] `pl_zscore_market`/`roe_zscore_market` — not implemented, only after sector-granularity check

### 4. Remove

Retracted in full (2026-07-15 decision) — nothing is being excluded from the model's feature list.
Raw OHLCV/size absolutes and `lpa`/`vpa` all stay as model inputs, not just parquet reference
columns. The one exclusion that still stands is `sector`/`status`/`cnpj`/`cvm_code` (§8/list 1
above) — that's a lookahead-risk exclusion (`status` is a today-snapshot joined onto historical
rows), not a redundancy one, so it was never part of this retraction.
