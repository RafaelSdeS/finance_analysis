# `ml_dataset_top50_universe.parquet` — Pre-Training Audit

**Status update (2026-07-16, same day):** §1.1, §1.3, §1.4 fixed in source
and covered by new/extended regression tests
(`test_features.py::test_ratio_columns_nan_when_denominator_near_zero`,
`test_top50_universe.py`'s zero-fill checks, new `test_loaders.py`) — all
passing (`tests/run_all.py --group all`: 32/32). §2 (scaler) verified in
place, not modified.

**Rebuild completed and re-verified against the new
`ml_dataset_top50_universe.parquet` (191,001 rows, 165 cols, 111 tickers):**

| Finding | Before | After rebuild |
|---|---|---|
| §1.1 (`revenue_per_earning` etc.) | max up to 6.4×10¹⁵ | max 91,284 (`revenue_per_earning`) down to 300 (`earnings_yield`) — all traced to genuine small-but-nonzero denominators (e.g. real `pl=-0.0033`, real `roe=-0.000079` during distress quarters), not epsilon-guard artifacts. 0 rows above 1e6 abs outside the pre-existing, documented raw `pl` outlier bucket. |
| §1.2 (`zhist` cols) | 152 cols, missing | 165 cols, all 13 `*_zhist_5y` present |
| §1.3 (zero-fill) | ~19 cols 100% NaN on `has_fundamentals==0` | 0 remaining NaN on all checked columns |
| §1.4 (`div_yield_12m`) | max 1546 (154,600%) | max 2.19 (219%), 0 rows above 500% |

One nuance worth recording: `corr(pl, pvp_to_roe_ratio)` is still ≈1.0
post-fix — not a bug this time, just Pearson correlation still being
sensitive to a handful of genuinely-extreme co-occurring distress rows
(e.g. `AXIA6` 2022-11, `pl≈-1.1M` and `pvp_to_roe_ratio≈-11,400` on the same
rows, same real distress event) even at bounded, four-digit magnitude.
**Verdict: ready.**

---

Written 2026-07-16. Scope: the point-in-time rolling top-50 universe file
(`build_top50_universe.py`'s output), not the fixed-list universe covered by
`TOP50_ML_READINESS_AUDIT.md` / `TOP50_INDEPENDENT_AUDIT.md` — those two
audited `ml_dataset.parquet` filtered to a static 50-ticker list. This file
is a different artifact: 191,001 rows, 111 tickers (union across all
quarterly rebalances 2011–2026), zero-filled fundamentals where
`has_fundamentals==0`, **completely unscaled** (see §2).

## Verdict

**Not ready for training as currently materialized.** One critical bug
(§1.1, same class as an already-fixed one, unfixed in 5 sibling columns)
produces feature values up to **1.5 quadrillion** in the actual file on
disk. A second file-staleness issue (§1.2) means the file predates the most
recent per-ticker z-score feature addition. Both require a source fix
and/or rebuild before training.

---

## 1. Data quality audit

### 1.1 CRITICAL — 5 more `x / (y + 1e-8)` ratios blow up to quadrillion scale; the prior fix only covered one of six siblings — FIXED 2026-07-16 (pending rebuild)

`TOP50_INDEPENDENT_AUDIT.md` (finding 1) found and fixed
`dividend_coverage_ratio`'s `+1e-8` epsilon guard producing values up to
2.3×10¹⁵, then (finding 7) grepped every other `+1e-8` guard, measured each
denominator's zero-rate (1–3%), and concluded they were all "legitimate rare
distress signals" like `CLAUDE.md`'s documented `pl`-outlier policy — and
left them unfixed.

Measured directly against the actual materialized data (both this file and
`ml_dataset.parquet`), that conclusion doesn't hold. Five ratios show the
exact same failure mode as the fixed one:

| Column (`features.py`) | Denominator | Denom exact-zero rate | `\|value\|>1000` rows | Max `\|value\|` (this file) | Max (`ml_dataset.parquet`) |
|---|---|---|---|---|---|
| `revenue_per_earning` | `net_income` | 3.33% | 2.91% of non-null | **6.36×10¹⁵** | 1.45×10¹⁶ |
| `pvp_to_roe_ratio` | `roe` | 3.24% | 2.64% | **7.86×10¹¹** | 3.99×10¹⁴ |
| `payout_ratio` | `lpa` | 3.43% | 2.05% | **7.13×10⁷** | 2.40×10¹⁰ |
| `ebitda_margin` | `net_revenue` | 1.11% | 0.30% | **4.22×10¹³** | 2.91×10¹⁴ |
| `earnings_yield` (+ `earnings_yield_vs_selic`) | `pl` | 0.95% | 0.14% | **1.00×10⁸** | 1.00×10⁸ |

Worst offenders are not distressed penny stocks — e.g. `revenue_per_earning`'s
max is **BBDC3** (Bradesco, a top-5 bank) on 2017-04-28..05-03 where
`net_income` printed exactly `0.0`; `pvp_to_roe_ratio`'s worst is also BBDC3
(2013-10). The reasoning that "near-zero denominator = meaningful distress
signal" (valid for raw `pl` itself, per `CLAUDE.md`) doesn't transfer to
these *derived* ratios: unlike `pl` — where a large value has graduated
economic meaning (expensive relative to tiny-but-positive earnings) — a
value like `7.86×10¹¹` here carries no information beyond "the denominator
was numerically indistinguishable from the `1e-8` epsilon." All 5 are in
`scale_features.py`'s `RATIO_COLUMNS`, so wherever scaling is eventually
applied, `RobustScaler`'s linear transform won't clip these (same reasoning
`TOP50_INDEPENDENT_AUDIT.md` finding 1 already made for
`dividend_coverage_ratio`).

Also note: **the correlation table in §3 below is unreliable for any column
downstream of this bug** — `pl`/`pvp_to_roe_ratio` show `corr=1.0` in the raw
computation, which is very likely the shared-outlier-row artifact of this
exact bug (a handful of 1e11+ rows dominate a Pearson correlation), not a
genuine relationship. Re-run correlation analysis after the fix.

**Fix:** same pattern as the already-applied `dividend_coverage_ratio` fix —
set each ratio to `NaN` when its denominator isn't strictly bounded away
from zero (e.g. `abs(denom) < 1e-6`), instead of dividing by `denom + 1e-8`.
Applies to `features.py` lines 406 (`payout_ratio`), 424
(`revenue_per_earning`), 432 (`ebitda_margin`), 534 (`pvp_to_roe_ratio`), 537
(`earnings_yield`). Then rebuild `ml_dataset.parquet` and re-run
`build_top50_universe.py`.

**FIXED 2026-07-16:** added a shared `_safe_ratio()` helper in `features.py`
(NaN when `|denominator| <= 1e-6`) and applied it at all 6 sites above, plus
`peg_ratio` (line 531 — not in the original table but the identical
anti-pattern in the same code block, skew 32 / max ~11,423 in §3's
distribution pass; fixed alongside the others rather than left next to the
fix). Regression test:
`tests/build_dataset/test_features.py::test_ratio_columns_nan_when_denominator_near_zero`
(36/36 tests in that file pass). **Not yet reflected in the parquet on
disk** — requires the rebuild command above, not run as part of this
source fix per standing instructions.

### 1.2 HIGH — this file predates the R1 per-ticker z-score features; it's a stale build

`ml_dataset_top50_universe.parquet` was written 2026-07-15 16:30, but
`ml_dataset.parquet` was rebuilt again at 2026-07-15 17:36 (commit
`05db484`, adding 13 `*_zhist_5y` columns per-ticker own-history z-scores).
Confirmed directly: this file has 152 columns, `ml_dataset.parquet` has 165
— the 13-column difference is exactly the `zhist` set
(`pl_zhist_5y`, `pvp_zhist_5y`, `roe_zhist_5y`, `amihud_illiquidity_zhist_5y`,
etc.), entirely absent here. If the RL agent is meant to see these
features, **`python -m src.build_dataset.build_top50_universe` needs to be
re-run** — it's a stale snapshot, not a bug in the filtering logic itself.

### 1.3 MEDIUM — zero-fill policy is inconsistent across sibling fundamental-derived features — FIXED 2026-07-16 (pending rebuild)

`build_top50_universe.py::zero_fill_missing_fundamentals()` zero-fills ~60
named columns whenever `has_fundamentals==0` (1,553 rows, 0.81% of the
file) — correctly, verified 0 remaining NaN in that list for those rows.
But it misses a second tier of fundamental-derived columns computed
downstream (cross-sectional / Piotroski / trend features), which stay
**100% NaN** on those same 1,553 rows:

`market_cap`, `filing_lag_days`, `days_since_fundamental`,
`pl_zscore_sector`, `pvp_zscore_sector`, `roe_zscore_sector`,
`debt_equity_zscore_sector`, `pl_percentile_5y`, `f_roa_positive`,
`f_roa_improving`, `f_margin_improving`, `f_leverage_decreasing`,
`f_liquidity_improving`, `f_score`, `had_negative_earnings_5y`,
`roe_trend_4q`, `margin_trend_4q`, `debt_trend_4q`, `roa_trend_4q`.

Net effect: for the same 1,553 rows, the agent would see e.g. `roe=0.0`
(zero-filled, "looks like a real observation") sitting next to
`roe_zscore_sector=NaN` and `f_score=NaN` in the identical row — an
inconsistent missing-data signal for a state that's conceptually identical
across all these columns. Small in row-count (0.81%) but every one of those
rows carries ~19 live NaNs that will fail a downstream
`assert np.isfinite(obs).all()` unless something else fills them.
**Fix:** either extend `zero_fill_missing_fundamentals()`'s column list to
cover these, or (cleaner, since some are legitimately not just
"has_fundamentals" gated — see §1.4) confirm the downstream consumer's own
NaN policy handles them.

**FIXED 2026-07-16:** extended `zero_fill_missing_fundamentals()`'s column
list with all ~19 columns above (same mask, same mechanism). Regression
test added to `tests/build_dataset/test_top50_universe.py` (11/11 pass).
Not yet reflected in the parquet on disk — requires re-running
`build_top50_universe.py`.

### 1.4 LOW — `div_yield_12m` still has non-sane values after the "365-day window" fix, traced to one corrupted raw ticker — FIXED 2026-07-16 (pending rebuild)

`TOP50_INDEPENDENT_AUDIT.md` finding 2 fixed the window from 252 calendar
days to 365 — correct, but doesn't address a separate issue: `div_yield_12m`
still reaches **1546** (154,600% trailing yield) in this file, all 554
rows `>500%` traced to a single ticker, **PDGR3**. Root-caused directly:
`data/raw/dividends/PDGR3.parquet` has `value_per_share` in the
**hundreds of millions** for all 5 of its recorded dividend events (e.g.
168,557,520 on 2012-05-09, `type="UNKNOWN"`) — a raw-vendor unit/labeling
error (a real per-share BRL dividend is cents-to-low-single-digits, not
9 figures), not a formula bug. Confirmed isolated: scanned all 523
`data/raw/dividends/*.parquet` files, **only PDGR3** has any
`value_per_share > 1000`; 100% of its own 5 events are affected.
**Fix:** quarantine `PDGR3`'s dividend records (or the ticker entirely, if
it's not already handled elsewhere) pending a source data investigation —
not a code fix, a data-quality exclusion like the existing
`QUARANTINED_TICKERS` pattern.

**FIXED 2026-07-16:** added a generic sanity ceiling to
`loaders.py::load_dividends()` (drops any `value_per_share > 1000`, logged,
not silent) rather than a hardcoded ticker name — also catches a future
recurrence of this same vendor failure mode on a different ticker.
Regression test: `tests/build_dataset/test_loaders.py` (new file,
registered in `run_all.py`'s FAST group). Not yet reflected in the parquet
on disk — requires the rebuild command in §5.

### 1.5 Checked clean — structural integrity

- **0** duplicate `(ticker, trade_date)` pairs, **0** full-row duplicates.
- **0** weekend rows.
- **0** tickers with non-monotonic `trade_date` ordering.
- **0** columns with literal `+/-Inf` (the existing `clean_dataset()` inf→NaN
  pass holds).
- **0** constant columns except `adj_close_precision_degraded` (a known,
  intentional flag — only fires for the already-fixed MBRF3 precision bug,
  so 0 elsewhere is expected, not a bug).
- **0** near-constant (>99.5% single value) numeric columns.
- `rsi_14` correctly bounded [0, 100]; `status`/`sector` categorical, sane
  cardinality (2 / 37 values); `f_score` correctly bounded [0, 5] (Piotroski
  convention) outside the §1.3 NaN gap.
- NaN pattern across the other 90+ NaN-bearing columns matches the
  documented prefix/warm-up shape from `CLAUDE.md` (CAGR needing 5y history,
  sector z-scores needing peer/rolling warm-up, filing-lag/day-count fields)
  — nothing new beyond §1.3/§1.4.
- `log_return`: max `|value|` = 1.74 (a ~470% single-day move — extreme but
  plausible for a penny/distressed name over 15 years across 111 tickers;
  only 2 rows exceed `|log_return|>1`), consistent with
  `TOP50_ML_READINESS_AUDIT.md`'s existing per-ticker outlier findings.
- `beta_1y`: range [-0.67, 2.86], sane.

### 1.6 Episode continuity (informational, not a bug — an RL design consideration)

By construction (rolling top-50 membership, union across periods), 34 of
111 tickers have at least one gap >100 days between their retained rows
(e.g. `PSSA3` has a 3,288-day gap — dropped out of the top-50 for ~9 years,
then re-qualified). This is **correct** point-in-time behavior per
`TOP50_UNIVERSE_VALIDATION.md`'s design (not a data bug — the underlying
per-ticker feature history was computed continuously before filtering, so
no feature discontinuity is introduced), but whatever RL environment
consumes this file needs to treat a ticker's retained rows as
**non-contiguous episodes**, not a single continuous series — a training
loop that naively feeds consecutive rows per ticker without checking for
these gaps would silently splice unrelated time periods together.

---

## 2. Scaling and normalization

**This file receives no scaling at all.** `scale_features.py` (this repo's
`RobustScaler` fit) reads/writes only `ml_dataset.parquet` /
`data/processed/scalers/` — `build_top50_universe.py` has no scaler import
and never calls it. So every column here is in its raw, native unit:

- The 58 `RATIO_COLUMNS` (pl, pvp, margins, growth rates, etc.) are
  unscaled ratios/percentages, roughly O(1)-O(100) in the typical case but
  with the §1.1 blowups reaching 1e8-1e15 in the untreated tail.
- 13 raw-BRL-level columns (`market_cap`, `net_income`, `equity`,
  `net_revenue`, `total_debt`, `ebitda`, `ebit`, `net_debt`, `cash`,
  `total_assets`, `current_assets`, `current_liabilities`,
  `shares_outstanding`) span **6-8 orders of magnitude** across the
  universe (e.g. `market_cap` median 3.6e10, max 9.9e12 — a 278x
  median-to-max ratio; `net_debt` up to 427x). These are absolute levels
  that scale with company size, not stationary ratios — a shared-weight
  policy network processing all 111 tickers' rows will see the same column
  read ~10¹⁰ for a small-cap and ~10¹² for VALE3/ITUB4-scale names in the
  same feature slot.
- `rsi_14` (already [0,100]), percentile features (already [0,1] by
  construction), and z-score features (`*_zscore_sector`, already
  standardized) correctly need **no further scaling** — leaving them
  unscaled is correct, not an oversight.

**Update 2026-07-16, scope narrowed per user instruction:** whether/how this
file gets scaled downstream is out of scope here — not something this repo
or this audit resolves, and not something to infer from any other branch.
The scaler not being applied to `ml_dataset_top50_universe.parquet` right
now is expected, not a defect. What *was* verified, read-only, staying on
this branch only:

- `scaler_metadata.json`'s `scaled_columns` (58 entries) matches
  `scale_features.py`'s `RATIO_COLUMNS` exactly, same order.
- `fit_window.fit_end` (`2018-07-30`) matches `split_config.json`'s
  `train_end` exactly — no lookahead in the fit.
- Every fitted `center`/`scale` value (RobustScaler median/IQR) is finite
  and non-zero, including for the 6 columns fixed in §1.1 —
  `revenue_per_earning` center=7.65/scale=18.58, `pvp_to_roe_ratio`
  center=0.19/scale=0.64, `payout_ratio` center=0.047/scale=0.31,
  `peg_ratio` center=0.0005/scale=1.29, `earnings_yield`
  center=0.016/scale=0.052 — all sane in magnitude for what these ratios
  mean, not still stretched by the pre-fix outlier tail (RobustScaler's
  median/IQR are resistant enough that a <3% astronomical tail doesn't
  visibly move them — confirmed directly, not assumed).
- `feature_scaler.joblib` / `scaler_metadata.json` mtimes are fresh
  relative to the rebuilt `ml_dataset.parquet` (no stale-scaler repeat of
  §1.2).

**Conclusion: the scaler artifacts this repo produces are internally
correct and valid for the 58 columns they cover, verified in place.** They
were fit before today's §1.1 code fix, so re-fitting after the next
rebuild (§5) will still be worth doing, but is not fixing a defect in the
current artifacts — it's just refreshing them against corrected inputs.

One additional, unavoidable subtlety if a `StandardScaler`/`RobustScaler` is
fit downstream on this exact file: because `zero_fill_missing_fundamentals()`
maps "not yet reported" to a literal `0.0` for ~60 columns (1,553 rows,
0.81%), any global mean/std or median/IQR fit over this file will be
(slightly, given the small fraction) pulled by those synthetic zeros mixed
in with real economic zeros — same class of consideration `CLAUDE.md`
already documents for the consumer-side NaN→0 policy, just worth restating
here since this file is exactly where that mixing happens.

---

## 3. Feature distributions (highlights; full detail in audit script output)

Beyond the §1.1 blowups, skew/kurtosis is otherwise in line with what's
expected for financial ratio data (heavy right tails on growth-rate and
leverage columns are real, not artifacts) — e.g. `revenue_growth_yoy`
(skew 56), `total_debt_growth_yoy` (skew 53) reflect genuine occasional
large swings (a small-cap's revenue doubling YoY), not corruption. No
column showed near-zero variance. `amihud_illiquidity`'s skew (249) is
expected for a liquidity metric that's near-zero for most large-cap/day
observations with a long right tail on thin-trading days — already a
documented, intentional distress-style signal.

**Recommendation:** don't blanket-transform (log/winsorize) every skewed
column — most of this skew is real economic content the agent should see.
After fixing §1.1, re-run this distribution pass; several of the "extreme
skew" columns in the current report (`pl`, `payout_ratio`,
`pvp_to_roe_ratio`, `earnings_yield`) are almost certainly dominated by the
bug, not genuine tail risk, and should look much tamer once fixed.

---

## 4. Feature engineering review — redundancy

21 column pairs show `|correlation| > 0.97` (computed on the current,
pre-§1.1-fix file — re-run after the fix, per the caveat in §1.1). Legitimate,
expected redundancy, not bugs:

- `cagr_revenue_5y` / `cagr_revenue_5y_final` and `cagr_earnings_5y` /
  `_final` (1.0 — the `_final` variant is the CAGR-backfilled version of the
  same metric, intentionally kept side-by-side per `CLAUDE.md`)
- `selic` / `cdi` (0.9999 — well-known near-identical Brazilian policy/interbank
  rates)
- `log_return` / `excess_return` / `real_return` (0.999+ — definitionally
  close, differ only by small macro adjustments)
- `ma_20` / `ma_60`, `ebitda` / `ebit`, `ev_ebit` / `p_ebit`,
  `current_assets` / `current_liabilities` — naturally correlated by
  business/accounting structure, not identical

**Likely bug-driven (re-check after §1.1 fix):** `pl` / `pvp_to_roe_ratio`
(corr 1.0), `earnings_yield` / `earnings_yield_vs_selic` (1.0), `pl` /
`revenue_per_earning` (0.987) — all downstream of the unfixed epsilon guards.

No action recommended for the legitimate pairs — an RL policy network can
absorb some redundant inputs, and de-duplicating here trades a cheap,
easily-learned redundancy for a meaningfully increased risk of removing a
feature the agent actually needs in some regime. Worth revisiting only if
training is compute/sample-constrained enough that trimming the observation
space measurably helps.

---

## 5. Time-series consistency

All confirmed clean: monotonic per-ticker ordering, no lookahead (re-verified
`merge_asof(direction='backward')` semantics apply upstream, unchanged by
top50 filtering), no fabricated trading days. The only structural
discontinuity is the expected, by-design membership-gap behavior (§1.6).

---

## 6. RL-specific considerations

- **Observation-space conditioning:** was poor, driven by §1.1
  (quadrillion-scale outliers, now fixed in source, pending rebuild) and the
  10-order-of-magnitude raw-currency columns (§2 — unscaled by design, not
  this repo's decision to make).
- **Non-stationarity:** real and expected (macro rate regime shifts across
  2011-2026, sector composition drift) — not a preprocessing defect; an RL
  env training across this whole span should account for it in its own
  design (e.g. curriculum, regime features already present via
  `selic`/`cdi`/`rate_environment`-style columns), not something this
  dataset should paper over.
- **Sample efficiency risk:** the §1.3 NaN inconsistency (0.81% of rows with
  a partial zero-fill) is small enough to be unlikely to meaningfully hurt
  sample efficiency on its own, but combined with whatever the RL env does
  with residual NaNs (drop the row? crash the assert? silently propagate?)
  it's worth resolving rather than assuming it's harmless.
- **Episode construction:** per §1.6, 34/111 tickers have real membership
  gaps >100 days — the env's episode/windowing logic must not treat a
  ticker's retained rows as one continuous series across such a gap.

---

## 7. Final assessment

### Overall readiness
**Fixed in source, not yet ready as materialized on disk.** §1.1, §1.3,
§1.4 are all fixed and test-covered (2026-07-16) — §1.2 was already fixed by
the rebuild that preceded this fix pass. But none of today's source fixes
retroactively change the parquet files already on disk; a rebuild (§5) is
still required before training. §2 (scaler) was checked and found correct
for what it currently covers — no code change there.

### Issues by severity
1. **CRITICAL, FIXED 2026-07-16** — §1.1: 6 `+1e-8` epsilon-guard ratios
   blew up to quadrillion scale (`revenue_per_earning`, `pvp_to_roe_ratio`,
   `payout_ratio`, `ebitda_margin`, `peg_ratio`,
   `earnings_yield`/`earnings_yield_vs_selic`). Fixed via a shared
   `_safe_ratio()` helper in `features.py`; regression-tested.
2. **HIGH, FIXED** — §1.2: file was stale, missing the 13 `*_zhist_5y`
   columns; resolved by the rebuild that happened before this fix pass.
3. **MEDIUM, FIXED 2026-07-16** — §1.3: zero-fill policy extended to the
   ~19 missing sibling columns in `build_top50_universe.py`; regression-tested.
4. **LOW, FIXED 2026-07-16** — §1.4: `loaders.py::load_dividends()` now
   drops implausible `value_per_share` rows (>1000, logged) — a generic
   ceiling rather than a `PDGR3`-specific quarantine, so it also catches a
   future recurrence on a different ticker; regression-tested.
5. **Checked, not a defect** — §2: scaler artifacts verified correct for the
   58 columns they cover (column list, fit window, finite/sane
   center/scale). Whether/how this file is scaled downstream of this repo
   is out of scope.

### Recommended next step before training
- [ ] Rebuild: `python -m src.build_dataset.build_ml_dataset` →
      `python -m src.build_dataset.scale_features` →
      `python -m src.build_dataset.build_top50_universe` — materializes all
      four 2026-07-16 fixes (§1.1, §1.3, §1.4) into the actual training
      files and re-fits the scaler against the corrected data.
- [ ] `python tests/run_all.py --group all` to confirm the rebuild is clean.
- [ ] Spot-check per §5 below that the previously-astronomical columns now
      read in a sane range.

### Nice-to-have (optional)
- Re-run the §4 correlation pass after the §1.1 fix to get a trustworthy
  redundancy picture (current numbers are likely outlier-inflated for the
  affected columns).
- Consider surfacing §1.6's membership-gap table as a per-ticker
  `days_since_last_row` or explicit episode-boundary flag in the file
  itself, so any consumer gets a positive signal instead of having to
  re-derive gaps from `trade_date` diffs.

### Risks if trained against the file currently on disk (pre-rebuild)
- The §1.1 outliers are exactly the kind of finite-but-astronomical value
  that evades `inf`-checks and survives a linear scaler untouched —
  `TOP50_INDEPENDENT_AUDIT.md` already flagged this failure mode as a
  plausible source of gradient explosion / NaN loss for the (previously
  fixed) `dividend_coverage_ratio`; the fix for the other 6 columns is now
  in source but hasn't reached the parquet on disk yet.
- The ~19 §1.3 columns still read NaN (not zero-filled) on `has_fundamentals
  ==0` rows in the current file, and `div_yield_12m` still reaches 154,600%
  for `PDGR3` until the rebuild runs.

### Conclusion
All source fixes (§1.1, §1.3, §1.4) are in and regression-tested; §2 was
checked and needs no change. **Rebuild
(`build_ml_dataset.py → scale_features.py → build_top50_universe.py`) before
training** — that's the only remaining step to reach a ready dataset.
