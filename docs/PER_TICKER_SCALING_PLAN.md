# Per-Ticker Feature Scaling: Investigation + Implementation Plan

**Status:** Phase 0/1 (R1 + the split-window seam + versioning fix) **implemented, tested, and built** — see the Summary Lists' checkboxes for exactly what's done vs. still open. R2 and R4 remain design-only (see below for why). Originally written 2026-07-15 as a design-only investigation against `dataset_v4` (152 columns, 1,319,349 rows, 515 tickers, 2000-01-03 → 2026-07-14); revised same day to add the evaluation-methodology-agnosticism requirement (§3.5); implemented and rebuilt same day into `dataset_v5` (165 columns — the 13 new `*_zhist_5y` columns — same row/ticker counts) with a matching scaler refit (`scaler_metadata.json` now records the `fit_window` it was produced from).

**Question investigated:** should each ticker be normalized against its own historical distribution instead of the single global scaler in `src/build_dataset/scale_features.py`? PETR4 and a small-cap have incomparable distributions for P/E, margins, market cap, etc.; a pooled `RobustScaler` maps both onto one median/IQR and hides "this value is unusual *for this company*."

**Answer (summary):** yes, the hypothesis is sound — but "replace one scaler with ~515 fitted scalers" is the wrong mechanism. The recommended architecture is layered:

- **R1 (primary):** causal per-ticker rolling normalization computed as *features* in Stage 2 — the hypothesis implemented point-in-time, with zero fitted state, hence zero leakage surface and no cold-start problem.
- **R2:** extended cross-sectional per-date ranks (peer-relative view; aligns with `docs/TODO.md` §6.2).
- **R3:** keep the existing global `RobustScaler` (level view + backward compatibility) — output-identical today, but its fit window becomes *injected from the active split configuration* rather than hardcoded (§3.3).
- **R4 (optional, designed but not recommended as primary):** a fitted per-ticker scaler bank as a single parameter table with a ticker → sector → global fallback hierarchy — only if a stationary per-ticker *transform* turns out to be required downstream.
- **Cross-cutting requirement — evaluation-methodology agnosticism:** the project may move from today's single fixed split to rolling-window, expanding-window, or multi-fold evaluation. R1/R2 are stateless and unaffected by any such change. R3/R4 never hardcode a boundary: every fit takes an injected `(fit_start, fit_end)` window resolved from the active split configuration through one adapter (§3.5), and every persisted artifact is keyed by the window that produced it.

Sections 1–2 establish the current state and the constraints that force this shape. Section 3 gives the design and its rationale. Sections 4–8 cover feature-by-feature treatment, leakage, edge cases, performance, and migration. Section 9 is the risk → test table; section 10 the research comparison; summary checklists close the doc.

---

## 1. Current state (`scale_features.py`, `manifest.py`)

- `build_scaler()` (`scale_features.py:64`): `ColumnTransformer([("robust", RobustScaler(), RATIO_COLUMNS)], remainder="passthrough")`. `RATIO_COLUMNS` (`scale_features.py:44-61`) = **58 unitless ratio/growth/trend columns**; the other **94 columns pass through** (verified against `data/processed/scalers/scaler_metadata.json`).
- `fit_scaler_on_train_split()` (`scale_features.py:85`): fits on all rows with `trade_date <= train_end`, pooled across every ticker. Live `split_config.json`: `train_end = 2018-07-30`, `val_end = 2022-07-26`; rows 703,793 / 297,173 / 318,383. The function reads the one `train_end` straight from `SPLIT_CONFIG_PATH` — a hard coupling to the single-fixed-split methodology that §3.3/§3.5 remove.
- The fit is a **separate, deliberate post-build step** (`python -m src.build_dataset.scale_features`), not part of `build_ml_dataset.py`.
- `sync_dataset_version()` (`manifest.py:150`) snapshots parquet + manifest + split_config into `dataset_v{N}/` — **scalers are not snapshotted** (pre-existing gap, fixed as part of this plan regardless of which options are adopted).

### 1.1 Who actually consumes the scaler — prerequisite check

Grep across the repo: `transform_features()` (`scale_features.py:73`) has **no callers outside its own module and tests**. Nothing in this repo applies the scaler to data at training or inference time. The consumer is the `ml_agent` branch, and `docs/TOP50_INDEPENDENT_AUDIT.md:143` records that its `src/agent/data_pipeline.py` **fits its own scaler** — meaning `feature_scaler.joblib` may currently be a dead artifact.

> **Prerequisite before implementation:** confirm on the `ml_agent` branch whether `data_pipeline.py` loads `feature_scaler.joblib` or refits independently. If it refits, the highest-leverage change is R1/R2 (new *columns* in the parquet, which the agent branch consumes directly through the schema/dtype contract) — and R4 is moot until the agent branch is changed to load persisted scalers at all.

### 1.2 What the global scaler gets wrong (the motivating defect)

- The pooled median/IQR is dominated by whichever tickers have the most train rows — long-history large caps. A pooled `pl` IQR of, say, 15 makes a bank persistently sitting at P/E 5 and a utility persistently at P/E 12 look like two nearby constants; the *within-company* movement (bank re-rating from 5 to 8 — a 60% multiple expansion) is compressed to ~0.2 scaler units.
- Companies have persistent level offsets (sector, capital structure, accounting) that the global transform preserves as static offsets. The model must burn capacity learning each ticker's "normal" before it can detect deviation from it — exactly the information a per-ticker view would hand it directly.
- `docs/notes.md:27` already steers this way: *"Put more emphasis on percentage-based and adjusted features, their variance over time, and their relationship to the corresponding sector (absolute values are often less informative)."*

What the global scaler gets **right** and must not be lost: cross-sectional comparability. A P/E of 5 *is* cheaper than a P/E of 50, and value/quality effects live in that level information. Any design that only normalizes per ticker maps both companies' typical values to ~0 and destroys the level signal. This is why the recommendation is *layered views*, not replacement (§3).

---

## 2. Hard constraints discovered during investigation

These four facts kill the naive design ("fit one RobustScaler per ticker on its train rows, store 515 joblib files") before it starts:

1. **Fit-boundary cold start — intrinsic to fitted per-ticker scalers under *any* temporal evaluation.** At any fit boundary *t*, tickers listed at or shortly before *t* have zero or too-few fit rows, so a per-ticker scaler fit up to *t* simply does not exist for them. Under today's split (`train_end = 2018-07-30`) this is not a corner case: RDOR3 (IPO Dec-2020) is a top-50 name, and the 2020–2021 IPO wave (AMBP3, CASH3, and dozens more) is entirely uncovered. Changing the evaluation methodology does not fix it — expanding-window folds each re-encounter it at their own boundary, and rolling windows make it *worse* (a short trailing fit window also pushes long-listed tickers below minimum-observation thresholds). Any fitted-per-ticker design therefore needs a fallback hierarchy *per fit window* — and the fallback (global or sector stats) is exactly what the design was trying to escape, so a large fraction of the universe would silently get global scaling anyway.
2. **Stale train-era statistics.** A per-ticker scaler frozen on 2000–2018 data measures "unusual vs. that company 2008–2018." Applied to 2026 rows, it drifts: Petrobras pre- and post-2016 are financially different companies; hyperinflation-era survivors changed capital structure entirely. Per-ticker stats have far less data than pooled stats (median ~1,300–4,600 rows/ticker vs 700K pooled), so they are noisier *and* they go stale faster. The user's actual hypothesis — "how unusual is the current value relative to this company's own history" — is a *trailing-window* notion, which a static train-fit scaler does not implement.
3. **Fundamentals are quarterly step functions broadcast to daily rows.** Per-ticker distributional stats over *daily* rows of `pl`, `roe`, etc. are ~63× redundant (each quarterly value repeated across a quarter of trading days) and produce degenerate IQRs for short histories. Any per-ticker stat on fundamental columns must be computed over quarterly observations (per `reference_date`) and then broadcast, mirroring how `features.py::compute_fundamental_features` (`features.py:249`) already operates.
4. **Short/sparse histories are common.** From `ml_dataset.manifest.json` NaN rates: `pl` 29.6%, `market_cap` 26.6%, `shares_outstanding` 23.8% — many tickers have fundamentals-sparse or short histories (the reason `compute_split_dates` (`manifest.py:74`) splits on unique dates, not rows). Per-ticker median/IQR over 8 quarterly observations is a noisy estimator; the design needs explicit minimum-observation thresholds.

One more discovery that *helps*: the repo already contains working precedents for both recommended mechanisms.

- **Causal per-ticker normalization as features:** `volatility_20d_percentile`, `volatility_60d_percentile`, `price_percentile_5y`, `price_percentile_1y`, `pl_percentile_5y`, `drawdown_percentile` (`features.py::compute_advanced_features`, `features.py:392`) are rolling-window per-ticker ranks, built exactly this way to avoid lookahead (guarded by `test_volatility_percentile_no_lookahead`).
- **Cross-sectional normalization:** `pl/pvp/roe/debt_equity_zscore_sector`, `div_yield_sector_percentile`, `momentum_vs_market_*`, `momentum_vs_sector_*`, `beta_1y` (`cross_sectional.py:31`).

R1 and R2 are extensions of proven in-repo patterns, not new machinery.

---

## 3. Recommended architecture: three views per feature, not one transform

For the scale-sensitive feature families the model should see up to three *views*, each carrying information the others destroy:

| View | Carries | Mechanism | Status |
|---|---|---|---|
| **Level** (peer-comparable magnitude) | value vs. all stocks, all time — "P/E 5 is cheap in absolute terms" | global RobustScaler on `RATIO_COLUMNS` (unchanged) | exists (R3) |
| **Own-history** (per-ticker regime) | value vs. this company's trailing distribution — "P/E 8 is *expensive for this bank*" | causal rolling robust z-score per ticker, computed as a feature in Stage 2 | **new (R1)** |
| **Peer-relative** (cross-sectional standing today) | value vs. other stocks *on the same date* — "cheapest decile this quarter" | per-date percentile rank across the universe | partially exists; extend (R2) |

The model (or a feature-selection pass downstream) chooses which view matters per feature; nothing is thrown away. This layering is the standard resolution in cross-sectional equity ML (see §10) and is strictly additive to the current pipeline.

### 3.1 R1 — causal per-ticker rolling normalization (primary recommendation)

For a target column `x`, add:

```
x_zhist = (x_t − rolling_median(x, window, min_periods)) / rolling_IQR(x, window, min_periods)
```

computed **per ticker**, where the window at time *t* contains only observations ≤ *t* (shifted to exclude *t* itself is unnecessary here — including the current value in a ≥8-observation robust window biases the z-score toward 0 negligibly and keeps the implementation a plain `rolling()`; the leakage test in §6 is against *future* data, which this never touches).

- **Fundamental columns:** compute over the per-ticker sequence of *quarterly* observations (dedup by `reference_date`, matching the `q`-frame pattern already used at `features.py:249` and for `n_quarters_available` at `features.py:523`), window = 20 quarters (5y), `min_periods = 8`, then broadcast to daily rows through the existing merge — visibility still governed by `fundamentals_available_date`, so the broadcast inherits the no-lookahead guarantee of the main merge.
- **Daily columns** (volatility, liquidity): compute over daily rows, window = 1260 trading days (5y), `min_periods = 252`.
- **IQR floor:** where rolling IQR = 0 (constant window), emit NaN, not ±inf — `clean_dataset()` (`clean.py:6`) already maps inf→NaN, but emitting NaN directly keeps the prefix-shape property auditable.
- **Naming:** `{col}_zhist_5y` (e.g. `pl_zhist_5y`, `roe_zhist_5y`). Consistent with the existing `{col}_percentile_5y` convention; `zhist` distinguishes from the cross-sectional `_zscore_sector` family.

**Initial column set** (highest information density first; extending later is cheap):

- Fundamentals (quarterly-based): `pl`, `pvp`, `roe`, `net_margin`, `ebitda_margin`, `debt_equity`, `net_debt_ebitda`, `earnings_yield`, `book_to_market`, `current_ratio`, `asset_turnover` — 11 columns.
- Daily technicals/liquidity: `volatility_20d` already has a rolling percentile; add `amihud_illiquidity_zhist_5y` and `turnover_ratio_zhist_5y` — 2 columns.

~13 new columns total in phase 1. Deliberately *not* included: growth/trend columns (`*_yoy`, `*_qoq`, `*_trend_4q` are already differenced — first-order own-history-relative by construction), returns (already unit-free; per-ticker vol-normalization is a possible later addition, not phase 1), bounded columns, flags.

**Why this is the primary mechanism:**
- It implements the hypothesis *literally and point-in-time*: unusualness vs. the company's own trailing history, adapting as the company changes — where a train-fit scaler freezes at 2018.
- **No fitted state** → no scaler files, no inference lookup, no refit procedure, no train/test leakage surface at all. Inference on future data is the same code path as history — and the columns are valid unchanged under *any* evaluation methodology (fixed split, rolling, expanding, multi-fold): re-cutting the split never requires recomputing them.
- **Cold start degrades gracefully:** a newly listed ticker produces NaN until `min_periods` is reached — a *leading prefix* of NaN, exactly the shape the existing NaN policy expects (`test_final_dataset.py::T_prefix_rule`), already explained by `n_quarters_available`, and already handled by the consumer's flag+fill machinery (CLAUDE.md NaN policy).
- **Renames are free:** `apply_ticker_continuity()` (`continuity.py:22`) splices before features are computed, so VVAR3→VIIA3→BHIA3 history is one series when the window runs.

### 3.2 R2 — extended cross-sectional per-date ranks

Add market-wide percentile ranks (uniform [0,1] per date) for: `pl`, `pvp`, `roe`, `debt_equity`, `earnings_yield`, `net_margin`, plus a size rank on `market_cap` (the size factor lives here, not in per-ticker scaling of an absolute currency column). Naming: `{col}_rank_cs`. Implemented in `cross_sectional.py` (requires extending `CROSS_SECTIONAL_INPUT_COLS` — see the Pass-2 constraint in §5). Rank-per-date is outlier-immune and needs no fitted state; it is the Gu–Kelly–Xiu-standard normalization for cross-sectional equity ML (§10).

This is `docs/TODO.md` §6.2 nearly verbatim, and that item is **gated on the M5 diagnosis** — R2 lands in the plan as designed-and-ready, but implementation order should respect the M5 gating already agreed in TODO.md.

### 3.3 R3 — keep the global scaler; make its fit window injected

The global RobustScaler stays, output-identical under the current split config. Reasons: (a) level view, per §1.2; (b) the schema/dtype contract with the `ml_agent` branch stays intact; (c) migration reversibility — every phase of this plan is additive (§8).

Two changes, both methodology-neutral:

- **Boundary injection.** `fit_scaler_on_train_split()` hardcodes reading one `train_end` from `SPLIT_CONFIG_PATH`. Replace with a pure `fit_scaler(dataset, window: FitWindow) -> ColumnTransformer` (no file I/O; boundaries passed in), plus a thin `main()` that resolves fit windows from the active split configuration via the §3.5 adapter and fits one artifact per window. Under today's single-split config that yields exactly one window ending at `train_end` — the same artifact as today; under a future rolling/expanding/multi-fold config it yields one artifact per fold with zero further code change here.
- **Versioning.** Snapshot `scalers/` into `dataset_v{N}/` (fixes the §1 versioning gap).

### 3.4 R4 (optional) — fitted per-ticker scaler bank as a parameter table

Designed here in full so implementation can proceed directly *if* a stationary per-ticker transform is ever required (e.g. the agent team wants per-ticker-standardized *inputs*, not extra feature columns). **Not recommended as the primary mechanism** — constraints 1–2 in §2 (cold start, staleness) apply to it in full and are only mitigated, never removed, by fallbacks.

**Storage: one file, never 515 joblib files.** ~515 tickers × 58 columns × 2 stats is ~60K floats — a single tidy parquet, trivially versioned and diffable:

`data/processed/scalers/scaler_params.parquet`

| column | dtype | notes |
|---|---|---|
| `level` | str | `"ticker"`, `"sector"`, or `"global"` |
| `key` | str | ticker name, sector name, or `"__GLOBAL__"` |
| `column` | str | feature column name |
| `center` | f64 | median over train rows at that level |
| `scale` | f64 | IQR over train rows; **1.0 where IQR = 0** (sklearn `RobustScaler` semantics) |
| `n_obs` | int64 | non-NaN train observations behind the estimate |

plus `scaler_params_metadata.json`: `schema_version`, the `FitWindow` that produced it (`fold_id`, `fit_start`, `fit_end` — §3.5; never a bare hardcoded date), a fingerprint of the split config it was resolved from, `dataset_version` (the `dataset_v{N}` it was fit against), `min_obs` thresholds, full column list, `built_at`. Under a multi-fold evaluation there is one params file per fit window (§3.5 layout).

**Fit:** one pandas `groupby(["ticker"])[cols].quantile([.25, .5, .75])` pass over the rows inside the injected `FitWindow` (§3.5), fundamentals deduped to quarterly observations first (constraint 3); same again grouped by `sector`, and once globally. No sklearn objects are instantiated — correctness is instead *tested against* `RobustScaler` per ticker (§9).

**Transform (fallback hierarchy):** for each (ticker, column), use the `ticker`-level row if `n_obs ≥ MIN_OBS` (**8 quarterly** observations for fundamental columns, **250 daily** rows for daily columns); else the `sector` row (same threshold, computed across the sector's train rows); else the `global` row. The chosen level should be exposed (a per-ticker audit table in each window's metadata, not a per-row column) so "how much of the universe actually got per-ticker treatment" is a queryable fact per fit window rather than a hope — under rolling windows especially, this fraction will vary fold to fold. Implementation: melt/merge + vectorized arithmetic — cheaper than `ColumnTransformer`, and `transform_features()`'s column-order restoration contract (`scale_features.py:73`) is preserved.

**Loud failure, no silent fallback:** a missing `scaler_params.parquet`, a corrupted file, or a column-list mismatch against `scaler_params_metadata.json` raises immediately with the mismatch enumerated. This echoes the silent-inference-fallback concern in `docs/TODO.md` §1.4 — a transform silently degrading to global stats at inference is precisely the class of bug that plan worries about.

### 3.5 Split abstraction — the one seam between evaluation methodology and scaling

The current fixed walk-forward split (`train_end`/`val_end`) is an *instance*, not an assumption. Everything stateful in this plan touches the split through a single adapter, so a future change of evaluation methodology (re-cut boundaries, multiple folds, rolling or expanding windows) is a change to the split *config format* plus this one function — never to feature code or scaler math:

```
FitWindow = (fold_id: str, fit_start: Timestamp | None, fit_end: Timestamp)

iter_fit_windows(split_config: dict) -> list[FitWindow]
```

- Lives beside `compute_split_dates()`/`write_split_config()` in `manifest.py` (moving to a small `splits.py` only if/when multi-fold formats make it grow).
- Today's `split_config.json` (`train_end`/`val_end`) maps to `[("full", None, train_end)]` — one window, `fit_start = None` meaning "from the beginning of history" (expanding). A future rolling/expanding/multi-fold config format adds a branch here and nowhere else; `fit_start` is already in the tuple so rolling windows need no schema change.
- **Consumers:** R3 and R4 fitting iterate over the returned windows, producing one artifact per window. Every artifact's metadata records the `FitWindow` that produced it plus a fingerprint of the split config it was resolved from, so a params/split mismatch is detected at load time instead of silently transforming with stale boundaries.
- **Selection at use time is always explicit:** training code for fold *k* loads the artifact whose `fold_id`/`fit_end` matches fold *k*; live inference loads the designated production window (by convention the latest `fit_end`, but chosen by the caller — the loader has no implicit "latest" default). There is no "load the scaler" without saying which one.
- **R1/R2 never see the split at all** — they are pure functions of the raw series and need no adapter; this is a core reason R1 is the primary mechanism.

**Storage under multiple windows:** the current flat `data/processed/scalers/` layout is the degenerate single-window case and stays canonical while the config defines one window (backward compatibility with anything reading today's paths). When a config defines multiple windows, artifacts live in one subdirectory per window — `data/processed/scalers/{fold_id}/…` — each holding its own `feature_scaler.joblib` / `scaler_params.parquet` + metadata.

---

## 4. Feature treatment matrix (all 152 columns)

Every live column, by family, with its scaling treatment under this plan. "—" = untouched/passthrough, exactly as today.

| Family (producing module) | Columns | Level (R3) | Own-history (R1) | Peer-rank (R2) | Rationale |
|---|---|---|---|---|---|
| Identifiers/dates | `ticker`, `trade_date`, `reference_date`, `fundamentals_available_date`, `corporate_name`, `trade_name`, `cvm_code`, `cnpj`, `sector`, `status` | — | — | — | not features; `status` additionally banned as a training feature (CLAUDE.md lookahead trap) |
| Binary flags | `f_roa_positive`…`f_liquidity_improving`, `had_negative_earnings_5y`, `has_dividends`, `has_fundamentals`, `adj_close_precision_degraded`, `cagr_*_defined` | — | — | — | already {0,1} |
| Counts | `f_score`, `div_count_12m`, `n_quarters_available`, `filing_lag_days`, `days_since_fundamental` | — | — | — | small bounded ints; scaling adds nothing |
| Bounded percentiles/z (exist) | `volatility_*_percentile`, `price_percentile_*`, `pl_percentile_5y`, `drawdown_percentile`, `div_yield_sector_percentile`, `*_zscore_sector` | — | — (they *are* R1/R2-type outputs) | — | scaling a percentile is a no-op with extra steps |
| Returns | `log_return`, `overnight_gap`, `intraday_return`, `return_{1m,3m,6m,12m}`, `excess_return`, `real_return`, `momentum_vs_*` | — | deferred (vol-normalized returns are a possible later R1 addition) | — | already unit-free and cross-ticker comparable; a fitted per-ticker scaler on returns bakes in stale vol regimes |
| Raw price levels | `open/high/low/close`, `adj_*`, `ma_20`, `ma_60`, `div_value_recent` | — | — | — | never model features as levels (env/execution needs them raw); relative versions exist (`price_vs_ma*`, percentiles) |
| Raw volume/liquidity absolutes | `volume`, `volume_adjusted`, `traded_amount`, `num_trades` | — | via ratio forms only | — | per-ticker scale differences are real but already captured by `volume_ratio_20d`, `turnover_ratio`, `amihud_illiquidity` — R1 targets those ratios, not the raw counts |
| Absolute fundamentals (BRL) | `shares_outstanding`, `market_cap`, `net_income`, `equity`, `net_revenue`, `total_debt`, `ebitda`, `ebit`, `net_debt`, `cash`, `total_assets`, `current_assets`, `current_liabilities` | — | — | `market_cap_rank_cs` (size factor) | ratios already normalize these per company; own-history z of `net_revenue` ≈ `revenue_growth_yoy` (exists) |
| **Ratios/multiples/margins (core target)** | the 58 `RATIO_COLUMNS` | **RobustScaler (unchanged)** | **13-column initial set** (§3.1) | **6 ranks + size** (§3.2) | the three-view design, §3 |
| Growth/trend/QoQ | `*_growth_yoy`, `*_qoq`, `*_trend_4q`, `cagr_*` | RobustScaler (unchanged — they're in `RATIO_COLUMNS`) | — | — | already differenced = intrinsically own-history-relative |
| Technicals, bounded | `rsi_14` (0–100), `drawdown`, `price_vs_ma*`, `hl_ratio`, `true_range_ratio`, `beta_1y` | — | — | — | bounded or already relative |
| Technicals, unbounded | `volatility_20d/60d`, `volatility_ratio_20_60`, `amihud_illiquidity`, `turnover_ratio`, `volume_ratio_20d` | last four: RobustScaler (unchanged) | `amihud`, `turnover` zhist (§3.1); vol already has percentiles | — | liquidity is the most per-ticker-idiosyncratic family in the set |
| Macro | `selic`, `cdi`, `ipca`, `selic_trend_20d` | — | — | — | identical across tickers on a date — per-ticker scaling is meaningless by construction; rates are already in natural bounded units |
| Dividend | `div_yield_12m` | — | — | — | already a yield; sector percentile exists |
| Calendar features / explicit targets | **do not exist in this dataset** | | | | rewards are derived downstream from returns by the agent env; nothing to scale here — noted so the categorization is complete |

---

## 5. Pipeline integration points

Everything this plan touches, in build order:

| Location | Change | Option |
|---|---|---|
| `features.py` | new `compute_history_relative_features()` — quarterly-deduped rolling robust z for the 11 fundamental columns; daily rolling for the 2 liquidity columns. Runs inside Pass 1 of `compute_features_chunked()` (`build_ml_dataset.py:68`), after `recompute_valuation_daily` and `compute_advanced_features` (it consumes their outputs: re-anchored `pl`, `turnover_ratio`) | R1 |
| `cross_sectional.py` | extend `CROSS_SECTIONAL_INPUT_COLS` (+`earnings_yield`, `net_margin`, `market_cap`) and `CROSS_SECTIONAL_OUTPUT_COLS` (+7 rank columns); rank logic beside the existing sector z-scores. **Constraint:** Pass-2-only — per-date ranks computed inside a 150-ticker batch silently compare against the wrong universe (the docstring at `build_ml_dataset.py:68` already warns about exactly this) | R2 |
| `build_ml_dataset.py` | wire the new Pass-1 call; nothing else — Pass 2/3 flow unchanged | R1 |
| `scale_features.py` | R3: refactor `fit_scaler_on_train_split()` → pure `fit_scaler(dataset, window)` with boundaries injected via `iter_fit_windows()` (§3.5); output-identical under the current config. If R4 adopted: add `fit_scaler_params()`, `load_scaler_bank()` (explicit window selection, no implicit default), vectorized transform with fallback + loud validation; keep `build_scaler`/`transform_features` intact | R3/R4 |
| `paths.py` | `SCALER_PARAMS_PATH`, `SCALER_PARAMS_METADATA_PATH` | R4 |
| `manifest.py` | new `iter_fit_windows()` adapter beside `compute_split_dates()` (§3.5); `sync_dataset_version()` additionally copies `scalers/` into `dataset_v{N}/`; new R1/R2 columns flow into `column_stats`/`nan_regressions()` automatically (verify — the first post-R1 build **will and should** trip the >2pp NaN-regression *warning* for the new columns; it is non-fatal by design, note it in the build log) | all |
| `build_top50_universe.py` | unaffected — operates on the built parquet before any scaling concern; new columns ride along through `filter_to_top50_universe()` | — |
| Tests | §9 | all |
| Docs | `CLAUDE.md` (feature list, scaling caveat), `docs/DATA_PIPELINE.md` (§build steps), `docs/FEATURE_SCALING_AUDIT.md` (summary lists gain the new columns) | all |
| **Out of repo** (`ml_agent` branch) | (a) prerequisite check from §1.1; (b) opt into `*_zhist_5y` / `*_rank_cs` columns in observation building; (c) NaN→fill handling for the new columns (same flag+fill pattern as CAGR); (d) if R4: switch `data_pipeline.py` to `load_scaler_bank()` | follow-ups |

Also noted during investigation: `data/processed/README.md` is referenced by CLAUDE.md as the tracked ownership-boundary doc but **does not exist on this branch** — restore it independently of this plan.

---

## 6. Leakage prevention

**R1/R2 (stateless):** the correctness criterion is *truncation invariance* — for every ticker and time *t*,

```
f(D_{≤t})_t  ==  f(D)_t
```

i.e., the feature value at *t* computed from data up to *t* equals the value at *t* computed from the full dataset. Rolling windows and per-date ranks satisfy this by construction; the test (§9) verifies it empirically by recomputing on truncated data at sampled cutoffs, following the existing `test_volatility_percentile_no_lookahead` / `test_merge_honors_actual_filing_date` pattern. Fundamental-based zhist inherits filing-date visibility from the main `merge_asof` (§3.1), so the broadcast step adds no new leakage surface.

**R3:** fit only on rows inside the injected `FitWindow` resolved from the active split config (§3.5) — under today's config, identical to the current train-only fit. Guarded by `test_refit_on_train_split_is_reproducible` plus the new arbitrary-window test (§9 row 15).

**R4 (if adopted):** every `center`/`scale`/`n_obs` derives exclusively from rows inside the fit window that owns the artifact — at ticker, sector, *and* global levels (a sector fallback fit on full history would leak through the back door). The general invariant, valid under any evaluation methodology: **a row is only ever transformed with parameters from a fit window that ends at or before the point where that row becomes out-of-sample** — fold *k*'s val/test rows use fold *k*'s fit-window params; live inference uses the explicitly selected production window. Refit happens only in the deliberate post-rebuild step, writes new per-window artifacts, and stamps `dataset_version` plus the split-config fingerprint in metadata so a params/dataset/split mismatch is detectable at load. Snapshotting into `dataset_v{N}` makes every historical experiment reproducible with the scaler it actually used.

---

## 7. Edge cases

| Case | Handling |
|---|---|
| Newly listed ticker (post-`train_end` IPO) | R1: leading-prefix NaN until `min_periods`, consistent with prefix rule + `n_quarters_available`; consumer fills per its existing NaN policy. R4: no ticker row → sector → global fallback (this is most of why R4 is not primary) |
| Short history / sparse fundamentals | R1: `min_periods` (8 quarters / 252 days) → NaN below threshold. R4: `n_obs < MIN_OBS` → fallback level |
| Missing fundamentals entirely (`has_fundamentals = 0` stretches, merger-dropped legs) | inputs are NaN → zhist is NaN; robust stats `skipna`; no imputation anywhere (project rule) |
| Delisted tickers (CANCELADA) | full train history exists → normal treatment; verify one in tests (they matter for the survivorship-bias-aware universe) |
| Renames/mergers | continuity splicing (`continuity.py:22`) runs before features; windows see one spliced series. Merger legs with dropped fundamentals → NaN zhist over the old leg, resuming post-boundary; **verify TIMS3 and BHIA3 show no zhist discontinuity artifact at the splice date** (the adj-close reconciliation factor rescales levels — unitless ratios are unaffected, which the test confirms rather than assumes) |
| Quarantined tickers (WDCN3, CAMB4, LLIS3, CCTY3) | dropped in `quality_filters.py:17-29` before any of this runs |
| Constant feature in window | rolling IQR = 0 → NaN (R1); `scale = 1.0` (R4, sklearn semantics) |
| Extreme ratios (\|pl\| > 400k policy) | preserved: robust center/scale are outlier-immune in fit; transform is linear so extremes stay extreme (intentional, per CLAUDE.md); zhist of an extreme value is a large finite number — informative, not clipped |
| NaN propagation | NaN in → NaN out at that position, never imputed; window stats skip NaN (guarded by an R1 analog of `test_nan_preserved_not_imputed`) |
| Changing feature sets across builds | R4 metadata column list validated at load, mismatch raises with names enumerated; R1/R2 columns live in the parquet, so the existing manifest fingerprint + schema/dtype contract test cover drift |
| Missing/corrupted `scaler_params.parquet` | loud error at load; **never** silently fall back to global or to legacy joblib (TODO §1.4 concern) |
| Evaluation methodology changes (re-cut boundaries, rolling/expanding, multi-fold) | R1/R2: unaffected — stateless, never refit. R3/R4: the new split config flows through `iter_fit_windows()` (§3.5) → refit produces one artifact per window; loaders require an explicit window, and the split-config fingerprint in metadata catches use of stale artifacts against a changed split |

---

## 8. Performance and migration

**Cost estimates:**

- R1 fundamentals: rolling stats over ≤~105 quarterly observations per ticker × 515 tickers × 11 columns — noise, sub-second territory. R1 daily: rolling-1260 quantiles over 1.32M rows × 2 columns — pandas `rolling().quantile()` is the slow spot, minutes at worst inside Pass 1's existing per-batch loop; acceptable for a build that already takes a multi-pass streamed approach.
- Memory: +13 float64 columns on 150-ticker Pass-1 batches ≈ +40 MB per batch peak — absorbed by the existing chunk design. Disk: +13–20 columns ≈ +60–150 MB on `ml_dataset.parquet` (~505 MB today).
- R4: fit = three groupby-quantile passes over the fit-window rows (≤704K under the current config; seconds) — under a multi-fold evaluation this repeats per window, scaling linearly in fold count and staying trivial; params table ~90K rows < 1 MB per window; transform = merge + arithmetic, cheaper than the current `ColumnTransformer`. **Explicit answer to "are many scaler files acceptable": no per-ticker joblib files, ever — one parquet.**

**Migration (each phase additive, reversible by simply not reading the new columns; `dataset_v{N}` snapshots give immutable before/after):**

- **Phase 0 — tests first.** All CREATE-marked tests in §9 written and passing against fixtures before feature code lands.
- **Phase 1 — R1 columns** added in Stage 2; dataset rebuilds bump `dataset_v{N}`; NaN-regression warning expected once; scaler untouched. Downstream reads nothing new yet → zero break risk.
- **Phase 2 — R2 columns** (respecting the TODO M5 gating for actually *using* them in experiments).
- **Phase 3 — split seam + versioning fix:** the `iter_fit_windows()` adapter (§3.5), the pure `fit_scaler(dataset, window)` refactor (verify the produced scaler is identical to today's under the current config), and snapshotting `scalers/` into `dataset_v{N}` (can ship with Phase 1).
- **Phase 4 (conditional) — R4** params table written *alongside* the legacy joblib; both artifacts coexist; `ml_agent` migrates on its own schedule; legacy joblib retired only after the agent branch confirms it loads the bank (or confirms it never loaded the joblib at all, per §1.1 — in which case retire it immediately).
- **Rollback:** any phase — stop reading the new columns / delete the new artifact; R3 path is never modified.

---

## 9. Risk → test table

House pattern per `docs/TOP50_UNIVERSE_VALIDATION.md` §2/§3. All new tests belong in `tests/build_dataset/`, fast group (synthetic fixtures, `run_all.py` `FAST` list), except the two marked *data*.

| # | Risk | Test | Status |
|---|---|---|---|
| 1 | Global scaler behavior regresses (column order, NaN, scope, determinism, train-only fit) | `test_scale_features.py` (5 tests, `tests/build_dataset/test_scale_features.py:35-93`) | EXISTS |
| 2 | R1 feature peeks at future rows | `test_history_relative.py::test_zhist_truncation_invariance` — recompute on data truncated at sampled dates, compare last rows (pattern: `test_volatility_percentile_no_lookahead`) | **DONE** |
| 3 | R1 warm-up NaN not prefix-shaped | `::test_zhist_warmup_is_prefix_nan` — first non-NaN appears exactly at `min_periods`, none before, no interior holes introduced | **DONE** |
| 4 | R1 computed on daily-repeated fundamentals (63× redundancy bug) | `::test_zhist_uses_quarterly_observations_not_daily_rows` — every daily row within one quarter shares the same value | **DONE** |
| 5 | R1 NaN imputed or inf emitted | `::test_zhist_nan_preserved_and_iqr_zero_is_nan` | **DONE** |
| 6 | R1 broken across a rename splice | `::test_zhist_continuous_across_ticker_history_boundary` — two concatenated history blocks under one ticker label (what continuity splicing produces), assert no cold-restart at the boundary. Real TIMS3/BHIA3 spot check (data-group) not yet added — folded into row 14 | **DONE** (lighter form; data-group spot check still open) |
| 7 | R2 rank computed inside a ticker batch (wrong universe) | rank columns declared in `CROSS_SECTIONAL_OUTPUT_COLS` → existing `test_compute_features_chunked` Pass-2 discipline covers placement; add `test_cross_sectional.py::test_rank_cs_is_per_date_uniform` | EXISTS (placement) / **CREATE** (values) |
| 8 | R4 params ≠ sklearn RobustScaler per ticker | `test_scaler_params.py::test_params_match_sklearn_per_ticker` | **CREATE** (if R4) |
| 9 | R4 fallback hierarchy wrong (new ticker / short history / all-NaN → sector → global) | `::test_fallback_levels` incl. a zero-train-row ticker (the RDOR3 case) | **CREATE** (if R4) |
| 10 | R4 sector/global fallback fit leaks rows outside the fit window | `::test_all_levels_fit_inside_window_only` | **CREATE** (if R4) |
| 11 | R4 file round-trip / corruption / column drift | `::test_params_roundtrip`, `::test_corrupt_or_missing_raises_loudly`, `::test_column_mismatch_raises` | **CREATE** (if R4) |
| 12 | Scaler artifacts not reproducible per dataset version | extend `test_dataset_versioning.py` — `dataset_v{N}` snapshot includes `scalers/` | **DONE** |
| 13 | Schema/dtype contract with `ml_agent` breaks | existing contract test (`TOP50_UNIVERSE_VALIDATION.md` §3.2) — new columns extend, never mutate, existing dtypes | EXISTS (verify covers additions) |
| 14 | Built-dataset sanity of new columns (bounds, NaN rates, extreme counts) | extend `test_final_dataset.py` (*data* group) with zhist/rank column checks | **CREATE** |
| 15 | Fit boundary hardcoded / window not honored (any methodology) | `test_scale_features.py::test_fit_honors_arbitrary_window` — fit at several synthetic `(fit_start, fit_end)` windows, including the one that reproduces today's config; assert params depend only on in-window rows | **DONE** |
| 16 | Multi-window artifacts cross-contaminate or get implicitly selected | `::test_per_window_artifacts_independent_and_explicitly_selected` — two windows → two artifacts with distinct params; loading without naming a window raises | **CREATE** |

Phase 0 of the migration = rows 2–6, 12, 14 written and green before any pipeline code merges; row 15 lands with the Phase 3 refactor; rows 8–11 and 16 with Phase 4, if R4 proceeds.

---

## 10. Research context: why layered views beat every single-mechanism alternative

| Approach | Pro | Con | Verdict |
|---|---|---|---|
| One global scaler (status quo) | stable statistics (700K rows), preserves cross-sectional levels, one artifact | hides per-ticker regimes (§1.2); pooled stats dominated by long-history large caps | keep — as the *level* view only (R3) |
| Per-ticker fitted scalers | own-history view; stationary transform | cold start kills it for post-2018 IPOs (§2.1); stats stale by construction (§2.2); erases level info if it *replaces* global; noisy estimates on short histories | not primary; viable only as R4 with fallbacks, and only if downstream needs a transform rather than features |
| Per-sector fitted scalers | more stable than per-ticker, some group structure | wrong granularity for the actual hypothesis (a company vs. *its own* history, not vs. its sector); sector-relative info already exists as `*_zscore_sector` | subsumed — sector appears as the R4 fallback tier and the existing cross-sectional features |
| Robust vs. standard scaling | median/IQR immune to this dataset's deliberate extreme ratios (\|pl\| > 400k kept) | linear transform keeps extremes extreme (intentional here) | robust everywhere a scaler exists — already the house choice, unchanged |
| Rank / percentile per date (cross-sectional) | outlier-immune, stateless, no leakage, the standard in cross-sectional equity ML (Gu–Kelly–Xiu 2020 rank-normalize characteristics per date) | discards magnitude ("cheapest decile" ≠ "how cheap"); needs full-universe pass | adopt as R2, alongside — not instead of — magnitude views |
| Causal rolling per-ticker normalization (as features) | implements the hypothesis point-in-time; stateless → leakage-free by construction; adapts to regime change; degrades to prefix-NaN on cold start (already-handled shape); valid unchanged under any evaluation methodology (no refit when the split is re-cut); precedented in-repo | warm-up cost (first 2y/8q per ticker yields NaN); window length is a hyperparameter; adds columns rather than transforming in place | **adopt as R1, primary** |
| Feature-specific normalization (bespoke rule per feature) | maximal correctness | that is exactly what §4 *is* — the matrix assigns each family its treatment; "one strategy per feature family" rather than one global rule | adopted, expressed as the treatment matrix |

The quant-practice consensus this lands on: **time-series-relative and cross-section-relative versions of a characteristic are different signals, and both are computed as point-in-time features; fitted scalers are kept for model-input conditioning only, fit train-only, robust, and few.** De Prado's leakage discipline (never let statistics computed on future data touch a training row) is satisfied structurally by R1/R2 (stateless) and procedurally by R3/R4 (injected fit windows + versioned per-window artifacts). The same split holds for evaluation-methodology risk: R1/R2 are immune to it structurally; R3/R4 confine it to the single `iter_fit_windows()` seam. `docs/RESEARCH_REFERENCES.md` is the place to log the GKX / de Prado citations when the doc is merged.

One honest caveat: R1 columns are *derived* from columns the model already sees, so information-theoretically they add nothing a sufficiently expressive model couldn't learn. Their value is inductive bias — handing the network a normalized deviation it would otherwise spend capacity approximating, which matters at this dataset's size (~700K train rows across 515 noisy series). That is also why this plan should not jump the M5 diagnosis queue (`docs/TODO.md`): if the edge dies before observations reach the policy, better observations won't resurrect it. Design now, implement per the M5→M6 sequencing.

---

## Summary lists

### 1. Unchanged
- [x] Global RobustScaler *outputs* (R3) — artifact identical under the current split config (the fit API gains injected windows, §3.3; refactor tracked in list 2)
- [x] All 94 passthrough columns' treatment
- [x] `build_top50_universe.py`, quality filters, continuity, quarantine machinery
- [x] NaN policy (no imputation; prefix rule; flags + consumer-side fills)

### 2. New — Phase 0/1 (R1 + split seam + versioning fix) — IMPLEMENTED 2026-07-15
- [x] Tests: rows 2–6, 12, 15 of §9 (written before implementation, all green under `tests/run_all.py --group fast`)
- [x] Row 14 (built-dataset sanity, *data* group) — `test_final_dataset.py`,
      `test_top_traded_quality.py`, `test_universe_integrity.py` all pass against the rebuilt
      165-column dataset (2026-07-15). A dedicated TIMS3/BHIA3 zhist-continuity spot check
      (beyond the synthetic splice test) is still not written — low priority, the synthetic
      test already exercises the same code path.
- [x] `features.py::compute_history_relative_features()` — 13 `*_zhist_5y` columns (11 quarterly-based, 2 daily-based)
- [x] Wire into Pass 1 of `compute_features_chunked()`
- [x] `iter_fit_windows()` split adapter in `manifest.py` (§3.5) + pure `fit_scaler(dataset, window)` refactor in `scale_features.py` — verified byte-identical to the old behavior under the current single-split config (`test_fit_honors_arbitrary_window`)
- [x] `sync_dataset_version()` snapshots `scalers/` into `dataset_v{N}/`
- [x] Docs: CLAUDE.md, `DATA_PIPELINE.md`, `FEATURE_SCALING_AUDIT.md` summary lists
- [x] Dataset rebuild + scaler refit — done 2026-07-15: `dataset_v5` (1,319,349 rows × 165
      columns, 13 real `*_zhist_5y` columns), `scaler_metadata.json` records
      `fit_window={fold_id: full, fit_start: null, fit_end: 2018-07-30}`, `dataset_v5/scalers/`
      confirms the versioning fix. Full data-group suite (8/8) green against the rebuild.

### 3. New — Phase 2 (R2, gated on M5 diagnosis per `docs/TODO.md` §6.2)
- [ ] `cross_sectional.py`: 7 `*_rank_cs` columns (6 ratios + `market_cap` size rank); extend input/output col lists
- [ ] Test row 7 (values)

### 4. Optional — Phase 4 (R4, only if downstream requires a per-ticker transform)
- [ ] Decision gate: §1.1 prerequisite answered *and* agent team wants transformed inputs over feature columns
- [ ] `scaler_params.parquet` + metadata (window-keyed, §3.4/§3.5), fit/load/transform in `scale_features.py`, `paths.py` constants
- [ ] Per-window artifact layout + explicit-selection loader (§3.5)
- [ ] Tests: rows 8–11, 16 of §9

### 5. Out-of-repo follow-ups (`ml_agent` branch)
- [ ] **Prerequisite:** confirm whether `src/agent/data_pipeline.py` loads `feature_scaler.joblib` or refits its own (determines whether the joblib is a live contract or dead weight)
- [ ] Opt observation builder into `*_zhist_5y` / `*_rank_cs` columns; extend flag+fill NaN handling to them
- [ ] If R4 adopted: switch agent to `load_scaler_bank()`
- [ ] Restore missing `data/processed/README.md` (referenced by CLAUDE.md, absent on this branch — independent of this plan)
