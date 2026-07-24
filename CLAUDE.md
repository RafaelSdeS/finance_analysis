# CLAUDE.md

Guidance for Claude Code working in this repo.

## Overview

**Project:** Brazilian-equity dataset pipeline for ML applications.

**Two stages** (all scripts run from project root):
1. **Data Collection** — staged prototype→validation→full-scale pipeline (checkpointing, logging, validation).
2. **Dataset Build** — merge raw data → derived features (technical, fundamental, macro) → clean → ML-ready parquet, no lookahead bias.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add BOLSAI_API_KEY=sk_...  (backfill only; .env is gitignored)
```

## Run Commands

### Stage 1: Collect Raw Data

```bash
# Backfill (one-time historical via BolsAI, 2000–present); resumes from checkpoints, idempotent
python -m src.data_collection.pipeline --mode full_scale            # all ~500+ tickers
python -m src.data_collection.pipeline --mode full_scale --dry-run  # preview ticker list
python -m src.data_collection.pipeline --mode prototype --tickers PETR4 VALE3

# Quarterly incremental refresh (free yfinance, no key; >99% cost savings)
python -m src.data_collection.pipeline --mode update

# Validate (cross-check vs yfinance, 1–15% tolerance on key ratios)
python tests/data_collection/validate_vs_yfinance.py
```

### Stage 2: Build ML Dataset

Prereq: Stage 1 complete (raw data in `data/raw/`). Merges prices + fundamentals + company info via `merge_asof` backward (no lookahead).

```bash
python -m src.build_dataset.build_ml_dataset            # → data/processed/ml_dataset.parquet
python tests/build_dataset/test_final_dataset.py        # schema, shape, lookahead, NaN, returns

# Fit the feature scaler (train-only, per split_config.json) — rerun after a rebuild
python -m src.build_dataset.scale_features               # → data/processed/scalers/{feature_scaler.joblib,scaler_metadata.json}
```

### Stage 3+: Modeling (not yet started)

No model or agent implementation exists in this repo. Prior modeling work on this branch (a
Stage 3 EIIE RL agent, a Stage 4 self-supervised conviction-model encoder, and an M-series →
risk_mandate → H-series research lineage exploring alpha/portfolio policies) was deleted on
2026-07-23 in a full reset — none of it produced a working, deployable result, and starting
over from Stage 2's output was judged cleaner than carrying that design forward. It remains
recoverable from git history (`git log` on `refactor` before that date) if ever needed for
reference, but nothing about its design should be assumed or reused without deliberately
re-reading it. Stage 2's output (`data/processed/ml_dataset.parquet` + the raw data in
`data/raw/`) is the only carryover — it's untouched by this reset.

### Utilities

```bash
python src/build_dataset/cagr_handler.py --ticker PETR4  # CAGR calculator
python src/visualizations/financial_view.py              # BBAS3 nominal vs inflation-adjusted vs SELIC (live yfinance)
jupyter notebook src/visualizations/exploration.ipynb    # full dataset validation + insights
```

### Tests

Plain Python scripts, no pytest. Unified test runner:

```bash
# Fast group (pure-code unit tests, no data files needed — used by CI)
python tests/run_all.py --group fast

# Data group (needs git-tracked data/raw/* and built data/processed/ml_dataset.parquet)
python tests/run_all.py --group data

# All tests
python tests/run_all.py --group all
```

**Test groups:**
- **Fast:** `test_features.py`, `test_merge.py`, `test_cross_sectional.py`, `test_compute_features_chunked.py`, `test_split_config.py`, `test_dataset_versioning.py`, `test_scale_features.py`, `test_company_siblings.py`, `test_ticker_continuity.py`
- **Data:** `test_final_dataset.py`, `test_top_traded_quality.py`, `test_universe_integrity.py`, `test_cagr_calculation.py`, `test_blue_chip_tickers.py`, `validate_vs_yfinance.py`, `test_collect_delisted.py`, `test_cvm_statements.py`

**Linting:**
```bash
ruff check .          # reports undefined names, unused imports/variables, bare-except
```

## Branches

- **main:** Stages 1–2 (data collection + dataset build). Latest stable.
- **build_dataset:** Stage 2 focus.
- **refactor:** Stages 1–2 only, same as main, as of the 2026-07-23 reset (see Stage 3+ note above). Modeling work restarts here from zero.
- **ml_agent:** a separate, earlier PPO agent (masked 279-ticker universe); unrelated to this branch's reset.

## Architecture

### Data Flow

```
BCB SGS (SELIC/CDI/IPCA) + BolsAI (OHLCV + quarterly fundamentals + dividends)
        ↓
data/raw/{prices,fundamentals,macro,dividends,company_info}/
        ↓ build_ml_dataset.py
  merge_asof(prices, fundamentals) [no lookahead] → left join company_info
  → compute_price_features()       [RSI, MA20/60, volatility, returns, drawdown]
  → compute_fundamental_features() [P/E, P/B, ROE, debt/equity, growth CAGR]
  → compute_macro_features()       [real return, excess return, rate environment]
  → fill_missing_cagr()            [backfill from earnings/revenue where API null]
  → clean (dupes, NaNs, outliers, sort)
        ↓
data/processed/ml_dataset.parquet  (one row per ticker+date)
  + ml_dataset.manifest.json, split_config.json, dataset_v{N}/ snapshot
        ↓ scale_features.py (separate, deliberate step — not run every build)
data/processed/scalers/feature_scaler.joblib  (train-only fit, per split_config.json)
```

### Key Modules

**Stage 1 (Data Collection)** — `src/data_collection/`, flat-function, source-agnostic:

| Module | Purpose |
|--------|---------|
| `config.py` | Shared config (tickers, keys, paths, retries, `DATA_SOURCE` per-type source switch) |
| `client.py` | BolsAI HTTP wrapper (retries, backoff): `make_client()`, `get_json()` |
| `checkpoint.py` | Resume state (JSON per collector) |
| `validate.py` | Quality gates (schemas, ranges, continuity) → `ValidationResult` |
| `collectors.py` | BolsAI: `collect_{macro,prices,fundamentals,company_info,dividends,corporate_events,sectors}()`; helper `_merge_save()` (idempotent append+dedup+validate+write) |
| `yf_collectors.py` | yfinance: `collect_{prices,fundamentals,dividends}_yf()` for `--mode update` |
| `pipeline.py` | Orchestration; `_collect()` dispatches to BolsAI/yfinance per `DATA_SOURCE` |
| `cvm_statements.py` + `cvm/` | CVM open-data collection (delisted fundamentals + real filing dates); thin `--step` CLI dispatching to `cvm/{http,crosswalk,statements,shares,ratios,company_info,filing_dates}.py`. `cvm/http.py` holds the one shared zip-download/retry implementation |

**Stage 2 (Dataset Build)** — `src/build_dataset/`, split by pipeline stage (each module below mirrors a section of the old monolithic `build_ml_dataset.py`):

| File | Purpose |
|------|---------|
| `build_ml_dataset.py` | Orchestration only: `main()` + the memory-bounded `compute_features_chunked()` (3-pass: per-ticker features → cross-sectional → clean+write) |
| `paths.py` | Shared path constants (avoids circular imports across the split modules) |
| `loaders.py` | `load_prices()`, `load_fundamentals()`, `load_company_info()`, `load_dividends()`, `company_siblings()` |
| `repair.py` | `repair_unadjusted_splits()` — rescales `adj_*` history where a split was left unadjusted |
| `continuity.py` | `apply_ticker_continuity()` — splices renamed/merged tickers |
| `quality_filters.py` | Coverage + filing-lag gates: `filter_tickers_with_no_fundamentals()`, `attach_filing_dates()`, `filter_excessive_filing_lag()` |
| `merge.py` | The 4 `merge_*` functions (prices+fundamentals, company_info, macro, dividends) |
| `features.py` | Per-ticker feature engineering: CAGR backfill, dividend yield, price technicals, fundamental ratios/trends, valuation re-anchoring, "advanced" contextual features, `compute_history_relative_features()` (per-ticker own-history z-scores, `*_zhist_5y`) |
| `cross_sectional.py` | `compute_cross_sectional_features()` — sector/market-relative features; needs the full universe at once, unlike everything in `features.py` |
| `clean.py` | `clean_dataset()` — final dedupe/inf-to-NaN pass |
| `manifest.py` | `write_manifest()`, `compute_split_dates()`, `iter_fit_windows()`, `write_split_config()`, `sync_dataset_version()` |
| `cagr_handler.py` | CAGR calc/fill (BolsAI first, backfill from earnings/revenue) |
| `scale_features.py` | Fits `ColumnTransformer` (RobustScaler on ratio columns, passthrough elsewhere) train-only, per `split_config.json`; saves `feature_scaler.joblib` + `scaler_metadata.json` |

## Critical Caveats

- **CAGR backfill is ON:** `fill_missing_cagr()` (which calls `fill_cagr_columns()` per ticker) runs unconditionally in `build_ml_dataset.py`'s main pipeline → dataset has `cagr_{earnings,revenue}_5y_final` populated. Coverage is ~60% from BolsAI; the backfill recovers an additional ~7%.
- **No lookahead (Stage 2) — ENFORCED:** `merge_asof(..., direction='backward')` on real CVM `fundamentals_available_date` (not fiscal period-end) — a price never sees a future fundamental. `volatility_*_percentile` use rolling-window rank, not global rank. Tests: `test_merge_honors_actual_filing_date`, `test_volatility_percentile_no_lookahead`. ✅ VERIFIED 2026-07-11.
- **Real filing dates (July 2026):** Fundamentals visible via CVM's `DT_RECEB` (received date), not fiscal `reference_date`. 41,530 filings from 1,223 companies, 100% coverage of 293-ticker universe; 4,657 rows (0.7%) would have violated a fixed 45/90-day buffer. Sourced from free, keyless CVM open-data portal; integrated via `src/data_collection/cvm/filing_dates.py` (`python -m src.data_collection.cvm_statements --step filing_dates`).
- **Known limitation — fundamentals *values* may be restated even though `fundamentals_available_date` is point-in-time (2026-07-23 audit, Issue 8):** `filing_dates.py` correctly takes the *earliest* CVM receipt (v1) as the availability date, but the fundamental *figures themselves* come from BolsAI's `/fundamentals/history`, which reflects whatever BolsAI's snapshot currently holds — almost certainly the latest restatement, not what was actually filed at v1. Where a company restated (common after auditor review of ITRs), a row dated at the original v1 filing date can carry corrected numbers nobody had access to at the time — an as-reported-vs-as-restated lookahead in fundamental factors, distinct from (and smaller than) the already-fixed filing-*date* lookahead above. **No fix with BolsAI alone** — CVM's own open-data ZIPs (`src/data_collection/cvm/statements.py`) contain every filing version and could source true as-first-reported v1 figures, but that's a larger sourcing project, not attempted here. Flagged so it isn't rediscovered as a bug; not yet quantified (no measurement of how many rows/how large a restatement gap this affects).
- **Unadjusted splits REPAIRED:** 67 corporate events (count as of the 2026-07-24 rebuild; grows as more history/tickers are collected) in BolsAI's `adj_*` columns were never back-adjusted, causing fake returns up to −99.99%. `repair_unadjusted_splits()` detects and rescales all pre-event rows *and volumes*: a 1:4 split divides prices by 4 and multiplies `volume`/`volume_adjusted` by 4 (same economic activity, more shares). Rescaling is critical for `amihud_illiquidity` and `turnover_ratio` features. `hl_ratio` uses `adj_high/adj_low` (not raw scales). WDCN3 quarantined (unfixable data corruption). ✅ VERIFIED 2026-07-11.
  - **Volume-scaling direction bug, FIXED 2026-07-24:** the "✅ Volume scaling VERIFIED 2026-07-15" claim above was wrong — the code was actually *dividing* `volume`/`volume_adjusted` by the split factor (same direction as price), which *compounds* the dollar-volume discontinuity instead of removing it (should multiply, per the same reasoning stated above). Latent since the volume-rescale feature was first added; never caught because no test asserted the *direction*, only that prices rescaled correctly. Fixed to `*= factor`; regression test now asserts the direction explicitly. Corrects `turnover_ratio`/`volume_ratio_20d` at every one of the 67 repaired events.
  - **Split-matcher persistence guard — investigated, NOT implemented (2026-07-24, re-confirmed 2026-07-24):** the single-day jump/tolerance match can't distinguish a genuine permanent split from a coincidental large one-day market move landing inside a recorded event's matching window. Two PRE-emptive guard designs (reject a match before applying it) were built and tested against the real 67-event dataset; both produced false rejections (27, then 8 more) against genuinely recorded `corporate_events.parquet` entries (PATI4's ~annual small bonus-share splits, SBSP3's clustered restructuring sequence, etc.) — ordinary volatility on illiquid/small-ratio tickers swamps any threshold loose enough to admit them. A third, POST-hoc design (audit whether an already-applied repair's post-jump price level held, rather than blocking it upfront) was tried too, against all 67 real applied events: 14/67 (21%) flagged at a 50% deviation / 20-row-median threshold. Traced two concretely: PDGR3 2025-11-03 (dev=81x) is a genuine subsequent speculative price recovery unrelated to repair correctness; AZEV4 2005-06-13 (dev=653x) traded so infrequently in 2005 that "20 rows later" spans 6 calendar years (2005→2011) — an illiquidity artifact, not a bad repair. Same false-positive wall as the first two attempts, just in audit rather than blocking form. Zero actual misfires found in the current dataset across all three attempts. Reverted all three; revisit only if a future repair is found to have actually misfired.
- **Ticker continuity & splicing (July 2026 fixes):** Renames/mergers/exchanges are spliced via `apply_ticker_continuity()` *after* `repair_unadjusted_splits()` (not before), so splits are repaired under each leg's original ticker name before being renamed onto the survivor. Splicing rules: (1) **rename** = same legal entity, splice prices + fundamentals, drop old ticker. (2) **merger** = exchange ratio, scale old-leg prices by ratio **and volume inversely by ratio** (keeps dollar volume = volume×price invariant across the splice, same rationale as split-repair volume scaling — otherwise `amihud_illiquidity` jumps by `ratio` right at the merger boundary), drop old-leg fundamentals. (3) **keep_separate** = parallel-trading acquirer (e.g., SulAmérica acquired by RDOR, which had its own IPO 2 years earlier), both legs stay as independent series; old treated as delisted. (4) **tender** = cash-out, no splice. Vendor aliases (ARZZ3→AZZA3, RRRP3→BRAV3, etc.) consolidated via `rename` entries where the new file contains full history under both names. Boundary-matching assumption (new ticker's first trade == splice point) is guarded: parallel-trading cases caught and rejected. Adj_close reconciliation factors (inherent basis mismatches between old/new vendor series) are validated [1/50, 50] sane range. Event rekeying: `repair.py` builds ticker-descendant chains from the map so splits recorded under old names (e.g., VVAR3) still match post-rename rows (BHIA3). ✅ All tests pass post-fix. ✅ TIMP3→TIMS3 factor sane (0.6963, not 6963).
- **`adj_close` 2-decimal vendor precision floor (deep-history microcaps):** BolsAI stores `adj_close`/`adj_open`/`adj_high`/`adj_low` at 2-decimal precision. For a handful of tickers with a large cumulative split/dividend adjustment factor, the true adjusted price underflows that floor — it either rounds to exactly `0.00` (raw `close` stays a normal nonzero price; confirmed in `data/raw/prices/UNIP6.parquet`'s earliest ~33 rows, 2026-07-21) or gets pinned at a tiny nonzero constant across several consecutive days while the real price keeps moving. **Not fixable — flag or mask, never drop or reconstruct**: there's no way to recover the lost precision, and (per the caveat above) `adj_close` must not be rebuilt from `data/raw/dividends`. `build_dataset/features.py::compute_price_features()` already masks non-positive `adj_close` to NaN before `log()` and flags the pinned-nonzero case via `adj_close_precision_degraded` (0/1; exact-2dp-quantized AND `<0.05`, so a genuinely low-priced-but-full-precision ticker like TIMS3 isn't misflagged). Any OTHER consumer computing its own `log(adj_close)` off the raw dataset must apply the same non-positive mask.
- **Returns ARE dividend-adjusted (total return), not price-only:** `log_return`/`return_{1m,3m,6m,12m}`/`excess_return`/`real_return` (`features.py`) are all derived from `adj_close`, and `adj_close` empirically bakes in dividend reinvestment, not just splits — confirmed by testing `adj_close/close` ratio drift against known dividends on split-free windows (e.g. BBAS3 post-split: predicted vs. observed ratio jumps matched to within ~0.04 pp per ex-dividend date). `div_yield_12m`/`div_count_12m` remain separately-tracked features on top of this, not double-counted into returns.
  - **Known, undocumented-by-vendor limitation — BolsAI/yfinance dividend-adjustment methodology diverges:** confirmed by direct measurement (145 tickers, BolsAI-only rows, split-free windows): median 4.9pp divergence between BolsAI's observed `adj_close` ratio drift and what `data/raw/dividends` alone predicts, often 20pp+. BolsAI's adjustment consistently implies *more* cumulative discount than our dividends table explains — i.e. our dividends table is missing some distribution type BolsAI's adjustment already correctly captures (bonus shares/subscription rights suspected, unconfirmed). **Do not "fix" this by recomputing `adj_close` from `data/raw/dividends`** — that would systematically under-adjust and regress returns. `validate_vs_yfinance.py:7` already flags this by skipping `adj_close` cross-validation as "uninformative." No fix available with current data; flagged here so it isn't rediscovered as a bug.
  - **Staleness across `--mode update` runs — FIXED:** yfinance's `auto_adjust=True` backward-adjusts a fetch window relative to "now" at fetch time. If each update only fetched rows after the last checkpoint (like every other collector), each quarterly batch would freeze at its own anchor and never get revisited — a dividend paid after one quarter's fetch would permanently fail to propagate into that quarter's already-stored `adj_close`, one small discontinuity per update, forever. `collect_prices_yf` (`yf_collectors.py`) now re-fetches its *entire* yfinance-sourced span every run via `_prices_fetch_start()` (anchored to the earliest yfinance row on disk, marked by `NaN num_trades`, not the latest), so the whole yfinance era stays internally consistent. Empirically verified this wasn't yet causing damage before the fix (max BolsAI→yfinance gap across 285 tickers was 1–3 days — this was the first `--mode update` run for all of them), but the fix prevents it from starting to matter after a few more quarterly cycles.
- **Valuation ratios re-anchored daily:** BolsAI computes `pl/pvp/market_cap/p_*/ev_*` at filing date; `recompute_valuation_daily()` rescales to current close (keeps `fundamentals_available_date` in output for any downstream consumer). Known ceiling: mid-quarter splits skew ratios until next filing (build warns).
- **All feature engineering is in Stage 2**, not deferred downstream (technicals, fundamental ratios, macro-adjusted, CAGR backfill, split repair, volatility rolling rank).
- **Per-ticker own-history z-scores (`*_zhist_5y`, July 2026):** `compute_history_relative_features()` (`features.py`) adds a causal rolling robust z-score — `(x - rolling_median) / rolling_IQR`, 5y window — for 11 fundamental ratios (`pl`, `pvp`, `roe`, `net_margin`, `ebitda_margin`, `debt_equity`, `net_debt_ebitda`, `earnings_yield`, `book_to_market`, `current_ratio`, `asset_turnover`) and 2 daily liquidity ratios (`amihud_illiquidity`, `turnover_ratio`). Answers "how unusual is this value for *this company*," distinct from the global `RobustScaler`'s cross-sectional level view (`scale_features.py`) and `cross_sectional.py`'s peer-relative view — see `docs/PER_TICKER_SCALING_PLAN.md`. Stateless (a plain trailing rolling stat, not a fitted transform): no train/test split to manage, valid unchanged under any evaluation methodology. Fundamentals are deduped to one row per `reference_date` before rolling (rolling the daily-forward-filled panel directly would be ~65x redundant), then mapped back onto every daily row of that quarter. Warm-up (< `FUND_ZHIST_MIN_QUARTERS`=8 quarters / `DAILY_ZHIST_MIN_DAYS`=252 days of history) is NaN, a leading prefix like every other rolling-window feature in this pipeline.
- **Scaler fit boundary is injected, not hardcoded (`iter_fit_windows()`, July 2026):** `scale_features.py`'s `fit_scaler(dataset, window)` takes a `FitWindow` (`manifest.py`) resolved from the active `split_config.json` via `iter_fit_windows()` — the one seam between the evaluation methodology (today: a single fixed split) and scaler fitting. A future rolling/expanding/multi-fold split format only changes `iter_fit_windows()`; `fit_scaler_on_train_split()` remains as a back-compat wrapper reproducing today's single-window behavior exactly. `sync_dataset_version()` now also snapshots `data/processed/scalers/` into `dataset_v{N}/`.
- **FIIs deferred:** stocks only (prices/fundamentals/dividends). FIIs are a separate asset class; add if scope expands to mixed-asset.
- **BolsAI:** key in `.env`, loaded by `config.load_env()` (stdlib parser). Backfill only — paid ~€0.10/1K calls. Caps: prices `limit<=5000` (date-window paginated), fundamentals `limit<=88` (use 80).
- **yfinance:** free incremental refresh. Prices/dividends full history to 2000; fundamentals only ~4–6 quarters (enough for quarterly refresh).
- **BCB series:** selic=11 (daily), cdi=12, ipca=433 — **NOT 432** (that's the annual meta target).
- **Benchmark:** BOVA11 (IBOV proxy ETF) collected automatically; prices only. **Now the true market series for `beta_1y`/`momentum_vs_market_*` (2026-07-24 fix):** these previously benchmarked against the equal-weighted mean return of whatever tickers happened to be in the collected panel that day — silently redefining "the market" as "the companies that survived to dataset-end," a benchmark-level survivorship bias distinct from the universe-selection-level one above. `build_ml_dataset.main()` now captures BOVA11's rows right after `apply_ticker_continuity()` (before the fundamentals-coverage filter drops it — an ETF has none by design), runs it through the same `compute_price_features()` as every other ticker, and threads the resulting return series into `compute_cross_sectional_features()` as a required `benchmark` argument. BOVA11 itself still never becomes a row in the output dataset (threaded through purely as an external reference series), so this changes only the *definition* of these two feature groups, not row/ticker counts or manifest shape. `momentum_vs_sector_*` is unaffected (still a real sector-peer comparison, no equivalent "benchmark" concept applies there).
- **Cross-sectional exclude-self mean, NaN-dilution bug FIXED 2026-07-24:** `cross_sectional.py`'s `_exclude_self_mean()` (used for `momentum_vs_market_*` before the BOVA11 fix above, and still the pattern to know about even though those columns no longer use it) derived its denominator from `groupby(...).transform("size")` — a blanket row count that includes tickers whose value is still NaN (e.g. a ticker in `return_12m`'s warm-up year) — while the numerator's `sum` already (correctly) skips those NaNs. Counting peers the numerator had already dropped biased every excluded-self mean toward zero, worst on thin/young-universe dates. Fixed to derive both sum and count from the value column's own `groupby(...).transform("count")`.
- **`days_since_fundamental` keyed to the wrong date, FIXED 2026-07-24:** measured `trade_date - reference_date` (the fiscal quarter-end the filing *describes*) instead of `trade_date - fundamentals_available_date` (when the market actually *saw* the filing) — understating true information age by the entire 45–90+ day filing lag, and inconsistent with `filing_lag_days`/`n_quarters_available` which already use the real availability date elsewhere in the same function.
- **`payout_ratio`/`dividend_coverage_ratio` annualization bug, FIXED 2026-07-24:** both used `div_value_recent` (the single most-recent ex-date's nominal payment, from `merge_dividends`' asof merge) as if it were the whole year's dividend — correct only for an annual payer, understating payout / inflating coverage for anyone paying quarterly or more often, and stair-stepping discontinuously at every ex-date. `compute_dividend_features()` now also computes `div_value_12m` (a trailing-12m nominal sum of per-event dividends, same window/convention as `div_yield_12m`/`div_count_12m` — just the un-normalized currency amount), and both ratios use it instead. `div_value_recent` is unchanged and still a legitimate "size of the last payment" feature in its own right.
- **Company info:** BolsAI-only (CVM metadata, rarely changes); refresh via `--mode full_scale` when new IPOs appear. Current dataset: 523 tickers total (373 ATIVO active + 85 CANCELADA delisted + 65 missing status); 4 quarantined (WDCN3 unadjusted splits unfixable, CAMB4 delisted 2019, LLIS3 delisted 2023, CCTY3 raw feed is not real trading data — mirrors CCRO3/Motiva's dead post-rename ticker across both BolsAI and yfinance).
- **`status` is a current-day snapshot, not point-in-time — do not use as a raw training feature:** `merge_company_info()` joins company_info's *today's* status (ATIVO/CANCELADA) onto every historical row of a ticker; confirmed 100% constant per ticker across its full history in the built dataset (2026-07-14 audit, `test_universe_integrity.py` §3.5). A model conditioned on `status` at a 2012 row would be seeing 2026 knowledge of whether that company survived — a feature-level lookahead trap, distinct from (and in addition to) the universe-selection-level survivorship bias in `TOP50_UNIVERSE_VALIDATION.md`. Left in the dataset deliberately (downstream point-in-time universe construction needs it to identify delisted names) — the burden is on any consumer training a model to exclude it from the point-in-time feature set. `sector` is the same kind of static join but carries far less outcome information, so it's lower-risk as a feature.
  - **Taint travels into derived columns too, FIXED 2026-07-24:** `manifest.LOOKAHEAD_TAINTED_COLS` used to list only `status`, but 6 `cross_sectional.py` columns are engineered directly from that same static, current-day `sector` join — `pl_zscore_sector`, `pvp_zscore_sector`, `roe_zscore_sector`, `debt_equity_zscore_sector`, `div_yield_sector_percentile`, `momentum_vs_sector_{1m,3m,12m}` — and carry the identical taint laundered into a numeric z-score/percentile/momentum figure that reads as clean. A consumer who dutifully dropped raw `status`/`sector` per the old list was still training on it through these. Now all 9 columns are recorded in `LOOKAHEAD_TAINTED_COLS`/the manifest's `lookahead_tainted_columns`. `momentum_vs_market_*` is *not* included here — it doesn't group by sector, but see the BOVA11-benchmark entry below for its own (different) survivorship concern, now fixed.
- **Data quality filters (Stage 2, enforced automatically):**
  - Filing lag filter: Drop fundamentals filed >180 days after quarter-end (0.9% of rows) — prevents lookahead from unreliable late filings
  - Close-price lookup: Replace BolsAI's stale close_price with actual close from `fundamentals_available_date` — prevents false >50% valuation jumps
  - Valuation re-anchoring: Rescale P/E, P/B, etc. to current close daily (not filing-date close) — keeps ratios current
  - Split repair: Detect and rescale unrecorded splits (67 events as of 2026-07-24) — prevents fake negative returns up to −99.99%
  - Sibling fill: Forward-fill missing company_info from same-CVM-company tickers (168,783 rows) — ensures all rows have sector/status metadata
  - Quarantine list: WDCN3 (raw close oscillates 6x, no repair), CAMB4 (delisted 2019, stale fundamentals), LLIS3 (delisted 2023, stale fundamentals) — eliminates data quality outliers.
- **NaN & extreme-value policy (implemented):**
  - Data quality filters (Stage 2):
    - Structural NaN (warm-up, pre-first-filing) trimmed by global start-date rule per universe.
    - Informative NaN (CAGR undefined from negative earnings/insufficient history) flagged: `cagr_earnings_defined`, `cagr_revenue_defined` (0/1); also tracks `n_quarters_available` (cumulative filing count) explaining all window-based NaNs.
    - Error NaN: prefix-shaped (no interior holes per ticker in merged fundamentals), detected via test `test_final_dataset.py::T_prefix_rule`. NaN-count regression vs previous build warned via `nan_regressions()` in `manifest.py` (logged but non-fatal, allows legitimate coverage changes).
    - Extreme ratio (144 rows |pl| > 400,000 dataset-wide, 95 in-universe, top-50): kept intact — denominators near zero are valid distress signals. No filled or clipped in the dataset; scaler's fit is robust (median/IQR) but transform is linear, so raw 400k → ~26k after scaling (still extreme, intentionally preserved). Model training handles via loss functions / clipping in the env.
  - Consumer-side (ml_agent env, not this repo): flags + neutral fills (e.g. fill CAGR NaN with 0), any NaN→0 transformation, hard `assert np.isfinite(obs).all()` before agent sees state, global start-date trim for the top-50 universe to drop pre-full-history rows.
- **Checkpoints/logs** (not git-tracked): Stage 1 `artifacts/checkpoints/{mode}/`, `artifacts/logs/collection/collection-*.log`.
- **Paths:** absolute via `Path(__file__).resolve().parents[N]`; always run from project root.
- **FutureWarnings suppressed:** `pct_change(fill_method=None)` for YoY growth; dropped all-NA columns per-file before concat. `repair.py`'s `volume`/`volume_adjusted` columns are cast to `float64` up front (2026-07-24 fix) instead of left `int64` until the final rescale-loop cleanup — in-place `*=`/`/=` on a slice of an `int64` column with a non-integer factor is a deprecated silent per-slice upcast pandas warns will become a hard error in a future version; the final `.round().astype("int64")` still converts back after all rescaling is done.

## Data on Disk

- **Raw (git-tracked):** full-scale universe, ~293 tickers + benchmark BOVA11, one parquet per ticker in `data/raw/{prices,fundamentals,dividends}/`. Prices/dividends current to 2026-06-30; fundamentals to 2026-03-31. Coverage isn't 100% uniform across types (e.g. a handful of tickers are missing a dividends file) — treat gaps as "not yet collected," not "confirmed zero," and re-run the relevant `collect_*` for that ticker to check. In the processed dataset this ambiguity is exposed directly: `has_dividends` (0/1, set in `merge_dividends()`) marks whether a ticker was ever collected at all, so `div_yield_12m == 0` can be told apart from "never collected" rather than silently reading as a confirmed zero.
- `data/raw/macro/{selic,cdi,ipca}.parquet` (one file per series) and `data/raw/company_info/company_info.parquet` (per-ticker static attributes: sector, cnpj, status, etc.) are market-wide reference tables, not per-ticker files.
- `data/raw/company_info/sectors.parquet` is a small aggregate `[sector name, ticker count]` table used to sanity-check how many companies fall in each sector — not a join key, not consumed by `build_ml_dataset.py`.
- `data/raw/corporate_events/corporate_events.parquet` is a market-wide split/inplit audit log; `company_info/sectors.parquet` and `corporate_events.parquet` are collected during `full_scale`/`prototype` runs only — skipped in `--mode update` so the free/keyless yfinance refresh path never needs a BolsAI key.
- **Processed:** `data/processed/ml_dataset.parquet` (created on first build), one row per ticker+date, plus `ml_dataset.manifest.json` (reproducibility snapshot) and `split_config.json` (walk-forward train/val/test date cutoffs — a filter, not materialized copies). Each build that actually changes output is also snapshotted immutably to `data/processed/dataset_v{N}/` (unchanged reruns don't bump `N`); cite `dataset_v{N}` when referencing exactly which build an experiment used. `data/processed/scalers/feature_scaler.joblib` + `scaler_metadata.json` are fit train-only (per `split_config.json`) via a separate, deliberate step (`scale_features.py`), not on every build. All of the above are gitignored and fully regenerable from `data/raw/`. `data/processed/README.md` is the one tracked exception (see its content for the full list) — it documents this ownership boundary; treat anything else that shows up in `data/processed/` (e.g. `ml_dataset_training.parquet`) as generated by pipelines outside this repo's `src/` (the `ml_agent` branch) and not something this repo can rebuild.

## Technology Stack

- **Python 3.10+** (`list[str]`, `dict | None`).
- **Data:** pandas, numpy, pyarrow.
- **APIs:** BolsAI REST (`httpx`, backfill), BCB SGS (requests, macro), `yfinance` (incremental).
- **Config:** stdlib `.env` parser (BolsAI key only).
- **Preprocessing:** scikit-learn (`ColumnTransformer`/`RobustScaler` in `scale_features.py`), `joblib` (scaler serialization) — Stage 2 direct dependencies.
- **Viz:** Plotly.
- **No test framework:** standalone `python script.py`.

## Knowledge Graph (graphify)

A persistent knowledge graph of this repo lives in `graphify-out/` (gitignored, regenerable). Built with the `graphify` skill (`/graphify`).

- **Query it first** for architecture/"how does X work"/"what calls Y" questions instead of re-reading source: `graphify query "<question>"` (BFS), `graphify path "A" "B"`, `graphify explain "<node>"`. The graph already exists — use it before a fresh scan.
- **Outputs:** `graphify-out/graph.html` (interactive), `GRAPH_REPORT.md` (god nodes, communities, surprising links), `graph.json` (raw).
- **Rebuild** after significant code/doc changes: `/graphify .` (full) or `/graphify . --update` (only new/changed files).
- **Semantic extraction backend:** code is AST-extracted (no key). Docs/papers use Gemini when `GEMINI_API_KEY`/`GOOGLE_API_KEY` is set (OpenAI-compatible endpoint; needs `graphifyy[gemini]` → the `openai` package); otherwise falls back to host-agent subagents.
