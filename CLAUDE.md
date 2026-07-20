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

### Stage 3: Train RL Agent (Iteration 1)

Prereq: Stage 2 complete (`data/processed/ml_dataset.parquet` + `top50_universe_membership.parquet`
on disk). EIIE portfolio-management agent (Jiang, Xu & Liang 2017), price-only, top-50 dynamic
quarterly universe, CDI-accruing cash, 2011–2026 window. Design + approved paper deviations:
`docs/eiie_agent/EIIE_AGENT_PLAN.md`.

```bash
python -m src.rl_agent.experiment --config configs/eiie_baseline.json --dry-run   # data + sanity checks only
python -m src.rl_agent.experiment --config configs/eiie_baseline.json             # full run: pretrain → OSBL backtest → baselines → report
python -m src.rl_agent.experiment --config configs/eiie_baseline.json --eval-split test  # final run (train+val pretrain, test backtest)
python -m src.rl_agent.sweep --config configs/eiie_baseline.json --seeds 1 2 3 4 -j 4    # parallel seed-ensemble / config sweep (per-job logs in experiments/sweep_logs/)
```
Output: `experiments/{run_name}_{timestamp}/` — `config.json`, `run_manifest.json` (seed, git commit,
dataset fingerprint, package versions), `sanity_report.txt`, `model.pt` (checkpoint), `report.html`
(agent vs. all 7 baselines: PV, reward curves, allocation, turnover/cost, metrics + bootstrap CIs),
`metrics_summary.json`, `report.json` (validation checklist).

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
- **Fast:** `test_features.py`, `test_merge.py`, `test_cross_sectional.py`, `test_compute_features_chunked.py`, `test_split_config.py`, `test_dataset_versioning.py`, `test_scale_features.py`, `test_company_siblings.py`, `test_ticker_continuity.py`; `tests/rl_agent/{test_config,test_data,test_pvm,test_environment,test_metrics,test_baselines,test_networks,test_train,test_sanity,test_plots,test_experiment,test_sweep}.py` (all synthetic-data — `test_train`/`test_experiment` exercise real gradient steps and the full orchestrator, just on tiny fabricated markets, never the real dataset or a real training run)
- **Data:** `test_final_dataset.py`, `test_top_traded_quality.py`, `test_universe_integrity.py`, `test_cagr_calculation.py`, `test_blue_chip_tickers.py`, `validate_vs_yfinance.py`, `test_collect_delisted.py`, `test_cvm_statements.py`, `tests/rl_agent/test_data_integration.py` (loads the real `PricePanel`: 172-wide global space, every in-window day has exactly 50 active members, no NaNs)

**Linting:**
```bash
ruff check .          # reports undefined names, unused imports/variables, bare-except
```

## Branches

- **main:** Stages 1–2 (data collection + dataset build). Latest stable.
- **build_dataset:** Stage 2 focus.
- **refactor:** adds Stage 3 iteration 1 (`src/rl_agent/`, this branch) — see `docs/eiie_agent/EIIE_AGENT_PLAN.md`.
- **ml_agent:** a separate, earlier PPO agent (masked 279-ticker universe); not this branch's Stage 3.

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

**Stage 3 (RL Agent, iteration 1)** — `src/rl_agent/`, price-only EIIE reproduction (`docs/eiie_agent/EIIE_AGENT_PLAN.md`):

| File | Purpose |
|------|---------|
| `config.py` | Frozen-dataclass `ExperimentConfig` ↔ JSON; every hyperparameter/date/cost rate config-driven, nothing hardcoded downstream |
| `paths.py` | Shared path constants for this package |
| `data.py` | `GlobalAssetIndex` (permanent 172-wide map, cash=0); `PricePanel.window_tensor()` (X_t, eq. 18) / `.price_relative()` (y_t, eq. 1, CDI-accruing cash); `validate_cdi_daily_percent()` units guard; observation prices read from the full `ml_dataset.parquet` (not the pre-built top50 file) so an entrant's lookback history isn't missing; `FEATURE_NORM` table lets `DataConfig.features` mix price-level channels (`close/high/low`, ÷value-at-t, paper eq. 18) with technical channels (`return_1m/3m/6m`, `price_vs_ma60`, `volatility_ratio_20_60`, `rsi_14`, `drawdown`, `volume_ratio_20d` — passthrough, a per-feature scale, masked/NaN→0) via `PricePanel.extra` |
| `pvm.py` | `PortfolioVectorMemory` — global-space `[T, 172]` weight buffer; `read_slots()`/`write()` bridge slot-space (network's fixed 50 inputs) and global space via `torch.gather`/`scatter`; a departing ticker's forced-sale liquidation falls out of the data structure, no special-case code |
| `environment.py` | `solve_mu`/`solve_mu_torch` (eq. 14, Theorem 1 fixed-point, converged vs. differentiable k-step); `drift_weights`/`drift_weights_torch` (eq. 7); `run_backtest()` — the one loop shared by the agent and every baseline |
| `networks.py` | `EIIECNN` (paper Fig. 2): kernel-height-1 convs keep each asset's stream independent, `w_{t-1}` inserted as an extra feature map before the final 1×1 conv, softmax over masked logits; `n_features` (conv1's input width) is just `len(cfg.data.features)`, so adding technical channels needs no network change |
| `train.py` | OSBL (Sec. 5.3): `sample_batch_starts()` (eq. 26 geometric recency bias), `train_step()` (entropy bonus scale-annealed per step, `entropy_schedule()`), `pretrain()` (held-out checkpoint-at-peak: scores the frozen policy every `checkpoint_eval_every` steps on a `checkpoint_holdout_days` slice carved off the train split's tail via `_score_holdout()`, restores + refreshes the PVM under the best-scoring `state_dict` at the end — returns `(losses, best_step, best_score)`), `run_online_backtest()` (interleaves `rolling_steps` updates via `run_backtest`'s `on_step` hook), checkpointing; `_PanelStore` precomputes all window tensors/price relatives once, GPU-resident (scales with channel count, ~200 MB at 3 channels), so per-step data prep is pure tensor indexing — verified bit-identical to the per-step numpy path |
| `sanity.py` | `run_sanity_checks()` — pre-training invariant gate (determinism, simplex weights, finite gradients/loss, baselines running cleanly); a dominant-asset toy market is a diagnostic, never a pass/fail gate |
| `baselines.py` | UBAH, UCRP, Best-Stock (hindsight), Random Portfolio/Rebalancing, Constant-Cash, BOVA11 — all through `run_backtest` except BOVA11 (evaluated directly from its own price series) |
| `metrics.py` | Full metrics suite (Sharpe/Sortino vs. CDI, Calmar, VaR/CVaR, turnover, cost drag, information ratio vs. BOVA11) + `block_bootstrap_ci()` |
| `plots.py` | `write_report()` — one self-contained HTML report (plotly.js embedded once): PV vs. every baseline, reward curves, allocation evolution, turnover/cost, metrics + CI table |
| `experiment.py` | CLI orchestrator: seed → load data → recompute a window-scoped split → sanity gate → pretrain → OSBL backtest → baselines → metrics → report → validation checklist |
| `sweep.py` | Parallel launcher: runs several experiment subprocesses at once (seed ensembles / config sweeps), bounded by `-j`, per-job logs |


## Critical Caveats

- **CAGR backfill is ON:** `fill_missing_cagr()` (which calls `fill_cagr_columns()` per ticker) runs unconditionally in `build_ml_dataset.py`'s main pipeline → dataset has `cagr_{earnings,revenue}_5y_final` populated. Coverage is ~60% from BolsAI; the backfill recovers an additional ~7%.
- **No lookahead (Stage 2) — ENFORCED:** `merge_asof(..., direction='backward')` on real CVM `fundamentals_available_date` (not fiscal period-end) — a price never sees a future fundamental. `volatility_*_percentile` use rolling-window rank, not global rank. Tests: `test_merge_honors_actual_filing_date`, `test_volatility_percentile_no_lookahead`. ✅ VERIFIED 2026-07-11.
- **Real filing dates (July 2026):** Fundamentals visible via CVM's `DT_RECEB` (received date), not fiscal `reference_date`. 41,530 filings from 1,223 companies, 100% coverage of 293-ticker universe; 4,657 rows (0.7%) would have violated a fixed 45/90-day buffer. Sourced from free, keyless CVM open-data portal; integrated via `src/data_collection/cvm/filing_dates.py` (`python -m src.data_collection.cvm_statements --step filing_dates`).
- **Unadjusted splits REPAIRED:** 53 corporate events in BolsAI's `adj_*` columns were never back-adjusted, causing fake returns up to −99.99%. `repair_unadjusted_splits()` detects and rescales all pre-event rows *and volumes*: a 1:4 split divides prices by 4 and multiplies `volume`/`volume_adjusted` by 4 (same economic activity, more shares). Rescaling is critical for `amihud_illiquidity` and `turnover_ratio` features. `hl_ratio` uses `adj_high/adj_low` (not raw scales). WDCN3 quarantined (unfixable data corruption). ✅ VERIFIED 2026-07-11. ✅ Volume scaling VERIFIED 2026-07-15.
- **Ticker continuity & splicing (July 2026 fixes):** Renames/mergers/exchanges are spliced via `apply_ticker_continuity()` *after* `repair_unadjusted_splits()` (not before), so splits are repaired under each leg's original ticker name before being renamed onto the survivor. Splicing rules: (1) **rename** = same legal entity, splice prices + fundamentals, drop old ticker. (2) **merger** = exchange ratio, scale old-leg prices by ratio **and volume inversely by ratio** (keeps dollar volume = volume×price invariant across the splice, same rationale as split-repair volume scaling — otherwise `amihud_illiquidity` jumps by `ratio` right at the merger boundary), drop old-leg fundamentals. (3) **keep_separate** = parallel-trading acquirer (e.g., SulAmérica acquired by RDOR, which had its own IPO 2 years earlier), both legs stay as independent series; old treated as delisted. (4) **tender** = cash-out, no splice. Vendor aliases (ARZZ3→AZZA3, RRRP3→BRAV3, etc.) consolidated via `rename` entries where the new file contains full history under both names. Boundary-matching assumption (new ticker's first trade == splice point) is guarded: parallel-trading cases caught and rejected. Adj_close reconciliation factors (inherent basis mismatches between old/new vendor series) are validated [1/50, 50] sane range. Event rekeying: `repair.py` builds ticker-descendant chains from the map so splits recorded under old names (e.g., VVAR3) still match post-rename rows (BHIA3). ✅ All tests pass post-fix. ✅ TIMP3→TIMS3 factor sane (0.6963, not 6963).
- **Returns ARE dividend-adjusted (total return), not price-only:** `log_return`/`return_{1m,3m,6m,12m}`/`excess_return`/`real_return` (`features.py`) are all derived from `adj_close`, and `adj_close` empirically bakes in dividend reinvestment, not just splits — confirmed by testing `adj_close/close` ratio drift against known dividends on split-free windows (e.g. BBAS3 post-split: predicted vs. observed ratio jumps matched to within ~0.04 pp per ex-dividend date). `div_yield_12m`/`div_count_12m` remain separately-tracked features on top of this, not double-counted into returns.
  - **Known, undocumented-by-vendor limitation — BolsAI/yfinance dividend-adjustment methodology diverges:** confirmed by direct measurement (145 tickers, BolsAI-only rows, split-free windows): median 4.9pp divergence between BolsAI's observed `adj_close` ratio drift and what `data/raw/dividends` alone predicts, often 20pp+. BolsAI's adjustment consistently implies *more* cumulative discount than our dividends table explains — i.e. our dividends table is missing some distribution type BolsAI's adjustment already correctly captures (bonus shares/subscription rights suspected, unconfirmed). **Do not "fix" this by recomputing `adj_close` from `data/raw/dividends`** — that would systematically under-adjust and regress returns. `validate_vs_yfinance.py:7` already flags this by skipping `adj_close` cross-validation as "uninformative." No fix available with current data; flagged here so it isn't rediscovered as a bug.
  - **Staleness across `--mode update` runs — FIXED:** yfinance's `auto_adjust=True` backward-adjusts a fetch window relative to "now" at fetch time. If each update only fetched rows after the last checkpoint (like every other collector), each quarterly batch would freeze at its own anchor and never get revisited — a dividend paid after one quarter's fetch would permanently fail to propagate into that quarter's already-stored `adj_close`, one small discontinuity per update, forever. `collect_prices_yf` (`yf_collectors.py`) now re-fetches its *entire* yfinance-sourced span every run via `_prices_fetch_start()` (anchored to the earliest yfinance row on disk, marked by `NaN num_trades`, not the latest), so the whole yfinance era stays internally consistent. Empirically verified this wasn't yet causing damage before the fix (max BolsAI→yfinance gap across 285 tickers was 1–3 days — this was the first `--mode update` run for all of them), but the fix prevents it from starting to matter after a few more quarterly cycles.
- **Valuation ratios re-anchored daily:** BolsAI computes `pl/pvp/market_cap/p_*/ev_*` at filing date; `recompute_valuation_daily()` rescales to current close (keeps `fundamentals_available_date` in output for agent state). Known ceiling: mid-quarter splits skew ratios until next filing (build warns).
- **All feature engineering is in Stage 2**, not deferred to the agent (technicals, fundamental ratios, macro-adjusted, CAGR backfill, split repair, volatility rolling rank).
- **Per-ticker own-history z-scores (`*_zhist_5y`, July 2026):** `compute_history_relative_features()` (`features.py`) adds a causal rolling robust z-score — `(x - rolling_median) / rolling_IQR`, 5y window — for 11 fundamental ratios (`pl`, `pvp`, `roe`, `net_margin`, `ebitda_margin`, `debt_equity`, `net_debt_ebitda`, `earnings_yield`, `book_to_market`, `current_ratio`, `asset_turnover`) and 2 daily liquidity ratios (`amihud_illiquidity`, `turnover_ratio`). Answers "how unusual is this value for *this company*," distinct from the global `RobustScaler`'s cross-sectional level view (`scale_features.py`) and `cross_sectional.py`'s peer-relative view — see `docs/PER_TICKER_SCALING_PLAN.md`. Stateless (a plain trailing rolling stat, not a fitted transform): no train/test split to manage, valid unchanged under any evaluation methodology. Fundamentals are deduped to one row per `reference_date` before rolling (rolling the daily-forward-filled panel directly would be ~65x redundant), then mapped back onto every daily row of that quarter. Warm-up (< `FUND_ZHIST_MIN_QUARTERS`=8 quarters / `DAILY_ZHIST_MIN_DAYS`=252 days of history) is NaN, a leading prefix like every other rolling-window feature in this pipeline.
- **Scaler fit boundary is injected, not hardcoded (`iter_fit_windows()`, July 2026):** `scale_features.py`'s `fit_scaler(dataset, window)` takes a `FitWindow` (`manifest.py`) resolved from the active `split_config.json` via `iter_fit_windows()` — the one seam between the evaluation methodology (today: a single fixed split) and scaler fitting. A future rolling/expanding/multi-fold split format only changes `iter_fit_windows()`; `fit_scaler_on_train_split()` remains as a back-compat wrapper reproducing today's single-window behavior exactly. `sync_dataset_version()` now also snapshots `data/processed/scalers/` into `dataset_v{N}/`.
- **Stage 3 training performance & reproducibility (July 2026, `TRAINING_SPEEDUP_PLAN.md`):**
  training was CPU-bound (tiny ~3k-param CNN, per-step numpy prep dominated); `train.py`'s
  `_PanelStore` now precomputes everything GPU-resident once — verified bit-identical, ~2×
  faster (~6 ms → ~2.5–3.4 ms/step). `train.compile` (`torch.compile`, config flag, default
  **off**) measured only 1.13× here and is NOT bit-identical to eager — leave off for runs
  that must reproduce exactly. Same-seed GPU runs drift *across processes* (cuDNN picks conv
  algorithms per-process; the sanity determinism gate only compares within-process) — CPU runs
  are exactly reproducible; `torch.use_deterministic_algorithms(True)` is deliberately not
  enabled. For seed ensembles / hyperparameter sweeps use `python -m src.rl_agent.sweep`
  (parallel subprocesses, ~0.5 GB GPU each; run-dir timestamps carry microseconds so
  same-second launches can't collide).
- **Stage 3 cash-attractor diagnosis (July 2026, `docs/eiie_agent/EIIE_DIAGNOSIS_PLAN.md`):** the agent's default
  failure mode is converging to 100% cash — CDI accrues ~8.65%/yr in log-space vs. equal-weight's
  ~8.22%, so unlike the paper's 0%-return cash asset, the training gradient has no restoring force
  pushing back once every asset score drifts down; softmax saturates (~1e-9/asset), the gradient
  vanishes, and more training makes it worse, not better. Fixes, both in `TrainConfig`:
  `entropy_beta_start/end/anneal_frac` (linear decay over the first `anneal_frac` of pretrain,
  flat at `entropy_beta_end` after — including through the whole online/live phase) forces early
  exploration instead of hoping a fixed value gets lucky; `checkpoint_holdout_days`/
  `checkpoint_eval_every` (`train.pretrain()`) periodically scores the frozen policy on a
  held-out tail of the TRAIN split (never val/test) and restores the best-scoring checkpoint
  instead of trusting wherever `pretrain_steps` happens to land — measured case: the same seed
  went from +47% (100k steps) to −71% (2M steps) on identical config, budget alone overfitting
  the policy past its peak. Even with both fixes, escaped policies are bistable (all-cash or
  all-in, no steady diversified sleeve) and gravitate to the highest-volatility names in the
  universe — a raw price CNN can't distinguish a real trend from a dead-cat bounce. Motivated the
  July 2026 technical-feature-channel work below.
- **Stage 3 technical feature channels (July 2026):** `DataConfig.features` can mix `close/high/low`
  with technical columns already computed by Stage 2 (`return_1m/3m/6m`, `price_vs_ma60`,
  `volatility_ratio_20_60`, `rsi_14`, `drawdown`, `volume_ratio_20d` — see `configs/eiie_features.json`).
  `data.FEATURE_NORM` routes each by kind: price levels divide by value-at-t (paper eq. 18); technicals
  passthrough (optionally rescaled, e.g. `rsi_14` ÷100) since they're already stationary — dividing a
  return by "the return at t" would be meaningless. **Technical columns are `ffill`-only, NOT `bfill`'d**
  in `load_price_panel`, unlike `close/high/low`: a price's pre-listing days are always masked out
  downstream so backfilling them is harmless, but a technical feature's own warm-up NaN (e.g.
  `return_6m` before 126 days of a ticker's own history) can still land on a day its slot IS active —
  bfilling would leak a later, now-defined value backward into that unmasked training row, a lookahead
  bug specific to technicals that prices never had. `pl`/`pvp` (PE/PB) are deliberately NOT wired in yet
  — 27–30% NaN and tails to ±2000 need a non-linear squash (pseudo-log or cross-sectional rank) plus a
  companion `*_isnan` mask channel, not plain clipping, or they blow out conv gradients.
- **FIIs deferred:** stocks only (prices/fundamentals/dividends). FIIs are a separate asset class; add if agent scope expands to mixed-asset.
- **BolsAI:** key in `.env`, loaded by `config.load_env()` (stdlib parser). Backfill only — paid ~€0.10/1K calls. Caps: prices `limit<=5000` (date-window paginated), fundamentals `limit<=88` (use 80).
- **yfinance:** free incremental refresh. Prices/dividends full history to 2000; fundamentals only ~4–6 quarters (enough for quarterly refresh).
- **BCB series:** selic=11 (daily), cdi=12, ipca=433 — **NOT 432** (that's the annual meta target).
- **Benchmark:** BOVA11 (IBOV proxy ETF) collected automatically; prices only.
- **Company info:** BolsAI-only (CVM metadata, rarely changes); refresh via `--mode full_scale` when new IPOs appear. Current dataset: 523 tickers total (373 ATIVO active + 85 CANCELADA delisted + 65 missing status); 4 quarantined (WDCN3 unadjusted splits unfixable, CAMB4 delisted 2019, LLIS3 delisted 2023, CCTY3 raw feed is not real trading data — mirrors CCRO3/Motiva's dead post-rename ticker across both BolsAI and yfinance).
- **`status` is a current-day snapshot, not point-in-time — do not use as a raw training feature:** `merge_company_info()` joins company_info's *today's* status (ATIVO/CANCELADA) onto every historical row of a ticker; confirmed 100% constant per ticker across its full history in the built dataset (2026-07-14 audit, `test_universe_integrity.py` §3.5). A model conditioned on `status` at a 2012 row would be seeing 2026 knowledge of whether that company survived — a feature-level lookahead trap, distinct from (and in addition to) the universe-selection-level survivorship bias in `TOP50_UNIVERSE_VALIDATION.md`. Left in the dataset deliberately (downstream point-in-time universe construction needs it to identify delisted names) — the burden is on any consumer training a model to exclude it from the point-in-time feature set. `sector` is the same kind of static join but carries far less outcome information, so it's lower-risk as a feature.
- **Data quality filters (Stage 2, enforced automatically):**
  - Filing lag filter: Drop fundamentals filed >180 days after quarter-end (0.9% of rows) — prevents lookahead from unreliable late filings
  - Close-price lookup: Replace BolsAI's stale close_price with actual close from `fundamentals_available_date` — prevents false >50% valuation jumps
  - Valuation re-anchoring: Rescale P/E, P/B, etc. to current close daily (not filing-date close) — keeps ratios current
  - Split repair: Detect and rescale unrecorded splits (53 events) — prevents fake negative returns up to −99.99%
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
- **FutureWarnings suppressed:** `pct_change(fill_method=None)` for YoY growth; dropped all-NA columns per-file before concat.

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
- **Preprocessing:** scikit-learn (`ColumnTransformer`/`RobustScaler` in `scale_features.py`), `joblib` (scaler serialization) — both were already installed for Stage 3 (agent), now also a Stage 2 direct dependency.
- **Viz:** Plotly.
- **No test framework:** standalone `python script.py`.

## Knowledge Graph (graphify)

A persistent knowledge graph of this repo lives in `graphify-out/` (gitignored, regenerable). Built with the `graphify` skill (`/graphify`).

- **Query it first** for architecture/"how does X work"/"what calls Y" questions instead of re-reading source: `graphify query "<question>"` (BFS), `graphify path "A" "B"`, `graphify explain "<node>"`. The graph already exists — use it before a fresh scan.
- **Outputs:** `graphify-out/graph.html` (interactive), `GRAPH_REPORT.md` (god nodes, communities, surprising links), `graph.json` (raw).
- **Rebuild** after significant code/doc changes: `/graphify .` (full) or `/graphify . --update` (only new/changed files).
- **Semantic extraction backend:** code is AST-extracted (no key). Docs/papers use Gemini when `GEMINI_API_KEY`/`GOOGLE_API_KEY` is set (OpenAI-compatible endpoint; needs `graphifyy[gemini]` → the `openai` package); otherwise falls back to host-agent subagents.
