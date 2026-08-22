# CLAUDE.md

Guidance for Claude Code working in this repo.

## Overview

**Project:** Brazilian-equity (+ US-equity, in progress) dataset pipeline for ML/portfolio applications.

**Three stages** (all scripts run from project root):
1. **Data Collection** — staged prototype→validation→full-scale pipeline (checkpointing, logging, validation). BR (`src/data_collection/br/`) is mature; US (`src/data_collection/us/` + `sec/`) is an active expansion, see `docs/US_EQUITIES_EXPANSION_PLAN.md`.
2. **Dataset Build** — merge raw data → derived features (technical, fundamental, macro) → clean → ML-ready parquet, no lookahead bias. BR is production; US (`build_us_dataset.py`) is validated at 500-ticker scale, full-universe scale-up in progress.
3. **Portfolio Construction** (`src/portfolio/`, active research, BR only) — point-in-time universe → LightGBM alpha forecaster → Ledoit-Wolf risk model → cost-aware convex optimizer → contrarian cash overlay, evaluated via walk-forward backtest. Not a shipped strategy — current finding is the ML pipeline doesn't yet beat equal-weight net of cost.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add BOLSAI_API_KEY=sk_...  (backfill only; .env is gitignored)
```

## Run Commands

### Stage 1: Collect Raw Data

```bash
# Backfill (one-time historical via BolsAI, 2000–present); resumes from checkpoints, idempotent
python -m src.data_collection.br.pipeline --mode full_scale            # all ~500+ tickers
python -m src.data_collection.br.pipeline --mode full_scale --dry-run  # preview ticker list
python -m src.data_collection.br.pipeline --mode prototype --tickers PETR4 VALE3

# Quarterly incremental refresh (free yfinance, no key; >99% cost savings)
python -m src.data_collection.br.pipeline --mode update

# Alternative one-command BR+US top-up (macro/dividends/prices/fundamentals, tail-only by
# default) -- separate entrypoint, NOT layered on top of pipeline.py's own --mode update,
# but routes fundamentals through the same DATA_SOURCE dispatch (pipeline._collect), so the
# CVM-vs-BolsAI-vs-yfinance choice is the one place either entrypoint reads from
python -m src.data_collection.refresh

# Delisted-company recovery (NOT run by the two commands above -- pipeline.py's
# per-ticker collectors gate on company_info status=ATIVO by construction, so a
# routine full_scale/update run alone silently regresses to a survivor-only
# universe; run these separately, same cadence as full_scale):
python -m src.data_collection.br.collect_delisted            # prices for delisted/never-collected tickers
python -m src.data_collection.br.cvm_statements               # CVM fundamentals + cancellation registry for delisted names
python -m src.build_dataset.terminal_events                   # after build_ml_dataset.py: realized payoff for tickers that died inside the panel

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

# US analogue (validated at 500-ticker scale; full-universe scale-up in progress, see Data on Disk below)
python -m src.build_dataset.build_us_dataset             # → data/processed/us_ml_dataset.parquet
```

### Stage 3: Portfolio Construction (`src/portfolio/`, active research)

Prereq: Stage 2 complete. BR only; not wired to the US dataset. Design docs: `docs/PORTFOLIO_ARCHITECTURE_PROPOSAL.md`
(what), `docs/PORTFOLIO_IMPLEMENTATION_PLAN.md` (how it was built), `docs/PORTFOLIO_IMPROVEMENT_PLAN.md`
(live research log — **read its "STOP" banner first**: pre-Phase-V sweep conclusions don't survive
correction for the number of configs compared).

```bash
python -m src.portfolio.run_baseline              # equal-weight + BOVA11 + 100%-CDI baselines, ~2s
python -m src.portfolio.run_alpha_diagnostic       # walk-forward LightGBM alpha only, rank-IC + naive alpha-sort, ~4min
python -m src.portfolio.run_full_backtest          # full alpha→Sigma→optimizer pipeline vs. all baselines
python -m src.portfolio.diagnose_contrarian        # contrarian overlay signal check (seconds, no training)
python -m src.portfolio.reanalyze                  # re-score a saved run without retraining
python -m src.portfolio.visualize_portfolio        # Plotly dashboard for a saved run
python -m src.portfolio.plot_tree                  # render one LightGBM tree (needs system `dot`/graphviz)
```

**Current state** (per `PORTFOLIO_IMPROVEMENT_PLAN.md`, last recorded run 2026-07-26): alpha signal is
real but weak (OOS rank-IC mean 0.056, positive on 65% of dates), but the full pipeline doesn't beat
equal-weight net of cost, and widening the universe made it worse. Behaves like a crisis hedge that
bleeds carry in calm markets. Honest, open research result — the project's own decision framework
(Phase V) explicitly allows "ship equal-weight, keep this as a research project" as a valid outcome.
Check that doc for the live status before citing any return/Sharpe number.

### Utilities

```bash
python src/build_dataset/cagr_handler.py --ticker PETR4  # CAGR calculator
python src/visualizations/financial_view.py              # BBAS3 nominal vs inflation-adjusted vs SELIC (live yfinance)
jupyter notebook src/visualizations/exploration.ipynb    # full dataset validation + insights
```

### Tests

Plain Python scripts (not a pytest suite — no fixtures/conftest, no `pytest` invocation to
collect the whole tree), each runnable standalone. Unified test runner:

```bash
python tests/run_all.py --group fast   # pure-code unit tests, no data files needed — used by CI
python tests/run_all.py --group data   # needs git-tracked data/raw/* + built ml_dataset.parquet
python tests/run_all.py --group all
```

The `FAST`/`DATA`/`NON_BLOCKING` lists in `tests/run_all.py` are the source of truth for exactly
which script is in which group — don't hand-copy the roster here, it drifts. Shape: `FAST` covers
`tests/build_dataset/`, `tests/portfolio/`, and `tests/data_collection/` unit tests (synthetic data,
runs anywhere); `DATA` covers final-dataset/universe/vendor-cross-validation checks that need the
real `data/raw/` + a built dataset; `NON_BLOCKING` is the subset of `DATA` that hits a live vendor
and shouldn't fail CI on vendor flakiness.

**New test files:** end with `if __name__ == "__main__": raise SystemExit(pytest.main([__file__]))`
and write real `test_*` functions with bare `assert` (see `tests/data_collection/test_pipeline_dispatch.py`)
— `pytest` is already pinned and it's the largest single convention already in use (15 of 50 files).
Don't migrate the other 32 hand-listed `if __name__ == "__main__": test_a(); test_b(); ...` files to
match — `tests/run_all.py`'s `main_block_drift()` check already guards those against a silently
uncalled test function, so the migration would buy consistency the guard already delivers as a
15-line check, for a 32-file diff. Convention for new files, guard for old ones.

**Linting:**
```bash
ruff check .          # reports undefined names, unused imports/variables, bare-except
```

## Branches

- **main:** Stages 1–2. Latest stable.
- **build_dataset:** Stage 2 focus.
- **refactor:** Stages 1–2 restarted here after a 2026-07-23 reset wiped prior modeling work (a Stage 3 RL agent + Stage 4 encoder + M/H-series research lineage — none reached a working result; recoverable from git history before that date but not to be reused without re-reading it). Since then this branch has grown a **new**, unrelated Stage 3 (`src/portfolio/`, see above).
- **ml_agent:** a separate, earlier PPO agent (masked 279-ticker universe); unrelated to this branch's reset.

## Architecture

### Data Flow

```
BCB SGS (SELIC/CDI/IPCA) + BolsAI (OHLCV + quarterly fundamentals + dividends)
        ↓
data/raw/br/{prices,fundamentals,macro,dividends,company_info}/
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

**Stage 1 (Data Collection)** — `src/data_collection/`, market-namespaced (`docs/DATA_COLLECTION_REORGANIZATION_PLAN.md`): shared infra at the package root, BR/US-only orchestration in `br/`/`us/`, `cvm/`+`sec/` unchanged (already unambiguous), one-time incident scripts in `one_off/` (never imported elsewhere):

| Module | Purpose |
|--------|---------|
| `config.py` | Shared config (tickers, keys, paths, retries, `DATA_SOURCE` per-type source switch); `RAW_DIR` parents `BR_RAW_DIR`/`US_RAW_DIR` |
| `client.py` | BolsAI HTTP wrapper (retries, backoff): `make_client()`, `get_json()` |
| `checkpoint.py` | Resume state (JSON per collector) |
| `validate.py` | Quality gates (schemas, ranges, continuity) → `ValidationResult` |
| `storage.py` | `_merge_save()` (idempotent append+dedup+validate+write) + `_chunk_dates()` — generic, used by every collector |
| `ratios.py` | Vendor-neutral fundamentals algebra: `compute_ratios()`, `FUND_FULL_COLS` — shared by the yfinance, CVM, and every SEC tier |
| `yf/` | yfinance collectors, split by concern: `_common.py` (shared fetch/repair helpers), `prices.py` (`collect_prices_yf`, `backfill_price_gap`), `fundamentals.py` (`collect_fundamentals_yf`), `dividends.py` (`collect_dividends_yf`, `collect_splits_yf`) — shared by BR's `--mode update` and US's full backfill |
| `br/collectors.py` | BolsAI: `collect_{prices,fundamentals,company_info,dividends,corporate_events,sectors}()` |
| `br/macro.py` | BCB SGS (SELIC/CDI/IPCA), free/keyless — split out of `collectors.py` as the only live-by-default function in a module whose every other function needs a paid BolsAI key |
| `br/pipeline.py` | BR orchestration CLI; dispatches to BolsAI/yfinance/CVM per `DATA_SOURCE` (default: yfinance prices/dividends, CVM fundamentals — BolsAI is opt-in only, see `docs/BOLSAI_EXIT_PLAN.md`). **Not a whole-panel rebuild tool**: `pipeline.py`'s fundamentals path always scopes to `_active_tickers()` (status == ATIVO) before calling `collect_fundamentals_cvm`/`build_fundamentals` — invisible at the call site, and the exact thing that would silently leave delisted-only tickers' fundamentals unmigrated (`docs/DATA_LAYER_CORRECTNESS_PLAN.md` §1 caught this at 115 stranded files). A full-crosswalk rebuild needs `cvm.ratios.build_fundamentals(tickers=None, rebuild=True)` directly. |
| `br/collect_delisted.py` | Price backfill for delisted/never-collected BR tickers |
| `br/cvm_statements.py` + `cvm/` | CVM open-data collection (delisted fundamentals + real filing dates); `--step` CLI over `cvm/{http,crosswalk,statements,shares,ratios,company_info,filing_dates}.py` |
| `br/stats.py` | Post-collection data audit (BR only) |
| `us/fred_collectors.py` | FRED (US macro), keyless `fredgraph.csv` |
| `us/pipeline.py` | US orchestration CLI (`--mode`/`--tickers`/`--dry-run`/`--steps`, same shape as `br/pipeline.py`) over the six `run_*` stage functions — universe, macro, prices, dividends, fundamentals, company_info |
| `sec/` | US fundamentals from SEC EDGAR: `http.py` (shared retry GET), `universe.py` (point-in-time filer roster), `crosswalk.py` (ticker↔CIK + `CIK_OVERRIDES`), `companyfacts.py` (XBRL 2009+/IFRS tier), `fds.py` (EX-27 tier, 1995–2000), `selected_financial_data.py` (Item 6 annual tier, fills 2001–2006 gap), `tenq.py` (10-Q inline-HTML tier, real Q1–Q3 resolution for that same 2001–2006 window; Q4 is derived cross-tier against item6's annual total), `fundamentals.py` (combines all tiers, point-in-time `fundamentals_available_date`). Extensive bug log: `docs/US_EQUITIES_EXPANSION_PLAN.md`. |
| `one_off/*` | One-time incident-fix scripts (confirmed BolsAI data gaps/corruption for specific tickers); not part of the regular pipeline |

**Stage 2 (Dataset Build)** — `src/build_dataset/`, split by pipeline stage:

| File | Purpose |
|------|---------|
| `build_ml_dataset.py` | Orchestration: `main()` + memory-bounded `compute_features_chunked()` (3-pass: per-ticker features → cross-sectional → clean+write) |
| `paths.py` | Shared path constants |
| `loaders.py` | `load_prices()`, `load_fundamentals()`, `load_company_info()`, `load_dividends()`, `company_siblings()` |
| `repair.py` | `repair_unadjusted_splits()` — rescales `adj_*` history where a split was left unadjusted |
| `continuity.py` | `apply_ticker_continuity()` — splices renamed/merged tickers |
| `quality_filters.py` | Coverage + filing-lag gates |
| `merge.py` | The 4 `merge_*` functions (prices+fundamentals, company_info, macro, dividends) |
| `features.py` | Per-ticker features: CAGR backfill, dividend yield, price technicals, fundamental ratios/trends, valuation re-anchoring, `compute_history_relative_features()` (`*_zhist_5y`) |
| `cross_sectional.py` | `compute_cross_sectional_features()` — sector/market-relative, needs the full universe at once |
| `clean.py` | `clean_dataset()` — final dedupe/inf-to-NaN pass |
| `manifest.py` | `write_manifest()`, `compute_split_dates()`, `iter_fit_windows()`, `write_split_config()`, `sync_dataset_version()` |
| `cagr_handler.py` | CAGR calc/fill (BolsAI first, backfill from earnings/revenue) |
| `scale_features.py` | Fits `ColumnTransformer` (RobustScaler on ratio cols) train-only, per `split_config.json` |
| `build_us_dataset.py` | US analogue — reuses BR's price/dividend/cross-sectional/cleaning stages unchanged, adds only what's genuinely different (sector mapping, macro, daily valuation, liquidity gate), skips BR-only steps that don't apply (split/continuity repair, filing-lag gate). Rationale: `docs/US_DATASET_BUILD_PLAN.md`. |

**Stage 3 (Portfolio Construction)** — `src/portfolio/`, BR only, active research. One file per concern, thin wrappers over Stage 2 output:

| File | Purpose |
|------|---------|
| `universe.py` | Thin wrapper over Stage 2's `build_top50_universe.py` (point-in-time, no-lookahead, quarterly-rebalanced membership) |
| `labels.py` | `forward_excess_return()` — forward H-day return over CDI, no-lookahead-by-construction |
| `features.py` | The literal ~121-column keep-list, checked live against `manifest.LOOKAHEAD_TAINTED_COLS` |
| `backtest.py` | `run_backtest()` — quarterly-rebalanced harness; weights drift with prices between rebalances (deliberate) |
| `risk.py` | `shrinkage_cov()` (Ledoit-Wolf) + cash-row augmentation + PSD/conditioning checks — raw sample cov over 30–500 assets is numerically unusable at this scale |
| `alpha.py` | Stage A: LightGBM regression on the forward-excess-return label, walk-forward retrain, purged + embargoed |
| `optimizer.py` | Stage B: cost-aware convex program (`cvxpy`) — alpha vs. risk-aversion vs. one-way turnover cost, with a contrarian exposure cap |
| `contrarian.py` | Layer 2 "cannons/violins" cash↔equity overlay — a 1-parameter economic rule, deliberately not a learned model (too few crisis episodes to fit one without overfitting) |
| `pipeline.py` | Wires alpha + risk + optimizer into one `weights_fn` the backtest harness drives |
| `metrics.py` | `full_report()` — annualized return, Sharpe, deflated Sharpe, max drawdown, turnover/holding-period, regime slices |
| `artifacts.py` | Persists backtest runs to `artifacts/backtests/` for re-analysis without retraining; append-only `trials.csv` for an honest `n_trials` |
| `run_baseline.py`, `run_alpha_diagnostic.py`, `run_full_backtest.py`, `diagnose_contrarian.py`, `reanalyze.py` | CLI drivers per research question — see Run Commands |
| `visualize_portfolio.py`, `plot_tree.py` | Plotly dashboard for a saved run; graphviz render of one LightGBM tree |

## Critical Caveats

Gotchas that will bite again if not known. Historical forensics (how each was found/measured) live in
git history and `docs/*AUDIT*.md`; this list is only the actionable "what's true now."

**Lookahead / point-in-time integrity**
- No lookahead (Stage 2), enforced: `merge_asof(..., direction='backward')` on real CVM `fundamentals_available_date` (`DT_RECEB`, not fiscal `reference_date`), not filing-date close. `volatility_*_percentile` uses rolling-window rank, not global. Tests: `test_merge_honors_actual_filing_date`, `test_volatility_percentile_no_lookahead`.
- Known unfixed gap: fundamental *values* (not the availability date) may reflect the latest restatement rather than what was actually filed at v1 — BolsAI's `/fundamentals/history` doesn't preserve filing versions. No fix possible without sourcing from CVM's raw ZIPs directly; not yet quantified.
- `status` (ATIVO/CANCELADA) is a current-day snapshot joined onto every historical row — a feature-level lookahead trap if used raw in training (constant per ticker across its whole history). `sector` is the same kind of static join, lower-risk. `manifest.LOOKAHEAD_TAINTED_COLS` also covers 6 `cross_sectional.py` columns derived from that same `sector` join (`*_zscore_sector`, `div_yield_sector_percentile`, `momentum_vs_sector_*`) — dropping raw `status`/`sector` alone is not enough.
- `days_since_fundamental` is keyed to `fundamentals_available_date` (when the market actually saw the filing), not `reference_date` (the fiscal period it describes).

**Splits, dividends & continuity**
- Unadjusted splits are repaired: `repair_unadjusted_splits()` rescales pre-event `adj_*` prices AND volumes (multiply by the split factor, not divide) for events BolsAI's `adj_*` never back-adjusted. WDCN3 is quarantined (unfixable corruption). A pre-emptive/post-hoc "is this match a real split" persistence guard was investigated 3 separate times and rejected each time — every design produced false rejections against genuinely recorded events (illiquid tickers' ordinary volatility swamps any workable threshold); don't re-attempt without new evidence of an actual misfire.
- Renames/mergers/exchanges are spliced via `apply_ticker_continuity()`, run *after* split repair (so each leg is repaired under its original name first). Rules: **rename** = same entity, splice + drop old; **merger** = scale old-leg price by exchange ratio and volume inversely (keeps dollar volume invariant), drop old fundamentals; **keep_separate** = parallel-trading acquirer, both legs stay independent; **tender** = cash-out, no splice (none currently in the map).
- Tickers that die *inside* the built panel (not spliced by continuity, not dropped for zero fundamentals) get a realized terminal payoff via `terminal_events.py`, sourced from CVM's own cancellation registry (`cvm/delistings.py`, not BolsAI): bankruptcy/liquidation reasons pay 0, everything else (voluntary cancellation, incorporation/merger) pays the ticker's last observed `adj_close`. Measured 2026-08-15: of the tickers that die inside the panel, most die *rising* (median final-60-trading-day return +2.9%) — acquisition-at-a-premium is the dominant BR terminal event, not wipeout, so this is closer to reality than the label's previous NaN. A ticker whose CVM registry status is still `ATIVO` despite no longer trading under that code is an unspliced rename, not a delisting — `terminal_events.find_rename_candidates()` reports those (e.g. `KROT3`→`COGN3`) for hand-adding to `ticker_continuity.json`; never auto-applied. This is a separate build step, run after `build_ml_dataset.py` (see Run Commands) — `forward_excess_return()` silently no-ops without it (`terminal_events=None`).
- `adj_close`/`adj_open`/`adj_high`/`adj_low` are 2-decimal precision in BolsAI; a few deep-history microcaps underflow that floor (rounds to `0.00` or pins at a tiny constant). Not fixable — never rebuild from `data/raw/br/dividends`. `compute_price_features()` masks non-positive `adj_close` before `log()` and flags `adj_close_precision_degraded`.
- Returns are total-return (dividend-adjusted), not price-only — `adj_close` already bakes in reinvestment. Known unfixed divergence: BolsAI's adjustment implies more cumulative discount than `data/raw/br/dividends` alone explains (~5pp median, likely bonus shares/subscription rights) — do **not** recompute `adj_close` from the dividends table, that would under-adjust and regress returns. `validate_vs_yfinance.py` already skips `adj_close` cross-validation as uninformative for this reason.
- yfinance collection edge cases fixed 2026-07-28: full-span refetch each `--mode update` run (so a late dividend still propagates into already-stored history); a live still-forming "today" bar is dropped before validation (`_drop_incomplete_today()`); an empty-but-no-exception response is retried instead of read as "no coverage" (`retry_on_empty`); OHLC bracket violations (not just non-positive values) are repaired (`_repair_bad_ohlc`).

**Feature engineering**
- All feature engineering lives in Stage 2, not deferred downstream.
- **Units convention (locked 2026-08-20, `docs/DATA_LAYER_CORRECTNESS_PLAN.md` §1):** every
  monetary value is a full currency unit — full BRL, full USD — never "thousands". This wasn't
  always true: BR fundamentals were stored in BRL thousands (a leftover BolsAI-API convention)
  while US was already full-dollar, silently breaking any ratio that crosses the two scales
  (`book_to_market`, `dividend_coverage_ratio` — both read 1000× too small). Enforced by
  `tests/build_dataset/test_unit_scale_invariants.py` (DATA group): per ticker, both markets,
  `vpa*shares==equity` · `lpa*shares==net_income` · `market_cap/shares==close` ·
  `book_to_market*pvp==1`, 10% band, fails on the worst offender — not a pooled statistic (a panel
  mixing correctly- and incorrectly-scaled tickers reads deceptively close to correct pooled).
- **Periodicity convention (`docs/DATA_LAYER_CORRECTNESS_PLAN.md` §3):** `_qoq` means an
  adjacent-quarter delta and is only accurately named on point-in-time inputs (e.g.
  `debt_equity_qoq`, `current_ratio_qoq`, `roa_qoq`). On a TTM input, a 1-lag diff is actually a
  single-quarter **year-over-year** change — subtracting two 4-quarter rolling sums cancels the
  three shared quarters — so `gross_margin`/`net_margin`/`roe`'s 1-lag columns are named
  `*_yoy_1q`, not `*_qoq` (renamed 2026-08-21; `roa_qoq` keeps its name — same TTM-numerator/
  point-in-time-denominator mix as `roe` but excluded from the rename by the plan). `*_trend_4q`
  (`diff(4)`, trailing year vs. prior year) is the complementary
  slow signal to `_qoq`'s fast one — keep both, they're not redundant.
- CAGR backfill is on unconditionally (`fill_missing_cagr()`): ~60% coverage from BolsAI + ~7% backfilled from earnings/revenue.
- Valuation ratios (P/E, P/B, etc.) are re-anchored to current close daily via `recompute_valuation_daily()`, not left at filing-date close. Known ceiling: mid-quarter splits skew ratios until the next filing.
- `*_zhist_5y`: causal rolling robust z-score (median/IQR, 5y window) per ticker for 11 fundamental ratios + 2 daily liquidity ratios — "how unusual for *this company*," distinct from the cross-sectional `RobustScaler` and `cross_sectional.py`'s peer-relative view. Stateless, no train/test split needed. Warm-up (<8 quarters / <252 days) is NaN.
- `iter_fit_windows()` (`manifest.py`) is the one seam between evaluation methodology and scaler fitting — a future rolling/multi-fold split format only changes this function.
- `payout_ratio`/`dividend_coverage_ratio` use `div_value_12m` (trailing-12m sum), not the single most-recent payment — the latter under/overstates for anyone paying more than annually.
- BOVA11 is the true market series for `beta_1y`/`momentum_vs_market_*` (not an equal-weighted mean of whatever tickers survived in the panel — that was a benchmark-level survivorship bias, now fixed). `momentum_vs_sector_*` uses a real sector-peer comparison, no equivalent concept applies.
- `cross_sectional.py`'s excluded-self mean derives both numerator and denominator from the same value column's own NaN-aware count (previously double-counted NaN peers into the denominator, biasing thin/young-universe dates toward zero).

**Data sources & limits**
- CVM's FCA `Codigo_Negociacao` (the ticker code itself) is 100% blank in every filing 2010-2017, populated only from 2018 on (verified live 2026-08-15, not just this repo's cache) — the crosswalk's real recoverable floor is 2018, not `cvm/http.py`'s `START_YEAR=2010`. It also reports the code *as of filing*, survivor-style (same failure mode as SEC's `company_tickers.json`): a renamed/delisted ticker stops appearing in any subsequent year, so no amount of re-scanning years recovers it (confirmed: `KROT3` appears in zero FCA years 2018-2026, only its 2019 rename `COGN3` does). `cvm/delistings.py`'s `cad_cia_aberta.csv` join has the taxonomy CVM does publish free (cancellation date/reason, current registry status) — but not a ticker-recovery path; it carries duplicate rows per CNPJ (stale pre-1978-rule registration episodes) and must keep the ATIVO/latest-cancellation row, not just deduplicate.
- BolsAI: key in `.env` via `config.load_env()` (stdlib parser, no dependency). No longer the default for anything — every `DATA_SOURCE` entry now points at a free source (see below) and `pipeline.py`'s `needs_bolsai` check only warns, doesn't hard-fail, when no key is set. Every BolsAI collector (`br/collectors.py`) stays in place and importable, reachable by flipping a `DATA_SOURCE` entry to `"bolsai"` (`.env` key required for that data type only). Paid ~€0.10/1K calls; caps: prices `limit<=5000`, fundamentals `limit<=88` (use 80).
- yfinance: free, keyless. Prices/dividends full history to 2000 (used for BR *and* US by default). BR fundamentals must NOT use yfinance (`DATA_SOURCE["fundamentals"]` default is `"cvm"`, not `"yfinance"`) — its BR financials are wrong in *level*, not just thin (~4–6 quarters): point-in-time balance-sheet items (equity, total_assets) have been observed dropping ~5x in a single quarter with no real event behind it (`docs/BOLSAI_EXIT_PLAN.md`'s "BUG-1"; regression-tested in `tests/data_collection/test_cvm_statements.py`). A second, independent reason for the same rule (found 2026-08-22, `docs/DATA_VERIFICATION_2026-08-22.md`): for BR tickers that are also US-listed ADRs, yfinance's `quarterly_balance_sheet`/`quarterly_financials` silently serve the **ADR's USD-denominated figures** under the BR `.SA` symbol, mislabeled `financialCurrency: "BRL"` — confirmed on VALE3 and PETR4 (ratio vs. CVM ≈ 5.2–5.3x, matching the FX rate), every other dual-listed name checked (ITUB4, BBDC4/3, ABEV3, SBSP3, CMIG4, BRKM5, GGBR4, CSNA3, TIMS3, UGPA3, VIVT3, PCAR3, MBRF3, SUZB3) was clean. Harmless today only because `DATA_SOURCE` keeps fundamentals off yfinance for BR — re-check any BR/ADR pair first if that switch is ever flipped. Prices/dividends are unaffected (`Ticker.history`/`info['currency']` correctly report BRL). US fundamentals are unaffected (separate `sec/` pipeline).
- CVM open data: free, keyless, `cvm/` submodules (`http.py`/`crosswalk.py`/`statements.py`/`shares.py`/`ratios.py`/`company_info.py`/`sectors.py`/`delistings.py`). Default source for BR fundamentals (a strict superset of BolsAI's own depth — both start 2010-12-31) and, since 2026-08-19, for `company_info`'s `status`/`sector` refresh too. Flows are standardized to TTM for every ticker regardless of what BolsAI itself stored per-ticker (was a near-coin-flip mix, corrupting cross-sectional comparisons — see `docs/BOLSAI_EXIT_PLAN.md` Task 0/1). Re-run cadence: `collect_statements()`/`collect_shares()`/`build_crosswalk()` all cache-and-skip every year except the current one, so re-running them (as `--mode update`'s CVM fundamentals stage now does automatically) is cheap.
- BCB series: `selic=11` (daily), `cdi=12`, `ipca=433` — **not 432** (that's the annual meta target, not the series).
- FIIs deferred: stocks only. Add if scope expands to mixed-asset.
- Company info: CVM CAD-sourced by default since 2026-08-19 (`cvm/company_info.py`'s `synthesize_company_info()`, reusing `cvm/delistings.py`'s registry download — not BolsAI's `/companies/` registry, which was itself a snapshot never refreshed for status transitions). A ticker reactivating from non-ATIVO to ATIVO is only trusted if its own price history was actually traded in the last 120 days (`_REACTIVATION_STALE_DAYS`) — CVM's `SIT` is company-level (CNPJ), not ticker-level, so a retired ticker code whose company keeps trading under a new one still reads ATIVO at CVM. Quarantined regardless of source: WDCN3 (unfixable splits), CAMB4/LLIS3 (delisted, stale fundamentals), CCTY3 (feed isn't real trading data).

**Data quality filters & NaN policy (Stage 2, automatic)**
- Filing lag filter drops fundamentals filed >180 days after quarter-end (~0.9% of rows).
- Close-price lookup replaces BolsAI's stale `close_price` with the actual close as of `fundamentals_available_date` (prevents false >50% valuation jumps).
- Sibling fill forward-fills missing `company_info` from same-CVM-company tickers.
- NaN taxonomy: **structural** (warm-up/pre-first-filing) trimmed by a global start-date rule; **informative** (CAGR undefined) flagged via `cagr_{earnings,revenue}_defined` + `n_quarters_available`, never silently filled; **error** NaN must be prefix-shaped only (`test_final_dataset.py::T_prefix_rule`), regressions warned via `nan_regressions()`; **extreme ratios** (e.g. |pl| > 400,000 near-zero-denominator distress cases) are kept intact, not clipped — the scaler's fit is robust but its transform is linear, so extremes stay extreme on purpose.

**Misc**
- Checkpoints/logs (not git-tracked): `artifacts/checkpoints/{mode}/`, `artifacts/logs/collection/`.
- Paths are always absolute via `Path(__file__).resolve().parents[N]`; run from project root.
- `pct_change(fill_method=None)` used for YoY growth to suppress FutureWarnings; `repair.py`'s volume columns are cast to `float64` before in-place rescaling (avoids a deprecated silent int64 upcast), converted back to `int64` after.

## Data on Disk

`data/raw/{br,us}/` are symmetric, market-namespaced raw trees. The one real asymmetry is
git-tracking, not layout: BR is git-tracked, US is gitignored (too large, rebuildable on demand).

- **Raw, BR (`data/raw/br/`, git-tracked):** 1,328 price files + benchmark BOVA11 (567 reach `ml_dataset.parquet`; the rest are delisted names kept for price history only, or filtered by Stage 2's quality gates), 612 fundamentals files, one parquet per ticker in `data/raw/br/{prices,fundamentals,dividends}/`. Prices current to 2026-08-14; fundamentals to 2026-06-30. Coverage isn't uniform (e.g. some tickers lack a dividends file) — treat gaps as "not yet collected," not "confirmed zero"; `has_dividends` (0/1) makes this explicit in the processed dataset.
- `data/raw/br/macro/{selic,cdi,ipca}.parquet` and `data/raw/br/company_info/company_info.parquet` are market-wide reference tables, not per-ticker. `company_info/sectors.parquet` is a small sanity-check aggregate, not a join key. `corporate_events/corporate_events.parquet` is a split/inplit audit log. Since 2026-08-19 all three are free (CVM CAD + yfinance `Ticker.splits`) and run in every mode, including `--mode update` (previously BolsAI-only and skipped there) — see `docs/BOLSAI_EXIT_PLAN.md` Task 4.
- **Processed:** `data/processed/ml_dataset.parquet` + `ml_dataset.manifest.json` (reproducibility snapshot) + `split_config.json` (train/val/test cutoffs, a filter not a copy). Each build that changes output is snapshotted to `data/processed/dataset_v{N}/`; cite `dataset_v{N}` when referencing a specific build. `data/processed/scalers/` fit train-only via the separate `scale_features.py` step. All gitignored, regenerable from `data/raw/`, except the tracked `data/processed/README.md`.
- **Raw, US (`data/raw/us/`, gitignored):** `macro/` (FRED), `prices/` (yfinance), `fundamentals/` (combined xbrl/ex27/item6 tiers), `sec/` (crosswalk, filings index, universe roster). `us_ml_dataset.parquet` has been built and validated at a scoped top-500-by-market-cap universe (Phase 6 of `docs/US_EQUITIES_EXPANSION_PLAN.md`); scaling the same code to the full ~10,432-ticker universe is Phase 6's current, in-progress step. Prices/fundamentals collection is gated by `sec/crosswalk.py`'s tier-1 crosswalk (SEC's `company_tickers.json`, current listings only) — survivor-only by construction, a decision accepted 2026-07-29 (`docs/US_COLLECTOR_FIX_PLAN.md` §4; free delisted-price recovery exists via Alpha Vantage but is rate-capped at 25 req/day, impractical at this universe's scale). `sec/universe.py` separately builds a genuinely survivorship-bias-free roster (every CIK that filed a 10-K/10-Q since 1994) purely to *measure* the gap — `build_us_dataset.py` now records the per-year coverage ratio as the manifest's `survivorship_coverage` field.

## Technology Stack

- **Python 3.10+** (`list[str]`, `dict | None`).
- **Data:** pandas, numpy, pyarrow.
- **APIs:** BolsAI REST (`httpx`, backfill), BCB SGS (requests, macro), `yfinance` (incremental).
- **Config:** stdlib `.env` parser (BolsAI key only).
- **Preprocessing:** scikit-learn (`ColumnTransformer`/`RobustScaler` in `scale_features.py`; `LedoitWolf` in Stage 3's `risk.py`), `joblib` (scaler serialization).
- **Stage 3 (portfolio):** `lightgbm` (alpha forecaster), `cvxpy` (convex optimizer), `scipy.stats` (deflated Sharpe ratio), `graphviz` (`plot_tree.py`; needs the system `dot` binary too).
- **Viz:** Plotly.
- **No test framework:** standalone `python script.py`.

## Knowledge Graph (graphify)

A persistent knowledge graph of this repo lives in `graphify-out/` (git-tracked — committed
deliberately since `feat: add graphify`, 2026-07-16 — but regenerable via the command below if
ever deleted). Built with the `graphify` skill (`/graphify`).

- **Query it first** for architecture/"how does X work"/"what calls Y" questions instead of re-reading source: `graphify query "<question>"` (BFS), `graphify path "A" "B"`, `graphify explain "<node>"`.
- **Outputs:** `graphify-out/graph.html` (interactive), `GRAPH_REPORT.md` (god nodes, communities, surprising links), `graph.json` (raw).
- **Rebuild** after significant code/doc changes: `/graphify .` (full) or `/graphify . --update` (only new/changed files).
- **Semantic extraction backend:** code is AST-extracted (no key). Docs/papers use Gemini when `GEMINI_API_KEY`/`GOOGLE_API_KEY` is set; otherwise falls back to host-agent subagents.
