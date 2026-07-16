# Graph Report - .  (2026-07-16)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 810 nodes · 1611 edges · 54 communities (40 shown, 14 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 102 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `76d9735b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- collectors.py
- test_utils.py
- build_ml_dataset.py
- features.py
- Deep RL Framework for Portfolio Management (Jiang et al. 2017)
- data_access.py
- scale_features.py
- cagr_handler.py
- test_features.py
- test_ticker_continuity.py
- compute_cross_sectional_features
- compute_advanced_features
- test_quality_filters.py
- compute_price_features
- validate.py
- http.py
- _fetch_year
- approx
- config.py
- cvm_statements.py
- test_dataset_versioning.py
- test_repair.py
- ratios.py
- test_scale_features.py
- run_all.py
- _collect
- validate_vs_yfinance.py
- collect_delisted.py
- ui.py
- inspect_all_data.py
- stats.py
- bolsai_api_validator.py
- test_log_return_basic
- build_top50_universe.py
- inspect_company_info.py
- fix_mrfg3_adj_close.py
- data_explorer.py
- load_env
- bolsai_api_macro_depth.py
- bolsai_api_price_depth.py
- bolsai_test_cagr.py
- test_load_dividends_drops_implausible_value_per_share
- test_prices_concat_dtype.py
- loaders.py
- test_rsi_no_down_days
- test_price_vs_ma_ratios
- app.py
- data_quality.py
- model_explorer.py
- training_analysis.py
- docs/specification.txt
- .github/workflows/ci.yml

## God Nodes (most connected - your core abstractions)
1. `compute_price_features()` - 34 edges
2. `Deep RL Framework for Portfolio Management (Jiang et al. 2017)` - 28 edges
3. `approx()` - 26 edges
4. `main()` - 21 edges
5. `compute_advanced_features()` - 18 edges
6. `_merge_save()` - 16 edges
7. `print_check()` - 16 edges
8. `apply_ticker_continuity()` - 15 edges
9. `compute_history_relative_features()` - 15 edges
10. `compute_features_chunked()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Filling Missing Data (flat fake price, 0 decay)` --semantically_similar_to--> `zero_fill_missing_fundamentals()`  [INFERRED] [semantically similar]
  docs/papers/deep_reinforcement_learning_framework_financial_portfolio_management.pdf.pdf → src/build_dataset/build_top50_universe.py
- `Log-Return Reward Function R` --semantically_similar_to--> `compute_price_features()`  [INFERRED] [semantically similar]
  docs/papers/deep_reinforcement_learning_framework_financial_portfolio_management.pdf.pdf → src/build_dataset/features.py
- `Price Relative Vector y_t = v_t / v_{t-1}` --semantically_similar_to--> `compute_price_features()`  [INFERRED] [semantically similar]
  docs/papers/deep_reinforcement_learning_framework_financial_portfolio_management.pdf.pdf → src/build_dataset/features.py
- `Sharpe Ratio (risk-adjusted return)` --semantically_similar_to--> `compute_macro_features()`  [INFERRED] [semantically similar]
  docs/papers/deep_reinforcement_learning_framework_financial_portfolio_management.pdf.pdf → src/build_dataset/features.py
- `Back-Test / Paper Trading (no future info)` --semantically_similar_to--> `merge_prices_and_fundamentals()`  [INFERRED] [semantically similar]
  docs/papers/deep_reinforcement_learning_framework_financial_portfolio_management.pdf.pdf → src/build_dataset/merge.py

## Import Cycles
- 1-file cycle: `src/data_collection/cvm_statements.py -> src/data_collection/cvm_statements.py`

## Hyperedges (group relationships)
- **ML Readiness Audit and Fix Flow** — docs_top50_universe_ml_readiness_audit_features_py, docs_top50_universe_ml_readiness_audit_build_top50_universe_py, docs_top50_universe_ml_readiness_audit_loaders_py, docs_top50_universe_ml_readiness_audit_test_features_py, docs_top50_universe_ml_readiness_audit_test_top50_universe_py, docs_top50_universe_ml_readiness_audit_test_loaders_py [EXTRACTED 1.00]
- **Universe Selection and Validation** — docs_top50_universe_validation_test_universe_integrity_py, docs_top50_universe_validation_ml_dataset_parquet, docs_top50_universe_ml_readiness_audit_build_top50_universe_py [INFERRED 0.85]
- **EIIE Framework Components** — docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_eiie, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_iie_minimachine, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_pvm, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_osbl, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_policy_networks [EXTRACTED 1.00]
- **RL Portfolio-Management MDP** — docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_rl_state, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_portfolio_vector_action, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_log_return_reward, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_deterministic_policy_gradient, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_price_tensor [EXTRACTED 1.00]
- **Survivorship-Safe Universe (paper <-> repo)** — docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_survival_bias, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_asset_preselection, src_build_dataset_build_top50_universe, docs_top50_universe_validation [INFERRED 0.80]

## Communities (54 total, 14 thin omitted)

### Community 0 - "collectors.py"
Cohesion: 0.05
Nodes (76): Client, main(), backfill_known_gaps.py — one-off historical backfill for confirmed BolsAI vendor, load(), mark_skip(), _path(), checkpoint.py — resume state, one JSON file per collector per mode.  Enables ide, Add `ticker` to the collector's negative cache (`cp["_skip"]`) and persist. (+68 more)

### Community 1 - "test_utils.py"
Cohesion: 0.06
Nodes (56): company_siblings(), cvm_code -> sorted tickers of the same company (PETR3/PETR4-style classes)., repair.py — rescale adj_* price history where a split/inplit was left unadjusted, Test P3: company_siblings() groups share classes of the same company by cvm_code, test_company_siblings(), check_outliers_zscore(), check_stale_prices(), main() (+48 more)

### Community 2 - "build_ml_dataset.py"
Cohesion: 0.06
Nodes (52): docs/ANOMALY_INVESTIGATION.md, docs/DATA_PIPELINE.md, Filling Missing Data (flat fake price, 0 decay), main(), build_ml_dataset.py ===================  Constrói um dataset final para Machine, compute_fundamental_features(), fill_missing_cagr(), Called on the fundamentals DataFrame BEFORE the asof merge. (+44 more)

### Community 3 - "features.py"
Cohesion: 0.07
Nodes (47): compute_features_chunked(), Three-pass, memory-bounded feature computation.      A fully unchunked pass OOM', clean_dataset(), clean.py — final pass: dedupe, inf->NaN, sort., compute_dividend_features(), compute_history_relative_features(), compute_macro_features(), Series (+39 more)

### Community 4 - "Deep RL Framework for Portfolio Management (Jiang et al. 2017)"
Cohesion: 0.07
Nodes (42): Deep RL Framework for Portfolio Management (Jiang et al. 2017), Asset Pre-Selection (top trading volume), Back-Test / Paper Trading (no future info), Baselines (UBAH, UCRP, Best Stock, OLMAR, PAMR...), Cash Bias / Quoted Currency (risk-free asset), Deterministic Policy Gradient, EIIE Topology (Ensemble of Identical Independent Evaluators), Final Accumulated Portfolio Value (fAPV) (+34 more)

### Community 5 - "data_access.py"
Cohesion: 0.14
Nodes (33): check_date_gaps(), check_duplicates(), check_lookahead(), check_nan_critical(), check_outliers_zscore(), check_return_spikes(), check_stale_prices(), check_zero_variance() (+25 more)

### Community 6 - "scale_features.py"
Cohesion: 0.10
Nodes (32): ColumnTransformer, Latest-Close Price Normalization, docs/PER_TICKER_SCALING_PLAN.md, docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md, compute_split_dates(), FitWindow, iter_fit_windows(), _manifest_fingerprint() (+24 more)

### Community 7 - "cagr_handler.py"
Cohesion: 0.11
Nodes (28): cagr_standard(), calc_annual_cagr(), _december_periods(), fill_cagr_columns(), get_cagr_statistics(), had_negative_base(), main(), DataFrame (+20 more)

### Community 8 - "test_features.py"
Cohesion: 0.11
Nodes (27): _fill_advanced_feature_columns(), DataFrame, RSI formula: 100 - (100 / (1 + avg_gain / avg_loss))., Mixed trend with both gains and losses → RSI should be valid (not NaN)., Pure downtrend → RSI should be very low (near 0)., Perfectly flat prices (zero gain, zero loss) → RSI = 50 (neutral), not NaN., Non-positive prices → NaN in log_return (no div-by-zero crashes)., Fundamental derived ratios: book_to_market, cash_ratio, etc.      earnings_yield (+19 more)

### Community 9 - "test_ticker_continuity.py"
Cohesion: 0.33
Nodes (18): apply_ticker_continuity(), continuity.py — splice renamed/merged tickers into their surviving series., Splice renamed/merged tickers into their surviving series.      Event types (dat, _fund(), _map(), _prices(), Test P2: apply_ticker_continuity() splices renamed/merged tickers correctly.  Pu, test_adj_close_reconciliation() (+10 more)

### Community 10 - "compute_cross_sectional_features"
Cohesion: 0.17
Nodes (18): compute_cross_sectional_features(), cross_sectional.py — sector/market-relative features (Pass 2 of compute_features, Sector/market-relative features: how does this stock compare to every     OTHER, approx(), _beta_fixture(), _fill_advanced_feature_columns(), DataFrame, 3 tickers, n_days of independent log_return, one sector (sector isn't     releva (+10 more)

### Community 11 - "compute_advanced_features"
Cohesion: 0.13
Nodes (20): compute_advanced_features(), Add context-aware, raw metrics (no thresholds or hardcoded rules).     Model lea, _advanced_features_fixture(), Minimal single-ticker frame with every column compute_advanced_features touches., volatility_20d_percentile at row i must not depend on rows after i (T1 regressio, n_quarters_available: cumulative count of distinct reference_date (quarterly fil, n_quarters_available counts per ticker independently (no bleed)., cagr_earnings_defined and cagr_revenue_defined equal notna() of their *_final co (+12 more)

### Community 12 - "test_quality_filters.py"
Cohesion: 0.11
Nodes (18): A ticker/quarter absent from the CVM register gets the statutory     deadline in, A filing can't precede its own quarter-end -- such a (data-error) row     must b, No filing_dates.parquet on disk at all -- every row gets the statutory     fallb, Rows before ORPHAN_PREFIX_TICKERS[ticker]['drop_before'] are removed;     everyt, Rows filed more than max_lag_days late are dropped; rows within the     threshol, Non-December quarter-ends get the 45-day ITR buffer; December     (annual/DFP fi, Drops quarantined tickers, tickers with zero fundamental rows, and     tickers w, The no-fundamentals report splits exclusions into: known non-company     (BOVA11 (+10 more)

### Community 13 - "compute_price_features"
Cohesion: 0.14
Nodes (17): Maximum Drawdown (MDD), OHLC Price Points (open/high/low/close), compute_price_features(), Volatility: std dev of log returns over window. Zero std when prices constant., Drawdown: (price - running_max) / running_max. All-time high → 0, crash → negati, HL ratio: (high - low) / close., Flags rows where adj_close is quantized to the 2-decimal vendor     precision fl, overnight_gap (prior close -> today's open) + intraday_return (today's     open (+9 more)

### Community 14 - "validate.py"
Cohesion: 0.34
Nodes (12): _common(), DataFrame, validate.py — lightweight per-collector data quality gate (runs before write)., validate_company_info(), validate_corporate_events(), validate_dividends(), validate_fundamentals(), validate_macro() (+4 more)

### Community 15 - "http.py"
Cohesion: 0.21
Nodes (14): build_crosswalk(), DataFrame, ticker -> cnpj, cvm_code, corporate_name, end_trading. Latest FCA wins per ticke, digits(), fetch_zip(), cvm/http.py — shared CVM open-data download plumbing.  Every CVM open-data sourc, One CVM yearly zip (FCA/DFP/ITR/FRE); None when the year isn't published (404)., read_csv() (+6 more)

### Community 16 - "_fetch_year"
Cohesion: 0.23
Nodes (13): collect_filing_dates(), _fetch_year(), DataFrame, cvm/filing_dates.py — CVM filing dates (real publication date per quarter).  Dow, One year's ITR/DFP register -> (cnpj, cvm_code, reference_date, received_date)., test_cvm_filing_dates.py ========================= _fetch_year() parses one CVM, Same (cnpj, cvm_code, quarter) filed twice (a restatement) -- the     market saw, _row() (+5 more)

### Community 17 - "approx"
Cohesion: 0.18
Nodes (14): Commission Rate (0.25%), Transaction Remainder Factor mu_t (transaction cost), approx(), Price features computed separately per ticker (no cross-contamination)., Approximate equality allowing for floating-point rounding., MA20/60: rolling mean of prices. First 19/59 rows should be NaN., volatility_ratio_20_60 = volatility_20d / volatility_60d -- a regime     signal, turnover_ratio = volume / shares_outstanding -- % of the float traded,     lives (+6 more)

### Community 18 - "config.py"
Cohesion: 0.24
Nodes (6): client.py — resilient HTTP helpers shared by all collectors.  Retries with expon, config.py — shared configuration for the data collection pipeline.  Loads .env (, cvm/company_info.py — CANCELADA (delisted) company_info rows from BolsAI's regis, cvm/crosswalk.py — FCA valor_mobiliario: ticker -> cnpj/cvm_code/corporate_name., cvm/shares.py — FRE capital_social -> shares-outstanding timeline per cnpj., cvm/statements.py — DFP/ITR DRE+BPA+BPP -> one wide quarterly frame per cnpj.

### Community 19 - "cvm_statements.py"
Cohesion: 0.22
Nodes (13): Append COMPANY_FIELDS rows for delisted tickers to company_info.parquet.     Sec, synthesize_company_info(), build_fundamentals(), _price_asof(), DataFrame, Series, Per-ticker fundamentals parquet for every crosswalk ticker that has a     prices, Last close at or before each reference date (NaN when none). (+5 more)

### Community 20 - "test_dataset_versioning.py"
Cohesion: 0.19
Nodes (12): nan_regressions doesn't report columns only in the new manifest (not a regressio, nan_regressions returns empty list when no column exceeds threshold., sync_dataset_version must copy scalers/ into dataset_v{N}/ too -- so an     expe, nan_regressions reports columns whose nan_pct rose by >threshold., test_content_change_creates_v2(), test_first_build_creates_v1(), test_nan_regressions_detects_increase(), test_nan_regressions_empty_when_no_increase() (+4 more)

### Community 21 - "test_repair.py"
Cohesion: 0.27
Nodes (12): _events_file(), _prices(), A recorded event whose |ln(1/factor)| is below MIN_DETECTABLE_JUMP is     filter, No corporate_events.parquet on disk (e.g. a --mode update run that never     col, A 2:1 split left unadjusted: pre-event adj_close is 2x too high relative     to, The audit log's factor direction is inconsistent (documented in     repair.py's, A jump matching the factor but years away from the recorded event date     (outs, test_repair_ignores_event_below_detectable_jump_threshold() (+4 more)

### Community 22 - "ratios.py"
Cohesion: 0.23
Nodes (10): compute_ratios(), cvm/ratios.py — BolsAI-schema fundamentals (ratios) for delisted tickers, built, Wide quarterly frame (one cnpj) + close_price/shares_outstanding columns     ->, load_statements(), DataFrame, All cached statement years -> wide frame: one row per cnpj+reference_date., Test 1b (CVM-derived fundamentals): ratio math on synthetic statements + cross-s, Statement values from CVM vs BolsAI's, same ticker+quarter. (+2 more)

### Community 23 - "test_scale_features.py"
Cohesion: 0.26
Nodes (10): Metadata must record the FitWindow that produced the artifact -- so a     params, fit_scaler must depend only on rows inside the injected FitWindow, not     on an, _synthetic_dataset(), _synthetic_dataset_full_ratio_columns(), test_fit_honors_arbitrary_window(), test_nan_preserved_not_imputed(), test_ratio_columns_scaled_others_untouched(), test_refit_on_train_split_is_reproducible() (+2 more)

### Community 24 - "run_all.py"
Cohesion: 0.38
Nodes (10): c(), main(), parse_subtests(), _print_coverage_report(), print_section(), print_summary(), Best-effort extraction of pytest -v per-test lines; empty for plain scripts., run() (+2 more)

### Community 25 - "_collect"
Cohesion: 0.25
Nodes (10): _collect(), Per-data-type source switch. Non-update modes (full_scale, prototype) are     th, BOVA11 (a benchmark ETF, not on BolsAI) always goes through yfinance     regardl, During `--mode update`, a data type missing from config.DATA_SOURCE     entirely, Regression test: full_scale/prototype are the one-time historical     backfill a, test_defaults_to_bolsai_when_data_type_unconfigured(), test_dispatches_to_bolsai_when_configured(), test_dispatches_to_yfinance_when_configured() (+2 more)

### Community 26 - "validate_vs_yfinance.py"
Cohesion: 0.25
Nodes (10): check_internal_consistency(), main(), _print_fund_rows(), validate_vs_yfinance.py ======================= Cross-validates BolsAI raw parqu, Compare a BolsAI column (BRL thousands) against a yfinance series (full BRL)., Recompute BolsAI's derived columns from its own raw columns, same row.     Curre, Returns False only if a real mismatch (>TOLERANCE_PCT%) is found., Returns False only on a real mismatch (>TOLERANCE_PCT% and <=200%).     Diffs >2 (+2 more)

### Community 27 - "collect_delisted.py"
Cohesion: 0.31
Nodes (6): candidate_tickers(), main(), collect_delisted.py — Stage 1 price backfill for delisted/never-collected ticker, Stock-like tickers with no prices parquet yet.      Suffix 3-8 pass on shape alo, Test 1a (delisted price backfill): candidate-list filter + delisting-date anchor, test_candidate_filter()

### Community 28 - "ui.py"
Cohesion: 0.25
Nodes (8): add_split_shading(), apply_split_filter(), DataFrame, Shared UI helpers for the explorer pages: sidebar filters, cached data access, a, Standard sidebar: tickers, date range, window/split filter, cache reset.      Re, Keep only rows inside the selected window/split spans., Shade train/val/test spans on a time-series figure (green/orange/red)., sidebar_filters()

### Community 29 - "inspect_all_data.py"
Cohesion: 0.39
Nodes (7): detect_date_column(), inspect_all(), inspect_file(), print_subtitle(), print_title(), Path, inspect_all_data.py ===================  Scans all folders inside:  data/raw/  a

### Community 30 - "stats.py"
Cohesion: 0.36
Nodes (7): _date_col(), main(), print_stats(), DataFrame, Path, stats.py — post-collection data audit.  Usage:     python -m src.data_collection, _stats_line()

### Community 31 - "bolsai_api_validator.py"
Cohesion: 0.67
Nodes (5): get(), print_header(), run(), show_dividends(), show_response()

### Community 32 - "test_log_return_basic"
Cohesion: 0.40
Nodes (5): Price Relative Vector y_t = v_t / v_{t-1}, Cumulative log returns over 21/63/126/252 day windows., Log returns: [100, 102, 101] → [NaN, log(1.02), log(101/102)]., test_log_return_basic(), test_return_windows()

### Community 33 - "build_top50_universe.py"
Cohesion: 0.40
Nodes (5): build_top50_universe.py, features.py, ml_dataset_top50_universe.parquet, test_features.py, test_top50_universe.py

### Community 34 - "inspect_company_info.py"
Cohesion: 0.60
Nodes (4): main(), print_header(), print_section(), inspect_company_info.py =======================  Inspeciona o parquet de company

### Community 35 - "fix_mrfg3_adj_close.py"
Cohesion: 0.50
Nodes (4): fix_one(), main(), Series, fix_mrfg3_adj_close.py — one-off repair for MRFG3's (spliced into MBRF3) chronic

### Community 36 - "data_explorer.py"
Cohesion: 0.50
Nodes (3): _load_one(), DataFrame, PAGE 1 — Data Explorer: inspect the datasets at every pipeline stage.

### Community 37 - "load_env"
Cohesion: 0.67
Nodes (3): load_env(), Path, Minimal .env loader. ponytail: 4 lines beats a python-dotenv dependency.

## Knowledge Gaps
- **12 isolated node(s):** `.github/workflows/ci.yml`, `docs/TOP50_INDEPENDENT_AUDIT.md`, `scale_features.py`, `loaders.py`, `test_features.py` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `collect_macro()` connect `collectors.py` to `build_ml_dataset.py`, `validate.py`?**
  _High betweenness centrality (0.211) - this node is a cross-community bridge._
- **Why does `docs/DATA_PIPELINE.md` connect `build_ml_dataset.py` to `collectors.py`?**
  _High betweenness centrality (0.211) - this node is a cross-community bridge._
- **Why does `compute_price_features()` connect `compute_price_features` to `test_log_return_basic`, `build_ml_dataset.py`, `features.py`, `Deep RL Framework for Portfolio Management (Jiang et al. 2017)`, `test_features.py`, `test_rsi_no_down_days`, `test_price_vs_ma_ratios`, `approx`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `compute_price_features()` (e.g. with `Log-Return Reward Function R` and `Maximum Drawdown (MDD)`) actually correct?**
  _`compute_price_features()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Deep RL Framework for Portfolio Management (Jiang et al. 2017)` (e.g. with `ml_dataset.parquet` and `build_ml_dataset.py`) actually correct?**
  _`Deep RL Framework for Portfolio Management (Jiang et al. 2017)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `.github/workflows/ci.yml`, `docs/TOP50_INDEPENDENT_AUDIT.md`, `scale_features.py` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `collectors.py` be split into smaller, more focused modules?**
  _Cohesion score 0.051055867415129644 - nodes in this community are weakly interconnected._