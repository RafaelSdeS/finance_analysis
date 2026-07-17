# Graph Report - .  (2026-07-16)

## Corpus Check
- 31 files · ~120,941 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1162 nodes · 2601 edges · 71 communities (58 shown, 13 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 118 edges (avg confidence: 0.74)
- Token cost: 127,814 input · 0 output

## Community Hubs (Navigation)
- Dataset Build Orchestration
- Company Sibling Grouping
- Backtest Metrics Suite
- RL Agent Baselines
- BolsAI Client & Checkpointing
- Price Panel & Asset Indexing
- Price Feature Engineering
- Data Quality Checks
- Differentiable Transaction Cost Model
- Dividend Adjustment & yfinance Backfill
- CAGR Calculation & Backfill
- Backtest Engine (Weights & Costs)
- RL Agent Experiment Config
- Portfolio Vector Memory
- Feature Scaler Fitting
- EIIE Paper Concepts
- Advanced Contextual Features
- Data Collection Pipeline Orchestration
- Repo Design Rationale (CLAUDE.md)
- Cross-Sectional Features
- Price Feature Edge Cases
- EIIE Agent Design Decisions
- Ticker Continuity & Splicing
- Fundamental Ratio Computation
- RL Experiment Orchestrator
- Filing Lag Quality Filters
- Top-50 Universe Construction
- Delisted Company Info Synthesis
- Collector Data Validation
- Chunked Feature Computation
- Per-Ticker History Z-Scores
- CVM Crosswalk & HTTP
- CVM Filing Dates Collection
- Project Docs & Roadmap
- Dataset Versioning & NaN Regression
- Test Runner (run_all.py)
- EIIE CNN Network
- Unadjusted Split Repair
- Metrics Test Suite
- Data Collection Config & .env
- Scaler Fit Window Tests
- yfinance Cross-Validation
- Dataset Cleaning Tests
- Delisted Ticker Backfill
- Explorer UI Helpers
- Raw Data Inspector
- Per-Ticker Feature Engineering
- Idempotent Parquet Merge/Save
- Dataset Stats CLI
- Transaction Cost Concepts
- BolsAI API Validator Tool
- Macro Feature Computation
- Top-50 Universe & Features
- Company Info Inspector
- MRFG3 One-Off Repair
- yfinance Ratio Recomputation
- Data Explorer Page
- BolsAI Macro Depth Test
- BolsAI Price Depth Test
- BolsAI CAGR Test Tool
- Dividend Plausibility Guard
- Price Concat Dtype Test
- Build Dataset Loaders
- Explorer Dependency Grouping
- Log Return Test
- Explorer App Entry Point
- Data Quality Explorer Page
- Model Explorer Page
- Training Analysis Page
- CI Workflow

## God Nodes (most connected - your core abstractions)
1. `PricePanel` - 45 edges
2. `EIIE Agent Plan (docs/EIIE_AGENT_PLAN.md)` - 40 edges
3. `compute_price_features()` - 34 edges
4. `BacktestResult` - 34 edges
5. `Deep RL Framework for Portfolio Management (Jiang et al. 2017)` - 30 edges
6. `PortfolioVectorMemory` - 27 edges
7. `approx()` - 26 edges
8. `run_experiment()` - 26 edges
9. `run_baseline()` - 25 edges
10. `GlobalAssetIndex` - 25 edges

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
- 1-file cycle: `src/rl_agent/__init__.py -> src/rl_agent/__init__.py`

## Hyperedges (group relationships)
- **EIIE Agent Data-to-Backtest Pipeline** — src_rl_agent_data, src_rl_agent_pvm, src_rl_agent_environment, src_rl_agent_networks, src_rl_agent_train, src_rl_agent_experiment [EXTRACTED 1.00]
- **Stage 2 Data-Integrity Guarantees** — claude_no_lookahead, claude_unadjusted_split_repair, claude_ticker_continuity_splicing, claude_status_lookahead_trap [INFERRED 0.85]
- **Three-Stage Pipeline Documentation Set** — claude, readme, docs_eiie_agent_plan [INFERRED 0.85]
- **ML Readiness Audit and Fix Flow** — docs_top50_universe_ml_readiness_audit_features_py, docs_top50_universe_ml_readiness_audit_build_top50_universe_py, docs_top50_universe_ml_readiness_audit_loaders_py, docs_top50_universe_ml_readiness_audit_test_features_py, docs_top50_universe_ml_readiness_audit_test_top50_universe_py, docs_top50_universe_ml_readiness_audit_test_loaders_py [EXTRACTED 1.00]
- **Universe Selection and Validation** — docs_top50_universe_validation_test_universe_integrity_py, docs_top50_universe_validation_ml_dataset_parquet, docs_top50_universe_ml_readiness_audit_build_top50_universe_py [INFERRED 0.85]
- **EIIE Framework Components** — docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_eiie, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_iie_minimachine, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_pvm, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_osbl, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_policy_networks [EXTRACTED 1.00]
- **RL Portfolio-Management MDP** — docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_rl_state, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_portfolio_vector_action, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_log_return_reward, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_deterministic_policy_gradient, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_price_tensor [EXTRACTED 1.00]
- **Survivorship-Safe Universe (paper <-> repo)** — docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_survival_bias, docs_papers_deep_reinforcement_learning_framework_financial_portfolio_management_pdf_asset_preselection, src_build_dataset_build_top50_universe, docs_top50_universe_validation [INFERRED 0.80]

## Communities (71 total, 13 thin omitted)

### Community 0 - "Dataset Build Orchestration"
Cohesion: 0.05
Nodes (65): Ticker Continuity & Splicing (rename/merger/keep_separate/tender), docs/DATA_PIPELINE.md, Back-Test / Paper Trading (no future info), Filling Missing Data (flat fake price, 0 decay), main(), build_ml_dataset.py ===================  Constrói um dataset final para Machine, continuity.py — splice renamed/merged tickers into their surviving series., compute_fundamental_features() (+57 more)

### Community 1 - "Company Sibling Grouping"
Cohesion: 0.06
Nodes (55): company_siblings(), cvm_code -> sorted tickers of the same company (PETR3/PETR4-style classes)., Test P3: company_siblings() groups share classes of the same company by cvm_code, test_company_siblings(), check_outliers_zscore(), check_stale_prices(), main(), Golden gate: collect failures, exit(1) if any. Inspector runs first. (+47 more)

### Community 2 - "Backtest Metrics Suite"
Cohesion: 0.10
Nodes (48): Figure, BacktestResult, annualized_return(), annualized_turnover(), block_bootstrap_ci(), cagr(), calmar_ratio(), final_apv() (+40 more)

### Community 3 - "RL Agent Baselines"
Cohesion: 0.10
Nodes (36): _active_gidx(), _best_stock_gidx(), _bova11_result(), constant_cash_weight_fn(), make_best_stock_weight_fn(), make_random_portfolio_weight_fn(), make_random_rebalancing_weight_fn(), ndarray (+28 more)

### Community 4 - "BolsAI Client & Checkpointing"
Cohesion: 0.10
Nodes (34): Client, python-bcb==0.3.3, load(), mark_skip(), _path(), checkpoint.py — resume state, one JSON file per collector per mode.  Enables ide, Add `ticker` to the collector's negative cache (`cp["_skip"]`) and persist., save() (+26 more)

### Community 5 - "Price Panel & Asset Indexing"
Cohesion: 0.10
Nodes (29): DataFrame, DatetimeIndex, DataConfig, _build_slot_calendar(), GlobalAssetIndex, _load_bova11(), load_price_panel(), ndarray (+21 more)

### Community 6 - "Price Feature Engineering"
Cohesion: 0.08
Nodes (36): Maximum Drawdown (MDD), OHLC Price Points (open/high/low/close), compute_price_features(), _fill_advanced_feature_columns(), DataFrame, Volatility: std dev of log returns over window. Zero std when prices constant., RSI formula: 100 - (100 / (1 + avg_gain / avg_loss))., Mixed trend with both gains and losses → RSI should be valid (not NaN). (+28 more)

### Community 7 - "Data Quality Checks"
Cohesion: 0.14
Nodes (33): check_date_gaps(), check_duplicates(), check_lookahead(), check_nan_critical(), check_outliers_zscore(), check_return_spikes(), check_stale_prices(), check_zero_variance() (+25 more)

### Community 8 - "Differentiable Transaction Cost Model"
Cohesion: 0.15
Nodes (32): Optimizer, drift_weights_torch(), Tensor, Batched, differentiable version of drift_weights (eq. 7), for     train.py's los, Batched, differentiable approximation of solve_mu (eq. 14) for the     training, solve_mu_torch(), agent_forward(), _batch_tensors() (+24 more)

### Community 9 - "Dividend Adjustment & yfinance Backfill"
Cohesion: 0.14
Nodes (27): BolsAI/yfinance Dividend-Adjustment Methodology Divergence, yfinance==0.2.66, main(), backfill_known_gaps.py — one-off historical backfill for confirmed BolsAI vendor, backfill_price_gap(), collect_dividends_yf(), collect_fundamentals_yf(), collect_prices_yf() (+19 more)

### Community 10 - "CAGR Calculation & Backfill"
Cohesion: 0.11
Nodes (28): cagr_standard(), calc_annual_cagr(), _december_periods(), fill_cagr_columns(), get_cagr_statistics(), had_negative_base(), main(), DataFrame (+20 more)

### Community 11 - "Backtest Engine (Weights & Costs)"
Cohesion: 0.12
Nodes (29): drift_weights(), ndarray, Simulate one policy (agent or baseline) over [start_idx, end_idx].      weight_f, w'_t (eq. 7): weights after one period's price movement, before any     rebalanc, Transaction remainder factor mu_t (eq. 14), converged fixed-point     iteration, run_backtest(), solve_mu(), _bisect_root() (+21 more)

### Community 12 - "RL Agent Experiment Config"
Cohesion: 0.14
Nodes (19): Path, ExperimentConfig, main(), EIIE portfolio-management agent (Jiang, Xu & Liang 2017), adapted to daily B3 da, _build_model(), sanity.py — automated invariant checks run before any real training (docs/EIIE_A, Run every gate. Called by experiment.py before pretraining starts;     a failing, run_sanity_checks() (+11 more)

### Community 13 - "Portfolio Vector Memory"
Cohesion: 0.13
Nodes (20): PortfolioVectorMemory, Tensor, pvm.py — Portfolio-Vector Memory (paper Sec. 5.2), extended for a dynamic top-50, Slot-space -> (n_global+1)-wide global-space row (w: [B, n_slots+1],     column, n_global: real global-space width (cash + N_union tickers, e.g.         172). Th, Full previous-period global weight vector(s), shape [..., n_global]         (dum, Previous-period weight in slot space + cash, for network input.         slot_gid, Store the network's output into PVM[row_idx] in global space.         w: [B, n_s (+12 more)

### Community 14 - "Feature Scaler Fitting"
Cohesion: 0.14
Nodes (23): ColumnTransformer, docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md, joblib==1.5.3, scikit-learn==1.9.0, FitWindow, iter_fit_windows(), A boundary a fitted scaler should train on: rows with     fit_start < trade_date, Resolve the fit window(s) a scaler should train on, from the active     split co (+15 more)

### Community 15 - "EIIE Paper Concepts"
Cohesion: 0.17
Nodes (23): Deep RL Framework for Portfolio Management (Jiang et al. 2017), Baselines (UBAH, UCRP, Best Stock, OLMAR, PAMR...), Deterministic Policy Gradient, EIIE Topology (Ensemble of Identical Independent Evaluators), Final Accumulated Portfolio Value (fAPV), IIE / Mini-Machine (per-asset evaluator), Latest-Close Price Normalization, Log-Return Reward Function R (+15 more)

### Community 16 - "Advanced Contextual Features"
Cohesion: 0.14
Nodes (22): compute_advanced_features(), Add context-aware, raw metrics (no thresholds or hardcoded rules).     Model lea, _advanced_features_fixture(), Rows with no filing (reference_date NaT) get has_fundamentals=0., Minimal single-ticker frame with every column compute_advanced_features touches., volatility_20d_percentile at row i must not depend on rows after i (T1 regressio, n_quarters_available: cumulative count of distinct reference_date (quarterly fil, n_quarters_available counts per ticker independently (no bleed). (+14 more)

### Community 17 - "Data Collection Pipeline Orchestration"
Cohesion: 0.14
Nodes (19): DATA_SOURCE Per-Type Source Switch (BolsAI vs yfinance), _active_tickers(), _collect(), main(), pipeline.py — orchestration + CLI for the staged data collection pipeline.  Same, Per-data-type source switch. Non-update modes (full_scale, prototype) are     th, Return tickers that matched BolsAI company info (exist on the platform)., Return only tickers with status='ATIVO' (exclude delisted/suspended). (+11 more)

### Community 18 - "Repo Design Rationale (CLAUDE.md)"
Cohesion: 0.11
Nodes (19): "No test framework" testing philosophy (plain python scripts), Per-Ticker Own-History Z-Scores (*_zhist_5y), Scaler Fit Boundary Injection (iter_fit_windows), `status` Field Lookahead Trap, Fast/Data Test Group Split, Unadjusted Splits Repair (53 corporate events), configs/eiie_baseline.json, data/processed/ml_dataset.parquet (+11 more)

### Community 19 - "Cross-Sectional Features"
Cohesion: 0.17
Nodes (18): compute_cross_sectional_features(), cross_sectional.py — sector/market-relative features (Pass 2 of compute_features, Sector/market-relative features: how does this stock compare to every     OTHER, approx(), _beta_fixture(), _fill_advanced_feature_columns(), DataFrame, 3 tickers, n_days of independent log_return, one sector (sector isn't     releva (+10 more)

### Community 20 - "Price Feature Edge Cases"
Cohesion: 0.10
Nodes (20): approx(), Pure uptrend (zero down-days in the window) → RSI = 100, not NaN.      loss=0 ma, Perfectly flat prices (zero gain, zero loss) → RSI = 50 (neutral), not NaN., Price features computed separately per ticker (no cross-contamination)., Approximate equality allowing for floating-point rounding., Fundamental derived ratios: book_to_market, cash_ratio, etc.      earnings_yield, div_yield_12m must cover a true trailing calendar year (365d).      Regression t, overnight_gap (prior close -> today's open) + intraday_return (today's     open (+12 more)

### Community 21 - "EIIE Agent Design Decisions"
Cohesion: 0.13
Nodes (19): No-Lookahead Guarantee (Stage 2, merge_asof backward), data/raw/macro/cdi.parquet, EIIE Agent Plan (docs/EIIE_AGENT_PLAN.md), 7-Baseline Evaluation Suite, CDI-Accruing Cash (paper deviation #1), EIIE (Ensemble of Identical Independent Evaluators), EIIE CNN Network Architecture (paper Fig. 2), Global Asset Indexing (172-wide, cash=index 0) (+11 more)

### Community 22 - "Ticker Continuity & Splicing"
Cohesion: 0.36
Nodes (18): docs/ANOMALY_INVESTIGATION.md, apply_ticker_continuity(), Splice renamed/merged tickers into their surviving series.      Event types (dat, _fund(), _map(), _prices(), Test P2: apply_ticker_continuity() splices renamed/merged tickers correctly.  Pu, test_adj_close_reconciliation() (+10 more)

### Community 23 - "Fundamental Ratio Computation"
Cohesion: 0.18
Nodes (17): build_fundamentals(), compute_ratios(), _price_asof(), DataFrame, Series, cvm/ratios.py — BolsAI-schema fundamentals (ratios) for delisted tickers, built, Per-ticker fundamentals parquet for every crosswalk ticker that has a     prices, Wide quarterly frame (one cnpj) + close_price/shares_outstanding columns     -> (+9 more)

### Community 24 - "RL Experiment Orchestrator"
Cohesion: 0.25
Nodes (17): compute_window_split(), _dataset_fingerprint(), _git_commit(), experiment.py — CLI orchestrator (docs/EIIE_AGENT_PLAN.md "Implementation phases, Recompute train/val/test cutoffs WITHIN this experiment's date window     (2011-, eval_split: 'val' for hyperparameter-selection runs (pretrain on     train, back, run_experiment(), main() (+9 more)

### Community 25 - "Filing Lag Quality Filters"
Cohesion: 0.11
Nodes (18): A ticker/quarter absent from the CVM register gets the statutory     deadline in, A filing can't precede its own quarter-end -- such a (data-error) row     must b, No filing_dates.parquet on disk at all -- every row gets the statutory     fallb, Rows before ORPHAN_PREFIX_TICKERS[ticker]['drop_before'] are removed;     everyt, Rows filed more than max_lag_days late are dropped; rows within the     threshol, Non-December quarter-ends get the 45-day ITR buffer; December     (annual/DFP fi, Drops quarantined tickers, tickers with zero fundamental rows, and     tickers w, The no-fundamentals report splits exclusions into: known non-company     (BOVA11 (+10 more)

### Community 26 - "Top-50 Universe Construction"
Cohesion: 0.21
Nodes (16): Asset Pre-Selection (top trading volume), docs/TOP50_INDEPENDENT_AUDIT.md, docs/TOP50_ML_READINESS_AUDIT.md, build_top50_membership(), filter_to_top50_universe(), main(), DataFrame, build_top50_universe.py — point-in-time top-50-by-volume universe filter.  Const (+8 more)

### Community 27 - "Delisted Company Info Synthesis"
Cohesion: 0.20
Nodes (10): cvm/company_info.py — CANCELADA (delisted) company_info rows from BolsAI's regis, Append COMPANY_FIELDS rows for delisted tickers to company_info.parquet.     Sec, synthesize_company_info(), cvm/crosswalk.py — FCA valor_mobiliario: ticker -> cnpj/cvm_code/corporate_name., cvm/shares.py — FRE capital_social -> shares-outstanding timeline per cnpj., collect_statements(), main(), cvm/statements.py — DFP/ITR DRE+BPA+BPP -> one wide quarterly frame per cnpj. (+2 more)

### Community 28 - "Collector Data Validation"
Cohesion: 0.34
Nodes (12): _common(), DataFrame, validate.py — lightweight per-collector data quality gate (runs before write)., validate_company_info(), validate_corporate_events(), validate_dividends(), validate_fundamentals(), validate_macro() (+4 more)

### Community 29 - "Chunked Feature Computation"
Cohesion: 0.19
Nodes (14): compute_features_chunked(), Three-pass, memory-bounded feature computation.      A fully unchunked pass OOM', clean.py — final pass: dedupe, inf->NaN, sort., compute_dividend_features(), Re-anchor BolsAI valuation ratios to the daily close.      The API computes pl/p, Compute rolling dividend yield and frequency after dividends are loaded., recompute_valuation_daily(), _chunked_pipeline_fixture() (+6 more)

### Community 30 - "Per-Ticker History Z-Scores"
Cohesion: 0.22
Nodes (15): compute_history_relative_features(), Per-ticker own-history z-scores (R1, docs/PER_TICKER_SCALING_PLAN.md).      Fund, _history_relative_fixture(), Fundamentals are forward-filled ~65 daily rows/quarter -- rolling     directly o, NaN input stays NaN (no imputation); a perfectly constant window     (IQR == 0), A rename/merger splice (continuity.py::apply_ticker_continuity) runs     before, One ticker, n_quarters distinct filings (days_per_quarter daily rows     each, f, A row's zhist value must not depend on any row after it -- the same     no-looka (+7 more)

### Community 31 - "CVM Crosswalk & HTTP"
Cohesion: 0.21
Nodes (14): build_crosswalk(), DataFrame, ticker -> cnpj, cvm_code, corporate_name, end_trading. Latest FCA wins per ticke, digits(), fetch_zip(), cvm/http.py — shared CVM open-data download plumbing.  Every CVM open-data sourc, One CVM yearly zip (FCA/DFP/ITR/FRE); None when the year isn't published (404)., read_csv() (+6 more)

### Community 32 - "CVM Filing Dates Collection"
Cohesion: 0.23
Nodes (13): collect_filing_dates(), _fetch_year(), DataFrame, cvm/filing_dates.py — CVM filing dates (real publication date per quarter).  Dow, One year's ITR/DFP register -> (cnpj, cvm_code, reference_date, received_date)., test_cvm_filing_dates.py ========================= _fetch_year() parses one CVM, Same (cnpj, cvm_code, quarter) filed twice (a restatement) -- the     market saw, _row() (+5 more)

### Community 33 - "Project Docs & Roadmap"
Cohesion: 0.14
Nodes (6): docs/ML_AGENT_ROADMAP.md, docs/specification.txt, docs/STAGE1_DATA_COLLECTION.md, docs/STAGE2_DATASET_BUILD.md, docs/STAGE3_ML_AGENT.md, docs/TODO.md

### Community 34 - "Dataset Versioning & NaN Regression"
Cohesion: 0.19
Nodes (12): nan_regressions doesn't report columns only in the new manifest (not a regressio, nan_regressions returns empty list when no column exceeds threshold., sync_dataset_version must copy scalers/ into dataset_v{N}/ too -- so an     expe, nan_regressions reports columns whose nan_pct rose by >threshold., test_content_change_creates_v2(), test_first_build_creates_v1(), test_nan_regressions_detects_increase(), test_nan_regressions_empty_when_no_increase() (+4 more)

### Community 35 - "Test Runner (run_all.py)"
Cohesion: 0.33
Nodes (11): coverage==7.6.9, c(), main(), parse_subtests(), _print_coverage_report(), print_section(), print_summary(), Best-effort extraction of pytest -v per-test lines; empty for plain scripts. (+3 more)

### Community 36 - "EIIE CNN Network"
Cohesion: 0.24
Nodes (9): EIIECNN, Tensor, networks.py — the EIIE CNN encoder (paper Fig. 2), docs/EIIE_AGENT_PLAN.md "EIIE, Fig. 2's fully convolutional EIIE: a chain of kernel-height-1     convolutions (, main(), Test: networks.py's EIIECNN -- output shape, simplex constraint, and slot maskin, test_forward_shape_and_simplex(), test_gradient_flows() (+1 more)

### Community 37 - "Unadjusted Split Repair"
Cohesion: 0.27
Nodes (12): _events_file(), _prices(), A recorded event whose |ln(1/factor)| is below MIN_DETECTABLE_JUMP is     filter, No corporate_events.parquet on disk (e.g. a --mode update run that never     col, A 2:1 split left unadjusted: pre-event adj_close is 2x too high relative     to, The audit log's factor direction is inconsistent (documented in     repair.py's, A jump matching the factor but years away from the recorded event date     (outs, test_repair_ignores_event_below_detectable_jump_threshold() (+4 more)

### Community 38 - "Metrics Test Suite"
Cohesion: 0.36
Nodes (12): main(), Test: metrics.py's performance metrics and block-bootstrap CIs, checked against, _result(), test_bootstrap_ci(), test_max_drawdown(), test_return_metrics(), test_sharpe_sortino(), test_summarize_smoke() (+4 more)

### Community 39 - "Data Collection Config & .env"
Cohesion: 0.23
Nodes (10): load_env(), Path, config.py — shared configuration for the data collection pipeline.  Loads .env (, Minimal .env loader. ponytail: 4 lines beats a python-dotenv dependency., CostConfig, EvalConfig, ExperimentMeta, ModelConfig (+2 more)

### Community 40 - "Scaler Fit Window Tests"
Cohesion: 0.26
Nodes (10): Metadata must record the FitWindow that produced the artifact -- so a     params, fit_scaler must depend only on rows inside the injected FitWindow, not     on an, _synthetic_dataset(), _synthetic_dataset_full_ratio_columns(), test_fit_honors_arbitrary_window(), test_nan_preserved_not_imputed(), test_ratio_columns_scaled_others_untouched(), test_refit_on_train_split_is_reproducible() (+2 more)

### Community 41 - "yfinance Cross-Validation"
Cohesion: 0.25
Nodes (10): check_internal_consistency(), main(), _print_fund_rows(), validate_vs_yfinance.py ======================= Cross-validates BolsAI raw parqu, Compare a BolsAI column (BRL thousands) against a yfinance series (full BRL)., Recompute BolsAI's derived columns from its own raw columns, same row.     Curre, Returns False only if a real mismatch (>TOLERANCE_PCT%) is found., Returns False only on a real mismatch (>TOLERANCE_PCT% and <=200%).     Diffs >2 (+2 more)

### Community 42 - "Dataset Cleaning Tests"
Cohesion: 0.29
Nodes (9): clean_dataset(), A row that's byte-for-byte identical to another (every column, not     just the, Same (ticker, trade_date) but a genuinely different value elsewhere is     NOT a, Literal inf/-inf (division-by-zero in a ratio or growth rate) must     become Na, Output must be sorted (ticker, trade_date) ascending with a clean     0..n-1 ind, test_exact_duplicate_row_removed(), test_inf_replaced_with_nan_other_columns_untouched(), test_near_duplicate_survives() (+1 more)

### Community 43 - "Delisted Ticker Backfill"
Cohesion: 0.31
Nodes (6): candidate_tickers(), main(), collect_delisted.py — Stage 1 price backfill for delisted/never-collected ticker, Stock-like tickers with no prices parquet yet.      Suffix 3-8 pass on shape alo, Test 1a (delisted price backfill): candidate-list filter + delisting-date anchor, test_candidate_filter()

### Community 44 - "Explorer UI Helpers"
Cohesion: 0.25
Nodes (8): add_split_shading(), apply_split_filter(), DataFrame, Shared UI helpers for the explorer pages: sidebar filters, cached data access, a, Standard sidebar: tickers, date range, window/split filter, cache reset.      Re, Keep only rows inside the selected window/split spans., Shade train/val/test spans on a time-series figure (green/orange/red)., sidebar_filters()

### Community 45 - "Raw Data Inspector"
Cohesion: 0.39
Nodes (7): detect_date_column(), inspect_all(), inspect_file(), print_subtitle(), print_title(), Path, inspect_all_data.py ===================  Scans all folders inside:  data/raw/  a

### Community 46 - "Per-Ticker Feature Engineering"
Cohesion: 0.25
Nodes (7): Series, features.py — per-ticker feature engineering (Pass 1 of compute_features_chunked, numerator / denominator, NaN where |denominator| isn't meaningfully     away fro, (x - rolling_median) / rolling_IQR over a trailing window ending at     each row, _rolling_robust_zscore(), _rsi(), _safe_ratio()

### Community 47 - "Idempotent Parquet Merge/Save"
Cohesion: 0.43
Nodes (7): _merge_save(), Append to existing parquet, dedup on date_col, validate, write. Idempotent., test_merge_save_new_rows_only.py ================================= _merge_save m, _row(), test_bad_row_in_new_batch_still_blocks(), test_bracket_check_tolerates_float_noise(), test_old_bad_row_does_not_block_new_good_row()

### Community 48 - "Dataset Stats CLI"
Cohesion: 0.36
Nodes (7): _date_col(), main(), print_stats(), DataFrame, Path, stats.py — post-collection data audit.  Usage:     python -m src.data_collection, _stats_line()

### Community 49 - "Transaction Cost Concepts"
Cohesion: 0.47
Nodes (6): Commission Rate (0.25%), Transaction Remainder Factor mu_t (transaction cost), turnover_ratio = volume / shares_outstanding -- % of the float traded,     lives, amihud_illiquidity = |log_return| / (volume * adj_close) -- price     impact per, test_amihud_illiquidity(), test_turnover_ratio()

### Community 50 - "BolsAI API Validator Tool"
Cohesion: 0.67
Nodes (5): get(), print_header(), run(), show_dividends(), show_response()

### Community 51 - "Macro Feature Computation"
Cohesion: 0.50
Nodes (5): Cash Bias / Quoted Currency (risk-free asset), Sharpe Ratio (risk-adjusted return), compute_macro_features(), Requires log_return (from compute_price_features) and selic/ipca already merged., get_selic()

### Community 52 - "Top-50 Universe & Features"
Cohesion: 0.40
Nodes (5): build_top50_universe.py, features.py, ml_dataset_top50_universe.parquet, test_features.py, test_top50_universe.py

### Community 53 - "Company Info Inspector"
Cohesion: 0.60
Nodes (4): main(), print_header(), print_section(), inspect_company_info.py =======================  Inspeciona o parquet de company

### Community 54 - "MRFG3 One-Off Repair"
Cohesion: 0.50
Nodes (4): fix_one(), main(), Series, fix_mrfg3_adj_close.py — one-off repair for MRFG3's (spliced into MBRF3) chronic

### Community 55 - "yfinance Ratio Recomputation"
Cohesion: 0.50
Nodes (4): _compute_ratios(), Recompute BolsAI-equivalent ratios from yfinance raw figures.     Formulas for m, test_ratios_no_inf.py ====================== Verifies _compute_ratios() never re, test_zero_denominator_yields_nan()

### Community 56 - "Data Explorer Page"
Cohesion: 0.50
Nodes (3): _load_one(), DataFrame, PAGE 1 — Data Explorer: inspect the datasets at every pipeline stage.

## Ambiguous Edges - Review These
- `collectors.py` → `python-bcb==0.3.3`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to
- `EIIE Agent Plan (docs/EIIE_AGENT_PLAN.md)` → `gymnasium==1.3.0`  [AMBIGUOUS]
  docs/EIIE_AGENT_PLAN.md · relation: conceptually_related_to
- `EIIE Agent Plan (docs/EIIE_AGENT_PLAN.md)` → `stable-baselines3==2.9.0`  [AMBIGUOUS]
  docs/EIIE_AGENT_PLAN.md · relation: conceptually_related_to
- `"No test framework" testing philosophy (plain python scripts)` → `pytest==8.3.4`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to

## Knowledge Gaps
- **28 isolated node(s):** `.github/workflows/ci.yml`, `docs/TOP50_INDEPENDENT_AUDIT.md`, `scale_features.py`, `loaders.py`, `test_features.py` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `collectors.py` and `python-bcb==0.3.3`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `EIIE Agent Plan (docs/EIIE_AGENT_PLAN.md)` and `gymnasium==1.3.0`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `EIIE Agent Plan (docs/EIIE_AGENT_PLAN.md)` and `stable-baselines3==2.9.0`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `"No test framework" testing philosophy (plain python scripts)` and `pytest==8.3.4`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `compute_price_features()` connect `Price Feature Engineering` to `Dataset Build Orchestration`, `Log Return Test`, `Per-Ticker Feature Engineering`, `EIIE Paper Concepts`, `Advanced Contextual Features`, `Transaction Cost Concepts`, `Price Feature Edge Cases`, `Top-50 Universe Construction`, `Chunked Feature Computation`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `EIIE Agent Plan (docs/EIIE_AGENT_PLAN.md)` connect `EIIE Agent Design Decisions` to `Dataset Build Orchestration`, `Project Docs & Roadmap`, `Backtest Metrics Suite`, `RL Agent Baselines`, `EIIE CNN Network`, `Test Runner (run_all.py)`, `Data Collection Config & .env`, `Differentiable Transaction Cost Model`, `RL Agent Experiment Config`, `Portfolio Vector Memory`, `EIIE Paper Concepts`, `Repo Design Rationale (CLAUDE.md)`, `RL Experiment Orchestrator`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `Deep RL Framework for Portfolio Management (Jiang et al. 2017)` connect `EIIE Paper Concepts` to `Dataset Build Orchestration`, `Project Docs & Roadmap`, `Price Feature Engineering`, `Transaction Cost Concepts`, `Macro Feature Computation`, `EIIE Agent Design Decisions`, `Top-50 Universe Construction`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._