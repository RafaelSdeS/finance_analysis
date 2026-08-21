# Graph Report - .  (2026-08-21)

## Corpus Check
- 173 files · ~246,879 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1774 nodes · 3668 edges · 138 communities (90 shown, 48 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- portfolio: Alpha
- tests/data_collection: Test Sec Companyfacts
- data_collection/sec: Universe
- data_collection/yf: Common
- data_collection: Checkpoint
- data_collection: Validate
- build_dataset: Quality Filters
- tests/data_collection: Test Pipeline Dispatch
- portfolio: Backtest
- CLAUDE
- tests/build_dataset: Test Merge
- tests/build_dataset: Test Top Traded Quality
- tests: Run All
- tests/build_dataset: Test Features
- data_collection/cvm: Ratios
- build_dataset: Cagr Handler
- tests/data_collection: Test Sec Selected Financial Data
- portfolio: Visualize Portfolio
- portfolio: Universe
- data_collection/cvm: Statements
- portfolio: Metrics
- tests/build_dataset: Test Features (2)
- tests/data_collection: Test Sec Fds
- tests/data_collection: Test Sec Fundamentals
- tests/build_dataset: Test Features (3)
- tests/build_dataset: Test Build Us Dataset
- data_collection/cvm: Sectors
- tests/build_dataset: Test Repair
- data_collection/sec: Fds
- tests/build_dataset: Test Cross Sectional
- tests/build_dataset: Test Compute Features Chunked
- build_dataset: Terminal Events
- data_collection/sec: Selected Financial Data
- tests: Test Utils
- data_collection/sec: Tenq
- tests/build_dataset: Test Universe Integrity
- build_dataset: Scale Features
- data_collection/sec: Crosswalk
- tests/build_dataset: Test Quality Filters
- portfolio: Artifacts
- tests/build_dataset: Test Features (4)
- tests/data_collection: Test Cvm Filing Dates
- tests/build_dataset: Test Manifest
- build_dataset: Manifest
- tests/build_dataset: Test History Relative
- tests/data_collection: Test Sec Cover Page
- portfolio: Contrarian
- data_collection: Config
- tests/data_collection: Test Sec Tenq
- build_dataset: Build Us Dataset
- tests/build_dataset: Test Dataset Versioning
- tests/build_dataset: Test Scale Features
- tests/build_dataset: Test Manifest Drift
- tests/data_collection: Validate Us Vs Vendor
- data_collection: Storage
- tests/build_dataset: Test Clean
- tests/data_collection: Test Client Fail Fast
- data_collection/us: Pipeline
- tests/data_collection: Test Prices Consecutive Failures
- tests/build_dataset: Test Artifact Coherence
- tests/data_collection: Test Prices Collect Dividends
- tests/data_collection: Validate Vs Yfinance
- tests/data_collection: Test Collect Delisted
- tests/data_collection: Test Refresh Folded Dividends
- tests/data_collection: Test Sec Company Info
- tests/data_collection: Test Refresh Tail Only
- tests/build_dataset: Test Loaders
- scripts/inspect: Inspect All Data
- tests/build_dataset: Test Unit Scale Invariants
- build_dataset: Build Us Dataset (2)
- data_collection/sec: Fds (2)
- tests/api: Bolsai Api Validator
- tests/data_collection: Test Prices Negative Cache
- scripts/inspect: Inspect Company Info
- data_collection/cvm: Statements (2)
- data_collection/one_off: Fix Mrfg3 Adj Close
- tests/build_dataset: Test No Hardcoded Data Paths
- tests/data_collection: Test Prices Yf Skip Existing
- Requirements
- Misc
- tests/api: Bolsai Api Macro Depth
- tests/api: Bolsai Api Price Depth
- tests/api: Bolsai Test Cagr
- tests/data_collection: Test Prices Concat Dtype
- Requirements (2)
- data_collection/br: Stats
- Misc
- data_collection/yf: Init
- tests/build_dataset: Test Features (5)
- tests/build_dataset: Test Features (6)
- tests/build_dataset: Test Features (7)
- tests/build_dataset: Test Features (8)
- Misc
- Misc
- .github/workflows: Ci
- Misc
- Misc
- Requirements (3)
- Requirements (4)
- data_collection/br: Collect Delisted
- data_collection/br: Collect Delisted (2)
- data_collection/br: Collectors (5)
- data_collection/br: Collectors (6)
- data_collection/br: Collectors (7)
- data_collection/br: Cvm Statements
- data_collection/br: Pipeline
- data_collection/br: Stats (4)
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc
- Misc

## God Nodes (most connected - your core abstractions)
1. `print_check()` - 57 edges
2. `print_header()` - 38 edges
3. `compute_price_features()` - 33 edges
4. `approx()` - 32 edges
5. `print_section_end()` - 30 edges
6. `main()` - 27 edges
7. `build_company_fundamentals()` - 25 edges
8. `main()` - 23 edges
9. `_merge_save()` - 22 edges
10. `compute_advanced_features()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `pytest==8.3.4` --conceptually_related_to--> `"No test framework" testing philosophy (plain python scripts)`  [AMBIGUOUS]
  requirements.txt → CLAUDE.md
- `test_bare_object_response()` --calls--> `collect_macro()`  [EXTRACTED]
  tests/data_collection/test_macro_bare_object.py → src/data_collection/br/macro.py
- `test_missing_observation_and_rename()` --calls--> `collect_macro_us()`  [EXTRACTED]
  tests/data_collection/test_fred_collectors.py → src/data_collection/us/fred_collectors.py
- `test_validate_us_fundamentals_warns_on_identity_violations()` --calls--> `validate_us_fundamentals()`  [EXTRACTED]
  tests/data_collection/test_sec_fundamentals.py → src/data_collection/validate.py
- `test_validate_us_fundamentals_warns_on_impossible_values()` --calls--> `validate_us_fundamentals()`  [EXTRACTED]
  tests/data_collection/test_sec_fundamentals.py → src/data_collection/validate.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Stage 2 Data-Integrity Guarantees** — claude_no_lookahead, claude_unadjusted_split_repair, claude_ticker_continuity_splicing, claude_status_lookahead_trap [INFERRED 0.85]
- **Three-Stage Pipeline Documentation Set** — claude, readme [INFERRED 0.85]

## Communities (138 total, 48 thin omitted)

### Community 0 - "portfolio: Alpha"
Cohesion: 0.05
Nodes (91): apply_ticker_continuity(), Splice renamed/merged tickers into their surviving series.      Event types (dat, fit(), _global_trading_dates(), _label_close_dates(), predict(), _purge_embargo_mask(), DataFrame (+83 more)

### Community 1 - "tests/data_collection: Test Sec Companyfacts"
Cohesion: 0.08
Nodes (57): _annual_only(), as_first_reported(), cluster_period_ends(), compute_us_ratios(), _derive_q4(), extract_line_items(), _facts_to_frame(), _period_months() (+49 more)

### Community 2 - "data_collection/sec: Universe"
Cohesion: 0.07
Nodes (41): Response, compress_archive_cache.py — one-time migration of sec/http.py's on-disk Archive-, fetch_companyfacts(), _cache_path(), _CachedResponse, get(), _is_archive_url(), Path (+33 more)

### Community 3 - "data_collection/yf: Common"
Cohesion: 0.11
Nodes (38): main(), backfill_known_gaps.py — one-off historical backfill for confirmed BolsAI vendor, _bolsai_junction_date(), _extract_dividends(), _last_completed_trading_day(), DataFrame, Timestamp, yf/_common.py — shared fetch/repair helpers behind every yfinance collector.  Sp (+30 more)

### Community 4 - "data_collection: Checkpoint"
Cohesion: 0.08
Nodes (38): Client, collect_dividends(), collect_fundamentals(), collect_prices(), collect_macro(), br/macro.py — BCB SGS collector (SELIC/CDI/IPCA), keyless.  Split out of collect, clear_skip(), load() (+30 more)

### Community 5 - "data_collection: Validate"
Cohesion: 0.11
Nodes (34): _common(), DataFrame, validate.py — lightweight per-collector data quality gate (runs before write)., Sanity gate for SEC fundamentals (combined xbrl/ex27/item6 tiers).      collect_, validate_company_info(), validate_corporate_events(), validate_dividends(), validate_fundamentals() (+26 more)

### Community 6 - "build_dataset: Quality Filters"
Cohesion: 0.10
Nodes (37): compute_features_chunked(), main(), build_ml_dataset.py ===================  Constrói um dataset final para Machine, Three-pass, memory-bounded feature computation.      `valuation_fn`: the daily v, compute_valuation_daily_us(), _is_non_common(), main(), build_us_dataset.py — US analogue of build_ml_dataset.py.  Full rationale/measur (+29 more)

### Community 7 - "tests/data_collection: Test Pipeline Dispatch"
Cohesion: 0.10
Nodes (32): _active_tickers(), _collect(), main(), pipeline.py — orchestration + CLI for the staged data collection pipeline.  Same, Per-data-type source switch, from config.DATA_SOURCE -- governs every mode     a, --mode update's price collection is yfinance-only (free, no BolsAI dependency --, _recover_stale_company_info_tickers(), run() (+24 more)

### Community 8 - "portfolio: Backtest"
Cohesion: 0.12
Nodes (32): buy_and_hold_curve(), cdi_curve(), equal_weight_fn(), DataFrame, DatetimeIndex, Series, Timestamp, backtest.py -- quarterly-rebalanced portfolio backtest harness (proposal §2.3/Ph (+24 more)

### Community 9 - "CLAUDE"
Cohesion: 0.07
Nodes (29): DATA_SOURCE Per-Type Source Switch (BolsAI vs yfinance), BolsAI/yfinance Dividend-Adjustment Methodology Divergence, No-Lookahead Guarantee (Stage 2, merge_asof backward), "No test framework" testing philosophy (plain python scripts), Per-Ticker Own-History Z-Scores (*_zhist_5y), `status` Field Lookahead Trap, Fast/Data Test Group Split, Ticker Continuity & Splicing (rename/merger/keep_separate/tender) (+21 more)

### Community 10 - "tests/build_dataset: Test Merge"
Cohesion: 0.10
Nodes (33): merge_company_info(), merge_dividends(), merge_macro(), merge_prices_and_fundamentals(), merge_asof(by="ticker") is a single grouped asof-join, not a loop --     the OLD, make_merge_batch_fn scopes the 4 merges (prices+fundamentals,     company_info,, test_make_merge_batch_fn_matches_unbatched_merge(), approx() (+25 more)

### Community 11 - "tests/build_dataset: Test Top Traded Quality"
Cohesion: 0.11
Nodes (31): check_outliers_zscore(), check_stale_prices(), main(), Golden gate: collect failures, exit(1) if any. Inspector runs first., Flag runs of >= run_len identical closes while volume > 0., Robust (median/MAD) z-score outlier flagging.      Two regimes, chosen per-colum, validate(), build_universe() (+23 more)

### Community 12 - "tests: Run All"
Cohesion: 0.10
Nodes (20): docs/ML_AGENT_ROADMAP.md, docs/STAGE1_DATA_COLLECTION.md, docs/STAGE2_DATASET_BUILD.md, docs/STAGE3_ML_AGENT.md, docs/TODO.md, coverage==7.6.9, c(), main() (+12 more)

### Community 13 - "tests/build_dataset: Test Features"
Cohesion: 0.11
Nodes (30): compute_advanced_features(), Add context-aware, raw metrics (no thresholds or hardcoded rules).     Model lea, _advanced_features_fixture(), The only surviving earnings_yield definition (compute_advanced_features,     1/(, price_percentile_1y must not depend on rows after it -- same     no-lookahead gu, A precision-floor adj_close of exactly 0.00 must not rank as the     window's al, turnover_ratio = volume / shares_outstanding -- % of the float traded,     lives, Mixed trend with both gains and losses → RSI should be valid (not NaN). (+22 more)

### Community 14 - "data_collection/cvm: Ratios"
Cohesion: 0.12
Nodes (29): _apply_share_events(), build_fundamentals(), collect_fundamentals_cvm(), compute_ratios(), _price_asof(), DataFrame, Series, cvm/ratios.py — BolsAI-schema fundamentals, rebuilt from CVM raw statements + sh (+21 more)

### Community 15 - "build_dataset: Cagr Handler"
Cohesion: 0.11
Nodes (28): _anchor_periods(), cagr_standard(), calc_annual_cagr(), fill_cagr_columns(), get_cagr_statistics(), had_negative_base(), main(), DataFrame (+20 more)

### Community 17 - "portfolio: Visualize Portfolio"
Cohesion: 0.12
Nodes (24): Figure, load_terminal_events(), terminal_events.py — realized payoff for BR tickers that die inside the built pa, None when the (optional, separately-run) build step hasn't been run     yet -- c, forward_excess_return(), DataFrame, Series, labels.py -- forward excess-return-over-CDI label (proposal §2.2, §3.3).  label_ (+16 more)

### Community 18 - "portfolio: Universe"
Cohesion: 0.12
Nodes (26): build_top50_membership(), filter_to_top50_universe(), main(), DataFrame, build_top50_universe.py — point-in-time top-50-by-volume universe filter.  Const, Return the (ticker, period_id, start, end) membership table — one row     per ti, Restrict df to rows whose (ticker, trade_date) falls in a period the     ticker, ponytail: zero-fill fundamental columns where has_fundamentals=0. Deliberate cho (+18 more)

### Community 19 - "data_collection/cvm: Statements"
Cohesion: 0.11
Nodes (26): build_crosswalk(), DataFrame, ticker -> cnpj, cvm_code, corporate_name, end_trading. Latest FCA wins per ticke, digits(), fetch_zip(), cvm/http.py — shared CVM open-data download plumbing.  Every CVM open-data sourc, One CVM yearly zip (FCA/DFP/ITR/FRE); None when the year isn't published (404)., read_csv() (+18 more)

### Community 20 - "portfolio: Metrics"
Cohesion: 0.17
Nodes (27): active_return_report(), annualized_return(), deflated_sharpe_ratio(), excess_over_cdi_sharpe(), full_report(), information_ratio(), max_drawdown(), newey_west_tstat() (+19 more)

### Community 21 - "tests/build_dataset: Test Features (2)"
Cohesion: 0.07
Nodes (29): _fill_advanced_feature_columns(), DataFrame, A ticker with no dividends collected at all (dividends df has zero     rows for, overnight_gap (prior close -> today's open) + intraday_return (today's     open, overnight_gap shares log_return's prior-close reference, so it needs     the ide, volatility_ratio_20_60 = volatility_20d / volatility_60d -- a regime     signal, MA20/60: rolling mean of prices. First 19/59 rows should be NaN., Volatility: std dev of log returns over window. Zero std when prices constant. (+21 more)

### Community 22 - "tests/data_collection: Test Sec Fds"
Cohesion: 0.13
Nodes (27): extract_line_items(), _fill_missing_multipliers(), infer_multiplier_from_trusted_tiers(), parse_fds(), Article-5 raw line items, scaled by <MULTIPLIER>. Non-Article-5 filings     retu, Borrow this CIK's own multiplier from a sibling exhibit for any row whose     <M, Second chance for ex27 rows _fill_missing_multipliers couldn't resolve at     al, ALL EX-27 exhibits' raw tag-value dicts in this filing -- a single filing can (+19 more)

### Community 23 - "tests/data_collection: Test Sec Fundamentals"
Cohesion: 0.12
Nodes (27): build_company_fundamentals(), _derive_annual_q4(), DataFrame, Absolute-floor rejection: a company that cleared the universe gate     cannot ge, One CIK's combined fundamentals across all three built tiers, one row     per fi, Q4 = item6's annual FY total - sum(tenq's Q1+Q2+Q3), for fiscal years     where, _reject_implausible_floors(), _q4_fixture() (+19 more)

### Community 24 - "tests/build_dataset: Test Features (3)"
Cohesion: 0.07
Nodes (28): approx(), div_yield_12m must cover a true trailing calendar year (365d).      Regression t, div_yield_12m must divide each dividend by the nominal (raw "close")     price A, A split falling inside the trailing 365-day window must not distort     div_yiel, price_vs_ma20/60 = adj_close / ma_20 / ma_60 -- scale-free by     construction (, True range: max(high-low, |high-prev_close|, |low-prev_close|) / close.     Unli, volume_ratio_20d = volume / volume.rolling(20).mean() -- flags unusual     volum, amihud_illiquidity = |log_return| / traded_amount -- price impact per     unit o (+20 more)

### Community 25 - "tests/build_dataset: Test Build Us Dataset"
Cohesion: 0.09
Nodes (22): approx(), _quarterly_income(), DTB3 is quoted as an annualized %; `selic` must come out as a genuine     daily-, Same leak class BR's merge_macro guards against: selic_trend_20d must     come o, The per-file-scan gate (build_us_dataset.py's fix for §8.0 Failure 1 --     load, 176 tickers share just 59 CIKs (2026-08-01 audit) because preferreds/     ETNs/b, cik_ticker_count lets a consumer spot the shared-fundamentals cases     the univ, Steady growth_per_year%-a-year net_income/net_revenue on a quarterly     grid st (+14 more)

### Community 26 - "data_collection/cvm: Sectors"
Cohesion: 0.11
Nodes (20): cvm/company_info.py — status (ATIVO/CANCELADA) refresh + new-row synthesis for c, Refresh `status`/`sector` for every ticker CVM's CAD registry resolves, and, _recently_traded(), synthesize_company_info(), cvm/crosswalk.py — FCA valor_mobiliario: ticker -> cnpj/cvm_code/corporate_name., build_delist_events(), DataFrame, cvm/delistings.py — CVM's own cancellation registry (cad_cia_aberta.csv): delist (+12 more)

### Community 27 - "tests/build_dataset: Test Repair"
Cohesion: 0.13
Nodes (25): _events_file(), _price_series(), _prices(), The audit log's factor direction is inconsistent (documented in     repair.py's, A jump matching the factor but years away from the recorded event date     (outs, A recorded event whose |ln(1/factor)| is below MIN_DETECTABLE_JUMP is     filter, No corporate_events.parquet on disk (e.g. a --mode update run that never     col, PETR4/ITUB4/SBSP3/... shape (docs/DATA_LAYER_FOLLOWUP_FINDINGS.md): a     single (+17 more)

### Community 28 - "data_collection/sec: Fds"
Cohesion: 0.12
Nodes (22): compute_ratios(), Vendor-neutral fundamentals algebra, shared by every source.  Moved out of yf_co, Recompute BolsAI-equivalent ratios from raw fundamentals figures.     Formulas f, build_cik_history(), extract_and_compute(), fetch_filing_text(), measure_prevalence(), DataFrame (+14 more)

### Community 29 - "tests/build_dataset: Test Cross Sectional"
Cohesion: 0.16
Nodes (22): compute_cross_sectional_features(), cross_sectional.py — sector/market-relative features (Pass 2 of compute_features, Sector/market-relative features: how does this stock compare to its     sector p, approx(), _benchmark(), _beta_fixture(), _fill_advanced_feature_columns(), DataFrame (+14 more)

### Community 30 - "tests/build_dataset: Test Compute Features Chunked"
Cohesion: 0.14
Nodes (22): compute_dividend_features(), compute_price_features(), features.py — per-ticker feature engineering (Pass 1 of compute_features_chunked, Re-anchor BolsAI valuation ratios to the daily close.      The API computes pl/p, numerator / denominator, NaN where |denominator| isn't meaningfully     away fro, Compute rolling dividend yield and frequency after dividends are loaded., recompute_valuation_daily(), _rsi() (+14 more)

### Community 31 - "build_dataset: Terminal Events"
Cohesion: 0.14
Nodes (22): apply_manual_overrides(), build_terminal_events(), _dead_tickers(), find_rename_candidates(), main(), DataFrame, Append MANUAL_TERMINAL_EVENTS for tickers build_terminal_events() couldn't     r, Dead tickers whose CVM registry status is still ATIVO -- the company     survive (+14 more)

### Community 32 - "data_collection/sec: Selected Financial Data"
Cohesion: 0.16
Nodes (23): build_cik_history(), extract_years(), find_selected_financial_data_table(), _find_year_columns(), _is_caption_only_row(), _is_placeholder(), _is_spacer(), _item6_heading_text() (+15 more)

### Community 33 - "tests: Test Utils"
Cohesion: 0.15
Nodes (19): main(), _panel(), test_labels.py -- synthetic checks for forward_excess_return (proposal §2.2): ha, main(), test_reanalyze.py -- checks for reanalyze.dsr_by_era: slicing a saved run's seri, _check_window_bounds(), main(), test_run_alpha_diagnostic.py -- checks for run_alpha_diagnostic.make_alpha_weigh (+11 more)

### Community 34 - "data_collection/sec: Tenq"
Cohesion: 0.16
Nodes (22): detect_unit_multiplier(), _normalize_label(), Scale factor implied by the filing's units caption ("(in millions)" etc.),     1, build_cik_history(), _current_quarter_items(), extract_statement(), find_statement_table(), parse_period_header() (+14 more)

### Community 35 - "tests/build_dataset: Test Universe Integrity"
Cohesion: 0.14
Nodes (19): company_siblings(), cvm_code -> sorted tickers of the same company (PETR3/PETR4-style classes)., Test P3: company_siblings() groups share classes of the same company by cvm_code, test_company_siblings(), check_no_duplicate_price_series(), check_no_price_oscillation(), check_schema_contract(), check_sibling_correlation() (+11 more)

### Community 36 - "build_dataset: Scale Features"
Cohesion: 0.20
Nodes (18): ColumnTransformer, FitWindow, iter_fit_windows(), A boundary a fitted scaler should train on: rows with     fit_start < trade_date, Resolve the fit window(s) a scaler should train on, from the active     split co, build_scaler(), fit_scaler(), fit_scaler_on_train_split() (+10 more)

### Community 37 - "data_collection/sec: Crosswalk"
Cohesion: 0.13
Nodes (13): sec/company_info.py — SIC code/description per company, the free US analog of BR, build_crosswalk_tier1(), DataFrame, sec/crosswalk.py — CIK <-> ticker mapping.  Tier 1 ONLY (this pass): SEC's compa, CIK -> ticker for every currently-listed company (survivors only)., collect_fundamentals_us(), sec/fundamentals.py — combine the XBRL (2007+), EX-27 (usably 1995-2000), and It, run_fundamentals() (+5 more)

### Community 38 - "tests/build_dataset: Test Quality Filters"
Cohesion: 0.11
Nodes (18): When a (cnpj, quarter) pair exists in filing_dates.parquet, its real     receive, A ticker/quarter absent from the CVM register gets the statutory     deadline in, A filing can't precede its own quarter-end -- such a (data-error) row     must b, No filing_dates.parquet on disk at all -- every row gets the statutory     fallb, Rows before ORPHAN_PREFIX_TICKERS[ticker]['drop_before'] are removed;     everyt, Rows filed more than max_lag_days late are dropped; rows within the     threshol, Non-December quarter-ends get the 45-day ITR buffer; December     (annual/DFP fi, Drops quarantined tickers, tickers with zero fundamental rows, and     tickers w (+10 more)

### Community 39 - "portfolio: Artifacts"
Cohesion: 0.16
Nodes (16): _config_hash(), latest_run(), load_run(), Path, artifacts.py -- persist backtest runs to disk (plan Phase V.0a/b). Getting three, Short, stable hash of the config dict -- names the run directory so     identica, Persist one backtest run: `config` is every param that defines it     (top_n, ho, Inverse of save_run: {"config": {...}, **series_and_frames}. (+8 more)

### Community 40 - "tests/build_dataset: Test Features (4)"
Cohesion: 0.12
Nodes (17): compute_fundamental_features(), Series, True where the row `lookback` ROWS back is also `lookback` real     quarters bac, Called on the fundamentals DataFrame BEFORE the asof merge.      `margin_col`: w, (x - rolling_median) / rolling_IQR over a trailing window ending at     each row, _rolling_robust_zscore(), _within_calendar_gap(), Fundamental derived ratios: book_to_market, cash_ratio, etc.      earnings_yield (+9 more)

### Community 41 - "tests/data_collection: Test Cvm Filing Dates"
Cohesion: 0.20
Nodes (15): collect_filing_dates(), _fetch_year(), DataFrame, cvm/filing_dates.py — CVM filing dates (real publication date per quarter).  Dow, One year's ITR/DFP register -> (cnpj, cvm_code, reference_date, received_date)., test_cvm_filing_dates.py ========================= _fetch_year() parses one CVM, A routine quarterly re-run where nothing new has been published (every     older, Same (cnpj, cvm_code, quarter) filed twice (a restatement) -- the     market saw (+7 more)

### Community 42 - "tests/build_dataset: Test Manifest"
Cohesion: 0.12
Nodes (16): Callers that don't run the real Stage 2 filter pipeline (most tests,     ad-hoc, selic/cdi are daily-percent rates, ipca is a monthly-percent rate --     a unit, A dataset that never merged macro series must not list units for     columns tha, write_manifest(parquet_path=...) streams column_stats from disk one     column a, status is a current-day snapshot joined onto every historical row (see     merge, The 6 cross_sectional.py columns engineered from the same static,     current-da, A dataset that never joined company_info (e.g. a narrow test fixture)     must n, quality_filters.filter_tickers_with_no_fundamentals's dropped_report     must be (+8 more)

### Community 43 - "build_dataset: Manifest"
Cohesion: 0.17
Nodes (14): Scaler Fit Boundary Injection (iter_fit_windows), compute_split_dates(), _manifest_fingerprint(), nan_regressions(), manifest.py — reproducibility manifest, walk-forward split config, and immutable, Walk-forward train/val/test cutoffs, one pair of dates for the whole dataset., Manifest fields that reflect actual output content (excludes build_at/git_commit, Report columns whose NaN% rose by >threshold percentage points.      Args: (+6 more)

### Community 44 - "tests/build_dataset: Test History Relative"
Cohesion: 0.22
Nodes (15): compute_history_relative_features(), Per-ticker own-history z-scores (R1, docs/PER_TICKER_SCALING_PLAN.md).      Fund, _history_relative_fixture(), Fundamentals are forward-filled ~65 daily rows/quarter -- rolling     directly o, NaN input stays NaN (no imputation); a perfectly constant window     (IQR == 0), A rename/merger splice (continuity.py::apply_ticker_continuity) runs     before, One ticker, n_quarters distinct filings (days_per_quarter daily rows     each, f, A row's zhist value must not depend on any row after it -- the same     no-looka (+7 more)

### Community 45 - "tests/data_collection: Test Sec Cover Page"
Cohesion: 0.20
Nodes (14): extract_shares_outstanding(), Timestamp, sec/cover_page.py — shares-outstanding cover-page parser (pre-2009 tiers).  Ever, Best-effort single shares-outstanding figure off a 10-K/10-Q cover     page, plu, _to_shares(), test_sec_cover_page.py ======================= Self-check for sec/cover_page.py', test_date_first_there_were_style(), test_extracts_aapl_sentence_style() (+6 more)

### Community 46 - "portfolio: Contrarian"
Cohesion: 0.17
Nodes (14): append_trial_log(), One row per run -- config + key metrics -- appended to trials.csv     (plan V.0b, add_smoothed_earnings_yield(), equity_exposure(), DataFrame, DatetimeIndex, contrarian.py -- Layer 2 of the two-layer design: the "buy at the sound of canno, CAPE-style fix for the trailing-earnings lag (2026-07-25 finding):     point-in- (+6 more)

### Community 47 - "data_collection: Config"
Cohesion: 0.13
Nodes (9): load_env(), Path, config.py — shared configuration for the data collection pipeline.  Loads .env (, Minimal .env loader. ponytail: 4 lines beats a python-dotenv dependency., fix_camb3_phantom_row.py — one-off repair for CAMB3's single phantom non-trading, test_fred_collectors.py ======================== Self-check for fred_collectors., test_missing_observation_and_rename(), test_macro_bare_object.py ========================= Verifies collect_macro() han (+1 more)

### Community 48 - "tests/data_collection: Test Sec Tenq"
Cohesion: 0.29
Nodes (14): DataFrame, test_sec_tenq.py ================= Self-check for sec/tenq.py's pure logic (no n, Reproduces AAPL's real 2004-02-10 10-Q (CIK 320193, accession     0001104659-04-, _real_income_table(), test_build_cik_history_as_first_reported_dedup(), test_build_cik_history_attaches_shares_outstanding_from_cover_page(), test_build_cik_history_end_to_end(), test_build_cik_history_skips_a_filing_whose_html_crashes_pd_read_html() (+6 more)

### Community 49 - "build_dataset: Build Us Dataset"
Cohesion: 0.14
Nodes (11): make_merge_batch_fn(), merge_company_info_us(), merge_macro_us(), _MergeBatcher, Series, Map a numeric SIC code to its coarse division name; NaN/unmatched -> NaN., Join ticker -> sic/sic_description/sector. No cvm_code sibling-fill,     no CVM, US analogue of merge.merge_macro(). Emits columns literally named     `selic`/`i (+3 more)

### Community 50 - "tests/build_dataset: Test Dataset Versioning"
Cohesion: 0.19
Nodes (12): nan_regressions doesn't report columns only in the new manifest (not a regressio, nan_regressions returns empty list when no column exceeds threshold., sync_dataset_version must copy scalers/ into dataset_v{N}/ too -- so an     expe, nan_regressions reports columns whose nan_pct rose by >threshold., test_content_change_creates_v2(), test_first_build_creates_v1(), test_nan_regressions_detects_increase(), test_nan_regressions_empty_when_no_increase() (+4 more)

### Community 51 - "tests/build_dataset: Test Scale Features"
Cohesion: 0.22
Nodes (12): Metadata must record the FitWindow that produced the artifact -- so a     params, status passes through the scaler untouched (not a ratio column) but     must nev, fit_scaler must depend only on rows inside the injected FitWindow, not     on an, _synthetic_dataset(), _synthetic_dataset_full_ratio_columns(), test_fit_honors_arbitrary_window(), test_nan_preserved_not_imputed(), test_ratio_columns_scaled_others_untouched() (+4 more)

### Community 52 - "tests/build_dataset: Test Manifest Drift"
Cohesion: 0.26
Nodes (12): check_market_drift(), check_us_empty_columns(), check_us_survivorship_coverage(), compare_manifests(), _latest_snapshot_manifest(), main(), Path, Market-agnostic on purpose: called for both BR and US. Gracefully     no-ops (pa (+4 more)

### Community 53 - "tests/data_collection: Validate Us Vs Vendor"
Cohesion: 0.23
Nodes (12): _av_daily(), check_internal_consistency(), check_tier_seams(), main(), _print_fund_rows(), validate_us_vs_vendor.py ========================= Cross-validates US raw parque, Compare our `end`-keyed column (raw USD) against a yfinance series (raw USD),, Recompute derived columns by hand from raw columns, same row. Currency/scale-imm (+4 more)

### Community 54 - "data_collection: Storage"
Cohesion: 0.24
Nodes (9): python-bcb==0.3.3, _chunk_dates(), _demo(), is_complete(), storage.py — shared parquet append/dedup/validate/write + date-window chunking., True if `path` exists AND already carries every column in `required_cols`., Yield (start, end) ISO windows of <= `years` each, to stay under API caps., test_chunk_dates_leap_year.py ============================== _chunk_dates used r (+1 more)

### Community 55 - "tests/build_dataset: Test Clean"
Cohesion: 0.24
Nodes (10): clean_dataset(), clean.py — final pass: dedupe, inf->NaN, sort., A row that's byte-for-byte identical to another (every column, not     just the, Same (ticker, trade_date) but a genuinely different value elsewhere is     NOT a, Literal inf/-inf (division-by-zero in a ratio or growth rate) must     become Na, Output must be sorted (ticker, trade_date) ascending with a clean     0..n-1 ind, test_exact_duplicate_row_removed(), test_inf_replaced_with_nan_other_columns_untouched() (+2 more)

### Community 56 - "tests/data_collection: Test Client Fail Fast"
Cohesion: 0.23
Nodes (8): get_json(), GET with backoff retry. Returns parsed JSON or raises after max retries., _FakeClient, _FakeResponse, test_client_fail_fast.py ========================= client.py's retry loops used, Always returns a retryable 500 -- exercises the retry-exhaustion path., test_no_sleep_when_out_of_retries(), test_sleeps_and_retries_when_another_attempt_remains()

### Community 57 - "data_collection/us: Pipeline"
Cohesion: 0.26
Nodes (11): _all_tickers(), main(), pipeline.py — orchestration + CLI for the US-equities collection pipeline (Phase, run(), run_company_info(), run_dividends(), run_macro(), run_prices() (+3 more)

### Community 58 - "tests/data_collection: Test Prices Consecutive Failures"
Cohesion: 0.30
Nodes (11): _fake_price_df(), test_prices_consecutive_failures.py ===================================== Self-c, fetch_results: dict ticker -> return value for _fetch_and_shape_prices     (None, _run(), test_long_failure_streak_aborts_loudly(), test_occasional_failures_interspersed_with_successes_do_not_abort(), test_resume_mode_still_aborts_past_its_own_much_higher_threshold(), test_resume_mode_tolerates_a_streak_past_the_normal_threshold() (+3 more)

### Community 59 - "tests/build_dataset: Test Artifact Coherence"
Cohesion: 0.35
Nodes (10): check_snapshot_matches(), check_split_manifest_agree(), check_terminal_events_not_stale(), _latest_snapshot(), main(), _manifest_path(), Path, Highest data/processed/dataset_v{N}/ next to output_path, or None.      BR-only (+2 more)

### Community 60 - "tests/data_collection: Test Prices Collect Dividends"
Cohesion: 0.24
Nodes (6): _FakeTickerNoActivity, _FakeTickerWithDividend, test_prices_collect_dividends.py ================================== Self-check f, test_collect_dividends_false_returns_none_and_never_touches_dividend_dir(), test_collect_dividends_true_no_activity_writes_no_dividend_file(), test_collect_dividends_true_writes_dividend_file_and_reports_changed()

### Community 61 - "tests/data_collection: Validate Vs Yfinance"
Cohesion: 0.25
Nodes (10): check_internal_consistency(), main(), _print_fund_rows(), validate_vs_yfinance.py ======================= Cross-validates BolsAI raw parqu, Compare a BolsAI column against a yfinance series -- both full BRL units     sin, Recompute BolsAI's derived columns from its own raw columns, same row.     Curre, Returns False only if a real mismatch (>TOLERANCE_PCT%) is found., Returns False only on a real mismatch (>TOLERANCE_PCT% and <=200%).     Diffs >2 (+2 more)

### Community 62 - "tests/data_collection: Test Collect Delisted"
Cohesion: 0.22
Nodes (6): candidate_tickers(), main(), Test 1a (delisted price backfill): candidate-list filter + delisting-date anchor, collect_prices() loads/saves its own checkpoint dict per call and isn't     safe, test_candidate_filter(), test_main_collects_all_tickers_in_one_call()

### Community 63 - "tests/data_collection: Test Refresh Folded Dividends"
Cohesion: 0.33
Nodes (8): Shared BR/US orchestration for the folded prices+dividends pass (see     module, _refresh_prices_and_dividends(), test_refresh_folded_dividends.py ================================== Self-check f, test_default_stages_single_pass_when_nothing_changed(), test_default_stages_two_pass_when_something_changed(), test_full_flag_skips_two_pass_split(), test_only_dividends_falls_back_to_standalone_collector(), test_only_prices_folds_no_dividends()

### Community 64 - "tests/data_collection: Test Sec Company Info"
Cohesion: 0.31
Nodes (8): collect_company_info(), DataFrame, ticker -> CIK (tier-1 crosswalk) -> submissions.json -> sic/sicDescription., _fake_submissions(), test_sec_company_info.py ========================= Self-check for sec/company_in, test_collect_company_info_extracts_sic_and_description(), test_resume_skips_already_resolved_tickers_and_checkpoints_progress(), test_unresolvable_ticker_and_failed_fetch_are_skipped_not_fatal()

### Community 65 - "tests/data_collection: Test Refresh Tail Only"
Cohesion: 0.36
Nodes (7): _prices_fetch_start(), Where to start the prices fetch from.      `tail_only` (default off): once the o, test_refresh_tail_only.py ========================== Self-check for the fast-ref, test_tail_only_false_still_starts_at_earliest_yfinance_row(), test_tail_only_true_starts_after_last_stored_row(), test_thin_file_ignores_tail_only_and_falls_back_to_floor(), _write_prices_fixture()

### Community 66 - "tests/build_dataset: Test Loaders"
Cohesion: 0.22
Nodes (8): A real BRL per-share dividend is at most low tens even in extreme     cases. Reg, tickers=None (default) must keep loading everything -- only a US-scale     calle, US fundamentals files carry per-line-item filing dates and XBRL/item6/     EX-27, optimize_dtypes=False (default) must leave BR's float64 precision     untouched, test_load_dividends_drops_implausible_value_per_share(), test_load_fundamentals_drops_provenance_columns(), test_load_fundamentals_optimize_dtypes_downcasts_numeric_keeps_cik(), test_load_prices_tickers_filter_loads_only_matching_files()

### Community 67 - "scripts/inspect: Inspect All Data"
Cohesion: 0.39
Nodes (7): detect_date_column(), inspect_all(), inspect_file(), print_subtitle(), print_title(), Path, inspect_all_data.py ===================  Scans all folders inside:  data/raw/br/

### Community 68 - "tests/build_dataset: Test Unit Scale Invariants"
Cohesion: 0.36
Nodes (7): check_identity(), check_margin_scale(), main(), §1 guard from docs/DATA_LAYER_CORRECTNESS_PLAN.md: money must be in one scale (f, Per-ticker median ratio; fails if the WORST ticker sits outside the band., All *_margin columns must read as the same convention, checked pooled     agains, run()

### Community 69 - "build_dataset: Build Us Dataset (2)"
Cohesion: 0.33
Nodes (6): build_universe_gate(), build_universe_gate_from_files(), _qualifying_tickers(), stats: per-ticker DataFrame indexed by ticker with n/med_close/med_dv     column, Quality/scale gate over the full US universe (plan §D1) -- a lifetime     per-ti, Same gate as build_universe_gate, computed from a per-file, column-     projecte

### Community 70 - "data_collection/sec: Fds (2)"
Cohesion: 0.40
Nodes (6): _fds_period_end(), _parse_fds_date(), Timestamp, EX-27's <FISCAL-YEAR-END> is "DEC-31-1994"-style, not ISO -- pandas parses it, This exhibit's own period end. <FISCAL-YEAR-END> only for a full-year     (PERIO, test_fiscal_year_end_never_used_as_a_quarterly_exhibits_own_period_end()

### Community 71 - "tests/api: Bolsai Api Validator"
Cohesion: 0.67
Nodes (5): get(), print_header(), run(), show_dividends(), show_response()

### Community 72 - "tests/data_collection: Test Prices Negative Cache"
Cohesion: 0.53
Nodes (5): _fake_checkpoint_store(), test_prices_negative_cache.py ============================== Self-check for coll, test_empty_runs_resets_when_coverage_confirmed_but_nothing_new(), test_empty_runs_resets_when_ticker_saves_real_rows(), test_negative_cache_skips_after_threshold_and_reprobes()

### Community 73 - "scripts/inspect: Inspect Company Info"
Cohesion: 0.60
Nodes (4): main(), print_header(), print_section(), inspect_company_info.py =======================  Inspeciona o parquet de company

### Community 74 - "data_collection/cvm: Statements (2)"
Cohesion: 0.40
Nodes (5): load_statements(), DataFrame, All cached statement years -> wide frame: one row per cnpj+reference_date., Statement values from CVM vs BolsAI's, same ticker+quarter., test_cross_source_vs_bolsai()

### Community 75 - "data_collection/one_off: Fix Mrfg3 Adj Close"
Cohesion: 0.50
Nodes (4): fix_one(), main(), Series, fix_mrfg3_adj_close.py — one-off repair for MRFG3's (spliced into MBRF3) chronic

### Community 76 - "tests/build_dataset: Test No Hardcoded Data Paths"
Cohesion: 0.67
Nodes (3): find_violations(), test_no_hardcoded_data_paths.py ================================ Guards the path, test_no_hardcoded_data_paths()

### Community 78 - "Requirements"
Cohesion: 0.67
Nodes (3): rich==15.0.0, stable-baselines3==2.9.0, tqdm==4.67.3

### Community 79 - "Misc"
Cohesion: 0.67
Nodes (3): DataFrame, Series, _shares_outstanding()

## Ambiguous Edges - Review These
- `"No test framework" testing philosophy (plain python scripts)` → `pytest==8.3.4`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to
- `python-bcb==0.3.3` → `collectors.py`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to

## Knowledge Gaps
- **24 isolated node(s):** `.github/workflows/ci.yml`, `docs/TODO.md`, `docs/STAGE1_DATA_COLLECTION.md`, `docs/STAGE2_DATASET_BUILD.md`, `docs/STAGE3_ML_AGENT.md` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **48 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `"No test framework" testing philosophy (plain python scripts)` and `pytest==8.3.4`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `python-bcb==0.3.3` and `collectors.py`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `compute_price_features()` connect `tests/build_dataset: Test Compute Features Chunked` to `build_dataset: Quality Filters`, `tests/build_dataset: Test Features`, `tests/build_dataset: Test Features (2)`, `tests/build_dataset: Test Features (3)`, `tests/build_dataset: Test Features (6)`, `tests/build_dataset: Test Features (8)`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `compute_advanced_features()` connect `tests/build_dataset: Test Features` to `tests/build_dataset: Test Features (2)`, `build_dataset: Quality Filters`, `tests/build_dataset: Test Compute Features Chunked`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `compute_ratios()` connect `data_collection/sec: Fds` to `data_collection/sec: Selected Financial Data`, `tests/data_collection: Test Sec Companyfacts`, `data_collection/sec: Tenq`, `data_collection/yf: Common`, `data_collection/sec: Crosswalk`, `tests/data_collection: Test Sec Fundamentals`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `print_check()` (e.g. with `main()` and `test_company_siblings()`) actually correct?**
  _`print_check()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `print_header()` (e.g. with `main()` and `main()`) actually correct?**
  _`print_header()` has 11 INFERRED edges - model-reasoned connections that need verification._