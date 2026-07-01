# TODO: Portfolio Allocation Agent

## Phase 1: Data Validation (3 tickers: PETR4, VALE3, WEGE3)

Goal: Validate API data accuracy and establish production-ready quarterly update pipeline.

### Done

- [x] Uncomment and test `fill_cagr_columns()` in `src/build_dataset/build_ml_dataset.py`
- [x] Collect BOVA11 (iShares Bovespa ETF / IBOV benchmark) in pipeline
- [x] **Quarterly update pipeline ready:** `--mode update` uses free yfinance for incremental refreshes (99% cost savings vs BolsAI)
  - yfinance-derived fundamentals verified to within 1–15% tolerance on key ratios (P/E, P/B, ROE)
  - Prices match BolsAI >99% (0.1–1% variance, typical market data variance)
  - Split-boundary handling + ratio computation verified on live data (PETR4, VALE3, WEGE3)
- [x] Source-agnostic architecture: `DATA_SOURCE` dict allows fallback to BolsAI per data type if yfinance breaks

### Blockers

- [ ] Run full pipeline (Stages 1–2) on 463+ tickers (data already collected; need to validate ml_dataset.parquet)
- [ ] Validate metrics: sample comparison of BolsAI prices + fundamentals against Yahoo Finance, B3 official
  - Note: minor diffs expected; flag only if >1% variance on key metrics (P/E, P/B, ROE, CAGR)
- [ ] Verify macro data (SELIC, CDI, IPCA) matches official BCB sources (spot check)
- [ ] Document any known discrepancies with links to reference sources

### Done

- [x] Macro data collection (SELIC, CDI, IPCA) via BCB SGS API
- [x] Stock prices and fundamentals collection via BolsAI API (PETR4, VALE3, WEGE3)
- [x] Dividend collection via BolsAI API (20-year history)
- [x] Generic API endpoint tester (`tests/api/bolsai_api_validator.py --path / --param`)
- [x] ML dataset builder: `merge_asof(prices, fundamentals)` + company info (no lookahead bias)
- [x] CAGR calculation and backfill logic (`cagr_handler.py`)
- [x] Interactive visualization: nominal price vs inflation-adjusted vs SELIC (`financial_view.py`)
- [x] Data validation & exploration notebook (`src/visualizations/exploration.ipynb`): coverage heatmap, completeness, liquidity, sector distribution, inflation-adjusted returns, fundamentals insights, dividend analysis

---

## Phase 2: Scale to Full Market

Once Phase 1 validation passes:

- [ ] Expand ticker coverage to all B3 equities
- [ ] Re-validate on full dataset
- [ ] Create `.env.example` documenting BolsAI API key requirement

- [ ] Technical indicators: momentum, moving averages, volatility, drawdowns
- [ ] Macro context features: interest rate regime, market regime (bull/bear), VIX-like volatility proxy
---

## Phase 3: RL Agent (ml_agent branch)

**Prerequisite:** Stage 2 (build_dataset branch) must be complete with:
- All 5 phases of dataset building (load, merge, **feature engineering**, clean, validate)
- Feature engineering: technical indicators (RSI, MA20/60, volatility), fundamentals (P/E, P/B, ROE), macro (SELIC, IPCA, real return)
- Output: `data/processed/ml_dataset.parquet` (feature-complete, no lookahead bias, ≥252 rows per ticker)

### Phase 3a: Foundation & Environment Design

**Goal:** Verify dataset quality, build PortfolioEnv, define state/action spaces, validate with random policy.

#### Dataset Verification (DO FIRST)
- [ ] **Run verification script** (see CLAUDE.md "Dataset Verification Plan"):
  - [ ] Inspect date range: earliest & latest date, total span in years
  - [ ] Check ticker coverage: unique tickers, rows per ticker, exclude short-history tickers
  - [ ] Verify feature completeness: all expected columns present
  - [ ] Measure NaN rates: overall <5%, per-column <30%, critical columns 0%
  - [ ] Check for duplicates: 0 duplicate (ticker, date) pairs
  - [ ] Verify temporal alignment: no lookahead bias (fundamental_date ≤ price_date)
  - [ ] Analyze distributions: returns ~0, no extreme outliers
  - [ ] Sector coverage: ≥3 sectors represented
  - [ ] Output: Use script `tests/agent/verify_dataset_for_training.py`
- [ ] **Determine train/val/test split dates** (based on actual data coverage):
  - [ ] Calculate 60/20/20 split by date (not rows)
  - [ ] Verify each split has ≥252 rows (minimum 1 year)
  - [ ] Document dates in config.py
  - [ ] Example: If data spans 2017-2026 (9 years), use: train=2017-06-30, val=2022-12-31, test=2026-06-30

#### Environment Setup
- [x] Update `requirements.txt`: add `torch==2.3.0`, `stable-baselines3==2.4.0`, `gymnasium==0.29.0`, `scikit-learn>=1.5.0`
- [x] Create `src/agent/` directory structure
- [x] **feature_engineering.py:** Functions to compute missing returns and prepare dataset
  - [x] `compute_returns()`: Log returns from prices (grouped by ticker)
  - [x] `prepare_training_dataset()`: Load dataset, compute returns, save to ml_dataset_training.parquet
  - [x] Handles NaN gracefully (first row per ticker is NaN, expected behavior)
- [x] **__init__.py:** Package exports (compute_returns, prepare_training_dataset)
- [x] **config.py:** Immutable `AgentConfig` dataclass with verified dates from dataset inspection
  - [x] Train/val/test date splits: 2000-01-03 / 2015-11-25 / 2021-03-13 / 2026-06-30
  - [x] Hyperparams: learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, entropy_coef=0.01
  - [x] Feature list: 23 state features (6 price + 14 fundamental + 3 macro)
  - [x] Paths: data_dir, model_dir, log_dir, dataset_path
  - [x] Validation on load (checks dataset exists, dates are valid)
  - [x] Method: `log_summary()` for debugging
- [x] **data_pipeline.py:** Dense tensors + train-only scaler
  - [x] Pivot long-format dataset → [6565 dates × 279 tickers × 23 features] numpy tensor
  - [x] Activity mask [dates × tickers] (46.2% cells active — universe grows 30 → 245 tickers)
  - [x] Universe: all tickers ≥252 rows (279; drops 9 stubs) — **full universe with masking, no survivorship bias**
  - [x] StandardScaler fit on train dates only → `data/models/feature_scaler.pkl`
  - [x] Output: `data/processed/agent_tensors.npz`
- [x] **env.py:** `PortfolioEnv(gymnasium.Env)` with time-varying universe
  - [x] State: normalized features (NaN→0 after z-score) + activity mask = 6,696-dim obs
  - [x] Action: 279-dim logits → **masked softmax** (inactive tickers get exactly 0 weight)
  - [x] Reward: daily log portfolio return; delisted-next-day positions carry flat
  - [x] Splits via `date_range="train"/"val"/"test"`
- [x] **test_env_basic.py:** All invariants pass
  - [x] Obs shapes across all 3 splits; weights sum to 1; inactive weight always 0
  - [x] 16,451 steps/sec (fast enough for 1M timesteps); deterministic under fixed seed

### Phase 3b: Training Infrastructure — DONE (smoke-tested; full run pending)

- [x] **trainer.py:** SB3 PPO ("MlpPolicy", CUDA), all hyperparams from `AgentConfig`
  - [x] `ValSharpeCallback`: deterministic val rollout every `eval_freq` rollouts
  - [x] JSONL logging → `data/logs/agent_training_YYYYMMDD-HHMMSS.jsonl`
  - [x] Checkpoints per eval + `agent_best.zip` (best val Sharpe) + `agent_final.zip`
  - [x] Early stopping: val Sharpe degrades `early_stopping_patience` (3) consecutive evals
  - [x] CLI: `--timesteps`, `--learning-rate`, `--device`
  - [x] Smoke run (12K steps, GPU) verified end-to-end
- [ ] **Full training run:** `python -m src.agent.trainer` (1M timesteps, ~hours on RTX 4060) — run when ready
- [ ] Hyperparameter tuning if val Sharpe plateaus: learning_rate first, then n_steps, batch_size

### Phase 3c: Evaluation & Backtesting — DONE

- [x] **metrics.py:** Sharpe, Sortino, max drawdown, cumulative/annualized return, win rate (unit-tested vs hand-computed values)
- [x] **evaluate.py:** All strategies rolled through the same env (portfolio math in one place)
  - [x] Baselines: equal-weight, market-cap (ffill, no lookahead), 1/vol (trailing 60d)
  - [x] Comparison table + `data/backtest/metrics.json` + `results.parquet`
  - [x] Plotly plots: cumulative value (log scale), drawdown, return distribution, top-10 weights timeline
- [x] **Found & fixed real data bug:** returns were computed from raw `close` → stock-split artifacts (±4.7 log returns, spurious 84%/yr equal-weight). Now from `adj_close` with corrupt-observation cleaning (|log r| > 1.0 → NaN, 332 rows / 0.04%). Post-fix equal-weight: 14.8%/yr, Sharpe 0.71 — realistic.

### Phase 3d: Deployment & Inference — DONE

- [x] **infer.py:** `predict_weights(date)` reuses env obs pipeline (state identical to training)
  - [x] Equal-weight fallback on load failure or invalid weights (never crashes)
- [x] **run_allocation.py:** CLI `--date --format csv|json --model`; sector-enriched output → `data/allocations/allocation_YYYY-MM-DD.{csv,json}`
- [x] **test_inference_output.py:** sum=1, sorted, no dupes, fallback path, pre-range date error

### Phase 3 Quality Gates (Before Merge)

- [x] **Correctness:** no lookahead (train-only scaler, backward merges, trailing vol); weights sum to 1 (1e-9); no NaN in values/weights; inactive tickers always 0 weight
- [ ] **Performance (needs full training run):**
  - [ ] Test Sharpe ≥ equal-weight baseline (0.71) — smoke agent merely matches it
  - [ ] Test max drawdown < 50%
  - [ ] Training curves converge; val Sharpe tracks training (not overfitting)
- [x] **Code:** type hints, docstrings, logger (no print in src modules), config-driven paths, files ≤300 lines
- [x] **Documentation:** `src/agent/README.md`, CLAUDE.md architecture guide updated with as-built design
  - [ ] Architecture diagram (state flow, module dependencies)

---

## Future Scope (Out of Phase 1–3)

- [ ] **FIIs (Real Estate Investment Trusts):** Deferred pending Phase 3 scope decision. If RL agent expands to mixed-asset allocation, add separate collectors for FII prices + distributions. API endpoints exist; collector skeleton pattern is proven. **Why deferred**: FIIs have different fundamentals (NAV/P-VP vs earnings) and require separate dataset build logic.

---

See `specification.txt` for complete system design (input features, model output, objective function, constraints, expected behavior).
