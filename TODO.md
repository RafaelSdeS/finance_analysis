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

**Goal:** Build PortfolioEnv, define state/action spaces, validate with random policy.

- [ ] Update `requirements.txt`: add `torch==2.3.0`, `stable-baselines3==2.4.0`, `gymnasium==0.29.0`, `scikit-learn>=1.5.0`
- [ ] Create `src/agent/` directory structure (with `__init__.py`)
- [ ] **config.py:** Immutable `AgentConfig` dataclass
  - [ ] Hyperparams: learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, entropy_coef=0.01
  - [ ] Feature list: list of feature column names from `ml_dataset.parquet`
  - [ ] Train/val/test date splits (60/20/20 by date, not rows)
  - [ ] Paths: data_dir, model_dir, log_dir
  - [ ] Action: n_tickers (inferred from data)
- [ ] **env.py:** `PortfolioEnv(gymnasium.Env)` class
  - [ ] `__init__`: Load `ml_dataset.parquet`, initialize scaler, store date ranges
  - [ ] State space: concatenated normalized features for all tickers [n_tickers × feature_dim]
  - [ ] Action space: `Box(0, 1, shape=(n_tickers,))` continuous weights (no sum constraint in space; enforce in step)
  - [ ] `reset(date_range)`: Return normalized state for first date in range
  - [ ] `step(action)`: Apply softmax weights, compute daily log return as reward, advance date
  - [ ] `_normalize_state()`: Z-score normalize using train scaler (fit-once-reuse principle)
  - [ ] Handle edge cases: insufficient data, missing values, date bounds
- [ ] **test_env_basic.py:** Validation script
  - [ ] Instantiate env, call reset(), verify state shape [n_tickers × feature_dim]
  - [ ] Step 100 times with random actions, verify portfolio value > 0
  - [ ] Verify weights sum to 1 after softmax in step()
  - [ ] Log: initial state, sample actions, sample returns

### Phase 3b: Training Infrastructure

**Goal:** Implement PPO training loop, logging, checkpointing, early stopping.

- [ ] **trainer.py:** Main training orchestration
  - [ ] Data pipeline: Load ml_dataset, create train/val/test date-splits
  - [ ] Scaler: Fit StandardScaler on train features only, save to `data/models/feature_scaler.pkl`
  - [ ] PPO initialization: use `stable-baselines3.PPO("MlpPolicy", env, learning_rate=3e-4, verbose=1)`
  - [ ] Training loop: call `agent.learn(total_timesteps=1_000_000)`
  - [ ] Logging (JSONL): Per 100 episodes, log {episode, train_return, train_sharpe, val_sharpe, val_max_dd}
  - [ ] Checkpointing: Save model every 100 episodes to `data/models/agent_checkpoint_ep{N}.pt`
  - [ ] Early stopping: Monitor val Sharpe over last 10 eval cycles, stop if degrades 3 cycles in a row
  - [ ] Config integration: All hyperparams from `AgentConfig`, zero hardcoding
- [ ] **Hyperparameter tuning placeholder**
  - [ ] Document which params to tune first: learning_rate, n_steps, batch_size
  - [ ] Add CLI flags: `--learning-rate 3e-4`, `--gamma 0.95`, `--episodes 1000`
- [ ] **Logging setup**
  - [ ] File: `data/logs/agent_training_YYYYMMDD-HHMMSS.jsonl` (structured, one JSON object per line)
  - [ ] Console: INFO level (episode progress, not step-level noise)
  - [ ] Log per-100-episodes: episode, average return, sharpe, max drawdown, learning rate

### Phase 3c: Evaluation & Backtesting

**Goal:** Backtest trained agent on test set, compute metrics, generate plots, compare baselines.

- [ ] **evaluate.py:** Backtesting harness
  - [ ] Load trained model + scaler from `data/models/`
  - [ ] Create fresh PortfolioEnv on test date range (deterministic, no exploration)
  - [ ] Step through test set, collect: portfolio_values, weights, returns, dates
  - [ ] Save results to `data/backtest/results.parquet` (columns: date, portfolio_value, weights, returns)
  - [ ] Compute metrics (on test set only):
    - [ ] Cumulative return: (V_final - V_0) / V_0
    - [ ] Annualized Sharpe: mean(returns) / std(returns) × sqrt(252)
    - [ ] Max drawdown: min( (peak - V_t) / peak )
    - [ ] Sortino ratio: mean(returns) / std(downside_returns) × sqrt(252)
    - [ ] Win rate: % of days with positive return
    - [ ] Sector exposure: distribution of weights by sector
  - [ ] Implement baseline policies (test set only):
    - [ ] Equal-weight: w_i = 1/n (daily rebalance)
    - [ ] Market-cap weight: w_i ∝ market_cap (from company_info)
    - [ ] 1/Vol weight: w_i ∝ 1/volatility (from dataset)
  - [ ] Compute same metrics for baselines
  - [ ] Output comparison table: agent vs 3 baselines on all metrics
- [ ] **Visualization module** (in evaluate.py or separate `viz.py`)
  - [ ] Cumulative value: agent vs 3 baselines (Plotly line chart, save to HTML)
  - [ ] Drawdown over time: agent only (Plotly area chart)
  - [ ] Sector allocation heatmap: date × sector (weights over time, Plotly)
  - [ ] Return distribution: histogram of daily returns (Plotly)
  - [ ] Weights timeline: stacked bar chart (top 10 holdings over time)
  - [ ] Save all plots to `data/backtest/plots/`
  - [ ] Summary: metrics.json with {sharpe, max_dd, sortino, win_rate, ...}

### Phase 3d: Deployment & Inference

**Goal:** Build inference script for daily portfolio allocation, integrate with production workflow.

- [ ] **infer.py:** Inference module
  - [ ] Function: `predict_weights(agent, latest_features_df) -> np.ndarray`
  - [ ] Input: DataFrame [n_tickers × feature_dim] (latest date from ml_dataset)
  - [ ] Load trained model + feature scaler
  - [ ] Normalize features using train scaler
  - [ ] Forward pass through policy (deterministic, no noise)
  - [ ] Apply softmax, return weights summing to 1
  - [ ] Error handling: return equal-weight if inference fails
- [ ] **run_allocation.py:** Daily entry point
  - [ ] Load latest `ml_dataset.parquet` (most recent date per ticker)
  - [ ] Call `predict_weights(agent, latest_features)`
  - [ ] Output: CSV with columns {ticker, weight, sector} (sorted by weight descending)
  - [ ] Alternative output: JSON with {timestamp, total_value, tickers: [{ticker, weight, sector}]}
  - [ ] Logging: timestamp, total_value, any warnings (extreme weights, NaNs, model errors)
  - [ ] CLI: `python src/agent/run_allocation.py --date 2026-07-01 --format csv`
- [ ] **Model versioning**
  - [ ] Save final model: `data/models/agent_final.pt`
  - [ ] Save metadata: training_date, test_sharpe, test_max_dd, scaler_version, feature_list
  - [ ] Fallback logic: If load fails, return equal-weight portfolio + log warning
- [ ] **Integration testing**
  - [ ] `test_inference_daily.py`: Load model, run inference, verify output shape & sum
  - [ ] `test_allocation_output.py`: Run run_allocation.py, verify CSV/JSON validity

### Phase 3 Quality Gates (Before Merge)

- [ ] **Correctness checks:**
  - [ ] No lookahead bias: all validation/test dates use only past data
  - [ ] Weights sum to 1.0 (within 1e-6 tolerance)
  - [ ] No NaN in portfolio values or weights
  - [ ] Portfolio value always ≥ initial capital (no rounding errors)

- [ ] **Performance checks:**
  - [ ] Test Sharpe ratio ≥ 0.5 (better than random baseline)
  - [ ] Test max drawdown < 50% (not catastrophic losses)
  - [ ] Training curves show convergence (loss/reward improving over time)
  - [ ] Validation Sharpe tracks training (not overfitting)

- [ ] **Code checks:**
  - [ ] Type hints on all functions
  - [ ] Docstrings on public API
  - [ ] No print() statements (use logger)
  - [ ] No hardcoded paths (use config)
  - [ ] Each file ≤ 300 lines
  - [ ] All imports at top, organized by stdlib/3rd-party/local

- [ ] **Documentation:**
  - [ ] README in `src/agent/`: quick-start guide, config example, output format
  - [ ] Inline comments on complex logic (RL algorithms, normalization)
  - [ ] Architecture diagram (state flow, module dependencies)

---

## Future Scope (Out of Phase 1–3)

- [ ] **FIIs (Real Estate Investment Trusts):** Deferred pending Phase 3 scope decision. If RL agent expands to mixed-asset allocation, add separate collectors for FII prices + distributions. API endpoints exist; collector skeleton pattern is proven. **Why deferred**: FIIs have different fundamentals (NAV/P-VP vs earnings) and require separate dataset build logic.

---

See `specification.txt` for complete system design (input features, model output, objective function, constraints, expected behavior).
