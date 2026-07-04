# ML Agent Roadmap (ml_agent branch)

Building the RL agent for portfolio allocation (Stage 3).

**See also:** `CLAUDE.md` ("Stage 3: ML Agent Architecture & Development Guide") for architectural decisions, coding conventions, and development workflow. `TODO.md` contains actionable task checklists for each phase below.

---

## Key Architectural Decisions (from CLAUDE.md)

This roadmap implements the following architectural decisions; **read CLAUDE.md for full rationale:**

1. **Anchored Rolling Windows (never a fixed split):** Training partitions the dataset into anchored windows (train always starts at the dataset's earliest date, test slides forward ~2 years per window); each window's train span is tail-carved into train/val for early stopping. The most recent window is the production model/default split. Prevents lookahead bias, respects market regime shifts, and produces multiple out-of-sample evaluations instead of one.
2. **Train-Only Scaler:** Fit StandardScaler on training data only; reuse for validation/test/inference. Stored as `data/models/feature_scaler.pkl`.
3. **State Space:** Concatenated normalized features `[n_tickers * feature_dim]`. Simple, learnable end-to-end.
4. **Action Space:** Continuous weights via softmax output (guarantees simplex: Σw_i=1, w_i≥0). No-shorting constraint built-in.
5. **Reward Function:** Daily log return `r_t = log(V_t / V_{t-1})` (simple, unambiguous; switch to Sharpe-based if variance is high).
6. **Algorithm:** PPO via `stable-baselines3` (vetted, production-ready, fast iteration).
7. **Code Style:** Immutable dataclasses (config), type hints (all functions), ≤300 lines per file, structured logging (JSONL).

---

## Phase 1: Foundation & Environment Design

**Task List:** See `TODO.md` → "Phase 3a: Foundation & Environment Design" for detailed checkboxes and sub-tasks.

### 1a. State Space Definition
From `ml_dataset.parquet`, each state (ticker, date) includes:
- **Price features:** OHLC, returns, volatility (rolling 20/60d), RSI
- **Fundamental features:** P/E, P/B, ROE, debt/equity, profit margin (quarterly, forward-filled)
- **Macro features:** SELIC rate, CDI, inflation (IPCA), lagged 1-252 days
- **Normalization:** Z-score per feature + ticker (handle cross-sectional differences)

**State vector:** Concatenate normalized features for all ~N tickers at each date → shape `[N * feature_dim]`

### 1b. Action Space & Portfolio Constraints
- **Action:** Continuous weights `w_t ∈ [0, 1]^N` where Σw_i = 1 (simplex)
- **Enforcement:** Use softmax output layer (guarantees simplex automatically)
- **Constraints:** No shorting, optional: add trading cost (proportional to |w_t - w_{t-1}|)

### 1c. Reward Function (Choose One)
**Option A (Simple):** Daily log return of portfolio
```
r_t = log(V_t / V_{t-1}) = log(Σ w_i * r_i,t)
```
Sparse signal, may converge slowly.

**Option B (Sharpe-based):** Exponential moving Sharpe ratio
```
sharpe_t = rolling_mean(r_t) / (rolling_std(r_t) + eps)
r_t = sharpe_t - sharpe_{t-1}
```
Encourages risk-adjusted returns; adds complexity.

**Recommended:** Start with Option A (return), switch to B if return variance is high.

### 1d. Setup Tasks
- [ ] Add ML dependencies to `requirements.txt`:
  - [ ] **Decision:** `torch` + custom policy, or `stable-baselines3` (PPO/A2C)?
    - `stable-baselines3`: Faster start, vetted algorithms, less code
    - `torch`: Full control, custom reward shaping, better for research
  - [ ] Suggested: `torch==2.3.0`, `stable-baselines3==2.4.0` (use both: SB3 wraps Torch)
- [ ] Create `src/agent/` directory structure:
  ```
  src/agent/
  ├─ __init__.py
  ├─ env.py           # PortfolioEnv (gym/gymnasium interface)
  ├─ policy.py        # Policy network (Actor-Critic or PPO)
  ├─ trainer.py       # Training loop
  ├─ evaluate.py      # Backtesting & metrics
  └─ config.py        # Hyperparams (learning_rate, gamma, etc.)
  ```
- [ ] Load `ml_dataset.parquet`, inspect shape and columns
- [ ] Implement `PortfolioEnv(gym.Env)`:
  - `__init__`: Store ticker list, feature scaler, initial capital
  - `reset()`: Return initial state (first date in train set)
  - `step(action)`: Apply weights, compute reward, return next state + reward + done
  - `render()`: (Optional) Print portfolio value and weights

## Phase 2: Training Loop & Infrastructure

**Task List:** See `TODO.md` → "Phase 3b: Training Infrastructure" for detailed checkboxes and sub-tasks.

### 2a. Data Pipeline
- [ ] Generate anchored rolling windows (train always from dataset start, test slides forward); tail-carve each window's train span into train/val by date (preserve temporal order)
- [ ] Normalize training features on *train set only* (prevent lookahead bias)
- [ ] Create train/val environments from respective date ranges
- [ ] Cache normalized datasets as parquets in `data/processed/agent_train.parquet` etc.

### 2b. Algorithm: PPO (Proximal Policy Optimization)
**Why PPO:** Stable, sample-efficient, handles continuous actions well, SOTA for portfolio RL.

**Hyperparameters to tune:**
```python
learning_rate = 3e-4
gamma = 0.99           # discount factor
gae_lambda = 0.95      # GAE smoothing
n_steps = 2048         # trajectory length per update
batch_size = 64
n_epochs = 10          # gradient updates per rollout
entropy_coef = 0.01    # exploration bonus
```

- [ ] Implement PPO from scratch *or* use `stable-baselines3.PPO`:
  ```python
  from stable_baselines3 import PPO
  agent = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=1)
  agent.learn(total_timesteps=1_000_000)
  ```
  (Recommended: SB3 for speed, custom loop for research)

### 2c. Training Loop (if custom)
```
for episode in range(num_episodes):
  state = env.reset()
  trajectory = []
  for t in range(max_steps):
    action = agent.policy(state)  # + noise for exploration
    next_state, reward, done, info = env.step(action)
    trajectory.append((state, action, reward, next_state, done))
    state = next_state
    if done: break
  
  advantages, returns = compute_gae(trajectory, gamma, gae_lambda)
  loss = ppo_loss(trajectory, advantages, returns, agent.policy)
  optimizer.step(loss)
  
  log: avg_return, portfolio_value, sharpe, max_dd
```

- [ ] Add real-time logging:
  - Per episode: portfolio value, return %, Sharpe, max drawdown, weights (sector %)
  - Per step: loss, gradient norm, policy entropy
  - Save to `data/logs/agent_training_YYYYMMDD.jsonl`
- [ ] Checkpoint every N episodes: `data/models/agent_checkpoint_ep{N}.pt`
- [ ] Early stopping: stop if val Sharpe degrades for M consecutive eval cycles

## Phase 3: Evaluation & Backtesting

**Task List:** See `TODO.md` → "Phase 3c: Evaluation & Backtesting" for detailed checkboxes and sub-tasks.

### 3a. Metrics (compute on *test set only*)
- **Cumulative Return:** `(V_final - V_init) / V_init`
- **Annualized Sharpe:** `mean(daily_returns) / std(daily_returns) * sqrt(252)`
- **Max Drawdown:** `min( (V_peak - V_t) / V_peak )`
- **Sortino Ratio:** Like Sharpe, but penalize downside variance only
- **Win Rate:** % of days with positive return
- **Sector Exposure:** Distribution of weights over sectors

### 3b. Baselines (for comparison on test set)
- **Equal-Weight:** w_i = 1/N for all i (rebalance daily)
- **Market-Cap Weight:** w_i ∝ market cap (buy and hold or rebalance)
- **1/Vol Weight:** w_i ∝ 1/volatility (inverse-volatility-weighted)

### 3c. Evaluation Script (`src/agent/evaluate.py`)
```python
def backtest(agent, env_test, metrics_to_compute):
  # Run agent on test set, collect trajectory
  values, returns, weights = []
  state = env_test.reset()
  while not done:
    action = agent.policy(state)  # deterministic
    state, reward, done, info = env_test.step(action)
    values.append(info['portfolio_value'])
    weights.append(action)
  
  # Compute metrics
  sharpe = compute_sharpe(returns)
  max_dd = compute_max_drawdown(values)
  ...
  
  # Save results
  results_df = pd.DataFrame({
    'date': env_test.dates,
    'portfolio_value': values,
    'weights': weights,  # N x num_tickers
    'action': ...
  })
  results_df.to_parquet('data/backtest/results.parquet')
  
  return {sharpe, max_dd, ...}
```

- [ ] Generate plots (Plotly or matplotlib):
  - Cumulative value: agent vs baselines
  - Drawdown over time
  - Sector allocation heatmap (date × sector)
  - Return distribution histogram
  - Weights timeline (stacked bar chart)
- [ ] Save plots to `data/backtest/plots/`
- [ ] Comparative table: agent vs 3 baselines on all metrics

## Phase 4: Deployment & Inference

**Task List:** See `TODO.md` → "Phase 3d: Deployment & Inference" for detailed checkboxes and sub-tasks.

### 4a. Inference Module (`src/agent/infer.py`)
```python
def predict_weights(agent, latest_features_df, n_tickers):
  """
  Input: latest_features_df with shape [n_tickers, feature_dim]
  Output: weights [n_tickers] summing to 1
  """
  state = normalize_features(latest_features_df, scaler)  # use train scaler
  with torch.no_grad():
    weights = agent.policy(torch.tensor(state)).numpy()
  return weights / weights.sum()  # ensure sum=1
```

### 4b. Integration Entry Point
- [ ] Script: `src/agent/run_allocation.py`
  - Load latest `ml_dataset.parquet` row per ticker
  - Call `predict_weights(agent, features)`
  - Output: CSV/JSON with (ticker, weight) pairs
  - Log: timestamp, total value, warnings (extreme weights, NaNs, etc.)
- [ ] Cron hook (optional): Run daily after market close to generate next-day weights

### 4c. Model Management
- [ ] Save final agent: `data/models/agent_final.pt` (+ metadata: train date, test sharpe, etc.)
- [ ] Version control: git-track `data/models/agent_final.pt` (or use DVC if >100MB)
- [ ] Fallback: If agent inference fails, default to equal-weight portfolio

---

## File Structure (After Completion)
```
src/agent/
├─ __init__.py
├─ config.py          # Hyperparameters, paths, feature list
├─ env.py             # PortfolioEnv, state normalization
├─ policy.py          # Policy network (MLP), action sampling
├─ trainer.py         # Training loop (manual or SB3 wrapper)
├─ evaluate.py        # Backtesting, metrics, plots
├─ infer.py           # Inference for live allocation
└─ run_allocation.py  # Daily entry point

data/
├─ processed/
│  ├─ ml_dataset.parquet           (from Stage 2)
│  ├─ agent_train.parquet          (train split, normalized)
│  ├─ agent_val.parquet
│  └─ agent_test.parquet
├─ models/
│  ├─ agent_checkpoint_ep50.pt
│  ├─ agent_checkpoint_ep100.pt
│  └─ agent_final.pt
├─ backtest/
│  ├─ results.parquet              (test set trajectory)
│  └─ plots/
│     ├─ cumulative_value.html
│     ├─ drawdown.html
│     ├─ sector_allocation.html
│     └─ comparison_metrics.json
└─ logs/
   └─ agent_training_*.jsonl
```

---

## Dependencies to Add
```bash
# Add to requirements.txt:
torch==2.3.0
stable-baselines3==2.4.0
gymnasium==0.29.0    # (replaces gym; needed for SB3)
scikit-learn==1.5.0  # (for scaler)
```

---

## Quality Gates (Before Merge to main)

**See `TODO.md` → "Phase 3 Quality Gates (Before Merge)" for the full checklist.**

After Phase 4, verify:
- **Correctness:** No lookahead bias, weights sum to 1, no NaN, portfolio value ≥ initial capital
- **Performance:** Test Sharpe ≥ 0.5, max drawdown < 50%, training curves converge, val Sharpe tracks training
- **Code Quality:** Type hints on all functions, docstrings on public API, no print() (use logger), no hardcoding (use config), each file ≤ 300 lines
- **Documentation:** README in `src/agent/`, inline comments on complex logic, architecture diagram

---

## Start Here
1. **Week 1:** Phase 1a–1d
   - Inspect `ml_dataset.parquet` columns (run `python tests/build_dataset/test_final_dataset.py`)
   - Build `PortfolioEnv` (gym interface)
   - Run 100 random episodes, log returns

2. **Week 2:** Phase 2
   - Implement PPO or use `stable-baselines3.PPO`
   - Train on toy dataset (5 tickers, 2 years of data)
   - Check convergence: portfolio value should trend upward

3. **Week 3:** Phase 3
   - Backtest on full dataset (train/val/test split)
   - Compare to baselines
   - Generate plots

4. **Week 4:** Phase 4
   - Write inference script
   - Test daily rebalancing logic
   - Ready for live integration

---

## Documentation Cross-Reference

| Document | Purpose | Key Sections |
|----------|---------|--------------|
| **CLAUDE.md** | Project guide & architecture | "Stage 3: ML Agent Architecture & Development Guide" — architectural decisions, coding conventions, module responsibilities, development workflow, assumptions, constraints |
| **TODO.md** | Actionable task checklists | "Phase 3: RL Agent" with Phase 3a–3d sub-sections + Quality Gates checklist |
| **ML_AGENT_ROADMAP.md** | Phase-by-phase technical guide | This file — design decisions, implementation examples, outputs per phase |

**Quick Links:**
- Data flow diagram: CLAUDE.md § "Architecture" → "Data Flow"
- File structure: CLAUDE.md § "Key Modules" → Stage 3 table
- Development workflow: CLAUDE.md § "Stage 3" → "Development Workflow"
- Assumptions & constraints: CLAUDE.md § "Stage 3" → "Key Assumptions & Constraints"
