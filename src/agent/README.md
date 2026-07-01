# ML Agent: Portfolio Allocation via Reinforcement Learning

RL agent for allocating capital across B3 (Brazilian) equities using PPO (Proximal Policy Optimization).

## Quick Start

### 1. Prepare Dataset (One-Time)

Ensure returns are computed and dataset is ready:

```bash
python src/agent/feature_engineering.py
```

Output: `data/processed/ml_dataset_training.parquet` (847,928 rows, 135 columns, returns computed)

### 2. Verify Configuration

Check agent config (paths, dates, features):

```bash
python src/agent/config.py
```

Output:
```
AGENT CONFIGURATION
  Dataset: data/processed/ml_dataset_training.parquet
  Temporal Splits: Train (2000-2015), Val (2015-2021), Test (2021-2026)
  State Features: 23 (price, fundamental, macro)
  Hyperparams: LR=3e-4, γ=0.99
```

### 3. Build Env Tensors (One-Time)

```bash
python -m src.agent.data_pipeline
```

Output: `data/processed/agent_tensors.npz` + `data/models/feature_scaler.pkl`

### 4. Train, Evaluate, Allocate

```bash
python -m src.agent.trainer --timesteps 12288     # smoke run (~1 min, GPU)
python -m src.agent.trainer                       # full run (1M timesteps)
python -m src.agent.evaluate                      # backtest agent_best.zip vs baselines
python -m src.agent.run_allocation --date 2026-06-29 --format csv
```

### 5. Tests

```bash
python tests/agent/verify_dataset_for_training.py
python tests/agent/test_env_basic.py
python tests/agent/test_backtest_metrics.py
python tests/agent/test_inference_output.py
```

## Architecture

### Modules

| Module | Purpose | Status |
|--------|---------|--------|
| `config.py` | Immutable configuration (frozen dataclass) | ✓ Complete |
| `feature_engineering.py` | Returns from `adj_close` (split-safe) + corrupt-row cleaning | ✓ Complete |
| `data_pipeline.py` | Dense tensors [dates×tickers×features] + mask + train-only scaler | ✓ Complete |
| `env.py` | PortfolioEnv: masked time-varying universe, masked softmax | ✓ Complete |
| `metrics.py` | Sharpe, Sortino, max DD, win rate (shared) | ✓ Complete |
| `trainer.py` | PPO training loop, val-Sharpe early stopping, checkpoints | ✓ Complete |
| `evaluate.py` | Backtest vs 3 baselines, metrics.json, Plotly plots | ✓ Complete |
| `infer.py` | `predict_weights(date)` + equal-weight fallback | ✓ Complete |
| `run_allocation.py` | Daily entry point (CSV/JSON) | ✓ Complete |

No `policy.py`: SB3's built-in `MlpPolicy` is used (custom net not needed for v1).

### Data Flow

```
ml_dataset.parquet (Stage 2 output)
         ↓
feature_engineering.py (compute returns if missing)
         ↓
ml_dataset_training.parquet
         ↓
config.py (defines train/val/test date ranges)
         ↓
env.py (PortfolioEnv loads and normalizes data per date range)
         ↓
trainer.py (PPO training on train set, early stopping on val set)
         ↓
evaluate.py (backtest on test set)
```

## Key Design Decisions

### 1. Returns Computation

**Why:** Missing `returns` feature from Stage 2.

**How:** `compute_returns()` computes log returns grouped by ticker:
```python
returns_t = log(close_t / close_{t-1})
```

**Implementation:** One-time in `feature_engineering.py`, saved to `ml_dataset_training.parquet`. Reusable by all downstream code.

**NaN Handling:** First row per ticker is NaN (no previous price). Expected behavior, model learns to ignore.

### 2. Feature Selection (State Space)

**23 features** (per ticker):
- **Price (6):** open, high, low, close, volume, returns
- **Fundamental (14):** valuation (pl, pvp), profitability (roe, roic, roa, margins), liquidity (current_ratio, cash_ratio), growth (earnings/revenue/ebitda growth YoY)
- **Macro (3):** selic (rate), cdi, ipca (inflation)

**Why not CAGR?** CAGR columns have 50-70% NaN. Rather than impute (creates synthetic data), we use YoY growth features which are real and more recent.

**NaN Strategy:** Model handles missing values. Fundamental data is sparse (quarterly, forward-filled). This is realistic — production inference will also have missing data.

### 3. Temporal Splits (By Date, Not Rows)

```
Train (60%): 2000-01-03 → 2015-11-25 (15.9 years, 326K rows)
Val   (20%): 2015-11-26 → 2021-03-13 (5.3 years, 198K rows)
Test  (20%): 2021-03-14 → 2026-06-30 (5.3 years, 324K rows)
```

**Why:** Prevents lookahead bias. Agent trains on history, validates on intermediate period, tests on most recent (realistic deployment scenario).

### 4. Configuration as Code (Not YAML)

**Why:** Immutable dataclass prevents accidental mutation, validates on load, integrates seamlessly with Python.

**Usage:**
```python
from src.agent.config import AgentConfig, DEFAULT_CONFIG

# Use defaults
config = DEFAULT_CONFIG

# Or override
config = AgentConfig(
    learning_rate=1e-4,
    total_timesteps=2_000_000,
)
config.log_summary()  # Print for debugging
```

## Dataset Details

From verification (`tests/agent/verify_dataset_for_training.py`):

| Metric | Value |
|--------|-------|
| **Date Range** | 2000-01-03 to 2026-06-30 (26.49 years) |
| **Tickers** | 288 (279 with full history) |
| **Total Rows** | 847,928 |
| **Columns** | 135 (after returns computation) |
| **Overall NaN Rate** | 15.34% (mostly high-NaN CAGR cols) |
| **Critical Columns** | 0% NaN (ticker, close, volume, sector) |
| **Sectors** | 44 (well diversified) |

## Next Steps

### Phase 1b: Environment

1. Implement `env.py` with PortfolioEnv class
   - Load dataset once in `__init__`
   - Reset to first date in train/val/test range
   - Step: apply softmax weights, compute reward, advance date
   - State: normalized feature vector

2. Test with random policy (`test_env_basic.py`)
   - Verify state shape, action application, reward computation
   - Run 100 episodes, check portfolio value > 0

### Phase 2: Training

1. Implement `trainer.py` (PPO training loop)
   - Instantiate env, agent (SB3 PPO)
   - Training loop with logging, checkpointing, early stopping

2. Implement `policy.py` (if custom, not SB3)
   - MLP actor-critic network

### Phase 3: Evaluation

1. Implement `evaluate.py` (backtesting)
   - Run on test set, compute metrics (Sharpe, max DD, Sortino)
   - Compare vs baselines
   - Generate plots

### Phase 4: Deployment

1. Implement `infer.py` (inference)
   - Load model, predict weights from latest features

2. Implement `run_allocation.py` (daily entry point)
   - Load latest data, call inference, output weights

## Debugging

### Check Dataset is Ready

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/ml_dataset_training.parquet')
print(f'Rows: {len(df)}, Cols: {len(df.columns)}')
print(f'Returns NaN: {df[\"returns\"].isnull().sum()}')
print(f'Features: {df.columns.tolist()[:20]}')
"
```

### Check Config is Valid

```bash
python src/agent/config.py
```

Should print configuration summary without errors.

### Trace Feature Engineering

```bash
python src/agent/feature_engineering.py --force
```

Forces recomputation of returns even if they exist (useful for debugging).

## References

- Architecture: See `CLAUDE.md` → "Stage 3: ML Agent Architecture & Development Guide"
- Roadmap: See `ML_AGENT_ROADMAP.md` → Phase 1–4 details
- Tasks: See `TODO.md` → Phase 3a–3d checkboxes
- Dataset verification: Run `python tests/agent/verify_dataset_for_training.py`
