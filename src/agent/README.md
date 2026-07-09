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

Output (splits shown are for the most recent anchored rolling window — see "Anchored Rolling Windows" below):
```
AgentConfig | ml_dataset_training.parquet | 23 features (6p+14f+3m)
  splits: train 2000-01-03→2020-06-03 | val 2020-06-04→2024-01-09 | test 2024-01-10→2026-01-10
  ppo: lr=0.0003 γ=0.99 λ=0.95 steps=1,000,000 batch=64 epochs=10
```

### 3. Build Env Tensors (One-Time)

```bash
python -m src.agent.data_pipeline
```

Output: `data/processed/agent_tensors.npz` + `artifacts/models/feature_scaler.pkl`

### 4. Train, Evaluate, Allocate

Training is always via anchored rolling windows — one PPO model per window, sequentially, in a single command:

```bash
python -m src.agent.trainer --timesteps 12288                # smoke run (per window, ~1 min/window, GPU)
python -m src.agent.trainer                                   # full run: 8 windows x 1M timesteps each
python -m src.agent.trainer --train-years 2 --test-years 1    # fast smoke: many small windows
python -m src.agent.evaluate                                  # backtest agent_best.zip (latest window) vs baselines
python -m src.agent.run_allocation --date 2026-06-29 --format csv
```

The most recent window's model is saved as `agent_best.zip`/`agent_final.zip` (the production model); earlier windows are namespaced `window_{id}_best.zip`/`window_{id}_final.zip`. The stitched out-of-sample curve across all windows is written to `artifacts/backtest/walkforward_results.parquet` + `walkforward_metrics.json`.

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
| `trainer.py` | PPO training loop per window, val-Sharpe early stopping, checkpoints; `main()` = sole rolling-window training CLI | ✓ Complete |
| `rolling_eval.py` | Window generation/training/eval orchestration (called by `trainer.py`); walk-forward stitching | ✓ Complete |
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
config.py (generate_windows() → anchored rolling windows; DEFAULT_CONFIG = most recent window)
         ↓
env.py (PortfolioEnv loads and normalizes data per date range)
         ↓
trainer.py + rolling_eval.py (PPO training per window, early stopping on each window's val tail)
         ↓
evaluate.py (backtest on the most recent window's test set)
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

### 3. Anchored Rolling Windows (By Date, Not Rows, Never a Fixed Split)

Training is never a single fixed train/val/test split — it always partitions
the dataset into anchored windows via `config.generate_windows()`:

```
Window 0: train 2000-01-03→2010-01-02 | test 2010-01-03→2012-01-03
Window 1: train 2000-01-03→2012-01-03 | test 2012-01-04→2014-01-04
...
Window 7: train 2000-01-03→2024-01-09 | test 2024-01-10→2026-01-10   ← most recent
```

Controlled by `AgentConfig` fields: `dataset_start`/`dataset_end` (2000-01-03 →
2026-06-30), `window_train_years` (10), `window_test_years` (2). Each window's
train span is further tail-carved by `window_val_fraction` (0.15) into
train/val for early stopping — e.g. window 7 becomes
`train 2000-01-03→2020-06-03 | val 2020-06-04→2024-01-09 | test 2024-01-10→2026-01-10`.
`DEFAULT_CONFIG` is always the **most recent** window's config, so
`evaluate.py`/`infer.py`/`run_allocation.py` automatically target the newest
held-out period.

**Why:** Prevents lookahead bias (test always follows train chronologically)
and produces 8+ independent out-of-sample evaluations across different market
regimes (2008 crash, COVID, rate hikes, ...) instead of one. The most recent
window doubles as the production model — trained on nearly all available
history, tested on the newest unseen period (realistic deployment scenario).

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
- Roadmap: See `docs/ML_AGENT_ROADMAP.md` → Phase 1–4 details
- Tasks: See `docs/TODO.md` → Phase 3a–3d checkboxes
- Dataset verification: Run `python tests/agent/verify_dataset_for_training.py`
