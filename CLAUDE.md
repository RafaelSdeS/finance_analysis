# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Project:** Brazilian-equity ML pipeline for reinforcement-learning-based portfolio allocation.

**Goal:** Collect daily stock prices and quarterly company fundamentals, then build a machine-learning dataset ready for model training. See `specification.txt` for system design and RL objective.
>>
**Pipeline:** Structured three-stage approach:
1. **Stage 1 (Data Collection):** Staged prototype→validation→full-scale pipeline with checkpointing, logging, validation
2. **Stage 2 (Dataset Build):** Merge raw data → add derived features (technical indicators, fundamentals, macro-adjusted) → clean → output ML-ready parquets (no lookahead bias)
3. **Stage 3 (Model):** RL agent training (future, separate branch; consumes feature-complete dataset from Stage 2)

All scripts run from project root.

## Setup

```bash
pip install -r requirements.txt
```

## Run Commands

### Stage 1: Collect Raw Data

**Initial Setup** (one-time, with BolsAI key):
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add: BOLSAI_API_KEY=sk_...
```

**Backfill Stage** (one-time: historical data via BolsAI, covers 2000–present):
```bash
python -m src.data_collection.pipeline --mode full_scale       # all ~500+ tickers
python -m src.data_collection.pipeline --mode full_scale --dry-run   # preview ticker list
python -m src.data_collection.pipeline --mode prototype --tickers PETR4 VALE3   # override
```
Resumes mid-run from checkpoints (idempotent: re-runs only fetch new data).

**Quarterly Incremental Updates** (no BolsAI key needed; uses free yfinance):
```bash
python -m src.data_collection.pipeline --mode update
```
Fetches only new trading days/quarters for prices/fundamentals/dividends from yfinance, merges into existing raw data. Replaces BolsAI for routine refreshes (>99% cost savings on API calls).

**Validation** (after any stage):
```bash
python tests/raw_data/validate_vs_yfinance.py
```
Cross-checks BolsAI data against yfinance (yfinance-derived fundamentals verified to within 1–15% tolerance on key ratios).

### Stage 2: Build ML Dataset

**Prerequisites:** Stage 1 must be complete (raw data in `data/raw/`).

Merges prices + fundamentals + company info (no lookahead bias; uses `merge_asof` backward):
```bash
python "src/2. build_dataset/build_ml_dataset.py"
```
Saves to `data/processed/ml_dataset.parquet` (one row per ticker + date).

**Validation** (after build):
```bash
python tests/processed_data/test_final_dataset.py
python tests/processed_data/test_final_dataset.py --file data/processed/ml_dataset.parquet
```
Checks schema, shape, lookahead, NaN counts, return distribution.

See `BUILD_DATASET_ROADMAP.md` for phase-by-phase implementation guide (5 phases: load, merge, feature engineering, clean, validate).

### Stage 3: Train ML Agent (ml_agent branch)

**Prerequisites:**
```bash
pip install torch stable-baselines3 gymnasium scikit-learn
```

**Quick Start** (see `ML_AGENT_ROADMAP.md` for detailed 4-phase guide):
```bash
# Phase 1: Environment setup
python -c "from src.agent.env import PortfolioEnv; print('Environment ready')"

# Phase 2: Train PPO agent
python src/agent/trainer.py --config src/agent/config.py

# Phase 3: Backtest on test set
python src/agent/evaluate.py --model data/models/agent_final.pt

# Phase 4: Daily inference
python src/agent/run_allocation.py --date 2026-06-29
```

**Outputs:**
- Trained model: `data/models/agent_final.pt`
- Backtest results: `data/backtest/results.parquet`
- Visualizations: `data/backtest/plots/*.html`
- Portfolio weights (daily): CSV/JSON with (ticker, weight) pairs

### Utilities

**CAGR calculator** (CLI and module):
```bash
python src/cagr_handler.py --ticker PETR4
```

**Visualization — Quick Snapshot** (BBAS3 nominal vs inflation-adjusted vs SELIC, live yfinance data):
```bash
python src/visualizations/financial_view.py
```

**Exploration Notebook** (full dataset validation + insights, Jupyter):
```bash
jupyter notebook src/visualizations/exploration.ipynb
```
Charts: price coverage, data completeness, liquidity, sector breakdown, inflation-adjusted returns (PETR4/VALE3/WEGE3/ITUB4), P/E/ROE/net margin by sector, market cap distribution, leverage, growth CAGR, dividend analysis.

### Tests

All tests are plain Python scripts (no pytest). Run from project root:

**Stages 1–2 (Data Collection & Dataset Build):**
```bash
python tests/processed_data/test_final_dataset.py
python tests/processed_data/test_final_dataset.py --file data/processed/ml_dataset.parquet
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY
python tests/api/bolsai_api_price_depth.py
python tests/api/bolsai_api_macro_depth.py
python tests/api/bolsai_test_cagr.py
python tests/raw_data/test_cagr_calculation.py
python tests/raw_data/test_ticker_data.py
python tests/raw_data/inspect_all_data.py
python tests/raw_data/inspect_company_info.py
```

**Stage 3 (ML Agent, ml_agent branch):**
```bash
# Phase 1: Environment validation
python tests/agent/test_env_basic.py           # Verify PortfolioEnv reset/step, state shapes

# Phase 2: Training validation
python tests/agent/test_training_convergence.py  # Run N episodes, verify loss decreases

# Phase 3: Evaluation validation
python tests/agent/test_backtest_metrics.py    # Run backtest, verify Sharpe/max_dd are reasonable

# Phase 4: Inference validation
python tests/agent/test_inference_output.py    # Verify weights sum to 1, no NaN, valid shape
```

**Development workflow:**
- After each phase implementation, run corresponding test
- Keep tests lightweight (no fixtures, direct function calls)
- Log results to console; save detailed output to `data/logs/test_*.log`

## Branches

- **main:** Base branch, stable. Stages 1–2 merged here once validated.
- **build_dataset:** Stage 2 (dataset building). Focus on merging raw data → ml_dataset.parquet. See `BUILD_DATASET_ROADMAP.md`.
- **ml_agent:** Stage 3 (RL agent). Separate effort after Stage 2 complete. See `ML_AGENT_ROADMAP.md` (if exists on that branch).
**Generic API Endpoint Tester** (explore BolsAI endpoints without writing code):
```bash
# Test specific endpoint + params
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY --path /dividends/PETR4 --param years=5

# Run full validation suite (9 checks)
python tests/api/bolsai_api_validator.py --api-key YOUR_API_KEY
```

## Architecture

### Data Flow

```
External APIs
├─ BCB SGS (macro: SELIC, CDI, IPCA)
└─ BolsAI (prices OHLCV + fundamentals quarterly + dividends)
        ↓
data/raw/
  ├─ prices/{TICKER}.parquet           (daily)
  ├─ fundamentals/{TICKER}.parquet     (quarterly)
  └─ macro/{selic,cdi,ipca}.parquet
        ↓ [Stage 2]
  ├─ dividends/{TICKER}.parquet        (historical, ~20 years)
  ├─ macro/{selic,cdi,ipca}.parquet
  └─ company_info/company_info.parquet
        ↓
build_ml_dataset.py
  → merge_asof(prices, fundamentals)   [no lookahead]
  → left join company_info
  → compute_price_features()           [RSI, MA20/60, volatility, returns, drawdown]
  → compute_fundamental_features()     [P/E, P/B, ROE, debt/equity, growth CAGR]
  → compute_macro_features()           [real return, excess return, rate environment]
  → fill_cagr_columns()                [backfill from earnings/revenue where API null]
  → clean (drop dupes, NaNs, outliers, sort)
        ↓
data/processed/ml_dataset.parquet      (one row per ticker+date)
        ↓ [Stage 3]
PortfolioEnv (gymnasium)
  → state: normalized features (price, fundamentals, macro)
  → action: portfolio weights (softmax)
  → reward: daily log return
        ↓
trainer.py (PPO)
  → train on [train_set: 60%], validate on [val_set: 20%]
  → save checkpoints
        ↓
data/models/agent_final.pt             (trained policy)
        ↓
evaluate.py: backtest on [test_set: 20%]
  → metrics: Sharpe, max drawdown, Sortino
  → comparison: vs equal-weight, market-cap, 1/vol baselines
        ↓
data/backtest/results.parquet          (trajectory, weights, returns)
data/processed/ml_dataset.parquet      (one row per ticker+date, includes dividend data)
```

### Key Modules

**Stages 1–2 (Data Collection & Dataset Build):**
**Stage 1 (Data Collection):**

| File | Purpose |
|------|---------|
| `src/data_collection/pipeline.py` | Orchestration (prototype/validation/full-scale modes, checkpointing) |
| `src/data_collection/collectors.py` | All collectors: BCB macro, BolsAI prices/fundamentals/company info |

**Stage 2 (Dataset Build, build_dataset branch):**

| File | Purpose |
|------|---------|
| `src/data_loading.py` | Utilities: `load_prices()`, `load_fundamentals()`, `load_macro_series()`, `inspect_dataset()` |
| `src/2. build_dataset/build_ml_dataset.py` | Orchestration: load → merge_asof → feature engineering → clean → save ml_dataset.parquet |
| `src/cagr_handler.py` | CAGR calculation/filling: use BolsAI values first, backfill from earnings/revenue. Uncomment line 143 in build_ml_dataset.py to enable. |

**Utilities:**

| File | Purpose |
|------|---------|
| `src/visualizations/financial_view.py` | Standalone Plotly chart: BBAS3 nominal/inflation-adjusted prices + SELIC overlay (uses `yfinance`). |

**Stage 3 (ML Agent, ml_agent branch):** See `ML_AGENT_ROADMAP.md` for detailed 4-phase guide (foundation → training → evaluation → deployment).

| File | Purpose | Status |
|------|---------|--------|
| `src/agent/__init__.py` | Package exports | Phase 1 |
| `src/agent/config.py` | Hyperparams: learning rate, gamma, feature list, train/val/test split dates, paths | Phase 1 |
| `src/agent/env.py` | PortfolioEnv (gymnasium interface): state space, action space, reward, normalization | Phase 1 |
| `src/agent/policy.py` | Policy network: MLP actor-critic or stable-baselines3 wrapper | Phase 2 |
| `src/agent/trainer.py` | Training loop: PPO (or SB3), checkpointing, logging, early stopping on val metrics | Phase 2 |
| `src/agent/evaluate.py` | Backtesting: metrics (Sharpe, max DD, Sortino), baseline comparisons, plots | Phase 3 |
| `src/agent/infer.py` | Inference: load trained agent, predict weights from feature state | Phase 4 |
| `src/agent/run_allocation.py` | Daily entry point: load latest data, predict weights, output CSV/JSON | Phase 4 |

## Branches

- **main:** Stages 1–2 (data collection + dataset build). Latest stable.
- **ml_agent:** Stage 3 (RL agent training). See `ML_AGENT_ROADMAP.md` for phase-by-phase implementation guide.

---

## Stage 3: ML Agent Architecture & Development Guide (ml_agent branch)

### Purpose & Scope

**Goal:** Build and train a reinforcement learning agent to allocate a portfolio across B3 equities by learning to maximize risk-adjusted returns (Sharpe ratio) using the feature-complete dataset from Stage 2.

**Scope:**
- **Input:** `data/processed/ml_dataset.parquet` (one row per ticker + date, with prices, fundamentals, macro features)
- **Environment:** PortfolioEnv (gymnasium interface) simulating daily portfolio rebalancing
- **Agent:** PPO (Proximal Policy Optimization) via stable-baselines3
- **Output:** Trained policy network, backtest results, daily allocation weights
- **Timeline:** 4 phases (foundation → training → evaluation → deployment)

### Key Architectural Decisions

#### 1. **Data Splits: Temporal, Not Random**
- **Train (60%):** Oldest dates → mid-period (learn from historical trends)
- **Val (20%):** Mid-period → near-recent (hyperparameter tuning, early stopping)
- **Test (20%):** Most recent → now (final evaluation, no touching during training)
- **Why:** Prevents lookahead bias and respects market regime shifts
- **Implementation:** Config specifies date ranges, not row counts

#### 2. **Normalization: Train-Set-Only Scaler**
- Fit StandardScaler on training data only
- Apply same scaler to val and test (prevents leakage)
- Store scaler in `data/models/feature_scaler.pkl` for inference
- **Why:** Production inference must use train-set statistics, not future data

#### 3. **State Space: Concatenated Normalized Features**
- Per-ticker features (normalized): prices, technicals, fundamentals, macro
- All tickers' features concatenated into one state vector: shape `[n_tickers * feature_dim]`
- Example: 50 tickers × 30 features = 1500-dim state
- **Why:** Simple, end-to-end learnable (agent discovers feature importance)

#### 4. **Action Space: Continuous Weights via Softmax**
- Raw network output → softmax → probability distribution (simplex: Σw_i = 1, w_i ≥ 0)
- No-shorting constraint built-in; no manual clipping
- **Why:** Mathematically clean, stable gradient flow

#### 5. **Reward Function: Daily Log Return (Simple)**
- `r_t = log(portfolio_value_t / portfolio_value_{t-1})`
- No transaction costs, no risk penalties (v1)
- **Future:** Switch to Sharpe-based reward if convergence is poor
- **Why:** Unambiguous signal; easier to debug than composite rewards

#### 6. **Algorithm: PPO via stable-baselines3**
- Use SB3's PPO implementation (vetted, production-ready)
- Hyperparameters: learning_rate=3e-4, gamma=0.99, gae_lambda=0.95
- **Alternative:** Custom torch loop if research requires fine-grained control
- **Why:** Faster iteration, fewer bugs, strong convergence guarantees

### Coding Conventions & Style

#### File Organization (Small File Principle)
- Each module ≤300 lines (config, env, trainer, evaluate, infer are separate)
- `__init__.py` exports public API only
- Helper functions (e.g., `_compute_returns()`) go in same file or a dedicated `_utils.py`

#### Configuration: Immutable Dataclasses
```python
# src/agent/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    train_split: float = 0.60  # Date range split
    feature_list: list[str] = field(default_factory=lambda: [...])
```
- All hyperparameters in config, zero hardcoding
- Frozen prevents accidental mutation

#### Type Hints: Mandatory
```python
def train_agent(config: AgentConfig, env: PortfolioEnv) -> PPO:
    """Train PPO on PortfolioEnv."""
    ...
```
- All function signatures include type hints
- Use `Optional[T]` explicitly, never bare `None`

#### Logging: Structured, Not Print
```python
import logging
logger = logging.getLogger(__name__)

# Good
logger.info(f"Epoch {ep}: sharpe={sharpe:.3f}, max_dd={max_dd:.3f}")

# Bad
print(f"Epoch {ep}: ...")
```
- File: `data/logs/agent_training_YYYYMMDD-HHMMSS.jsonl` (one JSON per line)
- Log levels: DEBUG (step-level), INFO (episode-level), WARNING (anomalies)

#### Testing: Lightweight, No pytest
- One-shot validation scripts in `tests/agent/`
- Example: `test_env_reset.py` manually checks PortfolioEnv reset logic
- No fixtures, no parametrization; just `if __name__ == '__main__':`

#### Naming: Snake Case + Descriptor
```python
# Good
portfolio_value_trajectory = [...]
compute_sharpe_ratio(returns)
class PortfolioEnv(gym.Env):

# Bad
pv = [...]
sharpe_fn(r)
class Env:
```

### Module Responsibilities & Data Flow

**src/agent/config.py**
- Single source of truth for hyperparams, paths, feature list
- No external reads (no `yaml`, no `.env`); all hardcoded or passed via CLI args
- Exports: `AgentConfig` dataclass

**src/agent/env.py**
- Implements `gymnasium.Env` interface
- Loads `ml_dataset.parquet` once in `__init__`
- `reset()`: Return normalized state for first date in date range
- `step(action)`: Apply weights, compute reward, advance date, return next state
- Handles train/val/test date ranges via config

**src/agent/trainer.py**
- Instantiates `PortfolioEnv` and PPO agent
- Training loop: collect trajectories, compute returns, update policy
- Logging: log metrics to file every N episodes
- Checkpointing: save model every N episodes
- Early stopping: monitor val Sharpe, stop if degrades for M cycles

**src/agent/evaluate.py**
- Instantiate fresh PortfolioEnv on test date range
- Run trained agent deterministically (no exploration noise)
- Compute metrics: Sharpe, max DD, Sortino, win rate
- Compare vs baselines (equal-weight, market-cap, 1/vol)
- Generate plots: cumulative value, drawdown, sector allocation, weights timeline
- Save: results.parquet, plots/*.html, metrics.json

**src/agent/infer.py**
- Load trained model and feature scaler
- Single function: `predict_weights(agent, latest_features_df) -> np.ndarray`
- Returns weights [n_tickers] summing to 1

**src/agent/run_allocation.py**
- Entry point for daily production use
- Load latest data from `ml_dataset.parquet`
- Call `predict_weights()`, output as CSV/JSON
- Logging: timestamp, total value, any warnings

### Key Assumptions & Constraints

#### Assumptions
1. **Feature Completeness:** All features in `ml_dataset.parquet` are present and up-to-date
2. **No Transaction Costs (v1):** Reward ignores trading friction; v2 can add cost penalties
3. **Daily Rebalancing:** Agent rebalances portfolio weights every day (high turnover; add cost constraints in v2)
4. **No Bankruptcy Risk:** All stocks remain liquid and tradeable throughout simulation
5. **Macro Stationarity:** SELIC, inflation regime doesn't change catastrophically (strong assumption; add regime detection in v2)

#### Constraints
1. **No Shorting:** Weights w_i ≥ 0 (enforced by softmax)
2. **Full Allocation:** Σw_i = 1 (no cash buffer; future versions can add cash_allocation feature)
3. **Training Data:** ≥2 years of history (train+val+test; if <2y, reduce test set)
4. **Feature NaNs:** Dataset must have <5% NaN in critical columns after Stage 2 cleaning

### Development Workflow

#### Incremental Phases (See ML_AGENT_ROADMAP.md)
1. **Phase 1 (Week 1):** Foundation - environment, state/action spaces, random policy testing
2. **Phase 2 (Week 2):** Training - PPO agent, hyperparameter tuning, convergence checks
3. **Phase 3 (Week 3):** Evaluation - backtesting, metrics, baseline comparisons, plots
4. **Phase 4 (Week 4):** Deployment - inference script, daily integration, fallback logic

#### Verification Checkpoints
- **After Phase 1:** `test_env_reset.py` passes, portfolio value tracks correctly for random policy
- **After Phase 2:** Agent loss curves downward, portfolio value trends positive over time
- **After Phase 3:** Test Sharpe ≥ baseline, max drawdown < 30%, consistent across runs
- **After Phase 4:** Daily inference produces valid weights [0, 1], sums to 1, logs complete

#### Branch Strategy
- Work on `ml_agent` branch throughout all 4 phases
- Commit after each phase with clear message: "phase: 1. foundation - env & state space"
- Merge to main only after Phase 3 passes all metrics

---

## Critical Caveats

### Stage 3 (ml_agent branch)
- ML agent code lives in `src/agent/` (see comprehensive guide above in "Stage 3: ML Agent Architecture & Development Guide")
- Detailed roadmap with 4 phases (foundation, training, eval, deploy) is in `ML_AGENT_ROADMAP.md`
- **Dependencies required:** `torch==2.3.0`, `stable-baselines3==2.4.0`, `gymnasium==0.29.0`, `scikit-learn>=1.5.0` (add to requirements.txt)
- **Data dependency:** Stage 2 must be complete; `data/processed/ml_dataset.parquet` must exist and pass validation
- **Temporal splits required:** Never split by rows; use date ranges (train/val/test by date order)
- **Scaler management:** Fit StandardScaler on train set only; save and load for inference (prevent lookahead)

### `fill_cagr_columns()` is Commented Out
Line 143 in `build_ml_dataset.py` has the call commented out:
```python
#ticker_df = fill_cagr_columns(ticker_df)
```
This means the dataset will be missing `cagr_earnings_5y_final` and `cagr_revenue_5y_final` columns at runtime. Uncomment when CAGR backfilling is needed.
### Stage 2: No Lookahead Bias (Temporal Merge)
The merge of prices + fundamentals uses `pd.merge_asof(..., direction='backward')` to ensure no price date ever sees a *future* fundamental (e.g., a price from 2026-04-01 gets the Q1 2026 fundamental dated 2026-03-31, not a later quarter). This is critical for valid backtesting.

**Check:** After merge, verify `fundamental_date <= price_date` for all rows.

### Stage 2 Feature Engineering (Phase 3)
All feature engineering happens in Stage 2, not deferred to Stage 3 (RL Agent). This includes:
- **Technical indicators:** RSI, moving averages, volatility, momentum, drawdown
- **Fundamental ratios:** P/E, P/B, ROE, leverage, growth CAGR
- **Macro-adjusted:** real return, excess return, rate environment
- **CAGR backfill:** `fill_cagr_columns()` (currently commented out at line 143 in `build_ml_dataset.py`)

Uncomment line 143 to enable CAGR backfilling. See `BUILD_DATASET_ROADMAP.md` Phase 3 for full feature list and Phase 4a for NaN handling strategy.

### FIIs Are Deferred
Pipeline collects **stocks only** (prices, fundamentals, dividends). Real Estate Investment Trusts (FIIs) are a separate asset class with different fundamentals (NAV/P-VP vs earnings/revenue) and distributions (monthly vs irregular). Will add if Phase 3 RL agent scope expands to mixed-asset allocation.

### BolsAI API Key Handling
- Stored in `.env` (copied from `.env.example`)
- `.env` is gitignored (never commit your key)
- Loaded by `config.load_env()` (stdlib parser, no `python-dotenv` dependency)
- Required for `src/data_collection/pipeline.py` and API validator tests

### Data Collection Pipeline Structure
- All collection lives in `src/data_collection/` with source-agnostic architecture
- **Backfill:** BolsAI API (paid, ~€0.10 per 1K calls) for one-time historical collection (2000–present)
- **Incremental updates:** yfinance (free) for quarterly refreshes; replaces BolsAI with 99% cost savings
- **Source switching:** `config.DATA_SOURCE` dict allows per-data-type fallback (e.g., if yfinance breaks, flip `DATA_SOURCE["prices"]="bolsai"` and retry)
- **Staged approach:** Prototype with 3–10 tickers first (BolsAI backfill), validate against yfinance, unlock full-scale, then use `--mode update` for routine refreshes
- **Checkpointing:** Pipeline resumes from `data/checkpoints/{mode}/` on interrupt (idempotent per mode: `prototype`, `full_scale`, `update`)
- **Logging:** All collector activity goes to `data/logs/collection-YYYYMMDD-HHMMSS.log`
- **BolsAI API caps (probed):** prices `limit<=5000` (date-window paginated), fundamentals `limit<=88` (use 80)
- **yfinance coverage:** prices/dividends have full history (back to 2000); fundamentals have ~4–6 quarters (sufficient for quarterly refresh)
- **BCB series:** selic=11 (daily rate), cdi=12, ipca=433. NOT 432 (that's the annual meta target)
- **Benchmark ticker:** BOVA11 (iShares Bovespa ETF, IBOV index proxy) collected automatically; prices only (no fundamentals/dividends, it's an ETF)
- **Company info:** BolsAI-only (CVM regulatory metadata, rarely changes); refresh manually via `--mode full_scale` when new IPOs appear

### Relative Paths
New pipeline uses absolute paths via `Path(__file__).resolve().parents[N]`. Run all commands from project root.

## Data on Disk

**Raw data** (tracked in git):
- Three prototype tickers + benchmark: PETR4, VALE3, WEGE3, BOVA11
- Location: `data/raw/prices/`, `data/raw/fundamentals/`, `data/raw/macro/`, `data/raw/company_info/`, `data/raw/dividends/`
- Status: prices current (2026-06-30, via yfinance `--mode update`), fundamentals current (2026-03-31 from BolsAI backfill), macro current (2026-06-30), dividends current

**Pipeline state** (NOT tracked in git):
- Checkpoints: `data/checkpoints/prototype/` and `data/checkpoints/full_scale/` (resume state per collector)
- Logs: `data/logs/collection-*.log` (timestamped collection runs)

**Processed dataset** (created on first `build_ml_dataset.py` run):
- Location: `data/processed/ml_dataset.parquet`
- One row per ticker + date (daily prices merged with quarterly fundamentals)

## Technology Stack

- **Python:** 3.10+ (uses `list[str]`, `dict | None` syntax)
- **Data:** pandas, numpy, pyarrow (parquet)
- **APIs:** BolsAI REST (direct `httpx`, backfill only), BCB SGS (direct requests, macro only), `yfinance` (production: incremental price/fundamental/dividend updates)
- **Config:** `python-dotenv` (load `.env` for API keys, BolsAI only; yfinance requires no key)
- **Logging:** Python built-in `logging` module (file + console)
- **Viz:** Plotly (existing `financial_view.py`)
- **ML/RL:** `torch==2.3.0`, `stable-baselines3==2.4.0`, `gymnasium==0.29.0` (Stage 3 only)
- **No test framework:** tests are standalone `python script.py` invocations (no pytest)

## Data Collection Modules (Stage 1)

Pipeline in `src/data_collection/` with flat-function architecture, supporting both BolsAI (backfill) and yfinance (incremental updates):

| Module | Purpose |
|--------|---------|
| `config.py` | Shared config (tickers, API keys, paths, retry limits, `DATA_SOURCE` dict for per-type source switching) |
| `client.py` | HTTP wrapper (retries, backoff, logging); `make_client()`, `get_json()` — BolsAI only |
| `checkpoint.py` | Resume state tracking (JSON per collector) |
| `validate.py` | Data quality gates (schemas, ranges, continuity); returns `ValidationResult` |
| `collectors.py` | BolsAI collectors: `collect_macro()`, `collect_prices()`, `collect_fundamentals()`, `collect_company_info()`, `collect_dividends()` |
| `yf_collectors.py` | yfinance collectors: `collect_prices_yf()`, `collect_fundamentals_yf()`, `collect_dividends_yf()` — used for `--mode update` |
| `pipeline.py` | Orchestration: `_collect()` dispatcher routes to BolsAI or yfinance per `DATA_SOURCE` config; supports `--mode update` |

**Helper:** `_merge_save()` in collectors.py — idempotent append + dedup + validate + write (shared by all collectors, source-agnostic)
