# Finance Analysis: RL Portfolio Allocation Agent

A reinforcement-learning system to learn dynamic portfolio allocation for Brazilian equities. Not price prediction—sequential decision-making on capital allocation under uncertainty, targeting returns above IBOV benchmarks.

## What This Is

Given a fixed monthly capital contribution (R$1000, inflation-adjusted), the model learns to allocate capital across equities, ETFs, and risk-free assets (SELIC/CDI). Each month it decides a portfolio allocation vector that maximizes long-term returns while managing volatility, drawdowns, and trading costs.

## Pipeline

Three stages: collect → build dataset → train (RL agent, `src/rl_agent/` on this branch).

### Stage 1: Raw Data Collection

**Macro data** (SELIC, CDI, IPCA from BCB SGS API):
```bash
python "src/1. collect_raw_data/bolsai_raw_data_collector.py" --start 1990-01-01 --end 2026-01-01
```

**Stock prices and fundamentals** (from BolsAI API; requires `--api-key`):
```bash
python "src/1. collect_raw_data/fetch_company_info.py" --api-key YOUR_API_KEY
```

### Stage 2: Build ML Dataset

Merges prices + quarterly fundamentals + company metadata into a single machine-learning-ready parquet:
```bash
python src/build_dataset/build_ml_dataset.py
```
Output: `data/processed/ml_dataset.parquet`

### Stage 3: Train RL Agent

**Iteration 1** (`src/rl_agent/`, this branch): a faithful reproduction of Jiang, Xu & Liang's EIIE
framework (`docs/papers/deep_reinforcement_learning_framework_financial_portfolio_management.pdf.pdf`)
— price-only features, the top-50 dynamic quarterly universe, CDI-accruing cash, B3's 0.03%
transaction cost — over the 2011–2026 pre-built top-50 window. Design, approved deviations from the
paper, and phase-by-phase status: `docs/eiie_agent/EIIE_AGENT_PLAN.md`.

```bash
python tests/run_all.py --group fast                                             # rl_agent unit/integration tests
python -m src.rl_agent.experiment --config configs/eiie_baseline.json --dry-run   # data + sanity checks, no training
python -m src.rl_agent.experiment --config configs/eiie_baseline.json            # full run -> experiments/{run_id}/report.html
```

A separate, earlier PPO agent (masked 279-ticker universe, equal-weight baseline Sharpe 0.71) lives
on the `ml_agent` branch; see CLAUDE.md there for its training commands.

## Setup

```bash
pip install -r requirements.txt
```

**Note:** BolsAI API key is required for Stage 1. Pass via `--api-key` flag.

## Current Data

**Prototype (main branch):** PETR4, VALE3, WEGE3 + BOVA11 (IBOV proxy ETF).  
**Macro:** SELIC, CDI, IPCA daily rates (1990–2026-06-30).  
**Data currency:** Prices/macro current to 2026-06-30; fundamentals to 2026-03-31. Refreshed via yfinance quarterly incremental updates.

## Visualization

Standalone Plotly chart: BBAS3 nominal price vs inflation-adjusted vs SELIC comparison.
```bash
python src/visualizations/financial_view.py
```

## See Also

- `CLAUDE.md` — development guide (run commands, architecture, caveats)
- `docs/specification.txt` — full RL system design (features, objective, constraints, expected behavior)
- `docs/TODO.md` — work roadmap
- `docs/STAGE1_DATA_COLLECTION.md`, `docs/STAGE2_DATASET_BUILD.md`, `docs/STAGE3_ML_AGENT.md` — how each stage actually works, with formulas and citations
- `docs/RESEARCH_REFERENCES.md` — papers behind the design choices
- `docs/eiie_agent/EIIE_AGENT_PLAN.md` — iteration-1 RL agent design, approved paper deviations, phase status (this branch)
- `docs/ML_AGENT_ROADMAP.md` — phase-by-phase RL agent build plan (`ml_agent` branch)
