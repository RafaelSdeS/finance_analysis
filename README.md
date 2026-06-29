# Finance Analysis: RL Portfolio Allocation Agent

A reinforcement-learning system to learn dynamic portfolio allocation for Brazilian equities. Not price prediction—sequential decision-making on capital allocation under uncertainty, targeting returns above IBOV benchmarks.

## What This Is

Given a fixed monthly capital contribution (R$1000, inflation-adjusted), the model learns to allocate capital across equities, ETFs, and risk-free assets (SELIC/CDI). Each month it decides a portfolio allocation vector that maximizes long-term returns while managing volatility, drawdowns, and trading costs.

## Pipeline

Three stages: collect → build dataset → train (RL agent not yet implemented).

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
python "src/2. build_dataset/build_ml_dataset.py
```
Output: `data/processed/ml_dataset.parquet`

### Stage 3: Train RL Agent

Not yet implemented. See `specification.txt` for system design.

## Setup

```bash
pip install -r requirements.txt
```

**Note:** BolsAI API key is required for Stage 1. Pass via `--api-key` flag.

## Current Data

Three tickers: PETR4 (Petrobrás), VALE3 (Vale), WEGE3 (WEG).  
Macro: SELIC, CDI, IPCA daily rates (1990 onward).

## Visualization

Standalone Plotly chart: BBAS3 nominal price vs inflation-adjusted vs SELIC comparison.
```bash
python src/visualizations/financial_view.py
```

## See Also

- `CLAUDE.md` — development guide (run commands, architecture, caveats)
- `specification.txt` — full RL system design (features, objective, constraints, expected behavior)
- `TODO.md` — work roadmap
