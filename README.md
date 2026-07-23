# Finance Analysis: Brazilian-Equity Dataset Pipeline

A data pipeline for Brazilian equities: collect raw prices/fundamentals/macro data, then build a
single ML-ready parquet with derived features (technical, fundamental, macro), no lookahead bias.

No model or agent is implemented yet — this repo currently covers data collection and dataset
build only. Modeling work (RL agent, or otherwise) starts fresh from here.

## Pipeline

Two stages: collect → build dataset.

### Stage 1: Raw Data Collection

```bash
# Backfill (one-time historical via BolsAI, 2000–present); resumes from checkpoints, idempotent
python -m src.data_collection.pipeline --mode full_scale

# Quarterly incremental refresh (free yfinance, no key)
python -m src.data_collection.pipeline --mode update
```

### Stage 2: Build ML Dataset

Merges prices + quarterly fundamentals + company metadata into a single machine-learning-ready parquet:
```bash
python -m src.build_dataset.build_ml_dataset
```
Output: `data/processed/ml_dataset.parquet`

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add BOLSAI_API_KEY=sk_...  (backfill only; .env is gitignored)
```

## Current Data

**Raw (git-tracked):** ~293 tickers + benchmark BOVA11, one parquet per ticker in `data/raw/`.
**Macro:** SELIC, CDI, IPCA daily rates from BCB SGS.
**Data currency:** Prices/macro current to 2026-06-30; fundamentals to 2026-03-31. Refreshed via yfinance quarterly incremental updates.

## Visualization

Standalone Plotly chart: BBAS3 nominal price vs inflation-adjusted vs SELIC comparison.
```bash
python src/visualizations/financial_view.py
```

## See Also

- `CLAUDE.md` — development guide (run commands, architecture, caveats)
- `docs/README.md` — docs index
