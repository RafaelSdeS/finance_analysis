# Finance Analysis: Brazilian-Equity Dataset Pipeline

A data pipeline for Brazilian equities (+ a US-equity expansion in progress): collect raw
prices/fundamentals/macro data, then build a single ML-ready parquet with derived features
(technical, fundamental, macro), no lookahead bias. On top of that, `src/portfolio/` is an
active, BR-only portfolio-construction research effort (LightGBM alpha forecaster + risk model
+ cost-aware optimizer, evaluated through a walk-forward backtest) — not a shipped strategy; see
`CLAUDE.md` for its current, honest status.

## Pipeline

Three stages: collect → build dataset → (research) portfolio construction.

### Stage 1: Raw Data Collection

```bash
# Backfill (one-time historical via BolsAI, 2000–present); resumes from checkpoints, idempotent
python -m src.data_collection.br.pipeline --mode full_scale

# Quarterly incremental refresh (free yfinance, no key)
python -m src.data_collection.br.pipeline --mode update
```

### Stage 2: Build ML Dataset

Merges prices + quarterly fundamentals + company metadata into a single machine-learning-ready parquet:
```bash
python -m src.build_dataset.build_ml_dataset
```
Output: `data/processed/ml_dataset.parquet`

### Stage 3: Portfolio Construction (active research, BR only)

```bash
python -m src.portfolio.run_baseline          # equal-weight + BOVA11 + 100%-CDI baselines
python -m src.portfolio.run_full_backtest      # full alpha -> risk model -> optimizer pipeline
```
See `CLAUDE.md`'s Stage 3 section for the current research status before citing any number from a run.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add BOLSAI_API_KEY=sk_...  (backfill only; .env is gitignored)
```

## Current Data

**BR, raw (git-tracked):** ~293 tickers + benchmark BOVA11, one parquet per ticker in `data/raw/br/`.
**Macro:** SELIC, CDI, IPCA daily rates from BCB SGS.
**Data currency:** Prices/macro current to 2026-06-30; fundamentals to 2026-03-31. Refreshed via yfinance quarterly incremental updates.
**US, raw (gitignored, rebuildable):** `data/raw/us/` — prices (yfinance), fundamentals (SEC EDGAR), macro (FRED). Validated at a top-500-by-market-cap scope; full ~10,432-ticker scale-up in progress, see `CLAUDE.md`.

## Visualization

Standalone Plotly chart: BBAS3 nominal price vs inflation-adjusted vs SELIC comparison.
```bash
python src/visualizations/financial_view.py
```

## See Also

- `CLAUDE.md` — development guide (run commands, architecture, caveats)
- `docs/README.md` — docs index
