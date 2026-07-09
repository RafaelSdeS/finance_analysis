# Runbook — full pipeline, data gathering → training

All commands run from the project root. Details: `CLAUDE.md`, `docs/STAGE3_ML_AGENT.md`.

## 0. Setup (once)

```bash
pip install -r requirements.txt
pip install torch stable-baselines3 gymnasium scikit-learn   # Stage 3 only
cp .env.example .env    # add BOLSAI_API_KEY=sk_...  (needed for backfill only)
```

## 1. Collect raw data (Stage 1)

```bash
# One-time historical backfill via BolsAI (paid API, resumes from checkpoints, idempotent)
python -m src.data_collection.pipeline --mode full_scale --dry-run   # preview ticker list first
python -m src.data_collection.pipeline --mode full_scale

# OR: quarterly incremental refresh via free yfinance (if backfill already done)
python -m src.data_collection.pipeline --mode update

# Validate against yfinance
python tests/data_collection/validate_vs_yfinance.py
```

## 2. Build ML dataset (Stage 2)

```bash
python -m src.build_dataset.build_ml_dataset          # → data/processed/ml_dataset.parquet
python tests/build_dataset/test_final_dataset.py      # schema, lookahead, NaN checks
python tests/agent/verify_dataset_for_training.py     # gates V1–V7 (must pass before Stage 3)
```

## 3. Train the agent (Stage 3)

```bash
# Data prep (rerun whenever the dataset changes)
python src/agent/feature_engineering.py
python -m src.agent.data_pipeline --universe-size 50  # → data/processed/agent_tensors.npz

# Optional sanity check: does the feature set have exploitable signal?
python -m src.agent.ranker_baseline

# Smoke test the env + a tiny training run
python tests/agent/test_env_basic.py
python -m src.agent.trainer --train-years 2 --test-years 1 --timesteps 2048 --universe-size 50 --bc-pretrain

# Full training (8 anchored rolling windows, 1M timesteps each, ~20 min total; production model = most recent window)
python -m src.agent.trainer --universe-size 50 --bc-pretrain
```

## 4. Evaluate & use

```bash
python -m src.agent.evaluate --model artifacts/models/agent_best.zip   # backtest vs baselines
python -m src.agent.rolling_eval --mode online_backtest --resume       # online retraining backtest
python -m src.agent.run_allocation --date 2026-06-29 --format csv      # daily weights
```

## Tests

```bash
python tests/run_all.py --group fast   # pure-code unit tests (CI)
python tests/run_all.py --group data   # needs raw + processed data
ruff check .
```
