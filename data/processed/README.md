# data/processed/

Gitignored — everything here is a local build artifact, not tracked.

- `ml_dataset.parquet`, `ml_dataset.manifest.json`, `split_config.json` — the
  current build, built by this repo. Regenerate anytime with
  `python -m src.build_dataset.build_ml_dataset`.
- `dataset_v{N}/` — immutable snapshots of the same three files, one per build
  whose content actually changed (unchanged reruns don't bump `N`). Cite
  `dataset_v{N}` when referencing exactly which build an experiment used.
- `ml_dataset_training.parquet` (and anything else not listed above) — **not**
  built by this repo. Produced by pipeline code that lives on the `ml_agent`
  branch (`src/agent/data_pipeline.py`), consumed by
  `src/visualizations/rolling_eval_results.ipynb` and `agent_performance.ipynb`.
  Don't edit or regenerate it from here — treat it as foreign input.
