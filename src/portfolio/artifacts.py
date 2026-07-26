"""
artifacts.py -- persist backtest runs to disk (plan Phase V.0a/b). Getting
three deflated-Sharpe variants on 2026-07-26 cost three full walk-forward
LightGBM retrains, because nothing survived past the in-memory run (the
dashboard only writes Plotly HTML, not the underlying curves). Save once,
re-analyze many times.

Also the append-only trial log (V.0b): every run appends one row (config +
key metrics) to trials.csv, so `n_trials` for deflated_sharpe_ratio is a
COUNTED fact -- len(pd.read_csv(TRIAL_LOG_PATH)) -- instead of a hand
estimate re-derived from reading this document's tables.
"""

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BACKTEST_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "backtests"
TRIAL_LOG_PATH = BACKTEST_DIR / "trials.csv"


def _config_hash(config: dict) -> str:
    """Short, stable hash of the config dict -- names the run directory so
    identical configs are trivially recognizable, without forcing the
    caller to invent a run name."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def save_run(config: dict, **series_and_frames) -> Path:
    """Persist one backtest run: `config` is every param that defines it
    (top_n, horizon_td, top_frac, hold_frac, use_exposure, window, ...);
    `**series_and_frames` is any number of named Series/DataFrames (equity
    curves, rebalance logs, the cdi series...). Returns the run directory.
    Pickled as one bundle (not parquet) because rebalance logs carry a
    dict-valued `weights` column parquet can't round-trip cleanly -- these
    are internal research artifacts, not a public data contract.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{_config_hash(config)}"
    path = BACKTEST_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=1, default=str))
    with open(path / "data.pkl", "wb") as f:
        pickle.dump(series_and_frames, f)
    return path


def load_run(path) -> dict:
    """Inverse of save_run: {"config": {...}, **series_and_frames}."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    with open(path / "data.pkl", "rb") as f:
        data = pickle.load(f)
    return {"config": config, **data}


def latest_run(name_contains: str = "") -> Path:
    """Most recent run directory, optionally filtered by a substring match
    on the directory name (e.g. a config hash) -- lets a re-analysis script
    default to "whatever I just ran" instead of requiring an explicit path."""
    runs = sorted(p for p in BACKTEST_DIR.glob("*") if p.is_dir() and name_contains in p.name)
    if not runs:
        raise FileNotFoundError(
            f"no saved runs in {BACKTEST_DIR}" + (f" matching {name_contains!r}" if name_contains else ""))
    return runs[-1]


def append_trial_log(config: dict, metrics: dict) -> None:
    """One row per run -- config + key metrics -- appended to trials.csv
    (plan V.0b). Read `len(pd.read_csv(TRIAL_LOG_PATH))` for the honest
    n_trials count next time a deflated Sharpe needs correcting, instead of
    re-counting sweep rows in PORTFOLIO_IMPROVEMENT_PLAN.md by hand."""
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **config, **metrics}
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(TRIAL_LOG_PATH, mode="a", header=not TRIAL_LOG_PATH.exists(), index=False)


if __name__ == "__main__":
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    orig_dir, orig_log = BACKTEST_DIR, TRIAL_LOG_PATH
    import src.portfolio.artifacts as _self
    _self.BACKTEST_DIR = tmp
    _self.TRIAL_LOG_PATH = tmp / "trials.csv"

    curve = pd.Series([1.0, 1.02, 1.01], index=pd.date_range("2020-01-01", periods=3))
    log = pd.DataFrame({"date": curve.index, "weights": [{"A": 0.5, "B": 0.5}] * 3})
    config = {"top_frac": 0.6, "hold_frac": 0.75, "window": "full"}

    run_path = _self.save_run(config, alpha_curve=curve, alpha_log=log)
    loaded = _self.load_run(run_path)
    assert loaded["config"] == config, "config didn't round-trip"
    assert loaded["alpha_curve"].equals(curve), "curve didn't round-trip"
    assert loaded["alpha_log"]["weights"].iloc[0] == {"A": 0.5, "B": 0.5}, "dict column didn't round-trip"
    assert _self.latest_run() == run_path, "latest_run didn't find the just-saved run"

    _self.append_trial_log(config, {"sharpe": 0.8})
    _self.append_trial_log({**config, "top_frac": 0.5}, {"sharpe": 0.6})
    trials = pd.read_csv(_self.TRIAL_LOG_PATH)
    assert len(trials) == 2, "trial log didn't accumulate rows across calls"
    assert set(trials["top_frac"]) == {0.6, 0.5}, "trial log didn't capture per-row config"

    shutil.rmtree(tmp)
    print("artifacts self-check OK | save/load round-trips config + curves + dict columns; "
          "trial log accumulates")
