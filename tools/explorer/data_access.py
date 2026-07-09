"""
Data-access layer for the pipeline explorer dashboard.

Everything that touches disk lives here: stage loaders, split spans, sector
map, env tensors + scalers, backtest results/metrics, training logs, and the
programmatic anomaly checks. No streamlit imports — this module is import-safe
from scripts and tests.

Self-check (no browser):
    python tools/explorer/data_access.py --check
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.config import DEFAULT_CONFIG, generate_windows, window_to_config  # noqa: E402

PRICES_DIR = ROOT / "data/raw/prices"
FUNDAMENTALS_DIR = ROOT / "data/raw/fundamentals"
COMPANY_INFO_PATH = ROOT / "data/raw/company_info/company_info.parquet"
PROCESSED_PATH = ROOT / "data/processed/ml_dataset.parquet"
TRAINING_PATH = ROOT / "data/processed/ml_dataset_training.parquet"
TENSORS_PATH = ROOT / "data/processed/agent_tensors.npz"
SCALER_PATH = ROOT / "artifacts/models/feature_scaler.pkl"
BACKTEST_DIR = ROOT / "artifacts/backtest"
LOGS_DIR = ROOT / "artifacts/logs"
ROLLING_EVAL_PATH = ROOT / "artifacts/models/rolling_eval_results.json"

BACKTEST_SOURCES = {
    "Last window (test)": "results.parquet",
    "Walk-forward (stitched)": "walkforward_results.parquet",
    "Online rollout": "online_results.parquet",
}
_METRICS_FILES = {
    "results.parquet": "metrics.json",
    "walkforward_results.parquet": "walkforward_metrics.json",
    "online_results.parquet": "online_metrics.json",
}


# --------------------------------------------------------------------------- basics

def all_tickers() -> list[str]:
    return sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))


def sector_map() -> pd.DataFrame:
    return pd.read_parquet(COMPANY_INFO_PATH, columns=["ticker", "sector"])


def split_spans() -> pd.DataFrame:
    """Anchored rolling windows as (window_id, split, start, end) rows, for shading/filtering."""
    windows = generate_windows(
        DEFAULT_CONFIG.dataset_start, DEFAULT_CONFIG.dataset_end,
        DEFAULT_CONFIG.window_train_years, DEFAULT_CONFIG.window_test_years,
    )
    rows = []
    for w in windows:
        cfg = window_to_config(w, DEFAULT_CONFIG)
        rows += [
            {"window_id": w.window_id, "split": "train", "start": cfg.train_start, "end": cfg.train_end},
            {"window_id": w.window_id, "split": "val", "start": cfg.val_start, "end": cfg.val_end},
            {"window_id": w.window_id, "split": "test", "start": cfg.test_start, "end": cfg.test_end},
        ]
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def _clip(df: pd.DataFrame, date_col: str, date_range: tuple | None) -> pd.DataFrame:
    if date_range is None:
        return df
    lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    return df[(df[date_col] >= lo) & (df[date_col] <= hi)]


# --------------------------------------------------------------------------- stage loaders

def load_raw_prices(tickers: list[str], date_range: tuple | None = None) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        f = PRICES_DIR / f"{t}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df["ticker"] = t
        df["date"] = pd.to_datetime(df["trade_date"])
        dfs.append(df.drop(columns=["trade_date"]))
    if not dfs:
        return pd.DataFrame()
    return _clip(pd.concat(dfs, ignore_index=True), "date", date_range)


def load_raw_fundamentals(tickers: list[str], date_range: tuple | None = None) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        f = FUNDAMENTALS_DIR / f"{t}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df["ticker"] = t
        df["date"] = pd.to_datetime(df["reference_date"])
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return _clip(pd.concat(dfs, ignore_index=True), "date", date_range)


def _load_parquet_slice(path: Path, tickers: list[str], date_range: tuple | None) -> pd.DataFrame:
    filters = [("ticker", "in", list(tickers))] if tickers else None
    df = pd.read_parquet(path, filters=filters)
    df["date"] = pd.to_datetime(df["trade_date"])
    return _clip(df, "date", date_range)


def load_processed(tickers: list[str], date_range: tuple | None = None) -> pd.DataFrame:
    return _load_parquet_slice(PROCESSED_PATH, tickers, date_range)


def load_training(tickers: list[str], date_range: tuple | None = None) -> pd.DataFrame:
    return _load_parquet_slice(TRAINING_PATH, tickers, date_range)


def load_tensors() -> dict:
    z = np.load(TENSORS_PATH, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_scalers() -> dict:
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def load_env_tensors(tickers: list[str], date_range: tuple | None = None,
                     scaler_cutoff: str | None = None) -> pd.DataFrame:
    """Un-pivot the dense [dates,tickers,features] tensor back to a tidy frame for selected tickers."""
    tensors = load_tensors()
    all_tick = tensors["tickers"]
    idx = [i for i, t in enumerate(all_tick) if t in tickers]
    if not idx:
        return pd.DataFrame()
    dates = pd.to_datetime(tensors["dates"])
    feat_names = DEFAULT_CONFIG.state_features
    feats = tensors["features"][:, idx, :]
    mask = tensors["mask"][:, idx]

    if scaler_cutoff is not None:
        scaler = load_scalers()[scaler_cutoff]
        feats = (feats - scaler.mean_) / scaler.scale_

    rows = []
    for j, t_idx in enumerate(idx):
        d = pd.DataFrame(feats[:, j, :], columns=feat_names)
        d["date"] = dates
        d["ticker"] = all_tick[t_idx]
        d["active"] = mask[:, j]
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["active"]]
    return _clip(out, "date", date_range)


STAGES = {
    "raw_prices": load_raw_prices,
    "raw_fundamentals": load_raw_fundamentals,
    "processed": load_processed,
    "training": load_training,
}


# --------------------------------------------------------------------------- model / backtest artifacts

def load_backtest(source_file: str, date_range: tuple | None = None) -> pd.DataFrame:
    df = pd.read_parquet(BACKTEST_DIR / source_file)
    df["date"] = pd.to_datetime(df["date"])
    return _clip(df, "date", date_range)


def load_metrics(source_file: str) -> dict:
    with open(BACKTEST_DIR / _METRICS_FILES[source_file]) as f:
        return json.load(f)


def load_bova11(date_range: tuple | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PRICES_DIR / "BOVA11.parquet")
    df["date"] = pd.to_datetime(df["trade_date"])
    return _clip(df, "date", date_range)


def load_rolling_eval() -> dict | None:
    if not ROLLING_EVAL_PATH.exists():
        return None
    with open(ROLLING_EVAL_PATH) as f:
        return json.load(f)


def load_training_logs() -> pd.DataFrame:
    """All trainer JSONL eval records, one row per checkpoint, tagged by run.

    Layout: artifacts/logs/agent/runs/<run_id>/{tag}.jsonl, tag = 'agent' or 'window_N'.
    Records: {timesteps, val_sharpe, val_max_drawdown, val_final_value, timestamp}.
    """
    rows = []
    for f in sorted((LOGS_DIR / "agent" / "runs").glob("*/*.jsonl")):
        run_id = f.parent.name
        tag = f.stem
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["tag"] = tag
            rec["run"] = f"{tag} @ {run_id}"
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- anomaly checks

def check_duplicates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    dup = df[df.duplicated(["ticker", date_col], keep=False)]
    return dup.assign(check="duplicate_row")


def check_date_gaps(df: pd.DataFrame, date_col: str = "date", max_gap_days: int = 10) -> pd.DataFrame:
    findings = []
    for t, g in df.sort_values(date_col).groupby("ticker"):
        gaps = g[date_col].diff().dt.days
        bad = g[gaps > max_gap_days]
        for _, row in bad.iterrows():
            findings.append({"ticker": t, "date": row[date_col], "check": "date_gap",
                              "value": f">{max_gap_days}d gap"})
    return pd.DataFrame(findings)


def check_return_spikes(df: pd.DataFrame, col: str = "log_return", threshold: float = 0.35) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    bad = df[df[col].abs() > threshold]
    return bad[["ticker", "date", col]].rename(columns={col: "value"}).assign(check="return_spike")


def check_stale_prices(df: pd.DataFrame, price_col: str = "close", run_len: int = 5) -> pd.DataFrame:
    if price_col not in df.columns or "volume" not in df.columns:
        return pd.DataFrame()
    findings = []
    for t, g in df.sort_values("date").groupby("ticker"):
        same = (g[price_col].diff() == 0) & (g["volume"] > 0)
        run = same.groupby((~same).cumsum()).cumsum()
        hits = g[run >= run_len]
        for _, row in hits.iterrows():
            findings.append({"ticker": t, "date": row["date"], "check": "stale_price", "value": row[price_col]})
    return pd.DataFrame(findings)


def check_nan_critical(df: pd.DataFrame, critical_cols: list[str]) -> pd.DataFrame:
    present = [c for c in critical_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    bad = df[df[present].isnull().any(axis=1)]
    return bad.assign(check="nan_critical")


def check_zero_variance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    findings = []
    for col in feature_cols:
        if col in df.columns and df[col].nunique(dropna=True) <= 1:
            findings.append({"ticker": "(all)", "date": pd.NaT, "check": "zero_variance", "value": col})
    return pd.DataFrame(findings)


def check_outliers_zscore(df: pd.DataFrame, feature_cols: list[str], threshold: float = 8.0) -> pd.DataFrame:
    findings = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or pd.isna(mad):
            continue
        z = 0.6745 * (s - med) / mad
        bad = df[z.abs() > threshold]
        for _, row in bad.iterrows():
            findings.append({"ticker": row.get("ticker", "?"), "date": row.get("date", pd.NaT),
                              "check": f"outlier:{col}", "value": row[col]})
    return pd.DataFrame(findings)


def check_lookahead(df: pd.DataFrame) -> pd.DataFrame:
    if "days_since_fundamental" not in df.columns:
        return pd.DataFrame()
    bad = df[df["days_since_fundamental"] < 0]
    return bad.assign(check="lookahead")


def run_all_checks(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    feature_cols = feature_cols or []
    parts = [
        check_duplicates(df),
        check_date_gaps(df),
        check_return_spikes(df),
        check_stale_prices(df),
        check_nan_critical(df, ["ticker", "date", "close", "volume", "sector"]),
        check_zero_variance(df, feature_cols),
        check_outliers_zscore(df, feature_cols),
        check_lookahead(df),
    ]
    parts = [p for p in parts if len(p) > 0]
    cols = ["ticker", "date", "check", "value"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True)


def tensor_cross_check(n_dates: int = 20, n_tickers: int = 5, seed: int = 0) -> tuple[int, int]:
    """Sample (date,ticker) cells; verify npz features == scaler(parquet pipeline output).

    Returns (checked, mismatches). Catches pivot misalignment between the tidy
    dataset and the dense tensors the agent actually consumes.
    """
    scalers = load_scalers()
    cutoff = sorted(scalers.keys())[0]
    scaler = scalers[cutoff]
    tensors = load_tensors()
    rng = np.random.default_rng(seed)
    sample_t = rng.choice(len(tensors["dates"]), size=min(n_dates, len(tensors["dates"])), replace=False)
    sample_k = rng.choice(len(tensors["tickers"]), size=min(n_tickers, len(tensors["tickers"])), replace=False)
    checked, mismatches = 0, 0
    for ki in sample_k:
        ticker = str(tensors["tickers"][ki])
        tidy = load_env_tensors([ticker], None, scaler_cutoff=cutoff)
        for ti in sample_t:
            if not tensors["mask"][ti, ki]:
                continue
            expected = (tensors["features"][ti, ki, :] - scaler.mean_) / scaler.scale_
            row = tidy[tidy["date"] == pd.Timestamp(tensors["dates"][ti])]
            if len(row) == 0:
                continue
            got = row[DEFAULT_CONFIG.state_features].to_numpy()[0]
            checked += 1
            if not np.allclose(got, expected, equal_nan=True, atol=1e-4):
                mismatches += 1
    return checked, mismatches


# --------------------------------------------------------------------------- self-check

def _selfcheck() -> None:
    tickers = ["PETR4", "VALE3"]
    date_range = ("2020-01-01", "2020-06-30")

    raw = load_raw_prices(tickers, date_range)
    assert len(raw) > 0 and {"ticker", "date", "close"} <= set(raw.columns)

    fund = load_raw_fundamentals(tickers, date_range)
    assert len(fund) > 0 and "ticker" in fund.columns

    proc = load_processed(tickers, date_range)
    assert len(proc) > 0 and "sector" in proc.columns

    train = load_training(tickers, date_range)
    assert len(train) > 0

    env_df = load_env_tensors(tickers, date_range)
    assert len(env_df) > 0 and set(DEFAULT_CONFIG.state_features) <= set(env_df.columns)

    cutoff = sorted(load_scalers().keys())[0]
    env_scaled = load_env_tensors(tickers, date_range, scaler_cutoff=cutoff)
    assert len(env_scaled) > 0

    spans = split_spans()
    assert {"window_id", "split", "start", "end"} <= set(spans.columns)

    findings = run_all_checks(proc, feature_cols=["close", "volume"])
    assert list(findings.columns) == ["ticker", "date", "check", "value"]

    bt = load_backtest("results.parquet")
    assert {"date", "log_return", "value_agent"} <= set(bt.columns)

    m = load_metrics("results.parquet")
    assert "agent" in m and "sharpe" in m["agent"]

    assert len(load_bova11(date_range)) > 0

    logs = load_training_logs()
    if len(logs) > 0:
        assert {"timesteps", "val_sharpe", "run", "tag"} <= set(logs.columns)

    # Pie-input sanity: latest snapshot weights are positive and sum to ~1
    w_cols = [c for c in bt.columns if c.startswith("w_")]
    snap = bt[w_cols].iloc[-1]
    assert (snap >= 0).all() and 0.5 < snap.sum() <= 1.001, f"weights sum {snap.sum()}"

    checked, mismatches = tensor_cross_check(n_dates=5, n_tickers=2)
    assert checked > 0 and mismatches == 0, f"tensor cross-check: {mismatches}/{checked} mismatches"

    print("✓ explorer data_access self-check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selfcheck()
    else:
        print(__doc__)
