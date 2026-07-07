"""
Shared UI helpers for the explorer pages: sidebar filters, cached data access,
and split-span shading. Pages import from here so filtering behaves identically
everywhere.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.explorer import data_access as da  # noqa: E402

SPLIT_COLORS = {"train": "green", "val": "orange", "test": "red"}


# Cached wrappers — da.* stays streamlit-free, caching lives at the UI boundary.
stage_df = st.cache_data(lambda stage, tickers, date_range: da.STAGES[stage](list(tickers), date_range))
env_tensors_df = st.cache_data(
    lambda tickers, date_range, cutoff=None: da.load_env_tensors(list(tickers), date_range, scaler_cutoff=cutoff)
)
backtest_df = st.cache_data(da.load_backtest)
metrics_dict = st.cache_data(da.load_metrics)
bova11_df = st.cache_data(da.load_bova11)
training_logs_df = st.cache_data(da.load_training_logs)
spans_df = st.cache_data(da.split_spans)
sector_series = st.cache_data(lambda: da.sector_map().set_index("ticker")["sector"])
tickers_list = st.cache_data(da.all_tickers)


def sidebar_filters(default_tickers: list[str] = ["PETR4"]) -> dict:
    """Standard sidebar: tickers, date range, window/split filter, cache reset.

    Returns dict(tickers, date_range, spans, window_choice, split_choice).
    """
    with st.sidebar:
        st.header("Filters")
        if st.button("Reload data (clear cache)"):
            st.cache_data.clear()

        tickers = st.multiselect("Tickers", tickers_list(), default=default_tickers)
        date_from = st.date_input("From", pd.Timestamp("2015-01-01"))
        date_to = st.date_input("To", pd.Timestamp.today())

        spans = spans_df()
        window_ids = sorted(spans["window_id"].unique())
        window_choice = st.selectbox("Window", ["(all)"] + [str(w) for w in window_ids])
        split_choice = st.selectbox("Split", ["(all)", "train", "val", "test"])

    return {
        "tickers": tickers,
        "date_range": (str(date_from), str(date_to)),
        "spans": spans,
        "window_choice": window_choice,
        "split_choice": split_choice,
    }


def apply_split_filter(df: pd.DataFrame, filters: dict, date_col: str = "date") -> pd.DataFrame:
    """Keep only rows inside the selected window/split spans."""
    window_choice, split_choice, spans = filters["window_choice"], filters["split_choice"], filters["spans"]
    if window_choice == "(all)" and split_choice == "(all)":
        return df
    sub = spans
    if window_choice != "(all)":
        sub = sub[sub["window_id"] == int(window_choice)]
    if split_choice != "(all)":
        sub = sub[sub["split"] == split_choice]
    if len(sub) == 0 or len(df) == 0:
        return df.iloc[0:0]
    masks = [(df[date_col] >= row.start) & (df[date_col] <= row.end) for row in sub.itertuples()]
    return df[np.logical_or.reduce(masks)]


def add_split_shading(fig, spans: pd.DataFrame) -> None:
    """Shade train/val/test spans on a time-series figure (green/orange/red)."""
    for row in spans.itertuples():
        fig.add_vrect(x0=row.start, x1=row.end, fillcolor=SPLIT_COLORS[row.split],
                      opacity=0.05, line_width=0)
