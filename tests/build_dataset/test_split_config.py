#!/usr/bin/env python3
"""
Walk-forward split boundary computation (compute_split_dates).

Run from project root: python tests/build_dataset/test_split_config.py
or: pytest tests/build_dataset/test_split_config.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset.manifest import compute_split_dates, iter_fit_windows


def test_split_is_time_ordered_no_overlap() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"trade_date": list(dates) * 3})  # 3 tickers, same calendar

    train_end, val_end = compute_split_dates(df, train_frac=0.7, val_frac=0.15)

    assert train_end < val_end
    assert train_end in dates
    assert val_end in dates


def test_split_robust_to_uneven_ticker_history() -> None:
    # Ticker A has the full 100-day history; ticker B only exists for the back
    # half. A row-count split would be dragged later by A's extra rows; the
    # date-based split must land on the same cutoff dates regardless.
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df_a = pd.DataFrame({"trade_date": dates})
    df_b = pd.DataFrame({"trade_date": dates[50:]})
    df_both = pd.concat([df_a, df_b], ignore_index=True)

    train_end_a, val_end_a = compute_split_dates(df_a, train_frac=0.7, val_frac=0.15)
    train_end_both, val_end_both = compute_split_dates(df_both, train_frac=0.7, val_frac=0.15)

    assert train_end_a == train_end_both
    assert val_end_a == val_end_both


def test_iter_fit_windows_resolves_todays_config_to_one_expanding_window() -> None:
    """Today's split_config.json shape (train_end/val_end) must resolve to a
    single expanding-from-start window ending at train_end -- the seam a
    future rolling/multi-fold config format would extend without touching any
    scaler-fitting code (docs/PER_TICKER_SCALING_PLAN.md §3.5)."""
    windows = iter_fit_windows({"train_end": "2018-07-30", "val_end": "2022-07-26"})

    assert len(windows) == 1
    assert windows[0].fold_id == "full"
    assert windows[0].fit_start is None
    assert windows[0].fit_end == pd.Timestamp("2018-07-30")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
