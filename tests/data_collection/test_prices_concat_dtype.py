"""
test_prices_concat_dtype.py
============================
Verifies collect_prices_yf's num_trades column is float64 (nan), not object
(None), so _merge_save's pd.concat with existing BolsAI-backfilled data never
raises the "empty or all-NA entries" FutureWarning.

Usage:
    python tests/data_collection/test_prices_concat_dtype.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_num_trades_concat_no_warning():
    df_old = pd.DataFrame({"ticker": ["PETR4"], "trade_date": pd.to_datetime(["2026-01-01"]),
                            "num_trades": np.array([12345.0])})
    df_new = pd.DataFrame({"ticker": ["PETR4"], "trade_date": pd.to_datetime(["2026-01-02"]),
                            "num_trades": np.nan})

    assert df_new["num_trades"].dtype == np.float64, \
        f"expected float64, got {df_new['num_trades'].dtype}"

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pd.concat([df_old, df_new], ignore_index=True)

    print("OK: num_trades stays float64 and concat raises no FutureWarning")


if __name__ == "__main__":
    test_num_trades_concat_no_warning()
