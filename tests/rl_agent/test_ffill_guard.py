"""Lookahead guard (EIIE_IMPROVEMENT_PLAN.md, Stage 0 pre-flight / E2 guard):
technical feature channels must be ffill-only through load_price_panel's
_dense -- a NaN warm-up prefix must stay NaN (window_tensor zeroes it),
never be bfilled with a later, now-defined value (a lookahead leak on
unmasked training rows). Prices, by contrast, ARE bfilled (paper Sec. 3.3
pre-listing flat fill; always masked downstream, so harmless).

Synthetic parquet files + monkeypatched path constants; standalone:
    python tests/rl_agent/test_ffill_guard.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import src.rl_agent.data as data_mod
from src.rl_agent.config import DataConfig


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ffill_guard_"))
    dates = pd.bdate_range("2015-01-05", periods=10)

    # AAAA3 trades all 10 days; its technical (return_1m) has a 4-day NaN
    # warm-up prefix ON ACTIVE DAYS. BBBB3 only starts trading on day 3
    # (price NaN before -> the bfill path prices legitimately take).
    rows = []
    for i, d in enumerate(dates):
        rows.append({"ticker": "AAAA3", "trade_date": d, "adj_close": 10.0 + i,
                     "adj_high": 10.5 + i, "adj_low": 9.5 + i,
                     "return_1m": np.nan if i < 4 else 0.1 + i / 100})
        if i >= 3:
            rows.append({"ticker": "BBBB3", "trade_date": d, "adj_close": 20.0 + i,
                         "adj_high": 20.5 + i, "adj_low": 19.5 + i, "return_1m": 0.2})
    pd.DataFrame(rows).to_parquet(tmp / "dataset.parquet")

    pd.DataFrame({"period_id": [1, 1], "ticker": ["AAAA3", "BBBB3"],
                  "start": [dates[0]] * 2, "end": [dates[-1]] * 2}).to_parquet(tmp / "membership.parquet")
    pd.DataFrame({"reference_date": dates, "cdi": 0.04}).to_parquet(tmp / "cdi.parquet")
    pd.DataFrame({"trade_date": dates, "adj_close": 100.0}).to_parquet(tmp / "bova11.parquet")

    data_mod.DATASET_PATH = tmp / "dataset.parquet"
    data_mod.MEMBERSHIP_PATH = tmp / "membership.parquet"
    data_mod.CDI_PATH = tmp / "cdi.parquet"
    data_mod.BOVA11_PATH = tmp / "bova11.parquet"

    cfg = DataConfig(window=3, features=("close", "high", "low", "return_1m"),
                     window_start=str(dates[0].date()), window_end=str(dates[-1].date()))
    panel = data_mod.load_price_panel(cfg, n_slots=2)

    ga = panel.asset_index.ticker_to_gidx["AAAA3"]
    gb = panel.asset_index.ticker_to_gidx["BBBB3"]

    # 1. Technical: NaN warm-up prefix preserved, NOT bfilled
    r1m = panel.extra["return_1m"]
    assert np.isnan(r1m[:4, ga]).all(), \
        f"LOOKAHEAD LEAK: technical warm-up prefix was bfilled: {r1m[:4, ga]}"
    # 2. First non-NaN equals the first ACTUAL data point (0.1 + 4/100), not a later value
    assert r1m[4, ga] == 0.1 + 4 / 100, f"first defined value wrong: {r1m[4, ga]}"
    # 3. Prices: pre-listing prefix IS bfilled flat to the first real price
    assert (panel.close[:3, gb] == 20.0 + 3).all(), \
        f"price pre-listing prefix not bfilled: {panel.close[:3, gb]}"
    # 4. window_tensor zero-fills the (unmasked) technical warm-up, never NaN
    X = panel.window_tensor(4, cfg.features)
    assert np.isfinite(X).all(), "NaN leaked through window_tensor"

    print("PASS: technicals ffill-only (warm-up NaN preserved), prices bfilled, no NaN reaches the network")


if __name__ == "__main__":
    main()
