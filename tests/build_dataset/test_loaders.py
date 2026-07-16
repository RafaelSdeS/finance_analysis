#!/usr/bin/env python3
"""
load_dividends()'s implausible-value_per_share sanity ceiling.

Run from project root: python tests/build_dataset/test_loaders.py
or: pytest tests/build_dataset/test_loaders.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_dataset import loaders  # noqa: E402


def test_load_dividends_drops_implausible_value_per_share(tmp_path, monkeypatch) -> None:
    """A real BRL per-share dividend is at most low tens even in extreme
    cases. Regression test for the PDGR3 raw-data bug (vendor unit/labeling
    error, value_per_share in the hundreds of millions -- inflated
    div_yield_12m to 154,600%, docs/TOP50_UNIVERSE_ML_READINESS_AUDIT.md
    §1.4): rows above the sanity ceiling must be dropped, real rows kept.
    """
    monkeypatch.setattr(loaders, "DIVIDENDS_DIR", tmp_path)

    good = pd.DataFrame({
        "ticker": ["AAAA3", "AAAA3"],
        "ex_date": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        "value_per_share": [0.5, 1.2],
    })
    good.to_parquet(tmp_path / "AAAA3.parquet")

    bad = pd.DataFrame({
        "ticker": ["PDGR3"],
        "ex_date": pd.to_datetime(["2012-05-09"]),
        "value_per_share": [168_557_520.0],
    })
    bad.to_parquet(tmp_path / "PDGR3.parquet")

    result = loaders.load_dividends()

    assert len(result) == 2
    assert set(result["ticker"]) == {"AAAA3"}
    assert sorted(result["value_per_share"]) == [0.5, 1.2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
