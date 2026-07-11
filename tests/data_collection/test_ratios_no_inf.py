"""
test_ratios_no_inf.py
======================
Verifies _compute_ratios() never returns a literal inf (e.g. net_revenue=0
for a pre-revenue/holding-company ticker must yield NaN, not inf, so raw
fundamentals parquet stays clean).

Usage:
    python tests/data_collection/test_ratios_no_inf.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.yf_collectors import _compute_ratios


def test_zero_denominator_yields_nan():
    out = _compute_ratios({
        "net_income": 100.0, "equity": 50.0, "net_revenue": 0.0,
        "total_assets": 200.0, "total_debt": 30.0, "ebitda": 10.0, "ebit": 5.0,
        "cash": 5.0, "current_assets": 20.0, "current_liabilities": 10.0,
        "shares_outstanding": 1000.0, "close_price": 1.0, "cost_of_revenue": 0.0,
    })
    for key in ("net_margin", "ebitda_margin", "p_sr", "ebit_margin"):
        assert math.isnan(out[key]), f"{key} should be NaN, got {out[key]}"
    assert not any(isinstance(v, float) and math.isinf(v) for v in out.values()), \
        "no field should ever be literal inf"

    print("OK: zero-denominator ratios come back as NaN, never inf")


if __name__ == "__main__":
    test_zero_denominator_yields_nan()
