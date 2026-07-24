"""
test_labels.py -- synthetic checks for forward_excess_return (proposal
§2.2): hand-computed expected values, so the vectorized shift/cumsum
implementation is checked against independent arithmetic, not itself.

Fast group (synthetic only). Run: python tests/portfolio/test_labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio.labels import forward_excess_return  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402

H = 3


def _panel():
    dates = pd.bdate_range("2020-01-01", periods=10)
    adj_close = [100.0 + i for i in range(10)]  # +1/day
    cdi = [0.01] * 10  # constant 0.01%/day
    degraded = [0] * 10
    degraded[2] = 1  # flag one row to check masking
    df = pd.DataFrame({
        "ticker": "AAA", "trade_date": dates, "adj_close": adj_close,
        "cdi": cdi, "adj_close_precision_degraded": degraded,
    })
    return df


def main():
    print_header("test_labels")
    passed = failed = 0

    df = _panel()
    label = forward_excess_return(df, horizon_td=H)

    # Hand-computed expected values, independent of the implementation's own
    # shift/cumsum machinery.
    adj = df["adj_close"].to_numpy()
    cdi_gross = 1 + df["cdi"].to_numpy() / 100
    for i in range(len(df) - H):
        expected_fwd_ret = adj[i + H] / adj[i] - 1
        expected_fwd_cdi = np.prod(cdi_gross[i + 1: i + 1 + H]) - 1
        expected = expected_fwd_ret - expected_fwd_cdi
        if i == 2:  # masked by adj_close_precision_degraded
            ok = pd.isna(label.iloc[i])
            print_check(f"row {i} masked (precision_degraded=1)", ok)
        else:
            ok = np.isclose(label.iloc[i], expected, atol=1e-10)
            print_check(f"row {i} matches hand-computed label", ok,
                        f"got {label.iloc[i]:.8f}, expected {expected:.8f}")
        passed, failed = passed + ok, failed + (not ok)

    trailing_ok = label.iloc[-H:].isna().all()
    print_check(f"last {H} rows are NaN (no forward window, never imputed)", bool(trailing_ok))
    passed, failed = passed + trailing_ok, failed + (not trailing_ok)

    # Row order independence: shuffle the input, confirm output realigns correctly.
    shuffled = df.sample(frac=1, random_state=0)
    shuffled_label = forward_excess_return(shuffled, horizon_td=H)
    realign_ok = np.allclose(
        shuffled_label.reindex(df.index).to_numpy(dtype=float),
        label.to_numpy(dtype=float), equal_nan=True,
    )
    print_check("output aligns to input index regardless of input row order", bool(realign_ok))
    passed, failed = passed + realign_ok, failed + (not realign_ok)

    # Non-positive adj_close at the base row must not divide-by-zero into a
    # bogus finite label -- it should come back NaN.
    df2 = _panel()
    df2.loc[0, "adj_close"] = 0.0
    label2 = forward_excess_return(df2, horizon_td=H)
    nonpositive_ok = pd.isna(label2.iloc[0])
    print_check("non-positive adj_close at the base row -> NaN label", bool(nonpositive_ok))
    passed, failed = passed + nonpositive_ok, failed + (not nonpositive_ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
