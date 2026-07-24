"""
test_risk.py -- checks shrinkage_cov against the textbook case for why it's
needed (n_assets > n_samples: raw sample covariance is singular by
construction), then validates conditioning vs raw sample cov on a REAL
trailing window from the real point-in-time liquid universe (proposal §2.5
"Done when").

Needs data/processed/ml_dataset.parquet -- data group.
Run: python tests/portfolio/test_risk.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.build_dataset.paths import OUTPUT_PATH  # noqa: E402
from src.portfolio import universe  # noqa: E402
from src.portfolio.risk import add_cash_row_col, condition_number, is_psd, shrinkage_cov  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_degenerate_case():
    """10 assets, 5 daily observations: a sample covariance from n < p
    observations is rank-deficient (rank <= n-1 = 4 < p = 10) by
    construction -- singular, not just poorly conditioned. Shrinkage must
    still return a full-rank, PSD, well-conditioned matrix."""
    passed = failed = 0
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=5)
    tickers = [f"T{i}" for i in range(10)]
    returns = pd.DataFrame(rng.normal(0, 0.01, (5, 10)), index=dates, columns=tickers)

    raw_cov = returns.cov()
    raw_rank = np.linalg.matrix_rank(raw_cov.to_numpy())
    raw_singular_ok = raw_rank < 10
    print_check("raw sample cov is rank-deficient with n_obs < n_assets (as expected)",
                bool(raw_singular_ok), f"rank={raw_rank}/10")
    passed, failed = passed + raw_singular_ok, failed + (not raw_singular_ok)

    shrunk = shrinkage_cov(returns)
    shrunk_psd_ok = is_psd(shrunk)
    print_check("shrinkage_cov is PSD even in the degenerate n<p regime", bool(shrunk_psd_ok))
    passed, failed = passed + shrunk_psd_ok, failed + (not shrunk_psd_ok)

    shrunk_cond = condition_number(shrunk)
    well_conditioned_ok = np.isfinite(shrunk_cond) and shrunk_cond < 1e6
    print_check("shrinkage_cov is well-conditioned (finite, not huge) despite n<p",
                bool(well_conditioned_ok), f"condition number={shrunk_cond:.2e}")
    passed, failed = passed + well_conditioned_ok, failed + (not well_conditioned_ok)

    return passed, failed


def test_empty_window():
    """A 0-row returns window (e.g. a rebalance right at the very start of
    history, before any return can even be computed) must return empty,
    like the 0-column case already does -- not crash inside LedoitWolf."""
    passed = failed = 0
    empty = pd.DataFrame(columns=["A", "B", "C"], dtype=float)
    result = shrinkage_cov(empty)
    ok = result.empty
    print_check("shrinkage_cov on a 0-row window returns empty instead of crashing", bool(ok))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_cash_row_col():
    passed = failed = 0
    sigma = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    with_cash = add_cash_row_col(sigma)
    shape_ok = with_cash.shape == (3, 3) and "cash" in with_cash.columns and "cash" in with_cash.index
    print_check("add_cash_row_col adds exactly one row and one column", bool(shape_ok))
    passed, failed = passed + shape_ok, failed + (not shape_ok)

    zero_ok = (with_cash.loc["cash"] == 0).all() and (with_cash["cash"] == 0).all()
    print_check("cash row and column are exactly zero", bool(zero_ok))
    passed, failed = passed + zero_ok, failed + (not zero_ok)

    psd_ok = is_psd(with_cash)
    print_check("cash-augmented Σ is still PSD (a zero block can't break it)", bool(psd_ok))
    passed, failed = passed + psd_ok, failed + (not psd_ok)

    return passed, failed


def test_real_window():
    passed = failed = 0
    df = pd.read_parquet(OUTPUT_PATH, columns=["ticker", "trade_date", "adj_close", "traded_amount"])
    membership = universe.liquid_universe(df[["ticker", "trade_date", "traded_amount"]], top_n=50)
    reb_dates = universe.rebalance_dates(membership)
    as_of = reb_dates[-2]  # a recent, well-populated rebalance date
    current_universe = universe.universe_at(membership, as_of)

    price_wide = df[df["ticker"].isin(current_universe)].pivot(
        index="trade_date", columns="ticker", values="adj_close"
    ).ffill()
    window = price_wide.loc[:as_of].tail(252).pct_change().iloc[1:]

    shrunk = shrinkage_cov(window)
    raw = window.dropna(axis=1, how="any").cov()

    real_data_ok = shrunk.shape[0] > 10
    print_check(f"real window ({as_of.date()}) yields a usable number of assets", bool(real_data_ok),
                f"{shrunk.shape[0]} assets, {len(window)} trading days")
    passed, failed = passed + real_data_ok, failed + (not real_data_ok)

    psd_ok = is_psd(shrunk)
    print_check("shrinkage_cov is PSD on the real window", bool(psd_ok))
    passed, failed = passed + psd_ok, failed + (not psd_ok)

    raw_cond = condition_number(raw)
    shrunk_cond = condition_number(shrunk)
    improves_ok = shrunk_cond < raw_cond
    print_check("shrinkage improves conditioning vs raw sample cov on the real window",
                bool(improves_ok), f"raw={raw_cond:.2e}, shrunk={shrunk_cond:.2e}")
    passed, failed = passed + improves_ok, failed + (not improves_ok)

    return passed, failed


def main():
    print_header("test_risk")
    p1, f1 = test_degenerate_case()
    p2, f2 = test_empty_window()
    p3, f3 = test_cash_row_col()
    p4, f4 = test_real_window()
    passed, failed = p1 + p2 + p3 + p4, f1 + f2 + f3 + f4
    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
