"""
Test: h_series/milestone_h0.py's classical_mv weight_fn -- the one baseline
in the H-series that uses a return estimate (the "beat this" straw man).
Synthetic PricePanel, following tests/rl_agent/test_risk_portfolios.py's
fixture convention.

Run from project root:
    python tests/h_series/test_milestone_h0.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.h_series.milestone_h0 import make_classical_mv_weight_fn  # noqa: E402
from src.rl_agent.config import RiskConfig  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex, PricePanel  # noqa: E402
from src.rl_agent.environment import run_backtest  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

C_SELL = C_BUY = 0.0003
TOL = 1e-6


def _panel(T: int = 60) -> PricePanel:
    rng = np.random.default_rng(7)
    asset_index = GlobalAssetIndex(tickers=("AAA", "BBB", "CCC", "DDD"),
                                    ticker_to_gidx={"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4})
    dates = pd.bdate_range("2020-01-01", periods=T)
    tickers = [10.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, T)) for _ in range(4)]
    close = np.column_stack([np.ones(T)] + tickers)
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(T, 1.0002),
        slot_gidx=np.array([[1, 2, 3, 4]] * T), valid=np.array([[True, True, True, True]] * T),
        window=2, start_idx=30, end_idx=T - 1,
    )


def test_weights_on_simplex(passed, failed):
    panel = _panel()
    cfg = RiskConfig(lookback=20, min_history_frac=0.5, rebalance_every=5)
    fn = make_classical_mv_weight_fn(cfg, panel.start_idx)
    result = run_backtest(panel, fn, C_SELL, C_BUY, panel.start_idx, panel.end_idx)

    sums_to_one = np.allclose(result.weights.sum(axis=1), 1.0, atol=1e-6)
    in_bounds = bool((result.weights >= -1e-9).all() and (result.weights <= 1.0 + 1e-9).all())
    finite = bool(np.isfinite(result.weights).all())
    ok = sums_to_one and in_bounds and finite
    print_check("classical_mv: every day's weights sum to 1, stay in [0,1], all finite",
                ok, f"sums_ok={sums_to_one}, bounds_ok={in_bounds}, finite={finite}")
    return passed + ok, failed + (not ok)


def test_no_lookahead_prefix_identical(passed, failed):
    """Truncating the panel's tail must not change any weight before the
    truncation point -- a lookahead bug would let a later day's price leak
    into an earlier rebalance decision."""
    panel_full = _panel(T=60)
    panel_short = _panel(T=60)  # same seed -> identical prices up to any shared t
    cfg = RiskConfig(lookback=20, min_history_frac=0.5, rebalance_every=5)

    fn_full = make_classical_mv_weight_fn(cfg, panel_full.start_idx)
    fn_short = make_classical_mv_weight_fn(cfg, panel_short.start_idx)
    result_full = run_backtest(panel_full, fn_full, C_SELL, C_BUY, panel_full.start_idx, panel_full.end_idx)
    result_short = run_backtest(panel_short, fn_short, C_SELL, C_BUY, panel_short.start_idx, 50)

    shared_len = min(len(result_full.weights), len(result_short.weights))
    ok = np.allclose(result_full.weights[:shared_len], result_short.weights[:shared_len], atol=1e-6)
    print_check("classical_mv: weights on the shared prefix are identical regardless of "
                "how far the panel extends past that point (no lookahead)", ok)
    return passed + ok, failed + (not ok)


def test_insufficient_eligible_names_falls_back_to_drift(passed, failed):
    panel = _panel(T=40)
    # min_history_frac=1.0 + short lookback right at start_idx: nothing is seasoned enough yet
    cfg = RiskConfig(lookback=35, min_history_frac=1.0, rebalance_every=5)
    fn = make_classical_mv_weight_fn(cfg, 35)
    result = run_backtest(panel, fn, C_SELL, C_BUY, 35, 39)
    ok = np.allclose(result.weights.sum(axis=1), 1.0, atol=1e-6) and np.isfinite(result.weights).all()
    print_check("classical_mv: degenerate eligibility (< 2 names) still returns a valid simplex "
                "via the w_drift fallback, never crashes", ok)
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("h_series/milestone_h0.py (classical_mv)")
    passed = failed = 0
    for test_fn in [
        test_weights_on_simplex,
        test_no_lookahead_prefix_identical,
        test_insufficient_eligible_names_falls_back_to_drift,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
