#!/usr/bin/env python3
"""
run_online_backtest: regression test for the day-indexing bug fixed 2026-07-09.

Bug: the main loop treated one env.step() call as covering exactly one day
(counter named day_idx, tqdm unit="day", baselines queried via act_fn(None,
day_idx)), but env.step() actually advances by rebalance_interval_days (21)
per call. This silently under-populated results by ~21x (24 rows instead of
~500 for a 501-day test span) and mis-indexed every baseline policy's date
lookup -- the exact same off-by-interval pattern found and fixed earlier in
tests/agent/test_excess_sharpe.py's stub model, except this time in
production code. Confirmed at the time: reported "annualized_return" values
of 588%/2420%/6260% for equal_weight/market_cap/inv_vol.

This test exercises the REAL production path (needs a trained model and
built tensors on disk) with retrain_every_days set far beyond the test
span, isolating the day-count fix from the (separately-tested) retraining
logic. Skips gracefully if no trained model exists yet.

Run from project root: python tests/agent/test_online_backtest_daily_granularity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.config import DEFAULT_CONFIG
from src.agent.env import PortfolioEnv
from src.agent.rolling_eval import run_online_backtest


def main() -> None:
    model_path = DEFAULT_CONFIG.model_dir / "agent_best.zip"
    if not model_path.exists():
        print(f"SKIP: no trained model at {model_path} — run trainer.py first to exercise this test")
        return

    expected_days = len(PortfolioEnv(DEFAULT_CONFIG, date_range="test").dates) - 1

    df, metrics = run_online_backtest(
        DEFAULT_CONFIG,
        retrain_every_days=10**9,  # far beyond the test span: isolates day-counting from retraining
        retrain_timesteps=1,
    )

    print(f"Test span days: {expected_days}, online_results rows: {len(df)}")
    assert len(df) == expected_days, (
        f"online backtest produced {len(df)} rows but the test span has {expected_days} days -- "
        f"this is the exact symptom of the day-indexing bug (env.step() consumes "
        f"rebalance_interval_days per call, not 1)"
    )
    print("✓ row count matches the real test-span day count (not ~1/21st of it)")

    date_span_days = (df["date"].max() - df["date"].min()).days
    assert date_span_days > expected_days * 0.9, (
        f"date column only spans {date_span_days} calendar days for {len(df)} rows -- "
        f"looks like dates aren't advancing per row"
    )
    print(f"✓ date column spans {date_span_days} calendar days across {len(df)} rows")

    # Baselines must reproduce sane, non-exploded annualized returns (not 100s of percent --
    # the exact symptom when ~500 real days get compressed into ~24 "annualization periods")
    for name in ("equal_weight", "market_cap", "inv_vol"):
        ann_ret = metrics[name]["annualized_return"]
        assert abs(ann_ret) < 2.0, (
            f"{name} annualized_return={ann_ret:.4f} is absurd (>200%/yr) -- "
            f"likely the day-compression bug reappeared"
        )
    print("✓ baseline annualized returns are sane (no 500%+ artifacts from day compression)")

    print("\nALL ONLINE-BACKTEST DAILY-GRANULARITY TESTS PASSED ✓")


if __name__ == "__main__":
    main()
