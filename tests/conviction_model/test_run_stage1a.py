"""
Test: conviction_model/run_stage1a.py's universe-construction and development-window
helpers (Phase 2, docs/conviction_model/REVIEW_REMEDIATION_PLAN.md) -- pure logic
(truncate_to_development_window) or a small synthetic parquet fixture
(point_in_time_union_tickers/top150_snapshot_tickers), no dependency on the real
data/processed/top150_universe_membership.parquet. Unlike run_stage1a.py's own main()
(a real training-run orchestrator, out of scope for the fast synthetic-test convention),
these two helpers are pure/data-independent enough to test directly here.

Run from project root:
    python tests/conviction_model/test_run_stage1a.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.run_stage1a import (  # noqa: E402
    point_in_time_union_tickers, top150_snapshot_tickers, truncate_to_development_window,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_truncate_to_development_window_excludes_reserved_span(passed, failed):
    calendar = pd.bdate_range("2010-01-01", periods=1000)
    panel = pd.DataFrame({"ticker": "AAA", "trade_date": calendar})
    truncated, dataset_end, dev_end = truncate_to_development_window(panel, reserved_holdout_years=1.0)

    nothing_past_dev_end = bool((truncated["trade_date"] <= dev_end).all())
    something_was_dropped = len(truncated) < len(panel)
    ends_before_dataset_end = dev_end < dataset_end
    ok = nothing_past_dev_end and something_was_dropped and ends_before_dataset_end
    print_check("truncate_to_development_window: drops every row after dataset_end minus "
                "reserved_holdout_years", ok,
                f"dataset_end={dataset_end.date()}, dev_end={dev_end.date()}, "
                f"kept={len(truncated)}/{len(panel)}")
    return passed + ok, failed + (not ok)


def test_truncate_to_development_window_boundary_is_inclusive(passed, failed):
    # Daily (not business-day) calendar so dev_end lands EXACTLY on a row in the panel --
    # confirms the split is <=dev_end (inclusive), matching split_train_holdout's own
    # "trailing window" convention, not a strictly-before cutoff that would silently drop
    # one extra day.
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="D")
    panel = pd.DataFrame({"ticker": "AAA", "trade_date": dates})
    truncated, dataset_end, dev_end = truncate_to_development_window(panel, reserved_holdout_years=1.0)

    expected_dev_end = pd.Timestamp("2023-01-01")
    boundary_row_kept = bool((truncated["trade_date"] == dev_end).any())
    ok = dev_end == expected_dev_end and boundary_row_kept
    print_check("truncate_to_development_window: the dev_end boundary row itself is kept "
                "(inclusive), not excluded", ok, f"dev_end={dev_end.date()}, "
                f"expected={expected_dev_end.date()}, boundary_row_kept={boundary_row_kept}")
    return passed + ok, failed + (not ok)


def _write_membership_fixture(path):
    """Synthetic top150_universe_membership.parquet-shaped fixture: 3 rebalance
    periods. DELISTED only appears in the EARLIEST period (simulating a name that
    qualified once, then was delisted/dropped out before the latest rebalance) --
    top150_snapshot_tickers (latest period only) must miss it;
    point_in_time_union_tickers (all periods) must include it."""
    rows = [
        {"ticker": "SURVIVOR", "period_id": "p0", "start": pd.Timestamp("2015-01-01")},
        {"ticker": "DELISTED", "period_id": "p0", "start": pd.Timestamp("2015-01-01")},
        {"ticker": "SURVIVOR", "period_id": "p1", "start": pd.Timestamp("2018-01-01")},
        {"ticker": "NEWCOMER", "period_id": "p1", "start": pd.Timestamp("2018-01-01")},
        {"ticker": "SURVIVOR", "period_id": "p2", "start": pd.Timestamp("2021-01-01")},
        {"ticker": "NEWCOMER", "period_id": "p2", "start": pd.Timestamp("2021-01-01")},
    ]
    pd.DataFrame(rows).to_parquet(path)


def test_point_in_time_union_includes_delisted_names(passed, failed):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "membership.parquet"
        _write_membership_fixture(path)
        union = point_in_time_union_tickers(membership_path=path)
    ok = set(union) == {"SURVIVOR", "DELISTED", "NEWCOMER"}
    print_check("point_in_time_union_tickers: includes a name only present in an EARLIER "
                "rebalance period (simulating a delisted/dropped-out ticker)", ok, f"union={union}")
    return passed + ok, failed + (not ok)


def test_snapshot_excludes_delisted_names(passed, failed):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "membership.parquet"
        _write_membership_fixture(path)
        snapshot = top150_snapshot_tickers(membership_path=path)
    ok = set(snapshot) == {"SURVIVOR", "NEWCOMER"} and "DELISTED" not in snapshot
    print_check("top150_snapshot_tickers: excludes a name only present in an EARLIER "
                "rebalance period -- the exact survivorship-bias gap the union fixes",
                ok, f"snapshot={snapshot}")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/run_stage1a.py (universe + development-window helpers)")
    passed = failed = 0
    for test_fn in [
        test_truncate_to_development_window_excludes_reserved_span,
        test_truncate_to_development_window_boundary_is_inclusive,
        test_point_in_time_union_includes_delisted_names,
        test_snapshot_excludes_delisted_names,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
