"""
Test: data.py's load_price_panel() against the real on-disk dataset
(docs/eiie_agent/EIIE_AGENT_PLAN.md Phase 2). Needs data/processed/ml_dataset.parquet,
data/processed/top50_universe_membership.parquet, data/raw/macro/cdi.parquet,
data/raw/prices/BOVA11.parquet -- DATA group, not FAST.

Run from project root:
    python tests/rl_agent/test_data_integration.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.config import DataConfig  # noqa: E402
from src.rl_agent.data import load_price_panel  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def main():
    print_header("test_data_integration")
    passed = failed = 0

    panel = load_price_panel(DataConfig(), n_slots=50)

    ok = panel.n_global == 172
    print_check("N_global = 171 union tickers + cash = 172", ok, f"got {panel.n_global}")
    passed, failed = passed + ok, failed + (not ok)

    ok = panel.dates[panel.start_idx] >= panel.dates[0]
    print_check("start_idx resolves to a real calendar date", ok, str(panel.dates[panel.start_idx].date()))
    passed, failed = passed + ok, failed + (not ok)

    ok = str(panel.dates[panel.start_idx].date()) <= "2011-01-31"
    print_check("experiment window starts at/near 2011-01-31", ok, str(panel.dates[panel.start_idx].date()))
    passed, failed = passed + ok, failed + (not ok)

    in_window = panel.valid[panel.start_idx:panel.end_idx + 1]
    ok = bool(np.all(in_window.sum(axis=1) == 50))
    print_check("every in-window trading day has exactly 50 active members", ok,
                f"min={in_window.sum(axis=1).min()}, max={in_window.sum(axis=1).max()}")
    passed, failed = passed + ok, failed + (not ok)

    ok = not np.any(np.isnan(panel.close)) and not np.any(np.isnan(panel.high)) and not np.any(np.isnan(panel.low))
    print_check("no NaN anywhere in dense price arrays (flat-fill covers all gaps)", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = bool(np.all(panel.cdi_factor > 1.0)) and bool(np.all(panel.cdi_factor < 1.01))
    print_check("cash factor (1 + cdi/100) in a sane per-day range", ok,
                f"range=[{panel.cdi_factor.min():.6f}, {panel.cdi_factor.max():.6f}]")
    passed, failed = passed + ok, failed + (not ok)

    ok = panel.bova11_close is not None and not np.all(np.isnan(panel.bova11_close[panel.start_idx:panel.end_idx + 1]))
    print_check("BOVA11 benchmark series loaded and covers the experiment window", ok)
    passed, failed = passed + ok, failed + (not ok)

    t = panel.start_idx + panel.window
    X = panel.window_tensor(t)
    ok = X.shape == (3, 50, panel.window) and not np.any(np.isnan(X)) and not np.any(np.isinf(X))
    print_check("window_tensor at a real in-window date: correct shape, finite", ok, str(X.shape))
    passed, failed = passed + ok, failed + (not ok)

    y = panel.price_relative(t)
    ok = y.shape == (172,) and not np.any(np.isnan(y)) and not np.any(np.isinf(y)) and np.all(y > 0)
    print_check("price_relative at a real in-window date: finite and positive", ok)
    passed, failed = passed + ok, failed + (not ok)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
