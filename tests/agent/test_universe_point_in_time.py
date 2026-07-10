"""
Unit tests for point-in-time (PIT) universe selection (synthetic panel, no data files).

DIAGNOSIS_PLAN.md F1: the pre-fix universe selector (`select_universe(cutoff=None)`,
data_pipeline.py's old default) picks top-N tickers by mean market cap over the FULL
sample, including data from after any given window's train_end — a future company's
eventual size can pull it into an early window's tradable universe. These tests first
demonstrate that leak on a synthetic panel (problem), then verify the cutoff-based
selection is truly future-blind (solution).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.data_pipeline import select_universe


def _make_caps_panel() -> pd.DataFrame:
    """3 tickers x 10 years of monthly caps. X starts tiny, becomes huge after year 5
    (an IPO-then-rocket story); Y and Z stay flat mid-size throughout."""
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")  # 10 years, monthly
    rows = []
    for i, d in enumerate(dates):
        year = i // 12
        x_cap = 1.0 if year < 5 else 1000.0  # tiny for 5y, then rockets
        rows.append({"ticker": "X", "trade_date": d, "market_cap": x_cap})
        rows.append({"ticker": "Y", "trade_date": d, "market_cap": 50.0})
        rows.append({"ticker": "Z", "trade_date": d, "market_cap": 40.0})
    return pd.DataFrame(rows)


def main():
    caps = _make_caps_panel()
    early_cutoff = "2003-01-01"  # inside X's "tiny" era, year 3

    print("Test 1 (problem): full-sample selection leaks future information")
    # Full-sample mean cap: X's average is dragged up by its post-year-5 rocket, so a
    # universe_size=2 selection over the WHOLE sample includes X even though X was never
    # actually a top-2 name during the early years.
    full_sample = select_universe(caps, ["X", "Y", "Z"], universe_size=2, cutoff=None)
    assert "X" in full_sample, (
        f"expected full-sample selection to leak X (future rocket) into the top-2, got {full_sample}"
    )
    print(f"  Full-sample top-2: {sorted(full_sample)} (X present — this is the leak)")

    print("\nTest 2 (solution): cutoff-based selection excludes future information")
    # As of the early cutoff, X is still tiny (cap=1.0) vs Y/Z (cap=50/40) — a
    # point-in-time selection must NOT include X.
    pit_selection = select_universe(caps, ["X", "Y", "Z"], universe_size=2, cutoff=early_cutoff)
    assert "X" not in pit_selection, (
        f"PIT selection at cutoff={early_cutoff} must not see X's future size, got {pit_selection}"
    )
    assert pit_selection == {"Y", "Z"}, f"expected {{Y, Z}}, got {pit_selection}"
    print(f"  PIT top-2 @ {early_cutoff}: {sorted(pit_selection)} (X correctly excluded)")

    print("\nTest 3 (solution): future-blindness property")
    # Selecting with a cutoff must be invariant to what happens strictly after that cutoff:
    # truncating the panel to only rows <= cutoff (simulating "the future hasn't happened
    # yet") must produce the identical selection as filtering the full panel by the same
    # cutoff.
    truncated = caps[caps["trade_date"] <= pd.Timestamp(early_cutoff)]
    sel_from_truncated = select_universe(truncated, ["X", "Y", "Z"], universe_size=2, cutoff=early_cutoff)
    sel_from_full = select_universe(caps, ["X", "Y", "Z"], universe_size=2, cutoff=early_cutoff)
    assert sel_from_truncated == sel_from_full, (
        f"cutoff selection must be future-blind: truncated={sel_from_truncated} vs full={sel_from_full}"
    )
    print(f"  Truncated-panel selection == full-panel selection @ cutoff: {sorted(sel_from_full)}")

    print("\nTest 4 (solution): per-window universes differ when caps drift")
    late_cutoff = "2009-01-01"  # year 9, well after X's rocket
    late_selection = select_universe(caps, ["X", "Y", "Z"], universe_size=2, cutoff=late_cutoff)
    assert late_selection != pit_selection, (
        "a later cutoff after a cap-drift event should select a different universe than an "
        "earlier cutoff, else per-window PIT selection is silently reusing one global list"
    )
    assert "X" in late_selection, f"expected X to qualify once it has actually grown, got {late_selection}"
    print(f"  Early @ {early_cutoff}: {sorted(pit_selection)}  |  Late @ {late_cutoff}: {sorted(late_selection)}")

    print("\nTest 5: cutoff selection respects universe_size and ticker_pool")
    small_pool = select_universe(caps, ["Y", "Z"], universe_size=1, cutoff=late_cutoff)
    assert small_pool == {"Y"}, f"expected {{Y}} (largest of the restricted pool), got {small_pool}"
    print(f"  Restricted pool ['Y','Z'], size=1 @ {late_cutoff}: {sorted(small_pool)}")

    print("\n✓ All universe point-in-time tests passed")


if __name__ == "__main__":
    main()
