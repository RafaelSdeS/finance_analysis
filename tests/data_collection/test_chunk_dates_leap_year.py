"""
test_chunk_dates_leap_year.py
==============================
_chunk_dates used raw datetime(s.year + years, s.month, s.day) to compute
each window's end, which raises ValueError whenever `s` is Feb 29 and
`s.year + years` isn't a leap year -- true every time for years=10 (adding
10 always shifts year%4 by 2, so the target year is never a multiple of 4).
Reachable via collect_macro's incremental path: BCB's daily selic/cdi series
can have a checkpoint last_date of Feb 28 in a leap year, making the next
start date Feb 29.

Usage:
    python tests/data_collection/test_chunk_dates_leap_year.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.storage import _chunk_dates


def test_chunk_dates_handles_leap_day_start():
    windows = list(_chunk_dates("2024-02-29", "2040-01-01", 10))

    assert windows, "must yield at least one window, not raise"
    first_start, first_end = windows[0]
    assert first_start == "2024-02-29"
    # 2024 + 10 = 2034, not a leap year -> clamped to Feb 28, not a crash.
    assert first_end == "2034-02-28", f"expected clamped 2034-02-28, got {first_end}"
    print("OK: _chunk_dates clamps a Feb-29 start + 10y to Feb 28 instead of raising")


if __name__ == "__main__":
    test_chunk_dates_handles_leap_day_start()
