"""Shared styled output helpers for test scripts (colors, boxes, glyphs).

TOLERANCES CATALOG -- every cross-vendor / fuzzy-match tolerance in the test
suite, in one place, so a future loosening is easy to spot and re-audit
(each one is individually justified inline where it's defined; this is the
index, not the reasoning):

  tests/data_collection/validate_vs_yfinance.py
    TOLERANCE_PCT = 25   -- BolsAI vs yfinance, prices AND fundamentals.
                            (Previously the fundamentals docstring claimed a
                            separate 70% -- stale, didn't match the code;
                            fixed to state the real shared value. If
                            fundamentals genuinely needs a wider band than
                            prices, that's a deliberate change to make here,
                            not something to silently drift back to.)
  tests/data_collection/test_cvm_statements.py
    TOLERANCE = 0.15 (15%) -- CVM-derived vs BolsAI-derived equity, cross-source.
  tests/data_collection/validate_us_vs_vendor.py
    TOLERANCE_PCT = 25   -- ours (SEC xbrl tier) vs yfinance, fundamentals AND prices
                            (Alpha Vantage). Matches BR's TOLERANCE_PCT above.
    SEAM_RATIO_MAX = 5, SEAM_RATE_CEILING = 0.05 -- tier-seam continuity check
                            (fundamentals_tier boundary jump); not cross-vendor,
                            same category as test_collect_delisted.py's anchor
                            tolerance below. Measured baseline 2026-08-05: 4.4%
                            flagged -- real unit-scaling bugs in the ex27/tenq/
                            item6 SEC parsers, see the script's own docstring.
  tests/build_dataset/test_final_dataset.py
    0.8 (80%) -- CAGR NaN "explained" coverage threshold (negative base year or
                 <20 quarters of history) before the coverage check counts as
                 acceptable rather than a regression.
  tests/data_collection/test_collect_delisted.py
    ANCHOR_TOLERANCE_DAYS = 7 -- not cross-vendor, but the same category:
                                 how far a live-verified last-trade-date
                                 anchor may drift before the test flags it.
  tests/build_dataset/test_unit_scale_invariants.py
    BAND = 0.01          -- NOT cross-vendor. These are algebraic identities on
                            one frame (market_cap == close*shares), so the band
                            is float-noise headroom, not vendor disagreement.
                            Measured 2026-08-23: 0 violations over 13M rows,
                            both markets. Was 0.10 + a per-ticker median until
                            then, which is blind to a per-QUARTER anchor error.
    SCALE_BANDS          -- per-column percent-vs-fraction bands, ~a decade
                            either side of measured, sized to catch a 100x
                            convention flip and nothing narrower.
  tests/build_dataset/test_us_final_dataset.py
    CAGR_*_EXPLAINED_FLOOR = 0.85 / 0.68 -- US analogue of test_final_dataset's
                            0.8 / 0.75. Measured 0.900 / 0.731.
    REF_REGRESSION_TICKER_CEILING = 0.01 -- share of tickers whose merged
                            reference_date ever moves backwards. BR measures
                            0.00%; RED on landing for US (50.09%) on purpose.
    STALE_FUNDAMENTAL_RATE_CEILING = 0.01 -- share of has_fundamentals rows
                            whose fiscal period is >400d old. BR 0.11%; RED on
                            landing for US (9.47%), same root cause.
    FROZEN_PL_RATE_CEILING = 0.01 -- measured 0/84,568.
  tests/data_collection/test_cvm_statements.py
    TTM flow cliff CEILING = 0.012 -- adjacent-quarter net_revenue ratios
                            outside [1/3, 3]. Measured 227/27,847 = 0.82%.
  tests/data_collection/test_br_data_quality.py
    _SELIC_CDI_MAX_SPREAD = 0.02 -- ~2.6x the observed max (0.00764).

RATE CEILINGS ONLY EARN THEIR KEEP WHILE THE BACKLOG THEY NAME EXISTS. A
ceiling left at its landing value after the backlog clears is pure slack: it
stays green through a regression all the way back up to the old number. Two
were re-measured 2026-08-23 -- _JUMP_RATE_CEILING (0.12 vs. 11.22% measured,
unchanged, still earning it) and _INF_RATE_CEILING (0.20 vs. 0.00% measured,
ratcheted to zero).

Last reviewed as a set: 2026-08-23. Add new tolerances here when introduced
elsewhere.
"""

import os
import sys
from shutil import get_terminal_size

import pandas as pd

COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
_CODES = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "dim": "2", "bold": "1"}


def c(color: str, text: str) -> str:
    """Wrap text in ANSI color code (or pass through if COLOR disabled)."""
    return f"\033[{_CODES[color]}m{text}\033[0m" if COLOR else text


WIDTH = max(60, min(get_terminal_size((100, 24)).columns, 100))

PASS, FAIL, SKIP = "PASSED", "FAILED", "SKIPPED"
_GLYPH = {PASS: c("green", "✓"), FAIL: c("red", "✗"), SKIP: c("yellow", "⚠")}


def print_header(title: str) -> None:
    """Print a bold box-drawn header."""
    pad = (WIDTH - len(title) - 2) // 2
    print(c("bold", "═" * pad + f" {title} " + "═" * (WIDTH - pad - len(title) - 2)))


def print_check(label: str, ok: bool, detail: str = "") -> None:
    """Print a single check result with glyph and color."""
    status = PASS if ok else FAIL
    glyph = _GLYPH[status]
    suffix = f" {detail}" if detail else ""
    print(f"  {glyph} {label}{suffix}")


def print_section_start(title: str) -> None:
    """Print a section header (left-aligned box-draw)."""
    header = f"┌─ {title} "
    print(c("bold", header + "─" * max(0, WIDTH - len(header))))


def print_section_end(passed: int, failed: int, skipped: int = 0) -> None:
    """Print section footer with counts."""
    counts = [f"{passed} passed"] if passed else []
    if failed:
        counts.append(f"{failed} failed")
    if skipped:
        counts.append(f"{skipped} skipped")
    footer = ", ".join(counts)
    print(f"└─ {footer}")


def print_separator() -> None:
    """Print a simple separator line."""
    print("─" * WIDTH)


def numeric_columns(df) -> list:
    """Numeric column names, dtype-only — df.select_dtypes() copies every
    matching column's data just to build the list, which OOMs on wide frames."""
    return [c for c, dt in df.dtypes.items() if pd.api.types.is_numeric_dtype(dt)]
