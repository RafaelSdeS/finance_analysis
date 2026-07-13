"""Shared styled output helpers for test scripts (colors, boxes, glyphs)."""

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
