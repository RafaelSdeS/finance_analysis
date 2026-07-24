#!/usr/bin/env python3
"""
Single entry point for the test suite: runs each test script, captures its
output, and prints a clean, colored, aligned report instead of dumping N
scripts' raw transcripts back to back.

Run from project root:
    python tests/run_all.py                # fast group (pure code, no data files needed)
    python tests/run_all.py --group data    # needs data/raw + a built ml_dataset.parquet
    python tests/run_all.py --group all
"""

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from shutil import get_terminal_size

ROOT = Path(__file__).resolve().parent.parent

# Pure-code tests: synthetic data only, run anywhere (used by CI).
FAST = [
    "tests/test_run_all_report.py",
    "tests/build_dataset/test_features.py",
    "tests/build_dataset/test_history_relative.py",
    "tests/build_dataset/test_merge.py",
    "tests/build_dataset/test_repair.py",
    "tests/build_dataset/test_quality_filters.py",
    "tests/build_dataset/test_clean.py",
    "tests/build_dataset/test_cross_sectional.py",
    "tests/build_dataset/test_compute_features_chunked.py",
    "tests/build_dataset/test_split_config.py",
    "tests/build_dataset/test_dataset_versioning.py",
    "tests/build_dataset/test_scale_features.py",
    "tests/build_dataset/test_company_siblings.py",
    "tests/build_dataset/test_ticker_continuity.py",
    "tests/build_dataset/test_top50_universe.py",
    "tests/build_dataset/test_loaders.py",
    "tests/build_dataset/test_manifest.py",
    "tests/data_collection/test_merge_save_new_rows_only.py",
    "tests/data_collection/test_macro_bare_object.py",
    "tests/data_collection/test_prices_concat_dtype.py",
    "tests/data_collection/test_ratios_no_inf.py",
    "tests/data_collection/test_skip_existing.py",
    "tests/data_collection/test_yf_collectors_demo.py",
    "tests/data_collection/test_pipeline_dispatch.py",
    "tests/data_collection/test_cvm_filing_dates.py",
]

# Needs data/raw/* on disk (git-tracked) and/or a built data/processed/ml_dataset.parquet.
DATA = [
    "tests/build_dataset/test_final_dataset.py",
    "tests/build_dataset/test_top_traded_quality.py",
    "tests/build_dataset/test_universe_integrity.py",
    "tests/data_collection/test_cagr_calculation.py",
    "tests/data_collection/test_blue_chip_tickers.py",
    "tests/data_collection/validate_vs_yfinance.py",
    "tests/data_collection/test_collect_delisted.py",
    "tests/data_collection/test_cvm_statements.py",
]

# Hits a live external vendor (yfinance) rather than only local/synthetic
# data. Vendor flakiness, rate-limits, or yfinance recomputing historical
# adjustments shouldn't fail CI for reasons unrelated to code correctness --
# still runs and its result is printed normally, it just doesn't flip the
# overall exit code.
NON_BLOCKING = {"tests/data_collection/validate_vs_yfinance.py"}

# --- color -------------------------------------------------------------

COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
_CODES = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "dim": "2", "bold": "1"}


def c(color: str, text: str) -> str:
    return f"\033[{_CODES[color]}m{text}\033[0m" if COLOR else text


WIDTH = max(60, min(get_terminal_size((100, 24)).columns, 100))

PASS, FAIL, SKIP = "PASSED", "FAILED", "SKIPPED"
_GLYPH = {PASS: c("green", "✓"), FAIL: c("red", "✗"), SKIP: c("yellow", "⚠")}

_PYTEST_LINE = re.compile(r"^\S+::(\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)\s+\[")


@dataclass
class ScriptResult:
    script: str
    ok: bool
    duration: float
    output: str
    subtests: list[tuple[str, str]] = field(default_factory=list)


def parse_subtests(output: str) -> list[tuple[str, str]]:
    """Best-effort extraction of pytest -v per-test lines; empty for plain scripts."""
    subtests = []
    for line in output.splitlines():
        m = _PYTEST_LINE.match(line)
        if m:
            status = m.group(2)
            subtests.append((m.group(1), FAIL if status == "ERROR" else status))
    return subtests


def run(script: str, coverage: bool = False) -> ScriptResult:
    print(c("cyan", f"▶ running {script}..."))
    start = time.monotonic()
    # --coverage: run under `coverage run --parallel-mode` instead of bare
    # python so each subprocess's .coverage.* file can be combined afterward
    # (each test script is its own subprocess, so plain `coverage run` on
    # run_all.py itself would only ever measure run_all.py, not src/).
    cmd = ([sys.executable, "-m", "coverage", "run", "--parallel-mode", script]
           if coverage else [sys.executable, script])
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        output_lines.append(line)
        print(c("dim", line), end="")
    proc.wait()
    duration = time.monotonic() - start
    output = "".join(output_lines)
    return ScriptResult(
        script=script,
        ok=proc.returncode == 0,
        duration=duration,
        output=output,
        subtests=parse_subtests(output),
    )


def print_section(result: ScriptResult) -> None:
    header = f"┌─ {result.script} "
    print(c("bold", header + "─" * max(0, WIDTH - len(header))))

    if result.subtests:
        for name, status in result.subtests:
            print(f"│ {_GLYPH[status]} {name}")
        passed = sum(1 for _, s in result.subtests if s == PASS)
        skipped = sum(1 for _, s in result.subtests if s == SKIP)
        failed = len(result.subtests) - passed - skipped
        tail = f"{passed} passed"
        if failed:
            tail += f", {failed} failed"
        if skipped:
            tail += f", {skipped} skipped"
        print(f"└─ {tail} · {result.duration:.2f}s")
    else:
        status = PASS if result.ok else FAIL
        print(f"└─ {_GLYPH[status]} {'passed' if result.ok else 'failed'} · {result.duration:.2f}s")

    if not result.ok:
        print(c("dim", "  │ output (last 25 lines):"))
        tail_lines = result.output.strip().splitlines()[-25:]
        for line in tail_lines:
            print(c("red", f"  │ {line}"))
    print()


def print_summary(results: list[ScriptResult]) -> None:
    title = " SUMMARY "
    pad = (WIDTH - len(title)) // 2
    print(c("bold", "═" * pad + title + "═" * (WIDTH - pad - len(title))))

    name_width = max(len(r.script) for r in results)
    for r in results:
        non_blocking = r.script in NON_BLOCKING
        if non_blocking and not r.ok:
            badge, glyph = c("yellow", " WARN "), _GLYPH[SKIP]
        else:
            badge = c("green", " PASS ") if r.ok else c("red", " FAIL ")
            glyph = _GLYPH[PASS] if r.ok else _GLYPH[FAIL]
        suffix = c("dim", "  (non-blocking: live vendor check)") if non_blocking else ""
        print(f"  {glyph} {badge} {r.script.ljust(name_width)}  {r.duration:6.2f}s{suffix}")

    print("─" * WIDTH)

    total = len(results)
    blocking_failed = sum(not r.ok for r in results if r.script not in NON_BLOCKING)
    non_blocking_failed = sum(not r.ok for r in results if r.script in NON_BLOCKING)
    passed = total - blocking_failed - non_blocking_failed
    skipped = sum(1 for r in results for _, s in r.subtests if s == SKIP)

    failed_str = c("red", f"Failed: {blocking_failed}") if blocking_failed else f"Failed: {blocking_failed}"
    skipped_str = c("yellow", f"Skipped: {skipped}") if skipped else f"Skipped: {skipped}"
    print(f"  Total: {total}    {c('green', f'Passed: {passed}')}    {failed_str}    {skipped_str}", end="")
    if non_blocking_failed:
        print(f"    {c('yellow', f'Non-blocking warnings: {non_blocking_failed}')}")
    else:
        print()


def _print_coverage_report() -> None:
    print()
    print(c("bold", "COVERAGE (src/)"))
    # subprocess.run() below writes directly to the inherited stdout fd; when
    # stdout isn't a TTY (piped, redirected, or any CI log) Python's own
    # print() calls above are block-buffered and would otherwise land AFTER
    # the subprocess's already-flushed output -- garbling the log order.
    sys.stdout.flush()
    subprocess.run([sys.executable, "-m", "coverage", "combine", "--quiet"], cwd=ROOT, check=False)
    subprocess.run([sys.executable, "-m", "coverage", "report", "--include=src/*", "-m"],
                   cwd=ROOT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["fast", "data", "all"], default="fast")
    parser.add_argument("--coverage", action="store_true",
                         help="print a src/ coverage report after running (informational, "
                              "requires the `coverage` package)")
    args = parser.parse_args()

    scripts = {"fast": FAST, "data": DATA, "all": FAST + DATA}[args.group]
    results = [run(script, coverage=args.coverage) for script in scripts]

    print()
    for result in results:
        print_section(result)
    print_summary(results)

    if args.coverage:
        _print_coverage_report()

    failed = sum(not r.ok for r in results if r.script not in NON_BLOCKING)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
