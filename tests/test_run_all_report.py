#!/usr/bin/env python3
"""Self-check for run_all.py's pytest-line parsing (regex-driven, worth a guard).

Run: python tests/test_run_all_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_all import parse_subtests, PASS, FAIL, SKIP

PYTEST_OUTPUT = """\
============================= test session starts ==============================
collecting ... collected 3 items

tests/build_dataset/test_split_config.py::test_a PASSED [ 33%]
tests/build_dataset/test_split_config.py::test_b FAILED [ 66%]
tests/build_dataset/test_split_config.py::test_c SKIPPED [100%]

============================== 1 failed, 1 passed, 1 skipped in 0.01s ===============================
"""

PLAIN_OUTPUT = """\
Ticker   : PETR4
Quarters : 96
TEST PASSED
"""


def demo() -> None:
    subtests = parse_subtests(PYTEST_OUTPUT)
    assert subtests == [("test_a", PASS), ("test_b", FAIL), ("test_c", SKIP)], subtests

    assert parse_subtests(PLAIN_OUTPUT) == []

    print("all checks passed")


if __name__ == "__main__":
    demo()
