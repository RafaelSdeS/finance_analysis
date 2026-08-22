"""
test_cvm_crosswalk.py
======================
Regression guard for cvm/crosswalk.py's _TICKER regex (fixed in e942e31,
"reject numeric CVM codes in FCA ticker crosswalk regex"): a few FCA filings
put the numeric CVM registration code (e.g. "023574") in Codigo_Negociacao
instead of a real B3 ticker -- confirmed on disk as tickers "11215"/"23574"/
"25585", each equal to that row's own cvm_code, purged in 3d73d74. The
(?=.*[A-Z]) lookahead added in e942e31 rejects any match with no letter.

Run from project root:
    python tests/data_collection/test_cvm_crosswalk.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_collection.cvm.crosswalk import _TICKER  # noqa: E402


def test_ticker_regex_rejects_pure_numeric_cvm_codes():
    for bad in ("11215", "23574", "25585", "112311", "000000"):
        assert not _TICKER.match(bad), f"{bad} must be rejected -- pure-numeric CVM code, not a real ticker"


def test_ticker_regex_accepts_real_tickers():
    for good in ("PETR4", "VALE3", "ABEV3", "ITSA4", "BPAC11", "WEGE3"):
        assert _TICKER.match(good), f"{good} must be accepted -- real B3 ticker"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
