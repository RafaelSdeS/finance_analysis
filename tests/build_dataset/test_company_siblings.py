"""
Test P3: company_siblings() groups share classes of the same company by cvm_code.

Pure code, synthetic data — fast group.

Run from project root:
    python tests/build_dataset/test_company_siblings.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.build_dataset.build_ml_dataset import company_siblings  # noqa: E402


def test_company_siblings():
    df = pd.DataFrame({
        "ticker": ["PETR4", "PETR3", "VALE3", "GHOST", "BLANK"],
        "cvm_code": ["9512", "9512", "4170", None, ""],
    })
    got = company_siblings(df)
    assert got["9512"] == ["PETR3", "PETR4"], got
    assert got["4170"] == ["VALE3"], got
    assert None not in got and "" not in got, "null/blank cvm_code must be excluded"
    assert len(got) == 2, got
    print("PASS  company_siblings")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_company_siblings() else 1)
