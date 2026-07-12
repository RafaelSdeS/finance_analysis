"""
Test 1b (CVM-derived fundamentals): ratio math on synthetic statements +
cross-source check against a BolsAI-sourced file when CVM caches exist.

Part 1 (always runs, pure code): a hand-built quarterly frame with known
values must produce the exact BolsAI-convention ratios (verified live against
BPAN4 2025-09-30: single-quarter flows, thousands for statements, R$ units
for market_cap) and pass validate_fundamentals + carry all FUND_COLS.

Part 2 (SKIPs until CVM caches are collected): for a ticker that has BOTH a
BolsAI fundamentals file and CVM statement coverage, raw statement values
(net_income, equity) must agree within tolerance — same cross-vendor bar as
validate_vs_yfinance.py.

Run from project root:
    python tests/data_collection/test_cvm_statements.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_collection import config, validate  # noqa: E402
from src.data_collection.cvm_statements import compute_ratios, load_statements  # noqa: E402

TOLERANCE = 0.15  # 15%, consistent with validate_vs_yfinance's loose band


def test_ratio_math():
    # One synthetic quarter, values in R$ thousands (like CVM/BolsAI):
    # net_income 100_000k, equity 1_000_000k, revenue 500_000k, assets 5_000_000k
    # close 10.00, shares 1_000_000_000 -> market_cap 10e9
    q = pd.DataFrame({
        "reference_date": [pd.Timestamp("2020-03-31")],
        "net_revenue": [500_000.0],
        "gross_profit": [200_000.0],
        "ebit": [150_000.0],
        "net_income": [100_000.0],
        "total_assets": [5_000_000.0],
        "current_assets": [800_000.0],
        "cash_caixa": [50_000.0],
        "cash_aplic": [150_000.0],
        "current_liabilities": [400_000.0],
        "debt_st": [100_000.0],
        "debt_lt": [900_000.0],
        "equity": [1_000_000.0],
        "close_price": [10.0],
        "shares_outstanding": [1_000_000_000],
    })
    out = compute_ratios(q, "TEST CO")
    r = out.iloc[0]

    assert abs(r["market_cap"] - 10e9) < 1, r["market_cap"]
    assert abs(r["pl"] - 100.0) < 0.01, r["pl"]          # 10e9 / 100_000k
    assert abs(r["pvp"] - 10.0) < 0.01, r["pvp"]         # 10e9 / 1_000_000k
    assert abs(r["p_sr"] - 20.0) < 0.01, r["p_sr"]
    assert abs(r["roe"] - 10.0) < 0.01, r["roe"]         # 100k/1000k * 100
    assert abs(r["roa"] - 2.0) < 0.01, r["roa"]
    assert abs(r["net_margin"] - 20.0) < 0.01, r["net_margin"]
    assert abs(r["gross_margin"] - 40.0) < 0.01, r["gross_margin"]
    assert abs(r["current_ratio"] - 2.0) < 0.01, r["current_ratio"]
    assert abs(r["cash"] - 200_000.0) < 0.01, r["cash"]
    assert abs(r["total_debt"] - 1_000_000.0) < 0.01, r["total_debt"]
    assert abs(r["net_debt"] - 800_000.0) < 0.01, r["net_debt"]
    assert abs(r["debt_equity"] - 1.0) < 0.01, r["debt_equity"]
    assert abs(r["lpa"] - 0.10) < 0.001, r["lpa"]        # 100_000k*1000 / 1e9 shares
    assert abs(r["vpa"] - 1.00) < 0.001, r["vpa"]
    assert r["ebitda"] == r["ebit"], "ebitda proxy must equal ebit"

    # schema gate: exactly what collect_fundamentals-written files must satisfy
    out["ticker"] = "XXXX3"
    vr = validate.validate_fundamentals(out)
    assert vr.passed, vr.errors
    missing = [c for c in validate.FUND_COLS if c not in out.columns]
    assert not missing, f"missing FUND_COLS: {missing}"
    print("PASS  ratio math + schema")
    return True


def test_cross_source_vs_bolsai():
    """Statement values from CVM vs BolsAI's, same ticker+quarter."""
    if not list(config.CVM_DIR.glob("stmt_*.parquet")):
        print("SKIP  cross-source: no CVM statement caches (run cvm_statements)")
        return True
    if not (config.CVM_DIR / "fca_crosswalk.parquet").exists():
        print("SKIP  cross-source: no crosswalk")
        return True

    xwalk = pd.read_parquet(config.CVM_DIR / "fca_crosswalk.parquet")
    stmts = load_statements()
    checked = 0
    for ticker in ("WEGE3", "PETR4", "VALE3"):
        bolsai_path = config.FUND_DIR / f"{ticker}.parquet"
        row = xwalk[xwalk["ticker"] == ticker]
        if not bolsai_path.exists() or row.empty:
            continue
        bolsai = pd.read_parquet(bolsai_path)
        cvm = stmts[stmts["cnpj"] == row.iloc[0]["cnpj"]]
        both = bolsai.merge(cvm, on="reference_date", suffixes=("_b", "_c"))
        both = both.dropna(subset=["net_income_b", "net_income_c", "equity_b", "equity_c"])
        if both.empty:
            continue
        for col in ("net_income", "equity"):
            b, c = both[f"{col}_b"], both[f"{col}_c"]
            rel = ((b - c).abs() / b.abs().clip(lower=1)).median()
            assert rel < TOLERANCE, f"{ticker} {col}: median rel diff {rel:.1%} > {TOLERANCE:.0%}"
        print(f"PASS  cross-source {ticker}: {len(both)} quarters within {TOLERANCE:.0%}")
        checked += 1
    if not checked:
        print("SKIP  cross-source: no overlapping ticker had both sources")
    return True


if __name__ == "__main__":
    ok = test_ratio_math() and test_cross_source_vs_bolsai()
    sys.exit(0 if ok else 1)
