"""
validate_vs_yfinance.py
=======================
Cross-validates BolsAI raw parquet data against yfinance for PETR4, VALE3, WEGE3.

Prices:       BolsAI close vs yfinance Close (auto_adjust=False), post-last-split only.
              (adj_close skipped — dividend-adjustment methods diverge, uninformative.)
Fundamentals: BolsAI net_revenue/net_income (BRL thousands, TTM) vs yfinance
              quarterly_financials (single-quarter -> rolling 4Q TTM).
CAGR:         Not re-checked here. Run: python src/cagr_handler.py --ticker PETR4

Usage (from project root):
    python tests/raw_data/validate_vs_yfinance.py
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS   = ["PETR4", "VALE3", "WEGE3"]
PROJECT   = Path(__file__).resolve().parents[2]
PRICE_DIR = PROJECT / "data/raw/prices"
FUND_DIR  = PROJECT / "data/raw/fundamentals"


def validate_prices(ticker):
    prices = pd.read_parquet(PRICE_DIR / f"{ticker}.parquet")
    start = str(prices["trade_date"].min().date())
    end   = str(prices["trade_date"].max().date())

    t = yf.Ticker(ticker + ".SA")
    hist = t.history(start=start, end=end, auto_adjust=False)[["Close"]]
    hist.index = hist.index.tz_localize(None)
    hist.index.name = "trade_date"
    hist = hist.rename(columns={"Close": "yf_close"})

    # yfinance Close is retroactively split-adjusted, BolsAI close is not.
    # They only agree after the last split, so drop everything before it.
    splits = t.splits
    if len(splits):
        cutoff = splits.index.max().tz_localize(None).normalize()
        prices = prices[prices["trade_date"] > cutoff]

    merged = prices.set_index("trade_date")[["close"]].join(hist, how="inner")
    if merged.empty:
        print("  Prices: N/A (no overlapping dates)")
        return

    pct = (merged["close"] - merged["yf_close"]) / merged["yf_close"] * 100
    print(f"  Rows compared : {len(merged)}  ({merged.index.min().date()} -> {merged.index.max().date()})")
    print(f"  Mean abs diff : {pct.abs().mean():.4f}%")
    print(f"  Max abs diff  : {pct.abs().max():.4f}%")
    print(f"  Mean signed   : {pct.mean():.4f}%  (+ = BolsAI higher)")
    print(f"  Within 1%     : {(pct.abs() < 1).mean()*100:.1f}% of rows")
    print(f"  Within 5%     : {(pct.abs() < 5).mean()*100:.1f}% of rows")

    flagged = merged[pct.abs() > 5]
    if flagged.empty:
        print("  Flagged >5%   : none")
    else:
        print(f"  Flagged >5%   : {len(flagged)} rows")
        for dt, row in flagged.iterrows():
            p = (row["close"] - row["yf_close"]) / row["yf_close"] * 100
            print(f"    {dt.date()}  BolsAI={row['close']:.2f}  yf={row['yf_close']:.2f}  diff={p:+.2f}%")


def validate_fundamentals(ticker):
    fund = pd.read_parquet(FUND_DIR / f"{ticker}.parquet")
    yt = yf.Ticker(ticker + ".SA")

    # Income statement: single-quarter -> rolling 4Q TTM (BolsAI reports TTM)
    try:
        qf = yt.quarterly_financials
    except Exception as e:
        print(f"  Income: N/A (yfinance error: {e})")
        qf = pd.DataFrame()

    for col, yf_row in [("net_revenue", "Total Revenue"), ("net_income", "Net Income")]:
        if yf_row not in qf.index:
            print(f"  {col}: N/A (yfinance row '{yf_row}' missing)")
            continue
        yf_q = pd.Series(qf.loc[yf_row], dtype=float).dropna().sort_index()
        yf_ttm = yf_q.rolling(4).sum().dropna()
        _print_fund_rows(col, col, fund, yf_ttm)

    # Balance sheet: point-in-time, NO rolling sum
    try:
        bs = yt.quarterly_balance_sheet
    except Exception as e:
        print(f"  Balance sheet: N/A (yfinance error: {e})")
        return

    for col, yf_row in [("equity", "Stockholders Equity"), ("total_assets", "Total Assets"),
                        ("total_debt", "Total Debt"), ("cash", "Cash And Cash Equivalents")]:
        if yf_row not in bs.index:
            print(f"  {col}: N/A (yfinance row '{yf_row}' missing)")
            continue
        yf_bs = pd.Series(bs.loc[yf_row], dtype=float).dropna().sort_index()
        _print_fund_rows(col, col, fund, yf_bs)


def _print_fund_rows(label, col, fund, yf_series):
    """Compare a BolsAI column (BRL thousands) against a yfinance series (full BRL)."""
    print(f"  {label}:")
    printed = False
    for dt, yf_val in yf_series.items():
        row = fund[fund["reference_date"] == dt]
        if row.empty or yf_val == 0:
            continue
        bolsai = row[col].values[0] * 1000  # BolsAI stores BRL thousands
        pct = (bolsai - yf_val) / abs(yf_val) * 100
        note = "  [likely currency mismatch — check reporting currency]" if abs(pct) > 200 else ""
        print(f"    {dt.date()}: BolsAI={bolsai/1e9:.2f}B  yf={yf_val/1e9:.2f}B  diff={pct:+.1f}%{note}")
        printed = True
    if not printed:
        print("    N/A (no overlapping quarter-end dates)")


def check_internal_consistency(ticker):
    """Recompute BolsAI's derived columns from its own raw columns, same row.
    Currency-immune (units cancel within a row). Tolerance 5%."""
    fund = pd.read_parquet(FUND_DIR / f"{ticker}.parquet").sort_values("reference_date")
    r = fund.iloc[-1]
    K = 1000  # financials are BRL thousands; market_cap/close_price are full BRL / per-share

    # (label, BolsAI value, recomputed value). See units note in validate run.
    checks = [
        ("market_cap",    r["market_cap"],   r["close_price"] * r["shares_outstanding"]),
        ("lpa",           r["lpa"],          r["net_income"] * K / r["shares_outstanding"]),
        ("vpa",           r["vpa"],          r["equity"] * K / r["shares_outstanding"]),
        ("pl",            r["pl"],           r["market_cap"] / (r["net_income"] * K)),
        ("pvp",           r["pvp"],          r["market_cap"] / (r["equity"] * K)),
        ("roe",           r["roe"],          r["net_income"] / r["equity"] * 100),
        ("roa",           r["roa"],          r["net_income"] / r["total_assets"] * 100),
        ("net_margin",    r["net_margin"],   r["net_income"] / r["net_revenue"] * 100),
        ("ebitda_margin", r["ebitda_margin"], r["ebitda"] / r["net_revenue"] * 100),
        ("net_debt",      r["net_debt"],     r["total_debt"] - r["cash"]),
        ("debt_equity",   r["debt_equity"],  r["total_debt"] / r["equity"]),
        ("ev_ebitda",     r["ev_ebitda"],    (r["market_cap"] + r["net_debt"] * K) / (r["ebitda"] * K)),
    ]

    print(f"  Latest quarter: {r['reference_date'].date()}")
    for label, bolsai, calc in checks:
        if pd.isna(bolsai) or pd.isna(calc):
            print(f"    {label:14s}: N/A (null input)")
            continue
        pct = (calc - bolsai) / abs(bolsai) * 100 if bolsai != 0 else float("inf")
        flag = "PASS" if abs(pct) < 5 else "FAIL"
        print(f"    {label:14s}: BolsAI={bolsai:>14.2f}  recomputed={calc:>14.2f}  diff={pct:+6.1f}%  {flag}")


def main():
    for ticker in TICKERS:
        print("\n" + "=" * 70)
        print(f"TICKER: {ticker}")
        print("=" * 70)
        print("\n[PRICES — BolsAI close vs yfinance Close, unadjusted, post-split]")
        validate_prices(ticker)
        print("\n[FUNDAMENTALS — BolsAI vs yfinance (income TTM, balance point-in-time)]")
        validate_fundamentals(ticker)
        print("\n[INTERNAL CONSISTENCY — BolsAI derived cols recomputed from raw, same row]")
        check_internal_consistency(ticker)

    print("\n" + "=" * 70)
    print("CAGR — not checked here. Run:")
    for t in TICKERS:
        print(f"  python src/cagr_handler.py --ticker {t}")


if __name__ == "__main__":
    main()
