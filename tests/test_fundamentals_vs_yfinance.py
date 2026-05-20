"""
test_fundamentals_vs_yfinance.py
================================
Compares fundamental values from our ml_dataset against Yahoo Finance.

For ratios that depend on price (P/E, P/B, Market Cap):
    - Uses the quarter-end closing price from yfinance history
      to reconstruct point-in-time values, matching Bolsai's methodology.

For flow metrics (Revenue, Net Income, EBITDA, EBIT):
    - Compares against yfinance TTM (annual), since Bolsai uses annual figures
      for December quarters.

For balance sheet items:
    - Compares against yfinance latest reported quarter.

Bolsai absolute values are in thousands of R$ → scaled ×1000.

Usage:
    python test_fundamentals_vs_yfinance.py
    python test_fundamentals_vs_yfinance.py --ticker VALE3
    python test_fundamentals_vs_yfinance.py --ticker PETR4 --tolerance 20
"""

import argparse
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DATASET_PATH = "../data/processed/ml_dataset.parquet"
BOLSAI_SCALE = 1000


# =============================================================================
# HELPERS
# =============================================================================

def pct_diff(a, b):
    if a is None or b is None or b == 0:
        return None
    return abs(a - b) / abs(b) * 100


def status(diff, tolerance):
    if diff is None:
        return "⚠  N/A"
    return f"✓ {diff:5.1f}%" if diff <= tolerance else f"✗ {diff:5.1f}%"


def fmt_ratio(val):
    return f"{val:.2f}" if val is not None else "N/A"


def fmt_abs(val):
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"{val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"{val/1e9:.2f}B"
    return f"{val/1e6:.2f}M"


def col0(df, *keys):
    """First column value for any of the given row keys."""
    if df is None or df.empty:
        return None
    for key in keys:
        if key in df.index:
            val = df.loc[key].iloc[0]
            return float(val) if pd.notna(val) else None
    return None


# =============================================================================
# FETCH FROM OUR DATASET
# =============================================================================

def get_our_data(ticker: str) -> dict:
    df     = pd.read_parquet(DATASET_PATH)
    df     = df[df["ticker"] == ticker].sort_values("reference_date")
    latest = df["reference_date"].max()
    row    = df[df["reference_date"] == latest].iloc[0]

    def s(col):
        v = row.get(col)
        return float(v) * BOLSAI_SCALE if v is not None and not pd.isna(v) else None

    def r(col):
        v = row.get(col)
        return float(v) if v is not None and not pd.isna(v) else None

    return {
        "reference_date":  latest,
        "close_price":     r("close_price"),   # quarter-end price from Bolsai
        "shares":          r("shares_outstanding"),
        # Ratios
        "pl":              r("pl"),
        "pvp":             r("pvp"),
        "roe":             r("roe"),
        "roa":             r("roa"),
        "net_margin":      r("net_margin"),
        "gross_margin":    r("gross_margin"),
        "ebitda_margin":   r("ebitda_margin"),
        "current_ratio":   r("current_ratio"),
        "debt_equity":     r("debt_equity"),
        # Absolute (scaled)
        "market_cap":      r("market_cap"),
        "net_revenue":     s("net_revenue"),
        "net_income":      s("net_income"),
        "ebitda":          s("ebitda"),
        "ebit":            s("ebit"),
        "total_assets":    s("total_assets"),
        "equity":          s("equity"),
        "total_debt":      s("total_debt"),
        "net_debt":        s("net_debt"),
        "cash":            s("cash"),
        # Per-share (for ratio reconstruction)
        "lpa":             r("lpa"),   # EPS
        "vpa":             r("vpa"),   # book value per share
    }


# =============================================================================
# FETCH FROM YFINANCE — point-in-time price + current fundamentals
# =============================================================================

def get_yfinance_data(ticker: str, reference_date: pd.Timestamp) -> dict:
    t    = yf.Ticker(f"{ticker}.SA")
    info = t.info

    # ── Point-in-time closing price on quarter-end date ───────────────────────
    date_str   = reference_date.strftime("%Y-%m-%d")
    next_day   = (reference_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    hist       = yf.download(f"{ticker}.SA", start=date_str, end=next_day,
                             progress=False, auto_adjust=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    pit_price = float(hist["Close"].iloc[0]) if not hist.empty else None

    # ── Financial statements ──────────────────────────────────────────────────
    try:
        is_annual  = t.income_stmt
        is_quarter = t.quarterly_income_stmt
        bs_quarter = t.quarterly_balance_sheet
    except Exception:
        is_annual  = pd.DataFrame()
        is_quarter = pd.DataFrame()
        bs_quarter = pd.DataFrame()

    def info_pct(key):
        v = info.get(key)
        return float(v) * 100 if v is not None else None

    shares = info.get("sharesOutstanding")

    # Reconstruct point-in-time ratios using quarter-end price
    lpa_yf  = col0(is_quarter, "Basic EPS", "Diluted EPS")
    vpa_yf  = None
    equity_q = col0(bs_quarter, "Stockholders Equity", "Common Stock Equity")
    if equity_q is not None and shares:
        vpa_yf = equity_q / shares

    pit_pe = pit_price / lpa_yf  if (pit_price and lpa_yf and lpa_yf > 0) else None
    pit_pb = pit_price / vpa_yf  if (pit_price and vpa_yf and vpa_yf > 0) else None
    pit_mc = pit_price * shares  if (pit_price and shares) else None

    return {
        "pit_price":        pit_price,
        # Point-in-time ratios (reconstructed)
        "pl_pit":           pit_pe,
        "pvp_pit":          pit_pb,
        "market_cap_pit":   pit_mc,
        # Current ratios from info (for reference)
        "pl_current":       info.get("trailingPE"),
        "pvp_current":      info.get("priceToBook"),
        "market_cap_current": info.get("marketCap"),
        # Margins/returns (TTM — no historical equivalent in yfinance)
        "roe":              info_pct("returnOnEquity"),
        "roa":              info_pct("returnOnAssets"),
        "net_margin":       info_pct("profitMargins"),
        "gross_margin":     info_pct("grossMargins"),
        "ebitda_margin":    info_pct("ebitdaMargins"),
        "current_ratio":    info.get("currentRatio"),
        "debt_equity":      info.get("debtToEquity"),
        # Income — annual TTM
        "net_revenue_ttm":  col0(is_annual, "Total Revenue"),
        "net_income_ttm":   col0(is_annual, "Net Income", "Net Income Common Stockholders"),
        "ebitda_ttm":       col0(is_annual, "EBITDA", "Normalized EBITDA"),
        "ebit_ttm":         col0(is_annual, "EBIT", "Operating Income"),
        # Balance sheet — latest quarter
        "total_assets_q":   col0(bs_quarter, "Total Assets"),
        "equity_q":         equity_q,
        "total_debt_q":     col0(bs_quarter, "Total Debt"),
        "cash_q":           col0(bs_quarter, "Cash And Cash Equivalents",
                                 "Cash Cash Equivalents And Short Term Investments"),
    }


# =============================================================================
# PRINT SECTION
# =============================================================================

def print_section(title, rows, tolerance):
    print()
    print(f"── {title} {'─' * (80 - len(title))}")
    print(f"{'Metric':<25} {'Ours (Bolsai)':>16} {'YF (matched)':>16} {'Diff':>12}")
    print("─" * 72)
    diffs = []
    for label, our_val, yf_val, kind in rows:
        fmt  = fmt_ratio if kind == "ratio" else fmt_abs
        diff = pct_diff(our_val, yf_val)
        if diff is not None:
            diffs.append(diff)
        print(f"{label:<25} {fmt(our_val):>16} {fmt(yf_val):>16} {status(diff, tolerance):>12}")
    return diffs


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",    default="PETR4")
    parser.add_argument("--tolerance", type=float, default=15.0)
    args = parser.parse_args()

    print(f"Ticker    : {args.ticker}")
    print(f"Tolerance : {args.tolerance}%")

    print("\nFetching from our dataset...")
    ours = get_our_data(args.ticker)
    ref  = ours["reference_date"]
    print(f"Latest quarter : {ref.date()}")
    print(f"Bolsai price   : R${ours['close_price']:.2f}")

    print("Fetching from Yahoo Finance...")
    yfd = get_yfinance_data(args.ticker, ref)
    print(f"YF price on {ref.date()} : R${yfd['pit_price']:.2f}" if yfd["pit_price"] else "YF price: N/A")

    print()
    print("=" * 72)
    print("FUNDAMENTALS COMPARISON (point-in-time matched)")
    print("=" * 72)

    all_diffs = []

    all_diffs += print_section("PRICE-DEPENDENT RATIOS (quarter-end price)", [
        ("P/E",              ours["pl"],           yfd["pl_pit"],           "ratio"),
        ("P/B",              ours["pvp"],          yfd["pvp_pit"],          "ratio"),
        ("Market Cap",       ours["market_cap"],   yfd["market_cap_pit"],   "abs"),
    ], args.tolerance)

    all_diffs += print_section("MARGINS & RETURNS (TTM — best available)", [
        ("ROE (%)",          ours["roe"],          yfd["roe"],              "ratio"),
        ("ROA (%)",          ours["roa"],          yfd["roa"],              "ratio"),
        ("Net Margin (%)",   ours["net_margin"],   yfd["net_margin"],       "ratio"),
        ("Gross Margin (%)", ours["gross_margin"], yfd["gross_margin"],     "ratio"),
        ("EBITDA Margin (%)",ours["ebitda_margin"],yfd["ebitda_margin"],    "ratio"),
        ("Current Ratio",    ours["current_ratio"],yfd["current_ratio"],    "ratio"),
    ], args.tolerance)

    all_diffs += print_section("INCOME STATEMENT (annual / TTM)", [
        ("Net Revenue",      ours["net_revenue"],  yfd["net_revenue_ttm"],  "abs"),
        ("Net Income",       ours["net_income"],   yfd["net_income_ttm"],   "abs"),
        ("EBITDA",           ours["ebitda"],       yfd["ebitda_ttm"],       "abs"),
        ("EBIT",             ours["ebit"],         yfd["ebit_ttm"],         "abs"),
    ], args.tolerance)

    all_diffs += print_section("BALANCE SHEET (latest quarter)", [
        ("Total Assets",     ours["total_assets"], yfd["total_assets_q"],   "abs"),
        ("Equity",           ours["equity"],       yfd["equity_q"],         "abs"),
        ("Total Debt",       ours["total_debt"],   yfd["total_debt_q"],     "abs"),
        ("Cash",             ours["cash"],         yfd["cash_q"],           "abs"),
    ], args.tolerance)

    print()
    print("=" * 72)
    if all_diffs:
        passed = sum(1 for d in all_diffs if d <= args.tolerance)
        print(f"Mean absolute diff : {sum(all_diffs)/len(all_diffs):.1f}%")
        print(f"Within tolerance   : {passed}/{len(all_diffs)} fields")

    print()
    print("Notes:")
    print(f"  - P/E, P/B, Market Cap use yfinance price on {ref.date()} to match Bolsai")
    print("  - Margins/returns: yfinance TTM vs Bolsai annual (Dec quarter)")
    print("  - Balance sheet: latest reported quarter from both sources")
    print("  - Debt/Equity: yfinance may report as % (×100) — divide by 100 if >10")


if __name__ == "__main__":
    main()