"""
validate_us_vs_vendor.py
=========================
Cross-validates US raw parquet data against independent vendors, mirroring
validate_vs_yfinance.py's role for BR. Provenance is inverted from BR though:
US prices already ARE yfinance (yf_collectors.py), and US fundamentals are
SEC EDGAR parsed by 3 homegrown regex/HTML tiers (ex27/tenq/item6) plus one
structured-API tier (xbrl) -- see CLAUDE.md's sec/ module row. So:

Fundamentals: our SEC-derived xbrl-tier rows vs yfinance quarterly_financials/
              balance_sheet -- a genuinely independent vendor here (unlike BR,
              where BolsAI IS the primary source). Units are raw full USD on
              BOTH sides (no *1000 like BR) and NOT TTM-rolled (both already
              single-quarter). Covers the xbrl tier only -- yfinance has ~5
              quarters of history, nowhere near the ex27/tenq/item6 (1994-2006)
              range. See tier-seam check below for that gap.
Internal:     Derived ratio columns recomputed by hand from raw columns, same
              row -- currency/scale-immune. No market_cap/pl/pvp/ev_* (US
              fundamentals carry no joined price at collection time).
Tier seams:   Universe-wide (not just the 4 sample tickers). Flags implausible
              value jumps across a fundamentals_tier boundary on the same
              ticker -- the only proxy check available for the three homegrown
              1994-2006 parsers, since no free vendor covers that range.
Prices:       Alpha Vantage TIME_SERIES_DAILY (raw as-traded) vs our `close`
              (also nominal -- yf_collectors.py reverse-adjusts every split
              since US always fetches from floor="1900-01-01"). Key-gated
              (free tier, 25 req/day) -- SKIPs cleanly with no key.
              outputsize=compact (~100 most recent trading days) -- AV's free
              tier stopped serving outputsize=full (confirmed 2026-08-06, now
              a premium-only parameter); compact still exercises the same
              reverse-split math on any split that landed in that window.

Usage (from project root):
    python tests/data_collection/validate_us_vs_vendor.py
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.data_collection import config  # noqa: E402
from test_utils import print_header  # noqa: E402

TICKERS = ["AAPL", "XOM", "KO", "JNJ"]
PRICE_DIR = ROOT / "data/raw/us/prices"
FUND_DIR = ROOT / "data/raw/us/fundamentals"
TOLERANCE_PCT = 25  # vendor differences; matches BR's validate_vs_yfinance.py

SEAM_RATIO_MAX = 5        # flag a tier-boundary value jump >5x or <1/5x
SEAM_RATE_CEILING = 0.05  # measured baseline 2026-08-05: 4.4% of seams flagged
SEAM_COLS = ["net_revenue", "equity", "total_assets"]
SEAM_MAX_GAP_DAYS = 400  # don't read a multi-year hole in coverage as a "seam"


def validate_fundamentals(ticker) -> bool:
    """xbrl-tier only (yfinance's ~5-quarter window can't reach ex27/tenq/item6)."""
    path = FUND_DIR / f"{ticker}.parquet"
    if not path.exists():
        print("  Fundamentals: SKIP (no data collected for this ticker)")
        return True
    fund = pd.read_parquet(path)
    fund = fund[(fund["fundamentals_tier"] == "xbrl") & (fund["period_months"] == 3)]
    if fund.empty:
        print("  Fundamentals: N/A (no xbrl rows)")
        return True

    yt = yf.Ticker(ticker)
    ok = True

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
        ok = _print_fund_rows(col, col, fund, yf_q) and ok

    try:
        bs = yt.quarterly_balance_sheet
    except Exception as e:
        print(f"  Balance sheet: N/A (yfinance error: {e})")
        return ok

    for col, yf_row in [("equity", "Stockholders Equity"), ("total_assets", "Total Assets")]:
        if yf_row not in bs.index:
            print(f"  {col}: N/A (yfinance row '{yf_row}' missing)")
            continue
        yf_bs = pd.Series(bs.loc[yf_row], dtype=float).dropna().sort_index()
        ok = _print_fund_rows(col, col, fund, yf_bs) and ok

    return ok


def _print_fund_rows(label, col, fund, yf_series) -> bool:
    """Compare our `end`-keyed column (raw USD) against a yfinance series (raw USD),
    fuzzy-matched within +/-10 days (our `end` is the true fiscal end, yfinance keys
    on calendar quarter-ends). When more than one of our rows falls in that window,
    picks the temporally NEAREST one, not whichever happens to be first in `fund`'s
    current row order (real bug, confirmed 2026-08-12: unrelated to actual proximity)."""
    print(f"  {label}:")
    printed = False
    ok = True
    for dt, yf_val in yf_series.items():
        dt = dt.tz_localize(None) if dt.tzinfo else dt
        near = fund[(fund["end"] - dt).abs() <= pd.Timedelta(days=10)]
        if near.empty or yf_val == 0:
            continue
        nearest_idx = (near["end"] - dt).abs().idxmin()
        ours = near.loc[nearest_idx, col]
        if pd.isna(ours):
            continue
        pct = (ours - yf_val) / abs(yf_val) * 100
        print(f"    {dt.date()}: ours={ours/1e9:.2f}B  yf={yf_val/1e9:.2f}B  diff={pct:+.1f}%")
        printed = True
        if abs(pct) > TOLERANCE_PCT:
            ok = False
    if not printed:
        print("    N/A (no overlapping quarter-end dates)")
    return ok


def check_internal_consistency(ticker) -> bool:
    """Recompute derived columns by hand from raw columns, same row. Currency/scale-immune."""
    path = FUND_DIR / f"{ticker}.parquet"
    if not path.exists():
        print("  Internal consistency: SKIP (no data collected for this ticker)")
        return True
    fund = pd.read_parquet(path).sort_values("end")
    r = fund.iloc[-1]
    ok = True

    # r.get(...) not r[...] -- real bug, confirmed 2026-08-12: a company can
    # legitimately lack one of these ~13 raw columns (e.g. no net_revenue
    # concept ever reported, same gap check_tier_seams already accounts for
    # when picking which columns to load), and a bare KeyError here used to
    # crash this ticker's WHOLE validation instead of reporting that one
    # check as N/A like every other missing-input case already does below.
    def g(col):
        return r.get(col, float("nan"))

    checks = [
        ("lpa",            g("lpa"),            g("net_income") / g("shares_outstanding")),
        ("vpa",            g("vpa"),            g("equity") / g("shares_outstanding")),
        ("roe",            g("roe"),            g("net_income") / g("equity") * 100),
        ("roa",            g("roa"),            g("net_income") / g("total_assets") * 100),
        ("net_margin",     g("net_margin"),     g("net_income") / g("net_revenue") * 100),
        ("ebitda_margin",  g("ebitda_margin"),  g("ebitda") / g("net_revenue") * 100),
        ("net_debt",       g("net_debt"),       g("total_debt") - g("cash")),
        ("debt_equity",    g("debt_equity"),    g("total_debt") / g("equity")),
        ("current_ratio",  g("current_ratio"),  g("current_assets") / g("current_liabilities")),
        ("asset_turnover", g("asset_turnover"), g("net_revenue") / g("total_assets")),
    ]

    print(f"  Latest quarter: {r['end'].date()}")
    for label, ours, calc in checks:
        if pd.isna(ours) or pd.isna(calc):
            print(f"    {label:14s}: N/A (null input)")
            continue
        # ours == 0 alone used to force pct=inf unconditionally -- real bug,
        # confirmed 2026-08-12: a legitimately debt-free company (total_debt
        # == 0, so both `ours` and the recomputed `calc` are genuinely 0)
        # was reported as a spurious FAIL. Only a real ours-vs-calc mismatch
        # around a zero `ours` should count as infinite relative error.
        if ours == 0:
            pct = 0.0 if calc == 0 else float("inf")
        else:
            pct = (calc - ours) / abs(ours) * 100
        flag = "PASS" if abs(pct) < TOLERANCE_PCT else "FAIL"
        print(f"    {label:14s}: ours={ours:>14.2f}  recomputed={calc:>14.2f}  diff={pct:+6.1f}%  {flag}")
        if flag == "FAIL":
            ok = False
    return ok


def check_tier_seams() -> bool:
    """Universe-wide: flag implausible value jumps across a fundamentals_tier boundary
    on the same ticker. Proxy check for the ex27/tenq/item6 (1994-2006) parsers, which no
    free vendor's history reaches -- see validate_fundamentals()'s xbrl-only limitation."""
    files = sorted(FUND_DIR.glob("*.parquet"))
    if not files:
        print("  SKIP: data/raw/us/fundamentals not collected yet")
        return True

    seams, offenders = 0, []
    for f in files:
        # Some tickers' parquet lacks a SEAM_COLS column entirely (dropped pre-write if
        # all-NaN for that company, e.g. no net_revenue concept ever reported) -- peek
        # the schema (no data read) before picking which columns to actually load.
        schema_cols = set(pq.ParquetFile(f).schema.names)
        cols = ["end", "fundamentals_tier", "period_months"] + [c for c in SEAM_COLS if c in schema_cols]
        d = pd.read_parquet(f, columns=cols)
        if d["fundamentals_tier"].nunique() < 2:
            continue
        d = d.sort_values("end").reset_index(drop=True)
        prev_tier = d["fundamentals_tier"].shift()
        gap_days = (d["end"] - d["end"].shift()).dt.days
        is_seam = prev_tier.notna() & (prev_tier != d["fundamentals_tier"]) & (gap_days <= SEAM_MAX_GAP_DAYS)

        for col in [c for c in SEAM_COLS if c in d.columns]:
            v = d[col].astype(float)
            if col == "net_revenue":
                v = v / (d["period_months"].fillna(3) / 3.0)  # normalize flows to per-quarter
            # `ratio > 0` and `np.isfinite(ratio)` used to gate the flag check
            # itself -- real bug, confirmed 2026-08-12: that silently EXCLUDED
            # a collapse-to-zero (ratio == 0, still correctly < 1/SEAM_RATIO_MAX
            # on its own) and a jump-from-zero (ratio == inf, still correctly
            # > SEAM_RATIO_MAX on its own) from ever being flagged -- exactly
            # the two most severe scaling failures this check exists to catch,
            # while `seams` (the denominator) still counted them, understating
            # the true flagged rate. Only genuine NaN (0/0, no real prior/
            # current value to compare) should be excluded.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (v / v.shift()).abs()
            flagged = is_seam & ratio.notna() & (
                (ratio > SEAM_RATIO_MAX) | (ratio < 1 / SEAM_RATIO_MAX)
            )
            seams += int(is_seam.sum())
            for i in d.index[flagged]:
                offenders.append((f.stem, col, prev_tier[i], d["fundamentals_tier"][i],
                                   d["end"][i].date(), ratio[i]))

    rate = len(offenders) / seams if seams else 0.0
    print(f"  Seam observations : {seams}")
    print(f"  Flagged (>{SEAM_RATIO_MAX}x)   : {len(offenders)} ({rate*100:.1f}%)")
    if offenders:
        offenders.sort(key=lambda o: max(o[5], 1 / o[5]), reverse=True)
        for tkr, col, frm, to, end, ratio in offenders[:15]:
            print(f"    {tkr:6s} {col:12s} {frm}->{to}  end={end}  ratio={ratio:.4g}")
    return rate <= SEAM_RATE_CEILING


def _av_daily(ticker):
    if not config.ALPHAVANTAGE_API_KEY:
        return None
    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}&outputsize=compact&apikey={config.ALPHAVANTAGE_API_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  Prices: N/A (Alpha Vantage request failed: {e})")
        return None

    ts = data.get("Time Series (Daily)")
    if not ts:
        note = data.get("Note") or data.get("Information") or data.get("Error Message") or str(data)[:200]
        print(f"  Prices: N/A (Alpha Vantage: {note})")
        return None

    av = pd.DataFrame.from_dict(ts, orient="index", dtype=float)
    av.index = pd.to_datetime(av.index)
    av.index.name = "trade_date"
    return av["4. close"].rename("av_close")


def validate_prices(ticker) -> bool:
    if not config.ALPHAVANTAGE_API_KEY:
        print("  Prices: SKIP (no ALPHAVANTAGE_API_KEY set)")
        return True

    path = PRICE_DIR / f"{ticker}.parquet"
    if not path.exists():
        print("  Prices: SKIP (no data collected for this ticker)")
        return True
    prices = pd.read_parquet(path)
    av_close = _av_daily(ticker)
    if av_close is None:
        return True

    merged = prices.set_index("trade_date")[["close"]].join(av_close, how="inner")
    if merged.empty:
        print("  Prices: N/A (no overlapping dates)")
        return True

    pct = (merged["close"] - merged["av_close"]) / merged["av_close"] * 100
    print(f"  Rows compared : {len(merged)}  ({merged.index.min().date()} -> {merged.index.max().date()})")
    print(f"  Mean abs diff : {pct.abs().mean():.4f}%")
    print(f"  Max abs diff  : {pct.abs().max():.4f}%")
    print(f"  Within 1%     : {(pct.abs() < 1).mean()*100:.1f}% of rows")
    print(f"  Within 5%     : {(pct.abs() < 5).mean()*100:.1f}% of rows")

    flagged = merged[pct.abs() > TOLERANCE_PCT]
    if flagged.empty:
        print(f"  Flagged >{TOLERANCE_PCT}%  : none")
    else:
        print(f"  Flagged >{TOLERANCE_PCT}%  : {len(flagged)} rows")
        for dt, row in flagged.head(10).iterrows():
            p = (row["close"] - row["av_close"]) / row["av_close"] * 100
            print(f"    {dt.date()}  ours={row['close']:.2f}  av={row['av_close']:.2f}  diff={p:+.2f}%")
    return flagged.empty


def main():
    results = []
    for ticker in TICKERS:
        print()
        print_header(f"TICKER: {ticker}")
        print("\n[FUNDAMENTALS — ours (xbrl tier) vs yfinance]")
        results.append(validate_fundamentals(ticker))
        print("\n[INTERNAL CONSISTENCY — derived cols recomputed from raw, same row]")
        results.append(check_internal_consistency(ticker))
        print("\n[PRICES — ours vs Alpha Vantage, raw as-traded, last ~100 trading days]")
        results.append(validate_prices(ticker))

    print()
    print_header("TIER SEAMS (universe-wide)")
    print("\n[Value continuity across fundamentals_tier boundaries -- ex27/tenq/item6/xbrl]")
    results.append(check_tier_seams())

    overall = all(results)
    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
