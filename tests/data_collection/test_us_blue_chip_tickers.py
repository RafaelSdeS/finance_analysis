"""
test_us_blue_chip_tickers.py
=============================
US analogue of test_blue_chip_tickers.py: spot-check the raw data for a
fixed list of well-established, long-history US tickers (AAPL, MSFT, ...).
Same rationale -- these are the names most likely to surface a collection/
validation bug first, and most costly to get wrong if missed.

Reuses the same schema/sanity gate collectors run at write-time
(src/data_collection/validate.py) instead of re-implementing checks, plus
one extra guard specific to this list: these tickers must actually be
present with a long history, not just internally consistent.

Skips gracefully (prints SKIP, still exits 0) if data/raw/us isn't collected
yet -- it's gitignored, unlike BR's git-tracked data/raw/.

Usage:
    python tests/data_collection/test_us_blue_chip_tickers.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_collection"))

import validate  # noqa: E402

BLUE_CHIPS = [
    "AAPL", "MSFT", "GOOGL", "JPM", "JNJ",
    "XOM", "KO", "WMT", "PG", "DIS",
]

MIN_PRICE_ROWS = 1000  # ~4 years of trading days; these tickers have decades

RAW = ROOT / "data" / "raw" / "us"


def main():
    if not RAW.exists():
        print("SKIP: data/raw/us not collected yet (gitignored, rebuild via src/data_collection/us/pipeline.py)")
        return

    checks = []

    for ticker in BLUE_CHIPS:
        prices_path = RAW / "prices" / f"{ticker}.parquet"
        fund_path = RAW / "fundamentals" / f"{ticker}.parquet"
        div_path = RAW / "dividends" / f"{ticker}.parquet"

        checks.append((f"{ticker}: prices file exists", prices_path.exists()))
        checks.append((f"{ticker}: fundamentals file exists", fund_path.exists()))
        checks.append((f"{ticker}: dividends file exists", div_path.exists()))

        if prices_path.exists():
            df = pd.read_parquet(prices_path)
            r = validate.validate_prices(df)
            checks.append((f"{ticker}: prices valid [{'; '.join(r.errors) or 'ok'}]", r.passed))
            checks.append((f"{ticker}: prices has >= {MIN_PRICE_ROWS} rows [{len(df)}]",
                            len(df) >= MIN_PRICE_ROWS))

        if fund_path.exists():
            df = pd.read_parquet(fund_path)
            r = validate.validate_us_fundamentals(df)
            checks.append((f"{ticker}: fundamentals valid [{'; '.join(r.errors) or 'ok'}]", r.passed))

        if div_path.exists():
            df = pd.read_parquet(div_path)
            r = validate.validate_dividends(df)
            checks.append((f"{ticker}: dividends valid [{'; '.join(r.errors) or 'ok'}]", r.passed))

    failed = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failed += not ok

    print()
    if failed:
        print(f"VALIDATION FAILED: {failed} check(s)")
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
