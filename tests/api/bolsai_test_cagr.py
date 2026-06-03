"""
test_cagr.py
============
Checks cagr_earnings_5y values returned by Bolsai
across all quarters for a given ticker.

Usage:
    python test_cagr.py --api-key YOUR_KEY
    python test_cagr.py --api-key YOUR_KEY --ticker VALE3
"""

import argparse
import httpx

BASE_URL = "https://api.usebolsai.com/api/v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--ticker", default="PETR4")
    args = parser.parse_args()

    r = httpx.get(
        f"{BASE_URL}/fundamentals/{args.ticker}/history",
        headers={"X-API-Key": args.api_key},
        params={"limit": 80},
        timeout=30,
    )

    print(f"Status : {r.status_code}")

    if r.status_code != 200:
        print(f"Error  : {r.text[:300]}")
        return

    history = r.json().get("history", [])

    print(f"Ticker  : {args.ticker}")
    print(f"Quarters: {len(history)}")
    print()

    null_quarters     = []
    non_null_quarters = []

    for q in history:
        date      = q.get("reference_date", "?")
        earn_cagr = q.get("cagr_earnings_5y")
        if earn_cagr is None:
            null_quarters.append(date)
        else:
            non_null_quarters.append(date)

    # Full table
    print(f"{'reference_date':<20} {'cagr_revenue_5y':>18} {'cagr_earnings_5y':>20}")
    print("-" * 62)
    for q in history:
        date      = q.get("reference_date", "?")
        rev_cagr  = q.get("cagr_revenue_5y")
        earn_cagr = q.get("cagr_earnings_5y")
        marker    = "  ← NULL" if earn_cagr is None else ""
        print(f"{date:<20} {str(rev_cagr):>18} {str(earn_cagr):>20}{marker}")

    # Summary
    print()
    print("=" * 62)
    print(f"Total quarters    : {len(history)}")
    print(f"Non-null          : {len(non_null_quarters)}")
    print(f"Null              : {len(null_quarters)}")
    if null_quarters:
        print(f"Null range        : {null_quarters[0]}  →  {null_quarters[-1]}")
    if non_null_quarters:
        print(f"Non-null range    : {non_null_quarters[0]}  →  {non_null_quarters[-1]}")

    # Diagnosis
    print()
    if not null_quarters:
        print("✓ No nulls — cagr_earnings_5y is fully populated.")
    elif not non_null_quarters:
        print("✗ All nulls — Bolsai has no cagr_earnings_5y for this ticker.")
    else:
        # Check if nulls are at the start (expected) or scattered (unexpected)
        last_null_idx   = history.index(next(q for q in history if q.get("reference_date") == null_quarters[-1]))
        first_nonnull_idx = history.index(next(q for q in history if q.get("reference_date") == non_null_quarters[0]))

        if last_null_idx < first_nonnull_idx:
            print("✓ Nulls are only in early quarters — expected behavior.")
            print("  Bolsai needs 5 years of earnings history to compute the CAGR.")
        else:
            print("✗ Nulls are scattered — unexpected, may be a data quality issue.")


if __name__ == "__main__":
    main()