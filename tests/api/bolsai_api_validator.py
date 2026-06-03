#!/usr/bin/env python3
"""
bolsai_api_validator.py
=======================

Validates the BolsaAI endpoints used by the ML pipeline.

Checks:
    1. Stocks listing
    2. Fundamentals history
    3. Price history
    4. Dividends
    5. Financial statements
    6. Macro data
    7. Endpoint limits
    8. Invalid tickers
    9. Multiple ticker smoke test

Usage:
    python bolsai_api_validator.py --api-key sk_xxxxxxxxx
"""

import argparse
import json
import httpx

BASE_URL = "https://api.usebolsai.com/api/v1"

TEST_TICKERS = [
    "WEGE3",
    "PETR4",
    "VALE3",
]


def get(client, path, params=None):
    url = f"{BASE_URL}/{path.lstrip('/')}"

    try:
        r = client.get(url, params=params or {})

        print(
            f"[{r.status_code}] "
            f"GET {r.request.url}"
        )

        if r.status_code == 200:
            return r.json()

        print("\nERROR RESPONSE:")
        print(r.text[:1000])
        print()

        return None

    except Exception as e:
        print(f"REQUEST FAILED: {e}")
        return None


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def show_response(data):

    if not data:
        return

    print("\nTop-level keys:")
    print(list(data.keys()))

    print()

    for key, value in data.items():

        if isinstance(value, list):

            print(f"{key}: list ({len(value)} items)")

            if value:
                print(
                    json.dumps(
                        value[0],
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )[:800]
                )

            print()

        elif isinstance(value, dict):

            print(f"{key}: dict")
            print(
                json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )[:800]
            )
            print()

        else:
            print(f"{key}: {value}")

    print()


def show_dividends(data):

    if not data:
        return

    print("\nDividend Summary")
    print("-" * 40)

    fields = [
        "ticker",
        "dividend_yield_ttm",
        "ttm_per_share",
        "current_price",
        "total_payments",
    ]

    for field in fields:
        if field in data:
            print(f"{field}: {data[field]}")

    annual = data.get("annual_summary", [])
    payments = data.get("payments", [])

    print()
    print(f"annual_summary rows: {len(annual)}")
    print(f"payments rows:       {len(payments)}")

    if annual:
        print("\nFirst annual summary row:")
        print(
            json.dumps(
                annual[0],
                indent=2,
                ensure_ascii=False,
            )
        )

    if payments:
        print("\nFirst payment:")
        print(
            json.dumps(
                payments[0],
                indent=2,
                ensure_ascii=False,
            )
        )


def run(api_key):

    client = httpx.Client(
        timeout=20,
        headers={
            "X-API-Key": api_key
        },
        follow_redirects=True,
    )

    ticker = TEST_TICKERS[0]

    # ------------------------------------------------------------------
    # STOCKS
    # ------------------------------------------------------------------

    print_header("1. STOCKS LIST")

    data = get(
        client,
        "/stocks/",
        {
            "limit": 10,
            "offset": 0,
        },
    )

    show_response(data)

    # ------------------------------------------------------------------
    # FUNDAMENTALS
    # ------------------------------------------------------------------

    print_header("2. FUNDAMENTALS HISTORY")

    data = get(
        client,
        f"/fundamentals/{ticker}/history",
        {
            "limit": 2,
        },
    )

    show_response(data)

    # ------------------------------------------------------------------
    # PRICE HISTORY
    # ------------------------------------------------------------------

    print_header("3. PRICE HISTORY")

    data = get(
        client,
        f"/stocks/{ticker}/history",
        {
            "limit": 2,
        },
    )

    show_response(data)

    data80 = get(
        client,
        f"/stocks/{ticker}/history",
        {
            "limit": 80,
        },
    )

    if data80:

        prices = data80.get("prices", [])

        print(
            f"\nlimit=80 returned "
            f"{len(prices)} records"
        )

    # ------------------------------------------------------------------
    # DIVIDENDS
    # ------------------------------------------------------------------

    print_header("4. DIVIDENDS")

    data = get(
        client,
        f"/dividends/{ticker}",
        {
            "years": 5,
        },
    )

    show_dividends(data)

    # ------------------------------------------------------------------
    # FINANCIALS
    # ------------------------------------------------------------------

    print_header("5. FINANCIAL STATEMENTS")

    data = get(
        client,
        f"/financials/{ticker}",
        {
            "report_type": "DFP",
            "statement_type": "DRE",
            "limit": 2,
        },
    )

    show_response(data)

    # ------------------------------------------------------------------
    # MACRO
    # ------------------------------------------------------------------

    print_header("6. MACRO - SELIC")

    data = get(
        client,
        "/macro/selic",
        {
            "limit": 5,
        },
    )

    show_response(data)

    # ------------------------------------------------------------------
    # LIMIT TEST
    # ------------------------------------------------------------------

    print_header("7. FUNDAMENTALS LIMIT DISCOVERY")

    for limit in [20, 40, 80, 100, 200, 500]:

        r = client.get(
            f"{BASE_URL}/fundamentals/{ticker}/history",
            params={
                "limit": limit
            }
        )

        if r.status_code == 200:

            payload = r.json()

            print(
                f"limit={limit:4d} "
                f"✓ count={payload.get('count', '?')}"
            )

        else:

            print(
                f"limit={limit:4d} "
                f"✗ status={r.status_code}"
            )

    # ------------------------------------------------------------------
    # BAD TICKERS
    # ------------------------------------------------------------------

    print_header("8. INVALID TICKERS")

    bad_tickers = [
        "ABC5",
        "AAP4",
        "INEXISTENTE99",
    ]

    for bad in bad_tickers:

        r = client.get(
            f"{BASE_URL}/fundamentals/{bad}/history",
            params={"limit": 2},
        )

        print(
            f"{bad:<15} "
            f"status={r.status_code}"
        )

    # ------------------------------------------------------------------
    # SMOKE TEST
    # ------------------------------------------------------------------

    print_header("9. MULTIPLE TICKER SMOKE TEST")

    for ticker in TEST_TICKERS:

        r = client.get(
            f"{BASE_URL}/fundamentals/{ticker}/history",
            params={
                "limit": 2
            },
        )

        if r.status_code == 200:

            d = r.json()

            print(
                f"{ticker:<8} "
                f"✓ "
                f"name='{d.get('corporate_name', '?')}' "
                f"count={d.get('count', '?')}"
            )

        else:

            print(
                f"{ticker:<8} "
                f"✗ status={r.status_code}"
            )

    client.close()

    print()
    print("=" * 80)
    print("VALIDATION FINISHED")
    print("=" * 80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Validate BolsaAI endpoints"
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="BolsaAI API key",
    )

    args = parser.parse_args()

    run(args.api_key)