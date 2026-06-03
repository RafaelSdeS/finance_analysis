"""
bolsai_api_price_depth.py
=========================

Tests how far back the Bolsai API returns price history for a given ticker.

Usage:
    python bolsai_api_price_depth.py \
        --api-key YOUR_API_KEY \
        --ticker VALE3
"""

import argparse
from datetime import datetime, timedelta

import httpx

BASE_URL = "https://api.usebolsai.com/api/v1"
LIMIT = 80


def main():

    parser = argparse.ArgumentParser(
        description="Test Bolsai price history depth."
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="Bolsai API key"
    )

    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol (e.g. PETR4, VALE3, WEGE3)"
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()

    client = httpx.Client(
        timeout=30,
        headers={"X-API-Key": args.api_key},
        follow_redirects=True,
    )

    print(f"Testing price history depth for {ticker}...")
    print()

    all_dates = []
    current_end = None
    page = 1

    while True:

        params = {
            "start": "1900-01-01",
            "limit": LIMIT,
        }

        if current_end:
            params["end"] = current_end

        print(f"Page {page}: {params}")

        try:
            response = client.get(
                f"{BASE_URL}/stocks/{ticker}/history",
                params=params,
            )

        except Exception as e:
            print(f"Request failed: {e}")
            break

        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print(response.text[:500])
            break

        data = response.json()
        prices = data.get("prices", [])

        if not prices:
            print("No more records returned.")
            break

        dates = [p["trade_date"] for p in prices]

        oldest = min(dates)
        newest = max(dates)

        print(
            f"Received {len(prices)} records "
            f"({oldest} -> {newest})"
        )

        all_dates.extend(dates)

        # If fewer than LIMIT records were returned,
        # we've probably reached the last page.
        if len(prices) < LIMIT:
            print("Last page reached.")
            break

        oldest_dt = datetime.strptime(oldest, "%Y-%m-%d")

        current_end = (
            oldest_dt - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        page += 1

    client.close()

    unique_dates = sorted(set(all_dates))

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)

    if unique_dates:
        print(f"Ticker        : {ticker}")
        print(f"Total records : {len(unique_dates)}")
        print(f"Earliest date : {unique_dates[0]}")
        print(f"Latest date   : {unique_dates[-1]}")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()