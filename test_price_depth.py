"""
test_price_depth.py
===================
Tests how far back the Bolsai API returns price history.

Usage:
    python test_price_depth.py --api-key YOUR_KEY
"""

import argparse
import httpx

BASE_URL = "https://api.usebolsai.com/api/v1"
TICKER = "PETR4"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    client = httpx.Client(
        timeout=30,
        headers={"X-API-Key": args.api_key},
        follow_redirects=True,
    )

    print(f"Testing price history depth for {TICKER}...")
    print()

    r = client.get(
        f"{BASE_URL}/stocks/{TICKER}/history",
        params={"start": "2000-01-01", "limit": 80},
    )

    print(f"Status: {r.status_code}")

    if r.status_code != 200:
        print(f"Error: {r.text[:300]}")
        client.close()
        return

    data = r.json()
    prices = data.get("prices", [])

    print(f"Records returned: {len(prices)}")

    if prices:
        print(f"Earliest date  : {prices[0]['trade_date']}")
        print(f"Latest date    : {prices[-1]['trade_date']}")
    else:
        print("No prices returned.")

    client.close()


if __name__ == "__main__":
    main()