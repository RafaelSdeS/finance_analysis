"""
bolsai_api_macro_depth.py
=========================

Tests how far back the Bolsai API returns macroeconomic data
for SELIC, CDI, IPCA, or any supported macro series.

Usage:
    python bolsai_api_macro_depth.py \
        --api-key YOUR_API_KEY \
        --series selic

    python bolsai_api_macro_depth.py \
        --api-key YOUR_API_KEY \
        --series cdi

    python bolsai_api_macro_depth.py \
        --api-key YOUR_API_KEY \
        --series ipca
"""

import argparse
from datetime import datetime, timedelta

import httpx

BASE_URL = "https://api.usebolsai.com/api/v1"
LIMIT = 5000

VALID_SERIES = [
    "selic",
    "selic_target",
    "ipca",
    "cdi",
    "usd_brl",
    "eur_brl",
]


def main():

    parser = argparse.ArgumentParser(
        description="Test Bolsai macro series depth."
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="Bolsai API key"
    )

    parser.add_argument(
        "--series",
        required=True,
        choices=VALID_SERIES,
        help=(
            "Macro series name "
            "(selic, selic_target, ipca, cdi, usd_brl, eur_brl)"
        )
    )

    args = parser.parse_args()

    series = args.series.lower()

    client = httpx.Client(
        timeout=30,
        headers={"X-API-Key": args.api_key},
        follow_redirects=True,
    )

    print(f"Testing macro history depth for '{series}'...")
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

        print(f"Batch {page}: {params}")

        try:
            response = client.get(
                f"{BASE_URL}/macro/{series}",
                params=params,
            )

        except Exception as e:
            print(f"Request failed: {e}")
            break

        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print(response.text[:1000])
            break

        payload = response.json()

        records = payload.get("data", [])

        if not records:
            print("No more records returned.")
            break

        dates = [r["date"] for r in records]

        oldest = min(dates)
        newest = max(dates)

        print(
            f"Received {len(records)} records "
            f"({oldest} -> {newest})"
        )

        all_dates.extend(dates)

        if len(records) < LIMIT:
            print("Last batch reached.")
            break

        oldest_dt = datetime.strptime(
            oldest,
            "%Y-%m-%d"
        )

        current_end = (
            oldest_dt - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        page += 1

    client.close()

    unique_dates = sorted(set(all_dates))

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    if unique_dates:
        print(f"Series        : {series}")
        print(f"Total records : {len(unique_dates)}")
        print(f"Earliest date : {unique_dates[0]}")
        print(f"Latest date   : {unique_dates[-1]}")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()