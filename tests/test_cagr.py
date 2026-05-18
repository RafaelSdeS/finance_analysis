"""
test_cagr.py
============

Downloads Bolsai fundamentals and RECOMPUTES CAGR manually
to compare against the API values.

This helps validate:
- forward-filling
- incorrect CAGR values
- unstable earnings CAGR
- missing values

Usage:
    python test_cagr.py --api-key YOUR_KEY
    python test_cagr.py --ticker PETR4 --api-key YOUR_KEY
"""

import argparse
import math

import httpx
import pandas as pd

BASE_URL = "https://api.usebolsai.com/api/v1"


# -----------------------------------------------------------------------------
# CAGR
# -----------------------------------------------------------------------------

def compute_cagr(initial, final, years=5):

    if initial is None or final is None:
        return None

    try:
        initial = float(initial)
        final = float(final)
    except:
        return None

    # CAGR invalid for non-positive values
    if initial <= 0 or final <= 0:
        return None

    try:
        cagr = ((final / initial) ** (1 / years) - 1) * 100
        return round(cagr, 2)

    except:
        return None


# -----------------------------------------------------------------------------
# STATUS
# -----------------------------------------------------------------------------

def classify(value):

    if value is None or pd.isna(value):
        return "MISSING"

    if abs(value) > 100:
        return "EXTREME"

    return "OK"


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--api-key", required=True)

    parser.add_argument(
        "--ticker",
        default="PETR4"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=80
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # API REQUEST
    # -------------------------------------------------------------------------

    r = httpx.get(
        f"{BASE_URL}/fundamentals/{args.ticker}/history",
        headers={"X-API-Key": args.api_key},
        params={"limit": args.limit},
        timeout=30,
    )

    print(f"Status: {r.status_code}")

    if r.status_code != 200:
        print(f"Error: {r.text[:300]}")
        return

    history = r.json().get("history", [])

    if not history:
        print("No history returned.")
        return

    df = pd.DataFrame(history)

    # -------------------------------------------------------------------------
    # PREPARE
    # -------------------------------------------------------------------------

    df["reference_date"] = pd.to_datetime(df["reference_date"])

    df = df.sort_values("reference_date").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # CHECK AVAILABLE COLUMNS
    # -------------------------------------------------------------------------

    print("\nAVAILABLE COLUMNS:")
    print(df.columns.tolist())

    # -------------------------------------------------------------------------
    # YOU MAY NEED TO CHANGE THESE COLUMN NAMES
    # -------------------------------------------------------------------------

    revenue_col = "revenue"
    earnings_col = "net_income"

    if revenue_col not in df.columns:
        print(f"\nERROR: '{revenue_col}' column not found.")
        return

    if earnings_col not in df.columns:
        print(f"\nERROR: '{earnings_col}' column not found.")
        return

    # -------------------------------------------------------------------------
    # COMPUTE MANUAL CAGR
    # -------------------------------------------------------------------------

    manual_rev_cagr = []
    manual_earn_cagr = []

    for i in range(len(df)):

        # Need 5 years earlier (~20 quarters)
        prev_i = i - 20

        if prev_i < 0:
            manual_rev_cagr.append(None)
            manual_earn_cagr.append(None)
            continue

        current_rev = df.iloc[i][revenue_col]
        old_rev = df.iloc[prev_i][revenue_col]

        current_earn = df.iloc[i][earnings_col]
        old_earn = df.iloc[prev_i][earnings_col]

        rev_cagr = compute_cagr(
            old_rev,
            current_rev,
            years=5
        )

        earn_cagr = compute_cagr(
            old_earn,
            current_earn,
            years=5
        )

        manual_rev_cagr.append(rev_cagr)
        manual_earn_cagr.append(earn_cagr)

    df["manual_rev_cagr_5y"] = manual_rev_cagr
    df["manual_earn_cagr_5y"] = manual_earn_cagr

    # -------------------------------------------------------------------------
    # DISPLAY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 140)
    print("API CAGR vs MANUAL CAGR")
    print("=" * 140)

    header = (
        f"{'date':<12}"
        f"{'api_rev':>12}"
        f"{'manual_rev':>14}"
        f"{'api_earn':>14}"
        f"{'manual_earn':>16}"
        f"{'status':>12}"
    )

    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():

        date = row["reference_date"].strftime("%Y-%m-%d")

        api_rev = row.get("cagr_revenue_5y")
        api_earn = row.get("cagr_earnings_5y")

        man_rev = row.get("manual_rev_cagr_5y")
        man_earn = row.get("manual_earn_cagr_5y")

        status = classify(man_earn)

        print(
            f"{date:<12}"
            f"{str(api_rev):>12}"
            f"{str(man_rev):>14}"
            f"{str(api_earn):>14}"
            f"{str(man_earn):>16}"
            f"{status:>12}"
        )

    # -------------------------------------------------------------------------
    # DIFFERENCE ANALYSIS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 140)
    print("LARGE DIFFERENCES")
    print("=" * 140)

    found = False

    for _, row in df.iterrows():

        api = row.get("cagr_revenue_5y")
        manual = row.get("manual_rev_cagr_5y")

        if api is None or manual is None:
            continue

        try:
            diff = abs(float(api) - float(manual))

            if diff > 5:

                found = True

                print(
                    f"{row['reference_date'].strftime('%Y-%m-%d')} | "
                    f"API={api} | "
                    f"MANUAL={manual} | "
                    f"DIFF={diff:.2f}"
                )

        except:
            pass

    if not found:
        print("No major differences found.")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    print("""
Interpretation:
- Manual CAGR uses actual 5-year growth computation.
- Differences may indicate:
    * forward-filling
    * TTM calculations
    * adjusted financials
    * API preprocessing
- Missing manual CAGR means:
    * insufficient history
    * negative/zero values
    * invalid CAGR math
""")


if __name__ == "__main__":
    main()