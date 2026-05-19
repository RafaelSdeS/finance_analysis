"""
fetch_company_info.py
=====================

Busca informações das empresas na API da BolsAI
utilizando os tickers já coletados anteriormente
na pasta data/raw/prices.

Salva:
    data/raw/company_info/company_info.parquet

Uso:
    python fetch_company_info.py --api-key SUA_CHAVE
"""

from pathlib import Path
from time import sleep
import argparse
import requests
import pandas as pd


BASE_URL = "https://api.usebolsai.com/api/v1/companies/"

# Columns the API actually returns (no subsector, segment, etc.)
COMPANY_FIELDS = [
    "ticker",
    "ticker_primary",
    "corporate_name",
    "trade_name",
    "cvm_code",
    "cnpj",
    "sector",
    "status",
]


# =============================================================================
# LOAD TICKERS
# =============================================================================

def load_tickers(prices_dir: Path) -> list[str]:
    return sorted(
        file.stem.upper()
        for file in prices_dir.glob("*.parquet")
    )


# =============================================================================
# SEARCH CANDIDATES FROM API
# =============================================================================

def search_companies(search_term: str, api_key: str) -> list[dict]:
    response = requests.get(
        BASE_URL,
        headers={"X-API-Key": api_key},
        params={"search": search_term, "limit": 20},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


# =============================================================================
# FETCH COMPANY DATA — tries multiple search terms to handle edge cases
# =============================================================================

def fetch_company_data(ticker: str, api_key: str) -> dict | None:
    """
    Tries search terms of decreasing length until an exact ticker match
    is found. Handles short tickers (e.g. WEGE3 → 'wege', 'weg') and
    tickers where the first 3 chars aren't specific enough.

    Search term candidates (in order):
        - ticker without digit suffix: PETR4 → 'petr', WEGE3 → 'wege'
        - first 3 chars:              PETR4 → 'pet',  CASH3 → 'cas'
        - first 2 chars (last resort)
    """

    # Strip trailing digit(s) to get the base name
    base = ticker.rstrip("0123456789")  # PETR4 → PETR, WEGE3 → WEGE

    search_candidates = dict.fromkeys([
        base.lower(),           # most specific
        base[:3].lower(),       # fallback
        base[:2].lower(),       # last resort
    ])  # dict preserves insertion order and deduplicates

    for search_term in search_candidates:

        if not search_term:
            continue

        try:
            candidates = search_companies(search_term, api_key)
        except requests.RequestException as e:
            print(f"  [REQUEST ERROR] search='{search_term}': {e}")
            continue

        print(f"  search='{search_term}' → {len(candidates)} result(s)")

        for company in candidates:
            api_ticker = str(company.get("ticker_primary", "")).strip().upper()
            if api_ticker == ticker:
                print(f"  Matched: {ticker}")
                return {
                    "ticker":           ticker,
                    "ticker_primary":   company.get("ticker_primary"),
                    "corporate_name":   company.get("corporate_name"),
                    "trade_name":       company.get("trade_name"),
                    "cvm_code":         company.get("cvm_code"),
                    "cnpj":             company.get("cnpj"),
                    "sector":           company.get("sector"),
                    "status":           company.get("status"),
                }

        sleep(0.2)  # small pause between search attempts

    print(f"  [WARNING] No exact match found for {ticker}")
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--prices-dir", default="../data/raw/prices")
    parser.add_argument("--output", default="../data/raw/company_info/company_info.parquet")
    args = parser.parse_args()

    prices_dir  = Path(args.prices_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers(prices_dir)

    print("=" * 60)
    print(f"FETCHING COMPANY INFO — {len(tickers)} ticker(s)")
    print("=" * 60)

    rows = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        try:
            row = fetch_company_data(ticker, args.api_key)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"  [ERROR] {e}")
        sleep(0.3)

    df = pd.DataFrame(rows, columns=COMPANY_FIELDS)

    if not df.empty:
        df = (
            df.drop_duplicates(subset=["ticker"])
            .sort_values("ticker")
            .reset_index(drop=True)
        )

    df.to_parquet(output_path, index=False)

    print()
    print("=" * 60)
    print(f"Saved {len(df)} companies → {output_path}")
    print("=" * 60)
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()