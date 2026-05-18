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


# =============================================================================
# LOAD TICKERS
# =============================================================================

def load_tickers(prices_dir: Path):
    """
    Descobre os tickers a partir dos arquivos parquet
    dentro de data/raw/prices.
    """

    tickers = []

    for file in prices_dir.glob("*.parquet"):
        tickers.append(file.stem.upper())

    return sorted(set(tickers))


# =============================================================================
# FETCH COMPANY DATA
# =============================================================================

def fetch_company_data(ticker: str, api_key: str):

    headers = {
        "X-API-Key": api_key
    }

    # PETR4 -> "petr"
    search_term = ticker[:3].lower()

    params = {
        "search": search_term,
        "limit": 20
    }

    response = requests.get(
        BASE_URL,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    json_data = response.json()

    data = json_data.get("data", [])

    print()
    print(f"Ticker: {ticker}")
    print(f"Search term: {search_term}")
    print(f"Results returned: {len(data)}")

    if not data:
        print("No data returned.")
        return None

    # DEBUG
    for company in data:

        api_ticker = company.get("ticker_primary")

        print(f"API returned ticker: {api_ticker}")

    # =========================================================
    # Find exact ticker
    # =========================================================

    for company in data:

        api_ticker = (
            str(company.get("ticker_primary", ""))
            .strip()
            .upper()
        )

        local_ticker = ticker.strip().upper()

        if api_ticker == local_ticker:

            print(f"Matched ticker: {ticker}")

            return {
                "ticker": ticker,
                "ticker_primary": company.get("ticker_primary"),
                "corporate_name": company.get("corporate_name"),
                "trade_name": company.get("trade_name"),
                "cvm_code": company.get("cvm_code"),
                "cnpj": company.get("cnpj"),
                "sector": company.get("sector"),
                "status": company.get("status"),
            }

    print(f"[WARNING] Exact match not found for {ticker}")

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api-key",
        required=True,
        help="BolsAI API Key"
    )

    parser.add_argument(
        "--prices-dir",
        default="data/raw/prices",
        help="Folder containing price parquet files"
    )

    parser.add_argument(
        "--output",
        default="data/raw/company_info/company_info.parquet",
        help="Output parquet file"
    )

    args = parser.parse_args()

    prices_dir = Path(args.prices_dir)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================
    # Load tickers
    # =========================================================

    tickers = load_tickers(prices_dir)

    print("=" * 80)
    print("FETCHING COMPANY INFO")
    print("=" * 80)
    print(f"Tickers found: {len(tickers)}")
    print()

    rows = []

    # =========================================================
    # Fetch data
    # =========================================================

    for i, ticker in enumerate(tickers, start=1):

        print("-" * 80)
        print(f"[{i}/{len(tickers)}] Processing {ticker}")

        try:

            row = fetch_company_data(
                ticker=ticker,
                api_key=args.api_key
            )

            if row is not None:
                rows.append(row)

        except requests.HTTPError as e:

            print(f"[HTTP ERROR] {ticker}")
            print(e)

        except requests.RequestException as e:

            print(f"[REQUEST ERROR] {ticker}")
            print(e)

        except Exception as e:

            print(f"[UNKNOWN ERROR] {ticker}")
            print(e)

        # evita spam na API
        sleep(0.3)

    # =========================================================
    # Build dataframe
    # =========================================================

    df = pd.DataFrame(rows)

    # remove duplicados
    if not df.empty:

        df = (
            df
            .drop_duplicates(subset=["ticker"])
            .sort_values("ticker")
            .reset_index(drop=True)
        )

    # =========================================================
    # Save parquet
    # =========================================================

    df.to_parquet(
        output_path,
        index=False
    )

    # =========================================================
    # Final summary
    # =========================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"Companies collected: {len(df)}")
    print(f"Saved to: {output_path}")

    print()

    if not df.empty:

        print(df.to_string(index=False))

    else:

        print("No company data collected.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()