"""
cvm/filing_dates.py — CVM filing dates (real publication date per quarter).

Downloads the CVM open-data filing registers (ITR quarterly, DFP annual) and
extracts, for every company and fiscal quarter, the date CVM actually
*received* the filing (DT_RECEB) — i.e. the real publication date of the
numbers, as opposed to the fiscal quarter-end (DT_REFER) that BolsAI reports
as `reference_date`.

Stage 2 uses this to make fundamentals visible in the dataset only from their
true release date (measured Q1-2025: median lag 44 days, but 8.6% of companies
file late, up to 443 days — a fixed statutory buffer can't cover those).

Output:
    data/raw/filing_dates/filing_dates.parquet
        cnpj (digits only), cvm_code, reference_date, received_date, report_type

Run once, then re-run quarterly (only missing/current years are downloaded):
    python -m src.data_collection.cvm_statements --step filing_dates
"""

from datetime import date

import pandas as pd

from .. import config
from . import http

OUTPUT_PATH = config.RAW_DIR / "filing_dates/filing_dates.parquet"


def _fetch_year(report_type: str, year: int) -> pd.DataFrame | None:
    """One year's ITR/DFP register -> (cnpj, cvm_code, reference_date, received_date)."""
    zf = http.fetch_zip(report_type.upper(), year)
    if zf is None:
        return None
    try:
        register_name = f"{report_type.lower()}_cia_aberta_{year}.csv"
        rows = http.read_csv(zf, register_name)
        if not rows:
            print(f"  {report_type} {year}: empty/missing register")
            return None

        df = pd.DataFrame({
            "cnpj": [r.get("CNPJ_CIA", "") for r in rows],
            "cvm_code": [r.get("CD_CVM", "") for r in rows],
            "reference_date": [r.get("DT_REFER", "") for r in rows],
            "received_date": [r.get("DT_RECEB", "") for r in rows],
        })
        df["cnpj"] = df["cnpj"].str.replace(r"\D", "", regex=True)
        df["reference_date"] = pd.to_datetime(df["reference_date"], errors="coerce")
        df["received_date"] = pd.to_datetime(df["received_date"], errors="coerce")
        df = df.dropna(subset=["cnpj", "reference_date", "received_date"])

        if df.empty:
            print(f"  {report_type} {year}: no valid rows after parsing")
            return None

        # First public availability: earliest receipt across filing versions
        # (later versions are restatements — the market saw the numbers at v1)
        df = (
            df.groupby(["cnpj", "cvm_code", "reference_date"], as_index=False)["received_date"]
            .min()
        )
        df["report_type"] = report_type.upper()
        return df
    except Exception as e:
        print(f"  {report_type} {year}: unexpected error: {e}")
        return None


def collect_filing_dates() -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current_year = date.today().year

    existing = pd.read_parquet(OUTPUT_PATH) if OUTPUT_PATH.exists() else None
    covered: set[tuple[str, int]] = set()
    if existing is not None:
        covered = {
            (t, y)
            for t, y in zip(existing["report_type"], existing["reference_date"].dt.year)
        }

    total_collected = 0
    total_companies = set()

    for report_type in ("itr", "dfp"):
        for year in range(http.START_YEAR, current_year + 1):
            key = (report_type.upper(), year)
            # current year is always refreshed — new filings arrive all quarter
            if key in covered and year < current_year:
                print(f"  {report_type} {year}: already collected, skipping")
                continue
            df = _fetch_year(report_type, year)
            if df is None:
                continue
            print(f"  {report_type} {year}: {len(df)} filings, saving immediately...")

            # Remove any existing rows for this (type, year) before appending
            if existing is not None:
                existing = existing[~(
                    (existing["report_type"] == key[0])
                    & (existing["reference_date"].dt.year == year)
                )]

            # Append to existing, deduplicate, and save immediately
            if existing is not None:
                result = pd.concat([existing, df], ignore_index=True)
            else:
                result = df.copy()

            result = (
                result.drop_duplicates(
                    subset=["cnpj", "reference_date", "report_type"], keep="last"
                )
                .sort_values(["cnpj", "reference_date"])
                .reset_index(drop=True)
            )

            # Sanity gate: drop corrupted rows
            bad = result["received_date"] < result["reference_date"]
            if bad.any():
                print(f"    WARNING: dropping {bad.sum()} corrupted rows "
                      f"(received before quarter-end)")
                result = result[~bad]

            result.to_parquet(OUTPUT_PATH, index=False)
            existing = result
            total_collected += len(df)
            total_companies.update(result["cnpj"].unique())
            print(f"    ✓ {len(result)} unique filings so far, "
                  f"{len(total_companies)} companies")

    print(f"\nFinal: {len(result)} filings ({len(total_companies)} companies, "
          f"{result['reference_date'].min().date()} → {result['reference_date'].max().date()}) "
          f"saved to {OUTPUT_PATH}")
    return result
