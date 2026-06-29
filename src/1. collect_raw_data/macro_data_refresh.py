import pandas as pd
import requests
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/raw/macro"

END_DATE = "2016-12-31"
DATE_COL = "reference_date"


BCB_SERIES = {
    "ipca": 433,
    "selic": 11,
    "cdi": 12
}


# -----------------------------
# FETCH BCB SGS (NO API KEY)
# -----------------------------
def fetch_bcb(series_id: int, start="1990-01-01", end=END_DATE):
    import pandas as pd
    import requests

    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    all_data = []

    chunk_size_years = 5
    current = start

    while current < end:
        chunk_end = min(current + pd.DateOffset(years=chunk_size_years), end)

        params = {
            "formato": "json",
            "dataInicial": current.strftime("%d/%m/%Y"),
            "dataFinal": chunk_end.strftime("%d/%m/%Y"),
        }

        r = requests.get(url, params=params, timeout=30)

        if r.status_code == 406:
            # fallback: reduz chunk (BCB é instável)
            chunk_size_years = max(1, chunk_size_years // 2)
            continue

        r.raise_for_status()

        data = r.json()

        if data:
            all_data.extend(data)

        current = chunk_end + pd.Timedelta(days=1)

    if not all_data:
        return pd.DataFrame(columns=[DATE_COL, "value"])

    df = pd.DataFrame(all_data)
    df.columns = [DATE_COL, "value"]

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True)

    df["value"] = (
        df["value"]
        .str.replace(",", ".", regex=False)
    )

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL)

    return df


# -----------------------------
# MERGE SAFE BACKFILL
# -----------------------------
def merge_backfill(df_local, df_external, col_name):
    df_local = df_local.copy()

    df_local[DATE_COL] = pd.to_datetime(df_local[DATE_COL])

    # garante limite histórico externo
    df_external = df_external[df_external[DATE_COL] <= END_DATE]

    # renomeia coluna de valor
    df_external = df_external.rename(columns={"value": col_name})

    # merge
    merged = pd.concat([df_external, df_local], ignore_index=True)

    # remove duplicatas por data (prioriza último valor carregado)
    merged = merged.sort_values(DATE_COL)
    merged = merged.drop_duplicates(subset=[DATE_COL], keep="last")

    return merged


# -----------------------------
# PROCESS FILE
# -----------------------------
def process_file(path: Path, key: str):
    print(f"Processando {path}")

    df = pd.read_parquet(path)

    if DATE_COL not in df.columns:
        raise ValueError(f"{path} não tem coluna '{DATE_COL}'")

    series_id = BCB_SERIES[key]

    external = fetch_bcb(series_id)

    df = merge_backfill(df, external, key)

    df.to_parquet(path, index=False)

    print(f"[OK] atualizado: {path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    for key in ["ipca", "selic", "cdi"]:
        path = DATA_DIR / f"{key}.parquet"

        if not path.exists():
            print(f"[SKIP] não encontrado: {path}")
            continue

        process_file(path, key)


if __name__ == "__main__":
    main()