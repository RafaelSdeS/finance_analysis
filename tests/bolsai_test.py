"""
bolsai_test.py
==============
Testa todos os endpoints usados no bolsai_ml_dataset.py
usando apenas 3 tickers conhecidos (WEGE3, PETR4, VALE3).

Imprime o schema real de cada response para validar
campos, chaves e limites antes de rodar o script completo.

Uso:
  python bolsai_test.py --api-key sk_SUA_CHAVE
"""

import httpx
import json
import argparse

BASE_URL  = "https://api.usebolsai.com/api/v1"
TEST_TICKERS = ["WEGE3", "PETR4", "VALE3"]

def get(client, path, params=None):
    url = f"{BASE_URL}/{path.lstrip('/')}"
    r = client.get(url, params=params or {})
    print(f"  [{r.status_code}] GET {url} params={params}")
    if r.status_code == 200:
        return r.json()
    else:
        print(f"  ERROR: {r.text[:200]}")
        return None

def show(data, max_records=2):
    """Imprime estrutura resumida do response."""
    if data is None:
        return
    # Chaves de topo
    top_keys = list(data.keys())
    print(f"  Top-level keys: {top_keys}")
    # Para cada chave que é lista, mostra 1 elemento
    for k, v in data.items():
        if isinstance(v, list) and v:
            print(f"  '{k}' (list, {len(v)} items) → primeiro elemento:")
            print(f"    {json.dumps(v[0], ensure_ascii=False, default=str)[:300]}")
        elif not isinstance(v, (dict, list)):
            print(f"  '{k}': {v}")
    print()

def run(api_key):
    client = httpx.Client(
        timeout=15,
        headers={"X-API-Key": api_key},
        follow_redirects=True,
    )
    ticker = TEST_TICKERS[0]  # WEGE3 para maioria dos testes

    print("=" * 60)
    print("1. LISTA DE STOCKS (paginação)")
    print("=" * 60)
    data = get(client, "/stocks/", {"limit": 10, "offset": 0})
    show(data)

    print("=" * 60)
    print("2. FUNDAMENTALS HISTORY")
    print("=" * 60)
    data = get(client, f"/fundamentals/{ticker}/history", {"limit": 2})
    show(data)

    print("=" * 60)
    print("3. PRICE HISTORY")
    print("=" * 60)
    data = get(client, f"/stocks/{ticker}/history", {"limit": 2})
    show(data)
    # Testa também limit=80 para confirmar o máximo
    data80 = get(client, f"/stocks/{ticker}/history", {"limit": 80})
    if data80:
        prices = data80.get("prices", [])
        print(f"  limit=80 → {len(prices)} registros retornados")
    print()

    print("=" * 60)
    print("4. DIVIDENDOS")
    print("=" * 60)
    data = get(client, f"/stocks/{ticker}/dividends", {"limit": 5})
    show(data)

    print("=" * 60)
    print("5. FINANCIALS (DFP / DRE)")
    print("=" * 60)
    data = get(client, f"/financials/{ticker}",
               {"report_type": "DFP", "statement_type": "DRE", "limit": 2})
    show(data)

    print("=" * 60)
    print("6. MACRO — SELIC")
    print("=" * 60)
    data = get(client, "/macro/selic", {"limit": 5})
    show(data)

    print("=" * 60)
    print("7. LIMITE MÁXIMO — descoberta automática")
    print("=" * 60)
    for lim in [80, 100, 200, 500]:
        r = client.get(
            f"{BASE_URL}/fundamentals/{ticker}/history",
            params={"limit": lim}
        )
        status = "✓ OK" if r.status_code == 200 else f"✗ {r.status_code}"
        count = r.json().get("count", "?") if r.status_code == 200 else "-"
        print(f"  limit={lim:4d} → {status}  (count={count})")
    print()

    print("=" * 60)
    print("8. TICKERS PROBLEMÁTICOS — validação do filtro")
    print("=" * 60)
    bad_tickers = ["ABC 5", "AAP 4", "INEXISTENTE99"]
    for t in bad_tickers:
        r = client.get(
            f"{BASE_URL}/fundamentals/{t}/history",
            params={"limit": 2}
        )
        print(f"  ticker='{t}' → {r.status_code}")
    print()

    print("=" * 60)
    print("9. MÚLTIPLOS TICKERS — smoke test completo")
    print("=" * 60)
    for t in TEST_TICKERS:
        r = client.get(f"{BASE_URL}/fundamentals/{t}/history", params={"limit": 2})
        if r.status_code == 200:
            d = r.json()
            count = d.get("count", "?")
            name  = d.get("corporate_name", "?")
            print(f"  {t}: ✓  corporate_name='{name}'  count={count}")
        else:
            print(f"  {t}: ✗  status={r.status_code}")
    print()

    client.close()
    print("Teste concluído.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True)
    args = p.parse_args()
    run(args.api_key)