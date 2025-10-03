# Finance Analysis

Este projeto é um script em Python para análise de ações brasileiras, comparando o desempenho de uma ação (como BBAS3) com a taxa SELIC e ajustando valores pela inflação (IPCA). Ele gera gráficos interativos usando Plotly.

## Funcionalidades

- Baixa dados históricos de ações usando `yfinance`.
- Calcula preço ajustado pela inflação (IPCA).
- Calcula total return com reinvestimento de dividendos.
- Compara com a SELIC mensal acumulada.
- Plota gráficos interativos comparando ação vs SELIC.

## Requisitos

Instale os pacotes necessários com:

```bash
pip install -r requirements.txt
