# --- Imports ---
from bcb import sgs
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Função para baixar e preparar dados de ações ---
def get_stock_data(ticker_symbol, start="2000-01-01", end="2025-01-01"):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(start=start, end=end)[['Close']]
    hist.index = hist.index.tz_localize(None)

    dividends = ticker.dividends
    dividends.index = dividends.index.tz_localize(None)

    splits = ticker.splits
    splits.index = splits.index.tz_localize(None)
    
    return hist, dividends, splits

# --- Função para baixar IPCA ---
def get_ipca(start="2000-01-01", end="2024-12-31"):
    ipca = sgs.get({'IPCA': 433}, start=start, end=end)
    ipca['IPCA'] = ipca['IPCA'] / 100 + 1
    ipca['IPCA_acumulado'] = ipca['IPCA'].cumprod()
    return ipca

# --- Ajuste pela inflação ---
def adjust_for_inflation(hist, ipca):
    ipca_index = ipca['IPCA_acumulado'].reindex(hist.index).interpolate(method='linear')
    ipca_index_real = ipca_index.iloc[-1] / ipca_index
    hist['Close_real'] = hist['Close'] * ipca_index_real
    return hist, ipca_index_real

# --- Total return ---
def compute_total_return(hist, dividends, ipca_index_real):
    dividends_adjusted = dividends.reindex(hist.index, fill_value=0) * ipca_index_real
    shares = 1 + (dividends_adjusted / hist['Close_real']).cumsum()
    hist['Total_return'] = hist['Close_real'] * shares
    return hist

# ============================
# Execução
# ============================

# 1. Dados
hist, dividends, splits = get_stock_data("BBAS3.SA")
ipca = get_ipca()

# 2. Ajuste inflação
hist, ipca_index_real = adjust_for_inflation(hist, ipca)

# 3. Total return
hist = compute_total_return(hist, dividends, ipca_index_real)

# ============================
# Gráfico interativo
# ============================

fig = go.Figure()

fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Preço Nominal'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['Close_real'], mode='lines', name='Preço Ajustado pela Inflação'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['Total_return'], mode='lines', name='Total Return (Dividendos Reinvestidos'))

fig.update_layout(
    title="BBAS3: Preço Nominal, Ajustado e Total Return",
    xaxis_title="Data",
    yaxis_title="R$",
    template="plotly_white",
    hovermode="x unified"
)

fig.show()

