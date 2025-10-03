# --- Imports ---
from bcb import sgs
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ============================
# Funções
# ============================

# --- Função para baixar dados de ações ---
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

# --- Função para baixar Selic acumulada no mês ---
def get_selic(start="2000-01-01", end="2024-12-31"):
    selic = sgs.get({'SELIC': 4390}, start=start, end=end)
    selic['SELIC'] = selic['SELIC'] / 100 + 1
    return selic

# --- Ajuste pela inflação ---
def adjust_for_inflation(hist, ipca):
    ipca_index = ipca['IPCA_acumulado'].reindex(hist.index).interpolate(method='linear')
    ipca_index_real = ipca_index.iloc[-1] / ipca_index
    hist['Close_real'] = hist['Close'] * ipca_index_real
    return hist, ipca_index_real

# --- Total return (dividendos reinvestidos) ---
def compute_total_return(hist, dividends, ipca_index_real):
    dividends_adjusted = dividends.reindex(hist.index, fill_value=0) * ipca_index_real
    shares = 1 + (dividends_adjusted / hist['Close_real']).cumsum()
    hist['Total_return'] = hist['Close_real'] * shares
    return hist

# --- Selic aplicada mensalmente sobre o valor inicial da ação ---
def apply_selic_monthly(hist, selic):
    # Resample mensal (fim do mês) e preencher valores
    selic_monthly = selic['SELIC'].resample('ME').mean().ffill()
    
    # Índice acumulado (composto)
    selic_index = selic_monthly.cumprod()
    
    # Aplicar sobre o valor inicial da ação
    selic_aplicada = hist['Close'].iloc[0] * selic_index / selic_index.iloc[0]
    
    # Reindexar para o mesmo índice do histórico da ação
    selic_aplicada = selic_aplicada.reindex(hist.index, method='ffill')
    
    return selic_aplicada

# ============================
# Execução
# ============================

# 1. Baixar dados
hist, dividends, splits = get_stock_data("BBAS3.SA")
ipca = get_ipca()
selic = get_selic()

# 2. Ajuste pela inflação
hist, ipca_index_real = adjust_for_inflation(hist, ipca)

# 3. Total return
hist = compute_total_return(hist, dividends, ipca_index_real)

# 4. Selic aplicada mensalmente sobre o valor inicial da ação
selic_aplicada = apply_selic_monthly(hist, selic)

# ============================
# Gráfico interativo
# ============================

fig = go.Figure()

fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Preço Nominal'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['Close_real'], mode='lines', name='Preço Ajustado pela Inflação'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['Total_return'], mode='lines', name='Total Return (Dividendos Reinvestidos)'))
fig.add_trace(go.Scatter(x=hist.index, y=selic_aplicada, mode='lines', name='Selic Mensal Aplicada'))

fig.update_layout(
    title="BBAS3: Preço Nominal, Ajustado, Total Return e Selic Mensal",
    xaxis_title="Data",
    yaxis_title="R$",
    template="plotly_white",
    hovermode="x unified"
)

fig.show()
