""" Create a CSV with daily data of Brazilian companies in the Ibovespa. It uses "relative" indicators. It includes:

1. Market Data (per asset):
- Historical prices (OHLC)
- Returns
- Volume
- Technical indicators:
  - Momentum
  - Moving averages
  - Volatility
  - Drawdowns

2. Fundamental Data (for equities only):
- P/E (Price/Earnings)
- P/B (Price/Book)
- ROE
- Dividend Yield
- Growth metrics
- Other data may also be included

3. Macro / Context Features:
- Risk-free rate (e.g., Selic/CDI)
- Trend of interest rates
- Market regime (bull / bear)
- Market volatility indicators

This dataset is intended for machine learning models and quantitative analysis, combining fundamental and technical indicators over time.
"""