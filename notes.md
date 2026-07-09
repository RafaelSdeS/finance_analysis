## Model objectives

* The model should think in the long term.
* Reward should be based on outperforming the market over the long run.
* The model should learn the fundamentals of long-term investing (buy quality companies at attractive prices, avoid structurally deteriorating companies, include dividend-paying stocks when appropriate).
* The agent should consistently outperform equal-weight, SELIC, BOVA11, and a market-cap-weighted benchmark.

---

## Environment & reward

* Buying CASH shouldn't incur transaction costs.
* Create a separate training/environment for periodic investing (DCA), optionally adjusting contributions for inflation.

---

## PPO & training experiments

* Enable a small entropy bonus (e.g., `ent_coef = 0.001`).
* Experiment with different `log_std_init` values (current: `-2`).

---

## Feature engineering

* Add more features to the training data.
* Put more emphasis on percentage-based and adjusted features, their variance over time, and their relationship to the corresponding sector (absolute values are often less informative).

---

## Data quality & preprocessing

* Investigate why processed fundamentals appear as blocky quarterly steps while raw fundamentals are smooth. Determine whether this is expected or indicates a preprocessing issue.
* Verify P/L outliers in the processed data and in `earnings_growth_yoy`.
* Fix missing features:

  * `pe_ratio`
  * `pb_ratio`
  * `date`

Overall NaN rate: 15.23% (17,428,274 / 114,470,280 cells)

Critical columns (should be 0% NaN):
  ✓ ticker           0.00%
  ✓ close            0.00%
  ✓ volume           0.00%
  ✓ sector           0.00%

Columns with >20% NaN:
  cagr_earnings_5y               69.46%
  cagr_earnings_5y_final         64.35%
  cagr_revenue_5y                49.88%
  cagr_revenue_5y_final          49.50%
  total_debt_growth_yoy          30.12%
  revenue_vs_earnings_growth_delta 26.44%
  revenue_growth_yoy             26.44%
  pvp_zscore_sector              26.21%
  pl_zscore_sector               26.18%
  peg_ratio                      25.97%
  ebitda_growth_yoy              25.73%
  earnings_growth_yoy            25.73%
  total_assets_growth_yoy        25.73%
  debt_equity_zscore_sector      25.68%
  roe_zscore_sector              25.65%
  dividend_coverage_ratio        25.29%
  payout_ratio                   25.29%
  current_ratio_qoq              23.80%
  net_margin_qoq                 23.52%
  gross_margin_qoq               23.52%
  p_sr                           23.09%
  margin_trend_4q                22.71%
  working_capital_ratio          22.67%
  current_ratio                  22.67%
  cash_ratio                     22.60%
  current_liabilities            22.60%
  net_margin                     22.58%
  revenue_per_earning            22.58%
  net_revenue                    22.58%
  asset_turnover                 22.58%
  ebitda_margin                  22.58%
  gross_margin                   22.58%
  ebit_margin                    22.58%
  debt_equity_qoq                22.12%
  roe_qoq                        22.12%
  pvp_to_roe_ratio               21.74%
  earnings_yield_vs_selic        21.74%
  pl_percentile_5y               21.74%
  close_price                    21.74%
  earnings_yield                 21.74%
  p_ebit                         21.74%
  ev_ebit                        21.74%
  pl                             21.74%
  pvp                            21.74%
  market_cap                     21.74%
  ev_ebitda                      21.74%
  book_to_market                 21.74%
  p_ebitda                       21.74%
  p_assets                       21.74%
  roa_trend_4q                   21.36%
  roe_trend_4q                   21.36%
  debt_trend_4q                  21.36%
  current_assets                 21.29%
  roe                            21.22%
  reference_date                 21.22%
  vpa                            21.22%
  lpa                            21.22%
  shares_outstanding             21.22%
  net_debt_equity                21.22%
  debt_equity                    21.22%
  roic                           21.22%
  net_debt_ebit                  21.22%
  net_debt_ebitda                21.22%
  net_income                     21.22%
  equity                         21.22%
  total_debt                     21.22%
  net_debt_to_assets             21.22%
  net_debt                       21.22%
  ebit                           21.22%
  ebitda                         21.22%
  cash                           21.22%
  total_assets                   21.22%
  ebit_over_assets               21.22%
  roa                            21.22%
  had_negative_earnings_5y       21.22%
  f_leverage_decreasing          21.22%
  f_liquidity_improving          21.22%
  f_score                        21.22%
  f_margin_improving             21.22%
  f_roa_improving                21.22%
  f_roa_positive                 21.22%
  days_since_fundamental         21.22%

Tickers with >20% NaN (may want to exclude):
  CCTY3    58.30%
  BMIN4    53.29%
  FIGE3    48.73%
  BAZA3    47.68%
  CEGR3    42.44%

✗ FAIL: Overall NaN rate ≥5% (imputation needed)c

* Improve handling of `NaN`/null fundamental data. Understand how the model currently interprets missing values.

---

## Visualization & debugging

* Determine which model or computation generates the "Portfolio Snapshot" section of the visualization.

---

## Benchmarks & evaluation

* Compare performance against:

  * Equal-weight portfolio
  * SELIC
  * BOVA11
  * Market-cap-weighted portfolio
* The goal is for the agent to decisively outperform these benchmarks over the long term.


- Use pl, pvp and other metrics directly from Bolsai API (maybe)
- Fix the imports/python execution bugs


Je les classerais ainsi:

    Momentum/rendement récent: rendements sur 1, 5, 20, 60 jours, car ils résument la tendance et les retournements.

    Volatilité: volatilité historique, ATR, variance réalisée, drawdown récent, car le RL doit apprendre à réduire le risque dans les phases instables.

    Corrélation et dispersion cross-asset: utiles pour l’allocation multi-actifs, car les poids dépendent aussi des dépendances entre actifs.

    Volume/liquidité: important pour la faisabilité réelle, surtout si tu trades des actifs moins liquides.

    Variables macro: SELIC, inflation, change, courbe des taux, stress de marché, car Lewin montre l’importance du régime au Brésil.

    Coûts de transaction et turnover: indispensables si tu veux éviter qu’un modèle trop nerveux surtrade.
Si tu veux une version vraiment efficace, je recommanderais de commencer avec un noyau de features simple mais solide:

    returns multi-horizons,

    volatilité glissante,

    drawdown,

    momentum,

    volume/liquidité,

    taux/régime macro,

    corrélation moyenne du portefeuille.

