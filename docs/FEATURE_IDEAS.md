# Feature Ideas (Backlog)

Candidate features for Stage 2 (`build_dataset/features.py`) to consider once Stage 3 modeling
resumes and feature selection is underway. Not scheduled — just parked here so they aren't
forgotten.

- [ ] **Accrual anomaly (Sloan 1996), balance-sheet version.** Low-accrual stocks tend to
      outperform high-accrual ones (earnings backed by cash vs. non-cash accruals). Computable
      from data already in `data/raw/br/fundamentals/*.parquet` — no new sourcing needed:
      `accruals = (ΔCA - Δcash) - (ΔCL) - D&A`, scaled by average `total_assets`, using
      `current_assets`, `current_liabilities`, `cash`, `total_assets`, and `ebitda - ebit` as a
      D&A proxy. Caveat: US evidence is strong, Brazilian-market evidence is weaker/mixed (thinner
      cross-section, accounting-regime differences) — treat as one more candidate signal, not a
      priority add.

- [ ] **Market breadth (mood proxy).** % of the universe trading above its own MA20/MA60 on a
      given date — a groupby/aggregate over per-ticker MAs already computed in
      `compute_price_features()`. No new sourcing. Cheap cross-sectional "risk-on/risk-off" signal.

- [ ] **Market realized volatility (mood proxy).** Rolling realized vol of BOVA11's return series
      (already collected as the benchmark for `beta_1y`/`momentum_vs_market_*`) — high vol ≈ market
      fear/stress regime. No new sourcing, just a rolling std on an existing series.

- [ ] **Piotroski F-score.** 9-point quality composite (profitability, leverage/liquidity change,
      operating efficiency), entirely computable from fields already in
      `data/raw/br/fundamentals/*.parquet` (`roe`, margins, `current_ratio`, `debt_equity`,
      `asset_turnover`, deltas quarter-over-quarter). No new sourcing.

- [ ] **News/social sentiment — parked, not cheap.** Would need a real text pipeline (scraping +
      NLP), a data source this repo doesn't have — a new data-collection project, not a
      feature-engineering task. A Brazil options-implied vol index (local VIX analogue) would be a
      more "real" mood signal than the two proxies above, but no known free public series for it.
