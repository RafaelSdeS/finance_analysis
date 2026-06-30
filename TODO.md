# TODO: Portfolio Allocation Agent

## Phase 1: Data Validation (3 tickers: PETR4, VALE3, WEGE3)

Goal: Validate API data accuracy against reliable sources before scaling.

### Blockers

- [ ] Uncomment and test `fill_cagr_columns()` in `src/2. build_dataset/build_ml_dataset.py` (line 143)
- [ ] Run full pipeline (Stages 1–2) on current 3 tickers
- [ ] Validate metrics: compare BolsAI prices + fundamentals against Yahoo Finance, B3 official, other data sources
  - Note: minor diffs expected; flag only if >1% variance on key metrics (P/E, P/B, ROE, CAGR)
- [ ] Verify macro data (SELIC, CDI, IPCA) matches official BCB sources
- [ ] Document any known discrepancies with links to reference sources

### Done

- [x] Macro data collection (SELIC, CDI, IPCA) via BCB SGS API
- [x] Stock prices and fundamentals collection via BolsAI API (PETR4, VALE3, WEGE3)
- [x] Dividend collection via BolsAI API (20-year history)
- [x] Generic API endpoint tester (`tests/api/bolsai_api_validator.py --path / --param`)
- [x] ML dataset builder: `merge_asof(prices, fundamentals)` + company info (no lookahead bias)
- [x] CAGR calculation and backfill logic (`cagr_handler.py`)
- [x] Interactive visualization: nominal price vs inflation-adjusted vs SELIC

---

## Phase 2: Scale to Full Market

Once Phase 1 validation passes:

- [ ] Expand ticker coverage to all B3 equities
- [ ] Re-validate on full dataset
- [ ] Create `.env.example` documenting BolsAI API key requirement

- [ ] Technical indicators: momentum, moving averages, volatility, drawdowns
- [ ] Macro context features: interest rate regime, market regime (bull/bear), VIX-like volatility proxy
---

## Phase 3: RL Agent

### Feature Engineering


### Environment & Simulation

- [ ] Portfolio state representation: current allocation weights, available cash, user risk profile
- [ ] Monthly-step simulation harness with fixed capital contributions (R$1000/mo, inflation-adjusted)
- [ ] Transaction cost model
- [ ] Dividend reinvestment handling

### Reward & Constraints

- [ ] Reward function: `return − λ1·volatility − λ2·max_drawdown − λ3·turnover`
- [ ] Constraints: position size limits, portfolio turnover cap (~10–20% per month), gradual rebalancing, always allow risk-free allocation

### Model & Evaluation

- [ ] RL model training (agent type: DQN, PPO, or actor-critic TBD)
- [ ] Backtesting harness: full historical simulation (no lookahead bias)
- [ ] Performance metrics: total return, annualized return, Sharpe ratio, max drawdown, Calmar ratio
- [ ] Benchmark comparison: IBOV (B3 Bovespa index)

---

## Future Scope (Out of Phase 1–3)

- [ ] **FIIs (Real Estate Investment Trusts):** Deferred pending Phase 3 scope decision. If RL agent expands to mixed-asset allocation, add separate collectors for FII prices + distributions. API endpoints exist; collector skeleton pattern is proven. **Why deferred**: FIIs have different fundamentals (NAV/P-VP vs earnings) and require separate dataset build logic.

---

See `specification.txt` for complete system design (input features, model output, objective function, constraints, expected behavior).
