# Stage 2: Dataset Build — How It Actually Works

Source: `src/build_dataset/`. This explains the implemented mechanics — for run commands see `CLAUDE.md`, citations reference `docs/RESEARCH_REFERENCES.md`.

## The organizing idea: no lookahead bias

Every temporal join in this stage — fundamentals onto prices, macro onto prices, dividends onto prices — uses `pd.merge_asof(..., direction="backward")`: for a given trading date, attach the most recent fundamental/macro/dividend value known **at or before** that date, never after. This is the single mechanical guarantee behind the "no lookahead" claim made throughout the project, and it's why `merge_asof` appears three separate times in `build_ml_dataset.py` rather than once. López de Prado (2018), *Advances in Financial Machine Learning*, Ch. 2 ("Backtesting under Realistic Conditions") — already in `RESEARCH_REFERENCES.md` and flagged there as mandatory reading — is the direct justification: a feature that "existed" on a given date must only reflect information actually available on that date, or backtest performance becomes an artifact of the leak rather than a real signal.

## Ticker inclusion policy

Before any feature is computed, tickers are filtered:

- `MIN_PRICE_ROWS = 10` — fewer rows than this means no usable history (the codebase notes one real example, a ticker with a single price row).
- Tickers present in prices but with **zero** fundamentals rows are dropped entirely. Sparse fundamentals (e.g. a ticker whose fundamentals only start in 2010) are fine — the agent handles NaNs in early rows — but *no* fundamentals at all is a hard exclusion, because this is meant to be a fundamentals-aware long-term agent, not a pure price-momentum one.
- After merging company info, rows are filtered to `status == "ATIVO"` only — a suspended or cancelled stock isn't tradeable, so it has no business being in a training set.

## Feature groups

### Price / technical features

Computed per-ticker on `adj_close` (non-positive prices masked to NaN first, to avoid divide-by-zero warnings in the log):

- `log_return = log(adj_close / adj_close.shift(1))`
- `volatility_20d`, `volatility_60d` — rolling standard deviation of log returns, 20- and 60-day windows
- `ma_20`, `ma_60` — simple moving averages of `adj_close`
- `hl_ratio = (high - low) / adj_close`
- `drawdown = (adj_close - cummax(adj_close)) / cummax(adj_close)` — running drawdown from the all-time high to date
- `rsi_14` — 14-period RSI (see below)
- `return_1m/3m/6m/12m` — cumulative log return over trailing 21/63/126/252 trading days

**RSI note, precisely:** the implementation is a simple-moving-average RSI —
```
gain = clip(delta, lower=0).rolling(14).mean()
loss = clip(-delta, upper=0)... .rolling(14).mean()
RSI = 100 - 100 / (1 + gain/loss)
```
— not Wilder's original exponentially-smoothed average (Wilder, J.W. (1978), *New Concepts in Technical Trading Systems*, the paper that introduced RSI). Both are legitimate implementations of the same indicator; the SMA variant is simpler and slightly more reactive to recent bars than Wilder's smoothing. Worth knowing if you ever compare this RSI against a charting platform that uses the classic Wilder version and see small discrepancies — that's why, not a bug.

### Fundamental features

Standard ratios (P/E, P/B, ROE, ROA, margins, leverage) plus YoY growth (`.pct_change(4)`, four quarters back) and QoQ trend (`.diff(1)`). A partial, 5-point Piotroski F-score is also computed — `f_roa_positive`, `f_roa_improving`, `f_margin_improving`, `f_leverage_decreasing`, `f_liquidity_improving`, summed into `f_score` (0–5). This deliberately omits the cash-flow-based components of Piotroski's original 9-point score (operating cash flow positive, cash flow > net income, no new share issuance) because the fundamentals feed doesn't carry cash-flow-statement data — a partial signal, not a full replication. These sector/quality-style ratio and growth features follow the factor-investing tradition of Fama & French (1993, 2012) — already in `RESEARCH_REFERENCES.md` — treating valuation, profitability, and leverage as distinct, separately-informative dimensions rather than collapsing them into one score.

### Macro features

- `excess_return = log_return - selic/252`
- `real_return = log_return - ipca/252`
- `selic_trend_20d = selic - selic.shift(20)`

The `/252` converts an annualized rate into a rough daily equivalent (a simple linear approximation, not compounded — acceptable at daily granularity). Adjusting for inflation this way follows Brière, Signori & Urevig (2012) on inflation-hedging portfolios (already in `RESEARCH_REFERENCES.md`), which is the rationale for treating real (IPCA-adjusted) return as a distinct feature from nominal return rather than assuming nominal returns are the whole story in a historically higher-inflation market like Brazil's.

### Cross-sectional features

Computed after the merge, since they need the full daily panel (all tickers, one date):

- **Sector z-scores**: `(value - group_mean) / group_std` for `pl`, `pvp`, `roe`, `debt_equity`, grouped by `(trade_date, sector)` — how expensive/profitable/levered a stock is *relative to its sector peers on that date*, not in absolute terms.
- **Rolling percentile ranks**: `volatility_20d`, `volatility_60d` (full-history rank), `price_percentile_5y` and `pl_percentile_5y` (5-year/1260-trading-day rolling rank), `drawdown_percentile` (1-year rolling rank).
- **Momentum decomposition**: `momentum_vs_market_{1m,3m,12m}` and `momentum_vs_sector_{1m,3m,12m}` — a ticker's trailing return minus the cross-sectional mean return (across all tickers, or within its sector) on that date. This is a momentum-factor construction in the spirit of Carhart (1997) (already in `RESEARCH_REFERENCES.md`), decomposed into market-relative and sector-relative components rather than one blended momentum number.

### Dividend features

`div_yield_12m` and `div_count_12m` are computed via a vectorized trailing-252-day window using `np.searchsorted` over sorted ex-dividend dates plus a cumulative sum of paid amounts — an O(log n) window lookup rather than a rolling-apply loop. Dividend yield as a standalone factor, especially relevant to Brazil's historically high-dividend-culture equities, follows Blakeslee et al. (2016) on dividend yield strategies in developed and emerging markets (already in `RESEARCH_REFERENCES.md`).

### CAGR backfill (`cagr_handler.py`)

A 3-tier priority merge, run on the fundamentals frame before it's joined to daily prices:

1. Use BolsAI's own reported `cagr_earnings_5y`/`cagr_revenue_5y` wherever present.
2. Where BolsAI's value is null, fall back to a locally computed CAGR (`calc_annual_cagr`, vectorized via numpy array slicing rather than a per-row loop): `((v_now/v_ago)**(1/years) - 1) * 100`, comparing each row to the value 20 quarters (5 years) prior.
3. Where the 5-year-ago base value itself was non-positive or missing, the metric is mathematically undefined and stays null — flagged via a companion `had_negative_earnings_5y` indicator rather than silently coerced to some placeholder number.

The combine is a literal `combine_first`: BolsAI's number always wins when present; the local calculation only fills gaps. The result lands in `cagr_earnings_5y_final` / `cagr_revenue_5y_final`.

## Data quality gates

`tests/build_dataset/test_final_dataset.py`'s `validate()` is the golden gate run against the final parquet, and it exists to catch exactly the failure modes the design above is meant to prevent:

- **No lookahead**: asserts `reference_date <= trade_date` for every row with a reference date — this is the direct, mechanical check on the `merge_asof(direction="backward")` guarantee above; if this ever fails, the no-lookahead claim is false.
- **No duplicate `(ticker, trade_date)` pairs** — catches a broken merge or a double-run that produced duplicate rows.
- **CAGR final columns present** — proves `fill_missing_cagr()` actually ran and populated the 3-tier merge, not just left BolsAI's raw (sparser) columns in place.
- **No NaN in `close`/`volume`** — these are the two columns nothing downstream can tolerate being missing.
- **Macro columns merged and not entirely null** — catches a broken macro join (e.g. a date-format mismatch that silently produced all-NaN SELIC/CDI/IPCA).
- **Every ticker has ≥252 rows** — one trading year minimum, consistent with the ticker-inclusion policy above; catches a ticker that slipped through with too little history to be usable.

## Commands

See `CLAUDE.md` → "Stage 2: Build ML Dataset" for the build and test commands.
