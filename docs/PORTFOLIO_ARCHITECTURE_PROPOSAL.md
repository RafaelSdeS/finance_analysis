# Research Strategy: Medium-to-Long-Term AI Portfolio Management (B3)

**Author role:** Lead AI Researcher / ML Systems Architect
**Status:** V1 architecture proposal — pre-implementation. No code.
**Dataset:** `data/processed/ml_dataset.parquet` — 1,308,104 daily ticker-date rows, 167 engineered features, point-in-time filing dates, split/continuity-repaired. See `CLAUDE.md` for the full data-quality provenance this proposal depends on.

---

## 0. TL;DR — the recommendation, up front

**Do not build deep RL for V1.** Build a **supervised cross-sectional return/quality forecaster feeding a cost-aware convex portfolio optimizer** ("predict-then-optimize"). This is the institutional standard, and for this specific problem it is not a compromise — it is the *correct* choice on the merits:

- The three behaviors you want to "emerge" (cash rotation, value-quality selection, low turnover) are each **directly and controllably producible** in a predict-then-optimize pipeline, most of them as closed-form consequences of the optimizer math rather than as fragile learned artifacts.
- Deep/offline RL on ~25 years of a **single realized macro history** is close to a worst-case setup for RL: catastrophic sample scarcity in the time dimension, no trustworthy simulator for counterfactuals, and a reward signal buried under return noise. It belongs in V3 as a *research bet with a high bar to clear*, not V1.

I'll defend this below, then specify the V1 architecture, the objective function, the state/time representation, the universe subset question, evaluation, and the steepest risks.

---

## 1. Independent critique of the premise

You asked me not to agree reflexively. Three pushes-back:

**1.1 "Emergent" is romanticized, and in a capital-allocation context it's a liability, not a virtue.**
Emergence requires *either* enormous data *or* strong priors. In markets you have neither: you have **one** realized path of Brazilian history (one 2008, one 2015–16 recession, one COVID, a rate-hiking cycle to ~26% in 2003, another to ~15% in 2025 that hadn't meaningfully reverted as of the dataset's mid-2026 end). A model that "discovers" cash rotation from that path has discovered *that path*, not a law. Worse: a fund's risk committee, drawdown post-mortems, and any future regulatory conversation all require you to answer *why did it move to cash in March*. An auditable `α → optimizer` pipeline answers that in one line. An emergent black box cannot. **Interpretability and control are features here, not training-wheels you discard later.**

**1.2 The behaviors you want are not equally hard to induce — and RL is hardest exactly where the optimizer is easiest.**
- Low turnover: **trivial** in a convex optimizer (one L1 penalty → a no-trade band). **Notoriously unstable** in RL (reward shaping that the agent games, or that dominates the return signal).
- Cash rotation under high SELIC: **closed-form** — it's the tangency portfolio with a risk-free asset in the opportunity set. In RL it's an emergent behavior you *hope* survives out-of-sample.
- Value-quality dualism: this is a **labeling/feature problem**, not an architecture problem. It lives in the α model regardless of whether the allocator is convex or RL.

**1.3 The sequence-model / Decision-Transformer instinct is mostly redundant with your own feature engineering.**
Your Stage-2 pipeline already bakes temporal context into the features: `*_zhist_5y` own-history z-scores, momentum, volatility percentiles, trends, `days_since_fundamental`, `n_quarters_available`. A Transformer/LSTM over raw history would re-derive signal you've *already* computed causally and leakage-audited. For V1, a per-row tabular model over the engineered features captures the temporal structure without adding an unaudited sequence model as a new leakage surface. (See §7 on when a sequence model earns its place.)

---

## 2. Recommended V1 architecture

```
                    ┌─────────────────────────────────────────┐
   per (ticker,     │  STAGE A — Alpha / Quality Forecaster    │
   date) feature ── │  GBM (LightGBM) cross-sectional          │──►  α_i,t  (expected
   vector (167)     │  learning-to-rank OR forward-excess-ret  │     forward excess return,
                    │  regression, purged walk-forward         │     per asset, per rebalance)
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
   Σ risk model ──► │  STAGE B — Cost-Aware Convex Optimizer   │──►  w_t  (target weights,
   (shrinkage /     │  max αᵀw − (λ/2)wᵀΣw − c₁‖Δw‖₁ − (c₂/2)‖Δw‖₂²│    incl. w_cash)
   factor model)    │  s.t. long-only, caps, cash∈opportunity  │
                    └──────────────────────────────────────────┘
                                       │
                              quarterly rebalance
                              (aligned to filing cadence)
```

### Stage A — the forecaster

- **Model:** Gradient-boosted trees (LightGBM/XGBoost). This is the honest default for ~167 **heterogeneous tabular** features on a cross-sectional panel. GBTs dominate tabular finance, handle non-linear **interactions natively** (critical for value-quality — see §4.2), tolerate the missingness patterns your NaN policy documents, and are cheap enough to walk-forward-refit dozens of times.
- **Target / head — pick one, test both:**
  - *(a) Learning-to-rank* (LambdaMART): rank assets cross-sectionally each period by forward excess return. Ranking is the natural objective — you care about *relative* attractiveness for allocation, and ranking is more robust to the fat-tailed, non-stationary *level* of Brazilian returns than pointwise regression.
  - *(b) Pointwise regression* on **forward excess return over CDI** at a long horizon (see §3.3). Simpler, gives a cardinal α the optimizer can trade off against the CDI carry directly.
  - Recommendation: **start with (b)** because the optimizer needs a cardinal α to compare against the risk-free carry; keep (a) as an ablation and a ranking-quality diagnostic.
- **What it must NOT see:** the lookahead-tainted columns in `manifest.LOOKAHEAD_TAINTED_COLS`, plus identifiers and non-stationary absolute levels. The full concrete exclude/keep partition of the real 167 columns is in **§4.4** — that partition is load-bearing, read it before building the feature matrix.

### Stage A′ — probabilistic (uncertainty-aware) forecasting — RECOMMENDED refinement

A point α discards exactly the information that separates *cheap quality* from *cheap junk*: two names with identical expected return can have wildly different return **distributions**, and a value trap is precisely a fat left tail. Feeding the optimizer a distribution instead of a point estimate is a genuine upgrade — but resist the temptation to bolt on four separate prediction heads ("mean, variance, downside, confidence"). Those collapse to **two real axes** — conditional *central tendency* and conditional *dispersion/downside* — because variance and downside are near-collinear for equities, and "confidence" is not a separate quantity (it's either inverse predicted variance = **aleatoric**, or model epistemic uncertainty = a different, harder thing; don't conflate them).

- **Get both axes from ONE model family, not four:** LightGBM **quantile regression** (`objective='quantile'`) at ~3 quantiles (10 / 50 / 90) — in practice **3 separate fits** (LightGBM's quantile objective has no native joint multi-quantile output; you train one model per target `alpha`, same features/hyperparameters). Still one family, not four unrelated heads:
  - **q50 → α** (expected excess return),
  - **q90 − q10 → dispersion** ("uncertainty"),
  - **q10 → downside** — the value-trap discriminator, directly.
  - **Known gotcha — quantile crossing:** independently-fit quantiles aren't guaranteed monotonic (a row can land q10 > q50 in the tails). Enforce ordering post-hoc (sort the 3 predictions per row) before computing `q90 − q10`, or the "dispersion" feature can go negative.
- **Two ways it feeds the optimizer (§5):** (1) **α-shrinkage** — pull each `α_i` toward zero in proportion to its predicted spread *before* it enters the program; this is the mechanism that yields the "more stable allocations" the richer forecast is supposed to buy. (2) **Swap the risk term** from symmetric variance `wᵀΣw` toward a **downside / CVaR** penalty on `q10`; still convex, and now the objective penalizes left-tail exposure specifically, not harmless symmetric wiggle — the value-trap-aware allocator.
- **You already have a cheap risk axis:** `volatility_20d/60d`, `volatility_ratio_20_60`, `beta_1y`, `amihud_illiquidity`, `drawdown`, `true_range_ratio`, and an EWMA/realized-vol covariance are "predicted variance" in its cheapest honest form. The quantile head must *beat* that baseline to earn its place.
- **Epistemic "confidence" is deferred:** if wanted in V2, **conformal prediction** on the quantile model gives calibrated intervals cheaply — no fifth head.
- **The hard guard (see §9.8):** the second moment is *harder* to forecast than the first (which is already ~all noise), so **do not let the optimizer trust any predicted interval until it is calibrated out-of-sample** — do realized returns land in the q10–q90 band ~80% of the time? If not, a shrinkage covariance beats the learned heads and you use that instead. Un-calibrated risk forecasts are worse than none.

`ponytail: one quantile forecaster over four heads; start with point-α + realized-vol Σ, add quantile risk term only after it beats that baseline on calibration + out-of-sample allocation.`

### Stage B — the allocator

A per-rebalance convex program (see §5 for the full objective). Cash (CDI accrual) is a **first-class asset in the opportunity set** with known return and ≈0 variance. Position caps enforce diversification; long-only matches the mandate. Solved with any convex solver (this is a tiny QP/SOCP — 30–500 assets).

### The risk model Σ — do not skip this

The optimizer is only as good as its covariance estimate, and a sample covariance of 30–500 assets over overlapping windows is numerically garbage (ill-conditioned, unstable). **Use Ledoit-Wolf shrinkage or a small factor model** (market + a few style factors you already have: value, quality, momentum, size). This is a real engineering task, not a footnote — a bad Σ silently wrecks the allocation and gets misattributed to the α model.

---

## 3. State and time representation

### 3.1 State = the cross-sectional feature panel, as-is
The "state" the forecaster consumes is the point-in-time feature vector per (ticker, date). No hand-built market-regime state variable is needed — the macro features (SELIC/CDI/IPCA level, real return, excess return, rate environment) **are** the regime encoding, and they're already causal and leakage-audited. The optimizer's state is `(α_t, Σ_t, w_{t-1})`.

### 3.2 Time = walk-forward, respecting `split_config.json`
- **Evaluation:** expanding/rolling walk-forward. Never a single random split. Refit the forecaster at each walk-forward step on data available up to that point.
- **Leakage in the labels:** forward-return labels *overlap* (a 12-month forward label at day *t* overlaps day *t+1*'s). Naive CV then leaks. Use **purged + embargoed cross-validation** (López de Prado): purge training samples whose label window overlaps the validation window, and embargo a buffer after it. This is the same leakage discipline your Stage-2 pipeline already enforces on merges — extend it to the modeling layer.
- **Scaler boundary:** fit any scaling train-only via the existing `iter_fit_windows()` / `FitWindow` seam. Do not re-invent the split boundary.

### 3.3 Rebalance cadence — the single most important low-turnover lever
**Rebalance quarterly, aligned to the fundamental filing cadence.** Fundamentals only change quarterly; a value thesis built on them cannot legitimately change daily. Daily rebalancing on quarterly signals is pure noise-trading that the 0.03% fee then taxes. Quarterly rebalancing:
- matches the information arrival rate,
- caps turnover structurally *before* the optimizer's cost penalty even engages,
- and makes multi-year holding the *default* rather than something you fight for.

Forecast **horizon** should be long to match — **6 to 12 months forward excess return**. (Test 6m vs 12m; longer horizon = lower turnover and closer to the "1–5 year" mandate, at the cost of fewer independent training labels — see §8 risk.)

---

## 4. How each Prime Directive goal is met — mechanistically

### 4.1 Cross-asset allocation & macro-regime awareness → *closed-form from the optimizer*
Put CDI in the opportunity set with return = the risk-free carry and ≈0 variance. **The `cdi` column *is* that carry — the optimizer's cash-leg return is directly in the dataset**, no modeling required for `α_cash`. The mean-variance optimum with a risk-free asset is the tangency portfolio blended with cash. **When risk-adjusted equity α falls below the CDI carry (high-SELIC regime, or α compressed by spiking risk), the optimizer mechanically shifts weight to cash.** When risk premia expand (α ≫ carry), it deploys. You do not train this behavior — it is the arithmetic of the objective.

The α model *also* sharpens the rotation because your feature set already contains the equity-risk-premium-vs-cash signal explicitly: **`earnings_yield_vs_selic`** (earnings yield minus the risk-free rate) is the single most important feature for this decision — it directly encodes "do equities out-yield cash right now," per-name. **This is the cleanest example of why the convex allocator beats RL here: you get the exact behavior you asked to "emerge," but provably and auditably — and the ERP signal is pre-engineered, not something the model has to rediscover.**

**Prefer deltas/spreads over absolute macro levels — a real overfitting risk in tree models.** SELIC ranged from ~1.9% (Aug 2020) to ~26.3% (Mar 2003) across the dataset's full 2000–2026 history (verified against `data/raw/br/macro/selic.parquet`) — a far wider swing than the 2022–25 hiking cycle alone (~2%→~15%) would suggest, and §9.3 already flags the non-stationarity this implies. A GBT split like `selic > 11.2%` risks encoding "the specific 2022–23 hiking episode," not a transferable regime rule — the split threshold is a historical accident, not a law. So the **spread/derived** features are the primary regime inputs the model should lean on: `earnings_yield_vs_selic` (equity-vs-cash spread — the actual decision variable), `selic_trend_20d` (direction, not level), `excess_return`, `real_return`. The **absolute levels** (`selic`, `cdi`, `ipca`, `ipca_daily_equiv`) stay in the feature set — a cash-rotation decision plausibly *does* depend on the level itself (15% cash is a qualitatively different opportunity cost than 2% cash, independent of any spread) — but treat them as secondary/conditioning inputs, not the primary lever, and **watch them**, not exclude them:
- Check SHAP/feature-importance isn't concentrating on a raw-level threshold tied to one historical episode (the overfitting signature this guidance exists to catch).
- Consider `monotone_constraints` in LightGBM on the spread features (e.g., predicted forward return non-decreasing in `earnings_yield_vs_selic`) — a stronger, structural regularizer against this exact failure mode than feature removal would be, and it doesn't cost you the level features' legitimate information.

### 4.2 Value-quality dualism (anti-value-trap) → *lives in α, learned but supervised*
This is where genuine learning happens, and it's the *right* place for it because it's **checkable**. The forecaster must learn the **interaction**: cheap × high-quality → positive forward return; cheap × distressed junk → not. GBTs model this interaction natively (a split on valuation *then* on a quality axis is exactly a decision-tree path). Your feature set is purpose-built for it, and you have a **ready-made quality composite**: the Piotroski score.

- **The value axis:** `pl`, `pvp`, `ev_ebitda`, `ev_ebit`, `p_sr`, `book_to_market`, `earnings_yield`, `peg_ratio`, `pvp_to_roe_ratio` — and their own-history context `pl_zhist_5y`, `pvp_zhist_5y`, `earnings_yield_zhist_5y`, `book_to_market_zhist_5y`, plus `pl_percentile_5y`. The `*_zhist_5y` "how unusual for *this* company" signals are exactly what distinguishes a temporary panic compression from a structural, permanent re-rating.
- **The quality axis (the value-trap discriminator):** **`f_score`** and its five components (`f_roa_positive`, `f_roa_improving`, `f_margin_improving`, `f_leverage_decreasing`, `f_liquidity_improving`) are a purpose-built accounting-quality/deterioration score — the single most direct anti-value-trap feature you have. Reinforced by `roe`, `roa`, `roic`, `net_debt_ebitda`, `debt_equity`, `current_ratio`, the trend features (`roe_trend_4q`, `margin_trend_4q`, `debt_trend_4q`, `roa_trend_4q`), and the distress flag `had_negative_earnings_5y`.
- **The interaction to expect the model to learn:** *low valuation percentile × high `f_score` × low `net_debt_ebitda`* → buy; *low valuation × low `f_score` × rising leverage* → the trap, avoid. Because both axes are pre-engineered, the model isn't discovering value investing from scratch — it's learning the *conditional weighting* of signals you've already computed, which is a far lower-variance learning problem than end-to-end RL.

**Diagnostic:** SHAP interaction values on (valuation × `f_score`) — if the model *isn't* pricing the value-trap interaction, you'll see it directly, and you can't with an emergent RL policy.

### 4.3 Friction-aware compounding & low turnover → *L1-dominant turnover penalty + quarterly cadence*
Two compounding mechanisms (see §5 for the math):
1. **Quarterly cadence** caps the *opportunities* to trade.
2. The **L1 turnover term `c₁‖Δw‖₁`** creates a **no-trade band**: a position only changes when the risk-adjusted α improvement exceeds the round-trip cost. Small α drift → no trade → winners run, positions persist across quarters and years. This is the Gârleanu-Pedersen / Boyd "multi-period trading via convex optimization" result: the optimal policy under linear costs is to trade *toward* the target only partially, damping churn. Multi-year holding periods are the emergent output — but emergent from convex math, not from a reward the agent might game. (An optional small `c₂‖Δw‖₂²` term smooths corner-solution jumpiness if observed — see §5 — but does not replace the no-trade mechanism.)

### 4.4 Feature hygiene — the concrete partition of the 167 columns

Of 167 columns, 46 must never be model inputs directly (identifiers, tainted, non-stationary levels — A+B+C+D below: 8+9+16+13). That leaves 121 real features (E below, including `sector` as a special-cased low-risk categorical — see its note). Verified column-by-column against the actual `data/processed/ml_dataset.parquet` schema and `manifest.LOOKAHEAD_TAINTED_COLS`; every one of the 167 real columns is accounted for exactly once below. Do **not** dump the raw frame into the GBM.

**A. Never features — identifiers / metadata (keep for joins, indexing, label construction only):**
`ticker`, `trade_date`, `reference_date`, `fundamentals_available_date`, `corporate_name`, `trade_name`, `cvm_code`, `cnpj`.

**B. Never features — lookahead-tainted (`manifest.LOOKAHEAD_TAINTED_COLS`, verified against source — 9 columns exactly):**
`status`, `pl_zscore_sector`, `pvp_zscore_sector`, `roe_zscore_sector`, `debt_equity_zscore_sector`, `div_yield_sector_percentile`, `momentum_vs_sector_1m`, `momentum_vs_sector_3m`, `momentum_vs_sector_12m`. These launder current-day survivorship/sector knowledge into numeric form. **Note the asymmetry:** `momentum_vs_market_{1m,3m,12m}` is **clean** (BOVA11-benchmarked since the 2026-07-24 fix) — **keep it**; only the `*_sector` variants are tainted. `beta_1y` is also clean (BOVA11) — keep.

**Correction: `sector` itself is NOT in `LOOKAHEAD_TAINTED_COLS`** (checked directly in `src/build_dataset/manifest.py` — only `status` plus the 8 derived columns above are listed). This matches `CLAUDE.md`'s own characterization: "`sector` is the same kind of static join but carries far less outcome information, so it's lower-risk as a feature" — it's still a current-day snapshot joined onto history, but not blanket-excluded. It's kept, with a caution note, in group E below rather than dropped here.

**C. Drop — non-stationary absolute price/volume levels (cross-sectionally meaningless; the normalized versions already exist):**
`open`, `high`, `low`, `close`, `adj_open`, `adj_high`, `adj_low`, `adj_close`, `volume`, `volume_adjusted`, `traded_amount`, `num_trades`, `ma_20`, `ma_60`, `lpa`, `vpa`. Use `price_vs_ma20`/`price_vs_ma60` instead of the raw MAs, `volume_ratio_20d`/`turnover_ratio`/`amihud_illiquidity` instead of raw volume. **`adj_close` is still needed to build the forward-return label and underlies every kept normalized feature — it just isn't itself an input.** Mask label rows where `adj_close_precision_degraded == 1`.

**D. Drop (or keep at most one log-size control) — raw currency fundamentals (collinear size proxies; their *ratios* are the real features):**
`net_income`, `equity`, `net_revenue`, `total_debt`, `ebitda`, `ebit`, `net_debt`, `cash`, `total_assets`, `current_assets`, `current_liabilities`, `shares_outstanding`, `market_cap`. If you want a size factor, keep exactly one: `log(market_cap)`. The rest are multicollinear scale, non-stationary, and already expressed as ratios elsewhere.

**E. KEEP — the real feature set (121), by group:**
- *Valuation:* `pl`, `pvp`, `ev_ebitda`, `ev_ebit`, `p_ebitda`, `p_ebit`, `p_sr`, `p_assets`, `book_to_market`, `earnings_yield`, `peg_ratio`, `pvp_to_roe_ratio`, `earnings_yield_vs_selic`.
- *Profitability / quality:* `gross_margin`, `net_margin`, `ebitda_margin`, `ebit_margin`, `roe`, `roa`, `roic`, `ebit_over_assets`, `asset_turnover`, `revenue_per_earning`.
- *Leverage / solvency:* `current_ratio`, `debt_equity`, `net_debt_equity`, `net_debt_ebitda`, `net_debt_ebit`, `cash_ratio`, `net_debt_to_assets`, `working_capital_ratio`.
- *Growth & change:* `cagr_revenue_5y`, `cagr_earnings_5y`, `cagr_earnings_5y_final`, `cagr_revenue_5y_final`, `revenue_growth_yoy`, `earnings_growth_yoy`, `ebitda_growth_yoy`, `total_assets_growth_yoy`, `total_debt_growth_yoy`, `revenue_vs_earnings_growth_delta`, `gross_margin_qoq`, `net_margin_qoq`, `roe_qoq`, `debt_equity_qoq`, `current_ratio_qoq`.
- *Piotroski / quality composite:* `f_score`, `f_roa_positive`, `f_roa_improving`, `f_margin_improving`, `f_leverage_decreasing`, `f_liquidity_improving`, `had_negative_earnings_5y`.
- *Fundamental trends:* `roe_trend_4q`, `margin_trend_4q`, `debt_trend_4q`, `roa_trend_4q`.
- *Price technicals (normalized):* `volatility_20d`, `volatility_60d`, `volatility_ratio_20_60`, `price_vs_ma20`, `price_vs_ma60`, `hl_ratio`, `true_range_ratio`, `drawdown`, `rsi_14`, `volume_ratio_20d`, `amihud_illiquidity`, `turnover_ratio`.
- *Trailing momentum / returns:* `log_return`, `overnight_gap`, `intraday_return`, `return_1m`, `return_3m`, `return_6m`, `return_12m`, `excess_return`, `real_return`, `momentum_vs_market_1m`, `momentum_vs_market_3m`, `momentum_vs_market_12m`, `beta_1y`.
- *Percentiles (own rolling, causal):* `volatility_20d_percentile`, `volatility_60d_percentile`, `price_percentile_5y`, `price_percentile_1y`, `pl_percentile_5y`, `drawdown_percentile`.
- *Own-history robust z (`*_zhist_5y`, 13):* `amihud_illiquidity_zhist_5y`, `turnover_ratio_zhist_5y`, `pl_zhist_5y`, `pvp_zhist_5y`, `roe_zhist_5y`, `net_margin_zhist_5y`, `ebitda_margin_zhist_5y`, `debt_equity_zhist_5y`, `net_debt_ebitda_zhist_5y`, `earnings_yield_zhist_5y`, `book_to_market_zhist_5y`, `current_ratio_zhist_5y`, `asset_turnover_zhist_5y`.
- *Dividends:* `div_yield_12m`, `div_count_12m`, `div_value_12m`, `div_value_recent`, `payout_ratio`, `dividend_coverage_ratio`, `has_dividends`.
- *Macro / regime:* `selic_trend_20d` (**primary regime input — a trend/delta, prefer this**; `earnings_yield_vs_selic` [already listed under Valuation] and `excess_return`/`real_return` [already listed under Momentum] are the other primary spread/delta regime signals discussed in §4.1 — not re-listed here, to keep the 121-feature count from double-counting); `selic`, `cdi`, `ipca`, `ipca_daily_equiv` (secondary/conditioning — keep, but watch for level-threshold overfitting per §4.1).
- *Static categorical (special case — keep, but read the note):* `sector`. Verified absent from `manifest.LOOKAHEAD_TAINTED_COLS` (see §4.4.B correction) — usable as a plain low-cardinality categorical for cross-sectional grouping. It's still a current-day join (a company's historical rows all show its 2026 sector), so don't derive anything from it beyond grouping/one-hot that would implicitly encode which sectors survived to today.
- *Info-age / NaN-explainer flags (let the GBM use these — they explain the NaNs):* `filing_lag_days`, `days_since_fundamental`, `n_quarters_available`, `has_fundamentals`, `cagr_earnings_defined`, `cagr_revenue_defined`, `adj_close_precision_degraded`.

**F. Special-role, not equity features:** `cdi` (and `selic`) double as the optimizer's cash-leg carry `α_cash` (§4.1); keep them in the α feature set too — they're legitimately predictive of equity returns *and* they parameterize the cash asset.

**Multicollinearity note (ponytail, but real):** the valuation block (`pl`/`pvp`/`ev_*`/`p_*`/`earnings_yield`/`book_to_market`) is highly collinear, as is the margin block. GBTs tolerate this for *prediction* but it wrecks SHAP *attribution* (credit splits arbitrarily among correlated features). If you rely on the §4.2 value-trap diagnostic, either prune to one representative per cluster or read SHAP at the cluster level, not per-column.

---

## 5. The objective function

At each rebalance *t*, solve:

```
maximize_w     αᵀw  −  (λ/2)·wᵀΣw  −  c₁·‖Δw‖₁  −  (c₂/2)·‖Δw‖₂²
subject to     Δw = w − w_{t-1}
               Σ_i w_i + w_cash = 1
               w ≥ 0                    (long-only mandate)
               w_i ≤ w_max              (diversification / concentration cap)
               w_cash ≥ 0
where          α_cash = CDI carry over the holding period, Σ row/col for cash ≈ 0
               c₁ = one-way execution cost ≈ 0.03% B3 fee + slippage/impact buffer (see §6, §9.5) —
                    NOT the round-trip fee; see note below the program
               c₂ = small, tuned empirically (starts at ~0 — see below)
               λ  = risk-aversion (tunes equity-vs-cash aggressiveness)
```

**Turnover penalty: L1-dominant elastic-net, not L2 — this matches the true cost structure.** The B3 fee is a **flat proportional fee**: cost scales linearly with trade size. That's the definition of an L1 cost — `c₁‖Δw‖₁` isn't an arbitrary regularizer choice, it's the direct mathematical image of "0.03% per unit traded." An L2-only penalty is the standard proxy for **market impact** (Almgren-Chriss: impact cost is quadratic in order size), which is a different, secondary cost component — not a substitute for the fee term. Defaulting to L2-only would:
- lose the **exact no-trade band** that makes §4.3's "winners run untouched" guarantee *provable* rather than merely likely — L2 nudges every position a little every quarter, proportional to the α gap, never zeroing a trade out exactly;
- mismatch the real cost model in §6 (fee is linear; only impact is quadratic).

So the default is **`c₁` dominant, `c₂` small (starting near 0, e.g. `c₂ ≈ c₁/10` or off entirely)** — a true no-trade band from the L1 term, with `c₂` available purely as a *smoothing* knob if backtests show the L1-only solution producing corner-solution jumpiness (abrupt full-position swaps rather than partial trims) in practice. Tune `c₂` empirically against the §8 turnover/holding-period diagnostics — don't assume it's needed; add it only if the L1-only backtest shows a problem it would fix. **V2 ablation:** sweep `c₂ ∈ {0, small, larger}` and report the turnover/Sharpe tradeoff curve; pure-L1 (`c₂ = 0`) remains the baseline to beat.

**Why this exact form induces what you want:**
- The **`c₁‖Δw‖₁`** term's subgradient is the no-trade region: asset *i* only moves if `|∂(αᵀw − risk)/∂w_i| > c₁`. This is *the* mathematically principled low-turnover mechanism — not a heuristic, not reward-shaping — and it is solved exactly by the convex solver in one shot (no gradient-descent "stickiness" concern; that's a training-time artifact this pipeline doesn't have).
- **`c₁` must be the one-way cost, not the round-trip fee — a bug to avoid, not a design choice.** `‖Δw‖₁` already charges once when a position opens and again, independently, when it later closes — a full buy-then-sell lifecycle naturally costs ≈2×c₁ from those two separately-charged legs. Setting `c₁` to the round-trip figure (2 × 0.03% = 0.06%) instead of the one-way fee (~0.03% + buffer) double-counts, artificially widening the no-trade band and understating achievable turnover/alpha. §9.5 already anchors the cost *floor* at 0.03% one-way (consistent with §6) — keep §5's `c₁` and §8's cost-sensitivity sweep anchored the same way, not to a pre-doubled number.
- **Cash in the opportunity set** delivers §4.1 for free.
- **`λ`** is your single interpretable dial for "aggressive deployment vs. wealth preservation," and it's inspectable — not buried in a policy network.

**Uncertainty-aware variant (Stage A′):** if you adopt the quantile forecaster, this objective upgrades two ways without leaving convexity: (1) replace `α` with the **shrunk** α (each `α_i` pulled toward 0 in proportion to its predicted q90−q10 spread — this is what buys the "more stable allocations"); (2) replace the symmetric `(λ/2)wᵀΣw` with a **downside/CVaR** penalty built on the q10 forecast, so the optimizer penalizes left-tail exposure (the value trap) rather than harmless symmetric variance. Gate this on the §9.8 calibration check — un-calibrated quantiles make it *worse* than plain mean-variance with a shrinkage Σ.

**Explicitly avoid** trying to encode all of this as an RL scalar reward like `Δwealth − turnover_penalty − risk_penalty`. That collapses three cleanly-separable, individually-tunable convex terms into one noisy signal the agent must disentangle from 25 years of one path. You'd be throwing away structure you already have in closed form.

### 5.1 Is the objective secretly myopic? — the honest answer to "does this reward long-term holding?"

This deserves a direct answer, because it's the failure mode that would quietly turn a "value investor" into a churner. **Long-horizon holding is enforced at three levels, in order of importance:**

1. **The label horizon is the real guarantee (deepest lever).** The forecaster is trained on **6–12-month forward excess return**, never on daily or next-day returns. A model is only as short-term as its supervision signal: one trained on daily targets *will* chase daily noise; one that has literally never been shown a sub-annual target **cannot** optimize for day-trading — it has no representation of it. This is baked into the objective at the supervision layer, not bolted on as a penalty. It is the single most important reason this design is structurally long-term.
2. **Quarterly rebalance cadence** makes intra-quarter trading *mechanically impossible* — the allocator simply isn't invoked more often than fundamentals change.
3. **The `c₁‖Δw‖₁` turnover term** creates the no-trade band that holds a position through noise once entered (§4.3), so winners compound across quarters and years.

**The honest nuance — V1's §5 optimizer is single-period (myopic-greedy), and that is a deliberate, bounded simplification, not an oversight.** Each quarter it maximizes a *one-step* objective; it does not *plan* over a multi-quarter horizon. Levers (1)–(3) mean it *behaves* long-term (slow-moving α + no-trade band + quarterly cadence), and we **select** designs on multi-year compounded outcomes (§8 measures holding-period distribution and compounded return, not per-period Sharpe) — so short-termism cannot hide in the evaluation. But a truly long-term-*aware* objective, one that explicitly values *not trading today because of the cost of trading again tomorrow*, is the **multi-period** convex program (Boyd, "Multi-Period Trading via Convex Optimization"; Gârleanu-Pedersen): optimize over a rolling horizon of *H* future rebalances with forecast α-decay and per-period costs. That is the principled upgrade, and it's the right V2 move **if and only if** the single-period + turnover-penalty version proves too twitchy in the §8 turnover diagnostics. Starting single-period is the lazy-correct call: it's convex, fast, and its holding behavior is empirically checkable before you pay for the multi-period machinery. `ponytail: single-period optimizer; upgrade to multi-period Boyd/G-P if turnover diagnostics show churn the penalty can't damp.`

**Bottom line:** the project does *not* optimize for daily trading — it is structurally incapable of it at the label layer, throttled at the cadence layer, and damped at the cost layer, with the only open question being *how explicitly* future holding is valued (single-period-with-penalty now, multi-period-planned later if needed).

---

## 6. Transaction cost & liquidity realism

The 0.03% B3 fee is the **floor**, not the cost. For a real fund the cost is fee + half-spread + market impact, and impact is **highly heterogeneous** across your universe — which is precisely why `amihud_illiquidity` and `turnover_ratio` exist in your features. Two consequences:

1. **The cost `c` in the objective must be per-asset and liquidity-scaled**, not a flat one-way 0.03%. Use the Amihud/turnover features to inflate `c` for thin names. A flat cost will make illiquid microcaps look tradeable when they are not.
2. This is a primary reason to **start on a liquid subset** (§7) where the flat-ish cost assumption is closest to true.

---

## 7. Universe: start with a liquid subset — yes, decisively

**Start with the liquid large-cap subset (~30–50 names), then expand.** The repo already has the scaffolding for exactly this: `TOP50_UNIVERSE_VALIDATION.md`, `test_blue_chip_tickers.py`, `test_top_traded_quality.py`. Reasons, in priority order:

1. **The cost model is only honest for liquid names.** Half the repo's data-quality caveats are microcap pathologies (2-decimal `adj_close` precision floor, pinned prices, oscillating raw closes, unfixable quarantines). Backtesting a *value* strategy on illiquid deep-history microcaps will manufacture alpha you can never capture after real slippage.
2. **It shrinks the data-quality noise floor**, so any signal you find is more likely real than an artifact of the exact vendor quirks `CLAUDE.md` catalogs.
3. **It de-risks the pipeline build.** Prove `α → Σ → optimizer → walk-forward → deflated-Sharpe` end-to-end on a clean, well-behaved universe first. Expanding the universe later is a data-filter change, not an architecture change.
4. The cross-sectional breadth loss is acceptable for V1: 30–50 names × ~25 years of quarterly rebalances still gives usable cross-sectional training rows, and the whole point of V1 is to validate the *machinery*, not to harvest the last basis point of breadth alpha.

**MANDATE — the liquid subset MUST be recomputed dynamically at every rebalance date, not fixed once:**
- At each quarterly rebalance date *t*, compute liquidity **using only trailing-12-month data available as of *t*** — trailing-12m median `traded_amount` (or `amihud_illiquidity` percentile), same causal discipline the rest of the pipeline already enforces on fundamentals via `fundamentals_available_date`.
- Keep only names in (e.g.) the top liquidity percentile band **as of that date**. The resulting ~30–50-name universe is **expected to differ from quarter to quarter** — a ticker liquid in 2015 but illiquid by 2020 drops out; a ticker that IPO'd and became liquid in 2019 enters. This churn in the *investable set* is a feature, not a bug — it's what "point-in-time" means.
- **Prohibited:** filtering the universe once using today's (2026) liquidity/traded-volume/survival status and applying that static list across the full backtest history. This silently reintroduces survivorship bias — the backtest would only ever "discover" that Company X was a great long-term hold using the fact that Company X is *still liquid and trading in 2026*, which the strategy could not have known in, say, 2010. This is the exact same failure mode `CLAUDE.md` already documents for `status`/`sector` (§4.4.B) — a current-day snapshot laundered into a feature that looks clean but encodes the future. Applying it to universe selection would undo the survivorship discipline the rest of the pipeline (BOVA11 benchmark fix, tainted-column list, etc.) was built to guarantee.
- **Implementation note:** write this once as a `(df, date, percentile_threshold) → tickers` helper and reuse it everywhere (backtest, Phase 2a equal-weight baseline, walk-forward loop) — a second, drifted implementation of "liquid universe" is how this mandate quietly gets violated in one code path and not another.

---

## 8. Evaluation methodology

The strategy is worthless if you can't tell skill from luck, and **this is genuinely hard here** (§9.4). Minimum bar:

- **Purged + embargoed walk-forward** (per §3.2). Report out-of-sample only.
- **Deflated Sharpe Ratio** (López de Prado) to correct for the number of configurations you tried — with quarterly rebalancing over ~25 years you have *few independent bets*, so a raw Sharpe is dangerously optimistic.
- **Turnover and holding-period distributions** as first-class metrics, not afterthoughts — they *are* the mandate. Report average holding period in years, annual turnover %, and the fraction of rebalances that were no-trades.
- **Regime-sliced performance:** report separately across the high-SELIC vs low-SELIC and crisis vs calm sub-periods. A strategy that only works in one regime of your single path is overfit to that path.
- **Cost sensitivity curve:** performance vs. assumed one-way `c` (0.03% floor → 0.15% → 0.3%, i.e. the bare fee up to ~10× it standing in for spread+impact); report the round-trip-equivalent (2×c) alongside for intuition. If alpha evaporates at realistic slippage, you don't have a strategy.
- **Baselines it must beat:** (i) equal-weight the liquid universe, (ii) buy-and-hold IBOV/BOVA11, (iii) 100% CDI. If it can't beat CDI net of cost, the whole "willingly migrate to cash" thesis says it *should* just hold CDI — and that's a valid, honest answer, not a failure.

---

## 9. Steepest research risks (ranked)

**9.1 Single-history overfitting / regime scarcity — THE dominant risk.**
You have ~1 macro path. Every temporal relationship you learn is conditioned on realized Brazilian history. This is why V1 leans on cross-sectional breadth (510 tickers currently in `ml_dataset.parquet` → liquid subset gives many cross-sectional samples per date even when independent *time* samples are few) and on the *low-parameter*, regularizable GBM + closed-form optimizer rather than a high-capacity policy net that will memorize the path. Mitigation is methodological, not architectural: heavy walk-forward, deflated metrics, regime-slicing, and refusing to trust one backtest.

**9.2 The as-restated fundamentals lookahead (`CLAUDE.md` Issue 8) — directly threatens the thesis.**
Fundamental *values* come from BolsAI's current snapshot (likely latest restatement) even though the *date* is point-in-time-correct. Your strategy's core edge is **value-quality selection from fundamentals** — the exact signals most contaminated by as-restated-vs-as-first-reported leakage. This will inflate backtest alpha precisely where you most want to trust it, and it is **unquantified**.

**The real fix — quantify with true data:** sample how many rows carry restated figures and the magnitude gap using CVM's versioned ZIPs (`src/data_collection/cvm/statements.py` already has the download machinery for every filing version), per the flagged sourcing project. This is the only way to actually *know* the size of the problem, and it may be worth a scoped data-sourcing effort ahead of modeling. Not yet attempted.

**Interim falsification test — a sensitivity/jitter check you can run today, before that sourcing project:** perturb the fundamental ratios that drive α (the §4.2 value and quality axes — `pl`, `pvp`, `roe`, `debt_equity`, `net_debt_ebitda`, etc.) with noise calibrated to a plausible restatement magnitude, rerun the full walk-forward backtest, and check whether the measured alpha (§8 metrics) survives.
- **What this test can and can't tell you:** if alpha collapses under a modest jitter, that's strong evidence the strategy is leaning on precision the market didn't actually have at the time — a red flag *before* you've paid for the CVM sourcing effort. If alpha survives, that's reassuring but **not proof** — jitter is a symmetric-noise proxy for restatement, and real restatements may be systematically biased (e.g., consistently *improving* reported figures after audit) rather than mean-zero noise, which a jitter test cannot catch. Treat a pass as "not yet falsified," not "confirmed clean."
- **Calibrating the jitter magnitude — be explicit, don't guess silently:** absent a measured restatement-magnitude distribution (which is what the CVM sourcing project would actually deliver), state the assumed magnitude directly in the writeup (e.g., "±10% on leverage/margin ratios, ±15% on earnings-derived ratios") so the test's conclusion is falsifiable against that stated assumption, not hidden inside an arbitrary default.
- This test is **evidence, not a substitute** for sourcing true as-first-reported v1 figures — it tells you whether the issue is urgent, not whether it's resolved.

**9.3 Non-stationarity / structural breaks.**
Brazil's rate regime (SELIC ~2%→~26% through the early 2000s, ~2%→~15% again in 2020–25), inflation history, and market microstructure all shift. A value multiple's meaning at SELIC 14% differs from SELIC 2%. Mitigation: the macro features let the model condition on regime, but don't over-trust extrapolation into a regime not in-sample. Rolling refit over expanding window; watch for feature-relationship drift.

**9.4 Statistical power at long horizon + low turnover.**
This is the uncomfortable one. Quarterly rebalancing × multi-year holding × ~25 years = **very few independent bets**. You may be *unable* to statistically distinguish skill from luck at the confidence a fund needs. This is inherent to the mandate, not fixable by architecture — it's a reason to (a) lean on cross-sectional breadth for statistical power, (b) set expectations with the fund up front, and (c) treat the deflated Sharpe honestly.

**9.5 Optimistic cost/liquidity model.** Covered in §6 — flat 0.03% ignores spread/impact; deadliest on the microcaps. Mitigated by the liquid-subset start and per-asset liquidity-scaled `c`.

**9.6 Universe-selection survivorship (residual).** Covered in §7 — must build the liquid subset point-in-time, not as-of-today.

**9.7 Risk-model instability.** A bad Σ misallocates and gets blamed on α. Mitigated by shrinkage/factor model (§2), but it's a real, ongoing estimation problem, not a one-time fix.

**9.8 Learned second-moment / quantile forecasts may be worse than none (Stage A′ risk).** If you adopt the probabilistic forecaster, remember the second moment is *harder* to estimate than the first, and the first is already ~all noise (§9.1, §9.4). An **un-calibrated** predicted variance/downside that the optimizer *trusts* misallocates worse than a plain shrinkage Σ. **Gate:** validate out-of-sample calibration before use — do realized returns fall in the q10–q90 band ~80% of the time, across regimes? If not, fall back to point-α + realized-vol/shrinkage covariance. Never let an unvalidated risk head into the objective.

---

## 10. When (and whether) to graduate to RL / sequence models

Frame these as **V2/V3 research bets that must beat the V1 convex baseline out-of-sample, net of cost, on the deflated metric** — not as the inevitable destination.

- **Sequence model (Temporal Fusion Transformer / LSTM) for the α head:** justified *only if* it beats the GBM on the same walk-forward. Its plausible edge is modeling temporal dynamics your engineered features don't already capture — but §1.3 argues that's a thin margin given how much temporal structure is pre-engineered. Cheap to test as an α-head swap; the rest of the pipeline is unchanged.
- **Offline RL / Decision Transformer for the allocator:** the *only* defensible RL flavor here (no simulator → no online RL; you have logged/backtest trajectories → offline). But offline RL on one history path with distribution shift is fragile, and the convex optimizer is *already* the "planner" delivering dynamic, multi-period, cost-aware behavior in closed form. The bar for RL to clear: demonstrably better multi-period cost-aware behavior than Gârleanu-Pedersen-style convex multi-period optimization — which is a **high** bar, because that convex approach is provably optimal under its (reasonable) assumptions. **Do not start here.**

---

## 11. Phased roadmap

- [ ] **Phase 0 — Interim restatement sensitivity check (§9.2).** Jitter fundamental ratios at a stated calibrated magnitude, rerun backtest, check alpha survives. Run before trusting any fundamental alpha; does not replace the CVM v1 sourcing project below, but is cheap and immediate.
- [ ] **Phase 0.5 — (optional, scoped) Quantify restatement via CVM versioned ZIPs (§9.2).** True fix; larger sourcing effort, not yet attempted.
- [ ] **Phase 1 — Point-in-time liquid universe (§7).** ~30–50 names, liquidity recomputed *dynamically at every rebalance date* via one shared `(df, date, threshold) → tickers` helper. Reuse `test_top_traded_quality.py` / `TOP50_UNIVERSE_VALIDATION.md` scaffolding. **Never** a static current-day filter.
- [ ] **Phase 2a — Equal-weight baseline.** Using the Phase 1 point-in-time universe, measure the §8 core metrics (return, turnover, Sharpe, cost-sensitivity, regime slices) for simple quarterly-rebalanced equal-weight. This is the floor everything else must beat.
- [ ] **Phase 2b — Forecaster (Stage A).** LightGBM, forward-excess-return regression head, purged/embargoed walk-forward, §4.4 tainted/dropped columns excluded, spreads prioritized over absolute macro levels (§4.1) with feature-importance monitoring. Baseline: does α rank forward returns out-of-sample at all, and does it beat Phase 2a?
- [ ] **Phase 3 — Risk model Σ.** Ledoit-Wolf shrinkage or small factor model. Validate conditioning/stability.
- [ ] **Phase 4 — Cost-aware convex optimizer (Stage B).** Cash asset, L1-dominant elastic-net turnover penalty (`c₁` primary, `c₂` tuned empirically from ~0), per-asset liquidity-scaled `c₁`, caps. Solve the §5 program per quarter.
- [ ] **Phase 5 — Full walk-forward backtest + evaluation (§8).** Deflated Sharpe, turnover/holding-period distributions, regime slices, cost-sensitivity curve, baselines (equal-weight from 2a, IBOV, 100% CDI).
- [ ] **Phase 6 — Expand universe** point-in-time, once the machinery is validated on the liquid subset.
- [ ] **Phase 7+ (optional research) — sequence-model α head; offline-RL allocator.** Each must beat the V1 baseline net of cost on the deflated metric, or it doesn't ship.

---

## 12. One-paragraph answer to "what is the optimal V1?"

The optimal V1 is a **LightGBM cross-sectional forecaster of 6–12-month forward excess return over CDI, feeding a quarterly cost-aware convex optimizer that holds CDI as a first-class asset and penalizes turnover with an L1-dominant elastic-net term**, evaluated by purged/embargoed walk-forward with deflated Sharpe on a **dynamically recomputed, point-in-time liquid subset** of ~30–50 B3 names. It delivers all three prime-directive behaviors — cash rotation (closed-form from the tangency math, sharpened by the `earnings_yield_vs_selic` spread), value-quality selection (learned but supervised and auditable in α, via the valuation × Piotroski `f_score` interaction), and low-turnover multi-year compounding (L1 no-trade band × quarterly cadence) — **without** the single-history overfitting, uninterpretability, and reward-gaming fragility that deep RL would import. RL and sequence models are deferred to explicit research phases with a high, quantified bar to beat this baseline. The steepest risks are single-history overfitting and the as-restated-fundamentals lookahead — the latter addressed immediately by an interim sensitivity/jitter test (Phase 0) pending a proper CVM v1-sourcing measurement, not by a fancier model.
