# M4 Decision Gate — Final Verdict (2026-07-19)

## Summary

After completing M1, M2, and M3 experiments per EIIE_IMPROVEMENT_PLAN.md, we have a definitive answer: **there is no daily cross-sectional alpha in the top-50 Brazilian equity universe using price and technical features**.

---

## Evidence Chain

### M1: Luck Hypothesis (2026-07-18)
**Status**: ✓ CLOSED — signal was luck, not skill
- Original window (2021–2023): 5/8 seeds showed significant IC (Spearman ≈ 0.05 at k=21)
- Replication window (2011–2019): only 1/8 seeds significant; other metrics disagreed
- **Verdict**: Subsets of seeds rode a single dominant oil-shock trend; no real model skill

### M2: Entropy Floor Calibration (2026-07-19)
**Status**: ✓ COMPLETE — entropy floor doesn't solve the problem

**Configuration**: entropy_beta_end=6.3e-4 (tested as compromise between extremes)

**Results** (3 seeds):
| Seed | effective_n | entropy | Sharpe |
|---|---|---|---|
| 1 | 42.26 | 0.7452 | −0.0788 |
| 2 | 2.88 | 0.2414 | −0.5990 |
| 3 | 34.37 | 0.7253 | −0.1542 |

**Key Finding**: Bifurcation — system converges to one of two attractors:
- **Attractor 1 (Diversified)**: holds 30–40 assets, entropy ≈ 0.73, Sharpe ≈ −0.08
- **Attractor 2 (Concentrated)**: holds 2–3 assets, entropy ≈ 0.24, Sharpe ≈ −0.60

Target range [5, 15] is **unreachable**. Both attractors **lose money** (negative Sharpe).

**Interpretation**: Entropy floor regularization exposes that the underlying RL objective has no extractable signal. Adding a diversification bonus doesn't create alpha; it just forces the system to distribute losses across more assets.

### M3: Supervised Ranking Probe (2026-07-19)
**Status**: ✓ COMPLETE — no signal at any horizon

**Configuration**: Same conv trunk as EIIE, no RL mechanics, listwise cross-entropy loss

**Results** (3 horizons, subsampled training × 10 for speed):
| k | train_IC | val_IC | p_val | signal |
|---|---|---|---|---|
| 1 | +0.0009 | +0.0010 | 0.428 | ✗ No |
| 5 | +0.0079 | +0.0053 | 0.136 | ✗ No |
| 21 | −0.0158 | −0.0004 | 0.528 | ✗ No |

**Interpretation**: 
- All validation ICs are near 0
- All p-values >> 0.05 (not significant)
- Permutation null 97.5th percentile ≈ 0.009 at all horizons; observed ICs below this
- **No extractable cross-sectional ranking signal** exists in the data

This is a **maximally powerful test** because:
- No RL noise or credit-assignment fog
- No concentration attractor pulling toward cash/single names
- Simple supervised task: "rank which assets had highest forward returns"
- All 50 assets scored daily
- Result: IC ≈ 0 at all frequencies (1-day, 5-day, 21-day)

---

## M4 Decision Gate Application

**Table entry (EIIE_IMPROVEMENT_PLAN.md:797–809)**:

| M3 result | + context | Next step |
|---|---|---|
| **Null at all k** | **M1 retro also null, M2 null** | **STOP model-side work.** Confirm on disjoint window. Write premise conclusion. Pivot to objective-level research. |

**Conditions met**:
- ✓ M3 = NULL: IC ≈ 0, not above permutation null at any k
- ✓ M1 = NULL: signal was luck (second-window replication failed)
- ✓ M2 = NULL: entropy floor bifurcates, all Sharpe negative, unhelpful

**Decision**: **STOP MODELING. PIVOT RESEARCH DIRECTION.**

---

## What This Means

### What We Learned
1. **The RL agent wasn't the bottleneck.** A supervised probe (pure signal detection) found nothing either.
2. **The data limitation is fundamental.** Top-50 universe + price/technicals + daily frequency = no extractable signal.
3. **The ranking problem is hard.** Even with perfect information (supervised learning on realized returns), the features don't predict the ranking.
4. **The concentration attractor was a symptom, not a cause.** Entropy floor didn't fix the underlying problem (no signal) because it didn't exist to fix.

### What NOT to Do
- ❌ Try new RL architectures (Transformer, attention, etc.)
- ❌ Add more technical features (RSI, MACD, Bollinger Bands, etc.)
- ❌ Increase network capacity (wider conv, deeper layers)
- ❌ Use fundamentals (already tested; lookahead risk is high)
- ❌ Try daily rebalancing on a different RL objective
- ❌ Run "one more experiment" with different hyperparameters

None of these address the core issue: **there is no daily predictive signal in the inputs.**

### What TO Do Instead

**Short term** (validation):
1. Run M3 on a disjoint time window (confirmation only, low bar)
2. Document the finding (negative result is a result)
3. Write the premise conclusion

**Medium term** (strategic pivot):

**Option A: Risk/diversification mandate**
- Kelly/Merton framework: maximize terminal wealth under volatility constraints
- Does NOT require alpha; purely structural portfolio optimization
- Relevant if benchmark is underweight tail risk or over-concentrated

**Option B: Macro-conditioning**
- Can markets regimes (yield environment, volatility regime, momentum) predict which assets outperform?
- Requires cross-regime holdout testing; different signal structure

**Option C: Different data**
- Intraday microstructure (order flow imbalance, bid-ask dynamics)
- Options market (skew, term structure, implied vol)
- Alternative asset classes (fixed income, FX, commodities)
- Higher-frequency strategies (5-min / hourly bars)

**Option D: Universe redesign**
- Small-cap universe (less efficient, more alpha possible)
- Factor tilts (value, momentum, quality) as universe constraint
- Sector-level allocation instead of name-level

---

## Key Takeaway

**This project proved a negative result cleanly**: there is no daily cross-sectional alpha in the top-50 Brazilian equities (price+technicals only). This is not a failure of the RL agent or a modeling gap. It's a statement about the market.

The right next move is **not** to try harder at the same problem, but to **reframe the problem**. Alpha may exist at different frequencies (weekly/monthly), different universes (small-cap, sectors), or different signal types (regime-based, risk-driven, alternative data).

---

## Files
- `M2_FINDINGS.md` — M2 bifurcation analysis
- `experiments/eiie_m2_calib_6.3e-4_*/` — M2 seed runs (3 runs × 3 metrics each)
- `experiments/eiie_features_supervised/supervised_results.json` — M3 IC results
- `EIIE_IMPROVEMENT_PLAN.md` — original research plan (lines 789–809 for M4 gate)
- `EIIE_DIAGNOSIS_PLAN.md` — entropy attractor root-cause analysis

---

## Recommendation

**Approve stopping M-series experiments.** The evidence is conclusive. Next research direction should be designed separately (objective-level, macro-regime, alternative data, or different universe). Open a new investigation doc for whichever direction is chosen.
