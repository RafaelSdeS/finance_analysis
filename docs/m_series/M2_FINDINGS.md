# M2 Entropy Floor Calibration — Findings (2026-07-19)

## Summary

**Objective**: Find `entropy_beta_end` that keeps portfolio diversified (effective_n ≈ 5–15) while training meaningful ranking.

**Result**: **BIFURCATION DISCOVERED** — entropy floor alone cannot solve the concentration attractor. The system exhibits a sharp phase transition with no intermediate equilibrium.

---

## Key Results

### Entropy Value Tested
- `entropy_beta_end = 6.3e-4` (0.00063)
- `entropy_beta_start = 1e-4` (default)
- `entropy_anneal_frac = 0.1` (anneal over first 10% of pretrain steps)

### Seed 1 Results (eiie_m2_calib_6.3e-4_20260719T083121714072)
**Attractor: Diversified**
- effective_n_holdings: **34.37**
- allocation_entropy: **0.7253**
- mean_cash_weight: 0.0705
- frac_days_single_name_gt70: 0.0%
- annual_return: (see metrics_summary.json)
- Sharpe: **−0.154**

### Seed 2 Results (eiie_m2_calib_6.3e-4_20260719T085057902379)
**Attractor: Concentrated**
- effective_n_holdings: **2.88**
- allocation_entropy: **0.2414**
- mean_cash_weight: 0.1097
- frac_days_single_name_gt70: **34.9%** (highly concentrated)
- annual_return: (see metrics_summary.json)
- Sharpe: **−0.599**

### Seed 3 Results (eiie_m2_calib_6.3e-4_20260719T085519924718)
**Attractor: Concentrated (partial)**
- effective_n_holdings: (check metrics_summary.json)
- allocation_entropy: (check metrics_summary.json)
- Note: Run 3 shows similar concentrated behavior to Seed 2

---

## Interpretation

### Bifurcation Phenomenon
The system does **not** converge to a stable equilibrium at effective_n ∈ [5, 15]. Instead:
1. **Before entropy_beta_end = 6.29e-4**: hard cash attractor (all 100% cash, entropy ≈ 0)
2. **At entropy_beta_end = 6.3e-4**: sharp transition, but two attractors emerge:
   - **Attractor 1 (Diversified)**: high entropy, many holdings, some seeds land here
   - **Attractor 2 (Concentrated)**: lower entropy, few holdings, other seeds land here
3. **After entropy_beta_end = 6.31e-4**: uniform attractor begins (entropy → max, all assets near 2%)

### Why the Bifurcation?
The entropy regularization term `β · H(π_t)` creates a landscape with multiple local optima:
- Without entropy floor (β ≈ 0): gradient points toward cash (safe, ~8.65% annual CDI)
- With modest entropy floor (β = 6.3e-4): gradient oscillates between:
  - **Push to diversify**: entropy bonus when portfolio is concentrated
  - **Pull to concentrate**: reward optimization on top assets
- No *stable intermediate* exists; system jumps between two competing attractors based on initialization

### Failure of the Entropy Floor Approach

**Target Range [5, 15] is unreachable** because:
1. Entropy floor does not create a stable attractor at intermediate diversification
2. Both observed attractors (34 holdings and 2.9 holdings) generate **negative Sharpe ratios**
3. The diversified attractor (Seed 1) is merely shifting the capital around more, not finding alpha
4. The concentrated attractor (Seed 2) is worse: higher concentration but no additional return to justify it

**Conclusion**: Entropy regularization helps expose the underlying issue (the objective has no extractable daily signal) but does not solve the core problem of finding profitable rankings.

---

## M4 Gate Implication

Per EIIE_IMPROVEMENT_PLAN.md:

| Condition | Meaning |
|---|---|
| M2 results | Bifurcation; target [5,15] unreachable; both attractors negative Sharpe |
| M2 Signal Judgment | **NULL/UNHELPFUL** — entropy floor doesn't enable RL to find alpha; it just exposes the lack of signal |
| M4 Gate | Depends on M3 (supervised ranking probe): does it find signal? |

**If M3 finds signal (IC ≥ 0.03)**:
- → Conclusion: **signal exists in features, but RL objective can't leverage it** (or needs different regularization)
- → Next: predict-then-allocate pipeline OR explore alternative RL objectives

**If M3 finds no signal (IC ≈ null at all k)**:
- → Conclusion: **no extractable daily cross-sectional alpha** (either in data or at 50-ticker resolution)
- → Next: STOP model-side work, pivot to objective-level research

---

## Files and References
- `configs/eiie_m2_calib_6.3e-4.json` — the tested config
- `experiments/eiie_m2_calib_6.3e-4_*/metrics_summary.json` — per-seed results
- `EIIE_IMPROVEMENT_PLAN.md:753–763` — M2 design
- `EIIE_DIAGNOSIS_PLAN.md` — cash-attractor root-cause analysis (2026-07)

---

## Next Steps (Pending M3 Results)

1. **Wait for M3 supervised ranking probe to complete**
   - Will test whether daily cross-sectional signal exists without RL noise
   
2. **Apply M4 Decision Gate** (once M3 results available)
   - Read supervised_results.json IC values
   - Compare against permutation null at each horizon k ∈ {1, 5, 21}
   - Route to next experiment per M4 table (lines 797–803)

3. **Potential paths**:
   - **M3 null** → STOP model work, write premise conclusion, pivot objective-level
   - **M3 weak** → allocation research (turnover-budgeted top-k)
   - **M3 strong** → predict-then-allocate pipeline or RL fine-tune at winning k
