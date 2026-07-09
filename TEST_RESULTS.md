# Testing M1.1 + M2.1 Fixes

**Date:** 2026-07-09  
**Changes tested:** 
- M1.1: Excess-of-SELIC metrics (subtract risk-free rate)
- M2.1: Fixed `_cap_weights` to redistribute overflow to stocks, not CASH

---

## Smoke Test Results ✓

### 1. Excess Metrics Verified
```
Agent raw Sharpe:     1.5918
Agent excess Sharpe: -0.5753  ← Honest: destroys alpha
SELIC excess Sharpe:  0.0000  ← Correct: risk-free rate minus itself = 0
```

**Verdict:** Excess metrics working correctly. Shows agent has negative alpha when SELIC carry removed.

### 2. Cap Weights Fix Verified
Before fix: One-hot action → 90% CASH (overflow forced)  
After fix:  One-hot action → 49/50 stocks with 1.87% CASH (overflow redistributed)

```
One-hot action [10, 0, 0, ...]:
  Active stocks: 49 (not 1!)
  Max weight: 0.1000 (cap enforced)
  CASH: 0.0187 (not 0.9!)
  Total: 1.0000 ✓
```

**Verdict:** Cap redistribution working. Policy can now express diversified portfolios.

---

## Full Retrain (In Progress)

Running: `python -m src.agent.trainer --timesteps 100000 --train-years 1 --test-years 1 --universe-size 50 --bc-pretrain`

**Purpose:** Verify training doesn't break and observe policy behavior changes.

Expected observations:
- [ ] Training completes without errors
- [ ] Per-window metrics include excess_sharpe
- [ ] Cash weights are policy decisions, not projection artifacts
- [ ] Effective_N increases for stocks (less forced concentration)
- [ ] Model file is valid and can be loaded

---

## Next Tests (if retrain succeeds)

1. **Backtest new model** - check if fixing the bias helps or if there's deeper skill issue
2. **Compare to old model** - see if policy behavior changed (should be less cash-heavy)
3. **Run evaluation** - generate full metrics.json with excess Sharpe for all baselines
4. **Check rolling_eval_results.json** - verify multi-window excess metrics

---

## Status

- **M1.1:** ✅ VERIFIED
- **M2.1:** ✅ VERIFIED  
- **Full pipeline:** 🔄 TESTING (retrain in progress)

Time to completion: ~10-15 min
