# M1.1 + M2.1 Changes Summary

## What Was Fixed

### M1.1: Excess-of-SELIC Metrics ✅
**Problem:** Agent's Sharpe ratio (1.59) looked good but was mostly fake SELIC carry. Raw Sharpe with rf=0 hides the risk-free rate's contribution.

**Solution:** 
- Added `excess_sharpe_ratio()` and `excess_sortino_ratio()` functions
- These subtract daily SELIC returns from strategy returns before calculating Sharpe
- Updated backtest pipeline to report both raw and excess metrics

**Impact:**
- Agent's true excess-of-SELIC Sharpe = **-0.58** (destroys alpha)
- SELIC excess Sharpe = **0.00** (correct: risk-free minus itself = 0)
- Agent scores 25th percentile on excess metrics vs random policies

**Files changed:**
- `src/agent/metrics.py` - new functions
- `src/agent/evaluate.py` - pass selic_returns to compute_all, report excess percentile
- `src/agent/rolling_eval.py` - add SELIC baseline, use excess_sharpe for WIN/loss decisions

---

### M2.1: Fix `_cap_weights` Dead Code ✅
**Problem:** When policy tried to concentrate weights (e.g., action[0]=10), the cap at 0.10 would clip it, then the overflow went to CASH instead of redistributing to other stocks. This forced ~90% CASH allocations through the projection, not the policy.

**Root cause:** Softmax sums to 1 (including CASH). After capping, the check `if stock_sum > 1.0` never fires → dead code. All overflow went to CASH via `CASH = 1 - stock_sum`.

**Solution:**
- Rewrote `_cap_weights()` to use iterative redistribution (water-filling)
- Capped overflow is redistributed proportionally to active uncapped stocks
- CASH keeps its softmax weight (not forced to absorb overflow)
- Converges in ≤n_stocks iterations

**Impact:**
- One-hot action now diversifies to 49/50 stocks with 1.87% CASH
- **Before fix:** 1 stock (0.10) + 90% CASH
- **After fix:** 49 stocks diversified + 1.87% CASH
- Effective_N jumps from ~2 to ~38 for stocks

**Files changed:**
- `src/agent/env.py` - rewrite _cap_weights logic
- `tests/agent/test_env_basic.py` - updated cap test to verify redistribution

---

## Testing

### Smoke Test Results ✓
```
Excess Metrics:
  ✓ Agent excess_sharpe = -0.5753 (verified)
  ✓ SELIC excess_sharpe = 0.0000 (verified)
  
Cap Weights:
  ✓ One-hot action → 49/50 stocks (verified)
  ✓ CASH = 1.87% not 90% (verified)
  ✓ Weights sum to 1.0 (verified)
  ✓ Cap enforced at 0.10 (verified)
```

### Full Retrain (In Progress)
Running: `python -m src.agent.trainer --timesteps 100000 --train-years 1 --test-years 1 --universe-size 50 --bc-pretrain`

Status: Currently training window 9+  
Expected: Should complete in ~5-10 more minutes  
Command: `tail -50 /tmp/retrain.log | grep -E "window|Sharpe|complete"`

---

## How to Verify Results

### 1. Check new metrics are in backtest
```bash
python -c "import json; m = json.load(open('artifacts/backtest/metrics.json')); \
print(f\"Agent excess_sharpe: {m['agent'].get('excess_sharpe', 'N/A')}\")"
```

### 2. Run evaluationon the trained model
```bash
python -m src.agent.evaluate --model artifacts/models/agent_best.zip
```

### 3. Check rolling window results
```bash
python -c "import json; r = json.load(open('artifacts/models/rolling_eval_results.json')); \
wins = sum(1 for w in r['windows'] if w['metrics']['agent'].get('excess_sharpe', 0) > w['metrics']['equal_weight'].get('excess_sharpe', -999)); \
print(f\"Agent excess_sharpe wins: {wins}/{len(r['windows'])}\")"
```

---

## What This Means

**Before these fixes:**
- Metrics were biased (rf=0 in 12% SELIC regime)
- Cap forced 90% cash through projection
- Can't tell if policy learned anything

**After these fixes:**
- Metrics are honest (excess-of-SELIC)
- Cap allows policy to express decisions
- Can measure actual skill vs baselines

**Next steps:**
- M1.2/M1.3: Real staleness check (detect model/config mismatch)
- M3.1: Cash-aware reward (make cash timing = 0 expected reward)
- M4.1: Cash-blended baselines (agent must beat similar-risk strategies)

---

## Commits

1. `eb0d59f` - M1.1: Add excess-of-SELIC Sharpe and Sortino metrics
2. `d0e1b21` - M1.1 (rolling_eval): Add excess metrics to rolling windows
3. `4472b17` - M2.1: Fix _cap_weights dead-code bug that forced cash
