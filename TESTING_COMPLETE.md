# Testing Complete: M1.1 + M2.1 Fixes Verified ✅

**Date:** 2026-07-09  
**Status:** Both fixes verified, tested end-to-end  
**Commit:** 006b17e

---

## Executive Summary

Two critical bugs were fixed and verified:

1. **M1.1: Excess-of-SELIC Metrics** — Revealed agent destroys alpha (-0.58 excess Sharpe)
2. **M2.1: Cap Weights Bug** — Allowed policies to express diversified portfolios (Effective_N 1.7→17.4)

The fixes expose an **uncomfortable truth**: with honest metrics and a working action space, the agent shows **no evidence of skill** (p=0.73). This is progress—we can now diagnose why.

---

## Before vs After

### The Measurement Problem (M1.1)

| Metric | Before | After |
|--------|--------|-------|
| **Agent Sharpe** | 1.59 | 1.59 (raw unchanged) |
| **Agent Excess Sharpe** | Hidden bias | **-0.58** (honest) |
| **What it means** | Looks smart | Destroys alpha |

**The fix:** Subtract daily SELIC before calculating Sharpe. This removes the "free carry" and reveals actual investment skill (or lack thereof).

---

### The Action Space Problem (M2.1)

| Behavior | Before | After |
|----------|--------|-------|
| **One-hot action** | 90% forced into CASH | 49 stocks + 1.87% CASH |
| **Effective_N** | 1.7 (concentrated) | 17.4 (diversified) |
| **Root cause** | Softmax sums to 1, but cap overflow went to CASH | Fixed: overflow redistributed to uncapped stocks |

**The fix:** Rewrite `_cap_weights()` to do water-filling redistribution instead of forcing overflow into CASH.

---

## Full Retrain Results (25 windows, 2001-2026)

### Honest Per-Window Performance (Excess-of-SELIC Sharpe)

```
Agent beats Equal-Weight: 12/25 windows (48%)
```

Sample windows:
```
w0: 2001-2002  agent=+0.641  ew=+0.632  diff=+0.009  WIN
w1: 2002-2003  agent=-0.176  ew=-0.204  diff=+0.028  WIN
w2: 2003-2004  agent=+3.633  ew=+3.675  diff=-0.043  loss
...
```

### Statistical Verdict

```
Regime-neutral t-test (daily agent − equal-weight):
  Mean daily excess return: +0.000017 (essentially zero)
  Std excess return:        +0.003760
  t-stat: +0.346
  p-value: 0.7290
  
Result: NOT significant at α=0.05
Interpretation: Cannot reject H0 that mean excess = 0
```

### Improved Metrics (M2.1 Effect)

```
Effective N (stocks):  17.4 ± 4.3  (was 1.7 before fix)
Max weight:            0.107 ± 0.004 (cap enforced)
Avg daily turnover:    0.200 ± 0.036
Max drawdown:          0.207 ± 0.106
```

The jump in Effective_N proves the cap fix is allowing diversification.

---

## What This Means

### ✅ Good News
- **Fixes work correctly** — Both M1.1 and M2.1 verified end-to-end
- **Metrics are now honest** — Excess Sharpe removes bias, cap allows expressions
- **Pipeline is healthy** — Can run full retrains, generate reliable results
- **We can diagnose properly** — Honest metrics enable root-cause analysis

### ⚠️ Challenging News
- **No statistical evidence of skill** — p=0.73 (cannot reject "random")
- **48% win rate** — Coin-flip level (50% would be random)
- **Zero mean excess return** — Agent adds noise, not alpha
- **This is the true state** — Not a measurement error, not a test bug

### 🔍 What's Next (Investigation)

The agent could be failing for several reasons:

1. **Features lack signal** (M5.3)
   - Test: Are the 19 features actually predictive of next-month returns?
   - Tools: Run IC analysis on test windows, compare to ranker baseline

2. **Representation bottleneck** (M6.1)
   - Test: Can a permutation-equivariant policy (per-ticker encoder) do better?
   - Issue: Flat MLP must learn ticker-slot-specific weights → 50× more parameters

3. **Reward design (even post-fix)** (M3.1)
   - Test: Try cash-aware reward where cash timing = 0 expected reward
   - Current: Excess vs equity-only EW still rewards going to cash

4. **Algorithm failure** (M6.5)
   - Test: Can the agent even memorize training data? (overfit sanity check)
   - If not, PPO optimization itself is the bottleneck

---

## How to Proceed

### Option A: Root-Cause Diagnosis (Recommended)
Implement M5 (Diagnose Where Edge Dies) to isolate the real bottleneck:
```
M5.1: Backtest BC-init (does RL destroy a good warm-start?)
M5.2: Overfit sanity check (can agent memorize training data?)
M5.3: Feature signal audit (do features carry IC on test windows?)
M5.5: Return attribution (split return into carry/timing/selection)
```

**Outcome:** A one-paragraph diagnosis naming the exact failure mode.

### Option B: Skip Diagnosis, Try Fixes
Implement potential solutions before understanding the problem:
```
M3.1: Cash-aware reward (may help if reward design is issue)
M6.1: Permutation-equivariant policy (may help if representation is issue)
```

**Risk:** Optimizing blindly; might waste effort on non-bottleneck.

### Option C: Accept the Result
Conclude that this problem/dataset/method combination doesn't support alpha:
```
- 12/25 windows beat EW (coin-flip)
- No statistical evidence (p=0.73)
- Honest measurement confirms: no skill
```

**Next:** Focus on a different problem or dataset.

---

## Key Learnings

1. **Honest metrics matter** — Raw Sharpe hid SELIC carry. Excess Sharpe revealed truth.
2. **Environment design matters** — A bug in the action space forced degeneracy.
3. **Multi-window testing matters** — Single backtest can be luck. 25-window average reveals truth.
4. **Statistical testing matters** — 12/25 wins sounds good until p-value says it's noise.

These fixes enable **real science**: honest measurement, reproducible results, proper diagnosis.

---

## Files

- **CHANGES_SUMMARY.md** — Technical details of M1.1 + M2.1
- **TEST_RESULTS.md** — Test methodology and findings
- **check_fixes.sh** — Repeatable verification
- **TODO.md** — Full roadmap (updated)

---

## Commits

```
eb0d59f - M1.1: Add excess-of-SELIC Sharpe and Sortino metrics
d0e1b21 - M1.1 (rolling_eval): Add excess metrics to rolling windows  
4472b17 - M2.1: Fix _cap_weights dead-code bug that forced cash
006b17e - test: Verify M1.1 + M2.1 fixes with full retrain
```

All changes are safe to keep; you can branch or revert as needed.
