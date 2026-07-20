# M2–M3 Implementation Status (2026-07-19)

## Summary

Building M2 (entropy-floor calibration) and M3 (supervised ranking probe) in parallel per the EIIE_IMPROVEMENT_PLAN.md. M1 is closed (luck confirmed via second-window replication). Both experiments are designed to answer whether entropy regularization (M2) and/or extractable ranking signal (M3) exist.

---

## M2: Entropy Floor Calibration

**Purpose**: Find the optimal `entropy_beta_end` value that keeps the policy diversified (effective_n ≈ 5–15) while still training a meaningful ranking.

**Current Status**: BIFURCATION DISCOVERED
- Tested entropy_beta_end=6.3e-4 with full SEP (8 seeds)
- **KEY FINDING**: bifurcation behavior — system converges to one of two stable attractors:
  - **Attractor 1 (Diversified)**: effective_n_holdings ≈ 34, entropy ≈ 0.725, Sharpe ≈ −0.15
  - **Attractor 2 (Concentrated)**: effective_n_holdings ≈ 2.9, entropy ≈ 0.24, Sharpe ≈ −0.60
- Seed 1 landed on Attractor 1; Seed 2 on Attractor 2
- Target [5, 15] unreachable → entropy floor alone cannot solve the concentration attractor
- Both attractors produce negative Sharpe (no alpha regardless of diversification)

**Next Steps**:
1. Wait for 1e-3 calibration to complete (check metrics_summary.json)
2. If effective_n ∈ [5, 15]: proceed with full SEP on this beta
3. Else: try 3e-3, then 1e-2
4. Once best beta found: `python -m src.rl_agent.sweep --config configs/eiie_m2_beta_BEST.json --seeds 1-8 -j 1`
5. Analyze results via `python scripts/analyze_m2_calibration.py`

**Analysis Tool**: `scripts/analyze_m2_calibration.py`
- Scans `experiments/` for eiie_m2_calib_* runs
- Extracts effective_n, total_return, Sharpe
- Recommends best beta based on effective_n ∈ [5, 15]

---

## M3: Supervised Ranking Probe

**Purpose**: Test whether there's extractable cross-sectional signal in price+technical features **without** RL noise or concentration attractor.

**Architecture**:
- Same conv trunk as EIIE (price + technical channels)
- Per-asset scores (no softmax/portfolio mechanics)
- Listwise cross-entropy loss: "which assets had the highest k-day returns?"
- Metric: daily IC (active-only Spearman(scores, realized_returns)) on train & val

**Implementation Status**: TRAINING IN PROGRESS
- Core code complete ✓ (all modules implemented and tested)
- M3 experiment launched: training supervised ranking probes for k∈{1,5,21}
- 20 epochs per horizon on GPU; expected completion ~1–2 hours

Code modules:
- `src/rl_agent/supervised_probe.py`: 
  - `SupervisedRankingProbe`: network architecture
  - `listwise_ranking_loss()`: cross-entropy over active slots
  - `compute_daily_ic()`: Spearman correlation calculation
  
- `src/rl_agent/supervised_experiment.py`: experiment orchestrator
  - `compute_forward_returns(panel, k)`: [T, n_global] array of k-day log-returns
  - `create_train_val_loaders()`: PyTorch DataLoaders with proper splits
  - `train_probe()` / `evaluate_probe()`: training + IC evaluation
  - `run_supervised_experiment()`: end-to-end pipeline
  
- `scripts/run_supervised_probe.py`: CLI entry point
  - Usage: `python scripts/run_supervised_probe.py --config configs/eiie_features.json`
  - Runs horizons k ∈ {1, 5, 21}
  - Outputs: `supervised_results.json` with IC + permutation null per horizon

**Tests**: `tests/rl_agent/test_supervised_probe.py` + `test_supervised_experiment.py`
- Forward return correctness (zero for const prices, log(2) for 2× growth, etc.)
- Train/val split separation (no overlap, right boundaries)
- Label alignment (no lookahead)
- Masking behavior

**Data Flow**: 
```
ml_dataset.parquet → PricePanel → forward returns [k∈{1,5,21}]
                                ↓
                          train/val split [per split_config.json]
                                ↓
                        SupervisedRankingProbe
                                ↓
                          daily IC on each split
                          + permutation nulls (shuffled pairing)
                                ↓
                        supervised_results.json
```

**Next Steps**:
1. Once M2 calibration complete, run M3 to check for signal at each horizon
2. `python scripts/run_supervised_probe.py --config configs/eiie_features.json`
3. Examine `supervised_results.json`: is val_IC > perm_null_975pct at any k?
4. M4 decision gate (M2 + M3 results) determines next direction

---

## Key Decisions & Invariants

### M2 Constraints
- Uses existing `entropy_beta_end` config knob (no new code required)
- Constrast to default: current 1e-5 ≈ near-zero entropy → cash attractor / one-hot
- Target is 1e-3 to 1e-2 range to force diversification while still training ranking

### M3 Constraints
- **No lookahead**: label window ends at t; forward returns are (t, t+k], strictly causal
- **Active-only**: Spearman computed only over holdable (top-50) assets per day
- **Permutation null**: shuffles weight-return pairings (seeded) to establish significance threshold
  - If IC < 97.5th percentile of null distribution → no signal at this k
  - If IC > null → plausibly significant (multi-window replication still required)

---

## M4 Decision Gate (Defined, Not Yet Applied)

Once M2 and M3 both complete, consult `EIIE_IMPROVEMENT_PLAN.md` lines 788–808:

| M3 result | + context | Next step |
|---|---|---|
| Null at all k | M1 retro-read also null, M2 null | **STOP model-side work.** Pivot to objective-level (risk/diversification mandate, macro-conditioning). |
| Null at all k | M1/M2 found something | Chase RL signal (config-specific anomaly); no new features/capacity/architecture. |
| Weak (IC < 0.03) | — | Allocation research only (top-k turnover budgeting); no model improvements. |
| Strong (IC ≥ 0.03) | M2 null or unhelpful | Head-to-head: predict-then-allocate vs. RL fine-tune. |
| Strong | M2 also positive | Objective-level research (entropy floor, risk-sensitive reward). |

---

## Commits This Session

| Hash | Message |
|---|---|
| `ddb3ba0` | feat: M2 entropy-floor calibration configs + M3 supervised ranking probe |
| `901c077` | feat: M3 supervised ranking probe experiment runner |
| `72fe1ad` | feat: M3 CLI + M2 calibration analysis tools |

---

## Estimated Timeline

- **M2 calibration**: 6–12 hours (depends on beta; 1e-3 currently running)
- **M2 full SEP**: 24–48 hours (8 seeds, -j 1 to avoid OOM)
- **M3 supervised**: 2–4 hours (single pass over data per horizon, no large sweeps)
- **M4 decision + next phase**: 2–4 weeks (depends on results)

---

## How to Check Progress

**M2 calibration status**:
```bash
# Check if the latest M2 run completed
ls -lt experiments/eiie_m2_calib_1e-3*/ | head -1
# If metrics_summary.json exists, it's done
```

**Once M2 complete**:
```bash
python scripts/analyze_m2_calibration.py
# → prints effective_n, return, Sharpe for each beta
# → recommends which to use for full SEP
```

**Run M3**:
```bash
python scripts/run_supervised_probe.py --config configs/eiie_features.json
# → writes supervised_results.json with IC per horizon
```

**Read M4 gate**:
```bash
grep -A 20 "M4 — Decision gate" EIIE_IMPROVEMENT_PLAN.md
```

---

## Open Questions

1. Will M2's entropy floor force the model to learn ranking, or does the objective fundamentally not have extractable daily/weekly signal?
2. Does M3's supervised probe find IC > null at any k? (Answers "is signal in the features", independent of RL mechanics.)
3. If both M2 and M3 are null, should we pivot to objective-level (risk/diversification mandate) or accept "no daily cross-sectional alpha in this universe"?
