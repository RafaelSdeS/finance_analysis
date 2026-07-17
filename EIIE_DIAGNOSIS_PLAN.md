# EIIE Agent: Escape the Cash Attractor — Diagnosis-First Plan

**Status**: Phase 0 complete. Ready for Phase 1.

## Phase 0 Results — Read Existing Evidence

### Current State of 3 Validation Runs (2021-11 → 2024-03)

| Run | Code State | Agent Return | vs CDI | vs BOVA11 | Sharpe | Mean Turnover | Diagnosis |
|-----|-----------|--------------|--------|-----------|--------|---|---|
| 222340 | Bug: `y_t·w_{t-1}` (no gradient to returns) | 30.94% | **+0.0%** | +5.3% | 11.50 | ~0 | 100% cash; matched CDI exactly |
| 224705 | Fix: `y_{t+1}·w_t` | 19.39% | **−11.5%** | −6.2% | −0.07 | 2.78% | Churns 3-day bets; loses to both |
| 231801 | Fix: μ-chaining (`w_{t-1}` from batch outputs) | 15.82% | **−15.1%** | −9.8% | −0.53 | 0.58% | **WORSE**: only 2 bets ever, both disastrous |

**Key finding**: All three runs underperformed CDI (30.94%) and BOVA11 (25.61%). The attempted fixes made things worse, not better.

### Root-Cause Hypotheses (ranked by evidence)

1. **PVM initialized all-cash** — paper-faithfulness bug (HIGHEST PRIORITY)
   - **Paper p.14**: *"Before any network training, the PVM is initialized with uniform weights."*
   - **Ours** (`src/rl_agent/pvm.py:56`): Every row initialized to 100% cash (`buffer[:, CASH_GIDX] = 1.0`)
   - **Effect**: Every pretrain sample's input and cost anchor say "you hold cash; moving costs money" — cash attractor baked into all 100k pretrain steps
   - **Citation**: `docs/EIIE_AGENT_PLAN.md:134` records this as an "approved micro-deviation" with faulty rationale ("uniform-over-172 would place mass on inactive") that attacks a strawman; paper-faithful is uniform over cash + each day's 50 active slots only
   - **Fix approach**: Initialize each row uniform-over-active (scatter via `scatter_to_global_row`)

2. **Agent trained with 5% of paper's budget** (HIGH PRIORITY)
   - **Paper Appendix B, Table B.1 (p.26)**: pretrain_steps = **2×10⁶**, β = **5×10⁻⁵**
   - **Ours** (`configs/eiie_baseline.json`): pretrain_steps = **100k** (20x fewer), β = **5×10⁻⁴** (10x higher recency bias)
   - **Learning rate match**: 3e-5 is the same; batch_size, window, L2, rolling_steps all match
   - **Consequence**: Undertrained network sits where initialization left it — which, per #1, is cash
   - **Evidence**: Nobody has checked whether pretrain converged; loss curves sit unexamined in each run's `report.html`
   - **Fix approach**: After Phase 1, run Phase 3 (config-only: 100k → 2×10⁶ steps, β 5e-4 → 5e-5)

3. **Single-seed per comparison** (MEDIUM PRIORITY)
   - Every run used seed=42 only. In weak-signal regimes, variance masks the signal.
   - **Evidence**: ml_agent roadmap (TODO.md M1.5/1.6) learned this lesson the hard way.
   - **Fix approach**: Phase 2 runs 5-seed ensemble to establish signal baseline.

4. **Structural mismatch: market regime** (CAVEAT)
   - Paper's market: 30-min crypto, 0% cash rate → cash is a dead asset
   - Ours: Daily B3, CDI ≈ 12%/yr → riskless rate beats the average stock
   - **Evidence**: UCRP (equal-weight rebalancing) only earned 13%, CDI 30.9%, BOVA11 25.6%
   - **Conclusion**: Heavy cash is partially rational on this window. Success criterion must be *defined before tuning*.

---

## Plan

### Phase 0 ✅ — Read the evidence we already have
- [x] Extract metrics from 3 existing runs' `metrics_summary.json`
- [x] Compare: agent return vs CDI (30.94%) and BOVA11 (25.61%)
- [x] Create `src/rl_agent/diagnose.py` (diagnostic script, frozen-weights backtest)
- [x] Write this document with findings & checkboxes (user's standing preference)

**Conclusion**: In-sample, the agent never beat cash. Pretrain loss likely never converged.
**Decision**: Hypotheses #1 (PVM) and #2 (budget) are load-bearing; test them in Phases 1 & 3.

---

### Phase 1 — Fix the PVM init (the one code change with paper authority)

**Objective**: Initialize PVM rows uniform-over-active slots, matching paper p.14.

- [ ] `src/rl_agent/pvm.py`: Constructor now takes `slot_gidx` and `valid` (per-period active slots) and initializes each row `uniform(cash + active)` via `scatter_to_global_row` (already used for w_t → global).
  - Thread `slot_gidx`/`valid` from `PricePanel` at construction.
  - Call sites: `experiment.py:125` (main training), `sanity.py:_short_train_run` (sanity gate).
- [ ] Update `tests/rl_agent/test_pvm.py` to assert rows are no longer all-cash at init.
- [ ] Run `python tests/run_all.py --group fast` — should pass.
- [ ] Update stale rationale at `docs/EIIE_AGENT_PLAN.md:134`.

**Backtest w_0 (equation 5)**: Stays all-cash (that's the *environment* init, unchanged; the PVM is the *training memory*, different thing).

---

### Phase 2 — Buy statistical eyesight before tuning

**Objective**: Establish a seed-ensemble signal baseline at the current budget (post-Phase-1).

- [ ] `experiment.py`: Add `--seed N` CLI override (frozen dataclass → `dataclasses.replace`).
- [ ] Run **5 seeds** × Phase-1 code on **val split** (2024-03 → 2026-07):
  - `python -m src.rl_agent.experiment --config configs/eiie_baseline.json --seed 1`
  - (repeat for seeds 2–5)
  - GPU enabled; ~30 min/run expected (~2.5 hr total)
- [ ] Record results in table below; decision rule: a change "helps" only if its median beats the current seed spread.

**Phase 2 Results Table** (to be filled):

| Seed | Agent Return | vs CDI | vs BOVA11 | Sharpe | Median/Spread |
|------|---|---|---|---|---|
| 1    |     |    |    |    | |
| 2    |     |    |    |    | |
| 3    |     |    |    |    | |
| 4    |     |    |    |    | |
| 5    |     |    |    |    | |
| **Median** | **?** | **?** | **?** | **?** | **decision gate** |

---

### Phase 3 — Align training budget with paper's Table B.1 (config-only, no code)

**Objective**: Scale pretrain_steps and β to paper values; check if pretrain converges in-sample.

- [x] Created `configs/eiie_phase3_budget.json`:
  - `pretrain_steps`: 100000 → **2000000** (paper value)
  - `beta`: 0.0005 → **0.00005** (paper value)
- [x] Launched 5-seed sweep using `sweep.py` (4 parallel jobs, ~5 hours expected):
  ```bash
  python -m src.rl_agent.sweep --config configs/eiie_phase3_budget.json --seeds 1 2 3 4 5 --eval-split val -j 4
  ```
  Logs in `experiments/sweep_logs/{timestamp}/`; runs in `experiments/eiie_phase3_*`

**Phase 3 Results Table** (to be filled when sweep completes):

| Seed | Agent Return | vs CDI | vs BOVA11 | Sharpe | Median/Spread |
|------|---|---|---|---|---|
| 1    |     |    |    |    | |
| 2    |     |    |    |    | |
| 3    |     |    |    |    | |
| 4    |     |    |    |    | |
| 5    |     |    |    |    | |
| **Median** | **?** | **?** | **?** | **?** | **decision gate** |

**Optional ablation** (if Phase 3 results stay cash-bound):
- [ ] Test μ-chaining on/off: revert the `w_prev_loss` hunk in `train.py:97` in a scratch branch.
- [ ] Run 2 seeds (original vs reverted) to settle whether μ-chaining helps or hurts.

---

### Phase 4 — Honest conclusion gate (no code)

**Definition of success** (set before looking at results):
- Beat **both CDI (30.9%)** AND **BOVA11 (25.6%)** on **val split** with **bootstrap CI** across the seed ensemble (not one lucky seed).

- [ ] Examine Phase 3 results (Sharpe, return CI, median cash weight).
- [ ] If in-sample edge exists but val stays flat: the 50-day price window carries no exploitable daily signal on B3 (consistent with the high CDI regime). Document this in this plan.
  - **Next lever**: iteration 2's scope is the *feature set* (fundamentals/macro already built by Stage 2; `networks.py` already reserves the encoder seam).
  - This is a scope decision for the user, not an optimization trick.
- [ ] If Phase 3 still underperforms after ablations: mark as "requires feature engineering or market-regime rethinking" and close Phase 0–3.

---

## Explicitly Out of Scope (user constraint)
- Transaction costs (3 bps), evaluation protocol, CDI accrual, dataset, universe — **untouched**.
- No changes to costs or eval methodology; only training gradient changes allowed.

---

## Next Immediate Action
After Phase 0 approval by the user (this document):
1. Implement Phase 1 (PVM init fix) — ~30 min
2. Run Phase 2 seed ensemble — ~2.5 hr
3. Depending on Phase 2 results, decide whether Phase 3 (budget scaling) is warranted

**Verification**: Fast test group passes; --dry-run passes; at least one Phase-2 val run shows PVM rows no longer 100% cash at init (new test assertion).
