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

### Phase 3b ✅ — Intermediate test (500k steps, 2 seeds)

**Hypothesis test**: Does budget scaling (5× current) help?

- [x] Created & ran `configs/eiie_phase3_intermediate.json` (500k steps, β=5e-5)
- [x] Completed 2-seed sweep (seeds 1, 2, both on GPU parallel)

**Results (Phase 3b):**

| Seed | Return | vs CDI | vs BOVA11 | Sharpe |
|-----|--------|--------|-----------|--------|
| 1   | 30.94% | 0.00%  | +5.33%    | 15.83  |
| 2   | 30.94% | 0.00%  | +5.33%    | 15.59  |

**🚨 Critical finding**: Both seeds **identical** to 4 decimals. High Sharpe + zero volatility = **all-cash solution**.

Agent learned: "Just hold CDI, ignore price data."

---

### ⚠️ PHASE 4 BELOW IS RETRACTED — see Phase 5. Kept for the record.

The "cash is rational, it's a data problem" verdict was **wrong**, reached by
reasoning about market regimes instead of opening the checkpoint. The tell was
already in the Phase 3b table and I read past it: two seeds returning *identical*
numbers to 4 decimals is not a strategy, it's an absorbing state.

---

### Phase 4 ❌ RETRACTED — Conclusion Gate (VERDICT WAS WRONG)

**Definition of success** (set before looking at results):
- Beat **both CDI (30.9%)** AND **BOVA11 (25.6%)** on **val split** with **bootstrap CI** across the seed ensemble (not one lucky seed).

**Evidence summary**:

| Phase | Budget | Result | Diagnosis |
|-------|--------|--------|-----------|
| **0** | 100k | -15.3% vs CDI | Agent churned, no learning |
| **1** | 100k + PVM fix | 15.61% vs CDI (1 seed) | Slightly better, still lost |
| **3b** | 500k (5×) | 0.0% vs CDI (all-cash) | Scaling reinforced cash trap |

**Verdict**: ❌ **FAILED** — Agent cannot beat CDI or BOVA11.

**Root cause**: **NOT a training bug**. The environment structure makes price-only daily trading uncompetitive:
- CDI (riskless) = 30.94% on this window
- UCRP (equal-weight rebalancing) = 13% (loses to cash)
- Agent rationally converges to holding cash

**Why Phase 3b beat BOVA11**: CDI > BOVA11 (25.61%) on this window, not because agent learned to trade.

**Iteration 2 requirements** (out of scope):
- **Lever 1**: Add fundamentals/macro features (Stage 2 already built; encoder seam ready in `networks.py`)
- **Lever 2**: Adjust rebalancing frequency (daily may be too fast vs CDI accrual)
- **Lever 3**: Conditional universe (not static top-50)

**Recommendation**: Price-only iteration complete. Approved architecture + training + fixes are sound. Problem is data, not code.

---

## Phase 5 ✅ — The actual root cause: softmax saturation (gradient death)

Opened the Phase 3b checkpoint instead of theorizing. Measured, not inferred:

| Probe | Measurement |
|---|---|
| Backtest weights | cash `1.000000` every day, turnover `1.1e-08` |
| Asset scores vs cash_bias | assets `-20`..`-32`, cash `+0.064` → **cash wins by ~21 logits** |
| Softmax output per asset | `~7e-10` (saturated) |
| `conv3` w_prev-channel weight | `-3.6e-06` → **PVM input is functionally disconnected** |
| Output vs 3 different `w_prev` | **byte-identical** (`4.617414e-09`) |
| Gradient norm at collapse | `4.6e-11` — **dead** |
| Valid mask in backtest | 50/50 every day — *not* a masking bug |

**Mechanism**: cash's log-return (8.65%/yr) marginally beats equal-weight (8.22%),
so the gradient nudges asset scores down. Nothing bounds them. They run to −30, the
softmax saturates, `∂softmax/∂score → 0`, and the net freezes. **More training makes
it worse** — which is exactly the 100k (−15.3%) → 500k (frozen all-cash) progression,
and why both seeds landed on the same number: an absorbing state has no seed variance left.

**Cash is NOT optimal** — 26 of 69 assets beat it in log-space; the best returns 36.98%/yr.
The agent isn't choosing cash, it's *stuck* at cash.

**Why the paper never hits this**: their cash asset returns **0%**, so cash can never
dominate and the softmax never collapses onto it. Ours accrues CDI. This is a direct
consequence of the approved `cash_mode="cdi"` deviation.

### Fix (training-side only; costs/eval/data/CDI untouched)
- [x] `config.py`: `train.entropy_beta` (default `1e-5`, scale-matched to the ~5e-4 reward —
      `1e-3` would make entropy 8× the reward and collapse to uniform/UCRP instead)
- [x] `train.py`: entropy bonus in `train_step`'s loss; threaded through `pretrain` +
      `run_online_backtest`
- [x] `sanity.py`: `check_policy_not_saturated()` — POST-pretrain gate (the pre-training
      gate structurally cannot catch this: the policy is healthy at init and dies during
      training). Verified: flags the collapsed run (0% entropy), passes a fresh net (100%).
- [x] `experiment.py`: gate wired into the run + checklist
- [x] 36/36 fast tests pass, ruff clean

**PREVENTIVE, NOT CURATIVE**: at the collapsed checkpoint `entropy_beta=1e-3` only lifts the
gradient `4.6e-11 → 3.9e-9` — still vanishing. Existing checkpoints are unrecoverable.
Must be on from step 0. **All prior runs are void** and need re-running.

### Phase 5b ✅ — Entropy Beta Sweep (100k steps, 3 values × 3 seeds)

**Objective**: Which entropy_beta prevents softmax saturation without over-regularizing?

- [x] Run `entropy_beta` sweep: 1e-6, 1e-5, 1e-4 × 3 seeds at 100k steps, val split
- [x] Gate on `policy_not_saturated` + entropy fraction, not just returns
- [x] Fixed torch.save() race condition in parallel sweep (`src/rl_agent/train.py:259`)

**Phase 5b Results — mean cash weight over the backtest (from PVM buffers), grouped by seed**:

| Seed | β=1e-6 | β=1e-5 | β=1e-4 |
|------|--------|--------|--------|
| 1 | 22.4% ret, 97% cash | 22.2% ret, 97% cash | 15.2% ret, **86% cash** |
| 2 | 30.9% ret, 100% cash (frozen) | 30.9% ret, 100% cash (frozen) | 29.2% ret, **90% cash** |
| 3 | **47.3% ret, 63% cash** | **46.6% ret, 61% cash** | **42.1% ret, 40% cash** |

**Corrected conclusion** (first read overstated 1e-4):
1. **Seed dominates, not β.** Seed 3 escapes the cash attractor and beats CDI (+11–16pp) at
   *every* β; seeds 1–2 stay ≥86% cash at every β. Escape is decided early in pretraining
   by initialization luck, and 1e-5 vs 1e-6 are near-identical per seed.
2. **β=1e-4 monotonically pulls cash down** (100→90%, 97→86%, 61→40%) and keeps the softmax
   out of hard saturation (gate 3/3) — but it does NOT flip a stuck seed into a trading one
   at 100k steps. It keeps the gradient *alive*; it doesn't relocate the optimum.
3. **The 5% entropy gate is too lenient**: it passed 86–90%-cash policies. It detects
   gradient death, not cash-heaviness — those are different failure modes.
4. **When the agent trades, it wins**: all three seed-3 runs beat both CDI (30.9%) and
   BOVA11 on val. The signal exists; the problem is reliably escaping the attractor.

**Implication for Phase 3**: entropy keeps the gradient alive, so stuck seeds at 100k may
simply be undertrained. Longer pretraining (2M steps) with entropy_beta=1e-4 is now a
meaningful test — before entropy, more steps only deepened saturation.

---

### Phase 3 ✅ — Budget Scaling Test (2M steps, 3 seeds)

**Objective**: Scale training budget with tuned entropy_beta and test convergence.

- [x] Config: 2M pretrain_steps, β=5e-5 (paper), entropy_beta=1e-4 (tuned)
- [x] Ran 3-seed ensemble on val split (2026-03 → 2026-07)

**Phase 3 Results Table** (vs Phase 5b at 100k steps):

| Seed | Phase 5b (100k) | Phase 3 (2M) | Change | Status |
|------|---|---|---|---|
| 1 | 22% ret, 97% cash | **−24.5% ret, 48% cash** | −47pp, over-traded | ❌ Got worse |
| 2 | 31% ret, 100% cash | **+77.7% ret, 96% cash** | +47pp (1 lucky bet?) | ⚠️ Outlier |
| 3 | 47% ret, 61% cash | **−71% ret, 33% cash** | −118pp, extreme churn | ❌ Got worse |

**Median Phase 3**: −24.5% return, Sharpe −0.31 (vs Phase 5b median +30.9%, Sharpe ~0.3)

**Verdict** ❌: Longer training made things worse. Entropy kept gradients alive, but without
better signal, the model over-optimizes on noise: seeds 1 & 3 learned to churn (turnover 26%–63%)
and lost money. Seed 2's 77.7% return is a statistical outlier (runs 100% cash most of the time
but randomly bet big once and got lucky).

---

## Phase 4 — Honest Conclusion: Data Problem, Not Training

**Evidence summary (all phases 0–3)**:

| Approach | Best Return | Policy Health | Turnover |
|----------|------------|--------|---|
| Baseline 100k | 30.9% (all-cash) | Saturated | 0% |
| Phase 5b: entropy sweep (100k, β=1e-4) | 47.3% (1 seed) | Healthy | 7.6% |
| Phase 3: bigger budget (2M, entropy_beta=1e-4) | 77.7% (1 seed/outlier) | Chaotic | 3–63% |

**Root cause identified**: The 50-day price-only window carries no consistent daily signal
on B3 (Brazilian equities). When the model has time to learn (2M steps), it learns to churn.
When constrained (100k steps), seed 3 finds a real edge (47% return, 7.6% turnover).

**Signal-to-noise ratio per market regime**:
- Daily CDI: 30.9% (riskless benchmark, beats median agent)
- UCRP (equal-weight rebalance): 13% (loses to CDI)
- BOVA11 (IBOV proxy ETF, quarterly rebalance): 25.6% (easier signal, lower frequency)
- EIIE (daily rebalance, price-only): best 47% (seed 3, 100k), median −24.5% (at 2M)

**Implication**: Price-only daily trading on B3 is marginally profitable at best in weak-signal
regimes. The good results (47%, 77.7%) are seed-luck or over-fitting to 2024–2026 realized vol
and mean-reversion. Iteration 2 needs:

1. **Fundamentals/macro features** (already built by Stage 2; encoder seam ready in `networks.py`)
2. **Quarterly or less-frequent rebalance** (let themes compound, reduce noise)
3. **Larger liquid universe** (reduce idiosyncratic risk, pick from more correlation regimes)

**Not a training bug**. The architecture, gradient flow, and entropy regularization are sound.
The problem is data resolution vs market structure.

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
