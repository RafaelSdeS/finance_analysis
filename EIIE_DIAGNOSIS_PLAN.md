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

---

## Phase 5 — From a Lucky Seed to a Repeatable Escape (implemented, not yet run)

Digging into *why* seed 3's Phase-5b run (47% return, 100k steps, `entropy_beta=1e-6`) worked
showed it wasn't a coin flip on returns — the PVM held ~60% cash steadily the whole val window
(2018-07-30 → 2022-07-26), turnover was a boring ~5-7%/day outside the COVID crash spike, and
the equity sleeve concentrated in PRIO3/WEGE3, two of B3's real multi-year compounders over
that window. A found, low-order momentum heuristic, not a jackpot bet. But the *same* seed 3
given 20x more budget (Phase 3, 2M steps) collapsed to −71% — it overfit the heuristic away.
Seeds 1/2 never found the basin at all, at any budget. So "100k steps" was a sweet spot for
one seed's init, not a property of the method.

**Three levers implemented** (`src/rl_agent/config.py`, `train.py`, `experiment.py`,
`sanity.py`; still inside the no-cost/no-eval/no-data constraint):

- **A. Checkpoint-at-peak** — `pretrain()` now carves the last `checkpoint_holdout_days`
  (default 250, ~1yr) off the *train* split's tail, samples OSBL batches only up to that
  boundary, and every `checkpoint_eval_every` (default 5000) steps scores the frozen policy
  on that untouched holdout via the existing `run_backtest`/`agent_forward`/`total_return`
  path. Keeps the best-scoring `state_dict` and restores it (+ refreshes the PVM boundary row
  under the restored weights) instead of trusting wherever `pretrain_steps` happens to land.
  Directly targets the Phase-3 finding: budget no longer needs to be guessed, since overfitting
  past the peak can't survive the restore.
- **B. Annealed `entropy_beta`** — replaced the flat `entropy_beta` with
  `entropy_beta_start=1e-4` decaying linearly to `entropy_beta_end=1e-5` over the first
  `entropy_anneal_frac=0.1` of pretrain, then flat for the rest of pretrain and the whole
  online/live phase. A fixed beta only helped the seed that was already going to escape on its
  own (the Phase-5b sweep showed seed 3 escaping at *every* beta tested while seeds 1-2 stayed
  cash-bound at *every* beta) — forcing more exploration early is aimed at seeds 1/2, not seed 3.
- **C. Wider seed ensemble** — no code needed, `sweep.py --seeds` already supports it; just
  hasn't been run at n>3 yet.

**Status**: code + tests done (`tests/rl_agent/test_train.py::test_entropy_schedule`,
`::test_pretrain_checkpoint_at_peak`), fast group green, `--dry-run` verified. All existing
`configs/eiie_*.json` updated to the new field names (old `entropy_beta` key removed).

### Phase 5 Results (2026-07-17, seeds 1–8 completed; 9–10 Ctrl-C'd mid-run)

Val window (from each run's manifest, window-scoped split): **2021-11-30 → 2024-03-21**.
Benchmarks on this window: CDI 30.9%, BOVA11 25.6%, UBAH 37.0%, UCRP 13.1%.

| seed | return | sharpe | maxdd | turnover | mean cash | best_step | holdout_ret | behavior |
|-----:|-------:|-------:|------:|---------:|----------:|----------:|------------:|----------|
| 1 | 16.9% | −0.35 | 0.195 | 0.5% | 96.6% | 99999 | +5.9% | cash |
| 2 | 30.9% | — | 0.000 | 0.0% | 100% | 79999 | +3.8% | pure cash |
| 3 | 29.6% | 0.09 | 0.170 | 14.1% | 63.3% | 99999 | −3.7% | mixed |
| 4 | **41.9%** | 0.40 | 0.104 | 0.8% | 99.0% | 94999 | **+30.5%** | cash + 11 days of lottery bets |
| 5 | 25.1% | −0.24 | 0.116 | 0.7% | 97.8% | 34999 | +7.0% | cash |
| 6 | −15.4% | −0.22 | 0.491 | 7.6% | 0.3% | 64999 | +1.1% | all-in high-beta |
| 7 | 9.4% | −0.22 | 0.257 | 1.8% | 3.0% | 4999 | −9.1% | all-in high-beta |
| 8 | 5.5% | −0.09 | 0.338 | 5.8% | 0.5% | 54999 | +1.7% | all-in high-beta |

Median: 21.0% return, −0.15 Sharpe. Still below CDI.

**What the anneal actually did — bimodal, not balanced.** Before (Phase 5b), 1 seed in 3
escaped cash partially. Now 4 of 8 escape — but the escaped policies overshoot to ~0–3% cash,
all-in on equities. The system is bistable (all-cash vs all-in) and `entropy_beta_end=1e-5`
is too weak to hold a middle ground after the anneal window closes. No seed found a
persistent diversified sleeve.

**What the escaped seeds bought is the real indictment**: BHIA3, MGLU3, LWSA3, HAPV3, AZUL4 —
the distressed, highest-volatility fallen names of the 2021–2023 tightening cycle, exactly
the names that kept collapsing through this val window (seed 6: −15.4% with a 49% drawdown).
A price-only CNN reads a dead-cat bounce in a high-vol name as momentum. It has no notion of
quality — same conclusion as Phase 4, now with a sharper mechanism.

**Seed 4's 41.9% is not a strategy**: 11 days (of 576) with equity >10%, concentrated
single-name lunges into BHIA3/HAPV3/AZUL4 that happened to pay. Lottery tickets on top of CDI.

**The genuinely positive result — the holdout score transfers across seeds**:
corr(holdout_return, val_return) = 0.54 Pearson / 0.52 Spearman (n=8, p≈0.18 — suggestive,
not significant). Picking the seed with the best holdout return (a legal, train-only
selection rule) selects seed 4, which is also the best val seed. Within-run, though, most
best_steps landed near the end (the holdout rarely triggered real early stopping), and the
gate `policy_not_saturated` correctly flagged the cash-collapsed seeds (1, 2, 4).

**Verdict**: the training machinery now works as designed — escapes happen, collapse is
flagged, checkpoint selection carries signal. Performance is still poor because the policy's
asset-picking is anti-signal (buys crashing high-beta names). This is the Phase 4 data-problem
conclusion again, unchanged: price-only daily bars on B3 don't identify *what's worth
holding*, only *what moved recently*. Iteration 2 levers remain: fundamentals/macro features,
lower rebalance frequency, or both.

---

## Phase 6: Technical feature channels (2026-07-17, `configs/eiie_features.json`, seeds 1–8)

Same val window (2021-11-30 → 2024-03-21), same benchmarks (CDI 30.9%, BOVA11 25.6%,
UBAH 37.0%). One variable changed vs Phase 5: 3 → 11 input channels
(`return_1m/3m/6m`, `price_vs_ma60`, `volatility_ratio_20_60`, `rsi_14`, `drawdown`,
`volume_ratio_20d` on top of close/high/low). Cadence, costs, reward untouched.

| seed | return | sharpe | maxdd | turn/day | cost drag | mean cash | best_step | holdout_ret | book |
|-----:|-------:|-------:|------:|---------:|----------:|----------:|----------:|------------:|------|
| 1 | 18.6% | −0.51 | 0.090 | 1.4% | 0.3% | 93.7% | 9999 | −0.9% | cash + BHIA3 dust |
| 2 | 30.8% | 0.12 | 0.288 | 15.6% | 4.5% | 18.5% | 24999 | +17.3% | PETR3/4, BIDI11, EMBJ3 |
| 3 | 8.5% | −0.06 | 0.258 | 48.9% | 12.9% | 33.4% | 44999 | +3.0% | BHIA3/MGLU3/AZUL4 churn |
| 4 | 12.9% | −0.17 | 0.259 | 5.7% | 1.9% | 0.0% | 4999 | −9.5% | PETR3/4, GGBR4, PRIO3 |
| 5 | −14.9% | −0.50 | 0.407 | 15.1% | 3.8% | 36.5% | 99999 | +44.3% | PETR + BHIA3/AZUL4 mix |
| 6 | **53.2%** | 0.36 | 0.488 | 51.0% | 16.2% | 0.0% | 39999 | **+59.6%** | PETR3/4 (45%), PRIO3, GGBR4 |
| 7 | 27.4% | −0.35 | 0.025 | 0.9% | 0.2% | 96.8% | 9999 | −1.3% | pure cash |
| 8 | −6.2% | −0.50 | 0.251 | 1.4% | 0.3% | 93.0% | 69999 | +11.4% | cash + BHIA3 dust |

Median 15.8%, mean 16.3% — vs Phase 5's median 21.0%. Headline unchanged: ensemble still
below CDI. But the composition of the escaped policies changed qualitatively:

**The features did what they were added to do — the dead-cat appetite is damped.** Phase 5's
escaped seeds bought BHIA3/MGLU3/LWSA3/HAPV3/AZUL4 (the crashing high-beta names). Phase 6's
strong escapes (2, 4, 6) buy PETR3+PETR4, PRIO3, GGBR4 — the *actual winners* of the
2021–2023 tightening cycle (oil/commodity complex). Seed 6 buying both Petrobras share
classes at once is a coherent-scoring sanity signal, not a coincidence. Junk residue survives
only in the weaker seeds (3, 5) and as dust in the cash seeds.

**The new tax is churn, not picking.** Seed 6's gross return ≈ 53% + 16.2pp cost drag ≈ ~69%
before costs at 51%/day turnover; seed 3 burned 12.9pp churning junk. Concentration is also
extreme (avg max single name 41–69% among escapees). The signal improved; the *trading* of
the signal is now the dominant loss source. (Relevant evidence for the deferred
rebalance-frequency question — NOT for a turnover penalty, which is off the table.)

**Bistability persists**: 3 of 8 seeds (1, 7, 8) still sit at 93–97% cash.

**Holdout selection rule: weaker on average, still right at the top.** corr(holdout, val)
dropped to 0.22 Pearson / 0.10 Spearman (was 0.54/0.52) — seed 5 is the outlier (holdout
+44% → val −15%; it also ran all 100k steps, best_step=99999, i.e. it kept climbing on the
250-day holdout tail and overfit it). But argmax still works: the best-holdout seed (6,
+59.6%) is also the best val seed (+53.2%) — same as Phase 5. The legal train-only selection
rule picked the winner twice in a row.

**More steps would not help.** 6 of 8 best_steps landed ≤ 45k of 100k; the only seed that
used the full budget (5) is the worst on val. Consistent with Phase 3 (2M steps → −71%): the
ceiling is representational/regime, not budget. Checkpoint-at-peak is already harvesting the
peak that exists.

**Verdict**: features moved the picking from anti-signal to signal (real 2022 winners), but
the edge is spent on 15–50%/day turnover and single-name concentration, and n=8 with CI
[−0.50, +3.39] on the best seed means no statistical claim of skill yet. Next levers (user
to reopen): rebalance frequency (churn evidence above), and the deferred PE/PB opt-in group.

---

## Phase 7: Batch-size + valuation-channel ablations & the online-phase drift diagnosis (2026-07-17)

Runs analyzed: `eiie_features` seeds 1–8 (batch 50), `eiie_features_b100` seeds 1–3,
`eiie_valuation_b50`/`_b100` seeds 1–3 (adds `pl_zhist_5y`/`pvp_zhist_5y` + isnan masks, 15 ch).
All series extracted from each run's `report.html` (PV, allocation evolution, turnover).

### Finding 1 — The "mimics random_rebalancing early" behavior is a near-uniform policy, not copying

Runs whose backtest starts diversified (allocation entropy 0.9–1.0, max single name 1–6%:
features seed 4, valuation seeds 1–2) show daily-return correlation 0.86–0.98 with
`random_rebalancing` — because a ~uniform 50-stock portfolio IS approximately the
equal-weight portfolio that baseline approximates. Not a bug, not imitation.

### Finding 2 — The later divergence is the ONLINE phase re-concentrating the policy (the real defect)

Allocation entropy by backtest thirds falls monotonically in most runs while cash (or a
single name) climbs:

| run | ent t1→t3 | cash t1→t3 |
|---|---|---|
| features s1 | 0.80 → 0.35 | 0.77 → 0.90 |
| features s7 | 0.63 → 0.22 | 0.82 → 0.95 |
| features_b100 s1 | 0.81 → 0.43 | 0.75 → 0.87 |
| valuation_b100 s1 | 0.99 → 0.69 | 0.05 → 0.77 |

The pretrain-side fixes (entropy anneal, checkpoint-at-peak) protect only pretrain. During
the live phase, `rolling_steps=30` updates/day run with entropy at its floor (1e-5) and NO
checkpoint gate — the cash attractor re-forms mid-backtest, exactly the failure mode Phase 5
diagnosed, now in the one phase nothing guards. The good checkpoint the holdout rule selected
is progressively destroyed in production.

### Finding 3 — The "varies A LOT / all-in hopping" archetype is the bistable regime + online hopping

Seeds whose pretrain lands concentrated (features s3, s6; valuation s3: start entropy
0.28–0.41) hop between >70% single-name positions during online updates (features s6: 28
switches, 51%/day turnover). Outcome is a coin flip on the same mechanism: features s6 +53%,
valuation s3 −50/−57% with 0.68–0.75 max drawdown.

### Finding 4 — Ablation results: both new levers are dead or negative

- **Batch 100 vs 50**: per-seed final PV differs by <0.02 (s1 1.186 vs 1.206, s2 1.308 vs
  1.291). Dead lever; drop it.
- **Valuation channels (pl/pvp zhist)**: no mean improvement; seed 3 became catastrophic in
  BOTH batch variants (−50%, −57%, maxDD up to 0.75 — worst results in the whole ensemble).
  The extra channels raised the concentrated regime's confidence without improving picking.
  PE/PB opt-in group: tested, negative. Close it.

### Finding 5 — Still no statistical edge, and the average seed loses to doing nothing

Every Sharpe CI spans zero (width ~±1.2). Mean agent total return (features, 8 seeds) ≈ +16%
vs `constant_cash` (pure CDI) +31% and UBAH +37% on the same window. Seed variance (PV
0.85–1.53 on identical config) still dwarfs every config delta tested.

### Recommended next actions (ranked by information-per-cost)

- [ ] **7a — Frozen-policy ablation (config-only): `rolling_steps: 0`** on the features
  config, seeds 1–8. Directly measures whether online training helps or hurts. Prediction
  from Finding 2: frozen preserves the good checkpoints (s1/s7 stay diversified) and avoids
  s3-style hopping.
- [x] ~~**7b — Excess-CDI training reward**~~ **REJECTED by user (2026-07-17): the reward
  function stays as the paper defines it.** Not a lever.
- [ ] **7c — If online training survives 7a: raise `entropy_beta_end`** (e.g. hold 1e-4
  through the live phase) so the floor that protects pretrain also protects production.
- [x] ~~**7d — Concentration cap**~~ **REJECTED by user (2026-07-17): no hard per-asset
  caps; the model must learn its own constraints.** Not a lever.
- [x] **PE/PB opt-in group: tested and closed** (Finding 4).
- [ ] **Stop adding feature channels** until the live-phase stability is fixed — the
  bottleneck is the objective/online drift, not inputs.

---

## Phase 8: Evidence-first sequencing — diagnostics before levers (planned 2026-07-18)

Context: an external recommendation list (incremental feature groups, cross-sectional
relative strength, multi-day holding horizons k, depthwise-separable conv1, two-stream
valuation bypass, behavioral/consistency/ranking diagnostics) was reviewed. Verdict: the
diagnostics are immediately valuable; every training-side lever is **blocked on Phase 7a**,
because Finding 5 stands — seed variance (PV 0.85–1.53 on identical config) swamps every
config delta tested. A/B-testing feature groups on top of that noise guarantees false
conclusions. Fix the noise source first, then measure, then pick ONE lever with evidence.

### Step 1 — Resolve 7a: frozen-policy ablation (config-only) — DONE (2026-07-18)

- [x] **Config**: `configs/eiie_features_frozen.json` (`eiie_features.json` copy,
  `rolling_steps: 0`, name `eiie_features_frozen`).
- [x] **Bug found + fixed mid-sweep**: seed 5's first run collided with seed 8's on
  `experiment.py`'s output directory (`datetime.now()` timestamp formatted to the identical
  microsecond string under concurrent `sweep.py -j 4` launches — the old "timestamps carry
  microseconds, so same-second launches can't collide" assumption didn't hold in practice).
  Seed 8's write silently clobbered seed 5's artifacts. Fixed by appending `os.getpid()` to
  the directory name (`experiment.py`, `sweep.py` docstring updated to match) — PIDs are
  unique across every simultaneously-running process regardless of clock resolution. Seed 5
  re-run cleanly afterward, no collision.
- [x] Ran seeds 1–8: `python -m src.rl_agent.sweep --config configs/eiie_features_frozen.json --seeds 1 2 3 4 5 6 7 8 -j 4` (+ seed 5 individually after the fix).

**Results — frozen vs. each seed's Phase 6 online counterpart** (same window; CDI +30.9%, BOVA11 +25.6%):

| seed | frozen return | online return | frozen Sharpe | online Sharpe | verdict |
|---:|---:|---:|---:|---:|---|
| 1 | +24.5% | +18.6% | −0.23 | −0.51 | frozen better |
| 2 | +23.8% | **+30.8%** | −0.05 | **+0.12** | online better |
| 3 | **−17.7%** | **+8.5%** | −0.37 | −0.06 | online better |
| 4 | +12.6% | +12.9% | −0.17 | −0.17 | wash |
| 5 | −13.8% | −14.9% | −0.50 | −0.50 | wash |
| 6 | **+98.4%** | +53.2% | **+0.62** | 0.36 | frozen better |
| 7 | +28.5% | +27.4% | −0.08 | −0.35 | frozen better |
| 8 | −7.2% | −6.2% | −0.49 | −0.50 | wash |

Tally: 3 favor frozen (1, 6, 7), 2 favor online (2, 3), 3 wash (4, 5, 8).

**Verdict: mixed, does NOT cleanly confirm Finding 2's hypothesis.** Seed 3 is a direct
counterexample — online training *rescued* a policy that collapses when frozen (−17.7% →
+8.5%). This mostly reproduces Finding 5 again: the frozen/online switch is itself just
another config delta whose effect is comparable to or smaller than seed-to-seed variance.
Seed 6's headline +98.4% also carries a red flag, not a clean win: mean cash ≈0%, allocation
entropy 0.08 (near-total concentration), 198 argmax switches (hopping between single-name
all-in bets roughly every 3 days) — the same "lottery ticket that happened to pay" pattern
Phase 5 already flagged, not evidence of a stable diversified strategy.
- Decision rule (as specified) doesn't resolve cleanly on PV/Sharpe alone → proceed to
  Step 2/3's diagnostics (8-D3 ranking quality especially) rather than picking 7c or dropping
  online updates on this evidence alone.

### Step 2 — Build the three diagnostics (offline, no retraining) — CODE COMPLETE

- [x] **8-D1 Behavioral metrics** (was rec 5) — implemented in `metrics.py`:
  `allocation_entropy`, `effective_n_holdings`, `mean_cash_weight`,
  `frac_days_cash_above`, `frac_days_single_name_above`, `argmax_switches`,
  `mean_position_lifetime`. Wired into `MetricsSummary`/`summarize()` (so every run's
  `metrics_summary.json` gets them, agent AND every baseline) and into `plots.py`'s
  `_METRIC_FIELDS` (report.html table). Tested: `tests/rl_agent/test_metrics.py`
  (`test_behavioral_metrics`, `test_behavioral_metrics_degenerate`).
- [x] **8-D2 Cross-seed consistency** (was rec 6) — `diagnostics.py`'s
  `cross_seed_consistency()`: mean pairwise cosine similarity + top-10 Jaccard overlap of
  daily weight vectors + correlation of per-run mean-weight vectors, aligned on the
  intersection of dates across runs. Tested: `test_diagnostics.py`.
- [x] **8-D3 Ranking quality** (was rec 7) — **the master key**. `diagnostics.py`'s
  `ranking_quality()`: per k∈{1,5,21} (configurable), Spearman(agent non-cash weights,
  realized forward k-day return) and top-decile-hit-rate-in-top-10, both overall and
  restricted to days held <50% cash (Finding 5: heavy-cash days otherwise drown the signal).
  New `forward_return(panel, t, k)` helper (no forward-return method existed on `PricePanel`
  before this). Tested: `test_diagnostics.py` (perfect/inverted/mismatched-tickers cases).
- [x] **On-disk artifact needed by D2/D3**: `experiment.py` now saves `weights.npz`
  (full `(T, n_global)` weight matrix + dates + tickers — `report.html` only ever kept a
  lossy cash+top-9+other aggregation) and `model_pretrain.pt` (the pretrain-selected
  checkpoint, saved BEFORE the online backtest mutates the model further — this is what
  lets `diagnostics.py --replay` reconstruct a frozen-policy backtest for **any** past run,
  not just future `eiie_features_frozen` ones).
- Old runs (predating this change) have no `weights.npz` — use
  `python -m src.rl_agent.diagnostics --runs <dirs> --replay` to reconstruct weights via an
  inference-only backtest against `model_pretrain.pt` (or `model.pt` if that's all a run
  has — note: for pre-Phase-8 runs `model.pt` is the POST-online, already-degraded policy,
  since nothing saved the pretrain checkpoint separately before now).

### Step 3 — Re-read the existing runs through the new lenses

- [x] Applied 8-D2/8-D3 to the 8 `eiie_features_frozen` runs from Step 1 (native
  `weights.npz`, no `--replay` needed): `python -m src.rl_agent.diagnostics --runs experiments/eiie_features_frozen_*`.
- [ ] Still open: apply the same to the other 17 Phase 6+7 runs (features s1–8,
  features_b100 s1–3, valuation_b50/b100 s1–3) via `--replay`. Lower priority now — the
  frozen-run result below is already decisive enough to pick Step 4's branch.

**8-D3 result (all 8 seeds, k=1/5/21): flat, ~0 everywhere.** Mean Spearman(agent
non-cash weights, realized forward return) ranges −0.011 to +0.007 across every
seed/horizon — no seed, no horizon, shows a real correlation in either direction.
Top-10-hit-rate on realized top-decile winners: 6–10%, matching the ~5.8% baseline
random chance out of 171 assets would give. This is not "ranks winners low" (a
consistent anti-signal would itself be informative) — it's indistinguishable from the
agent's weights having no relationship at all to what happens next, on any single day.

**8-D2 result: mean pairwise cosine 0.39, top-10 Jaccard 0.19, mean-weight correlation
0.56.** Seeds DO converge on a shared aggregate preference for certain tickers (matches
Phase 6's PETR3/4, PRIO3, GGBR4 recurrence) even though day-to-day top-10 picks diverge a
lot (Jaccard 0.19). So training reliably finds *something* stable across seeds — it just
isn't something that predicts returns.

**Verdict: REPRESENTATION PROBLEM, not policy/entropy/churn.** This also explains Step
1's mixed frozen-vs-online result — if the underlying signal is ~zero either way, whether
online updates run barely matters; both conditions are watching noise move around a
near-zero-skill policy. Proceed to Step 4's feature-work branch.

### Step 4 — Branch on evidence (pick ONE, justified by 8-D3) — DECIDED: feature work

- [ ] **Feature work, incrementally** (decided by 8-D3's flat result above): first the
  8-channel subset (close/high/low + return_1m/3m/6m + drawdown + price_vs_ma60), then the
  momentum/oscillator group (rsi_14, volatility_ratio_20_60, volume_ratio_20d), then ONE
  cross-sectional relative-strength channel (Stage 2's `cross_sectional.py` already computes
  market/sector-relative features — wiring, not new computation). One group per sweep, and
  re-run 8-D3 after each — the whole point is to watch whether Spearman/hit-rate actually
  moves off zero, not just whether backtest PV moves (Step 1 showed PV alone is too noisy
  to attribute).
- [x] ~~**Holding-horizon experiment**~~ **NOT indicated** — 8-D3 shows picking is flat, not
  churny-but-correct, so more time between rebalances has nothing to lock in.
- [x] ~~**Architecture changes**~~ **stay closed** — 8-D3's flat result doesn't distinguish
  "wrong architecture" from "no signal in these inputs at any architecture"; the cheaper,
  correctly-ordered next step is trying different/more inputs before rebuilding the encoder.
  Noted for the record: the proposed `groups=11` depthwise conv1 is invalid as stated
  (PyTorch requires out_channels divisible by groups; conv1_out=2) — it would be a full
  depthwise-separable block, i.e. a structural change. The two-stream valuation bypass stays
  in the drawer: valuation channels tested negative (Finding 4), so the open question is
  whether they belong at all, not how to encode them.

---

## Phase 9: Cross-sectional PE z-score test (2026-07-18, `configs/eiie_pe_sector.json`, seeds 1–8)

Added `pl_zscore_sector` + `pl_zscore_sector_isnan` (13 channels total) — the untested,
peer-relative valuation feature chosen in Step 4 over the already-closed per-ticker
`pl_zhist_5y`. Measured directly before wiring: well-bounded (p1/p99 ≈ [−2.7, 2.8], max
≈ ±5.6 across the built dataset), unlike `pl_zhist_5y`'s occasional ~18k blowups — no
`log1p` squash needed, `0.2` passthrough scale + isnan mask (39% NaN coverage measured).

| seed | return | sharpe | mean cash | entropy | eff_n | switches | maxdd | turnover |
|-----:|-------:|-------:|----------:|--------:|------:|---------:|------:|---------:|
| 1 | −36.3% | −0.86 | 0.82 | 0.04 | 1.16 | 0 | 0.501 | 0.050 |
| 2 | +1.1% | −0.33 | 0.85 | 0.06 | 1.22 | 0 | 0.271 | 0.018 |
| 3 | +1.3% | −0.29 | 0.89 | 0.02 | 1.10 | 0 | 0.327 | 0.033 |
| 4 | +28.8% | −0.09 | 0.82 | 0.20 | 1.80 | 0 | 0.066 | 0.018 |
| 5 | +33.4% | 0.15 | 0.00 | 0.59 | 18.97 | 0 | 0.202 | 0.053 |
| 6 | **+74.8%** | **0.51** | 0.00 | 0.17 | 2.14 | 21 | 0.432 | 0.366 |
| 7 | +25.7% | −0.03 | 0.88 | 0.08 | 1.28 | 0 | 0.149 | 0.016 |
| 8 | −12.0% | −0.03 | 0.02 | 0.31 | 4.27 | 17 | 0.452 | 0.460 |

(CDI +30.9%, BOVA11 +25.6% on this window.) Seed 6 again the standout (PV 1.748).

**8-D3 ranking quality: still flat, including for seed 6 itself.** Every seed, every
k∈{1,5,21}: mean Spearman in [−0.012, +0.006], hit rate 6–9% (≈ the ~5.8% random-chance
baseline). Seed 6's OWN numbers: Spearman +0.004/+0.006/+0.002, hit rate ~0.06 — no better
than the worst seed in the batch. The new feature did not move the needle for anyone,
including the seed that made the most money.

**Why seed 6 succeeded — checked directly against its `weights.npz`**: top holdings by
mean weight are PETR3 (24.1%), PETR4 (11.6%), PRIO3 (8.3%), GGBR4 (4.4%) — the *exact same*
oil/commodity-complex names seed 6 concentrated into in Phase 6 (original 11-channel
features) and the Phase 8 frozen ablation, regardless of which feature set it was trained
on. This is not the PE-sector feature working — it's seed 6's initialization consistently
locking onto the same handful of names independent of input channels, which happened to
ride a real, large 2021–2023 commodity rally in this exact backtest window. Combined with
its own flat 8-D3 numbers, this is the "lottery ticket that happened to pay" pattern
(Phase 5/7) confirmed a third time, not evidence of a discovered edge.

**Verdict: 4 configs in (baseline 3ch, features 11ch, frozen, pe_sector 13ch), 8-D3 has
never moved off zero.** Chasing seed 6's specific concentrated bet further would be
overfitting to one historical coincidence, not building a generalizable edge. The
untested `momentum_vs_market_1m/3m` cross-sectional channel (Step 4's other candidate,
not yet tried — PE-sector was tested instead) remains the cheapest still-open lever: it's
the one piece of information the "Identical Independent Evaluators" architecture
structurally cannot derive on its own (no cross-asset comparison inside the encoder), so
it's a different bet than another valuation/technical variant. Recommended before any
further feature attempts: decide up front what a null result on THIS channel means for
the program overall (four price/technical/valuation attempts already flat is a strong
prior against "one more feature" fixing it) rather than open-endedly iterating.

### Standing constraints (user decisions, do not reopen)

- Reward function = paper's log-return reward. No excess-CDI reshaping, no turnover penalty.
- No hard per-asset weight caps. Constraints must be learned, not imposed.
