# EIIE Agent Investigation — Overall Assessment

**Scope**: everything from the original cash-attractor bug through five feature
configurations and the diagnostics built to evaluate them. Companion to
`EIIE_DIAGNOSIS_PLAN.md` (the phase-by-phase working log with full data tables) — this
document is the synthesis: what was done, why, what it means, and what's actually left
to try.

**Bottom line up front**: the training-side bugs are fixed and verified. The agent no
longer collapses to frozen cash, and the tooling to measure real skill (not just backtest
luck) now exists and works. But across five different feature sets — including two
genuinely new information types added specifically to test this — the agent has **never
once shown measurable skill at ranking which stocks will actually outperform**. Every
apparent "win" traces to the same mechanism: a seed's random initialization concentrating
into a small handful of commodity-complex stocks that happened to rally hard in this
specific 2021–2023 window, not to anything the model learned about the data.

---

## 1. Where this started: the cash attractor

Early runs converged to holding 100% cash and staying there — the softmax over asset
scores saturated (`~1e-9` per asset), gradients vanished (`4.6e-11`), and more training
made it *worse*, not better (100k steps: −15.3% vs CDI; 2M steps: frozen all-cash).

**Root cause** (measured, not theorized): the paper's cash asset returns 0%, so cash can
never dominate its softmax. This project's cash accrues CDI (~8.65%/yr in log-space,
*above* equal-weight's ~8.22%), so the training gradient had a one-directional pull —
every asset's score got nudged down, with nothing pushing back — until scores ran to
−20/−32, the softmax saturated, and the network froze holding cash forever.

**Fixes** (training-side only, reward/costs/data untouched):
- **Entropy bonus** (`entropy_beta_start/end`, annealed over the first 10% of pretrain,
  flat after) — a restoring force against saturation, scale-matched so it nudges the
  softmax without dictating the allocation (1e-3 would just maximize entropy into UCRP).
- **Checkpoint-at-peak** (`checkpoint_holdout_days`/`checkpoint_eval_every`) — periodically
  scores the frozen policy on a held-out tail of the *train* split and keeps the
  best-scoring checkpoint, instead of trusting wherever a fixed step budget lands. Measured
  case that motivated this: the same seed went from +47% (100k steps) to −71% (2M steps)
  on an identical config — budget alone overfits the policy past its peak.

These fixes work as designed: seeds now escape the cash attractor reliably, and the
holdout-selection rule has picked the best-performing seed correctly in every batch tested
so far. What they don't do — and were never going to do — is guarantee the escaped
policy is actually *good* at picking stocks. That question needed different tools.

---

## 2. Why backtest return alone couldn't answer "is it actually working"

By Phase 7, seed-ensemble sweeps showed a pattern that made this obvious: **seed variance
dwarfs every config change tested.** Identical configs produced final portfolio values
ranging 0.85–1.53×; every Sharpe ratio's bootstrap CI spanned zero. A single seed's
backtest return is mostly noise — comparing two configs by their return alone (as had
been done through Phase 7) risks reading noise as signal in either direction.

This motivated **Phase 8**: build diagnostics that don't depend on one noisy backtest
number, then re-evaluate everything through them.

### What was built

- **Behavioral metrics** (`metrics.py`): allocation entropy, effective number of holdings,
  cash fraction, single-name concentration, position-switch count, mean position
  lifetime — now computed automatically for every run (agent and every baseline) and
  written into `metrics_summary.json`/`report.html`.
- **`weights.npz`**: the full per-day weight matrix for every asset, persisted for the
  first time (`report.html` previously only kept a lossy cash+top-9+"other" aggregation).
  Needed as the raw material for everything below.
- **`model_pretrain.pt`**: the pretrain-selected checkpoint, saved *before* the online
  backtest can mutate it further — lets any past run's frozen-policy behavior be
  reconstructed after the fact.
- **Cross-seed consistency (8-D2)**: do independently-seeded runs on the same config
  converge on a similar allocation (a stable learned signal), or diverge (seeds exploiting
  noise)? Measures pairwise cosine similarity, top-10 overlap, and mean-weight correlation
  across seeds.
- **Ranking quality (8-D3) — the decisive one**: for each backtest day, does the agent's
  portfolio weight actually correlate with which stocks go on to outperform over the next
  1/5/21 days? This is the one metric that distinguishes *real* stock-picking skill from
  a policy that merely trades a lot, or happens to sit in the right names by chance.

### The frozen-policy ablation (Phase 8, Step 1) — and a real bug it surfaced

Before trusting any of this, one open question from Phase 7 needed resolving: is the
*online* training phase (30 gradient updates per backtest day, run at the entropy floor
with no checkpoint protection) helping or hurting? Tested directly: same 8 seeds, same
everything, `rolling_steps: 0` (no online updates) vs. the original online runs.

**Result: genuinely mixed.** 3 seeds improved when frozen, 2 got *worse* (seed 3 collapsed
from +8.5% online to −17.7% frozen — online training rescued it), 3 were a wash. This
doesn't cleanly indict the online phase; it mostly re-confirms that seed variance is the
dominant term, even in this specific ablation.

**Bug found and fixed along the way**: seed 5's run silently lost its output when two
concurrent sweep processes' microsecond-precision timestamps collided on the exact same
output directory string — the "timestamps carry microseconds, so same-second launches
can't collide" assumption baked into `experiment.py`/`sweep.py` didn't hold in practice.
Fixed by appending the process ID (unique across concurrent processes regardless of clock
resolution) to the output directory name. Verified clean across two subsequent 8-seed
sweeps with zero collisions.

---

## 3. Three feature experiments, evaluated through the new diagnostics

With the tooling in place, three different feature additions were tested — all isolated
against the same 11-channel baseline (`close/high/low` + `return_1m/3m/6m` +
`price_vs_ma60` + `volatility_ratio_20_60` + `rsi_14` + `drawdown` + `volume_ratio_20d`),
same 8 seeds, same window, same costs.

| Config | New channels | Mean return (median) | Best seed | 8-D3 mean Spearman (k=1/5/21) |
|---|---|---:|---:|---|
| Frozen ablation | *(none — rolling_steps=0)* | 18.6% (23.8%) | seed 6: +98.4% | ~0 (−0.01 to +0.01), all 8 seeds |
| **PE-sector** | `pl_zscore_sector` + isnan mask | 14.6% (25.7%) | seed 6: +74.8% | ~0 (−0.012 to +0.006), all 8 seeds |
| **Momentum-vs-market** | `momentum_vs_market_1m/3m` | 22.2% (5.2%) | seed 5: +179.1% | ~0 (−0.011 to +0.007), all 8 seeds |

(CDI +30.9%, BOVA11 +25.6% on this window, all three rows.)

**Why these two features specifically**: `pl_zscore_sector` (peer-relative PE, computed
in `cross_sectional.py`) was chosen over the *already-tested-and-failed*
`pl_zhist_5y` (own-history PE z-score, Phase 7: seed 3 went to −57% in both batch
variants) because it's a genuinely different question — "cheap vs. peers today," not
"cheap vs. its own history." `momentum_vs_market_1m/3m` was chosen because it's
categorically different from every other channel tried: the network's "Identical
Independent Evaluators" architecture means every asset's convolution stream only ever
sees *that asset's own* 50-day window — it has no path to compute "how am I doing
relative to everyone else today." That comparison is structurally invisible to the model
unless it's handed the answer directly, which this channel does.

**Both came back the same way: no ranking-quality signal, on any seed, at any horizon.**
Top-10 hit rate on realized top-decile winners sat at 6–10% in every config — statistically
indistinguishable from picking at random out of 171 assets (~5.8% expected by chance).

---

## 4. The recurring mechanism behind every apparent "win"

The best-performing seed in each config was individually inspected via its `weights.npz`
holdings, not just its return number:

- **Frozen ablation, seed 6** (+98.4%): PETR3 (24%), PETR4 (12%), PRIO3 (8%), GGBR4 (4%).
- **PE-sector, seed 6** (+74.8%): PETR3 (24.1%), PETR4 (11.6%), PRIO3 (8.3%), GGBR4 (4.4%)
  — the *identical* holdings, unchanged by the new feature.
- **Momentum-vs-market, seed 5** (+179.1%): PETR4 (20.3%), PETR3 (20.1%), PRIO3 (5.8%),
  GGBR4 (5.6%) — same complex, but seed 5 this time, not seed 6.
- In the same run, **seed 6 itself collapsed** to 81% cash + a small BHIA3/MGLU3 position
  (−17.8%) — the *old* dead-cat-bounce failure mode from before technical features were
  even added.

The pattern: whichever seed's random initialization happens to concentrate into the
Petrobras/oil-commodity complex wins big, because that complex had a real, large rally in
this exact 2021–2023 backtest window. Which seed that is **is not stable** — it moved from
6 to 5 the moment two new channels changed the training dynamics, and the seed that "found
it" before collapsed to the opposite failure mode instead. Combined with every one of
these winning runs showing *zero* ranking-quality signal in its own right, this reads as
initialization-driven concentration intersecting a real historical rally by chance, not a
discovered, transferable edge. It also mechanically explains why frozen vs. online, and
feature set A vs. B, all look "mixed" rather than cleanly decisive: you're mostly measuring
which arbitrary seed the coin flip landed on, not the thing being tested.

---

## 5. Overall assessment

**What's solid:**
- The cash-attractor bug is genuinely fixed — verified via direct gradient/softmax
  inspection, not inferred from returns.
- Checkpoint-at-peak reliably picks the best-performing seed in a batch when one exists
  (confirmed correlation between held-out train-tail score and true val performance in
  multiple batches).
- The diagnostic tooling (behavioral metrics, cross-seed consistency, ranking quality)
  works, is tested, and has now been exercised across three real feature configs with
  consistent, interpretable results.
- The sweep infrastructure is robust (PID-based output naming verified collision-free
  under real concurrent load after the fix).

**What's not solid:**
- No configuration tested — 3-channel baseline, 11-channel technicals, frozen policy,
  peer-relative valuation, or cross-sectional momentum — has produced a policy with
  measurable forward-return ranking skill. Five consecutive null results on the one
  metric built specifically to detect this.
- Every seed-ensemble result has larger variance from seed choice alone than from any
  feature or training change tested. Bootstrap CIs on Sharpe consistently span zero.
- The reliable "big win" pattern is explainable by initialization noise intersecting a
  real market regime (2021–2023 commodities), not by anything resembling learned skill.

**What this doesn't prove**: it doesn't prove no signal exists in principle, or that
this specific market/window has no exploitable structure at all. It's evidence against
"the missing piece is one more price/technical/valuation feature" and against "the online
training phase is the main problem" — both of which had reasonable a priori cases before
this round of testing.

---

## 6. Options going forward

Ranked by expected information per unit of effort, given the standing constraints (paper
log-return reward, no reward reshaping, no hard per-asset caps — the model must learn
its own constraints):

0. **Overfit sanity check — can the model memorize at all?** Train on a tiny subset (a
   few months, or even a few dozen windows) with no holdout selection and ask: can it
   drive training reward to the hindsight-optimal allocation — perfectly "predicting" the
   data it was trained on? This is the classic diagnostic that splits the failure space
   in two: if it *can't* memorize even a tiny sample, the optimizer/architecture/capacity
   is limiting and option 1 becomes near-certain to matter; if it *can*, the encoder and
   optimization are fine and the problem is signal/generalization, which deflates option 1
   and strengthens options 3/5. Cheapest test on this list (one config, one seed, minutes
   not hours) and it changes how every subsequent result is read — run it first.

1. **Increase model capacity.** `conv1_out_channels=2`, `conv2_out_channels=20` is a
   genuinely tiny network (~3k parameters). Five null results with a very thin encoder
   don't yet distinguish "no signal in these inputs" from "not enough capacity to extract
   a real but subtle signal." This is the natural next hypothesis specifically because
   it's the one variable not yet touched — every prior test changed *inputs*, none changed
   *capacity*. The constraint is also *structural*, not just a parameter count: the input
   grew from 3 to 11 (and 15) channels while conv1 still compresses everything into 2
   feature maps — the network physically lacks the bandwidth to carry 11 distinct
   time-series modalities forward. Concrete fix: bump `conv1_out_channels` to 16–32 and
   `conv2_out_channels` to 32–64 (~12k parameters — still small, but enough to
   definitively test the "choking on input width" hypothesis). Expectation-managed:
   five nulls across different feature *types* suggest the bottleneck may not just be
   channel count, so treat this as closing off a hypothesis, not as the likely fix.
   Cost: moderate (config-only change, needs its own sweep + 8-D3 re-check; slower
   training per step).

2. **Test the remaining cross-sectional variants cheaply before concluding.** `beta_1y`
   and the sector-restricted momentum (`momentum_vs_sector_1m/3m`) already exist in the
   built dataset and are wiring-only additions (same pattern as the two already done).
   Low cost, but low expected value given the pattern — included for completeness, not as
   a strong recommendation.

3. **Reconsider the premise at the data/objective level.** Five flat results across
   fundamentally different feature types is a real pattern, not noise. If capacity (option
   1) also comes back flat, the more honest fork is: does daily-frequency single-stock
   alpha exist at all in this 50-stock B3 universe from public price/fundamental data? If
   not, alternatives include a lower-frequency signal (weekly/monthly, distinct from the
   already-tested "reduce churn" idea — this is about signal horizon, not turnover cost),
   macro/regime-conditioning features (rate environment, real return — already computed in
   Stage 2, never wired into the agent), or redefining the objective away from
   stock-picking alpha entirely (e.g. a risk/diversification-focused mandate where the
   agent doesn't need genuine picking skill to be useful, given CDI/UBAH/UCRP already
   provide a strong, cheap floor on this window).

4. **Improve statistical power before drawing a final conclusion.** 8 seeds and one
   576-day window is not a lot of statistical power to detect a small true edge against
   this much seed-to-seed noise. More seeds (16–32) on the current best-diagnosed config,
   or testing across multiple non-overlapping historical windows, would sharpen confidence
   in "there is truly no signal here" before that becomes the final word. **Deprioritized**:
   if the true edge is so small it needs 32 seeds or multiple windows to detect over the
   noise floor, it's swamped by initialization noise in practice and arguably not an edge
   worth trading — revisit only if options 1/5 surface something borderline.

5. **Test a different architecture.** EIIE's "Identical Independent Evaluators" design
   means each asset's stream only ever sees its own 50-day window — cross-asset
   relationships are structurally invisible unless hand-fed as features (which is exactly
   what `momentum_vs_market` and `pl_zscore_sector` did, and neither helped). Everything
   so far has therefore tested "can EIIE extract signal from hand-engineered
   cross-sectional features," not "can a model discover cross-asset structure from raw
   data." A cross-asset architecture (attention/transformer over assets, or a GNN) removes
   that constraint. Sits between option 1 (capacity) and option 3 (redefining the problem)
   in effort — a real architecture change, not a config knob, so it's the step *after*
   option 1 comes back flat, not alongside it.

**Recommendation**: run option 0 (overfit check) first — it's minutes of compute and
determines how to read everything after it. Then option 1 (capacity) is the cheapest way
to close off the remaining "maybe the encoder is just too small" explanation, since
nothing else has tested it yet — with the concrete widening above (conv1 16–32, conv2
32–64), not a marginal bump. If that
also returns a flat 8-D3, the fork is option 5 (cross-asset architecture) vs. option 3
(reconsider the premise) — and at that point the honest prior is that daily-frequency
single-stock alpha in this universe may not exist in a form EIIE-style models can extract
from public data.
