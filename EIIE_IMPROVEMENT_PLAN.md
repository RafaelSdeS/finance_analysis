# EIIE Improvement Plan — Hypothesis-Driven Experiment Sequence

**Goal**: determine what is limiting the EIIE agent's performance by changing **one major
variable per experiment**, ordered by expected information gain per unit of effort.
Companion to `EIIE_INVESTIGATION_ASSESSMENT.md` (§6 options — this plan operationalizes
them) and `EIIE_DIAGNOSIS_PLAN.md` (the phase-by-phase log so far).

**Standing constraints** (unchanged throughout): paper log-return reward, no reward
reshaping, no hard per-asset caps, same 2021–2023 val window, same cost model.

---

## Standard evaluation protocol (referenced by every experiment as "SEP")

Unless an experiment says otherwise:

- **Sweep**: 8 seeds (1–8) via `python -m src.rl_agent.sweep -j 4`, same window, same
  costs, same base config as the 11-channel technical baseline.
- **Diagnostics** (all already implemented, auto-written per run):
  - **8-D3 ranking quality** — the decisive metric: mean daily Spearman between portfolio
    weight and forward returns at k=1/5/21, plus top-10 hit rate on realized top-decile
    winners (chance ≈ 5.8%).
  - **8-D2 cross-seed consistency** — pairwise cosine similarity, top-10 overlap,
    mean-weight correlation across seeds.
  - **Behavioral metrics** — allocation entropy, effective holdings, cash fraction,
    concentration, position lifetime.
  - Portfolio value vs. CDI/BOVA11/UBAH/UCRP + bootstrap CIs (context only — never the
    primary verdict, per Phase 7's seed-variance finding).
- **Signal threshold**: observed 8-D3 noise band so far is roughly −0.012 to +0.007.
  Call it **signal** only if mean Spearman ≥ **+0.02** at any horizon with the same sign
  in ≥ 6/8 seeds, or top-10 hit rate ≥ **12%** (≈2× chance) in ≥ 6/8 seeds. Anything
  inside the noise band is a **null**, regardless of backtest return.
- **Replication rule (multiple-comparisons guard)**: "any of 3 horizons" tested across
  6+ experiments means a marginal threshold crossing will eventually occur by chance.
  A crossing therefore only *counts* after replicating — same config, one fresh 8-seed
  sweep (seeds 9–16), same horizon, same sign. Until replicated it's "candidate
  signal," and nothing downstream re-baselines on it.
- **Attribution rule**: exactly one config field group changes vs. the immediately
  preceding accepted baseline config; the diff is recorded in the experiment's row below.

---

## High-level roadmap

| Stage | Experiment | Variable changed | Cost | Gate |
|---|---|---|---|---|
| 0 | E0 overfit check | training data size (tiny) | minutes, 1 seed | always run first |
| 0 | E0b optimizer check | learning rate | minutes ×3, 1 seed | only if E0 fails |
| 1 | E1 capacity | conv channel widths | 1 sweep | E0 result conditions reading |
| 1 | E1b capacity dose | widths, one step larger | 1 sweep | only if E1 moves the needle |
| 2 | E2a/E2b remaining features | one feature channel each | 1 sweep each | optional, completeness |
| 3 | E3 signal horizon | decision/reward frequency | code + 1 sweep | only after 0–1 conclude |
| 4 | E4 cross-asset architecture | network topology | large | only after E3 or by decision |

Stop rule: the first experiment that produces a real 8-D3 signal becomes the new
baseline; everything after it re-runs against that baseline. If Stages 0–2 all return
nulls, the Stage 3/4 conversation happens *before* spending on them (assessment §6
option 3 — reconsider the premise).

---

## Stage 0 — Can the model learn *anything*? (cheapest, most informative)

### E0 — Overfit sanity check

- [x] **RESULT (2026-07-18, seed 42, window 2022-01-03..2022-06-30, 124 days): FAIL, clean.**
  In-sample k=1 Spearman = **0.0013** (pass bar 0.3), top-10 hit rate 0.068 (chance
  0.058), argmax switches 65/123 day-pairs (day-varying, not static), mean cash
  fraction 0.000, mean top-1 weight 1.000 (fully one-hot). Train-window return
  +432.93% vs. best-stock buy-and-hold +44.56% (expected under the trap noted below,
  not evidence of memorization). **Saturation probe** (added mid-investigation, logs
  softmax entropy on a fixed reference day every 500 steps): entropy declined smoothly
  the ENTIRE 100k-step budget (3.93 → 0.70 @10% → 0.049 @50% → 0.0002 @final), loss
  improved 17× throughout — `SATURATED EARLY: False`, ruling out premature
  gradient-death as the explanation. **Holdings inspection** (`weights.npz`): PETR4
  (41d) + PETR3 (29d) + GGBR4 (16d) + PRIO3 (5d) = the identical oil/commodity complex
  documented in every "big win" seed across the whole prior investigation (real Jan–Jun
  2022 Petrobras rally, Ukraine-invasion oil shock). **Conclusion**: the network
  converges smoothly to completion and finds a real, non-degenerate, day-varying
  policy — but that policy is the same momentum/commodity-complex attractor already
  seen everywhere else, not day-ahead winner identification, even on data trained on
  directly. Three independent signals (ranking quality, saturation trajectory,
  holdings identity) agree, so **E0b was skipped** (see below) and Stage 1 proceeds
  with a strengthened, not just neutral, prior that capacity/architecture — not the
  optimizer — is the limiting factor.
- **Hypothesis**: the current architecture + optimizer can memorize a tiny training
  subset — i.e. drive its own training-window allocation toward the hindsight-optimal
  (day-by-day best-asset) policy.
- **Why**: five consecutive 8-D3 nulls cannot yet distinguish "no learnable signal in
  the inputs" from "the model/optimizer can't fit anything at all." This single check
  splits the entire failure space in two and conditions how every later result is read.
- **Implementation**:
  - New config `configs/eiie_overfit_check.json`: train window shortened to ~6 months
    (~126 days), `checkpoint_holdout_days: 0` (no holdout carve-out — we *want*
    overfitting), `rolling_steps: 0`, generous `pretrain_steps`, 1 seed, **commission
    rate 0** and **entropy 0** — with both off, the objective is exactly per-day
    log-return, whose optimum is exactly "all-in the next day's best asset," giving a
    crisp memorization target instead of a cost-blurred one. This is a diagnostic
    config, never a candidate model (same status as `sanity.py`'s dominant-asset toy
    market), so suspending the cost constraint is legitimate here and only here.
  - Evaluate the frozen post-pretrain policy **on the same tiny train window**. If
    `--eval-split` can't point at the train subset, add the minimal plumbing for it —
    nothing more.
- **Evaluation protocol**: single seed, no sweep. Primary metric: **8-D3 ranking
  quality computed on the training days themselves** (in-sample weight vs. next-day
  return Spearman) — the existing machinery, pointed at seen data. Secondary:
  train-subset log-return vs. UCRP / Best-Stock / the day-by-day hindsight-optimal
  line; `weights.npz` inspection.
- **Success**: in-sample k=1 Spearman strongly positive (≥ **0.3**) with day-*varying*
  concentrated weights, and return well above buy-and-hold Best-Stock (day-by-day
  switching dominates any single stock when costs are off). **Failure**: in-sample
  Spearman near 0, or returns explained by a static concentrated position.
  **Note the trap**: matching Best-Stock's *return* alone is NOT success — parking
  100% in the hindsight-best stock achieves that with zero memorization, which is
  exactly the static-concentration luck mode already documented. The in-sample
  Spearman is the criterion precisely because static concentration can't fake it.
- **Interpretation**: success → representation/optimization is *not* the bottleneck;
  E1 (capacity) is deflated, Stages 3–4 strengthened. Failure → optimizer/architecture/
  capacity is limiting; run E0b, then E1 with high prior.
- **Confounds**: zero-cost + zero-entropy are two config deltas from the baseline, but
  both are part of defining the memorization objective itself, not independent
  variables — nothing from E0 transfers as a "finding" about costs or entropy. PVM
  feedback (`w_{t-1}` as input) makes samples path-dependent; irrelevant to the
  pass/fail read but expect noisier early training.
  **Two confounds actually hit and fixed during execution**: (1) the first attempt used
  a 2015 H1 window; 24 of 171 union tickers (later IPOs, e.g. ASAI3/BPAC11/RDOR3) had
  zero price rows before that `window_end`, leaving their global-space price column
  entirely NaN — `0 * NaN = NaN` in `drift_weights`/`drift_weights_torch`'s global sum
  then poisoned every column regardless of that ticker's (always-zero) weight, crashing
  training with NaN loss from step 0. Fixed at two levels: the diagnostic's window moved
  to 2022 H1 (past every union ticker's IPO, last one 2021-07-14), AND `environment.py`'s
  `drift_weights`/`drift_weights_torch` now `nan_to_num` `y_t` before the multiply —
  behavior-preserving in every production config (`window_end` is always ~today, so no
  column is ever fully NaN) but closes the landmine for any future truncated-window
  experiment; regression test `test_drift_weights_nan_column_safe` in
  `tests/rl_agent/test_environment.py`. (2) The first entropy=0 run collapsed to 100%
  cash within the first few steps — CDI-accruing cash + zero entropy reproduces the
  exact cash-attractor bug entropy_beta exists to prevent, confounding the memorization
  read entirely (never got to attempt it). `DataConfig.cash_mode` does NOT fix this —
  confirmed it only selects the risk-free baseline for Sharpe/Sortino reporting in
  `experiment.py`, it never changes what `price_relative()` actually pays cash. Fixed by
  overriding `panel.cdi_factor = np.ones_like(...)` (zero-return cash, paper-faithful)
  directly in `scripts/e0_overfit_check.py` — script-local, no production math changed.
  Pre-flight artifacts produced alongside this: `scripts/sanity_check_pipeline.py`
  (forward/backward pass sanity on real data) and `tests/rl_agent/test_ffill_guard.py`
  (confirms technical channels are ffill-only, never bfilled, through the real loader).

### E0b — Optimizer check (conditional: only if E0 fails)

- [x] **SKIPPED (2026-07-18)**: E0 failed, which pre-authorized E0b, but the saturation
  probe (added specifically to check this) directly showed the optimizer was never
  stuck — entropy declined smoothly across the entire 100k-step budget, no early
  plateau. E0b's whole purpose is separating "wrong step size" from "not enough
  capacity"; that question is already answered by direct measurement, so running it
  would spend compute for near-zero expected new information. Decision made with the
  user rather than auto-run despite the literal pre-authorization, given evidence
  specifically undercut its premise. Proceeding to Stage 1 with a strengthened prior.
- **Hypothesis**: E0's failure is a learning-rate/optimization problem, not capacity.
- **Why**: before paying for a capacity sweep, a 3-point LR sweep on the same tiny
  subset costs minutes and separates "wrong step size" from "not enough parameters."
- **Implementation**: E0 config, `learning_rate` × {0.1×, 1×, 10×} of current. Nothing
  else changes.
- **Evaluation**: same as E0, three runs.
- **Success/failure**: any LR memorizes → optimizer was the limit. None memorize →
  capacity/architecture is the prime suspect; proceed to E1 with high prior.
- **If an LR fixes it, the five prior nulls are void**: they were all obtained under a
  mis-tuned optimizer and stop being evidence about features. Consequence: re-run the
  11-channel baseline sweep (full SEP) at the new LR *before* E1 — that re-run becomes
  the new accepted baseline, and the old null pattern must be re-established (or
  overturned) against it.
- **Confounds**: entropy schedule interacts with LR scale (both push logits); keep
  entropy at 0 as in E0 so LR is the only live variable.

---

## Stage 1 — Capacity

### E1 — Widen the encoder

- [x] **RESULT (2026-07-18, 8-seed sweep, `configs/eiie_capacity_e1.json`, seeds 1-8,
  val split, 576 days)**: **FAIL, unanimous 8/8.**
  - k=1 mean Spearman (primary metric, all days): {0.0020, 0.0026, 0.0022, -0.0010,
    0.0020, 0.0015, 0.0003, -0.0003} across the 8 seeds — every value inside the noise
    band, none within an order of magnitude of the 0.02 pass bar. 0/8 same-sign-pass.
  - top-10 hit rate (all days): {0.071, 0.066, 0.075, 0.088, 0.077, 0.076, 0.087, 0.083}
    — all well under the 0.12 bar. 0/8 pass.
  - **Signal threshold not crossed by either criterion, in any seed.** Clean, unambiguous
    null — not a marginal miss.
  - **Confound checks (both required before trusting this) came back clean**:
    (1) *checkpoint pinned at budget end?* No — 6/8 seeds picked an early best_step
    (4999-14999 of 100000); only 2 picked a late one (89999, 99999). Consistent with the
    documented early-peak-then-overfit pattern, not budget starvation — more
    `pretrain_steps` would not likely change this. (2) *entropy-beta scale mismatch
    (tiny-net beta on an 8x-wider conv1)?* Real, but didn't produce a new confounding
    failure mode — `mean_cash_weight` ranges 0.10-0.78 and `frac_days_single_name_gt70`
    ranges 0.08-0.80 across seeds (same cash-attractor / single-name-concentration
    regime as every prior sweep, just re-landing on different seeds' corners), not a
    qualitatively new behavior. Since the *symptom* is unchanged, the failure is not
    plausibly an artifact of the scale mismatch.
  - **Luck check**: only one seed (1839151) posted a positive return (+10.48%). Its
    top holdings by days-as-argmax: PETR3 (110d), PETR4 (93d), BHIA3 (64d), MGLU3 (31d),
    GGBR4 (21d), PRIO3 (20d) — the same commodity-complex pattern documented since the
    original investigation, not a new signal. Confirms the one apparent "win" is the
    established luck pattern, not capacity paying off.
  - **Verdict**: capacity (channel width) is not the bottleneck. Per the plan's own
    interpretation rule (E0 passed → failure here is strong evidence the bottleneck is
    signal/architecture, not size): **E1b (dose-response) is skipped** — its precondition
    ("if E1 shows movement") is not met. Proceed to Stage 2/4 per the plan's branch
    logic; capacity-scaling line of inquiry is closed.
- [ ] **Hypothesis**: `conv1_out_channels=2` is a structural bottleneck — 11 input
  channels compressed into 2 feature maps cannot carry distinct modalities forward —
  and widening to 16/64 lets a real but subtle signal through (8-D3 leaves the noise
  band).
- **Why**: the one variable never touched (every prior test changed inputs, not
  capacity), and the assessment's option 1. Expectation-managed: five nulls across
  feature *types* suggest this may close a hypothesis rather than fix the agent.
- **Implementation**: copy the 11-channel baseline config; change exactly
  `conv1_out_channels: 2 → 16` and `conv2_out_channels: 20 → 64` (~12k params). No
  other field changes.
- **Evaluation protocol**: full SEP (8 seeds, all diagnostics).
- **Success**: 8-D3 crosses the signal threshold. **Failure**: 8-D3 stays in the noise
  band — regardless of whether some seed posts a big return (Petrobras-complex luck is
  the established alternative explanation; check `weights.npz` holdings of any big
  winner against the PETR3/PETR4/PRIO3/GGBR4 pattern before crediting the change).
- **Interpretation**: with E0-success context, failure here is strong evidence the
  bottleneck is signal/architecture, not size → Stage 3/4 conversation. With E0-failure
  context, success here confirms capacity was the limit; failure points at architecture
  (E4) or optimization beyond LR.
- **Confounds**: (1) larger net may need more `pretrain_steps` to peak — checkpoint-at-
  peak already guards against over/under-training, but verify `best_step` isn't pinned
  at the end of the budget (if it is, the budget was too small; extend and re-run before
  concluding). (2) **Entropy beta was scale-matched to the tiny net's logit scale**
  ("nudges without dictating") — an 8× wider conv1 changes logit magnitudes, so the
  same beta may now be too weak (cash attractor returns) or too strong (forces UCRP).
  Before reading any E1 result, check cash fraction + allocation entropy + the existing
  gradient/softmax inspection on 1–2 seeds; if either failure mode appears, retune beta
  to restore the *same qualitative regime* first — that's calibration of the diagnostic
  environment, not a second experimental variable, but record the new value. (3) Slower
  per-step training and larger `_PanelStore` footprint — check GPU memory before
  `-j 4`. (4) cuDNN cross-process nondeterminism (known, documented) — compare
  distributions across seeds, never single runs; seed *numbers* are not comparable
  across architectures (different init dimensionality → different draws), so "seed 6"
  here has no relation to "seed 6" in prior sweeps.

### E1b — Capacity dose-response (conditional: only if E1 shows movement)

- [x] **SKIPPED (2026-07-18)**: precondition not met — E1 showed no movement (8/8 seeds
  flat null on both Spearman and top-10 hit). No dose-response to test.
- [ ] **Hypothesis**: if E1 moved 8-D3, more width moves it further (a dose-response
  confirms the mechanism is capacity, not a fluke).
- **Implementation**: 16/64 → 32/128, nothing else. Full SEP.
- **Interpretation**: monotone improvement → capacity story confirmed, tune from there.
  E1 gain not reproduced or reversed → treat E1's movement as noise; back to nulls.

---

## Stage 2 — Remaining cheap feature channels (completeness, low expectation)

Wiring-only additions (same pattern as `pl_zscore_sector` / `momentum_vs_market`,
already exercised twice). Run **one at a time** on whatever config Stage 1 leaves as
the accepted baseline. Low expected value given five nulls — these exist to make
"we tested every cheap input" literally true before any premise-level conclusion.

### E2a — `beta_1y`
- [ ] **Hypothesis**: market-beta exposure information improves ranking quality.
- **Implementation**: add `beta_1y` to `DataConfig.features` (+ `FEATURE_NORM` entry if
  missing). One channel, full SEP. **Lookahead guard**: both E2 channels are technical-
  kind and must go through the documented ffill-only path in `load_price_panel` — their
  own warm-up NaN prefix must NOT be bfilled (bfilling leaks a later value into an
  unmasked training row; this is the exact trap already documented for `return_6m`).
  Verify the new columns route as `ffill`-only, same as every existing technical channel.
- **Success/failure**: signal threshold, as always.

### E2b — `momentum_vs_sector_1m/3m`
- [ ] **Hypothesis**: sector-relative momentum carries signal that market-relative
  momentum (already tested, null) does not.
- **Implementation**: as E2a. Run *after* E2a resolves, not combined with it.
- **Confounds (both)**: correlated with existing momentum channels — a null is
  uninformative about the feature in isolation, only about its marginal value here.

---

## Stage 3 — Signal horizon (first premise-level change)

### E3 — Weekly decision frequency

- [ ] **Hypothesis**: daily-frequency single-stock alpha doesn't exist in this universe,
  but a weekly-horizon signal does — the nulls reflect the *timescale*, not the inputs.
- **Why**: assessment §6 option 3's most testable sub-claim. Distinct from the
  already-tested "reduce churn" idea: this changes the *reward horizon*, not just
  turnover cost.
- **Implementation**: rebalance every 5 trading days — `run_backtest` steps weekly,
  price relatives compound over the 5-day gap, PVM stores weekly weights. This is a
  real change to `environment.py`/`train.py` semantics, not a config knob — scope it
  properly before starting; keep daily paths intact behind a config field
  (`decision_period_days: 1|5`) so the change is one variable.
- **Evaluation**: full SEP; 8-D3 horizons shift to k=1/4/12 *weeks*. Note the sample
  count drops ~5× (≈115 decisions per backtest) — CIs widen; judge by ranking quality
  consistency across seeds, not returns.
- **Success/failure**: **the daily ±0.02 threshold does not transfer** — it was
  calibrated on ~576 decisions; at ~115 the null distribution of mean Spearman is
  wider, and reusing the daily bar would over-call signal. Recalibrate first: build the
  weekly noise band empirically (shuffle weights across days within each run, recompute
  mean Spearman ~1000×), then set the threshold at the 97.5th percentile of that
  permutation null. Consistency rule (same sign in ≥ 6/8 seeds) carries over unchanged.
- **Prior check (free, do before writing any code)**: the daily runs' k=21 8-D3 was
  already null in all five configs — weights never correlated with 21-day forward
  returns even when trained on daily reward. Not conclusive against a weekly signal
  (those policies optimized a daily objective), but it tempers E3's prior; weigh this
  before spending the implementation effort.
- **Confounds**: fewer decisions = less statistical power (the one place option 4's
  power concern genuinely bites); geometric batch-sampling recency bias
  (`sample_batch_starts`), entropy annealing, and `checkpoint_holdout_days` /
  `checkpoint_eval_every` were all tuned per-*day/step* — re-express each in weekly
  units deliberately rather than letting day-denominated defaults silently apply.

---

## Stage 4 — Cross-asset architecture (largest, last)

### E4 — Attention over assets

- [ ] **Hypothesis**: EIIE's independent-evaluator constraint is the bottleneck — a
  model that can compare assets *internally* (attention/transformer over the asset
  dimension, or a GNN) extracts cross-asset structure that hand-fed cross-sectional
  features could not.
- **Why**: assessment §6 option 5. Both hand-engineered cross-sectional features
  (peer-relative PE, market-relative momentum) returned nulls, but that only tested
  "can EIIE *consume* cross-sectional answers," not "can a model *discover* them."
- **Implementation**: new network class in `networks.py` (per-asset conv encoder →
  attention across the 50 slots → per-asset logit), selected by a config field; masking
  must respect the dynamic universe exactly as the softmax mask does today. Everything
  outside `networks.py` unchanged.
- **Evaluation**: full SEP, plus E0 re-run on the new architecture first (it must pass
  the overfit check before its sweep results mean anything).
- **Success/failure**: signal threshold.
- **Interpretation**: failure here, on top of Stages 0–3, is the strongest evidence
  yet for the assessment's honest prior — daily/weekly single-stock alpha in this
  50-stock B3 universe may not be extractable from public price/fundamental data —
  and the project fork becomes objective-level (risk/diversification mandate,
  macro-conditioning) rather than model-level.
- **Confounds**: more parameters *and* new topology change at once relative to E1 —
  unavoidable, but E1's result bounds the capacity contribution; slot-permutation
  invariance must be preserved (assets have no fixed identity across quarters) or the
  model can memorize slot positions — a new lookahead-adjacent trap E1 didn't have.

---

## Explicitly deprioritized

- **More seeds / more windows (option 4)**: an edge needing 32 seeds to detect is
  swamped by initialization noise in practice. Revisit only to adjudicate a borderline
  result from E1/E3/E4.
- **Any experiment combining two changes** (e.g. capacity + new feature): forbidden by
  the attribution rule unless a single-variable result explicitly motivates the
  interaction test.
