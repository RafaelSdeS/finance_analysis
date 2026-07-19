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
  - **Verdict**: capacity (channel width) is not the bottleneck. **E1b (dose-response)
    is skipped** — its precondition ("if E1 shows movement") is not met.
    *Correction (2026-07-18 design review)*: this verdict originally cited the
    E0-success branch ("E0 passed") — factually wrong, E0's recorded verdict is FAIL.
    The applicable branch (E0-failure + E1-failure) reads "failure points at
    architecture (E4) or optimization beyond LR" — and note E0b (optimization beyond
    LR) was skipped, not cleared. Superseded either way: see the Design Review section
    below, which re-reads E0 itself and reorders everything downstream.
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

---

# Design Review — 2026-07-18 (post-E0/E1) — v2 roadmap, SUPERSEDES Stages 2–4 above

Adversarial re-read of this plan after E0 and E1 concluded. Original stage text is kept
above for the record; where this section conflicts with it, this section wins. Findings
first (most severe first), then the revised roadmap.

## Finding 1 — the decisive instrument (8-D3) has three concrete defects. VERIFIED IN CODE.

The entire evidence base — five feature nulls, E0's FAIL, E1's 8/8 null — was read
through `diagnostics.ranking_quality()`. Read line-by-line today, it has three
independent problems:

1. **Scope bug — global space instead of active members** (`diagnostics.py:150,196-203`).
   `forward_return()` computes over all 171 non-cash *global* assets; Spearman and the
   top-decile cut then include ~121 assets/day that are not in the top-50 membership:
   some listed-but-non-member (real returns the agent is *masked from holding*), some
   pre-IPO/post-delisting (bfill/ffill-flat prices → forward return exactly 0.0). A real
   ranking signal on the 50 holdable names is diluted ~3.4× by pairs the agent cannot
   act on; the top-decile hit denominator includes unholdable names, mechanically
   depressing hit rates; on down days the 0.9-quantile cut can sit at ~0 so the flat
   phantom names *enter the "top decile"*.
2. **Tie-handling bug** (`diagnostics.py:158` — "ties are vanishingly unlikely in
   continuous returns, so no rank-averaging needed"). That assumption is violated by
   construction on every single day: ~121 weights are exactly 0.0 and a block of forward
   returns is exactly 0.0. `argsort`-based ranking assigns arbitrary distinct ranks
   inside these massive tie blocks. If both axes happen to tie-break in similar index
   order, the (w=0, fwd=0) block is spuriously concordant with itself — a small
   systematic *positive* bias is plausible, and suspicious in the data: 6/8 E1 seeds and
   E0 all landed at +0.001..+0.003 rather than symmetric noise. Testable: recompute with
   average ranks; if the positive lean vanishes, it was artifact.
3. **Circular threshold calibration**. The ±0.02 signal bar was set from "the observed
   noise band so far" — i.e., from the very metric now known to be diluted and
   tie-biased. Neither the observed values *nor the bar* mean what the plan thinks.
   There is no permutation null anywhere in the daily pipeline (the plan prescribes one
   for E3 only — the one place it hasn't been needed yet).

**Consequence**: every 8-D3 number in this file is unreliable in both directions. The
nulls are *probably* still nulls (dilution shrinks signal, and PV/hit-rate corroborate),
but "probably" is not what a decisive metric is for. Fix the instrument, then re-read
ALL committed runs for free — every run's `weights.npz` is in git.

## Finding 2 — even a fixed Spearman is low-powered for the policies this objective produces

For reward `log(w·y)` with `w = softmax(z)`: `∂L/∂z_j = w_j·(y_j/(w·y) − 1)` — the
gradient on each asset's logit scales with its *current weight*. EIIE's shared evaluator
means tail scores aren't pure init noise (one function scores all assets), but the loss
only ever disciplines the scorer on the high-weight region — "does this look like the
current winner," not "order all 50 by expected return." Observed policies hold
effective_n ≈ 1.2–2.4. A full-cross-section rank correlation grades an ordering the
objective never trained. Two consequences:
- **Metric side**: add a metric with full power at any concentration —
  **selection alpha**: forward log-return of the argmax (and weight-top-k) holdings
  minus the active-universe mean that day, with a permutation null. This measures
  exactly "is the pick better than random," which is what the top of the policy is
  actually trained to do.
- **Training side**: if we want the *ranking* to be trained (and measurable), the
  policy must be non-degenerate — see M2 (entropy floor). At eff_n ≈ 10 the loss
  disciplines the scorer on ~10 names/day instead of ~1.5, and it directly attacks the
  documented bistable corner-seeking (cash attractor / one-hot) at its mechanism.

## Finding 3 — E0's verdict is internally inconsistent: the model DID fit its training window

E0 recorded "FAIL, clean — cannot memorize." Its own numbers refute the "clean":
train-window return **+432.9%** vs. best-stock buy-and-hold **+44.6%** with **65 argmax
switches** — log-return 4.5× the best static position. The block's stated trap ("matching
Best-Stock via static parking is not memorization") explains matching, not a 10×
day-varying beat. A policy earning 4.5× the best static log-return *on its training data*
has fit substantial structure of that window (within-oil-complex switching), full stop.
What E0 actually established: **the optimizer and network fit the training objective
fine** (loss ↓17×, smooth convergence, monetized in-sample); the *Spearman bar* measures
a different capability (full cross-sectional ordering) that the objective never demands
— see Findings 1–2. The E0 "FAIL" fed the "capacity/architecture is limiting" prior that
motivated E1; that prior was misassigned, and E1's null is the confirmation. Also never
computed: the day-by-day **hindsight-optimal PV** (the block's own listed yardstick) —
without it, "how much memorization" was unquantifiable. One line on saved data; do in M1.

## Finding 4 — fitting vs. generalization: never separated, still

No sweep logs train-window diagnostics. E1 concluded "capacity isn't it" from val-only
metrics — which cannot distinguish "can't fit more" from "fits more, generalizes
nothing." (E4's spec even requires an E0-rerun for the new architecture; E1 skipped the
analogous check.) Fix once in `experiment.py`: after pretrain, also run the frozen model
over the train window and write `train_metrics.json` + the same diagnostics. Every
future sweep then doubles as a bias/variance read at zero incremental cost.

## Finding 5 — the standing constraint freezes the variable the evidence now points at

"Paper log-return reward, no reward reshaping" was right for the reproduction phase.
The reproduction phase is over, and it succeeded — at reproducing the *mechanism*:
maximizing empirical log-growth over a finite sample concentrates on the
highest-realized-growth corner (here, the oil complex). That is textbook behavior of
empirical growth-optimal (Kelly) estimation — extreme estimation-error sensitivity and
concentration (MacLean/Thorp/Ziemba 2011), the same estimation problem behind DeMiguel
et al. 2009's result that even optimized Markowitz loses to 1/N out of sample. Our
metrics tables show exactly that: UBAH +37%, UCRP +13%, agent −47%..+10%. The paper's
domain (Poloniex crypto, 30-minute bars, enormous cross-sectional dispersion, 0%-return
cash) is where this objective family found signal; daily B3 equities with CDI cash is
not that domain, and independent replications of EIIE on daily equities have generally
failed to reproduce the crypto results. Keeping the objective frozen while spending
sweeps on capacity/features/architecture inverts the evidence. **Decision point
(user sign-off required, it amends a standing constraint)**: permit *entropy floor*
(raise `entropy_beta_end` so the converged policy stays diversified) as the one
sanctioned objective-level change. It is standard policy-gradient regularization
(max-entropy RL), uses an existing config knob, zero new code, and is the minimal
change that makes the training objective and the skill metric point at the same thing.

## Finding 6 — statistical validity gaps

- **Single regime**: every conclusion is conditional on one val window (2021–2023:
  post-COVID recovery + commodity supercycle + election). SEP v2: any accepted signal
  must also replicate on a disjoint window before it re-baselines anything.
- **Permutation nulls everywhere**, not just E3: shuffle weight rows across days within
  a run (B≈1000, seeded), recompute each metric → per-run p-value. Kills the circular
  threshold (Finding 1.3) and the multiplicity hand-wraving in one move; the seeds 9–16
  replication rule stays for accepted signals.
- 6/8 same-sign alone is weak (p≈0.29 two-sided under null); it only works combined
  with calibrated magnitude bars — which don't exist until the permutation nulls do.

## Finding 7 — process/QA debt that already cost runs

- **Sanity-gate flake**: `sanity.py:119` `np.allclose` default rtol=1e-5 failed on a
  cuDNN rel-diff of ~9e-5 at step 3 — a whole seed lost + manual retry. Real seeding
  bugs diverge at O(1); set rtol=1e-3 for the GPU path.
- **OOM policy**: wide-net jobs are ~2.4 GB RSS each; `-j 4` + VS Code on a 15 GB box
  OOM-killed the editor twice and (indirectly) a sweep. `sweep.py` should check
  available RAM and clamp `-j` with a warning (psutil is available), and optionally
  retry a failed seed once.
- **Hand-rebuilt summaries**: per-seed table (Spearman/hit/entropy/eff_n/cash/return)
  was assembled by hand three times this week. `sweep.py` should write
  `sweep_summary.json` aggregating per-run metrics + diagnostics at the end.
- **Saturation probe**: built for E0, decisive once, then left in the E0 script.
  Promote the entropy/max-weight trajectory logging into `pretrain()` proper (cheap,
  answers "which corner, when" for every future run).
- **Missing regression armor for later stages**: E3's surgery has no golden test
  pinning `decision_period_days=1` to current behavior; E4 has no slot-permutation
  equivariance test spec. Both required before any such code is written.

## Finding 8 — portfolio cuts (ladder applied)

- **E2a/E2b (features): CUT.** Adding input channels to an objective that only
  disciplines the argmax, measured by a diluted metric, is spending sweeps to move a
  broken needle. Reinstate only if M3 shows signal and asks "which features carry it."
- **E3 (weekly): environment surgery deferred, question kept.** The horizon question
  collapses into a *label parameter* of the supervised probe (M3: k∈{1,5,21}) — no
  `environment.py`/`train.py` surgery, no recalibrated weekly thresholds, same
  information. Build weekly RL only if weekly signal is actually found.
- **E4 (attention RL): deferred, replaced by M5.** Attention cannot conjure signal a
  supervised probe can't find, and it inherits the concentration attractor. Test the
  architecture question inside the probe (attention layer in a supervised ranker) at
  ~1/10 the cost; port to RL only on a positive.

## Hidden assumptions surfaced

- A1: "8-D3 measures what training optimizes" — false (Findings 1–2). Load-bearing.
- A2: "Spearman ≈ 0 + big return ⇒ luck/no memorization" — E0's own numbers refute it.
- A3: "After optimizer cleared, capacity/architecture are the suspects" — the objective
  was never on the suspect list because a standing constraint froze it.
- A4: "The 2021–2023 window generalizes" — untested regime-conditionality.
- A5: "Entropy beta is nuisance calibration" — it is the concentration dial that
  decides whether ranking is trained *and* measurable. First-class variable.
- A6: "PETR-complex = luck" — never tested against "genuine momentum structure of the
  window." (M1's fixed metrics + hindsight yardstick partially adjudicate this.)

## Revised roadmap (v2) — supersedes Stages 2–4

| M | What | Code | Compute | Gate |
|---|---|---|---|---|
| M1 | Fix the instrument + retro re-read all runs | ~1 day | none (offline) | **blocks everything** |
| M2 | Entropy-floor sweep (existing knob) | none | calib + 1 sweep | M1 done + constraint sign-off |
| M3 | Supervised ranking probe, k∈{1,5,21} | ~1–2 days | cheap (supervised) | M1 done; parallel to M2 |
| M4 | Decision gate on M1+M2+M3 | — | — | all three read |
| M5 | Attention inside the probe | small | cheap | only per M4 |

### M1 — Fix the measuring instrument, re-read history (no training)
- [x] `ranking_quality()` fixed (2026-07-18, `src/rl_agent/diagnostics.py`): Spearman +
      decile universe now restricted to `panel.valid[t]` active members via
      `_active_stock_gidx()`; `_spearman` now uses `_rankdata` (tie-aware average-rank).
      Existing tests (fixed-universe synthetic fixtures) pass unchanged — the fix is a
      no-op there by construction, confirming backward compatibility.
- [x] Added `selection_alpha()`: forward log-return of argmax/weight-top-k holdings
      minus active-universe mean, per (k, top_k), with a permutation null (redraw a
      random top-k subset per day, B=1000 default) → 97.5th-pct threshold + p-value.
- [x] Added `spearman_permutation_null()`: recalibrates the signal threshold per run by
      shuffling weight-day/return-day pairing (day's own active-only rank vectors held
      fixed) → 97.5th-pct threshold + p-value, replacing the hand-set ±0.02/12% bar.
      **Caveat discovered building this**: the null specifically tests for *day-specific*
      timing information — a policy with a purely *static* factor tilt (same ranking
      reused every day) is degenerate under this null (shuffling day-pairing changes
      nothing if the "pairing" carries no day-specific content), which surfaced as a
      literal floating-point-only null distribution in early test iterations. Real
      trained policies vary their weight vector day to day (reacting to that day's
      input window), so this isn't a practical limitation for actual runs — but it's a
      sharp reminder that this null answers "is there timing information," not "is
      there a real factor tilt at all" (the latter needs `selection_alpha` + a
      multi-window check, not this null alone).
- [x] Tests added (`tests/rl_agent/test_diagnostics.py`, 22/22 pass): tie-averaging
      regression (`_rankdata` on a hand-verified tie block), active-only dilution case
      (phantom never-holdable tickers growing far faster than any holdable name — old
      code would have diluted/zeroed the hit rate; fixed code reproduces the
      undiluted perfect-ranking result exactly), `selection_alpha` known-skill vs.
      known-anti-skill cases, `spearman_permutation_null` day-varying skilled vs.
      random cases (the latter two needed day-varying synthetic weights specifically
      because of the static-tilt degeneracy above).
- [x] **Re-ran diagnostics over all recoverable committed experiment dirs** — E0, all 8
      E1 seeds, and all 8-seed sweeps for `eiie_features_frozen` (the original
      11-channel baseline), `eiie_momentum_market`, and `eiie_pe_sector` (36 runs
      total). **Four older families (`eiie_features`, `eiie_features_b100`,
      `eiie_valuation_b50`, `eiie_valuation_b100`) predate the `weights.npz` artifact
      and would need `--replay` (reload dataset + checkpoint per run) — not done here,
      flagged as a remaining gap, not silently skipped.**
  - **Process note**: recovering these directories required restoring 341 files across
    ~40 experiment dirs that were tracked in git HEAD but missing from the working
    tree (deleted at some point without the deletion being committed — not caused by
    this session's work). Restored via `git restore --source=HEAD -- experiments/`
    with explicit user sign-off before touching it, per the "investigate unexpected
    repo state before acting" rule. Nothing was lost; HEAD was always the source of
    truth.
  - **E0 (train window, seed 42, 123 days)**: fixed metric gives k=1 Spearman
    **0.042**, k=5 **0.084**, k=21 **0.111** — ALL significant vs. the permutation null
    (p ≤ 0.021, most p < 0.001). The originally reported ~0.001 was the dilution
    artifact. **Hindsight-optimal ceiling** (a true omniscient day-by-day best-picker,
    zero cost) for this window: **+173,925%** — E0's own +433% captures only ~22% of
    the available log-return "memorization budget" (log(1+4.33)/log(1+1739.25) ≈ 0.22),
    not the ~0% the original "clean FAIL" implied, but nowhere near full memorization
    either. **Verdict revised**: E0 shows real, partial, statistically confirmed
    in-sample ranking structure — "optimizer/architecture can't fit anything" is
    false; "fits everything" is also false. The tie-bias hypothesis (Finding 1.2, "does
    the +0.002 lean vanish under average-rank ties") is moot — the fix revealed
    structure an order of magnitude larger than any plausible tie artifact, not a
    small bias that vanished.
  - **E1 (8 seeds, widened encoder) — re-verdict**: k=1/5/21 Spearman now significant
    and POSITIVE in 5/8 seeds (same 5 at every horizon), climbing monotonically with
    horizon in every one of them (e.g. strongest: 0.025 → 0.048 → 0.070, all
    p=0.002); `selection_alpha` independently confirms the same 5 seeds significant at
    k=5/21 (p ≤ 0.014, mostly p<0.002), null in the other 3. The 5 "signal" seeds also
    have far higher `n_active_days` (257–519 of 576) than the 3 null seeds (68–161) —
    the split tracks how invested the policy stayed, not which seed number. **This
    overturns the "unanimous 8/8 null, capacity ruled out" verdict recorded above** —
    under the diluted metric this was invisible.
  - **Same pattern reproduces in ALL THREE other re-read families**
    (`eiie_features_frozen`, `eiie_momentum_market`, `eiie_pe_sector`) — each an
    independent 8-seed sweep with a *different* extra feature channel (or none):
    2–4 of 8 seeds per family show the identical signature (significant, positive,
    horizon-climbing Spearman + selection alpha, p<0.05 mostly p<0.005), the rest
    null/negative. Magnitudes are close to E1's across all four families (k=21 rho
    typically 0.07–0.10 in the "signal" seeds).
  - **Synthesis — why this is "candidate," not "signal"**: the pattern's
    *cross-config* consistency is itself the tell. It shows up at nearly identical
    magnitude regardless of which extra feature channel is present (momentum,
    PE-sector, wider encoder, or none) — that rules out "this specific feature carries
    real information" as the explanation (Stage 2's original premise), and instead
    points at something about the *base setup* (this architecture, this training
    objective, this one 2021–2023 window). Checked the strongest signal-seed's
    holdings in both E1 and the historical families: **same PETR3/PETR4/PRIO3/GGBR4
    commodity complex documented everywhere else in this investigation** — the real
    2022 Ukraine-war oil shock. A subset of seeds' policies happen to converge to
    riding that one genuine, long, single macro trend hard enough and long enough
    that it registers as a horizon-climbing, statistically significant correlation
    *within this window* — indistinguishable, using only this data, from actual
    multi-day forecasting skill. The signal getting stronger at k=21 than k=1 is
    consistent with "rode one slow trend for months" and equally consistent with "the
    market has weak multi-week momentum here"; this design review cannot separate
    them alone (Finding 5/6's single-regime concern, now concretely realized rather
    than theoretical).
  - **Per M1's own gate**: technically "a historical run shows real signal under the
    fixed instrument" — literally true in four separate configs. But because it's the
    *same* signal in the *same* magnitude regardless of config, it does not "jump the
    queue" toward any one feature/config. It jumps the queue toward the single
    concrete check that can actually distinguish luck from skill: **replicate on a
    disjoint window** (SEP v2, Finding 6) before any of this counts as more than
    candidate. Seeds 9–16 replication (same window) is necessary but not sufficient —
    a second window is the decisive test, since a real 2022-only oil shock would
    trivially "replicate" across seeds 9–16 (same window, same trend) while still
    being pure luck.
- [x] E0 addendum: hindsight-optimal PV computed above (+173,925%, recalibrates the
      original +433%/+44.6% comparison). Tie-bias hypothesis addressed above (moot).
- [x] `experiment.py`: train-window metrics/diagnostics per run (Finding 4). After
      saving `model_pretrain.pt`, runs a frozen inference-only backtest over the same
      span pretrain() trained on (`panel.start_idx` → `pretrain_end_idx`), writes
      `train_metrics.json` (n_days, total_return, ranking_quality at k=1/5/21). PVM
      buffer is snapshotted before and restored after — this walks a wider span than
      pretrain()'s own holdout refresh touches, and left unrestored would silently
      change the w_prev the online backtest reads at the train/val seam. Tests
      (`test_experiment.py`, 14/14 pass) check the file's shape AND that the agent's
      val-window return stays finite (the seam wasn't perturbed). The *next* sweep
      (e.g. the second-window replication M1 calls for) gets train/val together for
      free.
- [x] QA (Finding 7): sanity-gate `deterministic_seeding` now rtol=1e-3 (was 1e-5;
      cuDNN per-process algorithm selection was failing this at ~9e-5, not a real
      seeding bug — real bugs diverge at O(1)). `sweep.py`'s `EST_RAM_PER_JOB_MB`
      corrected 700→2500 (the E1 sweep measured ~2.3-2.4 GB/job via `/proc/<pid>/
      status`, not the ~0.5 GB the old estimate assumed for the smaller original
      network — the stale constant is *why* -j under-clamped and OOM-killed VS Code
      twice this session). `run_jobs()` now retries a failed job once before counting
      it as a real failure (covers exactly the cuDNN sanity-gate flake and transient
      OOM hit this session). `sweep.py` now writes `sweep_summary.json` (per-job
      status + `metrics_summary.json`, parsed from each job's own "Artifacts in"
      log line — race-free vs. scanning `experiments/` for the newest matching dir).
      `saturation_probe()` promoted from the E0 script into `train.py` proper,
      wired into `pretrain()` via an optional `saturation_log_path` (composes with
      any externally-supplied `on_step`, doesn't replace it) — every future real run
      gets `saturation_probe.json` for free via `experiment.py`. Tests:
      `tests/rl_agent/test_sweep.py` (+7, retry-once/retry-exhausted/artifact-log-
      parsing/summary-writing) and `tests/rl_agent/test_train.py` (+2, saturation
      log shape + on_step composition), all passing.
- **Tests**: synthetic known-skill weights → positive selection alpha & Spearman;
  shuffled → ~0 with calibrated null; tie-block regression case (a (w=0, fwd=0) block
  must contribute nothing); active-only scope case (planted signal on members must not
  be diluted by non-members). **Done — see above, 22/22 pass.**
- **Gate**: if ANY historical run shows real signal under the fixed instrument, that
  config jumps the queue. If all stay null, proceed with a trustworthy baseline.
  **Outcome: neither branch cleanly fires — see synthesis above. Candidate signal
  found, cross-config, most parsimoniously explained by single-window luck. Next
  action before M2/M3 proceed: run the SAME accepted config on a second, disjoint
  window (e.g. 2018–2020, pre-COVID, no oil shock) and re-apply this fixed
  instrument. If the signal vanishes on the second window, the luck explanation is
  confirmed and M2/M3 proceed exactly as planned below. If it persists, this becomes
  the actual M4 "Strong" row and reprioritizes ahead of M2/M3.**

### M2 — Entropy floor: make the objective train (and expose) more than the argmax
- [ ] **Requires sign-off** (amends "no reward reshaping" — Finding 5).
- [ ] Calibration mini-run (2–3 betas × 2 seeds, labeled calibration, not evidence):
      pick `entropy_beta_end` targeting effective_n ≈ 5–15 (too high → uniform, logits
      flatter than noise; too low → status quo). Candidates: 1e-3, 3e-3, 1e-2
      (current: 1e-5). Fallback if the knob can't hold the band: ε-uniform weight
      mixing — more code, only if needed.
- [ ] Full SEP at the chosen beta, judged by M1's fixed metrics. Also the product bar:
      does the diversified agent beat UCRP (=1/N — the bar DeMiguel showed is the hard
      one) net of costs?
- **Confound**: beta changes both the trained objective and metric power — that is the
  point; record it as the sanctioned deviation, one variable.

### M3 — Supervised ranking probe (the premise experiment)
- [ ] `src/rl_agent/supervised_probe.py`: same conv trunk (no softmax head, no
      `w_prev`), per-asset score; listwise softmax cross-entropy over each day's
      active cross-section against forward k-day returns (cross-sectionally
      standardized); masked to active slots. Horizons k∈{1,5,21} are configs, not code.
- [ ] Metric: daily IC (active-only Spearman(scores, realized)) on train AND val, vs.
      permutation null. This is the direct, maximally-powered answer to "is there
      extractable cross-sectional signal in these features at horizon k" — no RL
      credit-assignment noise, all 50 assets disciplined every day.
- [ ] Expectation calibration: Gu–Kelly–Xiu 2020 got OOS R² ≈ 0.4%/month with ~900
      features on US equities. Daily, price-only, 50 B3 names is a strictly harder ask.
      IC of 0.02–0.05 at k=5/21 would be a strong result here.
- **Tests**: label alignment (window ends t, label (t, t+k], zero overlap); masked-slot
  loss contribution exactly 0; leakage guard reuse (`test_ffill_guard` pattern);
  synthetic end-to-end (planted predictive feature → IC recovered; shuffled labels →
  IC ≈ null). Edge cases: delisting terminal returns; PETR3/PETR4 near-duplicate
  cross-section (note, don't fix yet).
- **Interpretation**: IC > null at some k → signal exists; bottleneck is confirmed as
  objective/policy, not representation → two-stage build (rank → portfolio
  construction) or RL fine-tune at that k. IC ≈ null at all k → the strongest premise
  answer this project can produce: no extractable daily/weekly cross-sectional signal
  in these inputs. That conclusion still needs the multi-window check before it's
  final (Finding 6).

### M4 — Decision gate (no compute) — EXPLICIT, to prevent "one more experiment" syndrome

Definitions (fixed *before* looking at M3 results): **null** = val IC inside the
permutation null's 95% band at every k. **Weak** = above the 97.5th percentile at some
k, but IC < 0.03. **Strong** = IC ≥ 0.03 at some k, same sign in ≥ 6/8 seeds.
(Calibration: Gu–Kelly–Xiu-level results on far richer data correspond to ~0.02–0.05
here; 0.03 is "clearly real, plausibly monetizable after costs.")

| M3 result | + context | Next step — and *only* this step |
|---|---|---|
| Null at all k | M1 retro-read also null, M2 null | **STOP model-side work.** One confirmation run on a disjoint window to seal it, then write the premise conclusion. Pivot is objective-level (risk/diversification mandate, macro-conditioning, different data). M5 optional as a final cheap falsification, then nothing else. |
| Null at all k | M1 retro-read or M2 found something | The RL-side result outranks the probe's null: chase that specific config/against a second window. No new feature/capacity/architecture experiments. |
| Weak | — | Allocation research only: can a portfolio layer monetize a weak IC after costs (turnover-budgeted top-k, etc.)? **No model-improvement experiments** — a weak signal doesn't fund an architecture search. |
| Strong | M2 null or unhelpful | Head-to-head: predict-then-allocate pipeline vs. RL fine-tune at the winning k. Winner (net of costs, both windows) becomes the system. |
| Strong | M2 also positive (entropy floor helped RL) | Objective-level regularization research: the evidence says signal exists AND the RL objective can express it when regularized — tune that axis (entropy floor, risk-sensitive reward à la Moody–Saffell), not inputs or topology. |

Standing rules: (1) any experiment not named in this table requires amending this plan
first, in writing, with the hypothesis it tests — no ad-hoc runs. (2) Every row's
outcome must replicate per SEP v2 (fresh seeds + disjoint window) before it
re-baselines anything. (3) Weekly env surgery (old E3) only enters via the Strong rows
and only if k=5 wins and native weekly RL is explicitly wanted.

### M5 — Attention inside the probe (conditional)
- [ ] Attention layer across the 50 slots in the supervised probe; slot-permutation
      equivariance test mandatory (shuffle slots → permuted outputs, bit-exact).
      Isolates "can cross-asset comparison find what independent evaluation can't" at
      prediction level, without RL noise or the concentration attractor.

## SEP v2 amendments (once M1 lands)
1. All Spearman/hit metrics active-members-only, tie-aware.
2. Selection alpha + permutation p-values accompany every run; thresholds come from
   the permutation null, not hand-set bands.
3. Train-window metrics logged for every run.
4. Any accepted signal must replicate (seeds 9–16) AND hold on a second, disjoint
   val window.

## References
- Jiang, Xu & Liang 2017, *A Deep RL Framework for the Financial Portfolio Management
  Problem* (arXiv:1706.10059) — source paper; domain is Poloniex crypto at 30-min bars.
- DeMiguel, Garlappi & Uppal 2009, *Optimal Versus Naive Diversification* (RFS 22-5) —
  1/N beats optimized portfolios OOS; the estimation-error frame for Finding 5.
- MacLean, Thorp & Ziemba 2011, *The Kelly Capital Growth Investment Criterion* —
  empirical growth-optimal concentration & sensitivity (the attractor's theory).
- Gu, Kelly & Xiu 2020, *Empirical Asset Pricing via Machine Learning* (RFS 33-5) —
  supervised cross-sectional NN benchmark; calibrates M3 expectations.
- Haarnoja et al. 2018 (SAC) / max-entropy RL — entropy regularization as standard
  practice, grounding M2's legitimacy.
- Cao et al. 2007 (ListNet) — listwise ranking loss used in M3.
- Moody & Saffell 2001, *Learning to Trade via Direct Reinforcement* (IEEE TNN) —
  precedent for risk-sensitive objective modification in trading RL (differential
  Sharpe), relevant if M4 lands on objective redesign.
