# Root-Cause Investigation Plan: Why the RL Agent Shows No Alpha

> Local file (companion to `TODO.md`). Created 2026-07-09 from a full-pipeline review.
> Check boxes as phases complete; record verdicts inline under each phase.

## Context

After the M1–M7 fixes (honest SELIC-excess metrics, HAC/bootstrap stats, cap-bug fix,
cash-aware reward, BC-pretrain log_std fix, 5M-step budget, multi-seed protocol), the
stitched 16-year walk-forward says: **agent excess-of-SELIC Sharpe 0.172 vs EW 0.185,
market-cap 0.181, inv-vol 0.165** — statistically indistinguishable. M5 already ruled out:
missing signal (IC 0.03–0.10 every window), backwards policy (ρ=+0.18 alignment with
ranker), broken optimization (in-sample +0.37 beats EW's +0.28). The logit_scale sweep
ruled out temperature-alone.

This plan comes from a fresh read of the full pipeline. It found **three unexamined
problems upstream of everything tried so far**, plus a statistical-power result that
reframes what "success" can mean. Phases are ordered to eliminate whole problem classes
cheaply before any further agent training.

---

## New findings (2026-07-09 review — not previously in TODO.md)

### F1. Survivorship + universe-selection lookahead (data layer) — CONFIRMED, measured
- `data_pipeline.py:52-54` picks the top-50 by **mean market cap over the full sample,
  including the future**. Measured: only 29/50 of the current universe would have been
  selected on 2011 data → **21 of window-0's 50 slots were chosen using post-2012
  information**.
- The dataset is survivor-only: 288 tickers, only 10 stop trading before mid-2025 —
  essentially **zero delistings/bankruptcies in 26 years of Brazilian equities**, impossible
  for a point-in-time universe. Collection started from today's listing.
- Impact: every learned feature→return relationship is conditioned on survival/growth, and
  no Sharpe measured on the non-PIT tensors is a legitimate backtest of a historically
  executable strategy. **Measured in Phase 0.2 (below): the bias is real but its sign is
  NOT a uniform inflation** — full-sample selection actually *understates* EW Sharpe in
  6/8 windows (mean gap −0.13 Sharpe), concentrated in the earliest windows where the
  candidate pool was thin. The takeaway is contamination of *which strategy is being
  tested*, not a one-directional overstatement of performance.

### F2. Signal was measured on the wrong universe (feature layer) — code-confirmed, measured
- `ic_analysis.py` computes IC on the **full ~288-name dataset**, but the agent trades only
  the **top-50 large caps**, where cross-sectional alpha (quality/value especially) is
  typically far weaker.
- `ranker_baseline.py:89` splits **by rows** (ticker-holdout with full temporal overlap),
  so the teacher's t=24 partly measures contemporaneous fit, not real-time knowledge.
- This mechanically explains the standing paradox: positive measured IC everywhere, yet the
  ranker-top-20% portfolio loses OOS (Sharpe −0.34, worst of the M5 four-way).
- **Measured in Phase 0.3 (below):** the headline IC 0.03–0.10 does NOT vanish on the
  tradable top-50 — mean |rank-IC| shrinks moderately (0.060 → 0.047, full → PIT top-50),
  but its **non-overlap t-stat collapses from 1.91 to 0.51**. Revised framing: the signal
  isn't absent on the tradable universe, but ~50 names/day isn't enough breadth to
  distinguish it from noise — F2 is really an instance of F3's power problem, one level
  earlier than expected (signal detection, not just strategy evaluation).

### F3. The success bar is statistically unreachable as defined (evaluation layer)
- SE of annualized Sharpe ≈ 1/√T-years: **±0.71 per 2y window, ±0.25 pooled over 16y** →
  minimum detectable effect at 80% power ≈ **0.70 Sharpe pooled**, while the theoretical
  ceiling of this strategy class (Grinold: IR ≈ IC·√breadth·TC; IC≈0.05, ~50 names,
  monthly, long-only TC≈0.3 per Clarke–de Silva–Thorley) is **≈0.2–0.4**. Even a perfect
  agent can't produce a significant time-series Sharpe here. EW itself earned only 0.185
  excess-of-SELIC Sharpe in 16y — the whole equity premium is inside one SE.
- Consequence: move the primary metric to where the power is — cross-sectional tests
  (~190 independent 21d cross-sections × 50 names resolve mean rank-IC ≈ 0.05) and the
  selection-residual stream. Time-series excess Sharpe becomes descriptive only.

### Minor findings (fix opportunistically)
- [ ] `selic` (and all regime features) absent from observations — cash *timing* is
      unlearnable from the current obs; either add regime features or stop expecting timing.
- [ ] `has_fundamentals` commented out while NaN→0 imputation maps "no filing" to "average
      company" (most of pre-2011 train span). CLAUDE.md's claim it is included is stale.
      If re-added: test that a no-filings ticker and an exactly-average ticker produce
      *different* observations post-imputation (currently identical — the problem).
- [ ] `env.py` episodes always reset to `_t=0`: every rollout replays the identical path;
      16 identical envs → batch diversity is action noise only. Random-start episodes.
- [ ] Feature selection provenance ("RF importance R²=0.233") smells in-sample/leaky —
      re-derive on PIT data with time-split validation when Phase 0 lands.

---

## Component-by-component verdicts

| Component | Verdict | Evidence |
|---|---|---|
| Label/return alignment | Sound (decide t, earn t+1…t+N) | env.py; test_eval_integrity.py |
| Reward design | Sound post-M3.1; λ-audit (3.2) open, low priority | test_env_basic 7b–7d |
| Env portfolio math | Sound post-M2.1; 10bps costs reasonable | TODO 1.7 audit |
| PPO optimization | Works (overfits in-sample when allowed) | M5 5.2 |
| Action geometry | Degenerate (pre-cap 83% one-name) but temperature ruled out as lone fix | M5, M2.3 |
| Evaluation stats | Machinery sound, aimed at an underpowered metric | F3 |
| **Universe/data** | **Contaminated: survivorship + selection lookahead, unquantified** | F1 |
| **Feature signal** | **Never measured on tradable universe; teacher leak-validated** | F2 |
| Validation/selection | Single contiguous 15% tail = one regime; val→test transfer unmeasured | 6.4 open |

**Synthesis:** the agent may be failing because the tradable signal doesn't exist where it
trades (F2), measured on data that flatters everything (F1), judged by a test that couldn't
certify success anyway (F3). All three sit below the RL layer.

---

## Work already done — verified, do not redo

- [x] M1: excess-of-SELIC metrics, provenance sidecars, freshness gates, HAC + bootstrap,
      n=200 null, eval-path leakage audit
- [x] M2.1 cap-redistribution fix (effective_n 1.7→22.6); M2.3 logit_scale sweep (3 seeds,
      no robust winner in {4,5,6})
- [x] M3.1 cash-aware previous-weight reward (twice-corrected, test-covered)
- [x] M4.1/4.3 blend baselines + agent-matched blend + ranker-top-20% baseline
- [x] M5: per-window IC (full universe — Phase 0 redoes on tradable), alignment check,
      concentration autopsy, attribution, BC-init/final/teacher three-way, in-sample
      overfit check
- [x] Training fixes: 5M budget, early-stopping threshold + checkpoint decoupling,
      BC log_std recalibration, online-backtest day-indexing fix
- [x] Infra: `--seed`/`--logit-scale`/`--no-promote`, BC-init snapshots,
      `tools/policy_diagnostics.py`, 3-seed protocol

Open TODO.md items absorbed here: M6.1 → Phase 2.1; Option C → Phase 2.2; 6.2 → Phase 2.3;
6.4 + 3.2 → Phase 3; 2.2 cash-mode → Phase 2.4; 7.3 harness → build when Phase 2 needs it.

---

## Phased roadmap

Ordering principle: each phase can kill an entire class of hypotheses for days of CPU work,
not weeks of GPU time. **No agent retraining until Phase 2.**

### Phase 0 — Fix the measuring stick (data + signal + power). ~2–3 days, CPU only.

**Objective:** know whether tradable, point-in-time signal exists in the top-50 universe,
and define success metrics that can detect the plausible effect size.

- [x] 0.1 **Point-in-time universe** in `data_pipeline.py`: top-50 by mean market cap using
      only data ≤ each window's `train_end` (per-window universes, like per-window scalers).
      Current mode stays behind a flag for A/B.
      - **Done (2026-07-10):** `select_universe(caps, pool, size, cutoff)` added; `build_tensors`
        takes `config.pit_universe` (default False = unchanged full-sample behavior). When
        True, per-window PIT selections are unioned into the tensor's ticker set and a
        `pit_membership_mask[cutoff, ticker]` array is saved; `env.py`'s
        `_load_normalized_tensors` ANDs the matching row into the activity mask so a stock
        only appears in state/action space for windows where it was actually top-N at
        decision time; `run_pipeline`'s per-window scaler fit is also PIT-restricted for
        consistency. CLI: `python -m src.agent.data_pipeline --universe-size 50 --pit-universe`.
        Real-data smoke check (top-10, 8 windows): windows 0–1 (pre-2011, thin listings)
        select small/illiquid names; windows 2–7 converge to blue chips
        (PETR4/VALE3/ITUB4/BBDC4/...) — only 3 tickers (ABEV3, B3SA3, VIVT3) turn over across
        the last 6 windows once the market matures, confirming this isn't just noise.
      - **Tests** (`tests/agent/test_universe_point_in_time.py`, fast group, synthetic
        panel): *problem* — ticker X is small until year 5 then huge; full-sample selection
        (`cutoff=None`) includes X in a universe_size=2 pick even though X was never
        actually top-2 early on (demonstrates the lookahead). *Solution* — cutoff-based
        selection excludes X pre-rocket; future-blindness property (truncating the panel
        after cutoff doesn't change the cutoff's selection); per-window selections differ
        once caps drift (rules out silently reusing one global list). All 5 cases pass.
        Also added `tests/agent/test_data_pipeline_universe.py` (data group) wrapping the
        module's own `_selfcheck_universe_filter`/`_selfcheck_pit_universe` — these existed
        as dead code (never wired into `__main__` or any runner) and running them for the
        first time surfaced a real latent bug: `_selfcheck_universe_filter`'s assertion
        didn't account for CASH being added on top of `universe_size`, always failing when
        CASH is in the universe. Fixed the assertion and registered both new test files in
        `tests/run_all.py`. Full suite: 20/20 passed.
- [x] 0.2 **Quantify the bias:** EW backtest PIT vs current universe per window.
      - **Done (2026-07-10):** `tools/quantify_survivorship_bias.py` (in-memory only, no
        production artifacts touched) rolls plain equal-weight through both universes'
        masks over each window's test span and diffs cumulative return / Sharpe.
        Added V8 to `verify_dataset_for_training.py` — report-only gate (always passes),
        prints the delisting count so the survivor-only limitation shows at every
        verification: **10/288 tickers (3.5%) end >90d before dataset end over 27 years** —
        confirms F1's "essentially survivor-only" claim with an exact number.
      - **Result — the bias is real but NOT a uniform inflation (revises the plan's
        original hypothesis):**
        ```
        window  test span              current_sharpe  pit_sharpe  gap(cur−pit)
        0       2010-01 → 2012-01      0.44            1.11        -0.67
        1       2012-01 → 2014-01      0.63            1.14        -0.50
        2       2014-01 → 2016-01     -0.34           -0.76        +0.43
        3       2016-01 → 2018-01      2.11            2.25        -0.14
        4       2018-01 → 2020-01      1.96            1.77        +0.19
        5       2020-01 → 2022-01     -0.02            0.20        -0.21
        6       2022-01 → 2024-01      0.68            0.70        -0.02
        7       2024-01 → 2026-01      0.30            0.44        -0.14
        mean gap: -0.13 Sharpe, -0.093 cumulative return. Current-mode overstates PIT in
        only 2/8 windows.
        ```
        **Interpretation:** the full-sample selector's contamination is largest in the
        earliest windows (0, 1) — but it *understates* rather than inflates Sharpe there.
        Mechanism: with few tickers having enough trading history by 2008–2011, the PIT
        top-50 for those early windows draws from a thin, then-emerging-company pool;
        several of those names had unusually strong 2010–2014 runs. The full-sample
        selector instead plugs in today's already-enormous blue chips (PETR4/VALE3/ITUB4/
        ...) from window 0 onward, which were already large and comparatively flat over
        that same 2010–2014 span. **The takeaway is not "backtests are inflated" — it's
        that the full-sample selector is testing a DIFFERENT, unrealized strategy** ("hold
        the stocks that will turn out to dominate 2026") in early windows, not the one a
        live 2010 investor could have run. The methodological problem (F1) stands
        regardless of sign: no Sharpe computed on the current (non-PIT) tensors is a
        legitimate backtest of any strategy that could have been executed historically.
        For the RL agent specifically (windows 2–7 dominate its training/eval budget,
        where PIT and full-sample universes are nearly identical — only 3 of 50 names turn
        over), this softens F1's practical impact on the agent's own headline numbers —
        but windows 0–1 remain contaminated, and Phase 1's supervised ceiling and any
        multi-window pooled statistic should use the PIT tensors, not the current ones.
        Full table: `artifacts/backtest/survivorship_bias.json`.
- [x] 0.3 **Tradable-universe IC audit:** `ic_analysis.py` gains a `--tickers`/universe
      filter; run per PIT top-50 per test window, 21d horizon, non-overlap t-stats.
      - **Done (2026-07-10):** `load_panel()` takes an optional `tickers` set (+ CLI
        `--tickers`), filtering before any pivot/forward-return computation. New
        `tools/tradable_universe_ic_audit.py` loads the panel once, then per window slices
        to that window's test span and to two ticker sets: the full listed universe and
        the PIT top-50 (`select_universe`, same selector as 0.1), reusing `daily_rank_ic`/
        `summarize` unchanged. Had to align feature/forward-return matrices by both date
        index AND ticker columns per feature (`_aligned_window`) — discovered that
        fundamental-feature pivots (e.g. `roe`) have fewer dates/tickers than price-based
        ones, since `pivot_table` drops slices that are entirely NaN (pre-2011 CVM era;
        the odd ticker with no fundamentals ever) — a real data-shape quirk, not a bug in
        the new code, but it would have silently misaligned cross-sections if unhandled.
      - **Result — signal shrinks moderately, but statistical confidence collapses:**
        ```
        window  test span          full |IC|  pit |IC|  full t-stat  pit t-stat
        0       2010→2012          0.067      0.059     1.64          0.66
        1       2012→2014          0.073      0.028     2.36         -0.11
        2       2014→2016          0.074      0.075     2.59          1.57
        3       2016→2018          0.025      0.043     0.66         -0.54
        4       2018→2020          0.034      0.034     1.14          0.25
        5       2020→2022          0.047      0.022     1.47          0.61
        6       2022→2024          0.065      0.052     1.48          0.86
        7       2024→2026          0.094      0.060     3.91          0.75
        mean:                      0.060      0.047     1.91          0.51
        ```
        (mean |rank-IC| across 20 of the agent's 21 `state_features` — `volume` is
        excluded by `ic_analysis.py`'s own `SKIP_COLS`, unrelated to this change; non-
        overlap t-stat, 21d horizon.) Full table: `artifacts/backtest/tradable_universe_ic.json`.
      - **Interpretation:** the headline "IC 0.03–0.10" from TODO.md's M5 diagnosis is
        confirmed on the full universe (mean 0.060, consistent with that range) — but on
        the top-50 the agent actually trades it drops to 0.047 (a real but moderate ~22%
        shrinkage, not the "IC≈0" the plan's F2 hypothesis worried about) while the
        **non-overlap t-stat collapses from 1.91 to 0.51** — nowhere near conventional
        significance even before any multiple-testing correction. This reframes F2: the
        cross-sectional signal is not *absent* on the tradable universe, but on ~50 names
        per day there isn't enough breadth to distinguish it from noise with the data
        available. This is the F3 power problem showing up one level earlier than
        expected (in signal detection, not just strategy evaluation) — directly motivates
        widening breadth (0.5's tilt-IC test, pooling across windows) before concluding
        "no signal" from any single-window number.
      - **Tests** (extended `tests/agent/test_ic_analysis.py`, fast group, 3 new cases):
        *problem* (Test 6) — synthetic 12-ticker panel where only a "SMALL" 6-ticker group
        carries signal (myfeat exactly ranks their next-day returns) and a "BIG" 6-ticker
        group's returns are a fixed derangement of its own feat order; full-universe IC
        (0.867, hand-verified via the Spearman formula) is driven almost entirely by the
        SMALL group. *Solution* (Test 7) — filtering to the BIG-only ("tradable") subset
        collapses IC to −0.086, matching the hand-derived value exactly, demonstrating the
        real audit's mechanism on a controlled case. (Test 8) — filter honored exactly:
        `features[feat].columns == requested set`, no leakage; `min_names` exceeding the
        filtered set's size correctly yields all-NaN days. Full suite: 20/20 passed.
- [ ] 0.4 **Honest teacher validation:** `ranker_baseline.py` → time split (train ≤ cutoff,
      test after); walk-forward IC per window.
      - **Tests** (`tests/agent/test_ranker_time_split.py`, fast group): *problem* —
        synthetic panel with a purely contemporaneous common-shock feature (predictive
        within a period across tickers, zero persistence): assert the OLD row-split scores
        it as high-IC while the time split scores ≈ 0 — this is the leak that manufactured
        t=24. *Solution* — a genuinely persistent synthetic signal scores IC > 0 under the
        time split (the honest validator still detects real signal), and train rows are
        strictly ≤ cutoff < all test rows.
- [ ] 0.5 **Power memo + new primary metrics:** minimum-detectable-effect table; implement
      (a) mean daily cross-sectional rank-IC of active tilts (weights − EW) vs forward
      returns, HAC-corrected; (b) selection-residual mean test (attribution.py + 1.5 stats).
      - **Tests** (`tests/agent/test_tilt_ic_metric.py`, fast group, mirrors the 1.5
        calibration style): *calibration under null* — random-tilt strategies on synthetic
        returns must reject at ≈ the nominal rate (the naive version over-rejecting is the
        problem demonstration, exactly like the 1.5 t-test audit). *Power under signal* —
        a synthetic strategy whose tilts = noisy forward-return ranks with known IC 0.05
        must come out significant on ~16y of synthetic days (verifies the metric can
        actually detect the effect size F3 says Sharpe can't).

**Interpretation:** PIT top-50 IC ≈ 0 (< ~0.02, non-overlap t < 2) → failure fully explained,
RL work stops, pivot to feature/universe research (Phase 1b). IC ≥ ~0.03 → proceed.

**Verdict (fill in):** _______

### Phase 1 — Establish the supervised ceiling. ~1 week, CPU. Gated on Phase 0.

**Objective:** the best non-RL portfolio from these features becomes the benchmark RL must
beat and the achievable-alpha estimate (industry order: forecast → construction → then RL).

- [ ] 1.1 **Walk-forward supervised strategy:** per-window time-split teacher → top-k EW
      (k ∈ {10, 20} only), 0.10 cap, 21d rebalance, 10bps costs, turnover band; all 8 PIT
      windows pooled; Phase-0 primary metrics + DSR for the small sweep.
      - **Tests** (`tests/agent/test_supervised_strategy.py`, fast group): turnover band
        unit test (rank change inside band → no trade; outside → trade; costs charged only
        on trades); weights always sum to 1, respect cap, only PIT-active names; no-signal
        synthetic input → strategy degenerates to ≈EW net of costs (guards against the
        construction itself manufacturing returns).
- [ ] 1.2 **Alpha waterfall:** long–short decile paper portfolio → long-only top-k → +cap →
      +costs → +21d holding, per window. Empirical transfer coefficient; explains the
      "positive IC but ranker loses" paradox mechanically.
      - **Tests** (same file): on a synthetic panel with a known planted IC, assert the
        waterfall is monotone non-increasing across stages and the long–short stage
        recovers the planted signal (positive spread); with zero planted signal every
        stage ≈ 0 minus costs. This validates the diagnostic instrument before its verdict
        is trusted — the waterfall IS the problem-vs-solution test for "where does IC die."
- [ ] 1.3 **Factor sanity check:** regress daily selection residual on market/size/momentum
      factors built from own data — is "alpha" a persistent style tilt?
      - **Test:** synthetic strategy constructed as pure size tilt → regression attributes
        ≈ all of its "alpha" to the size factor with residual ≈ 0 (instrument check).

**Interpretation:** supervised OOS ≤ 0 → signal doesn't survive construction+costs on liquid
names; root cause = insufficient tradable signal → Phase 1b (cross-sectional rank features,
horizons, new data, wider PIT universe with liquidity filters) before ANY further RL.
Supervised OOS > 0 → quantified ceiling; RL must beat it or be dropped.

**Verdict (fill in):** _______

### Phase 2 — Make the policy able to express the signal. ~1–2 weeks, GPU. **Gated on Phase 1 > 0.**

All under the 3-seed protocol with `--no-promote`.

- [ ] 2.1 **M6.1 permutation-equivariant policy** (shared per-ticker encoder + market
      pooling, custom SB3 policy). Highest-leverage RL change: parameter sharing across 50
      slots turns ~120 decisions × 50 names into ~6,000 effective samples; BC from the
      honest teacher becomes near-lossless.
      - **Tests** (`tests/agent/test_equivariant_policy.py`, fast group): *problem* —
        permute the ticker slots of an observation and show the CURRENT flat MlpPolicy's
        outputs do NOT permute accordingly (slot-specific weights; documents the broken
        inductive bias). *Solution* — equivariance property: for random permutations π,
        `policy(π(obs)) == π(policy(obs))` within tolerance, including mask and
        prev-weights channels; plus BC-fidelity smoke: after BC on a synthetic teacher,
        per-ticker score order matches the teacher's.
- [ ] 2.2 **Action reparameterization (Option C):** per-ticker scores → deterministic tilt
      rule (EW·(1+κ·cs-z(score)), capped, renormalized) or Dirichlet head. Removes the
      logit_scale saturation pathology entirely.
      - **Tests** (extend `tests/agent/test_env_basic.py`): *problem* — reuse the M5
        measurement as a replay assert: at logit_scale=10, a modestly differentiated score
        vector produces pre-cap top-1 weight > 0.8 (near-argmax saturation). *Solution* —
        tilt rule maps zero scores → exactly EW; is monotone in score; output on the capped
        simplex (sums to 1, per-stock ≤ cap, inactive = 0); κ sweeps concentration smoothly
        with pre-cap top-1 staying in a usable band (~0.1–0.6) — the acceptance band 2.3
        defined but temperature alone couldn't hit.
- [ ] 2.3 **Cross-sectional obs normalization (6.2):** per-date rank/z channels — matches
      how the scale-invariant tree teacher reads the features.
      - **Tests** (extend `tests/agent/test_feature_engineering.py`): rank/z computed
        within-date over active names only (no leakage across dates, NaN-safe); a global
        level shift applied to all names on one date leaves that date's rank channels
        unchanged (regime drift can't swamp cross-section — the problem being fixed).
- [ ] 2.4 Hygiene in the same retrains: random episode starts; per-env seeds; add `selic` +
      one regime feature to obs (or fix cash_mode to capped/no-cash and stop expecting
      timing skill).
      - **Tests** (extend `tests/agent/test_env_basic.py`): *problem* — two `reset()`s of
        the current env yield the identical start index (documents the deterministic-path
        issue). *Solution* — with random starts enabled: start indices vary across
        seeds/resets, episodes never overrun the split's bounds, and `date_range="test"`
        evaluation stays deterministic from day 0 (random starts must be train-only, or
        every backtest number becomes seed-dependent).

**Success:** equivariant BC-init ≈ teacher portfolio (sanity); PPO-final ≥ teacher on
primary metrics across ≥3 seeds.

**Verdict (fill in):** _______

### Phase 3 — Validation & selection hardening (parallel with Phase 2, small).

- [ ] 3.1 Measure **val→test rank correlation of checkpoints** (6.4) from existing JSONL;
      if ≈0, early stopping selects regime luck → purged/embargoed multi-fold validation.
      - **Test:** analysis first (no unit test); if purged multi-fold validation is then
        built, it gets the standard purging test — no train fold sample within the embargo
        of any val sample (synthetic dates, assert on indices).
- [ ] 3.2 λ reward-shaping decomposition audit from training logs (TODO 3.2).
      - **Test:** decomposition identity check in `test_env_basic.py` — logged
        (raw excess, penalty, cost) recompose to the step reward exactly.

### Testing policy for this plan

Every code change above ships with two kinds of plain-script asserts (repo convention —
no pytest, register each new file in `tests/run_all.py`, fast group unless it needs data):
1. a **problem test**: reproduce the failure mode on a small synthetic input — where the
   defect is in *current* code (F1 selector lookahead, ranker row-split leak, flat-MLP
   non-equivariance, softmax saturation, deterministic resets), write the assert against
   current behavior first and flip it with the fix, same pattern as
   `test_early_stopping.py`'s replay of the real trajectories that motivated it;
2. a **solution test**: the property the fix guarantees (future-blindness, time-split
   honesty, equivariance, capped-simplex tilt mapping, null-calibration of the new
   metrics), so it can't silently regress.
Analyses (0.2, 1.2's verdict, 3.1, the power memo) aren't unit-testable — their
"test" is the instrument-validation asserts above plus the recorded table in this file.

### Phase 4 — The verdict memo. ~1 day.

- [ ] Compare on PIT data with powered metrics, across seeds: supervised vs BC-init vs
      PPO-final. Write the explicit go/no-go:
      1. RL > supervised robustly → scale up RL.
      2. RL ≈ supervised → ship supervised as the product; RL only as a bounded experiment
         where it's uniquely suited (multi-period cost-aware trading/execution).
      3. Everything ≈ 0 → features/universe don't support alpha; documented negative →
         redirect to data (new features, wider PIT universe, delisted-company data).

---

## Deliberately NOT doing

- No reward re-redesign (M3.1 is sound and test-covered).
- No hyperparameter tuning (stays banned until the diagnosis lands).
- No new RL algorithms (SAC/TD3) — evidence points at representation/data, not optimizer.
- No autoencoder/embedding preprocessing — unsupervised compression preserves variance,
  not alpha; the supervised teacher is already the strong version of that idea.

## References

- Grinold & Kahn, *Active Portfolio Management* — IR ≈ IC·√breadth·TC.
- Clarke, de Silva, Thorley (2002) — transfer coefficient ≈ 0.3 for long-only + caps.
- Bailey & López de Prado — Deflated Sharpe Ratio; Sharpe SE ≈ 1/√T-years.
- López de Prado, *Advances in Financial ML* — purged CV, backtest overfitting, survivorship.
- Harvey, Liu & Zhu (2016) — multiple-testing haircuts.
