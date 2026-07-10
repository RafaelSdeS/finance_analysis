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
  but its **non-overlap t-stat collapses from 1.91 to 0.51**.
- **Measured in Phase 0.4 (below), and this is the more surprising half:** fixing the
  row-split leak itself (genuine walk-forward validation, honest per-window time split)
  does NOT collapse the signal on the full universe — mean |IC| 0.083, t~20 per window,
  every one of 8 windows individually significant. **Revised framing: the signal is real
  and temporally persistent; the problem is specifically cross-sectional breadth (~50 vs
  ~288 names), not a lookahead artifact.** F2 is an instance of F3's power problem
  showing up in signal detection, not evidence the signal doesn't exist.

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
- **Built and confirmed in Phase 0.5:** `metrics.tilt_ic_significance` (new), HAC-tested.
  Derived MDE (≈0.049 IC) lands almost exactly on 0.3's measured tradable-universe
  magnitude (0.047) — and a synthetic test at the project's real scale (190 rebalances ×
  50 names) with a planted IC=0.05 comes back p=2.2e-6, unambiguously significant. The
  formula and the built instrument agree: this is a resolvable effect size, the old
  metric just couldn't see it.

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
- [x] 0.4 **Honest teacher validation:** `ranker_baseline.py` → time split (train ≤ cutoff,
      test after); walk-forward IC per window.
      - **Done (2026-07-10):** `compute_ranker_ic(df, horizon, split="time"|"row")` — new
        default `"time"` splits by calendar date cutoff (no overlap); `"row"` (old
        behavior) kept only for comparison, documented as leaky. Refactored the function
        into `_compute_forward_returns`/`_select_features`/`_score_predictions` helpers
        (feature-selection now correctly uses TRAIN rows only, a small additional
        lookahead fix beyond the split itself — the old code picked "valid" feature
        columns using the full train+test array). `tools/ranker_walk_forward_ic.py`
        applies the honest split per the RL pipeline's own anchored rolling windows
        (`config.generate_windows`) instead of one global 50/50 split.
      - **Result — surprising, and revises F2's original framing:** the row-split leak is
        real (proven on synthetic data below) and worth fixing on principle, but on the
        actual dataset, fixing it does **not** collapse the signal:
        ```
        window  test span        n_train    n_test   mean_ic  t_stat
        0       2010→2012        104,405     58,192   0.064    13.6
        1       2012→2014        148,472     63,083   0.054    11.4
        2       2014→2016        198,095     64,949   0.068    16.8
        3       2016→2018        251,333     68,219   0.085    17.0
        4       2018→2020        306,777     73,329   0.096    23.3
        5       2020→2022        362,532    100,160   0.107    21.4
        6       2022→2024        422,708    121,055   0.118    33.2
        7       2024→2026        487,474    126,250   0.073    24.0
        mean |IC| = 0.083, mean t-stat = 20.1 — every one of the 8 walk-forward windows
        individually clears conventional significance, several by a wide margin.
        ```
        Full table: `artifacts/backtest/ranker_walk_forward_ic.json`.
      - **Interpretation:** on the full ~288-name universe, the honest walk-forward IC
        (0.083, t~20/window) is at least as strong as the original row-split's pooled
        t=24 — the earlier number wasn't fabricated by the leak after all, at least not on
        this dataset. Combined with 0.3's result (same date ranges, restricted to the
        tradable top-50: IC 0.047, t collapses to 0.51), the full picture is now clearer
        than either result alone: **the feature→return relationship is genuinely real and
        temporally persistent — the problem is specifically breadth (~50 vs ~288 names per
        cross-section), not a lookahead artifact.** This means F2 was half right: the
        row-split methodology WAS broken (confirmed below) and shouldn't be trusted as
        written, but the practical worry it raised — "maybe there's no real signal at
        all" — is not what's actually going on. Reframes Phase 1's job: build the
        supervised ceiling on a walk-forward-honest basis (now available via this tool),
        and treat the tradable universe's narrow breadth, not signal existence, as the
        thing to design around.
      - **Tests** (`tests/agent/test_ranker_time_split.py`, fast group, 4 cases, synthetic
        24-ticker/120-day regime-shift panel): *problem* (Test 1) — a feature that ranks
        returns perfectly in "era A" (first half of dates) but is exactly uncorrelated
        (hand-verified zero-Spearman derangement) in "era B" (second half): the row split
        reports mean_ic=0.51, t=12.6 — a spuriously strong "signal", because its test set
        still spans era A via held-out tickers. *Solution* (Test 2) — the time split's
        test set is confined to era B only (the true out-of-sample regime) and reports
        mean_ic=**exactly 0.0000**, matching the hand-derived value to the decimal —
        honestly exposing the lack of forward skill the row split hid. (Test 3) train ≤
        cutoff < test strictly enforced; (Test 4) unknown `split` value raises
        `ValueError`. Full suite: 21/21 passed.
- [x] 0.5 **Power memo + new primary metric.**
      - **Done (2026-07-10):** `src/agent/metrics.py` gains `daily_tilt_ic(weights,
        ew_weights, fwd_returns, active, min_names)` — daily cross-sectional rank-IC of
        ACTIVE TILTS (`weights − ew_weights`) vs forward returns, i.e. "does the strategy
        systematically overweight names that go on to outperform" — and
        `tilt_ic_significance(...)`, which HAC-tests the resulting series via the existing
        `hac_mean_test` (reused from 1.5, `lag` defaults to 1 rebalance-step since
        `daily_tilt_ic` already returns one value PER REBALANCE decision, not per trading
        day — consecutive entries are already `rebalance_interval_days` apart). Selection-
        residual significance testing (b) doesn't need new code: `attribution.py`'s
        `selection_residual_daily` stream is already a plain array, so it slots directly
        into the existing `hac_mean_test`/`block_bootstrap_mean_ci` from 1.5 — no new
        function required, just a usage note (see below).
      - **Power memo:**
        ```
        Metric                          Effective N            SE formula        MDE (80% power, 5% 2-sided)
        Time-series excess Sharpe       T_years                1/sqrt(T_years)   2.8x SE
          per 2y window                 2                      0.71              ~2.0 Sharpe
          pooled 16y (8 windows)        16                     0.25              ~0.70 Sharpe
        Cross-sectional tilt-IC         n_rebalances (~190      std_ic/sqrt(n)   2.8x SE
          (16y @ 21d rebalance,          @ 16y/21d, N=50/date)
           pooled across windows)                               ~0.24/sqrt(190)   ~0.049 IC
                                                                  = 0.017
        ```
        The Sharpe row restates F3 (theoretical ceiling ≈0.2–0.4 Sharpe is below what even
        16 pooled years can resolve). The tilt-IC row's `std_ic≈0.24` is the actual daily
        rank-IC std from this project's own data (`artifacts/backtest/ranker_metrics.json`,
        the single-feature IC run) — not a generic assumption. **MDE ≈0.049 lands almost
        exactly on 0.3's measured tradable-universe magnitude (mean |IC|=0.047)** — the new
        metric is sized, by construction of the actual numbers, to resolve precisely the
        effect Phase 0 spent three steps establishing is real.
      - **Empirical confirmation, not just the formula:** `test_tilt_ic_metric.py`'s power
        test builds a synthetic strategy at this project's actual scale (190 rebalance
        dates x 50 names) with a *planted* IC of 0.05 and recovers mean_ic=0.0512,
        **p=2.2e-6** — unambiguously significant, where a time-series Sharpe test at the
        same effective sample size (F3) could not even reliably detect a Sharpe 10x
        larger than the theoretical ceiling. This is the concrete demonstration that the
        evaluation pipeline's *instrument*, not just the underlying strategy, was the
        bottleneck for detecting skill at this effect size.
      - **Adoption note for Phase 1+:** `tilt_ic_significance` needs the SAME (weights,
        ew_weights, forward-return) triple `evaluate.py`'s `rollout()` already produces
        per backtest — wiring it into `evaluate.py`/`rolling_eval.py` as a headline metric
        (alongside excess Sharpe, not replacing it) is a small follow-up, not a new
        research question; deferred to whichever of Phase 1/2 first needs a real backtest
        scored by it, so it isn't built speculatively ahead of a caller.
      - **Tests** (`tests/agent/test_tilt_ic_metric.py`, fast group, mirrors the 1.5
        calibration style, 3 cases): *calibration under null* (Test 1) — 300 trials of
        random tilts independent of forward returns; false-positive rate 5.3% (nominal
        5%). *Power under signal* (Test 2, above) — plants Pearson rho=0.05 via a
        `a*standardized_fwd + b*noise` mixture (Spearman≈Pearson for small rho on
        jointly-Gaussian data), recovers mean_ic=0.0512, p=2.2e-6, all 190/190 days valid.
        (Test 3) `min_names`/`active` masking honored — under-threshold days are NaN, not
        silently included or zero-filled. Full suite: 22/22 passed.

**Interpretation:** PIT top-50 IC ≈ 0 (< ~0.02, non-overlap t < 2) → failure fully explained,
RL work stops, pivot to feature/universe research (Phase 1b). IC ≥ ~0.03 → proceed.

**Verdict (2026-07-10): PROCEED to Phase 1.** Tradable-universe mean |IC| = 0.047 —
above the ≥0.03 bar, and the standalone tilt-IC power test (0.5) confirms a signal this
size is detectable given real breadth (p=2.2e-6 on a planted IC=0.05 at the project's
actual scale). Full synthesis across all 5 sub-phases, revised twice from the original
hypotheses as evidence came in:
- **F1 (survivorship/lookahead):** real, but NOT a uniform inflation — full-sample
  selection understates EW Sharpe in 6/8 windows; the problem is contaminated
  *strategy identity* in early windows, not one-directional overstatement. Windows 2–7
  (where the agent spends most of its budget) are barely affected (PIT vs full-sample
  universes differ in only 3/50 names).
- **F2 (universe-mismatched IC + row-split leak):** the row-split leak is real and
  demonstrated on synthetic data (0.51 spurious IC vs exact 0.0 honest), but on the
  ACTUAL dataset an honest walk-forward split does not collapse the signal — full
  universe mean |IC|=0.083, t~20/window, more robust than the original flawed number.
  Restricted to the tradable top-50, IC survives in magnitude (0.047) but loses
  statistical confidence (t 1.91→0.51) purely from reduced breadth (~50 vs ~288 names).
- **F3 (power):** confirmed and now instrumented — the tilt-IC test's derived MDE
  (≈0.049) lands almost exactly on F2's measured tradable-universe magnitude (0.047),
  and empirically detects a same-sized planted signal at p=2.2e-6.
- **Bottom line:** there IS real, temporally-persistent, economically-sized signal
  (~0.03–0.10 IC) in this project's features. It was never provably absent — it was
  under-measured (full-universe IC, leaky validation) and under-tested (time-series
  Sharpe lacks the power to see it on ~50 names). Phase 1 asks the next honest
  question: does that signal survive actual portfolio construction (caps, costs,
  21-day holding) on the universe the agent actually trades?

### Phase 1 — Establish the supervised ceiling. ~1 week, CPU. Gated on Phase 0.

**Objective:** the best non-RL portfolio from these features becomes the benchmark RL must
beat and the achievable-alpha estimate (industry order: forecast → construction → then RL).

- [x] 1.1 **Walk-forward supervised strategy:** per-window time-split teacher → top-k EW
      (k ∈ {10, 20} only), 0.10 cap, 21d rebalance, 10bps costs, turnover band; all 8 PIT
      windows pooled; Phase-0 primary metrics + DSR for the small sweep.
      - **Done (2026-07-10):** new `src/agent/supervised_strategy.py`
        (`select_topk_weights`, `backtest_topk_strategy`) — deliberately does NOT reuse
        `PortfolioEnv`: a top-k equal-weight target with k≥10 never actually triggers the
        0.10 cap (1/10 = 0.10 exactly, 1/20 < 0.10), so a direct, independently-implemented
        backtest is simpler and doubles as a second check on env.py's portfolio math
        rather than depending on it. `tools/supervised_strategy_backtest.py` trains a
        HistGradientBoostingRegressor teacher per window on TRAIN-span PIT-active features
        only (raw/unscaled — tree models are scale-invariant, matching the 0.3/0.4
        convention), backtests k∈{10,20} **and an EW-of-all-PIT-active baseline (k=9999,
        i.e. `select_topk_weights` naturally degenerates to full EW when k exceeds the
        active count)** on the TEST span, pools all 8 windows, scores with `tilt_ic_significance`
        (0.5) + HAC/bootstrap + DSR. Two real bugs caught before real numbers came out: (a)
        forgot to filter constant/all-NaN feature columns per window — reused
        `ranker_baseline._select_features` (0.4) rather than reinventing it, since early
        PIT windows have thin enough history to hit exactly this; (b) the module's first
        draft made the very first allocation free ("no cost on discovery"), inconsistent
        with `env.py`'s own documented convention that episodes start at 100% CASH and the
        first deployment IS a real, costed decision — fixed to match.
      - **Result — the fair comparison is the headline, and it's a clear loss for the
        supervised strategy:**
        ```
                    cum_return  sharpe   HAC p    tilt_ic   tilt_ic p   DSR
        k10 (top-10)   3.71     0.374    0.151    -0.013    0.279       0.877
        k20 (top-20)   4.92     0.502    0.061    -0.008    0.505       0.951
        EW (all-active) 6.23    0.607    0.021    +0.041    0.243       0.979
        ```
        (pooled over all 8 PIT windows, 21d rebalance, 10bps costs, turnover band on;
        tilt_ic_n_days: 192 for k10/k20, only 44 for EW — expected, since EW's own tilt
        vs. its own benchmark is near-degenerate most days by construction.) Full table:
        `artifacts/backtest/supervised_strategy.json`.
      - **Interpretation:** plain equal-weight-of-the-PIT-active-universe *beats both*
        top-10 and top-20 teacher-selected portfolios, on both raw return and Sharpe, over
        the pooled 16-year walk-forward test — and EW is the only one of the three whose
        excess-of-zero clears conventional significance (HAC p=0.021). The teacher's
        tilt-IC is indistinguishable from zero (even slightly negative) for both k10/k20.
        This is the honest answer to Phase 1's question — "does the real signal (0.3/0.4)
        survive actual portfolio construction on the tradable universe?" — **not yet, at
        least not through this construction.** A plausible mechanism: concentrating into
        10–20 (of up to 50) names churns much more aggressively on rank reshuffles than a
        near-full EW book, so turnover-band-limited but still-frequent retargeting drags
        on returns disproportionately for the more concentrated portfolios — exactly the
        transfer-coefficient erosion Clarke–de Silva–Thorley predict for long-only,
        concentrated construction. Whether the loss is COSTS, the LONG-ONLY/top-k
        discretization itself, or the 21-day HOLD is what 1.2's waterfall isolates next.
        Caveat: the turnover band used here is the simplest possible (any set change
        triggers a full retarget); a more forgiving band (e.g. hold-vs-buy buffer) is an
        easy follow-up lever if 1.2 points at turnover as the dominant loss.
      - **Tests** (`tests/agent/test_supervised_strategy.py`, fast group, 8 cases):
        `select_topk_weights` invariants (sums to 1, respects cap by widening the selected
        count rather than iterative redistribution, only active names, empty-active
        doesn't crash). Turnover band — checked via COST directly (not aggregate P&L,
        which is confounded by periodic-reset-to-EW having its own return effect
        independent of cost): the band charges cost only on the set-changing first
        decision, zero on 2 subsequent same-set decisions; without the band every decision
        costs something even with an unchanged set; a genuine set change (ticker swap)
        correctly triggers a new target. No-signal degeneracy: random scores' mean daily
        return tracks full-EW's mean closely (doesn't manufacture an edge from noise).
        Portfolio invariants: daily weights always sum to 1; the cap binds at REBALANCE
        time (matching env.py exactly) but drifted weights may exceed it between
        rebalances by design. Full suite: 23/23 passed.
- [x] 1.2 **Alpha waterfall:** long–short decile paper portfolio → long-only top-k → +cap →
      +costs → +21d holding, per window. Empirical transfer coefficient; explains the
      "positive IC but ranker loses" paradox mechanically.
      - **Done (2026-07-10):** `compute_alpha_waterfall` added to `supervised_strategy.py`,
        reusing `select_topk_weights`/`backtest_topk_strategy`. 5 stages, all reported as
        mean PERIOD excess return vs EW-of-active (same units, directly comparable):
        long_short → long_only → long_only_capped → plus_costs → plus_holding. Stages 1–4
        are strictly nested degradations (monotone non-increasing expected); stage 5 uses
        the turnover band (a genuinely different, cost-*saving* mechanism vs stage 4's
        always-reset assumption) so it isn't chained into the same monotonicity claim — a
        synthetic test with an unchanging top-k set across periods actually showed
        stage 5 > stage 4 (the band correctly saving cost when nothing needs to trade),
        which is the band working as designed, not a defect. `tools/alpha_waterfall.py`
        runs it per window on real data, reusing 1.1's exact per-window teacher (same
        predictions, not a separately-fit model) — but with **k=5, not 1.1's 10/20**,
        specifically so the 0.10 cap is actually triggered (1/5=0.20 > cap, forcing
        `select_topk_weights` to widen the selection to 10 names — 1.1's k≥10 never
        triggered this stage at all).
      - **Result — mechanically explains 1.1, and pinpoints the specific killing blow:**
        ```
        stage               pooled (n_periods-weighted mean excess return/period)
        long_short                  +0.00245   (raw signal, both tails, no constraints)
        long_only                   +0.00059   (drop the short leg: -76%)
        long_only_capped            -0.00011   (0.10 cap widens 5->10 names: flips negative)
        plus_costs                  -0.00121   (10bps costs: further erosion)
        plus_holding                -0.00125   (realistic 21d drift + turnover band)
        ```
        Full per-window table: `artifacts/backtest/alpha_waterfall.json`.
      - **Interpretation:** the raw signal (long-short, +0.0025/period) is real but small,
        and is destroyed in two identifiable steps, not gradually: (1) dropping the short
        leg alone removes ~76% of it — expected for a signal that isn't dramatically
        stronger in one tail than the other; (2) **the 0.10 position cap is the specific
        stage that flips the sign** — for k=5 (a genuinely differentiated, high-conviction
        selection), the cap forces `select_topk_weights` to widen to 10 names to satisfy
        1/n≤0.10, and that dilution costs more edge than costs or holding do combined
        (cap step: -0.00070; costs+holding step: -0.00114, i.e. comparable magnitude, but
        the cap step is where the sign genuinely flips from positive to negative). This is
        the empirical transfer-coefficient loss the plan's references (Clarke–de Silva–
        Thorley) predict for long-only, position-constrained construction — quantified,
        not just cited. It also reconciles with 1.1: even at k=10/20 (where the cap
        never technically binds), the SAME underlying mechanism — a small raw edge that a
        long-only, capped, costed construction has very little room to express — explains
        why neither concentration level beat plain EW.
      - **Tests** (extended `tests/agent/test_supervised_strategy.py`, fast group, 2 cases):
        *solution* (Test 9) — a strong, hand-constructed monotone planted signal (scores
        exactly rank returns every period): long-short recovers a clearly positive spread,
        stages 1–4 decay monotonically, stage 5 stays positive (the planted signal survives
        holding when the set is stable, as expected). *problem* (Test 10, instrument
        validation) — pure noise scores and returns: long-short/long-only sit near zero
        (< 0.05 magnitude), and costs measurably pull the later stages below the pre-cost
        capped stage — confirms the instrument doesn't manufacture a signal from nothing
        before trusting its verdict on real data. Full suite: 23/23 passed.
- [x] 1.3 **Factor sanity check:** regress the waterfall's long-short signal on
      market/size/momentum factors built from own data — is "alpha" a persistent style tilt?
      - **Done (2026-07-10):** `regress_on_factors(strategy_returns, factor_returns)` added
        to `supervised_strategy.py` — plain OLS via `np.linalg.lstsq`, reports alpha
        (intercept), betas, R², and the residual series. `tools/factor_sanity_check.py`
        reuses 1.1/1.2's exact per-window teacher (same predictions again — third reuse of
        the same trained model, not three separately-fit ones) to get the long-short
        strategy's period-return series, and builds two INDEPENDENT factor-portfolio
        return series from the SAME PIT-active universe each period: size (SMB-style:
        smallest-market-cap decile return − largest-market-cap decile return) and momentum
        (WML-style, using the already-existing `momentum_vs_market_3m` state feature:
        highest-momentum decile − lowest-momentum decile). Regresses the pooled long-short
        series on both.
      - **Result:**
        ```
        n_periods: 180        strategy mean return: +0.00291
        R-squared (size + momentum): 0.2018
        beta_size: +0.133     beta_momentum: -0.291
        alpha (residual, i.e. NOT explained by size/momentum): +0.00125
        ```
        Full output: `artifacts/backtest/factor_sanity_check.json`.
      - **Interpretation:** moderate, not dominant, factor exposure — R²=0.20 means size
        and momentum jointly explain about a fifth of the long-short signal's period-to-
        period variance, leaving roughly **43% of the raw mean signal (0.00125 of 0.00291)
        as residual alpha** not attributable to either factor. The negative momentum beta
        (-0.29) is consistent with the feature set's composition: several of the model's
        top-ranked fundamental features (ROE, ROIC, quality/value scores) are classic
        value/quality signals, which are typically mildly anti-momentum (buying
        fundamentally strong names that haven't necessarily just rallied). **Verdict: the
        long-short signal is not simply a repackaged size or momentum trade — but it also
        isn't fully independent of them, and this check only screens two of the many
        possible style factors (no explicit value or quality factor was built, since
        several of the model's own inputs already ARE value/quality features — testing
        against them would be closer to circular than informative).** Combined with 1.1/1.2
        this doesn't change the practical conclusion (the raw signal, whatever its
        composition, doesn't survive long-only/capped/costed construction today) but does
        rule out the most common "it's not real, it's just a known factor" objection to
        the raw signal itself.
      - **Test** (extended `tests/agent/test_supervised_strategy.py`, fast group, 2 cases):
        *problem/instrument check* (Test 11) — a strategy constructed as a NOISELESS pure
        linear function of one factor (`1.7 × size_factor`) regresses to R²≈1.0000, beta
        recovered exactly (1.7000), residual_mean≈0 (1e-19) — confirms the instrument
        correctly attributes a pure tilt as a pure tilt. *solution* (Test 12) — a strategy
        with a real mean shift (+0.003) but statistically independent of the factor:
        R²=0.0002 (correctly low), alpha (intercept) recovers the true +0.003 mean almost
        exactly — confirms genuine, factor-independent alpha is NOT explained away by an
        unrelated regressor. Full suite: 23/23 passed.

**Phase 1 exit criterion — met.** Verdict: **the raw signal is real (0.3/0.4), modest in
size, largely independent of size/momentum (1.3), and does not currently survive long-only/
capped/costed portfolio construction (1.1) — specifically because the 0.10 position cap
forces a dilution that costs more than transaction costs and holding combined (1.2).** This
is a quantified, mechanistic answer, not a guess: Phase 2's RL work (permutation-equivariant
policy, action reparameterization) is worth pursuing only if it can express **calibrated,
low-turnover, graduated conviction** that a rigid top-k/cap-then-dilute construction cannot
— which is exactly what M6.1/Option C were already designed to attempt, now with a concrete
mechanism (not just "the agent hasn't learned to size bets") to test against.

**Interpretation:** supervised OOS ≤ 0 → signal doesn't survive construction+costs on liquid
names; root cause = insufficient tradable signal → Phase 1b (cross-sectional rank features,
horizons, new data, wider PIT universe with liquidity filters) before ANY further RL.
Supervised OOS > 0 → quantified ceiling; RL must beat it or be dropped.

**Verdict (fill in):** _______

### Phase 2 — Make the policy able to express the signal. ~1–2 weeks, GPU. **Gated on Phase 1 > 0.**

All under the 3-seed protocol with `--no-promote`.

- [x] 2.1 **M6.1 permutation-equivariant policy** (shared per-ticker encoder + market
      pooling, custom SB3 policy). Highest-leverage RL change: parameter sharing across 50
      slots turns ~120 decisions × 50 names into ~6,000 effective samples; BC from the
      honest teacher becomes near-lossless.
      - **Done (2026-07-10):** new `src/agent/policy.py`. `SharedTickerEncoder`: the SAME
        small MLP applied to every ticker slot (batched matmul, no cross-slot mixing),
        each slot's input augmented with a masked-mean/std cross-sectional context vector
        (a symmetric function of the other active slots, so it's order-independent too) —
        this is what lets the shared weights judge a stock RELATIVE to that day's market,
        not just in absolute terms. `EquivariantMlpExtractor` mimics SB3's `MlpExtractor`
        interface (`policy_net`, `value_net`, `forward_actor`, `forward_critic`,
        `latent_dim_pi`, `latent_dim_vf`) so `bc_pretrain.py`'s existing BC code works
        **completely unmodified** against it. `EquivariantActorCriticPolicy` wires it into
        SB3, additionally replacing SB3's default `action_net` — a full `Linear(n_tickers,
        n_tickers)` that would silently remix every ticker's latent into every output
        logit and destroy equivariance — with `nn.Identity()`, since the extractor's actor
        head already outputs per-ticker logits directly. Wired into `trainer.py` via a new
        `--policy {MlpPolicy,equivariant}` flag (default unchanged), threaded through
        `rolling_eval.run_rolling_eval`/`train_window`.
      - **Verified at three levels:** (1) pure-PyTorch — actor path exactly equivariant
        (max error 7e-9), critic path exactly invariant (1e-8), corrupting an INACTIVE
        slot's features changes zero active-slot outputs (context vector genuinely ignores
        masked names, error 0.0 exactly), architecture is trainable (loss drops >99.99% on
        a synthetic per-ticker target). (2) SB3 integration — `PPO` constructs with the
        custom policy against `PortfolioEnv`'s real spaces, `action_net` confirmed
        `Identity`, deterministic `predict()` stays exactly equivariant end-to-end through
        the full SB3 stack (4e-7), and `bc_pretrain.pretrain_policy`/`pretrain_value` run
        completely unmodified and measurably update the new architecture's parameters.
        (3) **Real pipeline smoke run** — `python -m src.agent.trainer --train-years 2
        --test-years 1 --timesteps 4096 --universe-size 50 --policy equivariant
        --bc-pretrain --no-promote`: all 24 windows trained end-to-end (BC-pretrain, PPO,
        val eval, checkpointing, final report) without a single crash.
      - **Incident during the smoke run, disclosed and fixed:** the run (correctly,
        `--no-promote`) skipped model promotion and `walkforward_*`, but still silently
        **overwrote the shared, gitignored `artifacts/models/rolling_eval_results.json`**
        — a pre-existing bug (that write was never gated by `promote`, unlike
        `save_walkforward`, which had already been fixed for the exact same failure mode
        once before per its own code comment: "caught happening twice... before this got
        gated here too" — this one instance was missed). Because the file is gitignored,
        the prior production content (presumably from the 8-window/5M-timestep run
        referenced in TODO.md) is **not recoverable** — only a real production retrain
        restores it. Fixed immediately: the write is now gated behind `promote` in
        `finalize_and_report`, matching `save_walkforward`. Added
        `tests/agent/test_rolling_eval_promote.py` (data group) wrapping this and three
        other previously dead-code self-checks in `rolling_eval.py` (reachable only via
        `python -m src.agent.rolling_eval` with no CLI mode — never run by any test
        runner, which is exactly why this gap went undetected) — including a direct
        regression test for this bug: `promote=False` must not create the file,
        `promote=True` must. Audited every other `config.model_dir`/`config.backtest_dir`
        write in `trainer.py`/`rolling_eval.py` for the same pattern — confirmed this was
        the only unguarded instance.
      - **Tests** (`tests/agent/test_equivariant_policy.py`, fast group, 5 cases + 4 cases
        in `test_equivariant_policy_integration.py`, data group): *problem* — a generic
        flat MLP violates equivariance under a ticker-slot permutation (max error 0.40,
        confirming the broken inductive bias this architecture exists to fix). *solution*
        — exact equivariance/invariance/masking as above, plus the full SB3-level checks.
        Full suite: 26/26 passed.
      - **Scope note:** this validates the architecture is CORRECT and INTEGRATES
        cleanly — it is not yet a full multi-seed comparative training result (that needs
        real timesteps and the 3-seed protocol from TODO.md M7.2, deferred to when Phase 2
        is complete and a proper comparison run is warranted, not before).
- [x] 2.2 **Action reparameterization (Option C):** per-ticker scores → deterministic tilt
      rule (EW·(1+κ·cs-z(score)), capped, renormalized) or Dirichlet head. Removes the
      logit_scale saturation pathology entirely.
      - **Done (2026-07-10):** new `AgentConfig.action_mode` ("softmax" default, unchanged;
        "tilt" new) + `tilt_kappa` (default 2.0). `env.py`'s `action_to_weights` dispatches
        on it; new `_tilt_to_weights`: `weights = cap(normalize(clip(EW · (1 + κ·cs_z(action)), 0)))`,
        where `cs_z` is the cross-sectional z-score of the raw action over active names
        only — the mapping is invariant to whatever raw scale the policy outputs, so there
        is no equivalent to softmax's saturate-once-logit-spreads-exceed-O(1/scale) failure
        mode. `kappa=0` is exactly equal-weight; reuses the existing, already-tested
        `_cap_weights` for the final cap enforcement (shared with the softmax path — same
        redistribution logic, not reinvented). Wired into `trainer.py` via
        `--action-mode {softmax,tilt}` / `--tilt-kappa` (defaults unchanged), and
        `action_mode`/`tilt_kappa` added to `model_provenance.SEMANTIC_FIELDS` (a saved
        model's action output means something different under each mode, exactly the
        criterion that list's own docstring uses to decide membership).
      - **A real, pre-existing bug in `_cap_weights` found and fixed while testing this**
        (not tilt-specific, but the tilt formula's tendency to zero out negative-tilt names
        exposed it): the redistribution loop, when it runs out of uncapped active stocks to
        absorb overflow mid-iteration (e.g. a thin ~10-name early-history window where
        equal-weight itself already sits at the 0.10 cap boundary), silently DROPPED that
        overflow rather than accounting for it — total mass then summed to <1, and the
        function's final unconditional renormalize-the-whole-vector step scaled the
        just-capped stocks back UP, re-violating the cap by construction (measured: 9
        stocks tied at 0.10277, all breaching the 0.10 cap by the same amount). Root cause
        was purely inside the loop, not the final-enforcement step — traced by watching the
        exact iteration where `n_uncapped` hit 0. Fixed by routing genuinely-
        unredistributable overflow to CASH (the only always-uncapped absorber) at the exact
        point it would otherwise vanish — **not** a reintroduction of the M2.1 dead-code
        bug (which sent essentially ALL normal-case overflow to CASH via an always-false
        condition); this only fires on the rare residual left after the stock pool is
        already fully saturated at the cap. Since `_cap_weights` is shared by both action
        modes, this could in principle have bitten the production softmax path too in a
        sufficiently thin/saturated window — it just hadn't been hit by the existing test
        suite.
      - **Verified in the real pipeline:** `python -m src.agent.trainer --train-years 2
        --test-years 1 --timesteps 4096 --universe-size 50 --action-mode tilt
        --bc-pretrain --no-promote`: all 24 windows trained end-to-end without a crash.
        This run also served as a live confirmation that 2.1's `rolling_eval_results.json`
        promote-gating fix works correctly in practice, not just in its unit test (log:
        "Skipping rolling_eval_results.json write (promote=False)").
      - **Known scope boundary, not fixed here:** `bc_pretrain.py`'s BC-target encoding
        (`_weights_to_logits`, a softmax-inverse: `log(w)/logit_scale`) and `evaluate.py`'s
        baseline-weight reproduction both assume `action_mode="softmax"`; under `"tilt"`
        they'd feed a softmax-shaped target into a z-score-based interpreter, giving a
        miscalibrated (not crashing, just not exactly-target) BC warm-start or baseline. A
        real tilt-mode training/comparison run should either skip `--bc-pretrain` or accept
        an imperfect warm-start (PPO still fine-tunes from there), and should roll baselines
        through a separately-configured `action_mode="softmax"` env (same tensors/costs,
        different action interpretation) for a fair comparison. Not attempted here —
        genuinely out of 2.2's scope (giving the AGENT a non-saturating action space),
        deferred to whichever future step first runs a real tilt-mode comparison.
      - **Tests** (extended `tests/agent/test_env_basic.py`, data group, 6 new checks +
        1 dedicated regression case): *problem* — replays the M5 measurement directly:
        softmax(action×10) on a modestly differentiated score (not an extreme one-hot)
        saturates to pre-cap top-1 0.625 at a representative ~38-active-name date.
        *solution* — tilt keeps top-1 at 0.094 (usable band) on the identical input; zero
        action maps to EXACTLY equal-weight; monotone in score (w(-1)=0.0000 <
        w(+1)=0.0732); κ sweep {0,1,2,4,8} → top-1 {0.026, 0.072, 0.094, 0.10, 0.10}, smooth
        and monotone, saturating gracefully at the cap rather than an exponential blowup.
        Dedicated regression case locks in the `_cap_weights` fix at the exact thin-universe
        date that found it (max weight 0.1000, was 0.10277). Full suite: 26/26 passed.
- [x] 2.3 **Cross-sectional obs normalization (6.2):** per-date rank/z channels — matches
      how the scale-invariant tree teacher reads the features.
      - **Done (2026-07-10):** new `data_pipeline.cross_sectional_transform(feat_tensor,
        mask, method)` — per-date mean-0/std-1 z-score or [-1,1] percentile rank, computed
        ONLY over that date's active names, fully vectorized (no Python loop over dates or
        features). New `AgentConfig.cross_sectional_features` (default False, unchanged)
        + `cross_sectional_method`. Required separating `base_state_features` (real
        parquet columns, used for the data load) from `state_features` (the tensor's final
        channel list, extended with `f"{feat}_cs"` names when the flag is on) — macro
        features (e.g. `ipca`) are excluded from the cross-sectional set since they're
        identical across every ticker on a given date, making their cross-section
        degenerate/undefined by construction. Wired into `build_tensors()`: computes the
        transform on the raw price+fundamental channels and concatenates it, so
        `env.py`/PortfolioEnv need zero changes (obs_dim, feature count, etc. are already
        fully generic over `n_features`). CLI: `python -m src.agent.data_pipeline
        --cross-sectional-features [--cross-sectional-method rank]`.
      - **Verified against real data (in-memory, no disk writes):** 22 base → 43 total
        channels on a 10-ticker smoke universe; active-cell std of the new channels ≈0.92
        (real, non-degenerate spread, not silently zero). New self-check
        `_selfcheck_cross_sectional_features` wired into `test_data_pipeline_universe.py`
        (previously-dead-code-avoidance convention from 0.1, reused here directly).
      - **Tests** (`tests/agent/test_cross_sectional_features.py`, fast group, 7 cases):
        z-score and rank correctness hand-verified against known values (test 1/2 match to
        4 decimals). *Problem vs solution* (test 3) — a market-wide +1000 shock on one
        synthetic date changes a global (whole-history) scaler's output on that date by
        4.42 z-score units, while the cross-sectional transform's output on the SAME date
        changes by 3.5e-05 — **126,254× smaller**, because a uniform shift cancels exactly
        in mean-subtraction. (Test 4) shocking one date leaves every other date's output
        bit-identical (no cross-date leakage). (Test 5) inactive/NaN names are excluded
        from the cross-section's mean/std, not leaked in, and map to exactly 0. (Test 6)
        <2 active names → 0, not NaN/crash. (Test 7) invalid method raises. Full suite:
        27/27 passed.
      - **Scope note (matching 2.1's honest framing):** this validates the transform is
        mathematically correct and integrates cleanly into the existing tensor pipeline —
        it does not rebuild production tensors (would overwrite the shared, gitignored
        `agent_tensors.npz`/`feature_scaler.pkl`) or run a training comparison. Adopting it
        for a real run needs `python -m src.agent.data_pipeline --universe-size 50
        --pit-universe --cross-sectional-features` (rebuilding tensors is an explicit,
        user-facing action, not something to do silently mid-diagnosis) followed by an
        ablation training comparison — deferred to Phase 2.4/4 alongside the other
        pending real-training-run items.
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
