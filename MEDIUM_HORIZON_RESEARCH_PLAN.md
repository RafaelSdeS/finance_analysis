# Medium-Horizon Portfolio Research Plan (H-series, 2026-07-19)

Greenfield program: long-only portfolio over the top-50 dynamic B3 universe at weekly
(k=5), monthly (k=21), or quarterly (k=63) horizons. Successor to the closed M-series
(EIIE/RL, `docs/M4_DECISION_FINAL.md`) and R-series (risk mandate, `RISK_MANDATE_PLAN.md`).

---

## 0. Prior empirical boundaries (evidence, not assumptions)

These are measured results from this repo, and they are stricter than "daily price alpha
is null":

1. **M3 supervised probe was null at k=1, k=5, AND k=21** (val IC ≈ 0, all p ≫ 0.05,
   permutation null 97.5th pct ≈ 0.009). The probe used price/technical channels through
   a conv trunk with listwise loss — maximal power, no RL noise. Conclusion: **price and
   technical features carry no cross-sectional ranking signal at any horizon up to
   monthly.** Not just "no daily alpha" — no price-derived alpha at monthly horizon
   either. Any hope of alpha in this program rests exclusively on the feature families
   M3 never touched: fundamentals, dividends, macro conditioning.
2. **R4: unconstrained risk-only construction loses to BOVA11 outright** (test Sharpe
   0.074 best variant vs 0.24 BOVA11). Min-variance/risk-parity/vol-target with a cash
   escape hatch collapses into the CDI attractor; the fix is structural — fully-invested,
   benchmark-relative construction — not a better optimizer.
3. **Two independent false positives from short windows** (M1's lucky seed, R2's
   "shorter lookback wins" that died on the disjoint window). Any finding from a single
   ~2-year window is presumptively noise.
4. **The current fixed test split (2024-03-22 → 2026-07-14) is ~28 monthly observations**
   — the minimum detectable Sharpe difference on it is enormous. It caused #3. The
   H-series replaces single-split evaluation with a stitched walk-forward OOS spine
   (§H0). `iter_fit_windows()` was built for exactly this seam.

**Honest prior:** in a 50-name large-cap universe this liquid, the base rate that
fundamental/macro cross-sectional signal survives costs is maybe 25–40%. The plan is
therefore front-loaded so the cheapest phase (H1, days of work, zero ML) carries the
kill decision, and every expensive architecture is gated behind it.

---

## 1. Design-space verdicts (decided now, not re-explored later)

### A. Target/label space
- **PRIMARY: forward k-day cross-sectional BOVA11-relative return, k ∈ {21, 63} —
  entering the H2 regression as its cross-sectional normalized rank, not the raw
  percentage.** A single idiosyncratic +50% print (M&A rumor, privatization squeeze)
  under squared-error loss would distort every coefficient in the cross-section;
  rank targets bound this by construction. Raw-return targets winsorized at 1st/99th
  percentile per cross-section are kept as an H2 robustness ablation only. Discovery happens at the IC level (≈90–180 IC observations over the
  walk-forward — enough power to detect IC ≈ 0.03), confirmation at the portfolio level
  (IR vs BOVA11). Discovering at the portfolio level directly is statistically hopeless:
  detecting an IR with t ≥ 2 over ~7y of stitched OOS requires IR ≈ 0.75. IC first,
  portfolio second.
- **Covariance: estimated, never learned.** Ledoit-Wolf shrinkage on daily returns,
  refit per rebalance. Predicting covariance components with ML adds parameters to the
  weakest-signal part of the problem.
- **k=5 (weekly) is deprioritized**: M3 killed price features at k=5, and fundamentals
  update quarterly — there is no feature family left that plausibly resolves at weekly
  frequency. Weekly enters only as a rebalance-frequency ablation in H2, never as a
  prediction horizon.
- **Rejected as targets:** market regimes (unfalsifiable labels; regime enters as a
  conditioning *feature* in H3, not a target), end-to-end weights via differentiable
  loss (see C), factor exposures as targets (they're inputs here, not outputs).

### B. Multi-frequency fusion
**Resample everything to the decision frequency (monthly). Full stop.**
The dual-stream gated network / multi-task multi-horizon head space solves a problem —
mixing frequencies inside one deep sequence model — that only exists if you insist on a
deep sequence model, which the sample size forbids (see C). The lazy correct fusion:
- Daily → monthly: realized vol (21/63d), momentum (63/126/252d skip-21, from
  `adj_close` so it's total-return), Amihud, drawdown — aggregates computed on daily
  data, sampled at each monthly decision date.
- Quarterly → monthly: PIT forward-fill on `fundamentals_available_date` (already
  enforced in Stage 2), **plus a `days_since_filing` information-age feature** — this is
  the entire "information decay modeling" agenda expressed as one column. Note on
  functional form: any monotone transform of it (e.g. exp(−t/τ)) is a **no-op wherever
  the pipeline ranks** — Spearman IC (H1) and rank-normalized features (H2) are invariant
  to monotone maps. The decay transform has bite only in cardinal roles: as a
  **multiplicative freshness gate** on other characteristics (fresh-value vs stale-value
  interaction, H2/H3) it uses f(t) = exp(−t/τ) with **τ fixed a priori at 45 calendar
  days (half a quarter)** — not tuned in H1 (multiplicity), at most a {30, 63}d
  sensitivity check in H2.
- Monthly macro: levels + 3m deltas (Selic direction, IPCA surprise proxy, FX, Brent/iron
  ore returns). Macro is common to all assets, so it enters as interaction/conditioning
  (H3), not as a per-asset feature — a cross-sectional model cannot rank assets on a
  feature they all share.
- All fundamental ratios enter as **cross-sectional ranks** at each decision date
  (kills the |pl| > 400k tail problem without clipping) plus the existing `*_zhist_5y`
  own-history channel where available. **Rank normalization convention (features AND
  targets): rank_i/(N_t + 1) − 0.5 → centered uniform [−0.5, 0.5], where N_t is the
  active eligible universe on that decision date.** The universe is fluid (delistings,
  liquidity drops — some dates have 46 names, not 50); raw ordinal ranks 1…N would make
  feature variance drift with N_t across history, silently distorting the regularized
  weights. Gaussian PIT (Φ⁻¹) is the fallback only if ridge diagnostics show the
  uniform's bounded support hurting fit — don't reintroduce tails without a reason.
- **Distribution yield = total cash distributions, JCP included — already true, verified
  2026-07-19:** the raw dividends files carry `type ∈ {Dividendo, JCP}` and
  `compute_dividend_features()` sums `value_per_share` with no type filter, so
  `div_yield_12m` is already the total-cash-distribution yield. No tax-differential
  parsing, no separate JCP treatment — do not "fix" this.

### C. Modeling framework
Sample-size arithmetic decides this, not taste. ~15y × 12 = ~180 monthly cross-sections
× 50 assets = 9,000 rows, heavily cross-correlated (effective N far lower), and only
~90–100 of those cross-sections are usable OOS.
- **Transformers / Mamba / TCN sequence models: REJECTED.** 10⁵–10⁶ params vs ~10³
  effective samples. M3 already ran the strongest cheap version of this bet (conv trunk,
  daily data, more samples than any monthly model will ever see) and found nothing.
- **RL: REJECTED.** The M-series proved signal detection, not policy optimization, is
  the binding constraint. Supervised IC screening is strictly more powerful per unit
  compute. RL re-enters only if a proven signal ever needs multi-period execution
  scheduling — a problem we do not have.
- **End-to-end DPO (differentiable optimizer as a learning device): REJECTED.**
  Backpropagating Sharpe through a solver over ~180 rebalances fits the optimizer to
  noise, and its gradient signal is exactly the portfolio-level statistic shown above to
  be undetectable. A convex solver as a **fixed construction step** (H2) is fine and is
  not "DPO" — nothing learns through it. Grid search over 2–3 construction
  hyperparameters on validation achieves everything an end-to-end loss would, honestly.
- **Listwise neural rankers / LambdaMART: DEFERRED.** A listwise loss cannot see signal
  that Spearman rank IC cannot; it only re-weights it. Considered in H3 only if trees
  beat linear.
- **ACCEPTED ladder: (H1) single-characteristic rank IC → (H2) regularized linear
  composite on ranks → (H3, conditional) small GBM + discrete macro conditioning.**
  Linear-on-ranks is the maximum model complexity this dataset has demonstrated any
  right to.

---

## 2. Shared evaluation protocol (all phases)

- **Walk-forward spine:** expanding-window refits (initial train ≥ 2011–2018, then
  step annually), each fold scoring only its own OOS year; stitch OOS predictions/
  weights into one continuous series 2019 → 2026 (~90 monthly decision dates). Fit
  windows resolved via `iter_fit_windows()` — no hardcoded split dates. The last
  2024-03 → 2026-07 segment doubles as the "untouched-until-the-end" confirmation
  window: hyperparameter selection uses only pre-2024 folds.
- **Overlap handling:** decisions step monthly; k=63 targets overlap 3×. All IC/return
  t-stats use Newey-West HAC with lag ≥ k; decision gates are additionally checked on
  the non-overlapping subgrid (every 3rd month for k=63).
- **Costs:** the existing per-side cost rate from `configs/` (as in M/R series), always
  reported net; every gate re-checked at 2× costs (R3 precedent).
- **Universe:** point-in-time `top50_universe_membership.parquet`; `status` excluded
  from features (lookahead trap, per CLAUDE.md); entrants get NaN-masked warm-ups, no
  backfill.
- **Multiplicity:** H1 screens ~15–20 characteristics → gates use Benjamini-Hochberg
  FDR at 10%, and the permutation null (shuffle dates, not assets — preserves
  cross-sectional correlation) is the significance floor, as in M3.
- **Reuse:** `metrics.py` (Sharpe/IR/`block_bootstrap_ci`), `run_backtest`,
  the baseline suite, and the R0 `risk_portfolios.py` machinery. No new harness.

---

## 3. Milestones

### H0 — Evaluation harness + power analysis (≈ days, pure code + stats)
**Status (2026-07-19): RUN. Complete.** `src/h_series/{paths,stats,spine,features,
milestone_h0}.py` + `tests/h_series/{test_stats,test_spine,test_features,
test_milestone_h0}.py` (registered in `tests/run_all.py`, 43/43 fast suite green).
Output: `H0_FINDINGS.md` / `H0_FINDINGS.json`.
**Hypothesis:** none — this phase establishes what is *detectable* so later gates are
pre-registered against reality instead of wishes.
- [x] Walk-forward spine implemented over the existing dataset (monthly decision grid,
      PIT feature matrix builder, stitched OOS).
- [x] Baselines on the spine, net of costs: EW-top50 (monthly UCRP), BOVA11,
      classical Ledoit-Wolf mean-variance (long-only, monthly, turnover-penalized) —
      the last one is the "classical multi-period MV" straw man later phases must beat.
      **Result:** stitched OOS = 91 monthly obs (2018-12-31 → 2026-07-14). BOVA11 total
      return 1.069, UCRP 0.678 (underperforms, NW-t=-0.90 vs BOVA11), min_variance 0.925
      (underperforms, NW-t=-0.49) — **min_variance's underperformance qualitatively
      reproduces the R-series' R4 verdict via an entirely different evaluation
      methodology** (stitched walk-forward vs. R4's single fixed test split), a strong
      cross-check that this new harness isn't silently broken. classical_mv shows the
      strongest point estimate (9.13x, active NW-t=2.32) but by far the widest bootstrap
      CI (`[1.14, 44.8]`, ~39x spread) — textbook naive-Markowitz instability (Michaud
      1989), not a robust bar; H2's target is a narrower CI at comparable-or-better
      return, not beating this point estimate (see `H0_FINDINGS.md`'s Interpretation
      section, generated by the script itself so this caveat persists on every rerun).
- [x] **Power analysis:** min detectable mean IC (t=2): 0.0300 @ k=21, 0.0519 @ k=63 —
      matches the ≈0.02–0.03 pre-registered estimate almost exactly. Min detectable
      annualized IR (t=2): 0.726 — matches the ≈0.7 estimate almost exactly.
- **Gate:** none (infrastructure). ✅ Baselines reproduce the expected qualitative
  pattern; power numbers written to `H0_FINDINGS.json` before H1 examined any signal.

### H1 — Single-characteristic IC screen (THE kill gate; zero ML)
**Status (2026-07-19): RUN. Verdict: PASS.** `src/h_series/milestone_h1.py` +
`tests/h_series/test_milestone_h1.py` (synthetic regression test verifying the
sector-neutral gate keeps a real stock-specific signal and kills a pure sector tilt).
Output: `H1_FINDINGS.md` / `H1_FINDINGS.json`. Screened over 184 monthly dates
(2011-04-29 → 2026-07-14, `features.WINDOW_START` — restricted to the
TOP50_ML_READINESS_AUDIT.md-validated fundamentals-complete era), universe verified
exactly 50 names every date.

**Two bugs caught by actually running this (both fixed, both worth knowing about):**
(1) `build_monthly_panel` initially had no window floor and screened the full
2000–2026 history, including the pre-2011 structural-fundamentals-NaN era — fixed via
`WINDOW_START="2011-04-01"`; re-running confirmed this did NOT change the survivor
count (that era's rows were already NaN-excluded), so it wasn't the source of the
result, but the unwindowed run was still a real methodology bug independent of outcome.
(2) The sector-neutral IC initially compared a sector-demeaned CHARACTERISTIC against a
NON-demeaned TARGET, structurally attenuating real within-sector signal (caught because
a synthetic test with a near-deterministic embedded signal measured IC 0.33 instead of
the expected ~0.99) — fixed by also sector-demeaning the target for the sector-neutral
variant (`fwd_rel_return_sector_neutral_k{k}` / `target_rank_sector_neutral_k{k}`,
standard Barra-style two-sided neutralization); the raw variant is unaffected.

**Hypothesis:** slow characteristics — value (`earnings_yield`, `book_to_market`,
EV/EBITDA), quality (ROE, net margin, debt/equity trend), dividend yield, 12-1
momentum, low-vol, filing-lag/info-age, and their `*_zhist_5y` variants — carry
cross-sectional predictive power at k ∈ {21, 63} that price/technicals (M3-null)
do not. This is the classic-factor-premia-exist-in-B3 bet, nothing more exotic.
- [x] Monthly PIT feature matrix; Spearman rank IC time series per characteristic per k,
      vs forward BOVA11-relative return; NW t-stats; BH-FDR across the screen.
- [x] **Both raw and sector-demeaned ICs.** The gate runs on sector-neutralized ranks
      only; raw is diagnostic. Confirms the addendum's concern was real: e.g. `roe`
      clears FDR + |t|≥2 on BOTH raw and sector-neutral, but its sector-neutral
      sign_consistency is only 0.50 — correctly excluded by the sign-consistency check,
      not by sector-neutralization itself in this case (both checks are doing real work).
- [x] Quintile spread portfolios per characteristic — all survivors show small,
      economically sane monthly spreads (0.002–0.03), nothing implausibly large.
- [x] Sign-consistency across 4 chronological sub-windows (≥60% required).
- **Gate result — 10 distinct characteristics survived** (sector-neutral, |NW-t|≥2,
  FDR 10%, sign-consistency ≥0.6, at either k): **value** (`book_to_market`, `pl`,
  `pvp`, `ev_ebitda`), **quality** (`net_margin`), **income** (`div_yield_12m`,
  JCP-inclusive), **momentum** (`momentum_vs_market_12m`, `momentum_vs_sector_12m`),
  **low-vol** (`volatility_60d`, negative sign), **liquidity** (`turnover_ratio`,
  negative sign — an illiquidity premium). Excluded despite raw significance:
  `earnings_yield` (weak sector-neutral, notably NOT mirroring `pl`'s strength in sign
  — a known, documented artifact of this dataset's P/E sign-discontinuity at
  near-zero/negative earnings, not something to "fix"), `roe` (sign-inconsistent),
  `amihud_illiquidity` (sub-threshold t at k=63), `roe_trend_4q`/`debt_equity`/
  `*_zhist_5y` variants (no signal). IC magnitudes (0.02–0.10) and quintile spreads are
  realistic, not degenerate; the surviving factor families (value/quality/momentum/
  low-vol/illiquidity) are exactly what EM-equity factor literature would predict —
  none of this looks like a pipeline artifact, but see the caveat below.
  - **PASS → H2 with the 10 survivors**, per this gate's own rule. This reverses the
    M-series/R-series "no alpha anywhere" pattern for the first time — genuinely
    consequential, but H1's job was only ever screening-level significance (IC), not
    portfolio-level proof; per §4's ratchet, this does NOT license skipping straight to
    a strong claim. **Same protocol as the M4/R4 gates: awaiting user sign-off before
    starting H2** (composite construction + benchmark-relative sizing), not pursued
    further without it.

### H2 — Linear composite + benchmark-relative construction
**Status (2026-07-19): RUN. Verdict: FAIL.** `src/h_series/{composite,milestone_h2}.py` +
`tests/h_series/{test_composite,test_milestone_h2}.py` (45/45 fast suite green). Output:
`H2_FINDINGS.md` / `H2_FINDINGS.json`. Anchor: cap-weight (`capw`) primary per user
sign-off (closest investable analog to BOVA11; H0 showed the EW anchor alone carries a
structural drag), equal-weight (`ew`) reported. Scope: Construction A (multiplicative +
additive tilt) + all 6 mandatory ablations; Construction B (TE-constrained SLSQP overlay)
correctly NOT built — deferred behind an A-looks-alive gate that A did not clear.

**Hypothesis:** surviving characteristics combine into a composite whose tilt over a
fully-invested, benchmark-relative portfolio clears BOVA11 net of costs. The construction
is structurally incapable of the R-series failure (no cash asset, fully invested) — but
turned out to have a DIFFERENT structural confound, not anticipated in the original
design (see Interpretation below).
- [x] Composite: ridge on characteristic ranks (λ ∈ {0.1,1,10,100,1000}, selected on
      pre-2024 stitched OOS IC only), refit per walk-forward fold. Target = normalized
      rank of forward relative return (raw, k=21 primary); sector-neutral and k=63
      variants also fit for ablations (i)/(vi). Chosen λ=100 (raw), 1000 (sector-neutral),
      100 (k=63). Freshness × characteristic interaction columns added for the 3
      filing-derived survivors (`pl`, `pvp`, `net_margin`).
- [x] Construction A: `w ∝ anchor × (1 + γ·score)` (multiplicative, primary) and
      `w = anchor + γ·(score−mean)/Σ|score|` (additive, carried per the reviewer's
      concern that a cap-weight anchor's multiplicative form structurally chokes
      small-cap conviction), both with an iterative cap-enforcement loop (`enforce_max_weight`,
      max weight 10%, single-pass clip-renormalize was verified to re-violate the cap) and
      a no-trade band (5% L1 vs the actual drifted portfolio, never a remembered prior
      target). γ ∈ {0.5, 1.0, 2.0} selected on pre-2024 net IR: γ_mult=2.0, γ_add=0.5.
- [ ] Construction B: not built — A did not clear the gate, so per the plan's own
      complexity ratchet (§4) there is nothing to hand to a TE-constrained overlay yet.
- [x] Rebalance-frequency ablation: quarterly (k=63 composite) vs monthly — quarterly IR
      1.05 vs monthly 1.12, no meaningful difference (weekly not built, scoped out — k=5
      was never screened by H1 either, per §1.A's design verdict).
- [x] **Ablations / attribution (mandatory), all run:**
  - (i) sector-demeaned composite: pre-2024 IR 0.96 (vs 1.12 raw) — not a pure
    sector-timing bet, the signal survives sector-neutralization.
  - (ii) size/beta-neutralized composite: pre-2024 IR 1.09 (vs 1.12) — not a disguised
    size or beta bet either.
  - (iii) composite vs `momentum_vs_market_12m` alone: pre-2024 IR 1.10 — **the ridge
    step over all 10 survivors adds essentially nothing over the single strongest
    characteristic used alone.**
  - (iv) date-permutation null (200 draws, whole score-cross-section shuffle): observed
    IR 1.12 vs. a null distribution centered at **1.27** (p=0.78) — see Interpretation.
  - (v) 2× costs: pre-2024 IR 1.11 — survives (costs were never the binding constraint
    here).
- **Gate (pre-registered):** pre-2024 net IR > 0 with bootstrap CI excluding 0 (**PASS**,
  IR=1.12, CI [0.63, 2.32]) — quintile monotonicity (**FAIL**, Q4 mean ≈ Q3 mean, top
  quintile does not clear the 4th) — direction replicates on 2024–2026 (**PASS**, mean
  active return positive, IR 2.01) — survives 2× costs (**PASS**, IR 1.11). **4 of 5
  criteria passed; quintile non-monotonicity alone fails the gate → verdict FAIL.**

**Interpretation — why FAIL is the right call, not just a technicality:** the untilted
cap-weight anchor ALONE (γ=0, no ridge tilt at all — just holding the top-50 universe
cap-weighted) already has a pre-2024 IR of **0.95** vs BOVA11 (t=2.19, significant on its
own) and a post-2024 IR of 1.82. The tilted primary variant's IR (1.12) is barely above
this baseline. This is a genuinely new structural fact this program hadn't measured
before (H0 only tested an equal-weight UCRP anchor; the R-series never tested a
cap-weight anchor at all): **the top-50 cap-weighted basket itself outperforms
BOVA11/IBOV over this window**, independent of any stock selection — plausibly an
index-composition effect (IBOV is broader and differently weighted than a clean top-50
cap-weighted basket). The date-permutation null (ablation iv) makes this precise: a
random, uncorrelated score fed through the identical anchor+construction machinery
produces IRs centered *above* the real composite's IR. **The composite's incremental
contribution beyond the anchor is statistically indistinguishable from noise** — the
positive, cost-surviving, replicating-looking numbers were mostly a beneficial
anchor/universe choice, not evidence of H1-survivor stock-picking skill combining into
portfolio-level alpha. Ablation (iii) independently corroborates this from a different
angle: the 10-characteristic ridge composite performs no better than momentum alone.
- **FAIL with H1 passed → per this gate's own rule, one iteration allowed on
  construction only (frozen signal), then stop.** Candidate construction-only next
  steps (not started, awaiting sign-off): (a) an anchor-adjusted gate — re-run the
  permutation null / gate against the untilted-anchor baseline as the comparator
  instead of BOVA11, isolating whatever residual skill exists after removing the
  now-discovered anchor effect; (b) Construction B (TE-constrained overlay vs the
  cap-weight anchor, not BOVA11). **Same protocol as every prior gate in this program:
  awaiting user sign-off before starting either, not pursued further without it.**

### H3 — Conditional nonlinearity (only if H2 passes)
**Hypothesis:** interactions (e.g. value works only in rate-cutting regimes; quality
only in stress) add OOS lift over the linear composite.
- [ ] LightGBM (depth ≤ 3, heavy regularization, monotonicity constraints where
      economically signed) on the same monthly matrix; walk-forward identical to H2.
- [ ] Discrete macro conditioning: 2-state Selic-cycle filter × linear composite
      (2× the parameters, not 200×).
- **Gate:** beats H2's composite stitched OOS net IR by a pre-registered margin (set in
  H0, ≈ +0.1 IR) — otherwise **keep the linear model**. Complexity must pay rent.

### H4 — Fallback: benchmark-relative risk shaping (STRICTLY TIMEBOXED, likely skip)
**The closet-indexing warning, stated up front:** with ±2–3% active bounds and sector
caps on a 50-name, highly correlated universe, the feasible region collapses onto the
benchmark — w_portfolio ≈ w_BOVA11 — and after rebalancing turnover and execution
friction the construction is mathematically predisposed to underperform BOVA11 by
approximately its own trading costs. H4's only live hypothesis is that a
max-diversification/min-variance tilt *within* those bounds buys enough drawdown
reduction to matter, which on this universe is a long shot.
- **Timebox: reuse of R0 `risk_portfolios.py` + the H0 spine only — one config sweep,
  no new engineering. If it needs more than that, it isn't worth it.**
- **Gate:** OOS return within CI of BOVA11 AND max-drawdown/Calmar strictly better,
  net, replicated on the untouched segment. Cost drag vs BOVA11 reported explicitly —
  if active share is < ~10%, declare closet-indexing and stop regardless of metrics.
- **If H1 fails, the rational default is to SKIP H4 entirely** and declare the top-50
  universe depleted for this feature set (see §4 close-out). H4 runs only if there is
  spare capacity while awaiting a pivot decision.

---

## 4. Global decision gates

- **No phase proceeds on point estimates from a single window** (M1/R2 precedent).
- **Any positive result must replicate in direction on data not used for any choice.**
- **Model complexity is one-way ratcheted UP only through gates H1→H2→H3**; a gate
  failure never triggers a more complex model, only a simpler one or a stop.
- **Program close-out condition:** H1 fail = the top-50 B3 universe is declared
  depleted of exploitable monthly/quarterly cross-sectional signal for this feature set
  — combined with the M-series (price alpha ∅ at k ≤ 21) and R-series (unconstrained
  risk-only ∅), that is three independent clean negatives on the same universe. The
  correct next move is a *dataset* move, not an architecture move: **Option D —
  small-cap/SMLL universe**, where liquidity premia and structural inefficiencies
  plausibly exist (at the price of real capacity/slippage modeling work — the current
  cost model does not transfer). Do not burn weeks engineering an H4 index-tracker
  just to lose to costs; H4 is opportunistic only (see its timebox). Option D is a new
  plan requiring user sign-off, same as the M4 and R4 gates.
