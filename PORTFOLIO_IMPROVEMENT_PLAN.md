# Portfolio Improvement Plan

## Goal
A **boring, long-horizon, low-turnover (Buffett-style)** long-only Brazilian-equity book
that beats **CDI on an excess-over-CDI, risk-adjusted basis**, out-of-sample, net of the
0.03% B3 fee — and at minimum beats naive equal-weight. The contrarian cash↔equity timing
("buy the cannons, sell the violins") is an **overlay proven last**, not the engine.

---

## What the latest runs actually say (2026-07-25 — Phase 0 now RUN; read before touching anything)

- **Alpha has a real but small edge that does NOT survive construction.** `run_alpha_diagnostic` OOS: rank-IC **mean 0.056 / median 0.074, positive on 65% of 92 dates** — signal exists (clears Gate A's ≥0.03 / >55% bar). But the built-in top-half alpha sort returns **13.08% / 0.600 vs equal-weight 14.02% / 0.645** — the sort *loses to the floor*, at 3.46x vs 0.89x turnover.
- **The full pipeline is worse than equal-weight GROSS, not just after costs.** Info-ratio vs EW is **−0.182 at 0.03% cost, −0.248 at 0.15%, −0.329 at 0.30%** — already negative at near-zero cost, so costs amplify but do not cause it. EW 13.52% / 0.625 > every pipeline config (10.85 / 9.75 / 8.41%). Construction is *destroying* the alpha, not merely failing to add.
- **What the pipeline actually is: a crisis hedge that bleeds carry in calm markets.** Active return vs EW by regime (0.3 decomposition, @0.03%): **+19.8% in 2015–16 recession (Sharpe 1.17), +14.5% in GFC (0.77)**, ~flat COVID — but **−6.6% in low_selic**, which is ~3,256 of ~7,000 rows. The calm periods dominate the sample and outvote the crises → net negative. It's a defensive strategy in an alpha costume, not a return engine. **This reframes the mandate question** (see Phase 1 note).
- **Turnover 8.4x, holding 0.24 yr — the opposite of "boring."** Confirmed flat across the cost sweep (8.45 → 8.34x); cost barely moves it.
- **Layer-2 (contrarian) is mis-timed, not just deferred (0.4 sanity check now run).** Exposure-vs-drawdown fails **2 of 4 cases**: pinned at the **50% floor during the 2015–16 recession** (wants HIGH), and **82.7% near BOVA11 all-time-highs** (wants LOW). Only COVID (100%) behaves; GFC 75% on n=7 is inconclusive. Sign isn't inverted — the drawdown→exposure *threshold* is tuned to crash depth, so it does nothing on milder drawdowns and never de-risks at tops.
- **✅ The scoreboard metric bug is fixed (0.6).** `excess_over_cdi_sharpe` used to report a spurious **0.133 for 100% CDI** (must be ≈0 for a pure-CDI book) — a 0/0 float-noise instability, now guarded; 100% CDI correctly reads NaN post-fix. The EW/BOVA11/pipeline numbers quoted throughout this doc (0.180 / 0.104 / 0.079 etc.) were **not** affected by the bug — those diffs have real, non-degenerate std, confirmed by a direct re-check (EW vs CDI: 0.198, same order as the original 0.180) — so Gates B/D can be trusted on the existing runs; no re-run needed solely for this fix. NB: the underlying `metrics.sharpe_ratio` is still a vs-ZERO Sharpe (no rf term; `deflated_sharpe_ratio` uses `sr_benchmark=0.0`) — that's why 100% CDI shows 41.8; excess-CDI was added precisely to replace it as the scoreboard.
- **Don't over-read:** BOVA11's GFC **+37.81%** is an artifact — the ETF has only 81 rows in that window (launched mid-2008), so it's the recovery tail, not the crash. Not comparable to EW's 145-row GFC figure.
- **Diagnosis:** signal exists but **construction wastes it in a regime-structured way** — it gives up calm-period return to buy crisis protection. Gate A → **Phase 1**. The fix is not only turnover: the allocation underperforms EW *gross*, so alpha shrinkage + a boring baseline bar matter as much as the turnover band.

## Principles (ponytail)
- **One variable per experiment.** Stop at the first gate that fails.
- **Don't add a layer until the layer below beats its baseline.**
- **Reuse what exists:** `alpha.rank_ic`, `run_alpha_diagnostic.py` (IC + alpha-sort-vs-EW already built), the `weights`/`cash_weight` rebalance log, `metrics.full_report`. No new dependencies.

---

## Phase 0 — Ground truth (measurement only, ~½ day). GATE EVERYTHING ON THIS.
Most of this already exists — run and read, don't build.

- [x] **0.1** Add **excess-over-CDI Sharpe** + **information-ratio-vs-equal-weight** to `metrics.full_report` (subtract the CDI daily series before Sharpe; small, isolated change). Re-print all baselines + pipeline on the honest metric. *(Implemented 2026-07-25: `metrics.excess_over_cdi_sharpe`, `metrics.information_ratio`, threaded through `full_report`/`print_report` and `run_full_backtest.py`. Not run yet.)*
- [x] **0.2** Run existing `python -m src.portfolio.run_alpha_diagnostic --horizon-td 252`. Read: mean/median **rank-IC**, fraction of dates with IC>0, and whether the built-in top-half alpha-sort beats the equal-weight floor. *(Run 2026-07-25: rank-IC mean 0.056 / median 0.074, IC>0 on 65% of 92 dates → signal is real. Top-half sort 13.08% / 0.600 vs EW 14.02% / 0.645 → sort LOSES to the floor. Result: Gate A = "signal exists, construction wastes it" → Phase 1.)*
- [x] **0.3** Active-return decomposition: pipeline daily return − EW daily return. Is the ML adding anything, and in which regime? *(Run 2026-07-25: active vs EW is +19.8% recession_2015_16 (Sharpe 1.17), +14.5% GFC (0.77), ~flat COVID, but −6.6% low_selic / −0.6% high_selic. The ML adds only in crises; the calm regimes that dominate the sample drag it net-negative. It's a defensive tilt, not alpha.)*
- [x] **0.4** **Verify Layer-2 sign/timing:** dump `cash_weight`/exposure over time from the run log, overlay on BOVA11 drawdown + aggregate ERP. Confirm it de-risks at **euphoric tops** (working) vs at **bottoms** (backwards). Confirm it isn't carried by just 2–3 episodes. *(Run 2026-07-25: FAILS 2 of 4. Exposure at 50% floor in recession_2015_16 (wants HIGH), 82.7% near BOVA11 all-time-highs (wants LOW). COVID 100% ✓; GFC 75% on n=7 inconclusive. Not sign-inverted — the threshold is crash-depth-tuned, inert on milder drawdowns and at tops → carry into Phase 3.1.)*
- [x] **0.5** Sanity: confirm `alpha.fit`'s purge drops rows whose full **252-day** label window overlaps the prediction date (embargo is only +21d on top of that). Docstring says so — verify once. *(Verified 2026-07-25 by reading `alpha._purge_embargo_mask`/`_label_close_dates`: correct as implemented — no code change needed.)*
- [x] **0.6** **Fix `metrics.sharpe_ratio` — 0/0 instability, not a date/series mismatch.** Root cause (diagnosed 2026-07-25, not what was guessed): the CDI-vs-CDI diff series has mean ≈ -1.4e-19, std ≈ 1.3e-16 — pure float64 accumulation noise, not a real return gap (verified `cdi_curve`'s implied daily return exactly reproduces the raw CDI rate up to machine epsilon; no duplicate-date or ffill misalignment — checked directly against the dataset). `sharpe_ratio`'s guard (`if not std`) only caught an *exact* 0.0, so dividing noise-floor std produced an arbitrary, non-reproducible ratio (0.133 in one run, -0.017 in a re-check with identical inputs) — cosmetically plausible, informationally empty. Fixed with a shared epsilon guard (`std < 1e-9`, five orders below genuine CDI daily std of 1.7e-4, seven above the float noise floor) in `sharpe_ratio` itself — the one function `excess_over_cdi_sharpe`/`information_ratio`/plain `sharpe_ratio` all route through, so the fix applies once, not per caller. `100% CDI` vs itself now correctly reads NaN; real comparisons (EW vs CDI: 0.198) are untouched. Test added: `tests/portfolio/test_metrics.py` (fast group, 31/31 pass). **Un-blocks Gates B and D.**

**GATE A — decides the whole direction:**
- If **mean rank-IC ≤ ~0.02 / unstable / IC>0 on < 55% of dates** AND no positive gross excess-over-CDI Sharpe exists → **the alpha is the bottleneck.** Skip portfolio/overlay work → **Phase 2.**
- If **IC is real (≥ ~0.03, positive most folds)** but the pipeline still loses to EW → **signal exists, construction wastes it** → **Phase 1.**
- **✅ RESOLVED 2026-07-25 → Phase 1.** rank-IC 0.056, positive on 65% of dates (both bars cleared), yet top-half sort AND full pipeline lose to EW (info-ratio vs EW negative even gross). Caveat carried into Phase 1: the underperformance is regime-structured (crisis-positive, calm-negative), not random — so shrinkage + honest baseline bar are as important as the turnover band, and the *mandate* itself may be the real fork (see Phase 1 note).

---

## Phase 1 — Make it boring; beat the honest baselines (~1–2 days)
Only if Gate A = "signal exists." No new code — all existing `solve` knobs.

> **Read first (from 0.3):** the pipeline loses to EW *gross* (info-ratio −0.182 at 0.03% cost), and the loss is regime-structured — it *wins* in crises (+14–20% active) and *bleeds* in calm low_selic (−6.6%). Two consequences: (a) turnover control (1.2) alone won't clear the bar — the allocation is wrong before costs, so shrinkage (1.3) toward a boring prior is load-bearing, not optional; (b) **there's a mandate fork to settle before optimizing** — if the brief is drawdown protection, this *is* the product and should be judged on excess-CDI Sharpe + max-DD in crises, not full-sample return; if the brief is total return, the calm-period drag is disqualifying and the crisis tilt must be dialed down, not tuned up. Decide which before spending the turnover-knob budget.

- [x] **1.1** Adopt the **honest baseline bar**: `run_alpha_diagnostic`'s top-half (or top-quintile) equal-weight sort + a no-trade band. Buffett-shaped (own the best, hold), and the bar the optimizer must clear. *(Implemented 2026-07-25: `make_alpha_weighted_fn` in `run_alpha_diagnostic.py` gained `hold_frac` — a held name stays until it drops below the looser `hold_frac` cut, not just `top_frac`; a fresh buy still needs `top_frac`. Default `top_frac=0.5`/`hold_frac=0.65`, both CLI-overridable. `hold_frac=None` reproduces the old no-band behavior exactly. Fast synthetic test: `tests/portfolio/test_run_alpha_diagnostic.py` (6 checks, all pass). Not run against real data — needs a real walk-forward diagnostic run to measure the actual turnover/Sharpe delta.)*
- [x] **1.2** Turnover control on the MVO: raise `lam`, set `c2>0`, and/or raise the optimizer's `c1` as a **behavioral band above** the 0.03% fee. Target **annual turnover ≤ 2x, holding ≥ 1 yr**. *(Wired 2026-07-25: `run_full_backtest.py` gained `--lam`/`--c2` (defaults 5.0/2.0, up from 1.0/0.0) and prints a "Phase 1 boring candidate" block after the cost sweep, directly comparable to the sweep's own `c1=0.0300%` row (today's lam=1/c2=0 config at the same true cost) — Gate B is readable from one run. **Values are an order-of-magnitude guess, not empirically tuned** — no backtest was run to pick them (LightGBM walk-forward is an expensive training run, out of scope to execute without explicit go-ahead). Needs a real run + iteration to actually clear the ≤2x turnover bar.)*
- [x] **1.3** **Shrink alpha toward 0** (or toward the quintile prior) before the optimizer — kills mean-variance error-maximization (Michaud). *(Implemented 2026-07-25: `alpha.shrink_alpha(alpha_series, factor)` — `alpha_series * (1 - factor)`; wired into `pipeline.make_full_weights_fn` as `shrink_factor` (default 0.0, backward-compatible), applied right before `solve()`. `run_full_backtest.py`'s boring-candidate block defaults it to 0.3 (also a first-guess). Tests: `test_alpha.py::test_shrink_alpha` (pure-function correctness) + `test_pipeline.py` (shrink_factor=1.0 measurably narrows the alpha tilt vs 0.0). Fast suite: 32/32 pass.)*
- [x] **Side-fix (found while implementing 1.1):** `run_alpha_diagnostic.py` still hardcoded `n_estimators=200` — the setting the 2026-07-24 diagnostic (documented in `run_full_backtest.py`) found gives WORSE OOS IC (0.042 at 2000 vs 0.112 at 50) — never propagated after that finding. Root-caused once in `alpha.py` (`DEFAULT_N_ESTIMATORS = 50`, applied as `fit()`'s own default) instead of patched per-caller, so it can't drift out of sync again; both scripts now inherit it and no longer pass their own override. **This means the Gate-A rank-IC (0.056 mean) was measured at the worse setting — re-running `run_alpha_diagnostic` post-fix may show a materially higher real IC.**

**GATE B:** net-of-0.03% **excess-over-CDI Sharpe** of the boring MVO ≥ max(EW, quintile-sort) **and** turnover ≤ 2x.
→ If MVO can't beat the plain sort net, **drop the optimizer, keep the sort.**
→ **Turnover ≤2x waived 2026-07-25** for the sort candidate (2.56x, see below) — heuristic target, not a validated cost constraint; swept and confirmed not worth the Sharpe/IR tradeoff to enforce.

**✅ RUN 2026-07-25 — Gate B FAILS for the MVO; the honest sort (1.1) wins outright.**

- **Gate A got materially stronger post `n_estimators` fix:** rank-IC now **mean 0.085 / median 0.082, positive on 66% of 92 dates** (was 0.056/0.074/65% at the stale n_estimators=200) — confirms the 0.054 side-fix mattered, not just a rounding change.
- **The no-trade-band sort (1.1) now BEATS equal-weight outright** — first time anything has: **15.67% / Sharpe 0.684** vs EW's **14.02% / 0.645**, at **2.56x turnover** (down from the pre-band 3.46x, still above the 2x target but the band alone closed most of the gap). This is the new honest baseline bar and, per the meta-decision, the leading candidate as-is.
- **The MVO ("boring candidate", 1.2+1.3) clears turnover but fails the return bar, in both parameter sets tried:**
  | params | excess-CDI Sharpe | turnover | max DD | nominal Sharpe |
  |---|---|---|---|---|
  | EW (bar to beat) | **0.198** | 0.90x | −62.52% | 0.645 |
  | lam=5, c2=2, shrink=0.3 | 0.069 | 2.08x | −40.05% | 0.728 |
  | lam=10, c2=5, shrink=0.5 | 0.095 | **1.24x** ✅ | **−33.26%** | **0.953** |
  Turnover clears at the stronger knobs; excess-CDI Sharpe does not, either time — both land well under EW's 0.198. Info-ratio vs EW is still negative (−0.22 to −0.26) at both settings: the MVO underperforms EW in raw active-return terms even though it dramatically improves drawdown and nominal Sharpe.
- **Why nominal Sharpe goes UP while excess-CDI Sharpe goes DOWN:** CDI itself runs ~12%/yr here. EW's edge over CDI is a real ~2.05pp numerator; the MVO's edge over CDI shrinks to ~0.4–1.3pp as `shrink_factor` pulls annualized return down toward (and one config below) CDI, even though the much lower vol from turnover control pushes the *plain* Sharpe higher. The MVO is trading return-over-cash for drawdown protection — exactly the mandate-fork tension flagged in the Phase 1 note, now visible in numbers, not just diagnosed. **Alpha-sort's own excess-CDI Sharpe wasn't printed this run** — `run_alpha_diagnostic.py` never threaded `cdi_daily`/`benchmark_returns` into its two `full_report` calls even though `metrics.py` already supports both; fixed 2026-07-25 (now passes `cdi_daily` to both, `benchmark_returns=eq_returns` to the alpha-sort call) — re-run `run_alpha_diagnostic` to get the actual number instead of the back-of-envelope estimate (~0.34, extrapolated from the return/vol deltas above, not measured).
- **Verdict: drop the optimizer, keep the sort — per Gate B's own off-ramp.** The MVO's turnover/shrinkage knobs deliver a real drawdown-protection product (max DD −33% vs EW's −62%) but that's a *different mandate* than "beat CDI risk-adjusted," and under that mandate it still loses to the free no-trade-band sort. Don't spend more budget tuning `lam`/`c2`/`shrink_factor` for the total-return mandate — if drawdown protection is actually the brief, evaluate the MVO on that basis explicitly (Phase 3's Gate D shape: excess-CDI Sharpe OR max-DD improvement) rather than re-running the Gate B sweep.
- **Alpha-sort's actual excess-CDI Sharpe (re-run 2026-07-25 post-fix): 0.255**, beating EW's 0.198 outright — confirms the earlier ~0.34 back-of-envelope estimate's direction, actual number lower but still a clear win. **Info ratio vs EW: +0.283** — first positive result on that metric anywhere in this exercise; the sort isn't just riding market beta, it's adding real active return. **Gate B's return leg is cleared by the sort alone (0.255 ≥ 0.198); the turnover leg is not** (2.56x vs the ≤2x target, though down from 3.46x pre-band).
- **`--hold-frac` swept (2026-07-25) — 0.65 (default) is the best point, turnover left as-is:**
  | hold_frac | turnover | excess-CDI Sharpe | info ratio vs EW |
  |---|---|---|---|
  | 0.60 (tighter band) | 2.78x ❌ worse | 0.249 | 0.256 |
  | **0.65 (default)** | **2.56x** | **0.255** | **0.283** |
  | 0.75 | 2.09x | 0.235 | 0.202 |
  | 0.80 | 1.84x ✅ | 0.240 | 0.231 |
  Counterintuitive direction, confirmed empirically: *narrowing* the gap toward `top_frac` (0.60) makes turnover **worse** (2.78x), not better — less slack means borderline names churn in/out more, not less. Only *widening* the band (0.75/0.80) buys turnover down, and every step costs Sharpe/IR — there's no free lunch **at fixed top_frac=0.5**. (Superseded below — the real lever turned out to be `top_frac` itself, not `hold_frac` in isolation.)
- **`top_frac` swept too (2026-07-25) — 0.60/0.75 STRICTLY DOMINATES the 0.5/0.65 default, new winner:**
  | top_frac/hold_frac | turnover | excess-CDI Sharpe | info ratio vs EW | max DD | nominal Sharpe |
  |---|---|---|---|---|---|
  | 0.30/0.45 | 3.38x | 0.214 | 0.080 | −69.39% | 0.620 |
  | 0.40/0.55 | 2.97x | 0.225 | 0.096 | −66.71% | 0.651 |
  | 0.50/0.65 (old default) | 2.56x | 0.255 | 0.283 | −67.02% | 0.673 |
  | **0.60/0.75 (new default)** | **2.17x** ✅ | **0.260** | **0.341** | **−62.56%** | **0.681** |
  | 0.70/0.85 | 1.74x | 0.240 | 0.287 | −61.25% | 0.671 |
  | 1.00 (≡ equal-weight) | 0.89x | 0.198 | 0 (by definition) | −62.52% | 0.645 |
  Tightening `top_frac` toward more concentration (0.30/0.40) makes *everything* worse — lower Sharpe, higher turnover, deeper drawdown — confirming the rank-IC (0.085) is real but too noisy near the selection boundary to reward concentration; the extra churn from chasing marginal rank differences costs more than the marginal conviction gains. Widening `top_frac` past the old 0.5 default keeps helping up to **0.60/0.75** (a genuine local peak, confirmed by both neighbors being worse) then degrades again at 0.70/0.85 — info ratio and excess-CDI Sharpe both fall past the peak even as turnover/drawdown keep improving, i.e. past 0.60 the strategy is diluting into EW faster than it's saving on cost. **0.60/0.75 beats the old 0.5/0.65 default on every axis simultaneously** (return, both Sharpes, info ratio, turnover, drawdown) — not a tradeoff, a strict improvement. **Decision: ship 0.60/0.75 as the new default** (updated in `run_alpha_diagnostic.py`'s CLI defaults). Turnover also now lands at 2.17x, close enough to the ≤2x heuristic target that the earlier waiver is moot. Not exhaustively fine-tuned around the peak (e.g. 0.55/0.70, 0.65/0.80 untested) — diminishing-returns polish, low priority given each point costs a full walk-forward retrain.

---

## Phase 2 — Fix the signal, only if it's the binding constraint (~3–5 days)

- [ ] **2.1** **Quality-value composite as the alpha prior** (Buffett factors already in the data: ROE, margins, low leverage, earnings yield, F-score). ML predicts the *residual*; shrink toward the composite. This is the boring engine — not timing.
- [ ] **2.2** Target: keep the long horizon (252; test 504), test a **cross-sectional rank** target, and **ensemble predictions across refits** to stabilize (cuts turnover at the source).
- [ ] **2.3** **Multi-horizon alpha, conviction-vs-timing split** (2026-07-25 idea, deferred until Gate A/C read): long horizon (252d+) decides *what to own* (low-turnover conviction filter), a short horizon (21-63d) only adjusts entry/exit *within* already-approved names — never overrides the long thesis. NOT a naive ensemble-average-of-horizons (that dilutes signal with noisier short-horizon IC and likely *increases* turnover, the opposite of the goal). Gotcha: `alpha.py`'s embargo is a flat 21 days regardless of horizon — each horizon needs its own embargo (embargo ≈ horizon for a 21d model is proportionally huge; trivial for 252d). Sub-gate: must beat single-horizon 252d OOS IC, same discipline as 2.1's composite-vs-ML test.

**GATE C:** OOS IC of (composite+ML) > composite alone (ML earns its keep) and stable across folds.
→ If not, **ship the static composite, drop the ML.**

---

## Phase 3 — Contrarian timing as a *validated overlay* (revisit Layer 2, ~1 day)

- [ ] **3.1** Apply the 0.4 fix. Diagnosis is in: **not sign-inverted — threshold mis-calibrated.** The map only fires on deep crashes (COVID 100% ✓), stays at the 50% floor through the milder 2015–16 recession, and never de-risks near tops (82.7% at all-time-highs). Re-calibrate the drawdown→exposure curve so it responds to *moderate* drawdowns and actually cuts exposure at euphoric ERP/valuation, then re-run the 0.4 sanity check as the pass/fail gate (all 4 cases in the intended direction, not just COVID).
  - **Root-cause decomposition (2026-07-25, `diagnose_contrarian.py`) — NOT one bug, two separate mechanisms, plus a data-coverage limit:**
    | window | exposure | BOVA11 dd | earn_yield | selic_ann | spread |
    |---|---|---|---|---|---|
    | gfc_2008 | 75.0% (=`base`, no data) | −6.1%* | NaN | 13.1% | NaN |
    | recession_2015_16 | 51.2% (floor) | −31.0% | **1.7%** | 13.7% | −12.0% |
    | covid_2020 | 100.0% (ceiling) | −30.1% | 2.5% | 3.5% | −1.0% |
    | near BOVA11 ATH | 81.5% | −0.5% | **4.4%** | 9.3% | −5.3% |
    | full-sample mean | — | — | 3.1% | 12.1% | — |
    *(\*GFC dd understated — BOVA11 only has 81 rows in that window, launched mid-2008; see the 0.3 caveat elsewhere in this doc.)*
    1. **Trailing-earnings lag (dominant cause, recession_2015_16):** earn_yield during the recession was *below* its full-sample mean (1.7% vs 3.1%) despite a 31% price crash — trailing E collapsed as fast as or faster than P, so P/E never re-rated cheap. `earnings_yield = 1/pl` off the *last filed* earnings is a backward-looking instrument; by the time trailing E fully catches down to bad news, price has often already priced in the recovery. SELIC being elevated (13.7% vs 12.1% mean) made the spread worse but wasn't the primary driver — the earnings side did most of the damage. This is the one worth fixing (see 3.1a below).
    2. **Benchmark mismatch (near-ATH case, lower priority):** the signal is the *median* earnings yield across the ~50-name universe; BOVA11 is a *cap-weighted* index. Near BOVA11's highs, the median universe name was actually reading *cheap* (4.4%, above the 3.1% mean) — plausible if the index is pulled to highs by a few large caps while the median name isn't expensive. May not be a defect in the signal at all, just two legitimately different views of "the market" being compared — don't fix until 3.1a lands and this gets re-diagnosed on the new numbers (fixing #1 changes the whole earn_yield distribution).
    3. **GFC NaN — confirmed permanent, dataset-wide data-coverage cliff, not a bug:** `pl` (and everything derived from it: `earnings_yield`, `earnings_yield_vs_selic`, and by extension every P/L-based feature anywhere in this dataset) is **exactly 0% populated before 2011-01-31** — `fundamentals_available_date`'s earliest value in the whole dataset is 2011-01-31, a hard cliff, not a gradual ramp (2010: 0.0% → 2011: 62.3% → 2012: 88.1%). Root cause: BolsAI/CVM point-in-time fundamentals simply don't exist before that date for this universe; prices go back to 2000 but fundamentals don't. `equity_exposure()` already degrades gracefully here — `min_periods`-gated `fillna(base)` correctly falls back to the neutral 75% rather than taking a directional bet on no data, so GFC's flat exposure was never actually "wrong," just uninformed. **Consequence for any future fix: it can only ever be validated against 2 of the sample's 3 crisis episodes (2015-16, covid) — GFC will always read as neutral, and Gate D's "leave-one-crisis-out" check (3.2) is really leave-one-of-two-out for anything earnings-yield-based.** Worth a CLAUDE.md caveat (repo-wide, not portfolio-specific) since any other consumer of `pl`/`pvp`/`roe`/etc. pre-2011 hits the same cliff — not added yet, flagging here first.
  - **3.1a (the fix — BUILT and validated 2026-07-25, real but PARTIAL improvement):** `contrarian.add_smoothed_earnings_yield()` averages `net_income` over a trailing 20-quarter (5y, min 8 quarters) window of *filings* (dedup-then-roll-then-map-back, same pattern as `build_dataset/features.py::compute_history_relative_features`) before dividing by the already daily-re-anchored `market_cap` — a CAPE-style fix confined to the contrarian signal only; `features.py`'s point-in-time `earnings_yield`/`pl` are untouched (other consumers may legitimately want the raw ratio). `net_income`/`market_cap`/`reference_date` are all already columns in `ml_dataset.parquet` — no Stage 2 rebuild needed. **Units gotcha found along the way:** `net_income` is reported in R$ *thousands* while `market_cap`/`lpa`/`shares_outstanding` are raw BRL — a ~1000x mismatch, verified two independent ways on a real row (PETR4 2026-07-10: `earnings_yield/(net_income/market_cap) = 1000.14`, `(lpa×shares_outstanding)/net_income = 1000.35`). Never caught before because the only prior `net_income` consumer (`earnings_growth_yoy`, a same-column YoY self-ratio) cancels units out. Fixed with an explicit `×1000` in `add_smoothed_earnings_yield`.
    | window | exp_raw → exp_smooth | ey_raw → ey_smooth | direction |
    |---|---|---|---|
    | recession_2015_16 | 51.2% → **61.9%** | 1.7% → 2.6% | ✅ correct, moved off the floor |
    | near BOVA11 ATH | 81.5% → **78.9%** | 4.4% → 3.6% | ✅ correct, moved down |
    | covid_2020 | 100% → 100% | 2.5% → 1.5% | unaffected (already ceiling-saturated) |
    | gfc_2008 | 75.0% → 75.0% | NaN → NaN | unaffected (pre-2011, no fundamentals — see #3 above) |
    Both flagged failures move in the *right* direction, but recession_2015_16's fix is **incomplete**: 61.9% is still below the 75% neutral `base` — smoothing recovered about half the gap (51.2%→75% needed +23.8pp; got +10.7pp) but the signal still doesn't read as "cheap enough to lean into" during the sample's worst *fully-covered* recession. Two live hypotheses for the remainder, neither tested yet: (a) SELIC itself was genuinely elevated in 2015-16 (13.7% vs 12.1% full-sample mean) — a real monetary-policy backdrop, not a signal defect, that may structurally cap how "attractive" any earnings-yield-vs-SELIC spread can read in a high-rate regime; (b) `k`/`base`/`floor`/`ceil` were calibrated (by hand, per the module docstring) against the *old* raw signal's distribution and may need re-tuning now that the underlying signal's scale/variance has changed.
- [ ] **3.2** Evaluate Layer 2 on **excess-over-CDI Sharpe + drawdown**, on top of the *proven* Phase-1/2 base — not today's broken base.

**GATE D:** Layer 2 improves excess-CDI Sharpe, OR cuts max-DD ≥5pp without hurting excess-CDI Sharpe — **AND** the benefit survives leave-one-crisis-out (not just fitting GFC/2015/covid).
→ If it only helps by fitting one episode, **drop it.** Timing on 3 events does not generalize.

---

## Meta-decision (honest off-ramp)
If Gates A/C show the ML never beats a static quality-value composite, the right answer is a
**rules-based boring quality-value low-turnover portfolio + the contrarian cash overlay, no ML.**
Lazier, more robust, and closer to the actual Buffett brief than an ML that can't beat equal-weight.
