# Conviction Model — Review Remediation Plan

Addresses the **confirmed** issues from the 2026-07-22 design/implementation/methodology
review (conviction_model + the reused `h_series` power-floor primitives). Scope rule:
only issues verified against the code this session. Speculative optimizations are
explicitly excluded (see "Deliberately NOT addressed", bottom).

Governing goal: make the **Phase-1 diagnostic gate trustworthy** and the **label/power
claims honest**, since those are the instruments the whole project uses to decide whether
to continue. Bug-fixing is secondary to that.

Each phase leaves the repo in a working, tested state. Model-changing items are framed as
**one-variable experiments** against a fixed baseline, never bundled. Training runs are
written but **not executed** (CLAUDE.md rule) — each phase's validation says who runs what.

Legend: `[IMPL]` code change · `[RESEARCH]` analysis/decision · `[EXP]` requires a training
run to evaluate · severity from the review (High/Med/Low).

---

## Phase 0 — Critical correctness fixes (no retrain required)

**Objective:** Remove confirmed bugs that silently corrupt label columns or the CPC
objective. All are small, isolated, unit-testable, and independent of each other.

**Implementation tasks**
- [x] `[IMPL]` **(3, Med/High) Drawdown-severity non-positive `adj_close` mask.**
  `labels.py::compute_drawdown_severity` computes `np.log(prices_wide)` raw; its sibling
  `trailing_volatility` masks `prices_wide.where(prices_wide > 0)`. Apply the identical
  mask here. Why: `log(0)→-inf`, `log(neg)→nan` produce `inf`/NaN severity for
  precision-degraded microcaps (the exact CLAUDE.md caveat). Trade-off: none — strictly
  correct. Dependencies: none. **Done** (commit `02c69d9`).
- [x] `[IMPL]` **(12b, Low) Guard vol≈0 in risk-adjusted labels.**
  `compute_risk_adjusted_excess_returns` divides by `trailing_vol` with no floor →
  `inf`/huge labels for pinned-constant (`adj_close_precision_degraded`) tickers. Mask
  those rows to NaN (reuse the existing flag) or apply a small vol floor. Trade-off: masks
  a handful of degenerate rows; acceptable — they're data artifacts, not signal.
  **Done** (commit `32b8d5e`): degenerate (finite, ≤1e-8) `trailing_vol` NaN'd out
  *before* dividing, so no divide-by-zero warning fires either.
- [x] `[IMPL]` **(12a, Low) Exclude the positive from CPC negative fallback.**
  `sample_cpc_negatives`' same-stock fallback (`same_pool[same_pool != pos]`) can select
  `pos + cpc_horizon` (the positive) as a negative for `<regime_gap_days`-history tickers,
  creating a contradictory InfoNCE label. Exclude `pos ± horizon` from the fallback pool.
  Trade-off: negligible; only touches very-short-history tickers. **Done** (commit
  `90d6553`): new `exclude_positions` param on `sample_cpc_negatives`, wired through both
  `build_cpc_batch` and `build_stage1b_batch`. Confirmed the bug was real before fixing
  (108 leaks / 30 seeds × 25 anchors on a synthetic short-history ticker; 0 after).

**Validation**
- [x] Extend `test_labels.py`: a synthetic panel with an interior `adj_close==0.0` and a
  pinned-constant series → `drawdown_severity` finite (not inf/NaN), degenerate-vol row
  masked. Add a below-`regime_gap_days` ticker case to `test_ssl_pretrain.py` asserting the
  positive never appears in the sampled negatives.
- [x] `python tests/run_all.py --group fast` green (52/52; test count grows as tests are
  added to existing files, so the per-run total climbs with each phase).

**Expected outcome:** Label columns finite on the real dataset; no `divide by zero
encountered in log` warning from `labels.py`.

**Exit criteria:** New assertions pass; fast suite green; a one-off `build_conviction_labels`
smoke run on ~5 real tickers produces no non-finite values outside documented warm-up NaNs.

**Phase 0 status: COMPLETE** (2026-07-22). The one item not literally executed is the
real-data smoke run in the exit criteria (touching `data/raw/`, out of scope for a
synthetic-only fix/test cycle and gated behind CLAUDE.md's "never run code you didn't
explicitly ask to run") — the synthetic regression tests directly exercise the exact
failure modes (masked-zero price, pinned-zero vol, short-history fallback leak) that
motivated each fix, which is the stronger guarantee for a pure-function bug fix.

---

## Phase 1 — Trustworthy Phase-1 diagnostics (the gate)

**Objective:** The Phase-1 battery is the project's go/no-go instrument and is currently
contaminated. Fix the diagnostics so a "pass" means what it claims — **before** re-baselining.
Pure code + synthetic tests; no training run needed to land this phase.

**Implementation tasks**
- [x] `[IMPL]` **(2, High) Exclude same-ticker temporal neighbors in diagnostic 1.**
  `neighbor_outcome_variance_ratio` drops only the self-match; the plan requires excluding
  same-ticker points within a short window. Nearest neighbors are dominated by the same
  ticker at adjacent month-ends (near-duplicate embeddings, overlapping 252-day outcomes),
  so the current PASS (0.798, gate ≤0.8) is largely an autocorrelation artifact. Thread
  `tickers` (and dates) into the function; over-fetch neighbors and filter out any neighbor
  with the same ticker within ±N months before taking `k`. Trade-off: fewer usable neighbors
  per point → slightly noisier ratio; acceptable and correct. **Done** (commit `0983d98`):
  optional `tickers`/`dates`/`exclude_window_days`/`search_multiplier` params, omitting them
  reproduces the original behavior exactly; `run_diagnostics.py`'s diag1 caller wired through.
  Regression test uses an overlapping-forward-window outcome model (mirrors the real
  `FORWARD_HORIZON` mechanism) to prove the artifact is real unfiltered and measurably
  weaker (not necessarily eliminated — surviving neighbors can still correlate with each
  other, just not with the anchor) after exclusion. Re-scoring the real Stage 1A checkpoint
  with this fix is deferred to Phase 2 (needs the encoder-universe fix first).
- [x] `[IMPL]` **(6, Med) Block the linear-probe split by ticker — diag3 AND diag4 done.**
  `diag3/diag4` pre-shuffle into an iid split (`rng.permutation`), placing near-duplicate
  adjacent-month rows of one ticker on both sides → inflated OOS R². Replace with a
  **ticker-disjoint** split (GroupShuffle by ticker) in `linear_probe_r2`'s callers.
  Trade-off: R² drops toward its honest value; that is the point. **diag3 done** (commit
  `a022380`): `diagnostics.py::group_blocked_train_mask` + a `train_mask` param on
  `linear_probe_r2`/`valuation_vs_volatility_probe` (omitting it reproduces the original
  positional split exactly); `diag3_valuation_vs_volatility` now builds one ticker-blocked
  mask shared by both probes. Regression test confirms an iid split reports a deceptively
  decent R² (0.37) on data with zero genuine cross-ticker signal, while a ticker-blocked
  split on the same data honestly reports ≈0/negative R² (−0.23).
  **diag4 turned out to have a DIFFERENT issue, not literally "block the split":**
  `quality_persistence_autocorrelation`'s caller (`diag4_quality_persistence`) never
  actually implemented a train/test split — its `rng.permutation` shuffled row order, then
  `LinearRegression().fit()` was called on the **entire** masked set (order doesn't affect
  what a linear fit learns), and `predict()` ran over **all** points, including the
  training rows themselves (in-sample). Presented to the user as options (a) add a genuine
  ticker-blocked OOS split, or (b) keep the in-sample fit and just delete the dead
  `rng.permutation` — **user chose (a)**. **Done** (commit `1e5cddb`): reuses
  `group_blocked_train_mask`; the probe now fits on TRAIN-ticker rows only and the reported
  persistence autocorrelation is computed only on HELD-OUT test-ticker rows' predictions.
  No permanent test file added (confirmed `run_diagnostics.py`'s other `diagN_*` functions
  have no test coverage either, and the plan's own Testing Strategy section doesn't list
  one — it's the real-checkpoint orchestration script, not part of the fast synthetic-test
  convention); verified instead via a synthetic smoke check confirming correct indexing
  (uses ~half the tickers, as expected from a 50/50 split) and a finite result. The
  primitives it calls (`group_blocked_train_mask`, `quality_persistence_autocorrelation`)
  are already fully unit-tested in `test_diagnostics.py`.
- [x] `[IMPL]` **(5, Med) Score diagnostics on the unpooled representation too.**
  Diagnostics run on the mean-pooled vector; Phase 2's regressor consumes the 4 *separate*
  branch embeddings. Add a second pass over the concatenated 4-branch vector and report both
  in `run_diagnostics`' JSON. Trade-off: ~2× diagnostic runtime; cheap. **Done** (commit
  `d1cf4b2`): `compute_embeddings` returns both `pooled [N,d]`/`unpooled [N,4*d]` from one
  model forward pass; `diag5_perturbation` gets a `pooled: bool` switch (it re-embeds from
  raw branch tensors, not from the precomputed array); the 7-diagnostic loop is factored
  into `run_diagnostic_battery(rep_label, ...)`, called once per representation with
  independent rng streams, output JSON nested under `"pooled"`/`"unpooled"` keys. Verified
  via a synthetic smoke check of the new shape/pooling logic (the full battery isn't
  smoke-tested end-to-end — diag1/diag2 need real data files, same as this file's existing
  non-fast-test status).
- [x] `[IMPL]` **(10, Low/Med) Add practical effect-size floors to the gates.**
  Gates 2/6 currently pass on significant-but-negligible effects (MI 0.0006, corr 0.14). Add
  a minimum-magnitude bar alongside the significance bar in the `GATES` table and record the
  absolute magnitude in the pass rationale. Trade-off: some current "passes" become
  "significant but below effect-size floor" — a more honest ledger. **Done** (commit
  `b4cc766`): `MIN_REGIME_MI=0.02` / `MIN_SMOOTHNESS_CORR=0.1` (Cohen's small-effect
  convention) AND-ed into `GATES[2]`/`GATES[6]`; gate description strings updated to state
  the combined criterion. New `tests/conviction_model/test_run_diagnostics.py` (added to
  the fast group) — unlike the other `diagN_*` functions, `GATES` is pure lambda logic with
  no data dependency, so it's directly unit-tested (fails below the magnitude floor even
  when significant, passes above both bars, still fails when not significant even above
  the floor).

**Validation**
- [x] Extend `test_diagnostics.py`: (a) diagnostic 1 with an injected same-ticker
  autocorrelated cluster shows the ratio *rises toward 1* once same-ticker neighbors are
  excluded; (b) a ticker-blocked probe on data where signal lives only cross-sectionally
  yields lower R² than the old iid split; (c) each gate's magnitude+significance combination
  evaluates correctly just above/below threshold.
- [x] `python tests/run_all.py --group fast` green (53/53 as of this phase's last commit).

**Expected outcome:** Diagnostic scores reflect genuine embedding structure, not
near-duplicate leakage; the gate table distinguishes "real effect" from "detectable but
trivial".

**Exit criteria:** New synthetic tests pass; `run_diagnostics.py` emits both pooled and
unpooled results plus magnitude-annotated gate decisions. (Re-scoring the *real* checkpoint
is deferred to Phase 2, since the checkpoint itself is on a biased universe.)

**Phase 1 status: COMPLETE** (2026-07-22). All four tasks landed
(`0983d98`, `a022380`, `1e5cddb`, `d1cf4b2`, `b4cc766`); fast suite 53/53, `ruff` clean
throughout. Next: Phase 2 (correct the encoder training universe + re-baseline Stage 1A —
requires a real training run, user-executed).

---

## Phase 2 — Correct the encoder training universe + re-baseline

**Objective:** The recorded Stage 1A baseline was trained/scored on the **current** top-150
snapshot — survivorship + look-ahead in universe selection, contradicting the design ("the
encoder always saw all names; only labels are universe-restricted"). Fix the universe, then
establish the true Phase-1 baseline using Phase-1's corrected diagnostics.

**Implementation tasks**
- [x] `[IMPL]` **(1, High) Replace the snapshot universe.**
  `run_stage1a.py::top150_snapshot_tickers` pins the encoder to the most-recent rebalance
  period. Change the default to the **full-history, all-names** panel (or, if a size cap is
  wanted, the point-in-time *union* universe ~360 names from `top150_universe_membership`,
  which is survivorship-safe by construction). Keep the snapshot only behind an explicit
  `--debug-snapshot` flag for fast smoke tests. Trade-off: larger panel → slower per step and
  more memory; `LazyPanelGatherer` already handles the memory (it exists for exactly this).
  **Done** (commit `079d95c`): new `point_in_time_union_tickers()` is the default (~360
  names, union across all rebalance periods); old snapshot behavior kept only behind
  `--debug-snapshot`. Verified the wider universe can't crash on a ticker absent from
  `ml_dataset.parquet` — `top150_universe_membership.parquet` is built by reading
  `ml_dataset.parquet` directly, so membership is a subset of the dataset by construction.
- [x] `[IMPL]` **(11, Med) Move the SSL checkpoint holdout off the Phase-7 window.**
  `split_train_holdout` carves the trailing calendar year — which overlaps Phase 7's reserved
  final holdout. Checkpoint-selecting on it leaks the final-eval window into the frozen
  encoder. Parameterize the holdout to sit *before* the reserved window (e.g. accept an
  explicit `holdout_end`/reserved-window cutoff). Trade-off: slightly less recent holdout for
  checkpoint scoring; correctness win dominates. **Done** (commit `079d95c`): new
  `truncate_to_development_window()` drops every row past `dataset_end -
  reserved_holdout_years` (default 2y) before `split_train_holdout` ever runs; applied in
  both `run_stage1a.py` and `run_stage1b.py` (new `--reserved-holdout-years` flag on each).
  New `tests/conviction_model/test_run_stage1a.py` (fast group): boundary inclusivity, and a
  synthetic membership fixture proving the union includes an earlier-period-only
  ("delisted") name that the snapshot misses.

**Experiments (user runs; code written here)**
- [ ] `[EXP]` Retrain Stage 1A (CPC-only) on the corrected universe.
- [ ] `[EXP]` Re-score with Phase-1's corrected diagnostics (pooled + unpooled).
- [ ] Record the result as **the** Stage 1A baseline that 1B/1C/1D must beat, replacing the
  biased `PHASE1_DIAGNOSTICS_20260721-*.json` numbers in the plan.

**Validation**
- [ ] Assert the training panel contains names absent from the current snapshot (i.e.
  survivorship-prone/delisted names are present).
- [ ] `[EXP]` Diagnostics run end-to-end on the new checkpoint; JSON written; plan's Stage 1A
  status block updated with the corrected numbers and the note that the prior baseline was
  biased.

**Expected outcome:** A defensible Stage 1A baseline on a survivorship-safe encoder. The
honest likelihood (given the prior probes were both R²<0 OOS) is that several gates look
*weaker* than the recorded 4/7 — that is a more truthful starting point, not a regression.

**Exit criteria:** Corrected checkpoint exists; corrected diagnostics recorded; plan updated;
old biased numbers struck through with a pointer to why.

**Dependencies:** Phase 1 (diagnostics must be fixed before re-scoring, or the new baseline
inherits the old contamination).

---

## Phase 3 — Power-floor honesty (`h_series` / `check_power_floor`)

**Objective:** The min-detectable-IC floor is fed nominal breadth/length and an independence
assumption; the realized IC test uses effective breadth (non-NaN pairs, `MIN_GROUP_N`) and
dependent cross-sections. Both bias the floor optimistic — and the v9 "top-150 promotes
k=126/k=252 to primary" claim rests on that optimism. Make the floor match the test.
Independent of Phases 0–2 (operates on labels/membership, not the encoder).

**Implementation tasks**
- [ ] `[IMPL]` **(F2, Med) Permutation-null empirical floor.**
  Add a floor that permutes forward returns within each decision date, recomputes
  `spearman_ic_by_group`, and takes the null mean-IC dispersion — this automatically captures
  effective breadth (F1) *and* cross-sectional dependence (F2), unlike the closed-form
  `1/√(n−1)`. Keep the closed-form as a labeled "pre-registered reference"; use the empirical
  one for the decision. Reuses the project's existing permutation-null convention.
- [ ] `[IMPL]` **(F1, Med) Recompute with effective inputs.**
  Recompute the closed-form floor using the **median realized non-NaN per-date pair count**
  as `n_assets` and the **OOS `ic.notna()` count** as `n_obs` (also closes the plan's own
  deferred-fix #2). Report reference-vs-effective side by side.
- [ ] `[IMPL]` **(F3, Low) Reword `_decision`.**
  `check_power_floor._decision` hard-gates on 0.035; v9 downgraded 0.035 to a *reference*.
  Reword to "feasibility: floor {below/above} the 0.035 reference — real gate is Phase-4
  significance," so the emitted string can't be misread as pass/fail.

**Validation**
- [ ] `test_walkforward.py` (or a new `test_power_floor.py`): permutation-null floor ≥
  closed-form floor on a synthetic dependent cross-section (sanity: dependence raises the
  floor); effective-input floor ≥ nominal floor when NaNs are injected.
- [ ] Re-run `check_power_floor` at n_assets=150 with both methods; record `PHASE0_POWER_FLOOR_v3.json`.

**Expected outcome:** Honest floors. Likely consequence: **k=126/k=252 may fall back from
"primary/powered" to "exploratory,"** since their v9 promotion depended on the optimistic
floor. k=21/k=63 should retain enough margin.

**Exit criteria:** v3 floor recorded; plan's primary/exploratory horizon split updated to
match the empirical floor; the v9 promotion claim re-stated conditionally.

---

## Phase 4 — Design-fidelity experiments (one variable at a time)

**Objective:** Resolve the confirmed design↔implementation deviations. Each **changes the
encoder or its inputs**, so each is a single-variable experiment scored against the Phase-2
baseline — never bundled. Land the code behind a config flag (default = current behavior),
run the A/B, keep only if diagnostics improve.

**Implementation + experiment tasks**
- [ ] `[IMPL]`+`[EXP]` **(8, Med) Cross-attention residual + LayerNorm + branch-id.**
  `EncoderCNN.forward` has no residual, no norm, no branch-identity embedding, so each
  "labeled sub-embedding" is a blend rather than "that branch, informed by others" —
  weakening the per-branch-ablation rationale. Add `updated = norm(tokens + attn(tokens))`
  and a learned per-branch embedding. Config flag `attn_residual: bool`. Experiment: does it
  improve the diagnostics (esp. unpooled probes) vs. baseline? Keep iff yes. Trade-off: tiny
  param increase; standard transformer hygiene.
- [ ] `[IMPL]`+`[EXP]` **(9, Low/Med) Normalize price levels by the close-at-t anchor.**
  `window_tensor` divides each price-level channel by its *own* anchor, collapsing high/low
  to 1.0 at t and destroying the anchor-day high/low/close relationship (deviates from the
  cited EIIE eq-18, which divides all by close-at-t). Change price-level channels to share the
  `adj_close`-at-t denominator. Config flag. Experiment: A/B on diagnostics. Trade-off:
  restores intraday-range signal; low risk.
- [ ] `[IMPL]` **(7, Med — doc or code) CPC single- vs multi-horizon.**
  Config `cpc_horizon` is scalar, but the plan repeatedly claims multi-horizon CPC and uses
  that to justify the multi-output labels. Two options, pick one and make code+doc agree:
  (a) cheap: correct the plan to say CPC is currently single-horizon; (b) `[EXP]`: implement
  multi-horizon CPC (positives at several k, summed InfoNCE) and A/B it. Recommend (a) now,
  (b) only if Phase-2 shows long-horizon structure is the gap. Trade-off: (a) is honesty at
  zero cost; (b) is a real experiment.

**Validation**
- [ ] Each flag defaults to current behavior; `test_encoder.py` gains a residual-path
  shape/gradient/non-identity check and a "close-anchor normalization" check.
- [ ] `[EXP]` For each experiment: retrain from the Phase-2 baseline init, re-score
  diagnostics, record a one-line keep/drop decision vs. baseline in the plan.

**Expected outcome:** The encoder matches its own design intent where it helps, with each
change's effect independently attributable (the plan's own 1A→1D discipline, applied to
architecture).

**Exit criteria:** Each experiment has a recorded keep/drop decision with before/after
diagnostics; kept changes' flags flipped to default-on; doc and code agree on CPC horizon.

**Dependencies:** Phases 1+2 (need trustworthy diagnostics and a real baseline to A/B against).

---

## Phase 5 — Research: does Stage 1B's alignment loss learn identity, not forward prediction?

**Objective:** The forward cross-modal alignment loss is trivially satisfiable by encoding
*company identity* (the positive shares identity with the anchor; the different-stock
negative doesn't), rather than genuine forward fundamental prediction. This is an argued
concern **corroborated** by the observed v1 signature (alignment train loss collapsed
0.30→0.05 in ~50 steps while holdout never improved) — but it is a hypothesis to test, not a
code-confirmed bug. Do the experiment before redesigning the loss.

**Research tasks**
- [ ] `[RESEARCH]` Define an identity-shortcut probe: from the alignment anchor embedding,
  fit a linear classifier for ticker identity. High accuracy + flat forward-prediction skill
  = shortcut confirmed.
- [ ] `[RESEARCH]` Measure whether the alignment positive's fundamentals embedding actually
  *changes* between t and t+`alignment_horizon` for the pairs sampled (if it rarely changes,
  "forward prediction" is vacuous regardless of identity).

**Experiments**
- [ ] `[EXP]` Ablation: retrain 1B with the different-stock-same-time negative **removed** vs.
  present. If diagnostics barely move, identity was carrying the loss.
- [ ] `[EXP]` If shortcut confirmed: redesign the positive to require *change* prediction
  (predict the Δ of the fundamentals embedding t→t+k, or contrast the *next filing's*
  embedding against the *current-quarter* embedding as a hard negative). A/B vs. the current
  form.

**Validation**
- [ ] Identity-probe accuracy recorded before/after the redesign; forward-prediction skill
  (does the anchor rank the true future fundamentals above a same-ticker *different-time*
  fundamentals?) recorded.

**Expected outcome:** A clear verdict on whether 1B tests what it claims, and — if not — a
loss form that does. This directly protects the project's central "fundamentally predictive,
not price-autocorrelation" bet.

**Exit criteria:** Probe + ablation results recorded; explicit keep/redesign decision for the
alignment loss written into the plan's Stage 1B block.

**Dependencies:** Phases 1+2 (trustworthy diagnostics + baseline).

---

## Deliberately NOT addressed (confirmed non-issues or negative-value work)

- **Overlap deflation `n_eff = n_obs/(lag+1)`** — verified *consistent* with the downstream
  Bartlett-kernel NW t-stat (triangular VIF for exact-overlap MA(lag) equals `lag+1`).
  "Fixing" it would introduce an inconsistency. Leave.
- **`min_detectable_ir` has no overlap term** — correct: it runs on non-overlapping monthly
  active returns, paired with `newey_west_tstat(lag=0)`. Leave.
- **`1/√(n−1)` vs `1/√(n−2)`** — `n−1` is the right Spearman null SE. Leave.
- **Shared train/eval RNG** (12c) — only affects exact cross-cadence reproducibility, not
  results. Not worth the churn now; revisit only if a reproducibility audit needs it.

---

## Cross-cutting notes

- **Testing convention:** all new tests are synthetic/assert-based, added to the `fast` group
  in `tests/run_all.py` (no real-data dependency), per repo convention.
- **Training runs:** every `[EXP]` is written but executed by the user (CLAUDE.md). Each
  produces a checkpoint + diagnostics JSON recorded in the plan.
- **One hypothesis at a time:** Phases 4/5 change the model; each item is flag-gated
  (default = current) and A/B'd in isolation so any diagnostic movement is attributable.
