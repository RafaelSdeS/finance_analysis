# Conviction Model — Design & Implementation Plan

**Status:** design draft, not yet coded. Independent exploratory track — not a
replacement for `src/rl_agent` (EIIE) or `src/h_series` (cross-sectional factor model).
A **shared encoder + shared pooled regressor** (see naming note below) trained across
the top-50 universe, producing a per-(ticker, date) prediction rather than a single
own/cash decision at one horizon.

**Naming note:** earlier drafts called this a "per-ticker conviction model," which
mischaracterized the architecture — the encoder and regressor are both shared/pooled
across all top-50 names, not independent per ticker (that was the deliberate fix for
the small-per-entity-data problem in v1 of this discussion). It's a global model that
scores each instrument, not a collection of per-instrument models. Kept the file name
(`conviction_model`, still accurate for what it *outputs*) but dropped "per-ticker" from
the description throughout.

**Revision note (v3):** incorporates a second review round —
1. multi-output horizon targets replace the single decay-weighted scalar label (biggest
   change: preserves the full return-path shape instead of collapsing it),
2. an explicit "raw features vs. PCA vs. plain autoencoder vs. SSL encoder" ablation
   ladder is now mandatory in Phase 2, not optional,
3. CatBoost and LightGBM are added to the model-class competition (a deliberate,
   justified exception to the earlier "no new dependencies" stance — see Phase 2),
4. uncertainty is elevated from "free output" to a required input to any downstream use
   of the scores,
5. signal validation (does higher conviction correlate with better realized outcomes)
   is now its own phase, decoupled from and prior to any portfolio-allocation stub,
6. an explicit, un-fixed limitation is recorded: the label is still an outcome
   regression, not a direct measure of decision quality (see Labels, below).

**Revision note (v4):** the encoder is the actual hypothesis this project tests — if the
representation doesn't capture useful structure, no downstream regressor recovers it,
and if it does, several different tabular models will likely work on top of it. Treated
that seriously:
1. Phase 1 is expanded from "build the encoder + one diagnostic" into a full,
   self-contained phase that must be resolved *before* any downstream regressor work
   starts — architecture, SSL objectives, an explicit definition of what a "good"
   embedding is, and a battery of diagnostics that test the embedding's intrinsic
   properties, independent of downstream regressor performance.
2. The fusion step changes from "cross-attend, then pool into one vector" to
   "cross-attend, then keep each branch's updated token as its own sub-embedding" — the
   downstream model gets 4 labeled sub-embeddings (daily/weekly/monthly/fundamentals),
   not one opaque 256-d vector, trading a little of cross-attention's fusion power for
   interpretability, per-branch ablation, and debuggability (a tree can now report which
   branch's embedding actually carries importance).
3. The SSL objective gets one addition, deliberately scoped small: a low-weight
   auxiliary probe predicting known valuation ratios from the embedding during
   pretraining (not a new primary loss) — a direct, honest answer to "why would SSL
   discover investing-relevant structure rather than, say, volatility clustering,"
   argued in Risks below.

**Revision note (v5):**
1. Phase 1's 4 losses are no longer introduced together — staged as 1A→1D, each stage
   adding exactly one loss and re-running the diagnostic battery, so any improvement (or
   lack of one) is attributable to the loss that was just added, not a bundle of four
   simultaneous changes (the same "isolate one variable at a time" discipline
   `H3A_MULTI_HORIZON_SCREENING_PLAN.md` already uses for characteristic screening).
2. The 8 diagnostics get explicit quantitative gates (below), not just "report and eyeball."
3. Uncertainty is split into aleatoric (irreducible — the future itself isn't
   predictable) vs. epistemic (the model hasn't seen anything like this) — documented as
   a distinction, not resolved; Phase 5's allocation layer will need it eventually.
4. An embedding-trajectory visualization (one ticker, embedded monthly over its full
   history, projected to 2D, annotated with known events) becomes a required Phase 1
   deliverable, not an afterthought.
5. Phase 2's full Cartesian product (4 representations × 5 model classes = 20 runs) is
   replaced with a staged design: pick the best representation first using one fixed
   "referee" model class, then run the model-class competition only on the winner.
6. The framing sentence "the encoder is the actual hypothesis this project tests" is
   sharpened into an explicit, falsifiable research question (below).

**Revision note (v6):** a third review round confirmed most of its concerns were
already addressed (survivorship bias via point-in-time membership, CDI-relative excess
returns, ticker/sector embeddings, the SSL-objective risk, the trajectory plot's
qualitative status — all present since earlier rounds, cross-checked against the actual
doc text rather than taken on faith). Three genuine additions:
1. **Phase renumbering:** the old Phase 3 (full quarterly-refit production harness —
   warm-start, cold-restart, drift logging) was real engineering investment that
   doesn't need to exist before knowing whether there's a signal. Split into a new,
   cheap Phase 3 (minimal walk-forward signal check) that feeds Phase 4 (the gate,
   unchanged position), with the expensive production harness demoted to Phase 5,
   built only if Phase 4 passes. Allocation moves to Phase 6, final holdout to Phase 7.
2. Phase 4 gets two companion metrics beyond rank-IC: a long-short decile spread (the
   standard, portfolio-optimizer-free factor test) and explicit sub-period stability
   (does the signal hold across multiple ~5-year windows, not just pooled — the same
   discipline that would have caught H2's single-path artifact earlier).
3. Market-cap-bucket embedding added alongside ticker/sector; CPC's negative sampling
   is now specified (same-stock-different-regime and different-stock-same-time
   negatives) instead of left generic; quantile regression is added as an uncertainty
   method Phase 2 compares against ensemble spread, not assumed inferior.

**Revision note (v7):** Phase 0's Step 0 was actually run against the real
`top50_universe_membership.parquet` (2026-07-21) — this is a measured result, not a
projection. `n_monthly_obs=305`. Per-horizon `min_detectable_ic`: k=21 → 0.0164, k=63 →
0.0283, k=126 → 0.0401, k=252 → 0.0567, k=504 → 0.0801. Against H1's realistic survivor
IC range (0.035-0.088): **k=21 and k=63 are adequately powered; k=126/252/504 are not**
(their floors sit above or near the low end of that range — a null result there can't be
told apart from "not enough data to tell"). This is not a knockout finding — the plan
does not drop the long horizons — but it does change how Phase 4 is allowed to interpret
them:
1. Labels keep all 6 outputs (5 horizons + drawdown severity) — the multi-output vector
   still exists to teach the encoder/regressor about opportunity *shape*, which doesn't
   require every horizon to be independently statistically testable.
2. Horizons split into **primary** (k=21, k=63 — confirmatory, can pass or fail the
   project) and **exploratory** (k=126, k=252, k=504, plus drawdown-severity's own
   504-day dependency — descriptive, reported, cannot independently fail the project).
3. Phase 4's gate is redefined around the primary pair only; exploratory-horizon results
   are still computed and reported (rank-IC, decile spread, stability) but labeled
   "underpowered exploratory" rather than pass/fail.
4. A configurable horizon loss weight is added to the regressor stage (Phase 2) so
   models with a combined multi-output loss can down-weight the exploratory horizons
   during fitting rather than treating a 6-vector as 6 equally-weighted targets by
   default. (v8 raised the exploratory default from 0.25 to 0.5 — the weight controls
   *learning* emphasis, decoupled from the *gate*; see v8 note above.)
See Phase 0, Labels, Module layout, Phase 2, and Phase 4 below for the concrete edits;
`docs/conviction_model/PHASE0_POWER_FLOOR.json` holds the raw numbers.

**Revision note (v8):** the confirmatory gate landing on k=21/k=63 (v7) was a statement
about what the 50-name/305-month sample can *prove*, and got conflated with what the model
should *learn* — the 0.25 exploratory loss weight starved long-horizon learning for a purely
statistical reason. The agent is meant to be capable of long-term investing (go with the
market, not just trade 1-3 month moves), so the two are separated:
1. **Learning ≠ gating.** Exploratory horizons stay in the training loss at a real weight
   (default raised 0.25 → 0.5) so the encoder/regressor actually learn long-horizon structure;
   the gate covering only the adequately-powered horizons (whichever Phase 0's floors admit —
   k=21/k=63/k=126/k=252 under the v9 top-150 universe) only limits what the project can
   *claim*, not what it optimizes. Prefer the jointly-optimized multi-output candidates
   (CatBoost native, MLP) so any still-underpowered long horizon borrows strength from the
   well-powered ones
   (term-structure regularization) instead of being fit in isolation.
2. **The one real lever for a *confirmable* long-term signal is a wider label universe.** The
   encoder already sees all 293 names; only the label/regressor universe is pinned to top-50.
   Widening it to ~top-150 drops `sigma_ic_null` from 1/√49=0.143 to 1/√149=0.082, bringing
   k=126 (floor ≈0.023) and k=252 (≈0.033) below H1's 0.035 lower bound — i.e. actually
   powered — with k=504 improving to ≈0.046 (still short). Recorded as the concrete path to
   testing the long-term hypothesis for real (see the final Risk); it needs a top-150
   membership file built and accepts less-liquid names, so it's a scoped follow-up, not a
   silent default.
3. **"Go with the market" floor** (Phase 6): the allocation layer defaults toward the
   benchmark / equal-weight under low conviction or high uncertainty, taking active tilts only
   where conviction clears a bar — a long-run market-tracking floor that also directly counters
   EIIE's documented all-cash/all-in bistable failure (CLAUDE.md). (Reframed in v9 — cash at
   SELIC is a first-class holding, not excluded by this floor; see below.)

**Revision note (v9):** third design conversation with the user —
1. **H1's 0.035 is a *reference expectation*, not a hard gate.** It's the lower end of H1's
   realized-survivor IC range, used only to ask "is an effect this size even plausible here?"
   The actual statistical bar everywhere downstream is significance (t>2 / permutation-null via
   `min_detectable_ic`), never "cleared 0.035." A horizon whose power floor sits *below* 0.035
   means "a realistically-sized effect, if real, would clear significance in this sample";
   *above* it means "even a good effect might not be distinguishable from noise here." So the
   floor-vs-0.035 comparison is a feasibility check on sample size, not a pass/fail on the
   signal — Phase 0/4 language clarified accordingly.
2. **Top-150 label universe adopted** (v8's high-leverage lever, now the plan, not a maybe).
   Built by reusing `build_dataset/build_top50_universe.py::build_top50_membership(df,
   top_n=150)` verbatim — the *same* point-in-time construction (trailing-252d `traded_amount`
   rank, quarterly rebalance, `merge_asof` backward with a non-trading tolerance,
   lock-until-next-rebalance, union-recovers-delisted) that produced
   `top50_universe_membership.parquet`, differing only in `top_n` and output path
   (`top150_universe_membership.parquet`). **No new anti-lookahead / survivorship logic is
   written or re-validated — it inherits `TOP50_UNIVERSE_VALIDATION.md`'s guarantees by
   construction** (the function is already fully parameterized on `top_n`). Only the membership
   table is needed here (`active_universe_by_date()` consumes it); the filtered wide-universe
   parquet is optional since the encoder reads full `ml_dataset.parquet` regardless.
   Consequence: at n_assets=150, `sigma_ic_null` drops 1/√49=0.143 → 1/√149=0.082, so **k=126
   (floor ≈0.023) and k=252 (≈0.033) join k=21/k=63 as primary/powered; only k=504 (≈0.046)
   stays exploratory** (still above 0.035, though much improved). Phase 0's power floor is
   re-run at n_assets=150 to record this formally (`min_detectable_ic` is closed-form; n_obs is
   unchanged by universe width, re-confirm ≈305 on the actual top-150 build).
3. **Cash-at-SELIC is a first-class holding; "go with the market" is a floor, not a
   constraint.** The agent may sit in CDI/SELIC cash whenever it sees fit (the labels are
   already CDI-relative, so cash is the natural neutral). "Go with the market" means only that
   the model should be *capable of* benchmark-comparable returns — never structurally stuck
   below a sensible benchmark (equity index *or* CDI) — not that it must hold equities. Phase
   6's floor is reworded: under weak conviction/high uncertainty the neutral default is a
   benchmark-comparable mix that explicitly includes cash-at-SELIC, not forced equity exposure.

## Research question

The central hypothesis: **there exists a latent representation of market state that is
substantially more useful for long-term investment decisions than manually engineered
features or simple feature compression methods.** The encoder is the mechanism used to
test that hypothesis — not preprocessing in service of a downstream regressor, the thing
actually being evaluated. Reframed from where this exploration started ("can a better RL
agent be found?") to a narrower, more falsifiable question: does a learned latent
representation of market state improve long-term investment decisions over conventional
tabular features? Phase 1's diagnostics (intrinsic) and Phase 2's representation
competition (extrinsic, vs. raw features/PCA/autoencoder) together are the test of this
question — everything after Phase 2 only matters if the answer is yes.

## Objective

For each (ticker, date) in the top-50 universe, predict a **vector** of risk-adjusted
CDI-relative excess returns across 5 horizons plus a path-drawdown severity measure —
not a single aggregated score — so that different shapes of opportunity (a sharp
short-term move vs. a slow multi-year compounder) stay visible as different outputs
instead of being collapsed into one number. A single-number "conviction" is available
as a simple downstream reduction for reporting, but it is not the training target. **The
gate-relevant reduction (Phase 4's secondary check) uses the primary pair (k=21/k=63) only**
— a mean over all 5 horizons can also be reported but is contaminated by the underpowered
exploratory horizons, so it never feeds a pass/fail. Every prediction is paired with an
uncertainty estimate, which is a required input to any downstream use of the scores, not
an optional add-on — see Phase 2 and Risks. **All 6 outputs are trained and reported;
under the v9 top-150 universe the confirmatory (primary) set is k=21/k=63/k=126/k=252** —
only k=504 and drawdown-severity remain exploratory (at the old top-50, only k=21/k=63 were
primary; the wider universe lowered the power floor — see the v9 note, Labels, Phase 4).

## Relationship to existing tracks — reuse, don't duplicate

| Need | Reused from | Why not new code |
|------|-------------|-------------------|
| Walk-forward expanding folds | `src/h_series/spine.py::iter_expanding_folds()` / `FoldWindow` | Already does exactly this; `step_months` is a free parameter. |
| Point-in-time top-150 membership (v9) | `src/h_series/spine.py::active_universe_by_date()` + a new `top150_universe_membership.parquet` built by `build_dataset/build_top50_universe.py::build_top50_membership(df, top_n=150)` | Builder is already `top_n`-parameterized; top-150 reuses the identical point-in-time / no-survivorship construction, only `top_n` + output path differ (v9). |
| Leakage-safe forward returns | `src/h_series/spine.py::build_forward_targets()` | Takes any `bench: pd.Series` — pass a CDI cumulative-return index instead of BOVA11 to get CDI-relative targets for free, no fork needed. |
| Monthly decision calendar | `src/h_series/spine.py::monthly_decision_dates()` | Same cadence convention as H-series. |
| Scaler fit-window seam | `src/build_dataset/manifest.py::iter_fit_windows()` / `FitWindow` | Reuse this seam instead of a new one if the encoder or labels need a fitted scaler. |
| Model-class competition, picked by OOS performance | Same convention as `H3_PORTFOLIO_CONSTRUCTION_PLAN.md` stage 2 | "Pick by walk-forward OOS, not preference" is already this project's house rule. |
| Signal validation before portfolio construction | Same convention as H-series' H0/H1 (rank-IC screening) before H3's portfolio-construction stage | This plan mirrors that ordering explicitly (Phase 4, below) — same reasoning: prove the signal exists before spending effort translating it into weights. |
| Bootstrap CIs, Sharpe/Sortino/Calmar | `src/rl_agent/metrics.py::block_bootstrap_ci()` and friends | Already implemented, tested. |
| Report layout convention | `src/rl_agent/plots.py::write_report()`, `experiments/{run}_{ts}/` output shape | Same reproducibility artifacts as EIIE. |
| Encoder framework | `torch` (already a dependency) | No new deep-learning framework. |
| Rank-IC / Spearman correlation utilities | `src/h_series` (used throughout H0-H1 for characteristic screening) | Reuse rather than re-derive rank-correlation code for Phase 4's signal validation. |

## Architecture summary

```
data/processed/ml_dataset.parquet (per ticker, all 293 — encoder sees full history
                                    even for names outside top-50 at a given date)
        │
        ▼
Multi-resolution branches (shared across all tickers):
  daily (60d, small dilated conv) | weekly (2y) | monthly (10y) | quarterly fundamentals
        │
        ▼
Cross-attention update (4 branch tokens attend to each other so each can condition on
the others) → 4 UPDATED branch tokens, kept SEPARATE — not pooled into one vector
        │  trained via 3 primary SSL losses + 1 minor auxiliary probe, added ONE AT A
        │  TIME in Phase 1 (stages 1A→1D) and kept only if each earns its diagnostics
        │  (see "What the latent is for" and Phase 1, below):
        │    - CPC: multi-horizon InfoNCE on future latent state. Negatives sampled
        │      deliberately, not generically: same-stock-different-regime (same ticker,
        │      a distant time window in a different valuation/volatility state) and
        │      different-stock-same-time (a different ticker at the same date, so
        │      market-wide co-movement alone can't trivially satisfy the objective) —
        │      sharpens what "state" the objective is forced to discriminate
        │    - masked reconstruction across branches (auxiliary regularizer)
        │    - cross-modal alignment, FORWARD not just same-time: price/macro state at
        │      t must predict the fundamentals branch's embedding at t+k (not merely
        │      agree with it at t) — biases the representation toward what's
        │      fundamentally predictive, not just price-autocorrelation-predictive
        │    - (minor, low weight) auxiliary linear probe predicting known valuation
        │      z-scores (`pl_zhist_5y`, `pvp_zhist_5y`) from the embedding during
        │      pretraining — a nudge, not a primary objective (see Risks). MASKED on
        │      rows where these are NaN (a ticker's first ~2 years, per CLAUDE.md's
        │      documented FUND_ZHIST_MIN_QUARTERS warm-up rule) — silently training
        │      against NaN would either error or (if naively filled) inject a fake
        │      target on exactly the rows least like the rest of that ticker's history
        ▼
[ 4 labeled sub-embeddings (daily/weekly/monthly/fundamentals) | ticker/sector/
market-cap-bucket embedding | cross-sectional features ] — fed to the downstream model
as separate, ablatable blocks
        │
        ▼
ONE pooled multi-output regressor, trained across all top-50 tickers × history — model
class AND input representation both picked in Phase 2 by walk-forward OOS competition
(see Phase 2: candidates include RF/HistGB/CatBoost/LightGBM/MLP, and raw-features/
PCA/autoencoder/SSL-encoder as competing input representations); tree-based candidates
report per-branch feature importance for free, giving a direct "is this branch useless"
signal without a separate ablation study
        │
        ▼
6 outputs per (ticker, date): risk-adjusted CDI-relative excess return at each of
{21, 63, 126, 252, 504} trading days + one path-drawdown-severity measure, each with an
uncertainty estimate (ensemble spread)
```

**Regressor, not classifier+calibration** — the targets are continuous. **Multi-output,
not a single aggregated scalar** (see Labels) — preserves the shape of the opportunity
instead of collapsing it. **Shared, not per-ticker** (see naming note) — pooling avoids
the small-per-entity-data problem and handles top-50 membership turnover for free.
**Separate branch embeddings, not one fused vector** — cross-attention still lets
branches inform each other, but the downstream model (and a human debugging it) sees
4 labeled pieces, not one opaque vector; dropping a branch is a one-line ablation, not
a retrain of the fusion layer.

## What the latent representation is for

The embedding represents **market state** — the joint condition implied by price,
fundamentals, macro, and dividend history at a given moment — rather than "company
characteristics" alone, because the encoder's inputs jointly define that state and
nothing in the architecture privileges the company-specific view over the
macro/market-cycle view. Two (ticker, date) pairs in genuinely similar market states
should land close together in latent space even if they're years apart.

This is the concrete criterion the SSL losses are chosen to serve:
- CPC's InfoNCE predicts future **latent state**, not future price level — pushes
  toward temporal consistency of state rather than pure price autocorrelation.
- Forward cross-modal alignment forces price/macro state at t to predict the
  fundamentals branch's future embedding, not just agree with it at t.
- Masked reconstruction is the weakest of the three for this goal and is kept as a
  regularizer, not the primary objective.
- The auxiliary valuation-probe nudge (Architecture, above) is a direct, if small,
  push toward "this embedding should be able to recover known valuation information,"
  which is otherwise not guaranteed by any of the other three losses.

### What a "good" embedding means, made concrete — 8 diagnostics (6 here + 2 original below), independent of downstream regressor performance

Phase 1 is not done when the encoder trains without error — it's done when this battery
is computed and reported. None of these need the regressor from Phase 2 to exist; that's
the point (isolates encoder quality from downstream-model quality, so a bad Phase 2
result can be traced to the right stage). All reuse `sklearn` (`KMeans`,
`LinearRegression`, `mutual_info_score` — already dependencies), no new libraries:

1. **Nearby embeddings → similar future outcomes.** For a sample of (ticker, date)
   points, compare the spread (variance) of realized 12-month risk-adjusted excess
   return among each point's k-nearest embedding neighbors (excluding the same ticker
   within a short window, to avoid trivially matching on autocorrelation) against the
   spread among k random points. A useful embedding's neighbor-outcome spread should be
   materially lower than random.
2. **Market regimes cluster together.** Unsupervised `KMeans`/GMM on embeddings; check
   cluster assignment against known regime indicators (SELIC-level tercile, realized
   market volatility tercile, market drawdown state) via mutual information, compared
   against a permutation-null baseline (reusing the project's existing permutation-null
   convention from H-series rather than inventing a new significance test).
3. **Valuation regimes emerge naturally — the direct test of the "SSL might learn
   volatility clustering instead of value" failure mode (Risks, below).** Linear probe
   (`LinearRegression`, standard representation-learning evaluation technique): fit
   embedding → known valuation z-score (`pl_zhist_5y`, `pvp_zhist_5y`), report R². Run
   the same probe against realized volatility as a comparison point — if the volatility
   probe's R² dominates the valuation probe's, that's direct, early evidence the encoder
   learned the wrong thing, cheaply caught before Phase 2's full competition. Same NaN
   mask as the auxiliary probe loss — rows without a defined 5-year z-score are excluded,
   not filled.
4. **Company quality represented consistently over time.** Fit a similar linear probe
   against a quality proxy (e.g. ROE z-score); check its autocorrelation at long lags
   (should be high — quality is structurally persistent) versus short-lag noise.
5. **Stability under small perturbations.** Add small Gaussian noise to raw inputs
   (calibrated to a small fraction of each feature's scale) and measure the resulting
   embedding delta — should be small and roughly proportional to the input perturbation
   (an empirical local-Lipschitz estimate), not a discontinuous jump.
6. **Smooth evolution absent new information.** Period-over-period embedding drift
   should be small on non-event days/quarters and larger around genuine informational
   surprise (an earnings filing date, a large realized move) — measured as the
   correlation between embedding-delta magnitude and a simple surprise proxy.

**Plus the original two checks, unchanged:**
7. *Latent-similarity check:* pairs of (ticker, date) with similar fundamental/technical
   readings at different points in time should have smaller embedding distance than
   random pairs.
8. *Embedding-value check (Phase 2, mandatory — see below):* a regressor trained on the
   learned embedding must out-perform the same regressor trained on raw features, PCA,
   and a plain (non-contrastive) autoencoder of the same dimensionality, on walk-forward
   OOS error. If it doesn't, the SSL machinery isn't earning its complexity and the
   honest conclusion is to drop it in favor of the simplest representation that matches
   performance — a gating check, not an optional nicety.

Diagnostics 1-7 are intrinsic (no downstream model needed) and gate whether Phase 2 is
even worth running; diagnostic 8 is extrinsic and gates whether the SSL encoder
specifically (vs. simpler representations) is worth keeping.

## Data & universe

- Encoder input: full `ml_dataset.parquet` history per ticker (not restricted to universe
  membership dates) — mirrors the existing rationale in `src/rl_agent/data.py`. Unchanged by
  the top-150 decision: the encoder always saw all 293 names.
- Label / regressor-training universe: restricted to **top-150** active membership (v9) via
  `active_universe_by_date()`, reading the new `top150_universe_membership.parquet` (built by
  `build_top50_membership(df, top_n=150)` — same point-in-time construction as top-50, see the
  reuse table and v9 note). Wider universe → lower `sigma_ic_null` → k=126/k=252 become
  primary/powered; also a breadth gain (Fundamental Law of Active Management).
- Decision cadence: monthly (`monthly_decision_dates()`), matching the H-series
  convention.
- **`status` is explicitly excluded from every input schema (encoder, ticker/sector
  embedding, cross-sectional features).** Per CLAUDE.md, `company_info`'s `status` field
  is a current-day snapshot joined onto every historical row — a documented,
  already-fixed-once feature-level lookahead trap in this exact dataset, not a
  hypothetical risk. `sector` is lower-risk (static but carries far less outcome
  information per the same CLAUDE.md note) and is fine to use.

### Input features, per branch — corrected: macro and dividends were named in prose
("the encoder captures price, fundamentals, macro, and dividend history") but never
actually assigned to a branch or column list anywhere before this. Fixed here, using
`ml_dataset.parquet` columns that already exist (no new Stage 2 work):

| Branch | Concrete columns |
|---|---|
| **Daily** (60d window) | `close`/`high`/`low`/`volume` (price-level, ÷value-at-t per eq. 18 convention), `return_1m/3m/6m`, `price_vs_ma60`, `volatility_ratio_20_60`, `rsi_14`, `drawdown`, `volume_ratio_20d` — the exact channel set already validated for EIIE (`rl_agent/data.py::FEATURE_NORM`), reused rather than re-chosen from scratch. Pure price/volume technicals only — no macro here (see Monthly, corrected below). **CDI is deliberately not a raw daily feature anywhere in this table** — it's used only as the cash benchmark in `build_forward_targets()` for the labels (below), not fed to the encoder directly, both because SELIC/CDI move closely together in Brazil and to avoid a mildly circular signal from feeding the label's own benchmark back in as an input. |
| **Weekly** (2y window) | Same price-level + technical channels as the daily branch, resampled to weekly resolution — **last daily row per period, ffill gaps** (`data.py::resample_branch_frame`): a point-sample at period end matching the decision-date convention, deliberately *not* a within-period average (an RSI averaged over a week isn't an RSI). Same rule for the Monthly branch. |
| **Monthly** (10y window) | Same price-level + technical channels, resampled to monthly resolution, **plus macro**: `excess_return` (`log_return − selic/252`), `real_return` (`log_return − ipca/252`), `selic_trend_20d` (`selic − selic.shift(20)`) — the actual 3 columns `compute_macro_features()` produces (corrected twice now: an earlier draft named a 4th column, `rate_environment`, that doesn't exist in the source, and had placed all 3 in the Daily branch — wrong cadence match. SELIC changes roughly every 45 days (COPOM meetings) and IPCA is a monthly print, so within a 60-day daily window these are almost constant and `selic_trend_20d` is usually exactly 0; a 10-year monthly window actually spans many full rate cycles, where the same columns carry real, varying signal instead of diluting a branch meant to specialize in fast daily dynamics). |
| **Quarterly fundamentals** | `pl`/`pvp`/`roe`/`net_margin`/`ebitda_margin`/`debt_equity`/`net_debt_ebitda`/`earnings_yield`/`book_to_market`/`current_ratio`/`asset_turnover` (+ their `*_zhist_5y` own-history z-scores), `cagr_earnings_5y_final`/`cagr_revenue_5y_final` (+ `_defined` flags, `n_quarters_available`). **Dividends folded in here**, not a separate branch: `div_yield_12m`, `div_count_12m`, `has_dividends` — already quarterly-cadence-ish (trailing 12m windows), same cadence-matching reasoning that now places macro in Monthly rather than Daily — kept as its own branch rather than merged with Monthly since fundamentals are company-specific, not market-wide, a different modality worth keeping separable for the per-branch ablation this architecture is built around. |
| **Ticker/sector/market-cap-bucket embedding** | Categorical: ticker ID, `sector` (not `status`), a market-cap bucket derived from `market_cap` (already computed via `recompute_valuation_daily()`). |
| **Cross-sectional features** | Reused directly from `src/build_dataset/cross_sectional.py::compute_cross_sectional_features()` — sector/market-relative features already computed in Stage 2, not reinvented here. |

`pl`/`pvp` raw levels are notoriously heavy-tailed (CLAUDE.md: 27-30% NaN, tails to
±2000) — this plan uses the `*_zhist_5y` robust z-scored versions as the primary input,
not the raw ratios, for exactly the reason CLAUDE.md already flags them as not yet
wired into `rl_agent`: raw levels would blow out conv/tree gradients without a
non-linear squash this plan doesn't otherwise need to build.

**`div_yield_12m`/`div_count_12m` are not double-counting dividends already baked into
price.** Per CLAUDE.md, `adj_close` (and everything derived from it — `return_1m/3m/6m`,
`price_vs_ma60`, etc., used throughout the daily/weekly/monthly branches) already bakes
in dividend reinvestment, confirmed empirically, not just splits. So the price-based
branches already reflect *that* a dividend happened, as part of aggregate total return.
The explicit dividend columns add something different, not redundant: dividend *policy*
(how much yield, how often) — a stock with the same adj_close-implied total return can
be a high-yield/low-appreciation name or a zero-yield/high-appreciation name, and that
distinction is exactly the kind of structural information this plan wants the
fundamentals branch to carry. CLAUDE.md reaches the same conclusion for the base
dataset ("not double-counted into returns") — this plan inherits that reasoning rather
than re-deriving it. One caveat worth carrying forward, not solving: CLAUDE.md also
documents a known, unexplained ~5pp median (sometimes 20pp+) divergence between what
BolsAI's `adj_close` methodology implies about dividend adjustment and what
`data/raw/dividends` alone would predict — so `div_yield_12m` (computed from the
dividends table) and the adj_close-implied effect won't perfectly reconcile. Expected,
not a bug if a diagnostic later notices it.

## Labels — multi-output, per-horizon, CDI-relative

For horizons k ∈ {21, 63, 126, 252, 504} trading days (~1/3/6/12/24 months):

1. Build a CDI cumulative-return series as the `bench` argument to
   `build_forward_targets(prices_wide, cdi_index, decision_dates, k, universe)` —
   forward k-day return over cash, per (ticker, decision_date), leakage-safe, already
   restricted to that date's top-50 universe. **`prices_wide` is `adj_close`** (matching
   `build_forward_targets`'s own docstring in `spine.py`), not raw `close` — made
   explicit here since it wasn't stated before. This means the label is already total
   return (dividend-inclusive, per CLAUDE.md's verified `adj_close` finding), which is
   exactly the right measure for "would you rather own this than cash" — no separate
   dividend adjustment needed on the label side, and none should be added.
2. Risk-adjust each horizon's raw excess return by its trailing realized volatility.
3. **No aggregation into a single scalar.** Predict the 5 horizon-specific risk-adjusted
   excess returns as separate regression targets, plus one additional target — max
   peak-to-trough decline in cumulative excess return along the daily path from
   decision date to the 24-month horizon's end — capturing path shape (a position that
   goes up then round-trips) as its own output rather than folding it into a decayed
   average. This directly preserves what a single aggregate would destroy: e.g.
   `+15%/+14%/+13%/-8%/-10%` (short-term-only opportunity) and
   `+0.1%/+0.2%/+0.3%/+0.4%/+40%` (long-term compounder) now stay visible as different
   6-vectors instead of collapsing to comparable scalars. Any single-number "conviction"
   for reporting is computed downstream from this vector (Objective, above), never
   trained as its own target.
4. `RandomForestRegressor` and most GBM libraries support multi-output natively or via
   a thin wrapper (`sklearn.multioutput.MultiOutputRegressor` for the ones that don't) —
   no custom multi-task architecture needed for the regressor stage. The encoder's CPC
   loss is already inherently multi-horizon (predicts multiple future-k latents), so
   multi-output labels are a natural fit, not new machinery layered on top of a
   single-horizon design.
5. **Primary vs. exploratory horizon status (v7 measured at top-50; v9 updated for top-150):**
   under the adopted top-150 universe, **k=21/k=63/k=126/k=252 are primary** — their power
   floors (0.0094/0.0163/0.023/0.033) sit below H1's 0.035 reference, so their Phase 4 results
   can confirm or fail the project. **Only k=504 (floor ≈0.046) and, by extension, the
   drawdown-severity target (same 504-day window) stay exploratory** — real 6-vectors still,
   still trained and reported, but a null result there means "underpowered," not "no signal,"
   and it can't independently fail the project. (At the old top-50 only k=21/k=63 were primary;
   the wider universe promoted k=126/k=252 — see v9 note. 0.035 is a reference, not a hard gate;
   the actual bar is significance.) This label stays attached to each horizon everywhere
   downstream — see Phase 2's loss weighting and Phase 4's redefined gate.

### Known, un-fixed limitation: this is still an outcome regression, not a decision-quality measure

A company can be genuinely undervalued at time t and still receive a bad label if an
unforeseeable shock (war, commodity crash, recession) hits afterward — the label
reflects what happened, not whether the decision was reasonable given what was knowable
at t. This is not something this plan fixes; any label ultimately grounded in realized
returns has this property, and avoiding it entirely would mean going back to
hand-crafted "quality/value" scores, which is exactly what the self-supervised
representation is meant to replace. The practical mitigation, not a fix: pooled across
thousands of (ticker, date) examples, exogenous shocks are assumed to be uncorrelated
with the true, unobservable "was this a reasonable decision at t" signal, so they act as
label *noise* the regressor averages out over the full training set rather than as a
systematic *bias*. That assumption is checked, not just asserted — Phase 2's residual
analysis should confirm prediction errors don't cluster predictably around
macro-shock periods (if they do, the errors are structured, not just noisy, and the
mitigation doesn't hold). Flagged here so this ceiling isn't rediscovered as a bug
later.

## Module layout — `src/conviction_model/`

| File | Purpose |
|------|---------|
| `config.py` | Frozen-dataclass config (branches, SSL hyperparameters, label horizons, retrain cadence, purge/embargo days, regressor + representation candidate sets, per-horizon loss weights — v7) ↔ JSON. |
| `paths.py` | Shared path constants for this package. |
| `data.py` | Loads `ml_dataset.parquet`, builds per-ticker daily/weekly/monthly/quarterly window tensors. |
| `encoder.py` | `EncoderCNN` — 4 branches + cross-attention update, returns 4 separate labeled sub-embeddings (not one pooled vector). |
| `ssl_pretrain.py` | 3 primary losses (CPC, masked reconstruction, forward cross-modal alignment) + 1 minor auxiliary valuation-probe loss, weighted sum, checkpointing. |
| `diagnostics.py` | The 6 intrinsic embedding-quality diagnostics + their quantitative gates (see Phase 1's gate table), run after each of stages 1A-1D on the frozen encoder, no regressor involved. Also renders the embedding-trajectory visualization (PCA projection, reuses `rl_agent/plots.py`'s plotly convention). |
| `baselines_repr.py` | The 3 non-SSL representations competed against the encoder in Phase 2: raw feature vector, PCA (`sklearn.decomposition.PCA`, already a dependency), plain autoencoder (reconstruction loss only, same branch structure as `encoder.py` minus the contrastive/alignment losses, isolates "does contrastive/alignment learning help over plain compression"). |
| `labels.py` | Calls `h_series.spine.build_forward_targets()` per horizon with the CDI bench; risk-adjusts; assembles the 6-output target vector (5 horizons + drawdown severity). No aggregation step. |
| `tree.py` | Multi-output regressor, staged per Phase 2: **2a** picks the best input representation (raw features / PCA / autoencoder / SSL encoder's 4 sub-embeddings from `baselines_repr.py`) using a fixed `RandomForestRegressor` referee; **2b** competes `RandomForestRegressor` / `HistGradientBoostingRegressor` / `CatBoostRegressor` / `LightGBM` / small MLP only on the 2a winner. Both picked by walk-forward OOS error (H3 stage 2 convention). **Not all candidates support multi-output natively** — `RandomForestRegressor` and `CatBoostRegressor` do; `HistGradientBoostingRegressor` and `LightGBM` need `sklearn.multioutput.MultiOutputRegressor` (fits 6 independent single-output models, no cross-output structure sharing, ~6x slower to fit) — noted so 2b's comparison is read correctly, not treated as apples-to-apples on training cost. **Uncertainty method is itself compared, not assumed:** ensemble spread (free for RF/CatBoost/HistGB/LightGBM) vs. quantile regression (predict Q10/Q50/Q90 per output — CatBoost/LightGBM/HistGB support quantile loss natively, RF doesn't without a wrapper) — picked alongside the model class by the same walk-forward OOS process, since quantile regression gives an asymmetric interval (useful — return distributions aren't symmetric) at the cost of 3x the outputs to fit. **Per-horizon loss weight (v7, revised v8):** `config.py`'s weight vector defaults to primary=1.0 (k=21, k=63), exploratory=0.5 (k=126/252/504, raised from 0.25 in v8 so long-horizon structure is actually learned — the weight is a *learning*-emphasis knob, not the statistical gate), drawdown=1.0 — `loss = Σ w_i · loss_i`. Exploratory stays below primary because Phase 0's floors say those horizons can't be independently *confirmed*, but stays well above zero because the agent is meant to learn long-term investing (v8); jointly-optimized candidates (CatBoost native, MLP) let the long horizons borrow strength from the well-powered short ones. Applies literally to the MLP candidate (an explicit combined loss) and to `CatBoostRegressor`'s native multi-output mode (per-output weighting supported); for the `MultiOutputRegressor`-wrapped candidates (HistGB, LightGBM), each output is fit independently, so the "weight" there only informs which outputs get more hyperparameter-tuning attention, not a jointly optimized loss — noted so the comparison across candidates isn't read as apples-to-apples on this dimension either. Weights are config, not hardcoded — adjustable if Phase 1/2 data suggests otherwise. |
| `walkforward.py` | Wrapper around `spine.iter_expanding_folds()`: regressor refit and encoder warm-start fine-tune share a quarterly cadence; full cold-restart encoder retrain every ~3 years (configurable) to guard against warm-start drift. Purge/embargo filter drops training rows whose 504-day label window extends past `train_end`. |
| `signal_validation.py` | Phase 4: rank-IC / Spearman correlation computed **per raw output** (each of the 6 predictions against its own realized value, walk-forward OOS) — corrected to match Phase 4's own text; an earlier draft of this row described it against a single "downstream-reduced" conviction number, which the Phase 4 section itself already fixed but this row hadn't caught up to. The reduced conviction number gets its own secondary check, reported alongside, not in place of, the per-output results. Also computes the long-short decile spread (gross and net-of-cost) and sub-period stability. Reuses H-series' rank-correlation utilities. **Primary/exploratory split (v9, top-150):** the pass/fail gate is computed from k=21/k=63/k=126/k=252; only k=504 and drawdown-severity are tagged `underpowered exploratory` and excluded from the gate decision. This is the primary success gate, prior to and independent of any allocation stub. |
| `backtest.py` | **Deferred, not designed in this plan** — see Phase 6. Only entered if Phase 4's signal-validation gate passes. |
| `experiment.py` | CLI orchestrator: load data → labels → walk-forward loop → signal validation → (if gated) backtest → report. |

## Implementation phases

### Phase 0 — Power-floor prerequisite, then label & fold plumbing

- [x] Step 0: compute H0's power floor for this plan's exact setup — **not** by running
      `milestone_h0.py` directly (corrected: that script is hardcoded to
      `K_HORIZONS = (21, 63)` and has no CLI/parameterization; this plan needs 5
      horizons, including 126/252/504, which it doesn't cover). Call the actual
      reusable primitive it's built on instead: `src/h_series/stats.py::min_detectable_ic(n_obs, n_assets, lag)`
      — already a clean, parameterized, closed-form function, not something to
      reimplement. `n_assets=50` already matches this plan's universe exactly (same
      constant `milestone_h0.py` uses). `lag` per horizon comes from
      `spine.py::hac_lag_for_horizon(k)` (also already reusable). Same prerequisite
      check `H3A_MULTI_HORIZON_SCREENING_PLAN.md` requires before screening at a new
      horizon — this plan just calls the primitive directly instead of trying to run a
      script that wasn't built for external parameters. Implemented as
      `src/conviction_model/check_power_floor.py`.
- [x] Compare the power floor against realistic effect sizes (H1's actual survivors:
      0.035-0.088) and record an explicit go/underpowered decision before Phase 1 starts.
      **Run against the real `top50_universe_membership.parquet` (2026-07-21):**
      `n_monthly_obs=305`, `n_assets=50`. Per-horizon floors: k=21 → 0.0164 (HAC lag 0mo),
      k=63 → 0.0283 (2mo), k=126 → 0.0401 (5mo), k=252 → 0.0567 (11mo), k=504 → 0.0801
      (23mo). **Decision (top-50): k=21/k=63 are GO (below H1's 0.035 reference); k=126/252/504
      are UNDERPOWERED** (floors exceed it, worst at k=504 — needs IC≈0.08, over 2x the
      largest realistic effect size observed elsewhere in this project). Per v7: this
      does not kill the long horizons, it demotes them to exploratory status — see
      Labels, Phase 2, and Phase 4. Raw output: `docs/conviction_model/PHASE0_POWER_FLOOR.json`.
      **0.035 is a reference expectation, not a hard gate (v9)** — the real bar is significance
      (t>2 / permutation-null); the floor-vs-0.035 comparison only asks whether a
      realistically-sized effect could be detected at this sample size.
- [x] **Re-run the power floor at n_assets=150 (v9), record `PHASE0_POWER_FLOOR.json` v2.**
      **Run against the real `top150_universe_membership.parquet` (2026-07-21):**
      `n_monthly_obs=305` (unchanged, as expected — universe width doesn't move the rebalance
      calendar), `n_assets=150`. Measured floors: k=21→0.0094, k=63→0.0162, k=126→0.0230,
      k=252→0.0325, k=504→0.0460 — matches the closed-form v9 projection almost exactly.
      **Decision: k=21/k=63/k=126/k=252 GO (all below H1's 0.035 reference); only k=504 stays
      UNDERPOWERED/exploratory.** Raw output: `docs/conviction_model/PHASE0_POWER_FLOOR_v2.json`.
- [x] **Build `top150_universe_membership.parquet` (v9):** ran
      `python -m src.build_dataset.build_top50_universe --top-n 150 --membership-only`, calling
      `build_top50_membership(df, top_n=150)` verbatim — identical point-in-time /
      union-recovers-delisted construction as top-50, only `top_n` + output path differ; inherits
      `TOP50_UNIVERSE_VALIDATION.md`'s no-lookahead/no-survivorship guarantees. 360 tickers ever
      qualified for top-150 across the full rebalance history. Membership table only (encoder
      reads full `ml_dataset.parquet`; filtered wide parquet not built, per plan).
- [x] `labels.py`: build the CDI cumulative-return series from `data/raw/macro/cdi.parquet`.
- [x] `labels.py`: call `build_forward_targets()` once per horizon (k ∈ {21,63,126,252,504}).
- [x] `labels.py`: risk-adjust each horizon's excess return by trailing realized volatility.
- [x] `labels.py`: compute the drawdown-severity target (peak-to-trough decline along the
      daily path to the 24-month horizon).
- [x] `labels.py`: assemble the 6-output target vector — assert no aggregation step exists.
- [x] `walkforward.py`: purge/embargo wrapper over `iter_expanding_folds()`.
- [ ] Confirm `status` is absent from every input schema (encoder inputs, ticker/sector
      embedding, cross-sectional features) — `sector` is fine, `status` is not. **Not
      yet checkable in code** — no input schema exists until Phase 1's `data.py`/
      `encoder.py` are built; carried forward as a Phase 1 gate item, not a Phase 0 gap.
- [x] Sanity-check the two worked examples (Labels, above) land as visibly different
      6-vectors, not similar scalars.
- [x] `tests/conviction_model/test_labels.py` passes, and `test_walkforward.py`'s
      **Phase 0** assertions pass (fold generation, purge/embargo — its warm-start/
      cold-restart assertions are a Phase 5 gate, not this one; see Testing strategy).

**Phase 0 status: complete.** All label/fold plumbing implemented and tested
(`src/conviction_model/{labels,walkforward,check_power_floor}.py`,
`tests/conviction_model/{test_labels,test_walkforward}.py`, 47/47 fast tests passing),
power floor computed against real data with an explicit primary/exploratory decision
recorded (v7). The one open item (`status` exclusion) moves to Phase 1 since it needs
Phase 1's code to exist before it can be confirmed.

### Phase 1 — Encoder definition + intrinsic quality diagnostics, staged loss by loss

Self-contained; this phase IS the hypothesis under test, not preprocessing for Phase 2.
Phase 2 does not start until Phase 1D is done and reported.

- [x] `data.py`: per-ticker daily/weekly/monthly/quarterly window tensors
      (`load_ticker_daily_frame`/`load_ticker_quarterly_frame` I/O,
      `resample_branch_frame` for weekly/monthly, `window_tensor` pure normalization —
      tested in `tests/conviction_model/test_data.py`).
- [~] Confirm `status` is absent from every input schema built by `data.py`
      (ticker/sector/market-cap embedding, cross-sectional features) — **partially
      resolved:** the 4 branch feature lists (`DAILY_FEATURES`/`WEEKLY_FEATURES`/
      `MONTHLY_FEATURES`/`QUARTERLY_FEATURES`) are confirmed `status`-free (price/
      technical/macro/fundamental columns only). The ticker/sector/market-cap-bucket
      embedding and cross-sectional-features blocks (Architecture diagram — fed to the
      downstream model *alongside* the encoder's 4 sub-embeddings, not through a branch)
      aren't built yet; still open until that loading code exists.
- [x] `encoder.py`: 4 branch sub-networks (dilated conv per resolution) — `_BranchCNN`
      (`Conv1d` + `AdaptiveAvgPool1d`, per-branch feature count threaded from `data.py`).
- [x] `encoder.py`: cross-attention update producing 4 SEPARATE labeled sub-embeddings
      (not one pooled vector) — shape-checked, gradient-checked, determinism-checked, and
      confirmed non-identity, all in `tests/conviction_model/test_encoder.py`.
- [x] `ssl_pretrain.py`: CPC loss (`info_nce_loss`, InfoNCE) + the two specified
      negative-sampling types (`sample_cpc_negatives`: same-stock-different-regime via a
      time-gap heuristic, different-stock-same-time) + `train_step` (one CPC gradient
      step through `EncoderCNN`, mean-pooling the 4 branch tokens into one "market state"
      vector for this loss only) — all pure/tested in
      `tests/conviction_model/test_ssl_pretrain.py`, no real data needed.
- [x] `diagnostics.py`: the 7 intrinsic diagnostic functions (neighbor-outcome variance
      ratio, regime mutual information, linear probe R² [valuation-vs-volatility uses it
      twice], quality-persistence autocorrelation, perturbation sensitivity, temporal
      smoothness, latent-similarity significance) — pure, gate comparison left to the
      caller, tested against known injected/rejected synthetic cases in
      `tests/conviction_model/test_diagnostics.py`. Diagnostic 8 is Phase 2's job, not
      built here.
- [x] **Batch-assembly glue built and pilot-verified against real data.**
      `config.py` (`SSLConfig`, Stage-1A fields only), `data.py::branch_windows_from_precomputed`/
      `build_frame_cache` (weekly/monthly resampled once per ticker, not per lookup), and
      `ssl_pretrain.py::sample_cpc_anchor_positions`/`build_cpc_batch`/`CPCPanelStore` (precomputes
      every panel position's window tensors once, batch assembly becomes `index_select` — mirrors
      `rl_agent/train.py`'s `_PanelStore` pattern, CLAUDE.md's `TRAINING_SPEEDUP_PLAN`). All pure
      pieces tested on synthetic data (`tests/conviction_model/{test_config,test_data,test_ssl_pretrain}.py`).
      Encoder input is full `ml_dataset.parquet` history (515 tickers, 2000–2026 — not the "293"
      figure elsewhere in this doc, which is stale; the top-50/150 universe only restricts *labels*,
      not the encoder, see Data & universe), not the top-50/150 label universe — corrected from this
      item's earlier wording.
      **Pilot/dry-run run** (`pretrain_pilot.py`, Stage 1A/CPC only — Stages 1B-1D aren't written yet,
      so this isn't the full 4-stage pilot the checklist below still needs): 20 steps, 5 tickers
      (PETR4/VALE3/ITUB4/BBAS3/ABEV3), `SSLConfig` defaults. Loss fell 1.5774→0.2742 (min 0.2672),
      finite throughout — real signal, not noise, on the first real-data run. Found and fixed two
      real bugs surfaced only by running against real data (exactly what the dry-run is for): (1)
      `window_tensor` raised on an empty pre-`as_of` slice instead of left-padding — legitimate for
      an `as_of` before a ticker's first week/month closes, not a caller error, and separately the
      `padded[-n_have:]` slice was broken at `n_have=0`; (2) two O(n_unique × n) performance bugs
      (redundant per-position resampling in `_gather_branch_batch`, and `sample_cpc_negatives`'
      boolean-scan dict construction) — fixed via `CPCPanelStore`'s precompute and a proper
      O(n log n) `_group_positions` groupby. Net: batch-assembly time/step **6634ms → 11ms**
      (~600x), full step time 6678ms → 53ms (~125x), grad step itself unchanged (~35-45ms). Loss
      trajectory bit-identical across every fix (confirms each was a pure speed fix, not a
      behavior change). One-time `CPCPanelStore` precompute: ~32s for this 5-ticker/27k-position
      pilot slice — scales with universe size, revisit (chunking) before the full ~515-ticker run
      if that becomes materially slower than acceptable.
- [x] **Stage 1A-1D training loop (the real, full-scale run)** — Stage 1A done (below); 1B-1D
      remain (their losses -- masked reconstruction, forward cross-modal alignment, valuation
      probe -- aren't written yet).
- [ ] **Pilot/dry-run** (same convention as `rl_agent/experiment.py --dry-run`): rehearse
      the full Phase 1 loop — all 4 stages, all diagnostics, the trajectory plot — on a
      small slice (a handful of tickers, a few years) before the full 50-name/~15-year
      run, to catch plumbing bugs and get a real wall-clock estimate. (The CPC-only
      `pretrain_pilot.py` dry-run above covers Stage 1A; this item is about rehearsing all 4
      stages together once 1B-1D exist.)
- [x] **Stage 1A — CPC only.** Train; run diagnostics 1-7 + the trivial form of 8 (vs.
      raw features only). Record as the baseline the other 3 stages must beat.
      **Real run (2026-07-21), checkpoint `stage1a-20260721-165853.pt`, 150-ticker universe,
      27006 (ticker, month-end) points, `run_diagnostics.py`:** 4/7 gates passed.
      PASS: [1] neighbor-outcome variance ratio 0.7983 (gate ≤0.8), [2] regime MI 0.0006 vs.
      null_p95 0.0002, [5] perturbation sensitivity 0.0101 (gate ≤1.0), [6] temporal smoothness
      corr 0.1406, p<0.0001. FAIL: [3] valuation R² -0.0531 vs. volatility R² -0.1582 (gate:
      val_r2 > vol_r2 and ≥0.05 -- expected at 1A, nothing in CPC pushes toward valuation
      structure until 1D's valuation-probe loss), [4] quality persistence autocorr 0.0833 (gate
      ≥0.3, same reasoning), [7] latent similarity gap 1.8922, p=0.18 (gate p<0.05 -- correct
      sign, underpowered at n=426 matched pairs, not clearly wrong). Baseline 1B must beat:
      `docs/conviction_model/PHASE1_DIAGNOSTICS_20260721-211343.json`. (Rerun after the
      2026-07-21 `drop_zero_adjclose`→`trailing_volatility` NaN-masking fix; numbers essentially
      unchanged from the pre-fix run, as expected — only 0.03% of rows were affected.)
- [ ] **Stage 1B — + forward cross-modal alignment.** Warm-start from 1A; retrain; rerun
      diagnostics; compare against 1A.
- [ ] **Stage 1C — + masked reconstruction.** Same pattern; compare against 1B.
- [ ] **Stage 1D — + auxiliary valuation probe** (NaN-masked on rows without a defined
      `pl_zhist_5y`/`pvp_zhist_5y`). Same pattern; compare against 1C.
- [ ] For each of 1B/1C/1D: explicit keep-or-drop decision against the quantitative gate
      table below, applied relative to the previous stage — record which losses made the
      final config and which were dropped.
- [ ] Embedding-trajectory visualization: one long-history, event-rich ticker (e.g.
      PETR3/PETR4), embedded monthly across its full history, PCA-projected to 2D
      (`sklearn.decomposition.PCA`), plotted via the `plotly` convention from
      `rl_agent/plots.py`, with known events annotated (2020 COVID crash, 2016
      impeachment, elections, commodity cycles) — required deliverable, not optional.
- [ ] Final Phase 1 report assembled: diagnostic results per stage, gate pass/fail per
      diagnostic, trajectory plot, final loss configuration.
- [x] `tests/conviction_model/test_encoder.py` and `tests/conviction_model/test_diagnostics.py`
      pass (see Testing strategy) — both green (51/51 fast suite), but this covers the
      diagnostic *functions'* correctness on synthetic cases, not a real encoder's actual
      diagnostic scores, which still needs Stage 1A's real training run above.

**Quantitative gates for the 8 diagnostics** (concrete numbers to force a decision, not
asserted as principled — some are statistically grounded, some are first-guess round
numbers flagged as adjustable if Phase 1 data suggests otherwise):

| # | Diagnostic | Gate | Basis |
|---|---|---|---|
| 1 | Neighbor-outcome similarity | neighbor-outcome variance ≥20% lower than random-pair variance | arbitrary, adjustable |
| 2 | Regime clustering | mutual information with regime indicators exceeds the permutation-null 95th percentile | statistically grounded (reuses H-series' permutation-null convention) |
| 3 | Valuation probe vs. volatility probe | valuation-probe R² > volatility-probe R², **and** valuation-probe R² ≥0.05 | ordering is well-motivated (directly tests the failure mode in Risks); the 0.05 floor is arbitrary, adjustable |
| 4 | Quality persistence | 12-month-lag autocorrelation of the quality-probe score ≥0.3 | arbitrary, adjustable |
| 5 | Perturbation stability | encoder's normalized input-to-embedding sensitivity ≤ raw-feature-space sensitivity | relative gate, no free parameter |
| 6 | Temporal smoothness vs. surprise | correlation between embedding-delta magnitude and the surprise proxy is positive, p<0.05 via permutation | statistically grounded |
| 7 | Latent similarity | matched-state pairs' embedding distance significantly smaller than random pairs, p<0.05 via permutation | statistically grounded |
| 8 | Embedding value (full form runs in Phase 2) | SSL representation's walk-forward OOS error beats PCA/raw/autoencoder by ≥10% | arbitrary, adjustable |

### Phase 2 — Representation selection, then model-class competition

Staged, not a full Cartesian product; starts only after Phase 1D is reported.

- [ ] `baselines_repr.py`: raw feature vector representation.
- [ ] `baselines_repr.py`: PCA representation (`sklearn.decomposition.PCA`).
- [ ] `baselines_repr.py`: plain autoencoder representation (reconstruction loss only,
      same branch structure as `encoder.py` minus the contrastive/alignment losses).
- [ ] **2a — Representation selection.** All 4 representations (raw/PCA/autoencoder/SSL)
      evaluated with one fixed referee model (`RandomForestRegressor`), picked by
      walk-forward OOS error. 4 runs, not 20.
- [ ] Record the 2a winner and whether the SSL encoder cleared diagnostic 8's gate
      (beats raw/PCA/autoencoder by ≥10% OOS error) — if not, proceed with the simplest
      representation that matches performance and say so explicitly.
- [ ] **2b — Model-class competition, winner-representation only.** `RandomForestRegressor`
      / `HistGradientBoostingRegressor` / `CatBoostRegressor` / `LightGBM` / small MLP,
      crossed only with the 2a winner, picked by walk-forward OOS error.
- [ ] Confirm the multi-output handling is correctly noted per candidate (native for
      RF/CatBoost; `MultiOutputRegressor` wrapper for HistGB/LightGBM) before comparing
      training cost across candidates.
- [ ] Wire the configurable per-horizon loss weight (v8: primary=1.0, exploratory=0.5,
      drawdown=1.0, from `config.py`) into every candidate that supports it (MLP,
      `CatBoostRegressor` native multi-output); document the `MultiOutputRegressor`-wrapped
      candidates' weaker form (independent per-output fits, no joint loss to weight).
- [ ] If the SSL representation won: record per-branch feature importance from the
      winning tree-based model as the branch-ablation result.
- [ ] Residual-clustering check for the "outcome regression, not decision-quality"
      limitation (Labels, above), run on the final winning configuration.
- [ ] `requirements.txt` updated with pinned CatBoost/LightGBM versions.
- [ ] `tests/conviction_model/test_tree.py` and `tests/conviction_model/test_baselines_repr.py`
      pass (see Testing strategy).

### Phase 3 — Minimal walk-forward signal check (cheap; feeds Phase 4, not the
production system)

Deliberately NOT the full quarterly-refit machinery — that's real engineering
investment (Phase 5) that shouldn't be built before knowing there's a signal to
operate on. This phase answers one question as cheaply as possible: fit once (or on a
handful of expanding folds, reusing Phase 0's fold generator), score out-of-sample,
enough to compute Phase 4's metrics.

- [ ] Fit the Phase 2-winning configuration once on the development slice.
- [ ] Score OOS across the remaining expanding folds (no warm-start/cold-restart/
      drift-logging machinery — that's Phase 5, only built if this gate passes).
- [ ] Produce the per-fold, per-output prediction/realized-outcome pairs Phase 4 needs.

### Phase 4 — Signal validation (the actual success gate, decoupled from allocation
and from the production system)

Same ordering H-series already uses (H0/H1 rank-IC screening precedes H3's portfolio
construction) — this is the first thing to check, before any portfolio-construction
*or* production-infrastructure work.

**Primary/exploratory split (v9, top-150 power floors):** every check below runs on all 6
outputs, but only k=21/k=63/k=126/k=252 (**primary** under the top-150 universe) feed the
pass/fail decision. k=504 and drawdown-severity (**exploratory** — floor ≈0.046, still above
H1's 0.035 reference) are computed and reported identically, tagged `underpowered exploratory`
— a negative result there is not evidence of no signal, a positive result is interesting but
can't be trusted at face value, and neither can independently fail the project. (At top-50 the
primary set was only k=21/k=63; 0.035 is a reference, the real bar is significance — v9.)

- [ ] `signal_validation.py`: walk-forward OOS rank-IC computed per raw output (each of
      the 5 horizon predictions and the drawdown-severity prediction against its own
      realized value) — not against a reduced conviction number.
- [ ] Secondary rank-IC check: the reduced single-number conviction vs. a blended
      realized outcome, reported alongside (not in place of) the per-output results. The
      reduced conviction here is the **primary-pair (k=21/k=63) reduction**, so the secondary
      check isn't contaminated by the underpowered exploratory horizons; a mean-of-5 version
      may be reported too but carries no gate weight.
- [ ] **Long-short decile spread:** top-decile-conviction minus bottom-decile-conviction
      realized forward return, per horizon — the standard factor test, needs no
      portfolio optimizer, and is a second, differently-shaped check on the same
      question rank-IC asks.
- [ ] **Same spread, net of a round-trip transaction cost.** Reuse the rate already
      established in this project (`c_buy`/`c_sell` = 0.0003, i.e. 0.03% per leg, in
      every `configs/eiie_*.json` and applied in `src/rl_agent/environment.py`) rather
      than inventing a new number. A gross spread that vanishes once a single round-trip
      (buy + sell, ~0.06%) is subtracted isn't a signal worth building Phase 5/6 on top
      of — this is reported alongside the gross spread, not instead of it, since the raw
      spread is still the more direct test of "is there a signal at all."
- [ ] **Sub-period stability:** rerun rank-IC and the decile spread on 3 non-overlapping
      sub-windows (e.g. roughly 2011-2015 / 2016-2020 / 2021-2025) in addition to the
      pooled result — a signal that only shows up pooled but not within sub-windows is
      the same single-path-artifact failure mode that produced H2's false result, and
      this project already has the tooling (`H5_OBJECTIVE_CALIBRATION_ROBUSTNESS_PLAN.md`'s
      stability testing) to reuse rather than invent fresh. Confirm the earliest sub-window
      (≈2011-2015) post-dates the Phase 2 freeze/development slice, so its stability result is
      genuinely OOS and not partly in-sample.
- [ ] Permutation-null / NW-HAC significance applied to every rank-IC and decile spread
      (not a bare correlation coefficient or return number) — same bar H-series uses
      elsewhere.
- [ ] **Gate decision, k=21/k=63/k=126/k=252 (primary under top-150):** does higher
      conviction correlate with better realized outcomes at these horizons, consistently
      across sub-periods, at a magnitude and significance (t>2 / permutation-null, not a fixed
      0.035) that clears the bar — this is what passes or fails the project.
- [ ] **Exploratory report, k=504 + drawdown-severity:** same metrics computed
      and published alongside the gate decision, explicitly labeled `underpowered
      exploratory`, with no pass/fail semantics attached.

### Phase 5 — Production walk-forward harness (only if Phase 4's gates pass)

Not built until Phase 4 passes — this is the expensive, continuously-operating version
of Phase 3's cheap check, worth building only once there's a demonstrated signal to
operate on.

- [ ] Wire Phase 0-2 into `experiment.py`.
- [ ] Quarterly regressor refit loop.
- [ ] Quarterly encoder warm-start fine-tune loop (same cadence as the regressor).
- [ ] Cold-restart trigger: full encoder retrain every ~3 years (configurable), not a
      warm-start, at that cadence.
- [ ] Embedding-drift logging across the retrain cadence (to check whether ~3 years is
      actually the right cold-restart interval — see Risks).
- [ ] Roll forward through history, excluding the final holdout window.
- [ ] Confirm Phase 5's results are consistent with Phase 3's cheap check (a large,
      unexplained gap between the two means the continuous-retraining machinery itself
      is doing something Phase 3 didn't capture — worth understanding before trusting
      either).
- [ ] `tests/conviction_model/test_walkforward.py`'s Phase 5 assertions (warm-start/
      cold-restart cadence) pass — deferred from Phase 0, see Testing strategy.

### Phase 6 — Allocation layer (only if Phase 4's gates pass)

- [ ] Not designed in this plan — explicitly out of scope until Phase 4 passes.
- [ ] When designed: **benchmark-comparable floor (v8/v9, "at least go with the market").**
      "Go with the market" is a *capability floor*, not a constraint: the model must be able to
      earn benchmark-comparable returns and never be structurally stuck *below* a sensible
      benchmark — where the relevant benchmarks are **both the equity index and CDI/SELIC cash.**
      Cash-at-SELIC is a first-class holding the agent may choose whenever it sees fit (the
      labels are already CDI-relative, so cash is the natural neutral). So the weak-conviction /
      high-uncertainty default is a benchmark-comparable mix — index-like exposure *or*
      cash-at-SELIC, chosen by the model — not forced equity and not a drift to an all-cash or
      all-in corner. This also counters EIIE's documented all-cash/all-in bistable failure
      mode (CLAUDE.md): the point is a deliberate cash choice, not a collapsed-gradient trap.
- [ ] When designed: must be uncertainty-aware (Objective) — a high-conviction,
      high-uncertainty prediction should not automatically receive a large allocation.
- [ ] When designed: must be transaction-cost-aware — reuse the same 0.03%-per-leg rate
      (`c_buy`/`c_sell`) and cost-application logic already implemented in
      `src/rl_agent/environment.py` rather than a separate cost model, and include a
      no-trade band so a conviction change too small to clear round-trip costs doesn't
      trigger churn. `rl_agent/metrics.py` already computes turnover and cost drag —
      reuse those for reporting rather than writing new ones.
- [ ] When designed: must address the aleatoric-vs-epistemic uncertainty distinction
      (Risks) rather than treating all uncertainty as one signal.

### Phase 7 — Final holdout

- [ ] Confirm Phases 0-6 are fully frozen (no further architecture/hyperparameter/
      representation changes) before this phase starts.
- [ ] Run the most recent ~1-2 years exactly once — same discipline as
      `rl_agent/experiment.py --eval-split test`.
- [ ] Report final result regardless of outcome (pass or fail is itself the deliverable).

## Validation / leakage guarantees (tested, not assumed)

- **Purge/embargo test:** no training row's label window (up to 504 trading days
  forward) extends past that fold's `train_end`.
- **Point-in-time universe test:** regressor training only uses dates where
  `active_universe_by_date()` says the ticker was actually in the top 50.
- **No test-period leakage into SSL pretraining or warm-start fine-tuning.**
- **Hyperparameter/model-class/representation freeze test:** Phase 2's competition is
  decided once on the development slice, frozen before Phase 3 starts.
- **The full 8-diagnostic battery** (neighbor-outcome similarity, regime clustering,
  valuation/quality linear probes, perturbation robustness, temporal smoothness,
  latent-similarity, embedding-value — see "What a 'good' embedding means").
- **Residual-clustering check** for the outcome-vs-decision-quality limitation.

## Testing strategy

Mirrors the repo's plain-Python, assert-based, no-pytest convention (`tests/run_all.py`,
`fast` group — all synthetic, no dependency on the real dataset). Each file below is
written against the module it tests as soon as that module exists (Phase 0-2, per the
checkboxes above) — the assertions are fixed now so implementation has a concrete target.

**Status (mirrors the Phase 0/1 status blocks above):** `test_labels`, `test_data`,
`test_encoder`, `test_diagnostics`, `test_ssl_pretrain`, and `test_walkforward`'s Phase 0
assertions are implemented and green; `test_tree`/`test_baselines_repr` (Phase 2) and
`test_walkforward`'s Phase 5 assertions are not yet built. Boxes below reflect this.

**`tests/conviction_model/test_labels.py`**
- [x] The `+0.1%/+0.2%/+0.3%/+0.4%/+40%` worked example produces the expected 6-vector
      (hand-computed) — the 24-month value dominates, drawdown-severity is low.
- [x] The `+15%/+14%/+13%/-8%/-10%` worked example produces a visibly different 6-vector
      from the above — not collapsed to a similar scalar, and drawdown-severity is
      materially higher (the round-trip is penalized).
- [x] Output has exactly 6 independent columns — no silent aggregation/mean/decay step.
- [x] A synthetic row whose 504-day label window crosses a given `train_end` is dropped
      by the purge/embargo filter; a row that doesn't cross it is kept. (Asserted in
      `test_walkforward.py`, where the shared purge filter lives.)
- [x] Swapping the `bench` argument (CDI series vs. a synthetic BOVA11-like series)
      produces different `fwd_rel_return` values — confirms the CDI bench is actually
      wired in, not accidentally defaulting to `build_forward_targets`'s BOVA11 usage
      elsewhere in `h_series`.
- [ ] Rows where `pl_zhist_5y`/`pvp_zhist_5y` are NaN are excluded from any
      valuation-dependent computation, not filled with 0 or a placeholder. (Belongs to the
      encoder/auxiliary-probe side — `labels.py` targets are returns-based and never touch
      `pl_zhist`; moved in spirit to `test_diagnostics`/`test_encoder`, tracked here until then.)

**`tests/conviction_model/test_walkforward.py`** — split by what's actually buildable at
each phase; the whole file isn't a single Phase 0 gate (an earlier draft implied it was,
but cold-restart/warm-start cadence doesn't exist as a concept until Phase 5):
- [x] **Phase 0 gate:** expanding-fold boundaries generated correctly from
      `iter_expanding_folds()`.
- [x] **Phase 0 gate:** the purge/embargo filter used here matches `labels.py`'s own
      filter output on the same synthetic panel (no drift between two independent
      implementations of the same rule).
- [ ] **Phase 5 gate (not before):** regressor-refit and encoder warm-start fold
      boundaries land on the same quarterly cadence (`step_months=3`).
- [ ] **Phase 5 gate (not before):** the cold-restart flag fires only at the configured
      ~3-year mark, not every fold.

**`tests/conviction_model/test_encoder.py`**
- [x] Forward pass on synthetic multi-resolution input returns 4 separate sub-embeddings
      with the expected shapes — not one pooled vector.
- [~] Gradients are finite for each of the 4 losses individually (1A-1D), checked in
      isolation, not just the combined loss. (CPC loss done; the other 3 losses aren't built
      until Stages 1B-1D, so this can't be fully checked yet.)
- [x] Deterministic output on CPU given a fixed seed (two forward passes match exactly).
- [x] Cross-attention update actually changes each branch token relative to its
      pre-attention value (not a no-op / identity fallback).

**`tests/conviction_model/test_diagnostics.py`**
- [x] Neighbor-outcome test: on synthetic embeddings with injected low-variance
      neighborhoods, recovers a materially lower neighbor-outcome variance than random.
- [x] Regime-clustering test: on synthetic embeddings with an injected cluster structure
      matching a synthetic regime label, mutual information clears the permutation-null
      95th percentile; on pure-noise embeddings, it does not.
- [x] Valuation linear probe: recovers a known injected linear relationship (R² close to
      the injected signal-to-noise ratio, not near 0).
- [x] Perturbation-stability test: flags an artificially discontinuous synthetic encoder
      (large output jump from a tiny input change) as failing the gate; a smooth
      synthetic encoder passes.
- [x] Temporal-smoothness test: distinguishes injected "surprise" timesteps (large
      embedding delta) from noise timesteps (small delta) at the stated significance level.
- [ ] Each of the 8 gate thresholds (Phase 1's table) evaluates to the correct pass/fail
      on constructed synthetic cases that are deliberately just above and just below
      each threshold.

**`tests/conviction_model/test_tree.py`**
- [ ] Multi-output regressor recovers a known synthetic generating function (low error
      against the true function, across all 6 outputs).
- [ ] Ensemble-spread uncertainty is higher on synthetic out-of-distribution inputs than
      on in-distribution inputs.
- [ ] The `MultiOutputRegressor` wrapper path is exercised for the non-natively-multi-
      output candidates (HistGB, LightGBM) and produces 6 independently fitted models.
- [ ] Per-horizon loss weight (v7): on a synthetic target where the exploratory outputs
      are pure noise and the primary outputs carry real signal, the weighted candidates
      (MLP, CatBoost native) fit the primary outputs measurably better than an unweighted
      (all-1.0) baseline — confirms the weight vector actually changes what's optimized,
      not just a config field that's read and ignored.

**`tests/conviction_model/test_baselines_repr.py`**
- [ ] PCA representation: output shape and explained-variance-ratio sanity check on
      synthetic data.
- [ ] Plain autoencoder: reconstruction loss decreases over synthetic training steps.
- [ ] All 4 representations (raw/PCA/autoencoder/SSL) produce comparably-dimensioned
      outputs, so Phase 2a's referee-model comparison isn't confounded by dimensionality
      alone.

All synthetic/fast, added to the `fast` group in `tests/run_all.py`.

## Risks & open questions

- **The central open question: why should self-supervised learning discover
  representations useful for investing at all?** Not obvious, and this project's
  biggest source of uncertainty — larger than any choice between RF/CatBoost/LightGBM
  downstream. Masked reconstruction could become good at reconstructing price noise
  without learning anything valuation-related; CPC's contrastive objective could easily
  cluster by realized volatility (a strong, easy-to-predict regularity in financial
  time series) rather than by "good investment opportunity" — those are genuinely
  different things a representation could organize around, and nothing in a vanilla
  CPC/reconstruction objective favors one over the other. This plan doesn't claim to
  resolve that question by design — it can't be resolved by design, only by measurement
  — but it does two concrete things about it rather than hoping: (1) the forward
  cross-modal alignment loss and the auxiliary valuation-probe nudge (Architecture,
  above) are specifically chosen to bias the objective toward fundamentally-predictive
  structure over pure price statistics, and (2) Phase 1's diagnostic 3 (valuation linear
  probe vs. volatility linear probe) is a direct, cheap, early test for exactly this
  failure mode, run before any further investment in Phase 2. If diagnostic 3 shows
  volatility dominating and valuation near-zero, that's the honest result — evidence the
  objective needs rethinking, not a data or engineering bug to patch around.
- **Outcome-regression limitation (Labels, above) is inherent, not fixed.** Mitigated
  only by pooling across many examples; the residual-clustering check in Phase 2 is the
  only safeguard that it's noise, not bias.
- **New dependencies (CatBoost, LightGBM).** A deliberate exception to this project's
  usual minimal-dependency stance, scoped narrowly to Phase 2's competition — pin
  versions in `requirements.txt` same as everything else.
- **Two different notions of "uncertainty" — aleatoric vs. epistemic, documented here,
  not resolved.** Ensemble spread (Phase 2) mostly reflects **aleatoric** uncertainty —
  genuine unpredictability in the outcome even given a well-understood state (a known
  chaotic environment). The latent-similarity/neighbor-density diagnostics (Phase 1)
  are closer to **epistemic** uncertainty — the model hasn't seen a state like this
  before, distinct from "this state is inherently unpredictable." These plausibly
  should drive allocation differently (small position under known-chaotic conditions,
  vs. possibly no position under genuinely novel ones), but that logic isn't designed
  here — deliberately deferred to Phase 6, named now so it isn't rediscovered as a
  missing distinction later.
- **Warm-start drift.** The ~3-year cold-restart interval is a starting guess — Phase 5
  should log embedding drift over time to check whether that cadence is right.
- **MLP uncertainty is undesigned** — would need MC-dropout or a deep ensemble if the
  MLP wins Phase 2's competition and quantile regression isn't a good fit for it either;
  not designed here.
- **Cross-attention fusion cost** — bounded (4 tokens), but Phase 1 should confirm it
  beats concatenation before treating it as load-bearing.
- **Top-50 membership turnover** — a ticker with little history when it first enters the
  top 50 has a poorly-trained embedding; no special handling proposed, flagged as a
  known ceiling.
- **Possible unit bug in `excess_return`/`real_return` (not this plan's code, but an input
  it was about to depend on) — discovered while implementing Phase 0, not yet fixed
  anywhere.** `compute_macro_features()` computes `excess_return = log_return - selic/252`
  and `real_return = log_return - ipca/252`, with an inline comment claiming selic/ipca
  are annual percentages. Checked directly against `data/raw/macro/selic.parquet`: its
  raw values are numerically identical in magnitude to `cdi.parquet`'s (~0.05 on the same
  date), and `src/rl_agent/data.py::validate_cdi_daily_percent` already establishes,
  with an explicit range-checked assertion, that CDI's raw value is a **daily** percent
  rate (`/100` for the daily decimal), not annual. If SELIC follows the same convention
  (very likely, given the numerical match), `/252` understates the daily rate by roughly
  2.5x relative to the correct `/100`. Corroborated in-repo: the same module already treats
  SELIC inconsistently — `compute_macro_features()` uses `selic/252` (`features.py:318`, annual
  assumption) while `earnings_yield_vs_selic` (`features.py:546`) uses `selic/100` (daily-percent
  assumption); two conventions for one column in one file is itself evidence the `/252` path is
  wrong, independent of the CDI cross-check. This plan's own `labels.py`
  (`load_cdi_daily_decimal`) is unaffected — it reads `cdi.parquet` directly with the
  correct `/100` convention, not through `compute_macro_features()`. But the Monthly
  branch's `excess_return`/`real_return` input features (Input features table, above)
  *are* this function's output, so if the bug is real, those two columns are scaled
  wrong throughout `ml_dataset.parquet` (not just for this plan — every consumer of
  those two columns). Out of scope to fix here: `compute_macro_features()` is shared,
  foundational, and several things beyond this plan depend on it — fixing it would mean
  a separate, explicit task with its own sign-off, not a side effect of Phase 0. Recorded
  so Phase 1 doesn't build on these two columns without first resolving this.
- **Coupling to `src/h_series`, which is itself still an active, unsettled research
  track** (per its own roadmap, H2 already failed once and H3/H3a aren't resolved yet).
  This plan reuses `spine.py`'s fold/universe/target functions rather than
  reimplementing them, which is the right call, but it does mean a future signature or
  behavior change in `spine.py` (made for H-series' own reasons) can silently break
  `conviction_model`. Cheap mitigation: this project's own test suite should pin down
  the exact behavior it depends on (fold boundaries, `build_forward_targets`'s output
  shape) rather than only relying on `h_series`'s own tests to catch a regression.
- **Phase 4's rank-IC gate could fail even with a "correct" architecture** if the
  outcome-regression limitation's noise is large relative to sample size at this
  universe size (50 names) — a plausible, honest failure mode, not a sign the plan was
  wrong, the same lesson H2's failure already taught this project once.
- **Measured, not speculative (v7): the 50-name/305-month sample is only large enough to
  make this bar at k=21/k=63.** Phase 0's actual power-floor run (above) confirms the
  above risk isn't hypothetical at k=126/252/504 — it's already true given today's
  dataset size. The mitigation adopted (primary/exploratory split, Phase 4, Labels) is a
  reporting-honesty fix, not a statistical-power fix; if the long-horizon signal is real,
  this plan currently has no way to confirm it at an acceptable false-positive rate. The
  only real fix would be a larger effective sample. Two levers:
  - **Wider label universe (the high-leverage one — ADOPTED in v9).** top-50 → top-150 lowers
    `sigma_ic_null` from 1/√49=0.143 to 1/√149=0.082, dropping the floors to k=126≈0.023,
    k=252≈0.033 (both now under H1's 0.035 lower bound — powered) and k=504≈0.046 (still
    short); `n_obs` is unchanged (universe width doesn't add monthly dates). Needs a top-150
    point-in-time membership file built and accepts less-liquid names; it's also a breadth gain
    in its own right (Fundamental Law of Active Management). This is the v8 path to a
    *confirmable* long-term signal, not just a *learned* one.
  - **Longer history** — limited; the dataset can't be extended backward past what's collected.
- **Deferred fix idea (#2, OOS-vs-full-sample power — apply when Phase 4 runs).** Phase 0's
  floors use the full `n_obs=305`, but Phase 4's rank-IC runs only on walk-forward OOS dates
  (the initial train window is consumed first), so the effective n is smaller and the true
  floors are higher — e.g. a ~5y initial train slice leaves ~245 OOS months, lifting k=63's
  floor from 0.0283 to ~0.032 and thinning its margin under 0.035. Before trusting the
  k=21/k=63 GO at face value, recompute `min_detectable_ic` with the actual OOS month count
  Phase 4 will have (not 305) and re-record the GO/marginal decision. Not done now — recorded
  so it isn't forgotten.
