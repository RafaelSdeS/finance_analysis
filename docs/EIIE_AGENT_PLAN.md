# EIIE Portfolio Agent — Iteration 1 Design & Implementation Plan

Faithful reproduction of **Jiang, Xu & Liang (2017)**, "A Deep Reinforcement Learning Framework
for the Financial Portfolio Management Problem" (`docs/papers/deep_reinforcement_learning_
framework_financial_portfolio_management.pdf.pdf`) — EIIE topology + Portfolio-Vector Memory
(PVM) + Online Stochastic Batch Learning (OSBL) + deterministic policy gradient on the explicit
log-return reward — adapted to daily B3 data, the top-50 most-traded Brazilian stocks, **price
data only** (no fundamentals/macro/sentiment features yet; those arrive in later iterations
without requiring a rearchitecture — see "Extensibility" below).

Status: design approved 2026-07-16. Implementation in progress (`src/rl_agent/`).

## Approved decisions (deviations from the paper, and why)

| # | Decision | Paper says | We do | Why |
|---|---|---|---|---|
| 1 | **Cash return** | Bitcoin, zero return (quote currency) | Accrues daily CDI | Idle BRL realistically earns ~CDI; makes "hold cash" and the constant-cash baseline meaningful |
| 2 | **Universe** | Fixed preselection at backtest start | Dynamic quarterly top-50, point-in-time | Reuses this repo's existing survivorship-safe universe construction instead of re-deriving one |
| 3 | **Online learning** | OSBL, always-on during backtest | Kept (paper-faithful) | Approved as recommended — matches the paper's actual reported protocol |
| 4 | **Transaction cost** | 0.25% (Poloniex max) | 0.03% (`c_s = c_p`) | Actual B3 rate |
| 5 | **Experiment window** | N/A (crypto, 2016-17) | 2011-01-31 → 2026-07-14 | The pre-built top-50 universe file's span — trimmed there for *fundamentals* completeness, which future iterations need even though iteration 1 doesn't use fundamentals |

## Facts verified against the actual data (do not re-derive these — re-check if the data changes)

- **N_union = 171** distinct tickers ever appear in the top-50 membership table
  (`data/processed/top50_universe_membership.parquet`) → global asset space = 171 + 1 cash = **172**.
  Read from the file at load time, never hardcoded.
- **Every quarterly period within 2011-2026 has exactly 50 members.** The only short periods
  (32/36/39/45 members) are 2001-2002, before this experiment's window — so slot-padding for a
  short period never triggers here, but the valid-mask machinery is kept anyway since it's also
  how departing tickers get liquidated at quarter boundaries.
- **CDI (`data/raw/macro/cdi.parquet`, BCB series 12) is a daily rate in percent**, not annualized
  and not a fraction. Verified: 2000-01-03 `cdi=0.0683` → `(1.000683)^252 ≈ 18.7%` p.a. (matches
  SELIC ~19% then); 2026-07-14 `cdi=0.0525` → `≈14%` p.a. Cash relative factor = `1 + cdi/100`.
- **Observation-price continuity vs universe file:** the pre-built
  `data/processed/ml_dataset_top50_universe.parquet` drops rows outside a ticker's membership
  periods, so an entrant's `n=50`-day lookback history is missing there. Prices are byte-identical
  across files, so **observation prices are read from the full `ml_dataset.parquet`** (3 columns:
  `adj_close/adj_high/adj_low`, pyarrow column projection) while **investability/scope stays
  gated by the membership file within the 2011-2026 window**. This keeps `X_t` causal without
  changing the chosen universe or fundamentals rationale.
- `adj_close` is total-return (dividends baked in, per root `CLAUDE.md`) — consistent with
  comparing against CDI-accruing cash.

## Data flow

```
data/processed/ml_dataset.parquet          data/processed/top50_universe_membership.parquet
  (adj_close/high/low, 3-col projection)      (period_id, ticker, start, end; 2011-2026 slice)
                    │                                          │
                    └──────────────┬───────────────────────────┘
                                   ▼
                    PricePanel (src/rl_agent/data.py)
        dense (T × 172) close/high/low matrices on the B3 trading calendar,
        forward-filled within each ticker's listed life, flat-filled (ratio=1)
        outside it; per-day slot_gidx[T,50] + valid[T,50] from membership
                                   │
        data/raw/macro/cdi.parquet ──► cash relative factor (1 + cdi/100)
        data/raw/prices/BOVA11.parquet ──► benchmark only, never a model input
                                   ▼
              window-scoped train/val/test split (recomputed for 2011-2026;
              see "Split protocol" — NOT the repo's full-dataset split_config.json)
                                   ▼
        X_t (3×50×50 slotted tensor) + w_{t-1} (PVM read)  ──►  EIIE  ──►  w_t
                                   │
                         PVM write (global space) + environment.py (μ, reward)
                                   ▼
              experiments/{run_id}/ (config, seed, dataset fingerprint,
              checkpoints, metrics, plots, validation report)
```

## Global asset indexing & PVM dynamic↔global mapping

The one genuine extension over the paper's fixed-ordering assumption, specified precisely because
it's load-bearing for correctness:

- **Permanent global index**: the 171 union tickers sorted alphabetically get fixed indices
  `1..171`; **cash = index 0**. `N_GLOBAL = 172`. Deterministic, stored in experiment artifacts,
  independent of any file's row order.
- **Network slots**: the EIIE always sees a fixed width of **m=50 risky-asset slots** (+ cash).
  Each day, the active (≤50) tickers fill slots **sorted by permanent global index** — deterministic
  and stable *within* a quarter (so `w_{t-1}` aligns to the same asset per slot day-to-day). Across
  quarters the composition changes, so no fixed slot maps to a fixed ticker long-term — the
  shared-weight CNN cannot learn slot position as a ticker-identity artifact.
- **Two coordinate systems, bridged by `gather`/`scatter` (vectorized, no Python loops)**:
  - *Slot space* `[B, 50]` — what the network consumes/produces.
  - *Global space* `[B, 172]` — where the PVM lives and where **all cost/reward math happens**.
  - `read`: `w_prev_slots = gather(PVM[t-1], slot_gidx) * valid_mask`; `w_prev_cash = PVM[t-1][:, 0]`.
  - `write`: `scatter(zeros(B,172), slot_gidx, w_slots)`, set column 0 = cash weight, store at `PVM[t]`.
- **Boundary liquidation happens in global space.** At a quarter boundary the drifted weights `w'_t`
  (eq. 7) may hold tickers absent from the new active-50. Because the transaction-cost solver
  operates on full 172-dim global vectors (drifted `w'_t` vs. target `w_t`), a departing ticker's
  target is 0 and its forced sale is priced correctly — even though it's not among the network's 50
  inputs that day. This is why cost math is global while the network is slotted.
- **Valid-mask** `[B, 50]` (bool): false for empty slots and any slot without a live price that day.
  Masked slots get softmax logit `-inf` → weight 0.

## Observation / state / action / reward

- **Price tensor** `X_t`, shape `(f=3, m=50, n=50)` (paper eq. 18): last `n` closes/highs/lows per
  active slot, each divided by the latest close (`v_t`); last column = 1. Masked slots filled with
  the neutral flat pattern (all ones). Dead/pre-listing spans flat-filled (paper §3.3, 0-decay).
  **Extensibility**: future feature sets (fundamentals, macro, sentiment) are additional `f`
  channels — a config change to the channel list, not an architecture change.
- **State** `s_t = (X_t, w_{t-1})` (eq. 20). **Action** `a_t = w_t`, softmax over
  `[cash_bias, 50 masked asset scores]` (eq. 19).
- **Reward** `R = (1/t_f) Σ ln(μ_t · y_t · w_{t-1})` (eq. 21-22). `y_t` is the global 172-vector:
  cash = `1 + cdi_t/100`, asset = `v_t/v_{t-1}`, dead/flat = 1.
- **Loss stability**: `loss = -mean(log(clamp(μ · (y·w), min=1e-12)))`;
  `torch.nn.utils.clip_grad_norm_(..., max_norm)` (config-driven, default 5.0).

## Transaction cost model

`μ_t` (eq. 14) solved by the Theorem-1 fixed-point iteration, initial guess
`μ₀ = c · Σ|w'_{t,i} − w_{t,i}|` (eq. 16).
- **Backtest**: iterate to tolerance `δ` (default 1e-10).
- **Training**: fixed `k` steps from `μ₀` (paper leaves `k` unspecified → **k=1**, config-driven),
  giving a differentiable `μ`.
- **Validation gate (hard requirement before any training run)**: unit tests assert the k-step
  differentiable `μ` matches the converged fixed-point solver within tolerance on random cases, and
  a finite-difference gradient check confirms autograd through `μ` matches numeric gradients.

## EIIE network (paper Fig. 2, CNN)

Per-asset identical weight-shared streams: Conv1 (1×3 kernel, 2 feature maps, ReLU) → Conv2
(1×(n−2) kernel → 20 maps of `m×1`, ReLU) → concat `w_{t-1}` as a 21st feature map → 1×1 conv → 50
scores → prepend a learnable cash bias → mask inactive slots → softmax → `w_t ∈ Δ^{51}`. Behind an
`Encoder` protocol (`forward(X, w_prev, mask) -> scores`) so RNN/LSTM variants and future
feature-branch encoders plug in without touching training/PVM/environment code.

## PVM, OSBL training, split protocol

- **PVM**: global `[T, 172]` buffer, each row initialized **all-cash** (`w_0 = (1, 0, ..., 0)`,
  eq. 5) — a valid simplex point the policy converges away from under training. (Micro-deviation
  from the paper's "uniform" init: uniform-over-172 would place mass on inactive assets.)
- **OSBL** (§5.3): mini-batches of `n_b=50` *consecutive* periods; batch start sampled with
  geometric decay `P(t_b) ∝ β(1−β)^{t−t_b−n_b}` (seeded generator). Pretrain on the train split;
  during backtest, after each period append the new data point and run `rolling_steps` (paper: 30)
  causal updates sampled from all past data.
- **Split, recomputed for this window, never hardcoded**: the repo's existing
  `data/processed/split_config.json` (`train_end=2018-07-30`) was computed over the *full*
  2000-2026 dataset and does not fit the 2011-2026 experiment window. A window-scoped split
  (default 70/15/15 temporal, same date-based method as `manifest.compute_split_dates`) is
  recomputed and written into each experiment's own directory; `iter_fit_windows()`-style
  resolution keeps the fit boundary a single seam, not a hardcoded constant. Final evaluation:
  pretrain on train+val, backtest on test (paper's "training immediately precedes backtest").

## Evaluation

- **Baselines** (all through the *same* environment, same costs, same dates — the agent is never
  evaluated alone): Buy & Hold (UBAH), Equal-Weight rebalanced (UCRP), Best-Stock (hindsight),
  Random Portfolio (seeded Dirichlet), Random Rebalancing, Constant-Cash (= pure CDI), BOVA11 index.
- **Metrics**: Total/Annualized Return, CAGR, Volatility, Sharpe (vs. CDI risk-free), Sortino,
  Calmar, Max Drawdown, historical VaR/CVaR (95%), Portfolio Turnover, transaction-cost drag, Win
  Rate, Information Ratio vs. BOVA11, final APV.
- **Statistical uncertainty**: block-bootstrap confidence intervals on returns/Sharpe over the
  periodic-return series, reported alongside point estimates so noisy results aren't over-read.
- **Report**: one self-contained plotly HTML per experiment — PV vs. all baselines (log scale),
  train/val reward curves, allocation-evolution area chart, turnover & cost over time, weight
  distribution, metrics+CI table.

## Leakage & causality guarantees (tested)

- Every `X_t` uses only data with `trade_date ≤ t` — window index math checked against the
  calendar; an off-by-one that peeks at `t+1` fails the test.
- Universe membership at period P uses only trailing data (guaranteed upstream by the build's
  `merge_asof(direction="backward")`); a test asserts no future membership leaks into earlier periods.
- Survivorship: a test asserts delisted (`status == "CANCELADA"`) tickers are present in the union
  set (mirrors the repo's existing survivorship guard), and that `status` is never read as a
  feature (per root `CLAUDE.md`'s documented lookahead trap).
- Timestamp-alignment assertions between prices, CDI, and membership on the shared calendar.

## Module layout

```
src/rl_agent/
  __init__.py     # exports grow as each module lands
  config.py       # frozen-dataclass ExperimentConfig <-> JSON
  data.py         # GlobalAssetIndex, PricePanel (prices + membership + CDI + BOVA11)
  pvm.py          # PortfolioVectorMemory: global [T,172] buffer, gather/scatter read/write
  environment.py  # numpy backtest engine (agent + all baselines): mu solver, weight evolution,
                  #   global-space cost/reward, forced-sale; + differentiable torch mu for training
  networks.py     # Encoder protocol + EIIECNN
  train.py        # OSBL: geometric sampler, pretrain loop, online rolling updates, checkpointing;
                  #   _PanelStore: all window tensors/price relatives precomputed once, GPU-resident
  baselines.py    # 7 baselines through environment.py with identical costs
  metrics.py      # full metric suite + block-bootstrap CIs
  plots.py        # plotly -> one self-contained HTML report per experiment
  sanity.py       # invariant-based pre-training checks
  experiment.py   # CLI orchestrator; --dry-run stops after sanity, no training
  sweep.py        # parallel launcher: seed ensembles / config sweeps as bounded subprocesses

configs/eiie_baseline.json
tests/rl_agent/            # FAST group (synthetic only, no data files)
```

Reused from this repo: `manifest.py`'s dataset-fingerprint/versioning conventions, `paths.py`
constants, `tests/run_all.py` runner, plotly (already a dependency). Torch 2.12 is installed.
**Not** using stable-baselines3/gymnasium — the paper's deterministic policy gradient is direct
differentiable maximization of the reward; a gym+SB3 wrapper would fight the PVM/OSBL design for
no benefit.

## Testing strategy

- **Unit (FAST group, synthetic data only)**: price-tensor construction/normalization (eq. 18);
  `y_t` incl. CDI cash factor + known-value check; μ solver vs. brute-force root + `μ∈(0,1]` +
  zero-trade⇒μ=1; differentiable-μ vs. converged solver + finite-difference gradient check; weight
  evolution (eq. 7); PVM gather/scatter round-trip + all-cash init + boundary read/write with a
  departing ticker; softmax simplex (sums to 1, ≥0, masked slots = 0); config round-trip; every
  metric vs. hand-computed values.
- **Integration**: tiny seeded synthetic market end-to-end (pretrain a few steps → backtest →
  metrics → plot file exists); checkpoint save/load/resume equivalence; two identical-seed runs
  produce identical PV series; experiment dir contains every required artifact.
- **Sanity (`sanity.py`, run automatically before training) — invariant-focused, not behavioral**:
  deterministic seeding; constant prices + zero cost ⇒ PV constant; costs strictly reduce PV;
  UBAH reproduces its closed-form value; equal-weight correct; weights always on the simplex;
  PV > 0 always; no NaN/Inf anywhere (obs, weights, loss, grads); finite gradients on first
  batches. A dominant-asset toy market is a **diagnostic only** (logs concentration/reward trend),
  never a pass/fail gate — real markets don't guarantee an agent should fully concentrate.
- **Experiment validation checklist (auto, end of run)**: sanity passed; no numerical instability;
  config+seed+dataset-version+model-version saved; metrics+baseline-comparison+plots generated.
  Any failure marks the run invalid in `report.json` and in console output.

## Reproducibility

A single config seed seeds Python/numpy/torch; deterministic geometric sampler. Same config ⇒
same results **on CPU**, verified bit-exact (the S1 feature-store A/B reproduced sanity losses to
the last decimal — `TRAINING_SPEEDUP_PLAN.md`). On GPU, same-seed runs are deterministic *within*
a process (what the sanity gate's determinism check compares) but drift *across* processes:
cuDNN selects conv algorithms per-process, and measured sanity losses differ in the ~3rd–4th
significant digit between launches on unchanged code. `torch.use_deterministic_algorithms(True)`
is deliberately NOT enabled (speed cost); switch it on only if exact cross-process GPU
reproduction ever matters. `train.compile` (config flag, default off) is likewise not
bit-identical to eager — measured 1.13× here, not worth its reproducibility cost. Every run logs:
config copy, seed, git commit, dataset version (`dataset_v1` + manifest fingerprint), package
versions, the permanent asset-index map, and model checkpoint hashes. Run-dir timestamps carry
microseconds so parallel sweep launches (`sweep.py`) never collide on a directory.

## Implementation phases

- [x] **Phase 1** — this plan doc + `config.py` + `configs/eiie_baseline.json` + package skeleton.
- [x] **Phase 2** — `data.py`: `GlobalAssetIndex`, `PricePanel` (prices + membership + CDI +
      BOVA11), slot calendar, flat-fill, CDI known-value assertion. Unit tests.
- [x] **Phase 3** — `pvm.py`: `PortfolioVectorMemory` (gather/scatter, all-cash init, boundary
      liquidation). Unit tests.
- [x] **Phase 4** — `environment.py` (μ solver, weight evolution, global cost/reward) +
      `metrics.py` (full suite + bootstrap CIs). Unit tests.
- [x] **Phase 5** — `baselines.py` (7 baselines through the shared environment). Tests — runnable
      end-to-end before any RL code exists.
- [x] **Phase 6** — `networks.py` (EIIE CNN), `train.py` (OSBL), `sanity.py`. Differentiable-μ
      validation gate + synthetic convergence diagnostic.
- [x] **Phase 7** — `plots.py`, `experiment.py` (CLI, `--dry-run`), validation checklist.
      Integration tests.
- [x] **Phase 8** — Update root `CLAUDE.md` (Stage 3) + `README.md`. Tests were registered into
      `tests/run_all.py`'s FAST/DATA groups incrementally as each phase landed, rather than batched
      here — kept every phase immediately CI-discoverable instead of leaving 6 phases' worth of
      tests unregistered in the meantime.

All 8 phases complete: 35 FAST-group tests + 1 DATA-group integration test, all passing, all
run on synthetic data except the one DATA-group test (which loads the real dataset to verify the
loader). **No real training run has been executed** — per this project's standing working rule,
that requires explicit user go-ahead. Ready for: `python -m src.rl_agent.experiment --config
configs/eiie_baseline.json --dry-run` (verifies wiring against the real dataset, no training) and,
on approval, a real experiment run.

Per the standing working rule: code is written and unit-tested per phase, but **no training run
is launched without explicit go-ahead**; test/dry-run commands are offered, never auto-executed.

## Verification (end-to-end, once all phases land)

```bash
python tests/run_all.py --group fast                                             # incl. tests/rl_agent/
python -m src.rl_agent.experiment --config configs/eiie_baseline.json --dry-run  # data+sanity, no training
python -m src.rl_agent.experiment --config configs/eiie_baseline.json            # full run (on approval)
# -> experiments/{run_id}/report.html: agent vs. 7 baselines, metric+CI table, allocation evolution
```

## Assumptions & ambiguities (documented where they bite in code)

- Training μ uses fixed **k=1** from `μ₀` (eq. 16); config-driven, not hardcoded.
- Daily periods (paper: 30-minute); trade executed at day-`t` close.
- Pretrain step count and `β` retuned on validation (train span here is ~2,700 daily periods vs.
  the paper's ~30k half-hour periods — a hyperparameter difference, not an architectural one).
- CDI = %/day (BCB series 12) — verified above; asserted in the loader at runtime.
- Zero slippage / zero market impact (paper Hypotheses 1-2) — reasonable at daily frequency for
  top-50 B3 liquidity.
- `status` is never used as a feature (survivorship trap documented in root `CLAUDE.md`); universe
  membership alone gates investability point-in-time.
- Observation prices are sourced from the full dataset for entrant lookback continuity; the
  experiment's universe/date scope remain exactly the pre-built 2011-2026 top-50 window.
