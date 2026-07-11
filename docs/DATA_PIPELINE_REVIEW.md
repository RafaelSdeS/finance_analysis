# Data Processing Pipeline Review — RL Portfolio Optimization

Scope: Stage 2 (`src/build_dataset/build_ml_dataset.py` → `data/processed/ml_dataset.parquet`) on the
`refactor` branch, which is what actually feeds the RL agent. The agent code itself (`src/agent/*`)
lives only on the unmerged `ml_agent` branch and is **not inspectable from here** — called out
explicitly wherever it matters, rather than guessed at.

Grounded in the current code as of 2026-07-11. Where a finding is already tracked in `TODO.md` or
`notes.md`, it's referenced, not repeated.

Every actionable item below is tagged `T#`, has a **Goal** (what "done" concretely means) and a
**Test** (how you'd know it's actually done, not just attempted). §7 is a priority-ordered index of
every `T#`.

---

## 0. Two findings to act on before anything else

- [x] **T1 — Fix the lookahead leak in `volatility_20d_percentile` / `volatility_60d_percentile`**
      (`build_ml_dataset.py:673-674`). DONE. `g["volatility_20d"].rank(pct=True)` ranked against the
      ticker's **entire series, future included** — no rolling/expanding window, unlike the neighboring
      percentile features (`price_percentile_5y`, `pl_percentile_5y`, `drawdown_percentile`) which already
      used `.rolling(window=..., min_periods=1).rank(...)`. Switched both columns to the same
      `window_252 = 252*5`-window rolling-rank pattern (the shared `window_252` definition was hoisted
      above both blocks so it isn't duplicated).
      **Test:** `test_volatility_percentile_no_lookahead` in `test_build_dataset_features.py` — computes
      `compute_advanced_features` on a 6-row synthetic ticker, then again on the same data truncated to the
      first 3 rows, and asserts rows 0-2 are identical between the two runs (a plain global `.rank()` would
      fail this, since truncating removes future rows that a global rank could see).

- [x] **T31 — Fix the fundamental-data publication-lag leak (bigger blast radius than T1).** DONE.
      Verified directly: `reference_date` in every `data/raw/fundamentals/*.parquet` file is the **fiscal
      quarter-end** (e.g. `2026-03-31`), not a real filing/disclosure date — confirmed by reading
      `PETR4.parquet`. B3-listed companies have a statutory window after quarter-end to actually file (CVM
      ITR/DFP deadlines run weeks to ~90 days past period-end for annuals) — real investors could not see
      that quarter's fundamentals on the reference date itself. `merge_asof(direction='backward')` on
      `reference_date` was making `pl`, `pvp`, `roe`, every growth/Piotroski/valuation-composite column
      "available" starting exactly on the quarter-end date, ~40+ columns affected every quarter.
      The "recover the real filing date" lead through BolsAI was chased and ruled out: `collectors.py`'s
      `_merge_save` saves whatever BolsAI's `/fundamentals/{ticker}/history` response returns verbatim (no
      column trimming), and the raw response has no filing/disclosure-date field at all — confirmed against
      `PETR4.parquet`'s full 39-column schema. First shipped as the statutory-buffer fallback: `merge_asof`
      joined on a synthesized `fundamentals_available_date = reference_date + lag` (45d quarterly / 90d
      Q4-annual, `FILING_LAG_DAYS_QUARTERLY`/`FILING_LAG_DAYS_ANNUAL`).
      **Upgraded same day to real dates.** CVM's own open-data portal
      (`dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{ITR,DFP}/DADOS/`) publishes a `DT_RECEB` field — the date
      CVM actually received each filing — keyed by CNPJ, free and keyless. New
      `src/data_collection/filing_dates.py` downloads the ITR+DFP registers 2010–2026, extracts
      `(cnpj, reference_date) → received_date`, saves incrementally per year (robust to
      network interruptions — a run that dies partway through keeps everything collected so far and
      resumes on re-run). `attach_filing_dates()` joins these onto fundamentals by `cnpj` + `reference_date`
      before the price merge, falling back to the statutory buffer only for quarters missing from the CVM
      register. `reference_date` itself is untouched and still drives all YoY/QoQ/trend calculations.
      Side effect caught and fixed in the same change: `recompute_valuation_daily`'s split-guard
      (`near_filing`) assumed a fundamental became visible right at `reference_date`; fixed to key off
      `fundamentals_available_date` instead (now retained in the output — not dropped — since "when did
      these numbers become public" is legitimate agent-visible state).
      **Tests:** `test_merge_applies_filing_lag` (statutory-buffer path) and
      `test_merge_honors_actual_filing_date` (real-date path, both directions: an early filer visible
      before the statutory deadline, a late filer after) in `test_build_dataset_features.py`.
      **Verified on real data, twice.** Statutory-only build: 668,014 rows, `min(trade_date -
reference_date) == 45` days, zero violations. Real-date build (2026-07-11): collector returned
      **41,530 filings, 1,223 companies, 100% coverage of all 293 tickers in the universe** (zero nulls,
      zero negative lags after dropping 2 corrupt register rows); rebuild reports **14,362/14,362 quarters
      (100%) using real CVM dates, zero statutory fallback needed**. Quantified the actual improvement over
      the statutory estimate: comparing real `fundamentals_available_date` to what the fixed buffer would
      have assumed, **4,657 rows (0.7%) had real filings >30 days later than the statutory estimate** — the
      residual leak a fixed buffer structurally cannot catch, now closed; 84.8% of rows also got _tighter_
      (fundamentals correctly visible earlier for companies that file ahead of the deadline). Full
      `tests/run_all.py --group all` (4/4 files) and all 18 `test_final_dataset.py` gates green against the
      real-date rebuild; `ruff check` clean.

- [x] **T2 — Add a feature-internal causality test (generalizes T1's fix into permanent coverage)**
      DONE — `test_volatility_percentile_no_lookahead` is exactly this generic truncation-diff pattern
      (full-history run vs. truncated run, diffed row-by-row), not scoped only to the two columns T1 fixed.
      `test_final_dataset.py` itself wasn't touched — kept in the fast/synthetic group per the original
      recommendation, cheaper and faster than running it against the full parquet. Only
      `volatility_20d_percentile`/`volatility_60d_percentile` are covered so far; extending the same
      pattern to `price_percentile_5y`, `momentum_vs_market_1m`, `pl_zscore_sector` etc. is still open if
      broader coverage is wanted.

---

## 1. Validating the processed dataset

### What's already covered (don't re-add)

| Check                                                                    | Where                                         | Notes                                                                                            |
| ------------------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Fundamental lookahead (`reference_date <= trade_date`)                   | `test_final_dataset.py:validate()`            | merge-level only                                                                                 |
| Duplicate `(ticker, trade_date)`                                         | same                                          |                                                                                                  |
| CAGR `_final` columns present                                            | same                                          |                                                                                                  |
| NaN in `close`/`volume`                                                  | same                                          | only these two columns are NaN-gated                                                             |
| inf/-inf anywhere                                                        | same                                          |                                                                                                  |
| Macro columns non-null-anywhere                                          | same                                          |                                                                                                  |
| Row count per ticker                                                     | same                                          | informational only, not gating                                                                   |
| P/L varies daily within quarter                                          | same                                          | regression guard for `recompute_valuation_daily`                                                 |
| `close_price` dropped, `has_fundamentals` present                        | same                                          |                                                                                                  |
| Feature formulas (log_return, RSI, MA, drawdown, ratios, rescaling math) | `test_build_dataset_features.py`              | synthetic data, fast group                                                                       |
| Stale-price runs (≥5 identical closes, volume>0)                         | `test_final_dataset.py:check_stale_prices`    | **print-only, does not fail the run**                                                            |
| Per-ticker outliers (MAD z-score, threshold 8)                           | `test_final_dataset.py:check_outliers_zscore` | **print-only, does not fail the run**                                                            |
| Raw-collection schema/range gates                                        | `src/data_collection/validate.py`             | future dates, negative prices/volume, gaps, null rates — runs at collection time, not build time |

### Gaps — checklist mapped to your 12 topics

- [!] **T3 — Survivorship bias: CONFIRMED STRUCTURAL, fix is upstream.** Investigated 2026-07-11:
  `company_info.parquet` has **293/293 companies with status `ATIVO` — zero delisted companies
  anywhere in the universe**. The dataset is survivorship-biased by construction; no build-time test
  can fix this, and any backtest result (agent or benchmark) is inflated by the absence of companies
  that went to zero. The fix is a data-collection task: extend the ticker universe to include
  delisted B3 companies with their real (truncated) histories. Until then, treat every performance
  number produced from this dataset as an upper bound. **Remains open as a Stage-1 collection task.**

- [x] **T4 — Timestamp alignment.** ✅ DONE (2026-07-11): `no weekend trade_date rows` gate added
      to `test_final_dataset.py:validate()` — passes (0 found). Gap continuity beyond that is already
      warned at collection time (`validate.py`, >5-calendar-day gaps); a full B3 holiday-calendar check
      was skipped deliberately (needs an external calendar dependency for marginal value).

- [x] **T5 — Merge correctness.** ✅ DONE (2026-07-11): `asof merge picks most recent filed quarter (sampled)` gate
      in `validate()` — independently recomputes, from the **raw** fundamentals files + the filing-lag
      constants, which quarter should be visible on each sampled trade_date (100 sampled rows × 3
      tickers) and asserts the dataset's merged `reference_date` matches. 0 mismatches.

- [x] **T6 — Unexpected distributions.** ✅ DONE (2026-07-11): every build now writes
      `data/processed/ml_dataset.manifest.json` with per-column `nan_pct/mean/std/p1/p50/p99` — the
      comparable snapshot exists. Automated diff-against-previous-build is not built (manifests overwrite
      in place); comparing two builds is currently a manual diff of saved manifests. Build the automated
      diff when there's a workflow that keeps historical manifests (couples naturally with T23's
      `dataset_v{N}` versioning).

- [x] **T7 — Outlier/staleness gating.** ✅ DONE (2026-07-11): `--strict` flag added to `test_final_dataset.py` —
      default keeps the anomaly report informational (legitimate extremes land there alongside errors);
      `--strict` turns any stale-price/outlier finding into exit 1. The one confirmed-garbage case the
      triage surfaced (WDCN3) was handled at the source instead — see T8.

- [x] **T8 — Adjusted-price consistency.** ✅ DONE (2026-07-11), and it uncovered the **worst data-quality bug in the
      dataset**: 53 corporate events (of 577 in the audit log) were never adjusted in BolsAI's `adj_*`
      columns — fake single-day "returns" up to −99.99% (ln −9.3) sat in `log_return` for ~40 tickers
      including CSNA3, CMIG4, SBSP3, TIMS3, poisoning returns/volatility/drawdown/momentum and any reward
      built on them. Verified against raw files: raw `close` is continuous at these events while `adj_close`
      jumps — upstream back-adjustment anchored wrong. Fixes shipped in `build_ml_dataset.py`:
  - `repair_unadjusted_splits()`: detects each event's jump `ln(1/factor)` in the adj series near the
    (month-granular) recorded date and rescales all `adj_*` history before it. Matches **both**
    directions (the audit log's factor convention is inconsistent: SBSP3 records 0.2 for a ×5 basis
    change, ETER3 records 100 for ÷100) and repairs chronologically with rescan, which correctly
    handles multi-step upstream messes (TIMS3's ÷10,000 applied as two ÷100 steps; SBSP3's three
    basis re-anchorings in one week; CASH3/CEGR3/DTCY3 glitch-and-revert pairs). 53 repairs applied,
    each printed for auditability. Known ceiling (ponytail comment): events with |ln(1/factor)| < 0.3
    are indistinguishable from market moves and left alone.
  - `hl_ratio` fixed to use `adj_high/adj_low` instead of raw `high/low` — the old formula mixed raw
    and adjusted price scales and was meaningless whenever cumulative adjustment ≠ 1.
  - **WDCN3 quarantined** (`QUARANTINED_TICKERS`, documented reason): its raw feed alternates between
    two price bases ~6× apart hundreds of times 2021-2025 — not a split, no factor to repair with.
    **Test:** `no unadjusted split jumps in log_return` gate in `validate()` — rescans every event
    (both directions) against the final dataset; 0 events leaking. Extreme-return census after repair:
    190 → 108 rows > |1.2| ex-WDCN3, all 1-5 per ticker penny-stock moves (informational).

- [x] **T9 — `has_fundamentals` consistency.** ✅ DONE (2026-07-11): gate asserts `pl/pvp/roe/net_income/market_cap`
      are all-NaN wherever `has_fundamentals == 0` (0 leaked values), plus the filing-lag gate asserts
      `days_since_fundamental >= 45` with no negatives. The original "< ~120 days" upper-bound idea was
      dropped — it was written pre-T31; with the statutory lag the legitimate staleness range extends to
      ~185 days (Q3 filing carried until Q4's +90d annual deadline), so a hard upper bound would false-fail.

- [x] **T10 — Reproducibility manifest.** ✅ DONE (2026-07-11): `write_manifest()` in `build_ml_dataset.py` writes
      `ml_dataset.manifest.json` per build — git commit, pandas/numpy versions, build timestamp, row/ticker
      counts, date range, column list, and per-column distribution stats (doubles as T6's snapshot).
      Input-file hashing skipped (hashing 300 parquets per build for marginal provenance value); row
      counts + git commit of the tracked raw data cover the same question in practice.

_(Data leakage/lookahead → T1/T2 above. Duplicate observations beyond exact-row dedup and
incorrect-join direction beyond T5 are already adequately covered or low-priority — see inline notes
in the original table; not worth separate tasks until intraday data is ingested.)_

---

## 2. Which engineered features actually matter for an RL portfolio agent

**What Stage 2 already computes** (grounded in `build_ml_dataset.py`, confirmed line ranges):

| Group                | Features                                                                                                       | Windows                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Returns              | `log_return`, `return_1m/3m/6m/12m`                                                                            | 1d, 21/63/126/252d rolling sum |
| Trend                | `ma_20`, `ma_60`                                                                                               | 20/60d                         |
| Volatility           | `volatility_20d`, `volatility_60d`                                                                             | 20/60d                         |
| Momentum osc.        | `rsi_14`                                                                                                       | 14d                            |
| Risk                 | `drawdown` (expanding), `drawdown_percentile` (252d)                                                           |                                |
| Volume-adjacent      | `hl_ratio`                                                                                                     | —                              |
| Dividends            | `div_yield_12m`, `div_count_12m`                                                                               | 252 calendar days              |
| Fundamentals         | value ratios, YoY growth, QoQ trend, partial Piotroski `f_score`                                               | 4q, 20q (CAGR)                 |
| Macro-adjusted       | `excess_return`, `real_return`, `selic_trend_20d`                                                              |                                |
| Cross-sectional      | `pl/pvp/roe/debt_equity_zscore_sector`, `div_yield_sector_percentile`, `momentum_vs_market/sector_{1m,3m,12m}` | same-day                       |
| History-relative     | `price_percentile_5y`, `pl_percentile_5y`                                                                      | 1260d                          |
| Valuation composites | `peg_ratio`, `pvp_to_roe_ratio`, `earnings_yield_vs_selic`, etc.                                               |                                |

**Gaps worth closing:**

- [ ] **T11 — Add rolling Sharpe.** Cheap, directly reward-relevant for an allocator agent.
      **Goal:** `sharpe_60d` (or similar) = `mean(log_return, w) / std(log_return, w) * sqrt(252)`, window
      ≈ 60-126d, added to `compute_price_features`.
      **Test:** unit test in `test_build_dataset_features.py` on a synthetic constant-drift series with
      known mean/std, assert the computed value matches the closed-form Sharpe within tolerance.

- [ ] **T12 — Add rolling Sortino** — only if the reward function or agent state needs to distinguish
      downside vol from total vol; otherwise it's redundant with `volatility_60d` + `drawdown`.
      **Goal:** decision made explicitly (add or skip) based on whether `src/agent`'s reward is
      asymmetric; if added, `sortino_60d` using downside deviation only.
      **Test:** same pattern as T11 — synthetic series with known downside deviation, assert formula
      match.

- [ ] **T13 — Add rolling Beta vs BOVA11.** A portfolio agent needs systematic-risk exposure per
      asset; BOVA11 is already merged as the benchmark, so this is a rolling covariance calc away.
      **Goal:** `beta_252d` = `rolling_cov(ticker log_return, BOVA11 log_return, 252) / rolling_var(BOVA11
log_return, 252)`.
      **Test:** unit test with a synthetic ticker series constructed as `beta * market_return + noise`,
      assert the computed beta recovers the known input beta within tolerance.

- [ ] **T14 — Add relative volume** (`volume / rolling_mean(volume, 20)`) — only if liquidity/impact
      matters to the agent's action space (position sizing vs. a stock's typical liquidity); skip
      OBV/volume-price indicators as TA-signal features with no clear RL-state justification.
      **Goal:** decision made explicit against the agent's actual action space (visible only on
      `ml_agent` — check there before building this blind).
      **Test:** if added, formula unit test same pattern as T11.

- [ ] **T15 — Add one market regime feature** — a single rolling market-wide volatility percentile,
      not a taxonomy of regime flags.
      **Goal:** the agent has explicit regime context instead of having to infer it from many tickers'
      `volatility_20d` implicitly.
      **Test:** formula unit test; also assert the feature is a same-day cross-sectional aggregate (no
      per-ticker leakage), same causality-test pattern as T2.

- [ ] **T16 — Add market-wide percentile rank for `return_1m/3m/12m`** (sector-level equivalent
      already exists via `*_zscore_sector`) — cheap, same `groupby("trade_date")` pattern already used in
      `compute_advanced_features`.
      **Goal:** `return_1m_pct_rank_market` (etc.) added alongside the existing sector versions.
      **Test:** formula unit test on synthetic same-day multi-ticker data with known rank order.

- [ ] **T17 — Verify `earnings_yield`'s double definition is intentional, not a bug.** Defined once in
      `compute_fundamental_features` (~line 500s) and again in `compute_advanced_features:769` as `1/pl` —
      the second silently overwrites the first.
      **Goal:** confirmed (via git blame / author judgment) that the second definition is meant to
      supersede the first, or fixed if it's an accidental collision.
      **Test:** if intentional, add a comment at the first definition noting it's superseded downstream;
      if not, rename one and add a test asserting both columns exist with their distinct formulas.

**Explicitly skip (not tasks):** full pairwise correlation matrix (N² feature space, no clear agent
need yet), a separately-named relative-strength indicator (`momentum_vs_market/sector_*` already
covers it), calendar features (weak prior for daily-rebalance RL, route through §3's ablation process
if you have a specific hypothesis rather than adding speculatively).

---

## 3. Signal vs. noise — how to actually decide

~89 engineered columns after `compute_advanced_features`, no feature-selection infrastructure yet.
Cheapest/most-diagnostic first — each step's "Test" is what makes it a checkable task, not just advice:

- [ ] **T18 — Correlation + VIF triage pass.** Cluster the fundamental-ratio and momentum families
      (`peg_ratio` vs `pl` vs `earnings_yield`; `momentum_vs_market_*` vs `momentum_vs_sector_*` vs
      `return_*`).
      **Goal:** a documented list of high-VIF (>10) columns whose information is captured elsewhere —
      drop candidates, not yet dropped.
      **Test:** one script (`tools/` or a notebook cell) that outputs a VIF table for all numeric feature
      columns against `data/processed/ml_dataset.parquet`; done when that table exists and is reviewed,
      not when VIF is merely "low" — this is a triage artifact, not a pass/fail gate.

- [ ] **T19 — Mutual information / correlation with realized forward return** (a proxy target, not the
      agent's actual reward).
      **Goal:** a ranked list of features by univariate signal, used only to _deprioritize_ what gets
      tested first in T21 — not to decide inclusion (RL agents use features combinatorially; zero
      univariate MI doesn't mean zero value).
      **Test:** script outputs MI/correlation per feature vs. forward N-day return; done when the ranked
      list exists.

- [ ] **T20 — PCA per feature family** (fundamentals, price-technicals, cross-sectional) — diagnostic,
      not production dimensionality reduction.
      **Goal:** a concrete number per family (e.g. "3 components explain 95% of variance in the 10-feature
      momentum family") to justify pruning decisions with evidence, not intuition.
      **Test:** script outputs explained-variance-ratio per family; done when the numbers exist and are
      reviewed.

- [ ] **T21 — Ablation studies against actual RL agent validation performance — the only test that
      decides inclusion.** Feature importance/SHAP/permutation on a supervised proxy model answers "does
      this predict returns," not "does this help the policy" — a feature can be useless for return
      prediction and still valuable as agent state (e.g. `has_fundamentals` tells the agent when to trust
      the fundamental block, without itself being predictive).
      **Goal:** for each feature _family_ (not individual column — too correlated for single-column
      ablation to be meaningful against RL's own training variance), a significance-tested verdict:
      removing it either significantly hurts validation performance (keep), significantly helps (drop —
      it was actively harmful), or neither (drop for simplicity, since it adds nothing).
      **Test:** was blocked on T28 (walk-forward split must exist first — fixed typo, this previously
      misreferenced T13/T14) — **unblocked**, `split_config.json` now exists (§4/§6). Fixed train/val
      split, N-seed baseline (full feature set) vs. N-seed leave-one-family-out runs, compared via the
      same HAC+bootstrap significance test already used in the agent's diagnosis work (per `notes.md`) —
      not point-estimate comparison.

- [ ] **T22 — Sequential/recursive feature elimination on the supervised proxy model only** — not
      worth building against the RL training loop directly; T21's per-family ablation is the RL-relevant
      substitute and far cheaper than per-feature RFE against real training wall-clock time.
      **Goal:** optional — only pursue if T18-T20 leave ambiguity T21 can't cheaply resolve at the
      per-column level.
      **Test:** standard sklearn RFE against the proxy model from T19; report which columns it drops.

---

## 4. Processed-data directory architecture

Previous state: `data/processed/` held three large files with no manifest, no versioning, and (per
CLAUDE.md) no clear ownership — `ml_dataset.parquet` is built by this repo, but
`ml_dataset_training.parquet` and `agent_tensors.npz` are produced by pipeline code that isn't in this
branch (`src/agent/data_pipeline.py` on `ml_agent`), invisible to anyone checking out `refactor` alone.
**All four gaps below are now closed** (2026-07-11).

- [x] **T23 — Version the output directory.** ✅ DONE. `sync_dataset_version()` in `build_ml_dataset.py`
      snapshots the current build (parquet + manifest + `split_config.json`) into
      `data/processed/dataset_v{N}/` after every build. `N` only bumps when the new manifest's content
      fingerprint (`rows`/`tickers`/`date_min`/`date_max`/`columns`/`column_stats` — deliberately excluding
      `built_at`/`git_commit`, which always differ) differs from the latest existing version; an unchanged
      rerun is skipped and logged rather than piling up another 250MB+ copy. Files are `shutil.copy2`'d
      from the just-written flat outputs, not re-serialized, so versioning costs a file copy, not a second
      `to_parquet()`. The flat `ml_dataset.parquet`/`.manifest.json`/`split_config.json` paths are left in
      place (every existing consumer — `test_final_dataset.py`, `test_ticker_data.py`, the CLAUDE.md-documented
      run commands — reads from there); `dataset_v{N}/` is purely additive.
      **Test:** `test_dataset_versioning.py` (3 cases) — first build creates `dataset_v1`; an identical
      rerun is skipped (no `dataset_v2`); a content change creates `dataset_v2`.

- [x] **T24 — Add `split_config.json`.** ✅ DONE, with one deliberate simplification from the original
      spec: no materialized `train/val/test.parquet` copies. `compute_split_dates()` picks walk-forward
      cutoffs over **unique `trade_date`**, not row count — a row-count split would let long-history
      tickers drag the boundary later than a short-history ticker would, since tickers don't all start on
      the same date. `write_split_config()` writes `train_end`/`val_end` (default 70/15/15) plus row counts
      per split to `data/processed/split_config.json`. Consumers filter `ml_dataset.parquet` by date against
      these two cutoffs at load time instead of three parquet files that could drift out of sync with the
      source.
      **Test:** `test_split_config.py` (2 cases) — `train_end < val_end` and both are real trading dates;
      the cutoff dates are provably identical whether or not a short-history ticker is mixed into the input
      (the specific failure mode a row-count split would have).
      **Note:** this also satisfies §6's **T28** (walk-forward split logic) — same deliverable, same test.

- [x] **T25 — Add `data/processed/scalers/feature_scaler.joblib` + `scaler_metadata.json`, fit only on
      the train split.** ✅ DONE. Implemented with `sklearn.compose.ColumnTransformer` (already an
      installed dependency, pinned for Stage 3) rather than hand-rolled per-column scaling code —
      `RobustScaler` on a `RATIO_COLUMNS` list (54 unitless ratios/margins/leverage/growth-rate columns:
      `pl`, `pvp`, `ev_ebitda`, `debt_equity`, `cagr_*`, etc.), `remainder="passthrough"` for everything
      else (percentiles, z-scores, binary flags, returns, prices, identifiers) — no need to enumerate the
      ~80 passthrough columns by hand. `RobustScaler` in the installed sklearn version (1.9.0, verified
      directly) ignores NaN when fitting (nanmedian/IQR) and leaves NaN as NaN on transform, matching the
      project's no-imputation invariant for free. Fit only on rows at/before `split_config.json`'s
      `train_end` (`fit_scaler_on_train_split()`). One real gotcha caught and fixed:
      `ColumnTransformer`'s pandas output groups columns by transformer (all `robust` columns, then all
      passthrough columns), not original order — `transform_features()` reindexes the output back to the
      input's column order so nothing downstream can silently read the wrong column by position.
      `scaler_metadata.json` (`center`/`scale`/`scaled_columns`/`passthrough_columns`) is derived from the
      fitted transformer's own attributes (`center_`, `scale_`, `feature_names_in_`), not hand-copied, so
      it can't drift from what was actually fit. Standalone script
      (`python -m src.build_dataset.scale_features`), not wired into `build_ml_dataset.py`'s `main()` —
      refitting the scaler is a deliberate action, not something that should happen silently on every
      build. `joblib` (already an implicit transitive dependency of sklearn) added to `requirements.txt`
      as a direct dependency since it's now imported directly.
      **Test:** `test_scale_features.py` (5 cases) — column-order reindex after transform, ratio columns
      actually change while passthrough columns and NaN don't, refit-on-train reproduces identical
      `center_`/`scale_` (proves it wasn't fit on val/test), and `RATIO_COLUMNS` never contains a known
      percentile/z-score/binary column.
      **Note:** this also satisfies §6's **T29** (feature scaling, train-only fit) — same deliverable, same
      test.

- [x] **T26 — Directory ownership documented.** ✅ DONE — broader than the original "validation_reports"
      framing, closes the actual gap named in this section's intro. `data/processed/README.md` documents
      which files this repo builds (`ml_dataset.parquet`, `.manifest.json`, `split_config.json`,
      `dataset_v{N}/`, `scalers/`) vs. the foreign `ml_dataset_training.parquet` produced by the `ml_agent`
      branch's `src/agent/data_pipeline.py`. Tracking this required fixing the `.gitignore` rule itself:
      `data/processed/` (directory-level ignore) silently blocks a `!data/processed/README.md` negation
      from taking effect — git won't descend into an ignored directory to evaluate per-file exceptions.
      Changed to `data/processed/*` (ignore contents, not the directory) so the negation actually works.
      `data/validation_reports/` (the original T6-adjacent ask) remains unbuilt — no automated
      diff-against-previous-build exists yet; comparing two manifests is still a manual step. Build that
      when there's a workflow that consumes historical manifests programmatically (natural next use of
      `dataset_v{N}`, now that the versioned history exists to diff against).

**Skip:** `data/interim/` staging (only worth it if build time becomes a bottleneck — it isn't yet), a
YAML-driven feature-config registry (a `FEATURES.md` doc or a `FEATURE_COLUMNS` dict co-located with
the code is enough for ~90 features from one script). Agent-side artifacts (`agent_tensors.npz`,
training parquet, checkpoints) belong versioned with the `ml_agent` code that produces them, not
restructured here.

---

## 5. Missing values

`notes.md` already documents a measured **15.23% overall NaN rate** with a per-column breakdown — use
that as the baseline rather than re-deriving it.

| Source                            | Example columns                                                 | Right default                                                                                                                   |
| --------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Rolling-window warm-up            | `ma_60`, `volatility_60d`, `return_12m` (first 252 rows/ticker) | **Preserve as NaN** — forward-filling fabricates a year of fake signal; let the agent's state pipeline mask/zero explicitly.    |
| Recent IPOs                       | any rolling feature, ticker's first N days                      | Same — preserve; absence of a 5-year price percentile in year one is information, not a defect.                                 |
| Missing fundamentals this quarter | fundamental-derived columns when `has_fundamentals=0`           | **Preserve, rely on the flag** — already the design (CLAUDE.md caveats), and the right one.                                     |
| Sparse macro (BCB gaps, holidays) | `selic`, `cdi`, `ipca`                                          | Forward-fill is defensible — rates persist between publications by nature, unlike fundamentals.                                 |
| Delisted companies                | trailing history for a delisted ticker                          | Don't fill through a delisting — the row should stop existing at the delisting date, not interpolate implied continued trading. |
| Single-stock sectors              | `*_zscore_sector` when `std<=0`                                 | Already correctly NaN'd (`std.where(std > 0)`, line 709) — no action needed.                                                    |

- [x] **T27 — Macro fill mechanism confirmed by inspection.** ✅ DONE (2026-07-11): `merge_macro()` uses
      `merge_asof(..., direction="backward")` on each series — i.e. last-published-value carry-forward,
      causal by construction, no gap-dropping and no interpolation. No change needed.

General principle for an RL pipeline: don't let Stage 2 decide how NaNs are _consumed_ — masking,
zero-filling, or learned-embedding-for-missing-state is a modeling decision that belongs in
`src/agent`, not baked irreversibly into the parquet via imputation. Statistical imputation
(mean/median/KNN) is not recommended anywhere in this pipeline — imputing a financial time-series value
implies information you didn't have at the time, which directly undermines the project's own
no-lookahead invariant. A high NaN rate on a specific feature is usually an upstream collection gap to
fix, not a downstream imputation problem.

---

## 6. Overall assessment

**What's solid:** the merge-level no-lookahead invariant (`merge_asof backward` + explicit test), the
formula-level unit tests in `test_build_dataset_features.py`, the `recompute_valuation_daily`
regression guard (a real, previously-fixed bug with a test protecting against regression — the exact
pattern T1/T2 extends), and the collection-time schema gates in `validate.py`.

**Structural gap, now closed:**

- [x] **T28 — Build leak-safe walk-forward train/val/test split logic.** ✅ DONE via §4's T24
      (`split_config.json`) — same deliverable, same test. Not implemented anywhere in `src/` before this
      (confirmed: zero `train_test_split`/`TimeSeriesSplit` hits repo-wide). This was blocking T21's entire
      ablation methodology and any rigorous performance claim about the agent; unblocked now. Whether
      `ml_agent`'s own training loop actually consumes `split_config.json` (vs. its own ad hoc split) is
      unverified from this branch — worth checking before relying on it for T21.

- [x] **T29 — Build feature scaling, fit train-only.** ✅ DONE via §4's T25
      (`scale_features.py`) — same deliverable, same test.

---

## 7. Priority-ordered task index

**✅ COMPLETED (2026-07-11 data-quality pass):**

1. ✅ **T31** — fixed the fundamental publication-lag leak (~40+ columns affected, every
   quarter, every ticker — bigger blast radius than T1, verified against real CVM data)
2. ✅ **T1** — fixed the volatility-percentile lookahead bug
3. ✅ **T2** — causality-test pattern implemented (currently covers the T1 columns; extending
   to more columns is still open, see §0)
4. ✅ **T4–T10, T27** — 5 new validation gates in `test_final_dataset.py` (18 total, all passing),
   the T8 split-repair (53 events fixed) + `hl_ratio` fix + WDCN3 quarantine in `build_ml_dataset.py`,
   `--strict` mode, and the build manifest. Details in §1.

**✅ COMPLETED (2026-07-11 `data/processed/` architecture pass):**

1. ✅ **T23** — `dataset_v{N}` versioning, dedup on unchanged rerun (fingerprint = manifest content,
   excluding `built_at`/`git_commit`)
2. ✅ **T24 / T28** — walk-forward `split_config.json` (date cutoffs, not materialized parquet copies),
   split on unique `trade_date` so it's robust to uneven per-ticker history — the single
   highest-priority item from the previous pass, now unblocking T21
3. ✅ **T25 / T29** — `feature_scaler.joblib` + `scaler_metadata.json`, `sklearn.ColumnTransformer`
   (`RobustScaler` on ratio columns, passthrough elsewhere), fit train-only, output reindexed back to
   original column order
4. ✅ **T26** — `data/processed/README.md` documents build ownership; `.gitignore` fixed
   (`data/processed/*`, not `data/processed/`) so the exception actually takes effect

**STRUCTURAL FINDINGS (not tasks, but context):**

- **T3** — ⚠️ survivorship bias CONFIRMED structural (universe is 100% `ATIVO`, zero delisted
  companies). Open Stage-1 collection task: backfill delisted B3 tickers. Until then all backtest
  numbers are upper bounds.

**DEFERRED (agent branch, `ml_agent`):**

- **T32** — audit agent observation for Markov sufficiency
- **T33** — audit train/inference feature parity
- Whether `ml_agent`'s training loop actually consumes the new `split_config.json` / `feature_scaler.joblib`
  rather than its own ad hoc split/scaling — unverified from this branch, worth checking before citing
  T21 results against them.

**REMAINING PRIORITIES (§2–§6):**

1. **T18–T20** — cheap feature triage (§3), do before T21
2. **T21–T22** — ablation-based feature decisions (§3) — unblocked now that T24/T28 exist
3. **T11–T17** — new feature candidates (§2), lowest priority
4. `data/validation_reports/` automated manifest-diff (T6/T26-adjacent) — couples naturally with the
   `dataset_v{N}` history that now exists to diff against; still manual today
5. **T30** — resolve cross-branch fragmentation (reproducibility risk, not urgent)

**Not worth doing at all:** YAML-driven feature-config system, full pairwise-correlation feature
matrix, calendar-feature suite, premature `data/interim/` caching, and a per-feature compute/storage
cost-accounting framework — this is a once-per-build batch script over ~500 tickers, not a hot path;
revisit only if build time actually becomes a bottleneck.

---

## 8. Out of scope here — flag for audit on `ml_agent`

Two questions a second reviewer raised are real and matter more than any single feature in §2, but
can't be answered from this branch — `src/agent/*` only exists on `ml_agent`, not `refactor` (see the
scope note at the top of this document). Recorded here so they don't get lost, not specced blind.

- [ ] **T32 — Audit whether the agent's observation satisfies the Markov property.** Reviewing
      features individually (§2) answers "is column X useful," not "does the complete state the agent sees
      each step contain enough information to act well" — a different, higher-level question. Concretely:
      does `src/agent/env.py`'s observation construction include portfolio weights, cash position, the
      previous action, current drawdown/exposure, and time-remaining-in-episode — typically appended at
      `env.step()` time, not baked into the static per-ticker dataset this repo builds?
      **Goal:** a checklist answer (present/absent) for each of those five items against the actual
      `env.py` observation-construction code.
      **Test:** not a code test — a read-through of `src/agent/env.py`'s observation/state builder on the
      `ml_agent` branch, checked off against the five items above.

- [ ] **T33 — Audit train/inference feature parity.** Several Stage-2 features are cross-sectional or
      full-history-dependent by construction (`price_percentile_5y`, `pl_zscore_sector`,
      `momentum_vs_market/sector_*`, and after T1's fix, the volatility percentiles too) — computing one of
      these for a single live trading day requires the same full historical/cross-sectional context used at
      training time. If `src/agent/infer.py` (confirmed to exist on `ml_agent`) recomputes these
      independently rather than reusing the exact Stage-2 functions, a train/inference mismatch is likely.
      **Goal:** confirmed that `infer.py` calls into `src/build_dataset`'s actual feature functions (or an
      exact port of them), not a reimplementation that can silently drift.
      **Test:** for a fixed historical date, assert `infer.py`'s computed feature vector for a ticker
      matches the corresponding row in `ml_dataset.parquet` exactly, for every cross-sectional/full-history
      feature.
