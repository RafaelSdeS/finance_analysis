# US collector — fix plan (2026-07-29)

Consolidates every open item from `US_COLLECTOR_BUG_AUDIT.md` (potholes 7-10) and
`US_EQUITIES_EXPANSION_PLAN.md` (Phase 0/3/6 open boxes, §8 decisions) into one ordered
list. Ranked by value/effort, not by document order. Items marked **NOT DOING** are
deliberate closures, not oversights.

## 0. Current state (read this first)

A full collection run is in flight (`run_us_full_scale prices fundamentals`, started
2026-07-29 15:24). Two corrections to the working assumptions around it:

- **The run is 10,432 tickers, not 2,462.** `run_prices()` passes the entire tier-1
  crosswalk (`cik_ticker_crosswalk.parquet`, 10,432 rows). The 2,462 price files on disk
  were partial progress from the killed runs, never a coverage ceiling. At the observed
  ~15 tickers/min (`YF_RATE_LIMIT_SLEEP = 1.0`, deliberately slow to avoid Yahoo
  throttling), the prices step alone is **~12 hours**, then fundamentals on top.
- **Code edits made now do not reach the running job.** Python imported
  `sec/`, `yf_collectors.py`, and `config.py` at process start. Every fix below therefore
  lands for the *next* run — which is fine, because fundamentals rebuilds from scratch
  anyway (see item 2).

**Sequencing rule:** do not touch the prices path or restart the run. Prices are the
expensive, throttle-sensitive half and are already correct. All fundamentals-side fixes
below batch into a single `run_us_full_scale fundamentals` re-run afterward.

---

## 1. `SEC_USER_AGENT` still ships the placeholder — pothole #7

`config.py:33` defaults to `contact@example.com`; SEC's fair-access policy asks for a real
contact so heavy traffic gets an email rather than a block. `load_env()` runs before the
`os.environ.get` on line 33, so a `.env` entry is picked up correctly (verified).

- [ ] Add to `.env` (gitignored, local only):
      `SEC_USER_AGENT=finance-analysis-research rafesilvadesouza@gmail.com`
- [ ] Add a **placeholder** line to `.env.example` (tracked — never the real address).

Takes effect next run. No behavioral change beyond the header value.

---

## 2. `collect_fundamentals_us` has no resume — pothole #8

Rebuilds every ticker from scratch and overwrites each parquet outright. This is *partly
deliberate*: a derivation fix (item6 cascade, predecessor cutoffs, CONCEPT_MAP below) must
reach every already-collected company, not just new ones — a naive checkpoint would freeze
old, wrong rows in place. The real gap is crash-resume, not incremental collection.

- [ ] Add `skip_existing: bool = False` to `collect_fundamentals_us`; in the per-ticker
      worker, return early if the output parquet already exists. Two lines.
- [ ] Default **off** so today's rebuild-everything semantics are unchanged; pass
      `skip_existing=True` only when resuming an interrupted run.

Do **not** wire in `checkpoint.py` + `_merge_save` here — the per-ticker output parquet
already is the idempotent unit, and `_merge_save`'s append-and-dedup is wrong for a table
that must be fully recomputed when a derivation changes.

---

## 3. 100%-NaN `net_revenue` in the XBRL tier — pothole #9

8 of 120 sampled tickers have a fully-empty revenue column; `CONCEPT_MAP` is missing their
tag vocabulary, suspected financials/REITs. Never measured beyond that sample.

Caveat that changes the fix: for a bank, "revenue" genuinely has no single XBRL analog
(net interest income + noninterest income, tagged separately). Part of this cluster is
**correctly empty**, and forcing a tag would fabricate a number. Measure before mapping.

- [ ] Sweep all collected `data/raw/us/fundamentals/*.parquet` for 100%-NaN `net_revenue`
      in xbrl-tier rows — get the real rate and the sector/filer-type cluster, not 8/120.
- [ ] For a sample of those CIKs, list which `us-gaap`/`ifrs-full` concepts matching
      `Revenue`/`Income` *are* populated in their companyfacts. Data-driven, not guessed.
- [ ] Add only the concepts that are genuine revenue analogs to `CONCEPT_MAP`'s ordered
      fallback list (same pattern as the existing IFRS additions).
- [ ] For filer types with no true analog, leave NaN and document it — same treatment as
      the acknowledged `total_debt` gap for banks. A flagged NaN beats an invented figure.

Steps 1-2 can run against the existing 1,848 files immediately; no need to wait for the
collection to finish.

---

## 4. Survivorship bias — the one that actually matters

Present and unfixed. Two independent halves, and the cheap half gates the expensive half.

**What's on disk already:** `sec/universe.py` builds a genuinely point-in-time,
survivorship-bias-free roster (`us_universe_roster.parquet`) — verified to drop Lehman,
WorldCom, Enron, and pre-buyout Twitter in the exact quarter each actually collapsed. It
is simply **not connected to what gets collected**: `run_us_full_scale._all_tickers()`
reads the *current-listings* crosswalk instead. The instrument exists; the wiring doesn't.

**First measurement (2026-07-29, both parquets already local):**

| | count |
|---|---|
| roster CIKs (ever filed 10-K/10-Q since 1994) | 43,366 |
| crosswalk CIKs (currently listed) | 8,017 |
| roster CIKs with **no** current ticker | 37,951 (87.5%) |
| └ last filed ≥ 2020 | 5,405 |
| └ last filed 2000-2019 | 26,608 |
| └ last filed < 2000 | 5,938 |

**Do not quote 87.5% as the equity survivorship gap.** The roster counts every 10-K/10-Q
filer, which sweeps in debt-only registrants (bond covenants, no public stock), subsidiary
co-registrants, LPs/trusts, and blank-check shells that never had listed common equity.
It's an upper bound, and an inflated one. The ~5,405 that last filed since 2020 are the
most credible "recently disappeared, plausibly once-investable" bound.

- [ ] **Tighten the number** before building anything: restrict roster CIKs to those with
      real equity characteristics (e.g. non-trivial filing history + a resolvable former
      ticker) to convert 37,951 into a defensible figure.
- [ ] **Run the never-executed Phase 0 test: does Alpha Vantage's free
      `LISTING_STATUS&state=delisted` return a delisted roster *with prices*?** This is one
      afternoon and it decides everything downstream. Fundamentals for dead CIKs are
      recoverable free (SEC keeps every filing; `build_company_fundamentals` already keys
      on CIK) — but **fundamentals without prices are near-worthless**: a company with no
      price series can't be a dataset row, can't contribute returns, and can't enter
      cross-sectional stats. Build the recovery ladder only if prices are obtainable.
- [ ] **Then decide, explicitly:**
      - AV serves delisted prices → build crosswalk tiers 2-4 (dead-company CIK recovery)
        and collect the dead universe. Note this forces the file-naming question: dead
        companies have no current ticker, so output must key on **CIK**, with ticker
        demoted to a join column (which is what the plan's own risk table already
        prescribes — "key everything on CIK, never on ticker").
      - AV doesn't → either accept the bias and *document it loudly* in the dataset
        manifest (mirroring BR's `TOP50_UNIVERSE_VALIDATION.md` treatment), or resolve
        §8's paid-data decision (Sharadar, ~$50/mo) if a bias-free US backtest is a hard
        requirement rather than a nice-to-have.

Until one of those lands, any US backtest is survivorship-biased and must be labelled so.

---

## 5. Stage 2 — `us_ml_dataset.parquet` (blocked on the running collection)

The largest single piece of remaining work, and the point of the whole exercise. Nothing
Stage-2-equivalent exists for US data yet.

- [ ] Audit which of BR's `src/build_dataset/` modules generalize as-is vs. need a US
      variant. Do this by reading, not assuming — the macro source differs (FRED vs BCB),
      and the US price path stores unadjusted + `adj_*` the same way BR does (confirmed in
      the running log: yfinance splits are reverse-adjusted to the same convention), so
      `repair.py`/`features.py` may reuse more than expected.
- [ ] Build the US assembly path: merge prices + fundamentals + FRED macro via
      `merge_asof(direction='backward')` on `fundamentals_available_date`, schema-aligned
      to BR per plan §5.5.
- [ ] Port the invariant tests: no-lookahead, prefix-shaped NaN, split repair,
      `fundamentals_tier` boundary sanity.
- [ ] Record the survivorship state (item 4) in the manifest as a first-class field, not a
      README footnote.

---

## 6. Lower value — do only if it starts costing something

- [ ] **Wire US into `pipeline.py` (`--market us`) — pothole #10.** Cosmetic today;
      `run_us_full_scale.py` works. Worth it only when someone other than you runs this.
- [ ] **`*_restated` companion column** (plan §Phase 4, deferred). US fundamentals are
      already correctly as-first-reported via `min(filed)`; this would additionally expose
      the restated value alongside. Genuinely nice-to-have.
- [ ] **Universe breadth (§8 decision 3).** Already resolved in practice — the running job
      collects all 10,432 crosswalk tickers, matching the plan's "collect broadly, filter
      at Stage 2" recommendation. Close this box once the run confirms it.
- [ ] **Phase 0's "confirm yfinance behaviour at scale"** — answered empirically rather
      than by experiment: throttling was hit at ~2,462 tickers on a 0.3s pace, fixed by
      the separate `YF_RATE_LIMIT_SLEEP = 1.0`. Mark it closed with that finding.

## NOT DOING (closed with rationale, not open work)

- **Phase 8 full-statement parsing** — explicitly deferred in the plan doc; the three-tier
  approach already reaches 1995.
- **24 OTC-exempt foreign ADRs with no fundamentals** (BAE, BMW, CSL, Deutsche Telekom,
  …) — verified to have *zero* SEC filings of any kind; Rule 12g3-2(b) exempts them.
  Nothing in EDGAR to collect. Accepted out of scope 2026-07-28.

---

## Appendix — BR-side open items (separate track, not part of this plan)

Listed for completeness since "all known issues" was the ask. Only the first is actionable.

- [ ] **As-reported vs as-restated fundamentals lookahead** (CLAUDE.md, 2026-07-23 audit
      Issue 8). `fundamentals_available_date` is correctly point-in-time, but the *figures*
      come from BolsAI's current snapshot — i.e. latest restatement. CVM's own open-data
      ZIPs (`cvm/statements.py`) hold every filing version and could source true v1
      figures. Real, unquantified, and a larger sourcing project. Note the US path does
      **not** have this bug — `companyfacts.py` takes `min(filed)` per concept/period.
- **Closed, no fix available:** `adj_close` 2-decimal vendor precision floor (flagged, not
  reconstructable); BolsAI/yfinance dividend-adjustment divergence (documented, recomputing
  would regress returns); split-matcher persistence guard (three designs built and tested
  against all 67 real events, all produced false rejections — reverted deliberately).
