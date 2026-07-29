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

## 3. 100%-NaN `net_revenue` in the XBRL tier — pothole #9 — ✅ DONE 2026-07-29

Measured for real across all 1,848 collected tickers (not the 8/120 sample): 170 (9.2%)
affected — 163 with no `net_revenue` column at all, 7 more (AB, CFNB, COLB, CVBF, DX, GS,
NLY) where the column exists (from an ex27/item6-tier row) but is 100%-NaN in the xbrl
tier. Inspected raw companyfacts for a sample (ABCB/AGNC/ARCC/AGM/GS/NLY/COLB) — confirmed
cluster is banks/thrifts/mortgage REITs/BDCs/GSEs, all reporting
`InterestIncomeExpenseNet` + `NoninterestIncome` instead of a single gross-revenue tag.

**Decision: not fixed by adding a concept.** Net interest income is already a spread (income
minus interest expense), not the same economic quantity as an industrial company's gross
`Revenues` — mapping it into `net_revenue` would silently corrupt every revenue-based ratio
(P/S, revenue CAGR, margins) for this whole sector cluster, worse than the NaN it would
replace. Documented next to `CONCEPT_MAP`'s `net_revenue` entry (`companyfacts.py`), same
treatment as the existing `total_debt`/banks gap. `US_COLLECTOR_BUG_AUDIT.md` item 9 closed
with this finding.

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
- [x] **Run the never-executed Phase 0 test: does Alpha Vantage's free
      `LISTING_STATUS&state=delisted` return a delisted roster *with prices*?** ✅ DONE
      2026-07-29, both YES:
      - `LISTING_STATUS&state=delisted` (real free key, not `demo`) returned **9,390** real
        symbols with `symbol/name/exchange/ipoDate/delistingDate/status`.
      - `TIME_SERIES_DAILY` on two of them (AA-W, AABA) returned real daily OHLCV data
        ending exactly on each symbol's `delistingDate` — confirmed against known
        history (AABA/Altaba delisted 2019-11-06, price series ends that exact date with
        real volume, not a placeholder).
      - **The catch:** free tier is capped at **25 requests/day, total**. At that rate,
        pulling prices for all 9,390 delisted symbols is ~376 days serially — technically
        free, not practically free at this universe's scale. `outputsize=compact` (~100
        rows) is what was tested; `full` history depth is unverified and doesn't change
        the rate-limit math anyway (still 1 request per symbol).
      - Not yet checked: whether AV's roster is ticker-only or also exposes/derivable-to a
        CIK (needed to join delisted prices back to SEC fundamentals for the same company).
- [ ] **Then decide, explicitly** (sharpened by the result above — this is no longer "does
      free work," it's "is paying for AV premium or Sharadar worth it"):
      - Pay for AV premium (lifts the 25/day cap) or Sharadar (~$50/mo, per §8) → build
        crosswalk tiers 2-4 (dead-company CIK recovery, to join AV's ticker-keyed delisted
        roster back to a CIK for fundamentals) and collect the dead universe. Forces the
        file-naming question: dead companies have no current ticker, so output must key on
        **CIK**, with ticker demoted to a join column (already prescribed by the plan's own
        risk table — "key everything on CIK, never on ticker").
      - Stay free tier → the 25/day cap makes a full recovery impractical; either accept
        the bias and *document it loudly* in the dataset manifest (mirroring BR's
        `TOP50_UNIVERSE_VALIDATION.md` treatment), or use the free tier for a small,
        deliberately-scoped subset (e.g. only the largest/most-known delisted names) rather
        than the full ~9,390.

Until one of those lands, any US backtest is survivorship-biased and must be labelled so.
This is a spend-money-or-accept-bias decision now, not a research question — the research
is done.

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
