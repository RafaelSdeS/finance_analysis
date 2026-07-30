# US collector audit — 2026-07-28

Read-only pass over `src/data_collection/fred_collectors.py` + `src/data_collection/sec/`.
Every "confirmed" claim below is measured against the data already on disk
(`data/raw/us/`, 1,251 fundamentals files / 2,055 price files), not inferred from the code.
Nothing was changed.

**Status (2026-07-29):** Bugs 1-6 fixed and tested. Potholes 7-10 still open. Three more real
bugs found in a follow-up pass at larger scale (1,848 fundamentals files) — see
`docs/US_EQUITIES_EXPANSION_PLAN.md`'s Phase 7 section for the write-up (item6.py year-detection
false positives + their fundamentals.py cascade; `_prices_fetch_start` trusting a truncated
first fetch).

**Status (2026-07-30):** Full-universe sweep (not a sample) found 2 more real bugs (11-12,
both fixed) plus 4 measured-but-checked-not-a-bug findings — see the dated section below.
Pothole #8 (`skip_existing`) turns out to already be implemented in code; the fix plan's
checkbox was just stale.

## Bugs

- [x] **1. Item 6 tier values are off by 10³–10⁶ and carry no ratios** — `sec/item6.py` +
  `sec/fundamentals.py:51-95`. Item 6 tables print figures under an "(in millions)" /
  "(in thousands)" caption; `_parse_value()` never reads that caption and
  `build_company_fundamentals()` never runs `compute_ratios()` on the gap tier (unlike
  xbrl and ex27, which both do). Confirmed per-ticker, same column, adjacent tiers:

  | ticker | item6 `net_revenue` median | same ticker, other tier |
  |---|---|---|
  | INTC | 3.42e4 | 1.36e10 (xbrl) |
  | IBM  | 9.14e4 | 7.39e10 (ex27) |
  | GE   | 1.74e4 | 4.12e10 (ex27) |

  Across a 120-ticker sample: 448 item6 rows, `roe` non-null = **0**. So the entire
  2001–2006 window is both a 10⁶ magnitude cliff sandwiched between two correct tiers
  *and* a total hole in every derived ratio. Fix: parse the units caption, scale, then
  call `compute_ratios(unit_scale=1)` like the other two tiers. Cheap guard: compare the
  scaled row against the nearest ex27/xbrl row for the same CIK — the ratio should be
  ~1, not ~1e6 — and reject the filing if not.

- [x] **2. Every fiscal Q4 loses its income statement** — `sec/companyfacts.py::_quarterly_only`.
  The 60–100 day duration filter drops annual durations, and most 10-K filers never tag a
  standalone Q4 duration, so revenue / net income / ebit / cashflow are NaN at each fiscal
  year-end while the balance sheet (instant concepts) survives. Confirmed on the same
  120-ticker sample: `net_revenue` NaN 22.9%, `net_income` 18.3%, `roe` 22.0%; **58.7%** of
  the NaN rows land in December vs **26.4%** of all rows, and per-ticker NaN count has
  median = 0.93 × (rows / 4) — exactly one per year. Fix: keep the annual duration when no
  quarterly sibling shares that `end`, and derive Q4 = FY − (Q1+Q2+Q3) for the flow items.

- [x] **3. `universe.fetch_quarter` freezes a partially-collected quarter forever** —
  `sec/universe.py:75-88`. Only `max(_quarters_through_now())` is re-fetched; a cache
  written mid-quarter becomes immutable the moment the next quarter starts. Right now
  `data/raw/us/sec/full_index/2026q3.parquet` holds **414 rows** (max `date_filed`
  2026-07-27) against ~6,500 for a full quarter. Rerun in October and ~6,000 Q3 filings
  are permanently missing from `edgar_10k10q_filings.parquet` — silently shrinking the
  ex27/item6 filing set and the survivorship-free roster. The module docstring only
  contemplates "a rare late-indexed filing"; this is systematic. Fix: also re-fetch when
  the cache's `date_filed` max is before the quarter's last day.

- [x] **4. `fds.measure_prevalence` is broken by the `.search()` → `.finditer()` refactor** —
  `sec/fds.py:196-213`. `parse_fds()` now returns a *list*, but the caller still treats it
  as a dict: `tags is not None` is True even for `[]` (so `has_ex27` is always True), and
  `(tags or {}).get("ARTICLE")` raises `AttributeError` on the first filing that actually
  *has* an EX-27. Diagnostic-only path — but it is the function the prevalence table in
  that module's own docstring is sourced from.

- [x] **5. `crosswalk.build_crosswalk_tier1` doesn't handle `http.get() → None`** —
  `sec/crosswalk.py:59-60`. `resp.text` raises `AttributeError` on any SEC hiccup. It is
  called lazily from `collect_fundamentals_us` when the crosswalk parquet is absent, so
  one transient failure takes down the whole batch before a single ticker is written.

## Potholes

- [x] **6. No backoff in the "retry-with-backoff" GET** — `sec/http.py:49-62`. Attempts fire
  0.12 s apart, so a 429 or 503 usually burns all 3 inside a quarter-second and returns
  None, silently dropping that filing (or, in `companyfacts`, that whole CIK). One
  `time.sleep(2 ** attempt)` before the retry.
- [ ] **7. `SEC_USER_AGENT` defaults to `contact@example.com`** (`config.py:33`) and isn't in
  `.env.example`. SEC's fair-access policy blocks placeholder UAs — the default is the one
  that ships.
- [ ] **8. `collect_fundamentals_us` has no resume** — `sec/fundamentals.py:139-167`. Every
  other collector here goes through `checkpoint.py` + `_merge_save`; this one refetches all
  ~1,250 tickers (companyfacts + every pre-2002 10-K + every 2001–2008 10-K) from scratch on
  rerun and overwrites each parquet outright.
- [x] **9. 8 of 120 sampled tickers have 100% NaN `net_revenue` in the xbrl tier** — measured
  for real 2026-07-29 across all 1,848 collected tickers (not just the 120 sample): 170
  (9.2%) affected, confirmed clustered in banks/thrifts/mortgage REITs/BDCs/GSEs (ABCB,
  AGNC, ARCC, AGM, GS, NLY, COLB and 163 more), verified via their raw companyfacts —
  they report `InterestIncomeExpenseNet` + `NoninterestIncome` instead of a single gross
  revenue tag. **Not fixed by adding a concept** — net interest income is already a spread,
  not comparable to industrial `Revenues`; mapping it in would silently corrupt every
  revenue-based ratio for this whole sector cluster. Documented as an intentional gap next
  to `CONCEPT_MAP`'s `net_revenue` entry, same treatment as the existing `total_debt`/banks
  acknowledgment.
- [ ] **10. Nothing US is wired into `pipeline.py`** — no `--market us`; `collect_macro_us`,
  `collect_prices_yf(price_dir=US_PRICES_DIR, suffix="", floor=...)` and
  `collect_fundamentals_us` are reachable only as module `__main__`s or manual calls.
  Current state reflects that: 2,055 price files vs 1,251 fundamentals vs 280 dividends.

## Checked, not a bug

- `cluster_period_ends` with `NaT` — `(NaT - ts).days` is `nan`, `nan <= 10` is False, so a
  NaT `end` becomes its own cluster instead of crashing.
- `end > fundamentals_available_date`: **0 rows** across the sample — the last-line-of-defense
  drop in `fundamentals.py` is holding.
- Duplicate `(ticker, end)`: **0 rows** — the cross-tier `_end_cluster` dedup is working.
- `fred_collectors.py`: clean. Full-series refetch + `_merge_save` dedup is genuinely
  idempotent; `"."` → NaN → drop is correct for FRED.

## 2026-07-30 follow-up: full data-quality sweep + one new real bug

Full sweep of everything on disk (5,446 price files, 2,289 fundamentals files, 14
macro series) rather than a sample, per a direct "is this data valid" request. Prices
and macro came back completely clean (0 `validate_prices` errors, 0 Inf, 0% macro NaN
across all 14 FRED series). Fundamentals surfaced one new real bug, plus several
things that measured as anomalies but checked out as not bugs.

- [x] **11. `item6._row_values` mis-parses footnote-reference markers as data,
  corrupting positional alignment — FIXED 2026-07-30.** A bare `"(3)"`-style cell
  referencing a table footnote (present in some year-columns of a row but not
  others) parses as a valid negative number under `_parse_value`, indistinguishable
  by shape from a real parenthesized negative. Left in, it doesn't just produce one
  wrong value — it inflates that row's token count past `n_years`, shifting EVERY
  later year's real value one position early. Confirmed on ORCL's actual 2006 10-K:
  "Total assets" read 2006 correctly (its own marker cell landed after the real
  value), then read 2005's marker cell as the 2005 value, corrupting 2005 through
  2002 too — the impossible `total_assets = -3,000,000` was only the visible
  symptom of a wrong ROW, not a wrong CELL. Also hit NEM (`total_assets = -2,000`,
  1998). Fixed in `_row_values`: strip marker-shaped tokens (`^\(\d{1,2}\)$`) only
  when doing so exactly resolves a token-count excess over `n_years` — a genuine
  small negative dollar figure in an already-aligned row is never touched. Verified
  against ORCL's real filing text end-to-end: all 5 years now correctly aligned and
  scaled (2006 total_assets = $29.029B, matching Oracle's real FY2006 figure).
  Regression tests in `test_sec_item6.py`.
- [x] **12. SEC fundamentals had never been through any write-time validation gate
  — FIXED 2026-07-30.** `collect_fundamentals_us` writes `df.to_parquet()` directly
  (`fundamentals.py:261`, pre-fix) — unlike every other collector in this codebase,
  which all route through `_merge_save()`'s validate-then-write. Added
  `validate.validate_us_fundamentals()` (warn-only, not block: `build_company_fundamentals`
  rebuilds a company's entire multi-decade history in one shot each run, so refusing
  the whole write over one bad historical row would cost far more good data than it
  protects — unlike BR's incremental-batch collectors, where blocking is safe).
  Wired into `collect_fundamentals_us`'s write loop; warnings logged before write.

### Checked, not a bug (2026-07-30)

- **item6 unit-scaling bug (audit item 1) still visible in on-disk data for
  INTC/IBM/GE-class tickers — this is DATA STALENESS, not a code regression.**
  Verified the actual committed fix (commit `dfc5b90`, 2026-07-29 09:24) end-to-end
  against INTC's real filing text: produces the correct $34.209B 2004 net_revenue.
  But `INTC.parquet`'s on-disk mtime is 2026-07-28 15:35 — collected BEFORE the fix
  and never touched since (`collect_fundamentals_us` has no per-ticker staleness
  tracking; `skip_existing` only skips re-fetching, it can't tell "stale" from
  "fresh"). The full-scale collection run is itself incomplete (5,446/10,432 price
  tickers, 2,289/10,432 fundamentals tickers — see fix plan's pothole #10 area) and
  appears to have stopped without error (no traceback in any log, just the last log
  line ending mid-batch at 2026-07-29 23:06). **Action, not a code fix:** a fresh
  `collect_fundamentals_us` run (default `skip_existing=False`, which rebuilds
  everything) is needed to propagate this and every other fundamentals-side fix to
  already-collected tickers. Not run as part of this audit — multi-hour, heavy SEC
  EDGAR traffic, needs an explicit go-ahead.
- **ICE/PJT/OGS carry ROE in the billions (e.g. ICE `roe` ≈ 1.53e9 for 2013-06-30)
  — genuine SEC-reported data, not a parsing bug.** Fetched ICE's raw companyfacts
  JSON directly: `StockholdersEquity` for CIK 1571949 (IntercontinentalExchange
  Group, Inc., the NEW merger holdco created for the NYSE Euronext acquisition) is
  literally tagged `val: 10` at 2013-06-30 — the only fact SEC's API exposes for
  that concept/period, no dimensional duplicate with a larger number hiding
  alongside it. Consistent with a newly-incorporated merger vehicle's real nominal
  pre-close capitalization, not a unit or concept-mapping error in our code. Same
  class of problem the codebase already has a policy for (CLAUDE.md: "denominators
  near zero are valid distress signals... kept intact") — left unclipped,
  consistent with that precedent, not flagged as a bug.
- **`adj_close` one-day moves >20x for 67 tickers (e.g. SRXH: 540937 → 270.47 over 3
  trading days in 2012)** — traced to `_fetch_and_shape_prices` in `yf_collectors.py`:
  `adj_close` comes straight from `yfinance`'s own `auto_adjust=True` history call,
  untouched by our split-reverse-adjustment code (which only rescales the
  unadjusted `open/high/low/close` columns). This is a vendor-side adjusted-close
  computation quirk on an obscure, multiply-reverse-split penny stock (SRXH did 6
  splits, several reverse), not a bug in our repair logic. Same class of issue
  CLAUDE.md already documents as "not fixable — flag or mask, never drop or
  reconstruct" for `adj_close`'s other vendor quirks. Not pursued further.
- **Fiscal-Q4 NaN rate still 4x elevated post-fix (18.9% Dec vs 4.8% other months,
  measured on the first 200 collected tickers) — expected, not a regression.**
  Down sharply from the pre-fix 58.7%/26.4% (audit item 2), but `_derive_q4` is
  deliberately conservative: it only derives a missing Q4 when EXACTLY 3 quarterly
  periods nest inside the fiscal year's own window, leaving it NaN rather than
  guessing otherwise (own docstring, `companyfacts.py`). The residual reflects
  companies whose quarterly history has its own gaps, where no safe derivation
  exists — working as designed.
