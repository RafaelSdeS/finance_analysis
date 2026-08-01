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
checkbox was just stale. **Follow-up the same day:** the bug-11 fix turned out incomplete —
re-collecting the stale tickers it named surfaced 3 more real bugs (13-15, all fixed) in the
same subsystem; see the second dated section below. `sec/item6.py` renamed to
`sec/selected_financial_data.py` in the same pass (content + bugfixes, not just the name —
see `CLAUDE.md`'s `sec/` module table for the rationale).

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
  Regression tests in `test_sec_selected_financial_data.py` (renamed from
  `test_sec_item6.py` the same day — see the follow-up section below).
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

## 2026-07-30 follow-up #2: the bug-11 fix was incomplete — 3 more real bugs

Per the "action, not a code fix" note above, a fresh `collect_fundamentals_us` run was
made to propagate bug 11's footnote-marker fix onto the 4 known-stale tickers (NEM,
ORCL, plus BOOM/ZION spotted in the same negative-`total_assets` flag list). NEM/ORCL
came back clean as expected — but BOOM and ZION still showed impossible negative
`total_assets` in the *freshly regenerated* output, proving they weren't stale leftovers
at all. Root-caused both live against their actual EDGAR filings, turning up two more,
distinct bugs in the same table-selection/extraction path, plus a third, unrelated bug
in `companyfacts.py` found the same day by the same full-sweep methodology.

- [x] **13. `find_item6_table` ranked keyword score above year coverage, picking a
  business-segment fragment over the real Item 6 table — FIXED 2026-07-30.** Confirmed
  on ZION's actual 2005 10-K: Item 6 there is incorporated by reference (no real table
  in the parsed document at all), so the old `(score, len(years), rows)` sort key let a
  3-year business-segment condensed income statement — which happens to spell out
  "Total assets"/"Net income (loss)"/"Total revenue" verbatim (score 3) — outrank the
  actual company-wide 5-year table, whose equivalent row is labeled just "Assets" under
  an "AT YEAR-END" header (score 2, since "Assets" alone doesn't match "TOTAL ASSETS").
  A genuine Item 6 table's defining trait is covering more of the requested history, not
  how many keywords its labels spell out. Fixed by reordering the sort key to
  `(len(years), score, rows)` — year count decides first.
- [x] **14. Unit-caption detection scanned the whole filing instead of the winning
  table's own caption, silently rescaling ZION 1000x too small — FIXED 2026-07-30.**
  Once bug 13's fix picked the *correct* table, its figures were still wrong: the
  winning table's own caption said "(Amounts in millions)", but `detect_unit_multiplier`
  scanned the entire filing text and picked up "thousands" instead — the dominant
  caption belonging to the much larger main financial statements elsewhere in the same
  combined submission. Fixed to prefer the winning table's own caption text (searched
  within the table's own flattened cells) and fall back to the whole-document scan only
  when the winning table states no units of its own.
- [x] **15. Colspan-duplicated HTML cells defeated the bug-11 footnote-marker guard,
  corrupting BOOM's `total_assets` row — FIXED 2026-07-30.** Confirmed on BOOM (Dynamic
  Materials) actual 2005 10-K: its "Total assets" row round-trips through
  `pandas.read_html` with every year's value duplicated into two adjacent columns (a
  colspan-to-columns rendering artifact specific to that row), stacked with one genuine
  footnote-marker cell. Stripping only the marker left 10 tokens for `n_years=5` — not
  an exact match — so bug 11's fix (which only strips markers when doing so exactly
  resolves a token-count excess) silently gave up and fell through to the first 5 raw,
  still-duplicated tokens, corrupting every year one position off. Fixed by collapsing
  adjacent equal-value token pairs first (only once a row is already known to have too
  many tokens, same conservative trigger as the footnote-marker check), recovering the
  true per-year values before the footnote marker is even considered.
- [x] **16. `companyfacts.as_first_reported` had no floor on implausibly ancient `end`
  dates, unlike item6's equivalent guard — FIXED 2026-07-30.** Unrelated to 13-15,
  found by the same full-sweep audit: NG, CLSK and TENX each carry a genuine XBRL fact
  with an `end` date decades before the company plausibly existed (NG:
  `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`,
  end=1984-12-04; TENX: `CashAndCashEquivalentsAtCarryingValue`, end=1967-05-25/08-25).
  Checked directly against SEC's own companyfacts API: every one carries `val=0`, a
  filer-side XBRL-tooling placeholder artifact, not something this repo produces.
  `item6.build_cik_history` already has an equivalent last-line-of-defense year bound
  (`_FISCAL_YEAR_MIN`/`_MAX`, bug 2 above) for its own version of this failure shape,
  but instant (balance-sheet) concepts have no `start`/duration for
  `as_first_reported`'s existing filters to catch this at all. Fixed by adding
  `_MIN_PLAUSIBLE_END = 1995-01-01` (the earliest era any tier in this pipeline claims
  fundamentals data from) and dropping any fact with `end` before it.
  `tests/data_collection/test_us_data_quality.py` now also rate-checks the xbrl tier for
  this pattern (`_MIN_PLAUSIBLE_XBRL_END`), tolerating the known-stale pre-fix count
  while still catching a new regression.

**Verified 2026-07-30:** full re-collection of both prices (9,593/10,432 tickers) and
fundamentals (8,142/10,432 tickers) with all of 11-16 in place. Final sweep:
`test_us_data_quality.py` reports 0 `validate_prices` errors / 0 Inf across all price
files, 0 gate errors / 0 Inf / 0 implausible xbrl end dates across all fundamentals
files, and exactly one remaining known-stale negative-`total_assets` file (RMSL, within
the test's rate ceiling) — down from the NEM/ORCL/BOOM/ZION set this section started
from. All 14 macro series still clean.

## 2026-07-30 follow-up #3: cross-checked item6 figures against real published financials — 2 more confirmed bugs, 1 open

A user request to verify dataset figures against real-world published numbers (not just
internal consistency) surfaced that the item6 tier's dollar figures, not just its dates
(bugs 11-16), can be silently wrong. Recent-quarter xbrl-tier data for AAPL/MSFT/AMZN/
GOOGL/META/TSLA/NVDA was cross-checked against known real results and matched almost
exactly (e.g. MSFT Q1 FY25 revenue $65.585B, Apple's EU-tax-charge quarter net income
$14.736B, Tesla's Q1 2025 slump to $409M net income) — the xbrl tier is trustworthy. The
item6 tier was not: a dataset-wide scan comparing every item6 row's `net_revenue` against
the nearest xbrl/ex27 row for the same ticker found 119 rows across 1,806 item6-bearing
tickers off by ≥10x, split into two directions with two different root causes.

- [x] **17. `detect_unit_multiplier`'s tie-break was hash-seed-dependent, not
  deterministic — FIXED 2026-07-30.** Confirmed on AAPL's actual 2005 10-K: the Item 6
  table states its governing caption once, "(In millions, except share and per share
  amounts)", then separately captions its shares-outstanding sub-row "(in thousands)" —
  a 1-vs-1 tie under the old mode-based `max(set(hits), key=hits.count)` selection, whose
  outcome depends on `set()`'s iteration order (Python's string hash randomization is on
  by default, so this could even flip between separate process runs). Confirmed live:
  AAPL's FY2001-2005 net_revenue stored 1000x too small (FY2001 $5,363,000 instead of the
  real $5,363,000,000); Home Depot's FY1994-1999 showed the identical exact-1000x pattern
  across 6 consecutive years. The same row also exposed a second bug: AAPL's FY2001 net
  LOSS of $(25) million rendered as two separate HTML cells, "(25" and ")" — `_parse_value`
  only recognizes a negative when both parens are in the same cell, so it silently stored
  a $25M *profit*. Fixed with `detect_unit_multiplier(text, prefer_first=True)` (takes the
  first units mention, not the mode, when scanning a single winning table's own text — the
  table's governing caption is always stated before any per-row exception) and a
  paren-merge step in `_row_values` (joins an unclosed "(NNN" cell with an immediately
  following ")" cell before parsing). Verified end-to-end against AAPL's real filings via
  `build_cik_history` directly: FY2001 net_revenue $5.363B, net_income **-$25M** (correct
  sign), total_assets $6.021B — exact match to the filed 10-K text.
- [x] **18. A share-count row's own local "(in thousands)" caption could be read as the
  whole table's governing caption — FIXED 2026-07-30.** Found re-collecting TXN
  post-fix-17: its real Item 6 is incorporated by reference (no table in the primary 10-K
  document at all, same shape as ZION/bug 13), so `build_cik_history` fell through to the
  full combined submission `.txt` and found TI's real annual-report table there. That
  table states NO table-wide dollar caption in its own cells (the real "(in millions)"
  caption lives in a preceding paragraph outside the parsed table) — its only units
  mention was "Average common and dilutive potential common shares outstanding ... in
  thousands", which fix 17's `prefer_first=True` then wrongly applied to net_revenue too,
  understating TXN's real FY2005 $13.392B revenue as $13.392M. Fixed by excluding any row
  whose label mentions "shares" from caption detection entirely (never from value
  extraction — `extract_years`'s own per-share exemption is untouched), so a table shaped
  like this correctly falls through to the whole-document scan instead (which correctly
  found "millions" as TI's dominant caption). Verified against TXN's real filing:
  FY2005 net_revenue now $13.392B, matching the filed figure exactly.
- [ ] **Open, NOT fixed: item6 has more than these 2 root causes.** Re-scanning
  dataset-wide after 17+18 and re-collecting the 30 tickers the original 119-row scan
  flagged: the "too small" bucket dropped from 54 to 38 rows, i.e. most of that list
  (AMD, ASYS, AZO, BBWI, CMI, EGAN, GAP [partially — 1994-97 fixed, 1999-2001 still
  wrong], GEF, GRC, HVT, KTCC, M, MBOT, PAYX, PKOH, RS, SHW, THO, TWAV, WSM, XOM) is
  **still wrong**, each for its own filing-specific reason, not 17 or 18. Two confirmed
  live: **PAYX** — its real 2002-filed table (correctly captioned "in thousands", not a
  caption bug at all) has label text duplicated across two columns with inconsistent
  blanks depending on row-indentation depth (sub-items like "Total revenues" carry their
  label only in column 1 with column 0 NaN; section headers carry it in both) — the
  extracted net_revenue (35,600) doesn't match ANY value in the real table at all,
  meaning `extract_years`'s row-label matching picked the wrong row entirely, a parsing
  bug distinct from both units-caption bugs above. **XOM** — shows the familiar
  ~1000x-too-small pattern for 1999-2002 (from one filing) but ALSO a garbage
  near-zero net_revenue for FY2003 and missing FY2004-2005 entirely (from later
  filings) — at least two more distinct failures in the same ticker's chained history,
  not diagnosed further. **Also still separately open (found the same day, different
  root cause, NOT an item6 bug):** the "too big" bucket (65 rows, mostly WMT/TXT/SWK/
  ZBRA/GAP-1994-97/ANF/APH/BIO/FLEX/EL and more) — item6 is actually CORRECT in these
  cases (verified WMT: item6 FY1999 revenue $137.634B matches Walmart's real reported
  figure exactly); it's the **ex27 tier's own comparator value that's wrong** for these
  tickers (e.g. WMT's ex27 row shows $139,208 — not even plausible at any thousands/
  millions/billions scale for a company that size), a separate bug in `fds.py`'s
  EX-27 `<MULTIPLIER>` handling, not investigated at all yet. **Net assessment:** the
  item6 (and possibly ex27) tiers have accumulated many *different*, filing-specific
  extraction bugs, not one remaining shared cause — continuing to fix them one ticker at
  a time has sharply diminishing returns per bug found. A more systematic approach (flag
  every item6/ex27 row whose value deviates implausibly from a neighboring tier for the
  same ticker, and either quarantine or manually review those specific rows, rather than
  trying to make the HTML table parser handle every real-world formatting variant) is
  likely more effective than continuing this per-ticker chase. Not started.

## 2026-07-30 follow-up #3: raw `Close` corrupted at the source for extreme multi-reverse-split penny stocks — confirmed NOT a units/decimal error

Separate from the `adj_close` one-day-move finding above (follow-up #1) and from
item6/ex27's own bugs (follow-up #2's numbered list) — this is yfinance's
`auto_adjust=False` **raw** `Close`, corrupted before any of our code touches it,
for tickers with an unusually large number of cumulative reverse splits.

- [x] **19. Implausible raw `Close` values (billions to quadrillions per share) —
  guarded, not fixable.** Confirmed on 9 tickers (ADTX, MRDN, XTIA, NXPL, JAGX,
  TOPS, PPCB, NUWE, BINI — later joined by more of the same shape as collection
  continued): 60-90% of each ticker's own row count shows a `Close` in the
  billions-to-quadrillions range (ADTX max $3.71e12/share, BINI max
  $3.00e17/share). A `$10M/share` sanity ceiling (`_MAX_PLAUSIBLE_PRICE` in
  `yf_collectors.py`, `_fetch_and_shape_prices`) now skips the whole ticker
  cleanly with a clear log message instead of letting it cascade into a
  confusing `validate_prices` bracket-violation failure.
  - **User asked directly: could this just be a misplaced decimal point (e.g.
    "$3.7 trillion" is really "$3.7")?** Checked properly rather than assumed.
    Answer: **no.** Traced ADTX's raw `Close` at the boundary of each of its 7
    real reverse splits (2022-09-14 through 2026-05-18): the value doesn't jump
    by a fixed factor at each split boundary the way a genuine reverse split
    would (nor does it match the recorded split ratio) — instead it just keeps
    getting **larger the further back in time you go**, roughly tracking how
    many of the 7 splits *hadn't happened yet* at that date:
    | date | raw Close |
    |---|---|
    | 2026-07-30 (today) | $0.003 — a genuinely plausible price for this ticker |
    | 2026-05-18 (last split) | $1.7 |
    | 2025-11-03 | $1,440 |
    | 2024-10-02 | $27M |
    | 2023-08-18 | $3.9B |
    | 2022-09-14 (first split) | $180B |
    | 2020-06-30 (oldest row) | $2.47 trillion |
    Today's price is genuinely correct (matches what a heavily-diluted,
    multiply-reverse-split penny stock actually trades at). A fixed unit/decimal
    error would distort every date by the *same* factor; this instead compounds
    — each additional reverse split further back in time multiplies the
    distortion again. That's the signature of yfinance's own split-adjustment
    computation compounding incorrectly for tickers with an unusually high
    reverse-split count, not a scale/currency mistake on any one field. There is
    no single divisor that recovers real history across the whole series (each
    era would need a different, split-count-dependent correction), and no
    independent source to derive the right per-era factor from — genuinely
    unrecoverable, same as `adj_close`'s other documented vendor quirks.
    Not pursued further; the guard is the correct, final handling.

## 2026-07-31 follow-up: the "too big" ex27 bucket root-caused and fixed — 866-ticker recollection measured

Follow-up #3's "too big" bucket (65 rows, WMT/TXT/SWK/ZBRA/GAP/ANF/APH/BIO/FLEX/EL
and more) was marked "not investigated at all yet." Investigated and fixed.

- [x] **20. EX-27 filings that omit `<MULTIPLIER>` understated figures by up to
  10⁶× — FIXED 2026-07-31.** `<MULTIPLIER>` is genuinely OPTIONAL per SEC's EX-27
  schema. Confirmed on WMT's real filings: 1995/1996 10-Ks tag it explicitly
  (1,000,000), but 1997-2000 omit the tag entirely — not malformed, simply
  absent — even though the raw figures are still reported at the same implicit
  millions scale (1997's real `TOTAL-ASSETS`=39,604 is Walmart's actual ~$39.6B,
  not $39,604). `extract_line_items` silently defaulted the absent tag to 1.0.
  Fixed by tracking whether `<MULTIPLIER>` was explicitly present
  (`fds_multiplier_explicit`) and, when absent, borrowing the multiplier from a
  sibling EX-27 exhibit of the SAME CIK that does declare one — a company's own
  scale convention is consistent across its own filings even when one year's
  exhibit omits the declaring tag (confirmed fixes WMT and SWK). `compute_ratios`'
  dollar/dollar ratios are scale-invariant, so only the raw dollar fields needed
  correcting; ratios are recomputed from the corrected values.
  **Measured after recollecting all 866 currently-collected ex27-bearing
  tickers:** 130 tickers had at least one row genuinely fixed this way; 202
  tickers have at least one row that's STILL unresolved because there's no
  sibling exhibit ANYWHERE in their own EX-27 history to borrow a real multiplier
  from (confirmed on TXT: every single collected exhibit omits the tag) — these
  stay honestly flagged (`fds_multiplier_explicit=False` AND `fds_multiplier==1.0`)
  rather than guessed at with zero evidence, same "flag, don't fabricate"
  precedent as every other genuinely-unrecoverable case in this doc. The
  remaining 534 tickers had no missing-multiplier issue at all.
- **New, smaller findings surfaced by the same recollection, not investigated
  further (out of scope for this fix, each affecting exactly 1-2 tickers):**
  BELFA/BELFB (literally the same company's two share classes, same CIK) both
  show `total_assets`=-7,600 for `end`=2001-12-31 in the item6 tier — same
  scattered-per-filing category as follow-up #3's "too small" bucket (PAYX/XOM),
  not the ex27 bug this section fixes. OXY shows `shares_outstanding`=-891,624,558
  for `end`=2016-03-31 in the xbrl tier — a different subsystem again, genuinely
  new, not root-caused.

## 2026-07-31: xbrl-tier `shares_outstanding` inflated ~800x–1,000,000x for isolated
## filings — FOUND (real, full-scale US Stage 2 build) + FIXED

- [x] **21. `companyfacts.py`'s xbrl-tier `shares_outstanding` silently accepted a
  fact under ANY unit key, and had no cross-period plausibility check at all —
  FIXED 2026-07-31.** Not found by an audit pass — found in the real, first
  full-scale `us_ml_dataset.parquet` build (2,960 tickers / 15.4M rows), via a
  `market_cap` std (4.98e14) wildly out of line with its own p99 (3.53e11).
  Traced to raw `shares_outstanding`: confirmed on **at least 27 real tickers**
  (AA, AEP, BTI, CB, CBRE, CBT, CCI, CCL, CG, CNA, CNI, CNNE, CNX, DCH, EOG, HII,
  HL, IVR, MAR, MSCI, PCG, TEX, TFC, UPWK, WRB, YUM, plus BCH/BSAC — see below)
  — one or more isolated quarterly filings inflated by ~800x to ~1,000,000x
  relative to that SAME ticker's own other filings. E.g. BTI's `shares_outstanding`
  is ~2.456 billion every fiscal year except FY2019 (filed 2020-03-26), which
  reads 2,456,520,738,000,000 (~2.46 quadrillion). Corrupts `market_cap` and
  every ratio derived from it (`pl`, `pvp`, `p_sr`, `p_assets`, `p_ebit`,
  `ev_ebit`, `book_to_market`, `earnings_yield`, `peg_ratio`,
  `pvp_to_roe_ratio`, their `*_zhist_5y` variants) for every daily row where
  the bad filing is the "current" one via `merge_asof` — and risks distorting
  `pl_zscore_sector`/`pvp_zscore_sector` for OTHER, unaffected tickers in the
  same sector on the same date too (a single absurd outlier inflates the
  sector's std for that date). One ticker (**LTM**) is the worst case: not an
  isolated quarter — wrong for every filing from 2022-12-31 through the most
  recent 2025-12-31 (3+ consecutive years), never self-correcting.
  **Root cause investigated, not fully resolved**: SEC's raw XBRL fact JSON
  carries no `decimals`/`scale` field (unlike EX-27's declared `<MULTIPLIER>`,
  bug #20 above — not directly reusable, XBRL has no equivalent), so `val` is
  contractually supposed to already be final-scale; whether the bad values
  originate from genuine filer-side XBRL tagging errors or from picking up an
  atypical filing's context (S-1/424B/10-K/A, `form` is captured but was never
  filtered on) remains genuinely uncertain. Two independent, defensive fixes
  landed instead of chasing the exact upstream cause:
  1. `_facts_to_frame` now only accepts a fact under the unit key actually
     expected for shares-denominated concepts (`"shares"` — verified against
     the real live SEC API, not assumed) instead of iterating any unit key
     with no check at all. Scoped to `_SHARES_UNIT_CONCEPTS`
     (`CommonStockSharesOutstanding`/`EntityCommonStockSharesOutstanding`)
     only — deliberately NOT extended to enforce `"USD"` for every other
     concept, since whether ifrs-full foreign filers' dollar facts are
     uniformly tagged `"USD"` is unverified and their current "any unit key"
     behavior is unchanged/working.
  2. New `_reject_sequential_outliers(df, col)`: walks a resolved per-period
     series chronologically and NaNs out any value whose ratio to the LAST
     ACCEPTED value (never a rejected one) exceeds 20x either direction, or
     that's non-positive — with a companion `{col}_rejected_outlier` boolean
     flag (never a guessed/reconstructed value, this repo's own "flag, don't
     fabricate" convention, same as bug #20 above and `loaders.load_dividends`'s
     implausible-`value_per_share` drop). Comparing only against the
     immediately PRECEDING raw value would not have been enough — LTM's 4
     consecutive bad years would only have caught the first transition, then
     treated each subsequent bad year as "consistent" with the previous
     (already bad) one; never re-anchoring on a rejected value fixes this.
     20x is well above any real stock split (rarely exceeds ~10-20x) and well
     below every one of the 27 measured corruption ratios (800x+). Generic
     over `_ATTACHED_ITEMS` (today: just `shares_outstanding`), not hardcoded
     — applies automatically to any future member of that set.
  Wired into `extract_line_items`'s attached-items loop, right before the
  nearest-match attach onto the real period grid. `validate.py`'s
  `validate_us_fundamentals` now also warns (not blocks) when a row has a
  rejected value, for collection-log visibility.
  **Deliberately out of scope**: applying the same sequential-outlier guard to
  flow/dollar concepts (`net_income`, `equity`, `total_assets`, `net_revenue`,
  ...) — those legitimately have far higher period-over-period volatility for
  smaller/cyclical/growth companies and would need a fundamentally different
  plausibility model (relative to a slower-moving anchor, not a fixed ratio
  threshold); a follow-up opportunity, not attempted here. The same
  "resolve a candidate concept, take `val` verbatim, no validation" extraction
  path is structurally shared by all 13 `CONCEPT_MAP` items, so the same bug
  CLASS is plausible for other concepts too — not confirmed/measured for any
  of them.
  **BCH/BSAC (Chilean bank ADRs) are a separate, unresolved, lower-confidence
  flag, NOT fixed by the guard above**: the exact same `shares_outstanding`
  value repeats verbatim across up to 5 consecutive fiscal years (e.g. BCH:
  101,017,081,114 for 2018/2020/2021/2022/2023/2024, with 2019 reading exactly
  0). Not a magnitude bug — Chilean banks genuinely have huge nominal local
  share counts — so `_reject_sequential_outliers` correctly does NOT flag this
  (values aren't wildly different, just suspiciously identical). Could be
  legitimate (a static float, genuinely unchanged) or a stale-value collection
  bug; `fundamentals.py`'s tier-combiner has no `ffill`/caching touching this
  column (ruled out as the cause there). Left uninvestigated — flagged here so
  it isn't rediscovered as a mystery, not bundled into this fix.
  **Regression tests**: `tests/data_collection/test_sec_companyfacts.py` — unit-key
  rejection, isolated-bad-quarter rejection (BTI-shaped), persistent-multi-year
  rejection without re-anchoring (LTM-shaped, the one that would catch a
  regression to "compare only to the immediately preceding raw value"), a
  plausible-large-jump NOT being rejected (a real-split-shaped fixture), and an
  end-to-end wiring check via `extract_line_items`. Existing
  `test_shares_outstanding_does_not_fragment_periods` updated to use
  `unit="shares"` (its old default `unit="USD"` fixture would now be rejected
  by fix #1 above — this was itself confirmation the fix works). Full fast
  suite green, `ruff check` clean.
  **Not yet done: recollection.** No raw XBRL JSON is cached anywhere in this
  pipeline (`fetch_companyfacts` re-fetches from SEC live every run) — fixing
  the code does NOT retroactively fix `data/raw/us/fundamentals/*.parquet`.
  Every affected ticker needs a live re-fetch, then `us_ml_dataset.parquet`
  needs a full Stage 2 rebuild. See the commands at the end of this session's
  summary / project memory.

- [x] **21b. `_reject_sequential_outliers`'s forward-only walk had its own real
  bug (a naive fix for #21 above would have shipped this) — FOUND (live
  recollection, 2026-08-01) + FIXED same day.** The user ran the targeted
  recollection for the 26 confirmed tickers + BCH/BSAC. Two of them (**CCI**,
  **TFC**) came back with 67/81 and 68/85 rows rejected respectively — not the
  1-3 rows expected, ~80% of their ENTIRE history. Root cause: CCI's and TFC's
  own FIRST-EVER XBRL-era `shares_outstanding` values (2008-12-31/2009-06-30,
  when this tier begins for them) are themselves the corrupted ones (confirmed
  on the real recollected data: CCI reads 288,464,431,000 / 290,792,627,000 for
  those two dates, then ~67 genuinely correct ~2.9e8-scale quarters follow). A
  forward-only walk anchors its `last_good` on that first (bad) value and then
  rejects every good quarter that follows — the exact "known, accepted
  limitation" #21's implementation had already named and documented, just more
  consequential in practice than estimated (a 2-row bug became a 67/68-row
  wipeout, real data loss on two large, real companies' entire modern
  history — Crown Castle and Truist Financial).
  **Fix:** `_reject_sequential_outliers` no longer walks strictly forward from
  index 0. It first clusters all valid values by rounded `log10` magnitude,
  picks the MAJORITY cluster as the seed (ties broken by whichever cluster's
  earliest member is chronologically first — LTM's real 4-good/4-bad tie,
  verified to still resolve correctly, unchanged behavior), then walks forward
  from the seed AND separately backward from the seed (two independent
  last-accepted trackers). This correctly handles all three real shapes now
  confirmed in this dataset: an isolated bad quarter surrounded by good ones
  (BTI/YUM), a persistent bad run following good history (LTM, forward walk
  from a seed in the good cluster), and a persistent bad run AT THE START of
  history followed by good data (CCI/TFC, backward walk from a seed in the
  good cluster now correctly flags the early bad values instead of the good
  majority).
  **Regression test**: `test_reject_sequential_outliers_does_not_anchor_on_a_bad_first_value`
  (CCI-shaped fixture, using the real measured values) — fails against the
  original forward-only implementation, confirming it's a genuine regression
  guard. All prior tests (including the LTM tie case) still pass unchanged.
  Full fast suite green, `ruff check` clean.
  **Recollection must be re-run**: the data currently on disk for the 28
  targeted tickers reflects the BUGGY first pass (CCI/TFC's ~67/68 good rows
  wrongly NaN'd) — re-running the same targeted recollection command
  (idempotent full overwrite, cheap, seconds) with the fixed code corrects it.
