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
