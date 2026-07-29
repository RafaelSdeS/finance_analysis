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
