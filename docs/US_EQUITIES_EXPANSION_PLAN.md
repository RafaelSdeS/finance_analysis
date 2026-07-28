# US Equities Expansion — Full Plan

**Status:** Phases 1-3 done (real data collected, full historical crawl run — 43,366 CIKs,
1994-2026). Phase 4 core logic done + verified against real AAPL data (per-CIK path only;
bulk-zip scale-up pending). Phase 0/5-8 not started.
**Date:** 2026-07-28.
**Goal:** extend Stage 1 (data collection) + Stage 2 (dataset build) to US equities —
prices, fundamentals, macro — as far back as free sources reliably allow, minimizing
survivorship bias.

**Motivation (Stage 3+):** pretrain a model on US data (far larger cross-section), then
fine-tune on Brazil. Crypto dropped from scope on 2026-07-28; revisit separately if ever needed.

---

## 1. What was actually verified (2026-07-28)

Every claim below was measured by probing the live APIs, not taken from documentation.
Re-run these probes before trusting any number here that is older than a few months.

| Probe | Result |
|---|---|
| `sec.gov/files/company_tickers.json` | **10,432** current ticker↔CIK entries, keyless |
| `data.sec.gov` bulk `companyfacts.zip` | **1.39 GB**, rebuilt daily (last-mod same day) |
| `data.sec.gov` bulk `submissions.zip` | **1.55 GB**, rebuilt daily |
| DERA "Financial Statement Data Sets" | **69 quarterly ZIPs, 2009q1 → 2026q1** |
| EDGAR `full-index` earliest usable | **1994Q1** (20,889 filings, 1,420 10-K-ish). 1993Q1 is effectively empty (14 lines) |
| AAPL `companyfacts` shape | 503 `us-gaap` concepts; every fact carries `accn` + **`filed`** + `form` + `fy`/`fp` |
| AAPL `NetIncomeLoss` earliest period | period end **2007-09-29**, but `filed` 2009-10-27 (prior-year comparative inside the first XBRL 10-K) |
| **Restatement visibility** | AAPL FY2008 net income = **$4.834B as first filed (2009-10-27)**, restated to **$6.119B (filed 2010-10-27)** — a 26% revision, both versions retrievable |
| yfinance depth, old NYSE names | GE / KO / IBM / XOM / PG all start **1962-01-02** (16,249 rows). AAPL 1980-12-12, F 1972-06-01, T 1983-11-21 |
| **yfinance delisted tickers** | **TWTR, ATVI, SIVBQ, LEHMQ, ENRNQ, WCOEQ, SEARQ, GMGMQ, FTDCQ → 0 rows.** Yahoo purges delisted symbols |
| EDGAR retains dead companies? | **Yes** — Enron (351 filings, 1996→2005), Lehman, Twitter (→2022-12-02), Blockbuster all present with `formerNames`… |
| …but their tickers? | **`tickers=[]` and `exchanges=[]` for every delisted company.** SEC clears the field when filing stops |
| **EX-27 Financial Data Schedule** | Structured tag-value financials (~30 fields) inside pre-XBRL filings. Sampled 10-Ks: **1996Q1 8/8 present, 1999Q1 6/8, 2001Q1 2/8, 2002Q1 0/8** |
| FRED | **Works with no API key** via `fredgraph.csv?id=<series>`. `CPIAUCNS` returns **1913-01-01 → 2026-06-01**, 1,363 monthly rows |
| Stooq (delisted-price fallback) | **Bot-blocked** — returns a JS challenge page, not CSV. Not usable |
| Alpha Vantage `LISTING_STATUS` | demo key returns 0 rows; needs a real free key. **Untested — Phase 0 item** |
| Current BR dataset (for scale comparison) | 1,308,104 rows, 510 tickers, 2000-01-03 → 2026-07-14 |
| BR raw footprint | 1,199 price files / 104 MB; fundamentals 21 MB |

---

## 2. Coverage map — the real answer to "how far back?"

```
        1913 ─────────────────────────────────────────────────────────► 2026
MACRO   ████████████████████████████████████████████████████████████████  FRED, keyless
                    1962 ──────────────────────────────────────────────►
PRICES              ████████████████████████████████████████████████████  yfinance (survivors only)
                              1994 ─────► 2001  gap  2006/07 ──────────►
FUND.                         ███████████████░░░░░░░░████████████████████
                              EX-27 FDS      ▲       XBRL companyfacts
                                             └─ fillable via Item 6 chaining (§3.4)
```

- **Macro: 1913.** Better than expected — FRED needs no key at all, and CPI reaches 1913.
- **Prices: 1962.** That is Yahoo's floor for legacy NYSE names, and it is a hard floor —
  not a per-ticker quirk. Every old blue-chip tested returned exactly 16,249 rows starting
  1962-01-02.
- **Fundamentals: 1994, gap closable.** The important correction to the earlier draft, which
  claimed a flat "2009 floor." EX-27 pushes structured fundamentals back to **1994**; XBRL
  comparatives already reach **~2007**; and the remaining 2001–2006 window is fillable by
  chaining Item 6 tables (§3.4). Coverage is *continuous* 1994→present, but the line-item set
  is **narrower** outside the 2007+ XBRL tier — that is the real cost, not missing years.

### 2.0 The three fundamentals tiers, compared

| Tier | Years | Difficulty | Data richness | Main risk | Verified on |
|---|---|---|---|---|---|
| **EX-27 FDS** | 1994–2001 | **Easiest** — fixed tag-value parse | ~30 fields; no cash flow / EBITDA | `<MULTIPLIER>` mis-scaling (easy to detect) | 8 filings/yr × 4 yrs |
| **Item 6 chaining** | 2001–2006 | **Middle** — HTML table location + normalization | Narrowest: revenue, net income, EPS, assets, LT debt, dividends; equity sometimes | Coverage skew — smaller filers could abbreviate/omit Item 6 | **2 companies** (INTC, KO) |
| **XBRL companyfacts** | ~2007–present | **Biggest grind** — tag heterogeneity across 10k filers | Richest; full statements, point-in-time, restatements visible | A missed tag yields silent NaN, not an error | 15 large caps |

**The 2007+ tier loses nothing — it is richer than the current BR dataset.** Measured across
10 companies, every raw line item the BR fundamentals schema derives from is present:

```
10/10  net_revenue  net_income  equity  total_assets  current_assets
       current_liabilities  cash  total_debt  shares_outstanding  eps
 9/10  D&A (for EBITDA)   capex
 8/10  ebit   gross_profit          <- not missing data; these companies simply don't
                                       present that subtotal (typical of financials).
                                       Derive from components.
10/10  cashflow_ops   dividends     <- NOT in the BR schema at all; BolsAI never supplied
                                       a cash-flow statement
```

So the tiering above describes the cost of reaching *backwards*, not a ceiling on the modern
era. 2007+ supports the full BR-equivalent ratio set **plus** cash-flow-derived features that
the BR pipeline cannot currently compute, **plus** true as-first-reported point-in-time values
(§3.3). Deep history is thinner than recent history — the normal shape of financial data, and
already true of the BR dataset (2000–2007 is thinner there than 2015+).

> **Read the "verified on" column carefully.** These probes prove each mechanism *exists* and
> parses; they do **not** measure coverage across the universe. Every company sampled was a
> large cap. Both weak tiers (EX-27, Item 6) are most likely to degrade exactly where it was
> not tested — the small-filer long tail, which SEC scaled-disclosure rules permit to
> abbreviate or omit Item 6 entirely. Phase 0 exists to turn these existence proofs into
> coverage measurements before any of this is treated as fact.

### 2.1 The gap is smaller than it looks, and it is fillable

EX-27 was eliminated (SEC File No. S7-05-00, proposed 2000, effective 2001); XBRL did not
become mandatory until 2009. But two measurements shrink the gap substantially:

**(a) XBRL comparatives already reach ~2007, not 2009.** Sampling 15 large caps'
`companyfacts` for their earliest reported period:

| | earliest period end |
|---|---|
| AAPL 2006-09-30 · INTC 2006-12-30 · GE 2006-12-31 · SBUX 2006-10-01 · ADBE 2006-12-01 | **2006** |
| MSFT · KO · XOM · PG · CAT · MMM · HD · NKE | **2007** |
| JNJ · WMT | **2008** |

Median earliest period end = **2007**. Those come free with the XBRL tier — no extra work,
just don't discard facts whose period predates the filing. (They must still be stamped with
their 2009/2010 `filed` date — see §5.2.)

**(b) The remaining ~2001–2006 window is genuinely extractable.** Verified, not assumed:

- Filings in this era are **HTML with inline financial statements**, not 1990s ASCII, and no
  longer rely on `EX-13` incorporation by reference (measured: KO/INTC 10-Ks from 2003/2005/2008
  all have `inline_IS=True`; the `EX-13` exhibit disappears after 2003).
- `pandas.read_html` parses them directly — no custom HTML parser needed. Intel's 2005 10-K
  yielded exactly **one** income-statement-like table out of 155, with clean values
  (`Net revenue 34,209`, `Cost of sales 14,463`, `Gross margin 19,746`, `R&D 4,778`) across
  three fiscal years, units declared in the header ("In Millions—Except Per Share Amounts").

See §3.4 for the recommended extraction strategy, which avoids full-statement parsing entirely.

**What is *not* worth attempting:** the pre-2001 ASCII era's by-reference exhibits. Verified on
KO's 1995 10-K405 (347 KB ASCII, statements incorporated into `EX-13.1`). That era is already
covered by EX-27 anyway (§3.1), so there is nothing to gain.

---

## 3. Problem A — going back further than 2009 (SOLVED to 1994)

### 3.1 EX-27 Financial Data Schedules (1994–2001)

Machine-readable, fixed-tag, one per filing. Verified structure (Coca-Cola FY1994):

```
<ARTICLE> 5
<MULTIPLIER> 1,000,000
<PERIOD-TYPE>    YEAR      <FISCAL-YEAR-END>  DEC-31-1994
<CASH> 1,386          <SECURITIES> 145        <RECEIVABLES> 1,470
<INVENTORY> 1,047     <CURRENT-ASSETS> 5,205  <PP&E> 6,157
<TOTAL-ASSETS> 13,873 <CURRENT-LIABILITIES> 6,177
<COMMON> 427          <OTHER-SE> 4,808        <TOTAL-LIABILITY-AND-EQUITY> 13,873
<SALES> 16,172        <TOTAL-REVENUES> 16,172 <CGS> 6,167
<INTEREST-EXPENSE> 199 <INCOME-PRETAX> 3,728  <INCOME-TAX> 1,174
<INCOME-CONTINUING> 2,554 ... <NET-INCOME> <EPS-PRIMARY> <EPS-DILUTED>
```

That is enough to derive the core ratio set the BR dataset already carries: `pl`, `pvp`,
`roe`, `roa`, `net_margin`, `gross_margin`, `debt_equity`, `current_ratio`,
`asset_turnover`, `lpa`, `vpa`, `earnings_yield`, `book_to_market`.

**Two gotchas, both must be handled:**

- **`<MULTIPLIER>` varies per filing** (1, 1,000, 1,000,000). Never assume. This is the same
  class of bug as the existing `K = 1000` BolsAI-thousands convention in `yf_collectors.py`,
  and the macro-unit mismatch already burned once (`PIPELINE_FORENSIC_AUDIT` Issue 1).
- **`<ARTICLE>` selects a different tag schema per industry**: Art. 5 = commercial/industrial,
  Art. 7 = insurance, Art. 9 = banks, Art. 6 = registered investment companies. A bank's FDS
  has no `<INVENTORY>`/`<CGS>`. Parse per-article; do not force one schema.

**Not in EX-27:** cash-flow statement, EBITDA, segment data, shares outstanding (sometimes).
So 1994–2001 will have a *narrower* feature set than 2009+. That is fine and expected — flag
it with a coverage indicator column rather than silently emitting NaN.

### 3.2 XBRL (2009–present, with 2007–2008 comparatives for free)

Because every fact carries its own `start`/`end` period *and* the `filed` date of the filing
that reported it, the first XBRL filings (2009) contain 1–2 years of **prior-period
comparatives**. AAPL's earliest `NetIncomeLoss` period ends 2007-09-29. So 2007–2008 arrives
free — but those rows are *not point-in-time* (they were first published in 2009), and must be
stamped with `filed`, never with the period end. See §5.2.

### 3.3 Bonus: this fixes a known BR limitation, for the US side

`CLAUDE.md` documents an unfixable-with-BolsAI issue: fundamentals *values* may be restated
even though `fundamentals_available_date` is point-in-time (as-reported vs as-restated
lookahead). **EDGAR does not have this problem.** Every fact carries the accession and filing
date of the specific filing that reported it, so taking the **earliest `filed` per
(concept, start, end)** yields true as-first-reported figures.

Measured proof: AAPL FY2008 net income was $4.834B as first filed, and $6.119B a year later
(retrospective revenue-recognition adoption). A naive "latest value wins" build would show a
2008 row carrying a number nobody had until October 2010.

**Design rule: the US fundamentals table stores as-first-reported values, keyed by `filed`
date.** Optionally also store the latest restated value in a parallel column
(`*_restated`) so the restatement gap becomes measurable — which would, as a side benefit,
finally let us *quantify* how bad the equivalent BR problem is.

### 3.4 Filling 2001–2006: chain Item 6 "Selected Financial Data" (recommended)

The obvious approach — parse each year's full income statement and balance sheet — is the
expensive one: table *selection* is the hard part, not table *parsing*. Measured on 2005
10-Ks: Intel produced exactly 1 income-statement-like table out of 155 (unambiguous), but
Coca-Cola produced 6, and the top-scoring one was not the income statement at all.

**A much cheaper route exists.** Until the SEC eliminated it in 2021, **Item 6 "Selected
Financial Data"** required a standardized **5-year** summary table in every 10-K. One table,
~10 well-known line items, five years at a time. Verified on Intel's 2010 10-K — a single
table yielding 2005–2009:

```
(In Millions, Except Per Share Amounts)   2009    2008    2007    2006    2005
Net revenue                             35,127  37,586  38,334  35,382  38,826
Gross margin                            19,561  20,844  19,904  18,218  23,049
Research and development                 5,653   5,722   5,755   5,873   5,145
Operating income                         5,711   8,954   8,216   5,652  12,090
Net income                               4,369   5,292   6,976   5,044   8,664
Earnings per share (Basic)                0.79    0.93    1.20    0.87    1.42
```

Coca-Cola's 2010 Item 6 likewise spans 2005–2009 (and its 2005 filing carries 5- and 10-year
compound growth columns on top).

**Why this solves the gap cleanly:**

1. **Chaining covers everything.** A 2006-filed 10-K's Item 6 covers 2001–2005; a 2010-filed
   one covers 2005–2009. Two filings per company span the entire hole.
2. **Overlap gives free cross-validation.** Consecutive filings' Item 6 tables overlap by four
   years, so every extracted number can be checked against three or four independent
   extractions. A parser bug or a misread column shows up immediately as a disagreement —
   this is a much stronger correctness signal than anything available for the EX-27 or XBRL
   tiers.
3. **Same point-in-time rule applies.** Item 6 in a 2010 filing shows *restated* 2005 figures.
   Prefer the **earliest filing that reports a given fiscal year** — identical `min(filed)`
   rule as §3.3, so no new concept is introduced.
4. **Table selection is far easier** than for a full income statement: look for a table whose
   header carries ≥4 distinct fiscal years and whose first column mentions total assets or
   net income. That heuristic found the right table on the first try for both test companies.

**Limits — state them, don't paper over them:**

- Item 6 carries a *narrower* line-item set than a full statement. Typically revenue, net
  income, EPS, total assets, long-term debt, dividends per share; shareowners' equity is
  common but **not universal**. So `pl`, `roa`, `net_margin`, `asset_turnover` and
  `earnings_yield` are derivable; `pvp`/`roe`/`current_ratio`/`debt_equity` only where equity
  and current items happen to be reported.
- No cash-flow statement, no EBITDA, no segment data.
- Smaller filers sometimes omitted or abbreviated Item 6 (scaled disclosure). Coverage must be
  **measured per year**, not assumed.

Net effect: the gap era gets the same *core* ratio coverage as the 1994–2001 EX-27 tier —
narrower than 2009+, but continuous and cross-validated. Full-statement parsing stays
available as an optional deepening pass later, only for companies where Item 6 came up short.

---

## 4. Problem B — survivorship bias (PARTIALLY solvable free; be honest about the rest)

### 4.1 What is and isn't achievable

| Layer | Free? | Verdict |
|---|---|---|
| **Who existed and when** (point-in-time universe roster, incl. dead companies) | ✅ Yes | Fully solvable from EDGAR `full-index`, 1994→present |
| **Fundamentals for dead companies** | ✅ Yes | Fully solvable — EDGAR keeps Enron's 351 filings, Lehman's, Twitter's |
| **CIK → ticker for dead companies** | ⚠️ Partial | SEC clears `tickers[]` on delisting; must recover from filing cover pages |
| **Prices for dead companies** | ❌ **No** | yfinance purges them (verified: TWTR/ATVI/LEHMQ/etc. all 0 rows); Stooq bot-blocked |

So: the universe roster and the fundamentals can be made survivorship-bias-free. **Prices
cannot, with free sources.** Anyone claiming otherwise for free data is wrong.

### 4.2 The design: quantify the bias even where it can't be removed

This is strictly better than the current BR situation, where survivorship bias is
acknowledged but never measured.

1. **Build a point-in-time universe roster** (`us_universe_roster.parquet`) from EDGAR
   `full-index`, ~130 quarterly files, 1994→present. A CIK counts as publicly reporting in
   quarter *Q* if it filed a 10-K/10-Q within a trailing window. This roster **includes every
   company that later died** — it is constructed from what was filed at the time.
2. **Join actual price coverage against that roster per year.** The result is a direct,
   publishable number: *"in 1999 the roster held N companies; we have prices for M of them;
   survivorship coverage = M/N."* Emit it as a build artifact, not a footnote.
3. **Use it as a feature-set gate.** Any Stage 3+ backtest must read this coverage table and
   either restrict to periods where coverage is adequate, or explicitly report the bias.
4. **Fundamentals-only analyses can be run bias-free** on 2009+ (and 1994–2001 via EX-27),
   because the dead companies' filings are all there.

### 4.3 Recovering tickers for dead companies

Needed to join EDGAR fundamentals to any price series at all. Ordered by reliability:

1. `company_tickers.json` — 10,432 current names. Free, exact, but survivors only.
2. **XBRL `dei:TradingSymbol`** on filing cover pages, 2009+ — present in the filing even
   after delisting, since filings are immutable. Recovers tickers for anything that died
   2009→present (Twitter, ATVI, SIVB, FRC…).
3. **Pre-2009 cover-page text** — the ticker usually appears near "Name of each exchange on
   which registered". Regex-extractable at moderate reliability; validate a sample by hand.
4. `submissions.zip` `formerNames` — catches renames (the US analogue of this repo's existing
   `apply_ticker_continuity()` splicing for BR), which is separately essential: a ticker
   change is *not* a delisting and must not be counted as one.

### 4.4 If price survivorship must actually be solved

It requires paid data. Options, cheapest first — **not** part of this plan, listed so the
decision is informed:

- **Sharadar SEP + ACTIONS** (Nasdaq Data Link) — survivorship-bias-free US EOD, ~$50/mo tier.
- **Norgate Data** — survivorship-bias-free, ~$US few hundred/yr.
- **CRSP via WRDS** — the academic gold standard; free *if* you have a university affiliation.
  Worth checking whether you do — it would be the single highest-value unlock here.
- **EODHD** — has delisted coverage, mid-priced.

---

## 5. Architecture

Reuse-first. The repo already solved most of these problems for Brazil; the US work is
mostly *pointing existing machinery at new sources*, plus one genuinely new module family.

### 5.1 Module layout

```
src/data_collection/
  config.py                  # + US_TICKERS, SEC_UA, FRED_SERIES, DATA_SOURCE entries  [DONE: FRED_SERIES, US_MACRO_DIR, US_PRICES_DIR, US_PROTOTYPE_TICKERS]
  yf_collectors.py           # REUSE for US prices (see 5.3) — [DONE: 3 optional params added, see Phase 2]
  fred_collectors.py         # NEW — tiny; mirrors the BCB macro collector          [DONE, 2026-07-28]
  sec/                       # NEW — mirrors the existing cvm/ package, same role
    http.py                  #   shared throttled GET + bulk-zip download (10 req/s cap)
    universe.py              #   full-index -> point-in-time roster (§4.2)
    crosswalk.py             #   CIK <-> ticker, incl. dead-company recovery (§4.3)
    companyfacts.py          #   XBRL facts -> as-first-reported tidy table (2009+)
    fds.py                   #   EX-27 Financial Data Schedule parser (1994-2001)
    ratios.py                #   raw line items -> the BR-compatible ratio schema
```

`sec/` deliberately mirrors `cvm/` — same shape of problem (free government bulk source,
real filing dates, zip downloads, versioned filings), so the same package structure applies.
`cvm/http.py` already holds "the one shared zip-download/retry implementation"; `sec/http.py`
is its sibling, not a refactor of it (different host, different rate-limit contract).

### 5.2 Point-in-time correctness (the thing that must not be got wrong)

The BR pipeline's central invariant is `merge_asof(..., direction='backward')` on a real
availability date. The US side must produce the identical contract:

| BR | US equivalent |
|---|---|
| `reference_date` (fiscal period end) | XBRL fact `end` / EX-27 `<PERIOD-END>` |
| `fundamentals_available_date` (CVM `DT_RECEB`) | XBRL fact **`filed`** / EX-27 filing's index date |

Rules:
- **Never** key a row on the period end. Key on `filed`.
- Where a period is reported by several filings, take **`min(filed)`** for the value
  (as-first-reported), per §3.3.
- The 2007–2008 comparative rows must carry their 2009 `filed` date. They are legitimately
  *unavailable* before then and `merge_asof` will correctly refuse to show them earlier.

This means the whole existing Stage 2 no-lookahead machinery works unchanged — no new
lookahead surface is introduced, provided the collector stamps `filed` correctly.

### 5.3 Prices — reuse, don't rewrite

`yf_collectors.py` is already source-generic: `_yf_symbol()` just applies
`config.TICKER_ALIASES` + `config.YF_SUFFIX`, and US tickers need **no suffix** (`""`).
Everything BR-specific in that file is inapplicable and simply won't trigger:

- `_bolsai_junction_date()` / `_reconcile_yfinance_junction()` — return `None`/no-op when
  there is no BolsAI-sourced row on disk (US data is pure-yfinance from row one). Confirmed
  by reading the functions: both short-circuit when `num_trades` is never non-NaN.
- `_prices_fetch_start()` — will refetch the full span every run, which is exactly right and
  is already the intended behaviour for a yfinance-only series.
- `_repair_nonpositive_ohlc()` — still wanted; it is a generic Yahoo glitch, not a BR one.

So: **no new price collector.** The work is a config entry and a universe list.

### 5.4 Storage budget

BR is 1,199 price files / 104 MB. US is ~10,432 tickers with much deeper history (old NYSE
names carry 16,249 rows vs BR's ~967-row average). Rough estimate: **1.5–3 GB** of price
parquet, plus **~3 GB** of EDGAR bulk zips (transient — parse and discard, keep only the
derived tables). Budget ~10 GB working space. Everything under `data/raw/us/` stays
gitignored; unlike the BR raw data, this is too big to git-track.

### 5.5 Keeping US and BR separate

Do **not** merge US rows into `ml_dataset.parquet`. Different currency, calendar, accounting
regime, macro block. Produce `data/processed/us_ml_dataset.parquet` with a schema deliberately
*aligned* to the BR one (same column names where the concept matches) so a Stage 3+ pretrain →
fine-tune handoff is a straight column-subset operation, not a translation layer.

---

## 6. Phased plan

Each phase ends green before the next starts, per the standing rule in this repo.

### Phase 0 — de-risk assumptions (½ day, no production code)

- [ ] Confirm EX-27 prevalence properly: sample **200+** 10-Ks across 1994–2001 (not 8) and
      report % with a parseable EX-27, broken down by `<ARTICLE>`.
- [ ] Parse 20 EX-27s across all four article types; confirm the field set and that
      `<MULTIPLIER>` handling is right.
- [ ] Get a free Alpha Vantage key and test `LISTING_STATUS&state=delisted` — does it return
      a delisted roster with dates, and does it serve *prices* for delisted symbols? If yes,
      §4.1's "prices ❌" downgrades to "partial ✅" and Phase 3 changes materially.
- [ ] Check whether you have any university/WRDS affiliation (CRSP access would change §4.4).
- [ ] Confirm yfinance mass-fetch behaviour at scale: rate limits / throttling on ~500
      sequential tickers, to size Phase 3 runtime.

**Gate:** if EX-27 prevalence is below ~50%, drop the 1994–2001 tier and revert to a 2009 floor.

### Phase 1 — macro (FRED) — ✅ DONE 2026-07-28

- [x] `fred_collectors.py`, keyless `fredgraph.csv?id=` path (no API key needed — verified).
- [x] Series: `FEDFUNDS`, `DGS2`/`DGS10`/`DGS30`, `CPIAUCSL`+`CPIAUCNS`, `PPIACO`, `UNRATE`,
      `GDPC1`, `INDPRO`, `T10Y2Y`, `VIXCLS`, `DTWEXBGS`, `M2SL`.
- [x] Written to `data/raw/us/macro/{series}.parquet`, one file per series — same layout as
      `data/raw/macro/`.
- [x] Frequency + unit documented in `config.py` next to `FRED_SERIES`, same convention as
      `BCB_SERIES`.

**Gate met:** all 13 series collected. `ppi` (PPIACO) actually reaches **1913-01-01** and
`industrial_production` (INDPRO) reaches **1919-01-01** — both deeper than the CPI figure used
to size expectations. Full result:

| series | rows | span |
|---|---|---|
| fed_funds | 864 | 1954-07 → 2026-06 |
| treasury_2y | 12,533 | 1976-06 → 2026-07 |
| treasury_10y | 16,125 | 1962-01 → 2026-07 |
| treasury_30y | 12,355 | 1977-02 → 2026-07 |
| cpi_sa | 953 | 1947-01 → 2026-06 |
| cpi_nsa | 1,361 | **1913-01** → 2026-06 |
| ppi | 1,362 | **1913-01** → 2026-06 |
| unemployment | 941 | 1948-01 → 2026-06 |
| real_gdp | 317 | 1947-01 → 2026-01 |
| industrial_production | 1,290 | **1919-01** → 2026-06 |
| term_spread_10y2y | 12,534 | 1976-06 → 2026-07 |
| vix | 9,238 | 1990-01 → 2026-07 |
| dollar_index | 5,154 | 2006-01 → 2026-07 |
| m2 | 809 | 1959-01 → 2026-05 |

Self-check: `tests/data_collection/test_fred_collectors.py` (FRED's `.`-for-missing marker,
mocked — no network in the test).

### Phase 2 — US prices, prototype — ✅ DONE 2026-07-28

- [x] `US_PROTOTYPE_TICKERS` in `config.py` (AAPL, GE, KO, IBM, XOM, PG — all verified deep).
- [x] `collect_prices_yf` reused, **not unchanged as originally scoped** — see correction below.
- [x] Validated against `validate.py`'s existing price gates.

**Gate met:** GE/KO/IBM/XOM/PG each landed 16,249 rows starting 1962-01-02; AAPL 11,496 rows
from 1980-12-12. Zero validation failures after the fix below.

**Correction to §5.3's "no new price collector" claim:** true in spirit, but not quite
"as-is" — three module-level globals were hardcoded inside functions that only took `path` as
a parameter, which would have collided US and BR data on a shared reuse:

1. `config.PRICES_DIR` — US prices would have landed in the *same directory* as BR prices.
   Fixed: `collect_prices_yf()` gained an optional `price_dir` param (defaults to
   `config.PRICES_DIR`, so BR callers are unaffected).
2. `config.YF_SUFFIX` (`.SA`) — would have requested `AAPL.SA` instead of `AAPL`. Fixed:
   `_yf_symbol()`/`_fetch_and_shape_prices()`/`collect_prices_yf()` gained an optional
   `suffix` param.
3. `config.START_DATE` (BR's 2000-01-01 floor) — the first prototype run silently capped at
   2000-01-01 instead of reaching 1962, since there was no existing file/checkpoint to infer
   an earlier start from. Fixed: `_prices_fetch_start()`/`collect_prices_yf()` gained an
   optional `floor` param; US calls pass `"1900-01-01"` so yfinance returns whatever it
   actually has.

All three are additive, default-preserving optional parameters — zero behavior change for
existing BR call sites (confirmed: full `fast` test suite green, 34/34, before and after).

**Unplanned bug found and fixed while running this phase:** fetching mid-session returns a
live, still-forming "today" bar that can be internally inconsistent — measured on XOM,
fetched 2026-07-28 during market hours: `low` (154.32) printed *above* `open` (154.17) by
15¢, because open/close print immediately while high/low keep updating from a
differently-lagged feed. That single bad row failed `validate_prices` for the **entire**
multi-decade batch via `_merge_save`'s all-or-nothing validation — silently discarding
thousands of otherwise-good historical rows, for any ticker collected during market hours.
Not BR/US-specific; latent in the shared `yf_collectors.py` path since `_prices_fetch_start`'s
full-span-refetch design was added. Fixed by extracting a `_drop_incomplete_today()` helper
(new self-check in `_demo()`) that drops any row dated today before it ever reaches
validation — safe because the next run's full-span refetch picks up the finalized close once
the session ends.

### Phase 3 — universe + survivorship instrumentation — CODE DONE, full crawl PENDING (2026-07-28)

- [x] `sec/http.py`: throttled GET (10 req/s cap), shared by every `sec/` module.
- [x] `sec/universe.py`: `parse_master_idx()` (pure, unit-tested), `fetch_quarter()`
      (per-quarter cache, current quarter always refetched), `build_filings()`,
      `build_roster()`, `compute_coverage()`.
- [x] `sec/crosswalk.py`: **tier-1 only** (`build_crosswalk_tier1()`, current listings via
      `company_tickers.json`) — the dead-company recovery ladder (§4.3 tiers 2-4) is not yet
      implemented; `compute_coverage()`'s docstring states this makes its number a **lower
      bound**, not final.
- [x] Self-checks: `tests/data_collection/test_sec_universe.py` (parser + coverage logic, no
      network) — registered in `run_all.py --group fast`.
- [x] **Spot-verified against real EDGAR data**, 8 quarters spanning each company's known
      collapse (not the full 130-quarter history yet — see below):

  | Company (CIK) | Last seen filing | What actually happened |
  |---|---|---|
  | Lehman Brothers Holdings (806085) | 10-Q, 2008-07-10 (2008Q3) | Filed Ch. 11 Sept 2008 — absent from 2009Q1 sample. Correct. |
  | WorldCom/MCI (723527) | 10-Q, 2001-11-14 (2001Q4) | Accounting scandal broke Jun 2002, Ch. 11 Jul 2002 — absent from 2002Q3 sample. Correct. |
  | Twitter, Inc. (1418091) | 10-K, 2014-03-06 (2014Q1) | Taken private by Musk Oct 2022 — absent from 2022Q4/2023Q1 samples. Correct. |
  | Enron Corp (1024401) | 10-Q, 2001-11-19 (2001Q4) | Filed Ch. 11 Dec 2001. Present right up to the collapse. Correct. |

  This is the actual mechanism working, not a claim: each company drops out of the roster in
  the exact quarter its real-world collapse says it should, with no manual curation.
- [x] Tier-1 crosswalk fetched for real: 10,432 current CIK↔ticker pairs (AAPL→320193 confirmed).

**Full historical crawl — ✅ RUN 2026-07-28** (in your terminal, per your instruction for
long/heavy scripts):

```
131/131 quarters (1994Q1 -> 2026Q3) -- 1,072,744 qualifying 10-K/10-Q filings
284,404 (cik, year) roster rows -- 43,366 distinct CIKs ever actively reporting
```

**Gate met on the complete dataset**, not just the 8-quarter spot-check:

| Company (CIK) | Active years (full roster) | Real history |
|---|---|---|
| Lehman Brothers Holdings (806085) | 1994 → 2008 | Collapsed Sept 2008. Correct. |
| Enron Corp (1024401) | 1996 → 2001 | Ch. 11 Dec 2001. Correct. |
| WorldCom/MCI (723527) | 1995 → 2006 | **Not a simple story** — Ch. 11 Jul 2002, but the SAME CIK resumed normal 10-K/10-Q filing after reorganizing as "MCI Inc," continuing until Verizon's Jan 2006 acquisition. The roster correctly captures the full bankruptcy-reorganization-then-acquisition lifecycle, not just "died in 2002." |
| Twitter, Inc. (1418091) | 2014 → 2022 | IPO Nov 2013 (first 10-K, fiscal 2013, filed 2014); taken private by Musk Oct 2022. Correct. |

**Two findings worth flagging, now that real numbers exist:**

1. **The tier-1 crosswalk resolves a ticker for only 5,414 of the 43,366 distinct roster
   CIKs (12.5%).** The other 87.5% are exactly the dead-company-ticker-recovery problem
   §4.3 describes — this is that gap, measured for the first time rather than estimated.
   Tiers 2-4 (not yet built) are what would close it.
2. **Active filer count nearly halved since its 1998-99 peak:** ~11,300 CIKs/year in
   1997-1999 (and again ~11,700 in 2008) down to 6,749 in 2026. This independently
   reproduces a well-documented phenomenon in the finance literature (fewer US public
   companies today than 25 years ago, sometimes called "the listing gap") — a good sanity
   check that the roster reflects real economic history, not a parsing artifact.

**Coverage table** (current state — 6 priced tickers vs. thousands-per-year roster):

```
year  roster_ciks  priced_ciks  coverage
1994         2351            5   0.21%
2008        11713            5   0.04%
2026         6749            5   0.07%
```

This ~0.05% figure is **expected and not a defect** — it reflects that only Phase 2's 6
prototype tickers are priced so far, not a flaw in the roster or crosswalk. It becomes the
real, meaningful survivorship-bias number once Phase 6 collects the full priced universe;
tracking it from here forward (rather than only computing it once at the end) means any
future regression in coverage is visible immediately rather than discovered late.

### Phase 4 — fundamentals, XBRL tier 2009+ — CORE LOGIC DONE 2026-07-28 (per-CIK path only)

- [x] `sec/companyfacts.py`: `fetch_companyfacts(cik)` → tidy `(end, val, filed, form, accn)`
      per concept (`_facts_to_frame`).
- [x] As-first-reported selection: `min(filed)` per `(start, end)` (`as_first_reported`).
      `*_restated` companion column **not yet added** (straightforward extension, deferred).
- [x] Ratio computation **reused, not reimplemented**: `yf_collectors._compute_ratios` was
      promoted to a public `compute_ratios(r, unit_scale=K)` (default `K=1000` preserves BR
      behavior exactly) — SEC's XBRL figures are already full-dollar (not thousands), so
      `sec/companyfacts.compute_us_ratios()` calls it with `unit_scale=1`. Same formulas
      already verified at 5% tolerance against live BolsAI data, now reused for US ratios too.
- [ ] `sec/http.py`'s bulk-zip path (`companyfacts.zip`, 1.4GB once vs 10k+ per-CIK calls) —
      **not built**; current implementation is per-CIK only (`fetch_companyfacts(cik)`), fine
      for prototyping a handful of companies, but scaling to the full universe (Phase 6) needs
      the bulk-zip path instead of one HTTP request per CIK.

**Gate met:** AAPL FY2008 net income comes out as **$4.834B** (as-first-reported), verified
directly against live data — `as_first_reported(facts, "NetIncomeLoss")` on real AAPL
companyfacts returns exactly `4834000000`, filed `2009-10-27`, not the `6119000000` restatement
filed a year later. Self-checked without network in `test_sec_companyfacts.py`.

**Two real bugs found and fixed while verifying against live AAPL data (not caught by design
alone — both needed actual data to surface):**

1. **Duration collision:** XBRL tags the same fiscal `end` with quarterly, half-year,
   9-month, AND annual durations simultaneously (a 10-Q's current-quarter + YTD comparatives;
   a 10-K's full year) — confirmed 96 of AAPL's `NetIncomeLoss` periods share an `end` with a
   different-duration sibling. Merging line items on `end` alone collided multiple duration
   variants into one row. Fixed: `_quarterly_only()` restricts duration concepts to ~60-100
   day periods before dedup; instant (balance-sheet) concepts are unaffected (no `start`, no
   ambiguity).
2. **Global-not-per-period concept resolution:** the original fallback logic picked the first
   non-empty concept **for the whole company** and used it for all periods. AAPL's own revenue
   tag moved `SalesRevenueNet` (2008-2018, 40 periods) → `"Revenues"` (2016-2018 transition
   label, only 8 periods) → `RevenueFromContractWithCustomerExcludingAssessedTax` (2017-2026,
   29 periods). Because `"Revenues"` happened to be non-empty, the original code used *only*
   its 8 periods, silently discarding 2008-2016 and most of 2019-2026. Fixed: `_resolve_item()`
   unions every concept's coverage **per period**, with fallback-list order breaking ties only
   where two concepts genuinely report the same `end` (transition-year overlap). Revenue
   coverage went from 8 periods to 66 (2008-01 → 2026-03) after the fix.

Both bugs would have been invisible from synthetic test data alone — they only showed up
against AAPL's real, messy, multi-era tagging history. Worth remembering for Phase 5/7: the
EX-27 and Item-6 tiers deserve the same real-data verification before trusting them, not just
unit tests against hand-built fixtures.

### Phase 5 — fundamentals, EX-27 tier 1994–2001 (3–5 days, gated on Phase 0)

- [ ] `sec/fds.py`: per-`<ARTICLE>` parsers, `<MULTIPLIER>` scaling, unit self-checks.
- [ ] Stamp `filed` from the filing index date.
- [ ] Emit a coverage flag column so the narrower 1994–2001 feature set is explicit.

**Gate:** a hand-checked sample (KO 1995 among them: total assets 13,873 × 1e6 = $13.873B)
reconciles to the published figures.

### Phase 6 — full-universe scale-up + Stage 2 build (1 week)

- [ ] Scale prices to the full ticker list; expect long runtimes and partial failures —
      reuse the existing checkpoint/resume machinery, don't invent new.
- [ ] `us_ml_dataset.parquet` via the Stage 2 pipeline, schema-aligned to BR (§5.5).
- [ ] Port the existing test suite's invariants (no-lookahead, prefix-NaN, split repair).

### Phase 7 — fundamentals, gap tier 2001–2006 via Item 6 chaining (4–6 days)

Closes the last hole (§3.4). Deliberately sequenced last: it is the only tier whose output can
be validated against the two tiers either side of it, so building it after Phases 4–5 means the
overlap years (2001 vs EX-27, 2006–2007 vs XBRL) become free correctness tests.

- [ ] `sec/item6.py`: locate the Item 6 table by heuristic (header carries ≥4 distinct fiscal
      years; first column mentions total assets or net income), parse with `pandas.read_html`.
- [ ] Normalize the known layout noise: NaN spacer columns, `$`/sign in separate columns,
      parenthesized negatives, footnote superscripts, units from the table header.
- [ ] Chain filings per CIK and keep the **earliest** report of each fiscal year (§3.3 rule).
- [ ] **Cross-validation harness** — the main deliverable, not an afterthought: assert
      overlapping extractions of the same (cik, fiscal_year) agree within tolerance, and
      report the disagreement rate. A high rate means the parser is wrong, not the data.
- [ ] Reconcile the boundary years against the neighbouring tiers: 2001 vs EX-27, 2006–2007
      vs XBRL. Any systematic offset here is a units or column-alignment bug.
- [ ] Emit per-year coverage; flag the narrower line-item set explicitly.

**Gate:** overlap disagreement rate < 2%, and boundary-year reconciliation against the EX-27
and XBRL tiers shows no systematic offset.

### Phase 8 — *deferred* — full-statement parsing

Only for companies where Phase 7's Item 6 came up short (omitted, abbreviated, or missing
equity), and only if the missing ratios prove to matter. Table *selection* is the hard part
(Intel: 1 candidate of 155; KO: 6, top-ranked one wrong). Do not start speculatively.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| XBRL tag heterogeneity silently drops revenue/income for many filers | **High** | Per-concept ordered fallback lists; assert non-null coverage % per year; fail the build if coverage regresses |
| EX-27 `<MULTIPLIER>` mis-scaling → 1e6-off figures | **High** | Explicit per-filing multiplier parse + magnitude sanity assertion vs a known-good sample |
| yfinance throttling / IP-blocking on ~10k sequential fetches | Medium | Existing checkpoint/resume + backoff; Phase 0 measures real throughput first |
| Price survivorship bias silently inflates any backtest | **High** | Cannot be removed free — §4.2 makes it *measured* and forces downstream consumers to read the coverage table |
| Ticker reuse (a dead ticker reassigned to a new company) corrupts series | Medium | Key everything on **CIK**, never on ticker; ticker is a display join only |
| 2007–2008 comparative rows treated as available before 2009 | **High** | Stamp `filed`, never period end (§5.2); assert min(`filed`) ≥ 2009 for XBRL-tier rows |
| Item 6 column misalignment silently shifts a year (2005 value read as 2004) | **High** | The overlap cross-validation harness (Phase 7) catches exactly this — a one-year shift shows as systematic disagreement across all overlapping filings |
| Item 6 omitted/abbreviated by smaller filers → gap-era coverage skews to large caps | Medium | Measure coverage per year *and* per size bucket; if it skews badly, the gap tier is large-cap-only and must be labelled as such |
| Three fundamentals tiers with different line-item sets create a discontinuity the model reads as signal | **High** | Emit an explicit `fundamentals_tier` column (`ex27` / `item6` / `xbrl`) so any tier-boundary artefact is attributable, not mysterious; never silently NaN-fill across tiers |
| Scope creep into a US modeling project before data is solid | Medium | This plan stops at Stage 2 output. No Stage 3 work until `us_ml_dataset.parquet` passes its tests |

---

## 8. Decisions still open

1. **Alpha Vantage / CRSP check (Phase 0).** If either yields delisted prices, §4 changes
   substantially — worth resolving before Phase 3.
2. **Paid price data.** If a genuinely survivorship-bias-free US backtest is a hard
   requirement rather than a nice-to-have, Sharadar (~$50/mo) is the cheapest credible route
   and should be decided now, not after Phase 6.
3. **Universe breadth.** All ~10,432 tickers, or a liquidity/market-cap screen? Full breadth
   pulls in thousands of microcaps and OTC shells whose data quality is poor (the BR pipeline
   already needed a quarantine list for exactly this). Suggest: collect broadly, filter at
   Stage 2, mirroring the existing BR approach.
