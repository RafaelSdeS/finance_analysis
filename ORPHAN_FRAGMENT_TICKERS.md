# Orphan Price-Fragment Tickers — flagged, not yet fixed

Written 2026-07-15, companion to `TOP50_INDEPENDENT_AUDIT.md`. Scope: a
dataset-wide scan for the "recycled B3 ticker code" pattern confirmed on
`BRDT3` and (with strong evidence, not yet coded) `CYRE4` — a raw price
file's earliest rows are stale data from an unrelated, long-dead earlier
holder of the same ticker symbol, followed by a multi-year silence, followed
by the real current listing's dense modern trading. Not real history of the
current entity; produces a nonsensical fake return the day trading resumes
(now caught by `MAX_RETURN_GAP_DAYS` in `features.py`, but the garbage rows
themselves are still in the raw data and still pollute `ma_20/60`, `rsi_14`,
and anything based on price *levels* rather than `log_return`).

## What's already fixed

- `BRDT3` (→`VBBR3`): dropped, see `quality_filters.ORPHAN_PREFIX_TICKERS`.
- `CCRO3`→`MOTV3`: a related but distinct bug (dead-ticker-stub *inside* a
  splice boundary, not an orphan prefix) — fixed via `old_last_date` in
  `ticker_continuity.json`.
- `UGPA3`: NOT this pattern — confirmed a genuine, currently-unfillable
  BolsAI/yfinance collection gap in one continuous real listing (see
  `yf_collectors.FLAT_RUN_PADDING`). Its 2010-2011 gap is now caught by the
  general `MAX_RETURN_GAP_DAYS` guard, not row-dropping.

## Methodology (reproduce with this)

```python
import pandas as pd
from pathlib import Path

GAP_YEARS_DAYS = 730  # 2 years -- above the confirmed-legitimate 47-53 day
                       # illiquid-microcap gaps, below UGPA3's confirmed-real 499-day gap

for f in sorted(Path("data/raw/prices").glob("*.parquet")):
    df = pd.read_parquet(f, columns=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    gap = df["trade_date"].diff().dt.days
    splits = gap[gap > GAP_YEARS_DAYS]
    # if len(splits): candidate ticker -- see columns below
```

## What "fixing" one of these means

Per-ticker, like the `BRDT3` and `UGPA3` investigations: cross-check
`company_info.parquet` (does a `cvm_code`/`cnpj` exist? does the corporate
name match a company plausibly trading this early?), `fundamentals` (do real
filings exist for the orphan-span dates?), and ideally an external source
(as was done here for `UGPA3`/`CYRE4`) confirming the entity's actual listing
date. If confirmed orphaned: add an entry to
`quality_filters.ORPHAN_PREFIX_TICKERS`. If confirmed a real gap in one
continuous entity: no code change needed, `MAX_RETURN_GAP_DAYS` already
covers it — just note it here as investigated.

**Do not bulk-apply row-dropping to this whole list** — `orphan_pct` shows
several tickers (e.g. `GPCP3` 92%, `RNEW11` 94%, `IGBR3` 99.6%) where the
"orphan" fragment is nearly the entire file; for those in particular, verify
before assuming ticker-code reuse rather than a real (if extreme) trading
halt.

## Priority notes

- **`CYRE4`**: already verified in the 2026-07-15 audit conversation (not in
  `company_info.parquet` at all; per external research the security didn't
  exist before Jan 2026, created via a Dec-2025 Cyrela share-class
  reorganization). Ready to fix the same way as `BRDT3` — just needs the
  `ORPHAN_PREFIX_TICKERS` entry added.
- **`LREN3`**: a **top-50 universe member** (rank 14) — its gap fragment
  resolves well before the 2011-04-01 training window (resumes 2005-04-01),
  so it doesn't currently affect top-50 training, but re-verify if the
  training start date ever moves earlier.
- Everything else below: outside the top-50 universe, unverified, lower
  priority.

## Full candidate list (117 tickers, excludes already-fixed `BRDT3`)

| ticker | orphan_rows | total_rows | orphan_pct | orphan_span_end | resumes | biggest_gap_days |
|---|---|---|---|---|---|---|
| CYRE4 | 257 | 385 | 66.8% | 2005-06-24 | 2026-01-02 | 7497 |
| SNSY3 | 5 | 684 | 0.7% | 2005-03-17 | 2020-12-11 | 5748 |
| NAFG3 | 2 | 42 | 4.8% | 2000-08-17 | 2015-08-19 | 5480 |
| SNSY5 | 206 | 1451 | 14.2% | 2005-12-19 | 2020-12-11 | 5471 |
| BPAR3 | 4 | 53 | 7.5% | 2019-02-21 | 2026-05-04 | 5076 |
| AFLU5 | 8 | 10 | 80.0% | 2008-10-08 | 2017-11-16 | 3326 |
| INEP4 | 3627 | 4533 | 80.0% | 2014-08-29 | 2022-11-18 | 3003 |
| BSLI3 | 1 | 1111 | 0.1% | 2001-07-03 | 2009-01-26 | 2764 |
| GPCP3 | 1666 | 1803 | 92.4% | 2013-05-16 | 2020-11-13 | 2738 |
| NUTR3 | 319 | 644 | 49.5% | 2017-09-29 | 2025-03-20 | 2729 |
| LUXM3 | 9 | 13 | 69.2% | 2013-01-02 | 2015-04-30 | 2691 |
| EUCA4 | 5 | 4124 | 0.1% | 2002-07-03 | 2009-11-13 | 2690 |
| TEND3 | 570 | 2850 | 20.0% | 2010-02-08 | 2017-05-04 | 2642 |
| CALI3 | 24 | 230 | 10.4% | 2018-11-05 | 2022-06-13 | 2563 |
| MWET3 | 326 | 361 | 90.3% | 2016-01-29 | 2022-11-14 | 2481 |
| INEP3 | 3 | 4460 | 0.1% | 2000-08-23 | 2007-05-17 | 2458 |
| FIGE3 | 20 | 101 | 19.8% | 2022-08-25 | 2025-11-27 | 2365 |
| AMCE3 | 2 | 71 | 2.8% | 2002-03-06 | 2008-03-13 | 2199 |
| BALM3 | 57 | 577 | 9.9% | 2011-08-01 | 2017-07-11 | 2171 |
| MEND3 | 1 | 2 | 50.0% | 2009-09-25 | 2015-05-27 | 2070 |
| SHUL3 | 6 | 9 | 66.7% | 2016-02-04 | 2021-06-18 | 1961 |
| RNEW3 | 650 | 997 | 65.2% | 2019-10-16 | 2025-02-14 | 1948 |
| RNEW11 | 2254 | 2404 | 93.8% | 2019-10-16 | 2025-02-14 | 1948 |
| PCAR3 | 39 | 1679 | 2.3% | 2010-03-08 | 2015-06-30 | 1940 |
| ILMD3 | 10 | 18 | 55.6% | 2005-09-21 | 2010-12-13 | 1909 |
| IGBR3 | 2493 | 2502 | 99.6% | 2018-04-30 | 2023-05-23 | 1849 |
| SJOS3 | 2 | 63 | 3.2% | 2002-05-10 | 2007-05-03 | 1819 |
| BAHI4 | 32 | 499 | 6.4% | 2002-03-12 | 2006-11-23 | 1717 |
| EMAE3 | 14 | 27 | 51.9% | 2019-08-28 | 2024-03-28 | 1674 |
| VAGV3 | 25 | 31 | 80.6% | 2005-04-27 | 2009-10-01 | 1618 |
| HETA3 | 1 | 106 | 0.9% | 2001-03-06 | 2005-08-04 | 1612 |
| BSLI4 | 3 | 1241 | 0.2% | 2004-09-24 | 2009-01-26 | 1585 |
| FTRX3 | 1 | 94 | 1.1% | 2005-03-11 | 2009-07-03 | 1575 |
| CORR4 | 94 | 112 | 83.9% | 2011-09-14 | 2015-12-08 | 1546 |
| VAGV4 | 1281 | 1357 | 94.4% | 2005-06-17 | 2009-09-10 | 1546 |
| CELP6 | 18 | 83 | 21.7% | 2015-04-24 | 2018-10-15 | 1523 |
| AHEB3 | 52 | 457 | 11.4% | 2013-05-27 | 2015-07-29 | 1504 |
| CNFB3 | 4 | 52 | 7.7% | 2001-11-29 | 2005-12-27 | 1489 |
| DOHL3 | 5 | 299 | 1.7% | 2005-08-08 | 2008-11-10 | 1455 |
| VULC3 | 1 | 3724 | 0.0% | 2000-11-20 | 2004-10-25 | 1435 |
| FIGE4 | 16 | 25 | 64.0% | 2006-02-02 | 2010-01-05 | 1433 |
| LARK4 | 23 | 24 | 95.8% | 2007-09-17 | 2010-02-12 | 1413 |
| BRGE7 | 70 | 79 | 88.6% | 2019-06-04 | 2023-04-12 | 1408 |
| MTSA3 | 7 | 62 | 11.3% | 2011-05-30 | 2015-04-06 | 1407 |
| REDE4 | 967 | 1116 | 86.6% | 2012-11-16 | 2016-09-05 | 1389 |
| LEVE3 | 6 | 3824 | 0.2% | 2007-09-19 | 2011-01-12 | 1389 |
| VPSC3 | 12 | 16 | 75.0% | 2004-03-22 | 2008-01-03 | 1382 |
| OGXP3 | 1336 | 1688 | 79.1% | 2013-10-30 | 2017-08-03 | 1373 |
| WMBY3 | 2 | 3 | 66.7% | 2008-02-27 | 2010-04-05 | 1358 |
| TROR4 | 1 | 454 | 0.2% | 2000-03-10 | 2003-11-26 | 1356 |
| PTNT3 | 13 | 1539 | 0.8% | 2005-05-18 | 2007-06-01 | 1346 |
| RHDS4 | 5 | 7 | 71.4% | 2003-11-18 | 2007-07-23 | 1343 |
| CLSC5 | 17 | 24 | 70.8% | 2008-08-06 | 2010-12-30 | 1320 |
| IGBR6 | 3 | 52 | 5.8% | 2003-08-13 | 2007-02-13 | 1280 |
| NAFG4 | 15 | 355 | 4.2% | 2004-11-23 | 2008-05-19 | 1273 |
| NORD3 | 40 | 1235 | 3.2% | 2005-05-05 | 2008-10-23 | 1267 |
| CELP7 | 252 | 489 | 51.5% | 2011-11-14 | 2015-04-28 | 1261 |
| VPTA3 | 6 | 38 | 15.8% | 2004-10-26 | 2007-03-29 | 1260 |
| SGEN3 | 5 | 78 | 6.4% | 2001-04-24 | 2004-10-05 | 1260 |
| IVIL3 | 1 | 2 | 50.0% | 2000-06-08 | 2003-11-19 | 1259 |
| AHEB6 | 38 | 83 | 45.8% | 2018-02-02 | 2020-06-08 | 1257 |
| GFTT3 | 1 | 2 | 50.0% | 2001-01-19 | 2004-06-29 | 1257 |
| GFTT4 | 1 | 2 | 50.0% | 2001-01-19 | 2004-06-29 | 1257 |
| MLFT3 | 1 | 123 | 0.8% | 2000-06-27 | 2003-11-21 | 1242 |
| CELP3 | 4 | 964 | 0.4% | 2011-08-08 | 2014-12-08 | 1218 |
| SOND3 | 27 | 32 | 84.4% | 2022-04-26 | 2024-10-10 | 1191 |
| CELP5 | 425 | 752 | 56.5% | 2011-11-14 | 2015-01-23 | 1166 |
| CTPC4 | 4 | 17 | 23.5% | 2004-02-11 | 2007-04-05 | 1149 |
| EKTR3 | 144 | 341 | 42.2% | 2013-01-23 | 2016-03-14 | 1146 |
| MNPR3 | 4 | 4438 | 0.1% | 2004-01-12 | 2007-02-21 | 1136 |
| ESTR3 | 154 | 175 | 88.0% | 2019-03-26 | 2021-07-12 | 1127 |
| TMGC6 | 7 | 20 | 35.0% | 2002-02-25 | 2005-03-21 | 1120 |
| EUCA3 | 14 | 1473 | 1.0% | 2012-03-20 | 2015-04-09 | 1115 |
| CPLE5 | 89 | 566 | 15.7% | 2018-04-30 | 2020-05-08 | 1082 |
| LETO3 | 1 | 14 | 7.1% | 2002-01-23 | 2005-01-07 | 1080 |
| RCSL3 | 18 | 3104 | 0.6% | 2006-01-26 | 2008-12-29 | 1068 |
| RCSL4 | 557 | 4803 | 11.6% | 2006-01-26 | 2008-12-29 | 1068 |
| CALI4 | 86 | 105 | 81.9% | 2013-10-10 | 2016-09-09 | 1065 |
| GRNL3 | 6 | 28 | 21.4% | 2004-12-21 | 2007-02-12 | 1055 |
| AZEV4 | 60 | 2758 | 2.2% | 2008-06-18 | 2011-05-02 | 1048 |
| DUQE3 | 20 | 21 | 95.2% | 2008-09-01 | 2011-07-11 | 1043 |
| LHER4 | 1 | 11 | 9.1% | 2010-09-03 | 2013-07-03 | 1034 |
| BMEB3 | 5 | 2564 | 0.2% | 2004-07-23 | 2006-08-31 | 1022 |
| MTIG3 | 1 | 476 | 0.2% | 2001-09-21 | 2004-07-08 | 1021 |
| MTIG4 | 1 | 2597 | 0.0% | 2001-09-21 | 2004-07-07 | 1020 |
| GLOB3 | 11 | 807 | 1.4% | 2000-10-31 | 2003-08-15 | 1018 |
| DHBI3 | 53 | 54 | 98.1% | 2012-01-16 | 2014-07-24 | 1014 |
| MGEL3 | 1 | 2 | 50.0% | 2008-06-06 | 2011-03-11 | 1008 |
| CTWR3 | 1 | 2 | 50.0% | 2000-04-26 | 2003-01-21 | 1000 |
| UGPA3 | 44 | 3744 | 1.2% | 2004-08-25 | 2007-05-07 | 985 |
| LLIS3 | 2911 | 2919 | 99.7% | 2020-06-05 | 2023-01-30 | 969 |
| SPRI6 | 383 | 441 | 86.8% | 2016-05-02 | 2018-12-26 | 968 |
| TIBR3 | 9 | 22 | 40.9% | 2004-11-19 | 2007-07-04 | 957 |
| VULC4 | 20 | 155 | 12.9% | 2002-04-05 | 2004-10-25 | 934 |
| BUET3 | 1 | 14 | 7.1% | 2002-09-17 | 2005-04-07 | 933 |
| SCLO3 | 25 | 39 | 64.1% | 2007-12-13 | 2010-03-18 | 926 |
| MAPT3 | 218 | 313 | 69.6% | 2016-10-25 | 2019-05-02 | 919 |
| SUZB6 | 28 | 43 | 65.1% | 2010-04-19 | 2012-09-12 | 917 |
| FESA3 | 3 | 1470 | 0.2% | 2000-06-06 | 2002-12-06 | 913 |
| LFFE3 | 1 | 144 | 0.7% | 2002-04-17 | 2004-10-05 | 902 |
| LREN3 | 3 | 5212 | 0.1% | 2002-10-28 | 2005-04-01 | 886 |
| AHEB5 | 51 | 133 | 38.3% | 2014-01-07 | 2016-05-20 | 864 |
| SLED4 | 40 | 3781 | 1.1% | 2000-03-02 | 2002-07-08 | 858 |
| FBMC3 | 38 | 39 | 97.4% | 2010-07-15 | 2012-11-06 | 845 |
| FGUI3 | 17 | 18 | 94.4% | 2005-01-31 | 2007-05-24 | 843 |
| LREN4 | 9 | 41 | 22.0% | 2001-01-31 | 2003-05-21 | 840 |
| FRAS3 | 1 | 3063 | 0.0% | 2003-10-17 | 2006-02-01 | 838 |
| MSPA3 | 2 | 257 | 0.8% | 2001-05-23 | 2003-09-05 | 835 |
| LFFE4 | 251 | 327 | 76.8% | 2011-09-06 | 2013-12-19 | 835 |
| SLED3 | 20 | 619 | 3.2% | 2001-06-22 | 2003-09-16 | 816 |
| MRSL3 | 2 | 12 | 16.7% | 2005-06-06 | 2007-08-24 | 809 |
| BALM4 | 47 | 1657 | 2.8% | 2006-03-10 | 2008-05-09 | 791 |
| CBMA3 | 1 | 742 | 0.1% | 2004-12-17 | 2007-02-02 | 777 |
| CAFE3 | 1 | 160 | 0.6% | 2000-01-17 | 2002-02-28 | 773 |
| TANC4 | 32 | 96 | 33.3% | 2001-03-08 | 2003-04-02 | 755 |
| FFTL3 | 11 | 76 | 14.5% | 2001-03-26 | 2003-04-09 | 744 |
| FLBR3 | 1 | 12 | 8.3% | 2000-01-13 | 2002-01-17 | 735 |

## Checklist

- [x] `BRDT3` — fixed.
- [ ] `CYRE4` — pre-verified, needs `ORPHAN_PREFIX_TICKERS` entry only.
- [ ] `LREN3` — top-50 member, currently harmless (out of training window), re-verify if training start date changes.
- [ ] Remaining 115 — unverified, lower priority, verify per-ticker before fixing (do not bulk-apply).
