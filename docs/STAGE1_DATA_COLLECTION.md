# Stage 1: Data Collection — How It Actually Works

Source: `src/data_collection/`. This explains the implemented mechanics — for run commands see `CLAUDE.md`, for the target design see `specification.txt`.

## Sources and why there are three

| Source | Data | Cost | Coverage |
|---|---|---|---|
| BolsAI API | prices, fundamentals, dividends, company info | paid, ~€0.10/1K calls | full B3 history, backfill |
| BCB SGS API | SELIC, CDI, IPCA | free | daily/monthly macro series |
| yfinance | prices, fundamentals, dividends | free | incremental refresh only |

BolsAI is the backfill source (one-time historical pull to `START_DATE = 2000-01-01`); yfinance is the ongoing incremental refresh (`--mode update`) so quarterly updates don't re-incur BolsAI's per-call cost. A `DATA_SOURCE` dict in `config.py` maps each data type (`prices`/`fundamentals`/`dividends`) to `"bolsai"` or `"yfinance"` independently — flipping one entry redirects that data type's collector without touching the others. `company_info` and macro have no yfinance equivalent and always come from BolsAI/BCB.

One BCB gotcha worth stating plainly: the SELIC series ID used is **11** (the daily rate), not **432** (the annual policy-meeting target). They're easy to confuse and return very different-looking numbers; the wrong one silently produces plausible-looking but wrong data.

## Ticker universe

`collectors.get_all_tickers()` paginates BolsAI's `/stocks/` endpoint and filters to the regex `^[A-Z0-9]{4}[3-8]$` — standard B3 equity tickers, which excludes BDRs (suffix 34/35) and FIIs/ETFs (suffix 11). `BOVA11` (the IBOV-proxy benchmark ETF) is force-included even though its suffix would otherwise be excluded, and is tagged in `YFINANCE_ONLY_TICKERS` since BolsAI doesn't carry ETFs — it always comes from yfinance regardless of the `DATA_SOURCE` setting for that data type.

Prototype mode uses a fixed 3-ticker sample (`PETR4, VALE3, WEGE3`) validated against yfinance before scaling up.

## HTTP layer and resilience

`client.py` wraps `httpx.Client` with a single retry/backoff routine (`get_json`):

- Retryable: `{429, 500, 502, 503, 504}`, plus connection/timeout errors, plus BCB's specific quirk of intermittently returning an empty `200` body (a JSON-decode failure on a 200 is treated as transient, not fatal).
- Backoff: `wait = min(BACKOFF_BASE * 2**attempt, BACKOFF_MAX)` = 1s, 2s, 4s… capped at 30s, across `MAX_RETRIES = 3` attempts.
- Any other 4xx **fails fast** — no retry, since a client error (bad ticker, malformed request) won't fix itself on retry.

yfinance collectors use a separate, simpler retry (`_retry` in `yf_collectors.py`): a bare `except Exception`, `wait = YF_RETRY_SLEEP * 2**attempt` (2s, 4s, 8s, uncapped), 3 attempts. The justification in-code is that yfinance is a different transport with no typed exceptions worth special-casing — same exponential-doubling idea, deliberately less machinery.

## Idempotency and checkpointing

Every collector writes through one shared function, `_merge_save`, regardless of source:

1. Concatenate new rows onto the existing parquet (if any).
2. `drop_duplicates(subset=["ticker", date_col], keep="last")` — **the newest fetch wins** on any overlapping date, so re-running a collector to patch a bad day is safe.
3. Run the relevant `validate.py` check; if it fails, **nothing is written** (fail closed, not partially).
4. Sort and save.

This is why the whole pipeline is safe to re-run: a crash mid-collection leaves the last successfully-validated parquet on disk, and the next run's `drop_duplicates` reconciles any overlap. Checkpoints (`checkpoint.py`) are one JSON file per collector per mode (`data/checkpoints/{mode}/{name}.json`), storing each ticker's last-collected date/quarter — namespaced by mode so `prototype`, `full_scale`, and `update` never share or clobber each other's resume state.

## Validation gates (`validate.py`)

A `ValidationResult` (`passed`, `warnings`, `errors`) is returned by one validator per data type. All share `_common`: reject empty frames, reject missing required columns, reject any date more than 2 days in the future, warn (don't fail) on duplicate dates. Beyond that:

- **Prices**: error on `close <= 0` or negative volume; warn (not error) on >5-calendar-day gaps between trading days — could be holidays, could be a real halt, worth a human's attention but not worth blocking the pipeline.
- **Fundamentals**: warns if `cagr_earnings_5y` is null in >50% of rows *after* the first 20 quarters (the first ~5 years structurally can't have a 5-year CAGR yet, so those rows are excluded from the check).
- **Company info**: requires ticker/name/CVM code/CNPJ; errors on duplicate tickers.
- **Macro / dividends**: errors if a macro column is entirely null, or if any dividend `value_per_share <= 0`.

## Per-collector mechanics worth knowing

- **Macro** (`collect_macro`): BCB needs date-range chunking (10-year windows) and Brazilian `dd/mm/yyyy` date formatting; a 404 from BCB means "no data published for this range" (e.g. a weekend), not an error.
- **Prices** (`collect_prices`): BolsAI caps requests at 5000 rows, so a full 2000-present backfill is chunked into 10-year windows (`PRICE_CHUNK_YEARS`, ≈2500 rows/window at ~250 trading days/year). Incremental runs (once a checkpoint exists) fetch only `(last_date+1, today)`.
- **Fundamentals** (`collect_fundamentals`): BolsAI rejects `limit >= 90`; `FUND_LIMIT = 80` covers all ~62 quarters currently available in one call. If a ticker ever exceeds 80 quarters, the fallback is to paginate via start/end params (confirmed to work, just not needed yet).
- **Company info** (`collect_company_info`): fetches the *entire* company list in one or two paginated calls rather than one request per ticker — a 500-ticker collection is 1-2 API calls, not 500.
- **Dividends** (`collect_dividends`): the only collector with no incremental checkpoint — it always re-fetches the full configured window (20 years) and relies purely on `_merge_save`'s dedup for idempotency, since dividend history is small enough that re-fetching is cheap.
- Per-ticker exceptions in `collect_prices`/`collect_fundamentals` are caught and logged, not fatal — one bad ticker doesn't abort a 400-ticker run.

## The yfinance incremental path

Since yfinance and BolsAI report figures in different conventions, `yf_collectors.py` reconciles them rather than storing raw yfinance output directly:

- **Units**: BolsAI stores fundamentals in BRL *thousands*; yfinance reports full BRL. A constant `K = 1000` scales every yfinance-derived fundamental figure to match.
- **Splits**: yfinance's `auto_adjust=False` history is BolsAI's unadjusted convention, but a split that occurs *within* the newly-fetched window needs correcting — the collector detects splits via `t.splits` and multiplies pre-split OHLC rows in the new batch by the split ratio. Already-stored historical rows are never rewritten.
- **Adjusted OHLC**: yfinance only gives an adjusted *close*; adjusted open/high/low are derived proportionally as `raw * (adj_close / close)`.
- **Documented divergences** (real, not hidden): `volume_adjusted` is set equal to raw volume because yfinance doesn't split-adjust volume the way BolsAI does; `traded_amount` is approximated as `close * volume`; `num_trades` has no yfinance equivalent and is left `None`; dividend `type` is always `"UNKNOWN"` since yfinance can't distinguish JCP (juros sobre capital próprio) from a regular dividendo, which matters for Brazilian tax treatment but isn't recoverable from the feed.
- **Fundamental ratios** (`_compute_ratios`): recomputes ~37 BolsAI-equivalent ratios (P/E, P/B, ROE, ROA, margins, leverage, EV multiples, etc.) from yfinance's raw income statement / balance sheet figures, using TTM sums (`rolling(4).sum()`) for flow items (revenue, net income, EBITDA) and point-in-time values for balance-sheet items (equity, assets, debt, cash). These formulas were checked at 5% tolerance against live BolsAI data (`tests/data_collection/validate_vs_yfinance.py`). One explicitly-flagged approximation: `roic = ebit / (total_debt + equity - cash) * 100` uses pre-tax EBIT rather than tax-effected NOPAT, because yfinance doesn't expose an effective tax rate cleanly enough to compute true NOPAT — an accepted simplification, not an oversight. 5-year CAGR columns are left null from yfinance (only ~1.5 years of quarterly depth available) and are backfilled later by `cagr_handler.py` (Stage 2) once merged with the longer BolsAI history.

## Pipeline orchestration (`pipeline.py`)

Stage order per run: `macro` (always, ticker-independent) → `company_info` (skipped in `--mode update`, to minimize BolsAI usage — run `full_scale`/`prototype` manually when a new IPO needs picking up) → `prices` / `fundamentals` / `dividends` (only for tickers whose `company_info.status == "ATIVO"`, plus benchmarks, which always get prices regardless of status).

`--dry-run` logs what *would* run (mode, ticker count/preview, per-type source) without any network calls — useful for confirming a full-scale run's scope before spending BolsAI credits. A stage-level exception aborts the whole `run()` call (returns `False`), but every stage that completed before the failure has already persisted its data and checkpoint, so a re-run resumes rather than restarts.

## Commands

See `CLAUDE.md` → "Stage 1: Collect Raw Data" for the exact CLI invocations (`--mode prototype/full_scale/update`, `--dry-run`, `--tickers`).
