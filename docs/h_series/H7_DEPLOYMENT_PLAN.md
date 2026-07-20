# H7 — Deployment Plan (Phase 3, Contingent — Not Scheduled)

**Status:** contingent draft. Not worth detailing further until H6's report is in hand.

## Objective

Turn a validated research pipeline into something that can actually generate a real monthly
target-weight list.

## Rationale

Explicitly contingent — nothing here is worth designing until H1-H6 collectively hold up net of
realistic costs. Designing deployment mechanics for an unproven strategy solves a problem that
may not need solving; listed here so it isn't forgotten, not because it's next.

## Design

Monthly decision-date scheduling aligned to the real CVM filing-date latency already handled in
Stage 2 (`src/data_collection/cvm/filing_dates.py`); a runnable script that pulls the latest data,
runs H3+H4's pipeline, and emits a target-weight list a person can act on. No automatic order
execution — explicitly a target-weight generator, not a trading bot, consistent with the caution
level maintained throughout this project so far.

## Implementation

Thin orchestration script reusing existing data-collection (`--mode update`) and the H3+H4
modules; no new modeling, purely operational wiring.

## Expected Outcome

A monthly, repeatable, low-effort way to generate target weights.

## Validation

Dry-run across the most recent **6** already-known months (not just one — a single date can't
distinguish a systematic issue from a one-off coincidence); confirm output matches what the
backtest would have produced for those dates, **within the vendor-discrepancy tolerance this
project already established** (`tests/data_collection/validate_vs_yfinance.py`'s existing
1-15% tolerance band on key ratios), not exact equality.

## Success Criteria

Script runs end-to-end without manual intervention beyond a data refresh; live-path target
weights match backtest-path weights within the established BolsAI/yfinance tolerance band for
all 6 dry-run dates; any date touched by a corporate event in the live window is flagged
explicitly (not silently compared) since `--mode update` doesn't run the same
split/merger-repair path the backtest's full-scale data went through.

## Failure Criteria

A mismatch **beyond** the known vendor-discrepancy tolerance, or on a date with no corporate
event to explain it — that combination indicates a real data-handling bug (e.g. a lookahead leak
or stale-data issue), as opposed to the already-understood, already-tolerated vendor divergence.

## Risks & Assumptions

- Assumes production data latency matches what the backtest assumed (real filing dates, not
  idealized ones) — already well-handled elsewhere in this project via the CVM filing-date work,
  but worth re-verifying specifically for the live path rather than assumed to carry over.
- **CVM's own publication latency is a distinct question from the filing-lag handling already
  built** (`filter_excessive_filing_lag()` handles late *filings*; it says nothing about whether
  CVM's open-data portal itself is current as of "now" for a live monthly run) — needs a direct
  check before trusting a live run's fundamentals freshness, not assumed safe by analogy.
- The BolsAI/yfinance vendor discrepancy (§Validation) is a known, already-documented
  characteristic of this project's data, not a new risk introduced by deployment — but it's the
  first place deployment will actually collide with it in practice.

## Next Decision Gate

N/A beyond this stage in the current plan. Deploying with real money is explicitly a separate,
later decision — not an automatic consequence of this stage succeeding.
