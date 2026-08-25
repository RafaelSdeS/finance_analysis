# Data Collection: Theory and Findings

Everything the hard way taught us about collecting this dataset, written as background
before you restart the work — not a runbook, just the concepts and findings that will
save you from re-discovering them by re-breaking them.

---

## 1. "Active" is a leakier concept than it sounds

The registry status a data vendor gives you (ATIVO/CANCELADA) is keyed to the
**company** (its tax ID / CNPJ), not to the **ticker code**. A company can be alive and
trading today under a ticker the registry has never linked to it, because it renamed,
merged, or refreshed its listing. This single fact is the root of almost every
ticker-specific oddity in this dataset. There are three genuinely different reasons a
ticker can look "missing," and they need different explanations, not one fix:

1. **Never seen at all.** Free official crosswalks (ticker ↔ company ID) are often
   survivor-style and only go back a limited number of years — a ticker retired before
   that window opened is invisible to them entirely.
2. **Genuinely delisted.** The company itself stopped trading (bankruptcy, going
   private, cancellation). Any vendor correctly stops returning data.
3. **Company alive, code retired.** The registry still marks the *company* active
   because it kept trading — just under a new symbol. Querying the old symbol 404s or
   returns nothing, which looks identical to case 2 from the outside, but the fix is
   completely different: this is a naming problem, not a coverage gap.

Case 3 is the one that fools you, because it presents exactly like a real coverage gap.
The tell: does this company trade today under a *different* symbol? If a "delisted"
ticker's company registry status still reads active despite no recent price activity,
that's the signature of a disguised rename, not a real delisting.

A second, related trap: some ticker suffixes are structurally ambiguous. A four-letter
+ "11" suffix is used both by fund shares (ETFs/REITs) and by real operating-company
"units" (a bundled package of common + preferred shares of one company). A naive regex
filter can't distinguish the two by shape alone — and a hand-maintained allowlist of
"known real units" silently goes stale and drops real, liquid company units over time.
The reliable test is external: does an official company-ID crosswalk actually resolve a
tax ID for that ticker? If yes, it's a real operating company, not a fund. **Lesson:
never hand-maintain an allowlist for something an authoritative source can already
answer.**

---

## 2. The vendors and what each is actually for

This dataset spans two markets (Brazil and the US), and neither market's data comes
from one source. Six vendors show up, each earning its place for a narrow, specific
reason — not because more sources are inherently better, but because no single vendor
covers the whole surface honestly. Knowing each one's actual role (and where it quietly
stops being trustworthy) matters more than knowing its name.

- **CVM (Brazil's securities regulator)** — the official, free, keyless source of
  truth for Brazilian company fundamentals and corporate registry status. Its filings
  are the authoritative record of what a company actually reported, when, and to whom
  it belongs (its tax ID, its sector, whether it's still registered as active).
  **Shortfalls:** its ticker↔company crosswalk is survivor-style and only extends back
  a limited number of years — a ticker retired before that window opened is invisible
  to it forever, no amount of re-scanning later years recovers it, because each year's
  filing only reports the ticker code *as of that filing*. Its "active" status is
  company-level, not ticker-level (see §1) — it can correctly say a *company* is alive
  while being useless for figuring out which *ticker code* it trades under today. Its
  registry of company records also carries duplicate rows per company (stale
  re-registration episodes from older regulatory eras), so a naive read overcounts
  unless you know to keep only the current/latest row per company. Its dividend
  disclosures are split across formats with different strengths: one gives per-share,
  per-class amounts with real payment dates but only for a bounded historical window
  before that specific disclosure format was retired; a continuous but coarser
  alternative disclosure covers the years the first one doesn't, but only as a total
  company-wide amount, with no share-class breakdown and no payment date — good enough
  to sanity-check a total, not to reconstruct a payment series. It has no opinion at
  all on foreign markets.

- **yfinance** — a free, keyless price/dividend/split vendor, in practice the most
  reliable source for exactly that slice of data in both markets. **Shortfalls:** it is
  not a fundamentals source of record for this project's home market — see §6's
  currency and scale findings for why. It only ever describes companies that are
  *currently listed*: query a retired ticker symbol and it simply has nothing to say,
  which is exactly why it can't answer "what happened to a delisted company" on its
  own. And even within its strong suit, precision quietly degrades at the edges: a
  meaningful share of deep-history, very-low-priced (microcap) names underflow its
  displayed decimal precision, rounding to zero or pinning at a tiny constant — not a
  bug in this project's handling of the data, a floor in what the vendor itself
  returns, unfixable by reprocessing.

- **A paid commercial aggregator (referred to here generically, since its identity
  matters less than its role)** — the one source in this whole stack that can answer
  "what did a company that's no longer listed look like while it was trading," because
  it retains historical coverage the free official sources never indexed and the
  free market-data vendor drops the moment a symbol stops trading. That's a genuinely
  irreplaceable capability. **Shortfalls:** it is not a reliable source for *current*
  data quality — see §6's price-quality and fundamentals-scale findings, both measured
  directly against it. Its dividend endpoint is also structurally thinner than it looks:
  it returns annual per-year summaries, not individual payment records — no ex-date, no
  pay-date, no per-payment granularity — which is enough for a coarse yield estimate but
  not for anything that needs to know exactly when a payment happened. And it carries
  the "rename phantom" failure mode described in §6: it can silently resolve a retired
  ticker code to a live successor company's current data. The practical shape this
  settles into: pay for what only a paid vendor can give you (deep delisted-company
  history), and don't pay for anything a free source already does better (current
  prices, current fundamentals, or anything needing payment-level dividend precision).

- **A national central bank's macroeconomic data service (Brazil)** — free, keyless,
  and the sole source for policy-rate, interbank-rate, and inflation series. Nothing
  else substitutes for it; there's no "vendor comparison" question here, only a
  correctness one — **the real shortfall is entirely on the consumer side, not the
  vendor's:** knowing exactly which named series is the *daily traded rate* your return
  calculations need, versus which superficially similar series is an annual policy
  *target* that looks plausible, updates on a similar cadence, and is the wrong number
  entirely. The vendor will happily hand you either one; it doesn't know which one your
  math assumes.

- **FRED (a US Federal Reserve data service)** — the US-market equivalent role: free,
  keyless, authoritative for US macro series. Structurally parallel to the Brazilian
  central bank's role above, and just as uniquely positioned — there is no serious
  alternative for public, free, official US macro time series. **Shortfall:** the same
  series-naming trap applies in principle (multiple similarly-named series covering
  overlapping but distinct concepts), and macro series of this kind are also routinely
  revised after first publication — a series pulled today can differ from what a market
  participant actually saw in real time on the original date, a subtler point-in-time
  risk than the price/fundamentals lookahead concerns elsewhere in this project.

- **SEC EDGAR (the US securities regulator's public filing system)** — the US
  equivalent of CVM: free, keyless, authoritative for what a US-listed company actually
  filed. **Shortfalls:** its *machine-readable* structured filing format only exists for
  recent history (roughly the last decade and a half); older filings exist only as
  less-structured formats from earlier reporting-technology eras, and there is a real
  gap window in between where neither the newest structured format nor the oldest
  exhibit format cleanly applies — that middle era needs its own, more effort-intensive
  parsing approach, pieced together from a slower quarterly filing format instead of a
  clean annual one. Getting continuous fundamentals coverage across a long US history
  means combining several different filing-format eras end to end, each parsed
  differently, not one clean API across the whole timeline. Its company-identifier
  universe is also a snapshot of *current* filers by default, the same survivor-style
  trap as CVM's crosswalk — a delisted company's identifier can drop out of the current
  roster the same way a retired ticker drops out of CVM's.

**The pattern across all six:** free-and-official is authoritative for compliance-grade
truth (what was actually filed, what a company's real regulatory status is) but is
usually incomplete for historical breadth, especially for anything no longer trading.
Free-and-commercial (yfinance) is excellent for current market data but was never
designed as a fundamentals source of record. Paid-and-commercial fills the one gap
neither free source can: deep historical coverage for things that no longer exist,
at a real financial cost and with real, measured quality tradeoffs everywhere else.
Picking "the best vendor" is the wrong frame — the right frame is picking the right
vendor *per data type*, which is exactly the conclusion §6 arrives at independently
from the fundamentals side.

---

## 3. Corporate actions: renames, mergers, and lookalikes are not the same fix

Knowing *that* a ticker changed identity isn't enough — you also need to know *how*,
because the wrong treatment silently corrupts the price series. Four distinct shapes
show up in a market with this much corporate activity:

- **Rename** — same legal entity, only the ticker code (or company name) changed. The
  two price series are literally one continuous history; splice them end-to-end.
- **Merger with a true successor** — the old entity legally ceased; shareholders
  received shares of a new company at some exchange ratio. Splice the series, but scale
  the old leg's price by the exchange ratio and its volume inversely, so dollar-volume
  stays invariant across the boundary — do not just stitch raw prices together across a
  ratio change.
- **Acquisition by an already-listed acquirer ("parallel-trading acquirer")** — the
  entity that absorbed the company was *already trading independently* before the deal.
  This one cannot be spliced: the acquirer's pre-deal history belongs to a completely
  different economic story than the target's. Treat the target as simply delisted at
  the acquisition date and leave both series independent. The tell here is chronology:
  if the "successor" already had years of price history predating the corporate event,
  it isn't a successor at all, it's a separate company that happened to buy this one.
- **Cash tender / going-private** — no successor security exists at all. Not a splice;
  this becomes a realized terminal payoff instead (see §4).

A subtler trap sits inside all of this: **ticker codes get reused by unrelated
companies over time.** A ticker that looks like the obvious sibling of a confirmed
rename (say, the other share class of the same stock) can turn out to have price
history *predating* the rename event by a decade or more — meaning that code was
already in use by a completely different, unrelated listing before the renamed company
reused it. Splicing on the assumption "same company, other share class" would silently
delete or overwrite real, unrelated history. The only defense is checking for
history predating the event before trusting an apparent sibling relationship.

**How to classify without guessing:** compare the last trade price of the old leg
against the first trade price of the new leg. A genuine rename or 1:1 merger produces
prices that are adjacent (within a few percent) on either side of the boundary — the
market doesn't reprice a company by 5x just because its ticker changed. A real exchange
ratio produces a *consistent, round* multiple across the boundary, not noise. If
neither holds, or if the "successor" has independent pre-event history, it's not a
splice case.

---

## 4. Splits and price-scale repair

Vendors sometimes fail to back-adjust historical prices for a real stock split or
reverse split (inplit) — the adjusted-close series should show a smooth line through
the event, but instead shows a step. The fix is a scale correction: multiply pre-event
adjusted prices (and volumes, inversely, to keep dollar volume consistent) by the split
factor, for the segment before the event.

The interesting failure mode is around **detection, not correction**: a real
split-driven jump is not distinguishable from noise by size alone — you need a matching
corporate-action record near the same date to confirm it's a real event and not just
volatility. And even confirmed corporate-action dates are often only month-precision, so
detection windows need to be generous around the recorded date, not exact-day matching.

One genuinely surprising finding: it's not always only the *adjusted* price columns
that need this repair. In one case, a ticker's *raw* (nominally unadjusted) OHLC columns
also silently carried a scale defect — normally raw OHLC is deliberately left untouched,
since it's supposed to show the real historical nominal price including jumps. This was
only caught because a downstream sanity invariant (something as simple as "market cap
divided by shares outstanding should equal the close price") flagged one ticker as
wildly wrong two processing stages later. **The lesson generalizes: a scale bug is often
invisible in the raw series and only becomes obvious once you check an invariant that
should hold regardless of scale.** Eyeballing raw data catches shape problems; it rarely
catches scale problems.

Order matters when both split-repair and identity-splicing (§2) apply to the same
company: repair each leg's own internal scale *first*, under its own original name,
before splicing the legs together. Splicing first would let an unrepaired scale error on
one leg propagate across the boundary into the other.

A related, permanent design decision worth internalizing: an automated "is this
suspicious price jump a real split or vendor corruption" classifier was attempted more
than once and abandoned every time — every version produced false positives on illiquid
tickers, where ordinary volatility for a thinly-traded name can swamp any workable
threshold. There is no clean statistical dividing line between "real 40% jump" and
"corrupted data that happens to look like a 40% jump" at low liquidity. The practical
answer that stuck: classify by hand against corroborating evidence (a real recorded
corporate event, or a cross-check against an independent series for the same
underlying company), not by threshold alone.

---

## 5. Delisting isn't one outcome, and "missing" isn't the honest default

When a company disappears from a live panel, the *reason* matters for what number you
should attach to it. Two broad outcomes:

- **Failure** (bankruptcy, liquidation) → the position is realistically worth close to
  nothing at the end.
- **Everything else** (voluntary cancellation, being acquired, merging into a
  successor) → the position was realistically worth close to its last observed traded
  price — often *more*, since acquisitions frequently happen at a premium.

The finding worth remembering: measured across a real market panel, **most companies
that die inside a dataset, die *rising***, not collapsing — acquisition-at-a-premium is
the dominant delisting mode in this market, not wipeout. A dataset that simply drops
delisted tickers, or worse, records their outcome as missing/undefined, is quietly
injecting a *negative* survivorship bias into anything trained on it — it's not neutral
to leave the outcome blank. A blank/NaN label there isn't a safe default, it's a
different, still-wrong assumption in disguise (effectively "unknown," which downstream
code often treats as "excluded," which is its own bias).

The practical implication: recovering a realistic terminal value for dead tickers is
worth real engineering effort, and the source of truth for *why* a company delisted
(the actual bankruptcy/cancellation registry) is more trustworthy for that judgment call
than inferring it from price action alone.

---

## 6. Vendor mismatch: the findings that justified splitting sources by data type

The single biggest lesson from this whole stage: **no one vendor is right about
everything, and the errors are not random — they cluster by data type in ways that
are only visible once you cross-check directly.**

- **Price/volume quality**: measured head-to-head on the same overlapping set of
  tickers, one free vendor beat a paid vendor on essentially every axis — more total
  rows, far fewer implausible >50% single-day jumps, zero non-positive close prices
  (the paid vendor had hundreds), and a much lower rate of prices that were suspiciously
  rounded to a coarse precision (a vendor-side floating-point/display artifact, not a
  real market fact). Paying for data does not guarantee better data.

- **Fundamentals — a scale bug, not just thinness.** One vendor's balance-sheet figures
  (equity, total assets) were observed dropping by roughly 5x in a single quarter with
  no real corporate event behind it — a parsing/units bug, not a reporting change. This
  kind of error is dangerous specifically because it's *plausible-looking* — a company's
  book value halving-and-then-some in one quarter doesn't scream "bug" the way a
  negative price does.

- **Fundamentals — an inconsistent convention, silently mixed.** The same vendor stored
  some tickers' flow figures (revenue, earnings) as trailing-twelve-month sums and
  others as single-quarter point-in-time values, essentially at random per ticker, with
  no flag distinguishing which. Any feature comparing companies cross-sectionally (a
  sector-relative ratio, a percentile rank) computed off this mix is comparing
  incompatible units without knowing it. This is a worse failure mode than an outright
  missing value, because it doesn't fail loudly — it just quietly produces wrong
  relative rankings.

- **Fundamentals — a currency mismatch hiding behind a correct-looking label.** For
  companies that are dual-listed (trading both domestically and, via an ADR, on a
  foreign exchange), a free vendor's fundamentals endpoint was found to silently serve
  the *foreign, dollar-denominated* figures under the *domestic* ticker symbol — while
  still labeling the currency field as the domestic currency. The ratio between the
  correct and the served figures matched the FX rate almost exactly. This is the most
  dangerous kind of vendor bug: the data is self-consistent, plausible-looking, and
  wrong only in a way you'd catch by independently cross-verifying against a source you
  trust — never by inspecting the file alone.

- **Coverage gaps aren't uniform either.** A live registry can simply be missing real,
  currently-trading companies — not stale, just never onboarded — confirmed by directly
  re-querying the vendor and finding the gap reproduces, not by assuming your own cache
  was out of date.

- **A vendor can also serve *someone else's* data under a ticker's name.** In more than
  one case, querying a vendor's live API for a specific (often delisted or renamed)
  ticker returned data that, on inspection, was actually another company's series
  entirely — sometimes the live successor's current data served under the old, retired
  symbol; sometimes a wholly unrelated company's numbers. This is distinct from "missing
  data" — it's confidently wrong data, and it will not announce itself. The only defense
  is periodically cross-checking a sample of tickers' data against an independent
  source rather than trusting internal consistency.

**The general finding underneath all of the above:** vendor errors are not evenly
distributed noise you can average away — they cluster by exactly the kind of query that
stresses a vendor's weak spot (delisted names, dual-listed names, thin-liquidity names,
reused symbols). The fix was never "pick the better vendor" — it was "figure out which
vendor is trustworthy for *which specific data type*, and only that type," because the
same vendor that's excellent at one thing (say, deep historical coverage of dead
tickers no free source has) can be actively wrong at another (current fundamentals for
the same company). Treat vendor choice as a per-data-type decision, not a global one,
and re-verify that decision with a direct, measured comparison rather than reputation.

---

## 7. State that lies about the present: negative caching

Any collection process that remembers "I already tried this and got nothing" runs the
risk of that memory becoming stale in a way that's invisible from the outside. If a
ticker failed to return data a few times in a row under old logic (say, because it was
queried under a retired code), a system that "gives up" on repeatedly-failing items
after a threshold will keep giving up even after the underlying reason for failure is
fixed — silently, with no error, just an absence. The absence looks identical to "there
really is no data," which is exactly the case it's least safe to assume.

The general principle: any cache of *failure*, not just of success, needs a way to be
told "the world changed, re-check this" — otherwise a fix to the collection logic can
be invisibly negated by stale memory of the old, broken behavior.

---

## 8. The single idea to keep

**A ticker is not a stable identity — it's a label that drifts, gets reused, and gets
shared ambiguously between unrelated instruments.** Almost every hard bug in this stage
traced back to treating the ticker string itself as the entity, instead of treating the
underlying company (or, failing that, the actual continuity of its price history) as
the entity and the ticker as just its current, temporary label. When a ticker's data
looks wrong, the productive question is "which real company is this actually describing
right now," not "what's wrong with this ticker's numbers" — the second question often
has no clean answer on its own terms, while the first one usually resolves it in one
step.
