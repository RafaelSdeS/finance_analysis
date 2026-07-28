"""sec/fundamentals.py — combine the XBRL (2007+) and EX-27 (usably 1995-2000)
tiers into one point-in-time fundamentals table per company.

The 2001-2006 gap between them (Phase 7, Item 6 chaining) is not filled here --
this just concatenates what the two built tiers already produce, tagging each
row's source via `fundamentals_tier` so the gap is visible as a data hole, not
silently interpolated.
"""

import logging

import pandas as pd

from . import companyfacts, fds

log = logging.getLogger("sec")

# XBRL preferred over EX-27 on any overlapping fiscal period end (richer, more
# reliable tier -- see plan §2.0). In practice the two tiers shouldn't overlap
# (EX-27 usably ends 2000, XBRL starts ~2006-2007), but a company whose XBRL
# comparatives reach unusually far back could clash; resolve deterministically
# rather than leave a silent duplicate `end`.
_TIER_PRIORITY = {"xbrl": 0, "ex27": 1}


def build_company_fundamentals(cik: int, filings: pd.DataFrame) -> pd.DataFrame:
    """One CIK's combined fundamentals across both built tiers, one row per
    fiscal period end, each stamped with `fundamentals_tier` and a real
    `fundamentals_available_date` (never the period end -- plan §5.2)."""
    frames = []

    facts = companyfacts.fetch_companyfacts(cik)
    if facts is not None:
        line_items = companyfacts.extract_line_items(facts)
        if not line_items.empty:
            xbrl = companyfacts.compute_us_ratios(line_items)
            xbrl["fundamentals_tier"] = "xbrl"
            frames.append(xbrl)

    ex27 = fds.build_cik_history(cik, filings)
    if not ex27.empty:
        ex27 = ex27.rename(columns={"fds_period_end": "end"})
        ex27["fundamentals_tier"] = "ex27"
        frames.append(ex27)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["cik"] = cik
    combined["_priority"] = combined["fundamentals_tier"].map(_TIER_PRIORITY)
    return (combined.sort_values(["end", "_priority"])
                     .drop_duplicates(subset="end", keep="first")
                     .drop(columns="_priority")
                     .sort_values("fundamentals_available_date")
                     .reset_index(drop=True))
