# test_raw_processed_reconciliation.py
#
# The anti-survivorship gate: verifies every raw BR price ticker and every
# ticker that dies inside the built panel is accounted for by a real,
# traceable mechanism (kept, a documented drop reason, a continuity splice,
# a terminal payoff, or a flagged rename candidate) -- never a silent
# disappearance. See docs/DATA_INTEGRITY_TEST_PLAN.md P1-b.
#
# BR-only, deliberately: US collection is gated by the SEC crosswalk
# (current listings only), so it's survivor-only by construction and a
# reconciliation identity there would pass trivially. The honest US
# equivalent (survivorship_coverage per-year ratio) is asserted in
# test_manifest_drift.py instead.
#
# Run from project root:
#   python tests/build_dataset/test_raw_processed_reconciliation.py

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.paths import CONTINUITY_PATH, OUTPUT_PATH, PRICES_DIR  # noqa: E402
from src.build_dataset.quality_filters import STALE_TICKER_DAYS  # noqa: E402
from src.build_dataset.terminal_events import _dead_tickers, find_rename_candidates  # noqa: E402
from src.data_collection.cvm.delistings import build_delist_events  # noqa: E402
from test_utils import print_header, print_check, print_separator  # noqa: E402

TERMINAL_EVENTS_PATH = ROOT / "data/processed/terminal_events.parquet"

KNOWN_DROP_REASONS = {
    "quarantined", "known_non_company", "delisted_stale",
    "redundant_sibling", "gap_unexplained", "too_short_history",
}
GAP_UNEXPLAINED_CEILING = 1  # currently 1 (REAG3); a new unexplained gap must be investigated, not silently absorbed


def check_ticker_reconciliation(manifest: dict, kept: set) -> list[tuple[str, bool, str]]:
    """raw == kept UNION dropped UNION spliced, as a SET identity -- NOT a
    count sum. |kept|+|dropped|+|spliced| overcounts by |kept & spliced|:
    continuity.json's `keep_separate` entries (parallel-trading acquirer)
    are deliberately in BOTH the continuity map and the kept panel. Coding
    this as 567+735+31==1328 fails today (it's 1333) and wrongly implies
    corruption -- found writing this test, 2026-08-16.
    """
    raw = {p.stem for p in PRICES_DIR.glob("*.parquet")}
    dropped = set()
    for bucket in manifest["dropped_no_fundamentals"].values():
        dropped |= set(bucket)
    continuity = json.loads(CONTINUITY_PATH.read_text())
    spliced = {e["old"] for e in continuity["events"]}

    residual = raw - kept - dropped - spliced
    checks = [
        ("every raw price ticker is kept, dropped-with-reason, or continuity-spliced",
         not residual, f"unaccounted: {sorted(residual)}" if residual else ""),
    ]
    orphan_kept = kept - raw
    checks.append(("every kept ticker has a raw price file", not orphan_kept,
                    f"kept but no raw file: {sorted(orphan_kept)}" if orphan_kept else ""))
    return checks


def check_drop_bucket_labels(manifest: dict) -> list[tuple[str, bool, str]]:
    dropped_no_fundamentals = manifest["dropped_no_fundamentals"]
    unknown_buckets = set(dropped_no_fundamentals) - KNOWN_DROP_REASONS
    checks = [
        ("every dropped_no_fundamentals bucket is a known, documented reason",
         not unknown_buckets, f"new unlabeled bucket(s): {sorted(unknown_buckets)}" if unknown_buckets else ""),
    ]
    gap = dropped_no_fundamentals.get("gap_unexplained", [])
    gap_ok = len(gap) <= GAP_UNEXPLAINED_CEILING
    checks.append((f"gap_unexplained has not grown past {GAP_UNEXPLAINED_CEILING}",
                    gap_ok, f"now {len(gap)}: {gap}" if not gap_ok else ""))
    return checks


def check_terminal_events_coverage(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Every ticker that dies inside the panel (terminal_events.py's own
    STALE_TICKER_DAYS=730 definition of "dead") must be resolved by one of
    the three mechanisms the codebase actually has: a continuity splice, a
    realized terminal_events.parquet payoff, or a flagged
    find_rename_candidates() entry (report-only, awaiting a hand-verified
    continuity.json add). Anything outside all three is a silent survivorship
    hole: a real delisting whose forward-return label stays NaN instead of
    reflecting what actually happened.

    REAL FINDING, 2026-08-16: this does NOT hold today. 48 tickers with
    processed status=CANCELADA (a confirmed real delisting, not a live
    company) fall outside all three paths -- including AMER3 (Lojas
    Americanas/Americanas S.A., the Jan-2023 accounting-fraud collapse: its
    panel ends 2023-01-19 after a ~68% one-week price crash, currently
    labeled NaN instead of the real outcome). Root cause, not a data gap:
    CVM's own registry still shows these entities' `sit` as ATIVO (40) or
    SUSPENSO(A) - DECISÃO ADM (8) at the company level even though the
    TICKER stopped trading -- build_terminal_events() correctly treats
    sit==ATIVO as "not a delisting, maybe an unspliced rename" per its own
    docstring, and find_rename_candidates() correctly declines to propose a
    redirect when no OTHER ticker under the same CNPJ is still trading. Both
    are working as designed; neither was designed for "the company stopped
    trading entirely without CVM ever recording a resolved cancellation
    reason." This is a real pipeline gap, not a test bug -- fixing it is an
    economic-classification judgment call (does sit=ATIVO-but-dead default
    to failure, to acquired, or need a fourth bucket?) that belongs to
    whoever owns terminal_events.py's payoff logic, not to this test.
    """
    continuity = json.loads(CONTINUITY_PATH.read_text())
    spliced = {e["old"] for e in continuity["events"]}
    terminal_events = (
        set(pd.read_parquet(TERMINAL_EVENTS_PATH)["ticker"])
        if TERMINAL_EVENTS_PATH.exists() else set()
    )

    delist_events = build_delist_events()
    rename_candidates = find_rename_candidates(df, delist_events)
    flagged = set(rename_candidates["ticker_old"]) if len(rename_candidates) else set()

    dead = _dead_tickers(df)
    uncovered = dead - spliced - terminal_events - flagged

    label = (f"every dead ticker (>{ STALE_TICKER_DAYS }d stale) is spliced, paid a terminal "
             f"event, or flagged as a rename candidate")
    detail = (f"{len(uncovered)} uncovered, e.g. {sorted(uncovered)[:15]}"
              f"{' ...' if len(uncovered) > 15 else ''}") if uncovered else ""
    return [(label, not uncovered, detail)]


def main() -> int:
    print_header("RAW -> PROCESSED RECONCILIATION (BR)")

    if not TERMINAL_EVENTS_PATH.exists():
        print("SKIP: terminal_events.parquet not built -- run "
              "`python -m src.build_dataset.terminal_events` after build_ml_dataset.py")

    manifest = json.loads(OUTPUT_PATH.with_suffix(".manifest.json").read_text())
    df = pd.read_parquet(OUTPUT_PATH, columns=["ticker", "trade_date", "adj_close"])
    kept = set(df["ticker"].unique())

    checks: list[tuple[str, bool, str]] = []
    checks += check_ticker_reconciliation(manifest, kept)
    checks += check_drop_bucket_labels(manifest)
    checks += check_terminal_events_coverage(df)

    failed = 0
    for label, ok, detail in checks:
        print_check(label, ok, detail)
        if not ok:
            failed += 1
    print_separator()
    print(f"{len(checks) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
