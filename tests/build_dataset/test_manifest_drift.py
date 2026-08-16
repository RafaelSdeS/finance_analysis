# test_manifest_drift.py
#
# Catches "the rebuild silently got worse": diffs the current processed
# manifest(s) against the latest dataset_v{N} snapshot. See
# docs/DATA_INTEGRITY_TEST_PLAN.md P0-a.
#
# Deliberately does NOT use manifest.nan_regressions() as a whole-panel gate.
# Calibrated 2026-08-16 against real dataset_v1 -> dataset_v2 (both
# legitimate builds, +57 tickers): it fires 107 false positives, because
# widening the universe with thinner/delisted names raises nan_pct almost
# everywhere at once. nan_pct isn't comparable across builds when panel
# composition changes -- and that's this project's explicit direction of
# travel. A real per-cohort version needs stats recorded in write_manifest()
# first; not done here.
#
# The column-level median-drift check below doesn't have that problem: it's
# calibrated the same way (v1 -> v2) and stays clean (max drift 0.071 across
# 156 columns, 0 exceed the 0.25 threshold used here) because a real
# distribution shift moves the median regardless of which tickers are in
# the panel.
#
# Run from project root:
#   python tests/build_dataset/test_manifest_drift.py

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.paths import OUTPUT_PATH, US_OUTPUT_PATH  # noqa: E402
from test_utils import print_header, print_check, print_separator  # noqa: E402

MEDIAN_DRIFT_THRESHOLD = 0.25  # x (p99-p1); calibrated 2026-08-16, max observed 0.071
ROW_DROP_CEILING = 0.02
# Tolerates a handful of quarantine drops (WDCN3, CAMB4/LLIS3, CCTY3, ...),
# not a universe collapse -- "tickers never drops" would fire on the
# project's own routine quarantine practice.
TICKER_DROP_RATE_CEILING = 0.05

# US-only: columns known-empty today (2026-08-16, see manifest's
# empty_columns). Dated allowlist, not a blanket skip -- a NEW empty column
# still fails this check.
US_KNOWN_EMPTY_COLUMNS = {
    "ebitda_margin", "ebitda_margin_zhist_5y", "ebitda_growth_yoy", "dividend_coverage_ratio",
}

US_LATEST_YEAR_COVERAGE_FLOOR = 0.5  # measured 2026-08-16: 2026 = 0.721


def _latest_snapshot_manifest(output_path: Path):
    """Highest dataset_v{N}/ that actually snapshots THIS market's output.

    dataset_v{N}/ directories are not market-namespaced -- OUTPUT_PATH.parent
    and US_OUTPUT_PATH.parent are literally the same directory
    (data/processed/), and sync_dataset_version() always writes the copied
    manifest under the fixed name "ml_dataset.manifest.json" regardless of
    which market's OUTPUT_PATH it was called with. A bare `dataset_v*` glob
    would happily hand a BR snapshot to a US comparison (wrong columns,
    wrong dollar scale) -- found running this test for the first time,
    2026-08-16. Matching on `output_path.name` inside the snapshot dir is
    what actually discriminates: today only BR ever gets snapshotted (no
    us_ml_dataset.parquet exists in any dataset_v{N}/), so this returns
    (None, None) for US until a US snapshot mechanism exists -- correctly,
    not by a hardcoded market check.
    """
    candidates = sorted(
        (int(m.group(1)), p)
        for p in output_path.parent.glob("dataset_v*")
        if (m := re.fullmatch(r"dataset_v(\d+)", p.name)) and (p / output_path.name).exists()
    )
    if not candidates:
        return None, None
    _, path = candidates[-1]
    manifest_path = path / "ml_dataset.manifest.json"
    if not manifest_path.exists():
        return path, None
    return path, json.loads(manifest_path.read_text())


def compare_manifests(prev: dict, curr: dict) -> list[str]:
    """Human-readable drift violations between two manifests. Pure function,
    no I/O -- directly unit-testable against real snapshots or synthetic
    fixtures (see _self_check_compare_manifests below)."""
    violations = []

    dropped_cols = set(prev["columns"]) - set(curr["columns"])
    if dropped_cols:
        violations.append(f"columns dropped vs previous build: {sorted(dropped_cols)}")

    row_drop = (prev["rows"] - curr["rows"]) / prev["rows"] if prev["rows"] else 0
    if row_drop > ROW_DROP_CEILING:
        violations.append(f"rows dropped {row_drop:.2%} (ceiling {ROW_DROP_CEILING:.0%}): "
                           f"{prev['rows']} -> {curr['rows']}")

    ticker_drop = (prev["tickers"] - curr["tickers"]) / prev["tickers"] if prev["tickers"] else 0
    if ticker_drop > TICKER_DROP_RATE_CEILING:
        violations.append(f"tickers dropped {ticker_drop:.2%} (ceiling {TICKER_DROP_RATE_CEILING:.0%}): "
                           f"{prev['tickers']} -> {curr['tickers']} -- a few quarantines are expected, "
                           f"a mass drop is not")

    if curr["date_max"] < prev["date_max"]:
        violations.append(f"date_max went backwards: {prev['date_max']} -> {curr['date_max']}")

    prev_stats, curr_stats = prev.get("column_stats", {}), curr.get("column_stats", {})
    for col in sorted(set(prev_stats) & set(curr_stats)):
        p1, p99 = prev_stats[col].get("p1"), prev_stats[col].get("p99")
        p50a, p50b = prev_stats[col].get("p50"), curr_stats[col].get("p50")
        if None in (p1, p99, p50a, p50b):
            continue
        iqr = abs(p99 - p1)
        if iqr == 0:
            continue
        drift = abs(p50b - p50a) / iqr
        if drift > MEDIAN_DRIFT_THRESHOLD:
            violations.append(f"{col}: median drift {drift:.3f} exceeds {MEDIAN_DRIFT_THRESHOLD} "
                               f"({p50a:.4g} -> {p50b:.4g})")

    return violations


def check_market_drift(output_path: Path, label: str) -> tuple[str, bool, str]:
    """Market-agnostic on purpose: called for both BR and US. Gracefully
    no-ops (passes with a note) when no manifest or no snapshot exists yet
    -- today that's the whole US half, since no us_dataset_v{N}/ directory
    exists (2026-08-16). Starts asserting automatically once one does."""
    manifest_path = output_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return (f"[{label}] manifest drift", True, "no manifest -- not built yet")
    curr = json.loads(manifest_path.read_text())

    snap_dir, prev = _latest_snapshot_manifest(output_path)
    if prev is None:
        detail = "no dataset_v{N} snapshot exists yet -- nothing to diff against"
        return (f"[{label}] manifest drift vs latest snapshot", True, detail)

    violations = compare_manifests(prev, curr)
    ok = not violations
    detail = "; ".join(violations[:5]) + (f" (+{len(violations) - 5} more)" if len(violations) > 5 else "")
    return (f"[{label}] manifest drift vs {snap_dir.name}", ok, detail)


def check_us_empty_columns() -> tuple[str, bool, str]:
    label = "[US] empty_columns is within the dated allowlist"
    manifest_path = US_OUTPUT_PATH.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return (label, True, "US dataset not built")
    manifest = json.loads(manifest_path.read_text())
    unexpected = set(manifest.get("empty_columns", [])) - US_KNOWN_EMPTY_COLUMNS
    return (label, not unexpected, f"new empty column(s): {sorted(unexpected)}" if unexpected else "")


def check_us_survivorship_coverage() -> list[tuple[str, bool, str]]:
    manifest_path = US_OUTPUT_PATH.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return [("[US] survivorship_coverage", True, "US dataset not built")]
    manifest = json.loads(manifest_path.read_text())
    coverage = manifest.get("survivorship_coverage")
    if not isinstance(coverage, list) or not coverage:
        return [("[US] survivorship_coverage is recorded", False, f"got: {coverage!r}")]

    checks = []
    bad_range = [r for r in coverage if not (0 < r["coverage"] <= 1)]
    checks.append(("[US] every year's coverage is in (0, 1]", not bad_range,
                    f"{bad_range[:3]}" if bad_range else ""))

    bad_ratio = [r for r in coverage if r["priced_ciks"] > r["roster_ciks"]]
    checks.append(("[US] priced_ciks never exceeds roster_ciks", not bad_ratio,
                    f"{bad_ratio[:3]}" if bad_ratio else ""))

    latest = max(coverage, key=lambda r: r["year"])
    floor_ok = latest["coverage"] >= US_LATEST_YEAR_COVERAGE_FLOOR
    checks.append((f"[US] latest year ({latest['year']}) coverage >= {US_LATEST_YEAR_COVERAGE_FLOOR:.0%}",
                    floor_ok, "" if floor_ok else f"got {latest['coverage']:.1%}"))
    return checks


def _self_check_compare_manifests():
    """compare_manifests is non-trivial branching logic (5 independent
    violation types) -- exercised directly against synthetic fixtures so a
    future edit that breaks it fails loudly here, not as a missed
    regression on real data."""
    base = {
        "columns": ["a", "b", "pl"], "rows": 1000, "tickers": 100, "date_max": "2026-01-01",
        "column_stats": {"pl": {"p1": 0.0, "p50": 10.0, "p99": 100.0}},
    }
    clean = dict(base, rows=1010, tickers=101, date_max="2026-01-02",
                 column_stats={"pl": {"p1": 0.0, "p50": 10.5, "p99": 100.0}})
    assert compare_manifests(base, clean) == [], "small, healthy drift must not flag"

    assert any("columns dropped" in v for v in compare_manifests(base, dict(base, columns=["a", "pl"])))
    assert any("rows dropped" in v for v in compare_manifests(base, dict(base, rows=500)))
    assert any("tickers dropped" in v for v in compare_manifests(base, dict(base, tickers=50)))
    assert any("date_max went backwards" in v
               for v in compare_manifests(base, dict(base, date_max="2025-01-01")))
    median_shift = dict(base, column_stats={"pl": {"p1": 0.0, "p50": 80.0, "p99": 100.0}})
    assert any("median drift" in v for v in compare_manifests(base, median_shift))


def main() -> int:
    _self_check_compare_manifests()

    print_header("MANIFEST DRIFT")
    checks: list[tuple[str, bool, str]] = [
        check_market_drift(OUTPUT_PATH, "BR"),
        check_market_drift(US_OUTPUT_PATH, "US"),
        check_us_empty_columns(),
    ]
    checks += check_us_survivorship_coverage()

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
