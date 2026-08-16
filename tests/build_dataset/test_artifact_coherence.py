# test_artifact_coherence.py
#
# Checks that the processed-artifact set (manifest, split_config, versioned
# snapshot, terminal_events) actually describes ONE consistent build, not a
# mix of files left over from different runs. None of these are caught by
# test_final_dataset.py (which only looks at the parquet's own contents) --
# see docs/DATA_INTEGRITY_TEST_PLAN.md P0-b.
#
# Feature scaling is deliberately out of scope here: nothing in src/portfolio/
# consumes feature_scaler.joblib (LightGBM is scale-invariant, see
# docs/PORTFOLIO_IMPLEMENTATION_PLAN.md pothole P4), so scale_features.py's
# output is not a build artifact this test needs to track.
#
# Run from project root:
#   python tests/build_dataset/test_artifact_coherence.py

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.build_dataset.paths import (  # noqa: E402
    OUTPUT_PATH, SPLIT_CONFIG_PATH, TERMINAL_EVENTS_PATH,
    US_OUTPUT_PATH, US_SPLIT_CONFIG_PATH,
)
from test_utils import print_header, print_check, print_separator  # noqa: E402


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")


def _latest_snapshot(output_path: Path) -> Path | None:
    """Highest data/processed/dataset_v{N}/ next to output_path, or None.

    BR-only in practice today (2026-08-16): no us_dataset_v{N}/ directory
    exists yet, sync_dataset_version() is only ever called with BR's
    OUTPUT_PATH -- so this returns None for the US output_path, which the
    caller must treat as "not applicable", not as a failure.
    """
    import re
    candidates = sorted(
        (int(m.group(1)), p)
        for p in output_path.parent.glob("dataset_v*")
        if (m := re.fullmatch(r"dataset_v(\d+)", p.name))
    )
    return candidates[-1][1] if candidates else None


def check_split_manifest_agree(output_path: Path, split_config_path: Path, label: str) -> list[tuple[str, bool, str]]:
    checks = []
    manifest_path = _manifest_path(output_path)
    if not manifest_path.exists() or not split_config_path.exists():
        checks.append((f"[{label}] manifest + split_config present", False,
                        f"missing: {manifest_path if not manifest_path.exists() else split_config_path}"))
        return checks

    manifest = json.loads(manifest_path.read_text())
    split_config = json.loads(split_config_path.read_text())
    same_build = manifest.get("built_at") == split_config.get("built_at")
    checks.append((
        f"[{label}] split_config.built_at == manifest.built_at", same_build,
        "" if same_build else f"split={split_config.get('built_at')} manifest={manifest.get('built_at')}",
    ))
    return checks


def check_snapshot_matches(output_path: Path, label: str) -> tuple[str, bool, str]:
    manifest_path = _manifest_path(output_path)
    snapshot_dir = _latest_snapshot(output_path)
    if snapshot_dir is None:
        return (f"[{label}] latest dataset_v{{N}} matches current manifest", True,
                "no snapshot directory exists -- not applicable")

    current = json.loads(manifest_path.read_text())
    snap_manifest_path = snapshot_dir / "ml_dataset.manifest.json"
    if not snap_manifest_path.exists():
        return (f"[{label}] latest dataset_v{{N}} matches current manifest", False,
                f"{snapshot_dir} has no ml_dataset.manifest.json")

    snap = json.loads(snap_manifest_path.read_text())
    fingerprint_keys = ("rows", "tickers", "date_min", "date_max", "columns", "column_stats")
    matches = all(current.get(k) == snap.get(k) for k in fingerprint_keys)
    detail = "" if matches else f"{snapshot_dir.name} is stale vs current build -- sync_dataset_version() didn't run"
    return (f"[{label}] latest {snapshot_dir.name} matches current manifest", matches, detail)


def check_terminal_events_not_stale() -> tuple[str, bool, str]:
    label = "terminal_events.parquet not older than the dataset build"
    if not TERMINAL_EVENTS_PATH.exists():
        return (label, False, f"missing: {TERMINAL_EVENTS_PATH} -- "
                "forward_excess_return() silently no-ops without it (terminal_events=None)")
    if not OUTPUT_PATH.exists():
        return (label, False, f"missing: {OUTPUT_PATH}")
    # terminal_events.parquet has no built_at column of its own (schema is
    # [ticker, delist_date, event_type, terminal_payoff]) -- mtime is the
    # only handle available for "was this regenerated against the current
    # panel or is it left over from an older build."
    stale = TERMINAL_EVENTS_PATH.stat().st_mtime < OUTPUT_PATH.stat().st_mtime
    detail = "" if not stale else "mtime predates ml_dataset.parquet -- rerun terminal_events after the last rebuild"
    return (label, not stale, detail)


def main() -> int:
    print_header("ARTIFACT COHERENCE")
    checks: list[tuple[str, bool, str]] = []

    checks += check_split_manifest_agree(OUTPUT_PATH, SPLIT_CONFIG_PATH, "BR")
    checks.append(check_snapshot_matches(OUTPUT_PATH, "BR"))
    checks.append(check_terminal_events_not_stale())

    if US_OUTPUT_PATH.exists():
        checks += check_split_manifest_agree(US_OUTPUT_PATH, US_SPLIT_CONFIG_PATH, "US")
        # No US snapshot check: sync_dataset_version() has never been called
        # with a US path and no us_dataset_v{N}/ directory exists (2026-08-16,
        # see docs/DATA_INTEGRITY_TEST_PLAN.md). Nothing to assert yet.
        # No US terminal_events check either -- the concept doesn't exist on
        # that side (US collection has no delisted-recovery/continuity path).
    else:
        print("  (US dataset not built -- skipping US checks)")

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
