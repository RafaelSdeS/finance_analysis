"""
Test: conviction_model/config.py's SSLConfig JSON round-trip.

Run from project root:
    python tests/conviction_model/test_config.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.config import SSLConfig  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_json_round_trip_preserves_all_fields(passed, failed):
    cfg = SSLConfig(cpc_horizon=63, batch_size=32, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ssl_config.json"
        cfg.to_json(path)
        loaded = SSLConfig.from_json(path)
    ok = loaded == cfg
    print_check("SSLConfig: to_json/from_json round-trips exactly", ok, f"got {loaded}, expected {cfg}")
    return passed + ok, failed + (not ok)


def test_defaults_are_frozen(passed, failed):
    cfg = SSLConfig()
    try:
        cfg.cpc_horizon = 99
        ok = False
    except AttributeError:
        ok = True
    print_check("SSLConfig: frozen dataclass rejects attribute mutation", ok)
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/config.py")
    passed = failed = 0
    for test_fn in [test_json_round_trip_preserves_all_fields, test_defaults_are_frozen]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
