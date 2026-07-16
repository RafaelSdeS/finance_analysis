"""
Test: config.py's ExperimentConfig round-trips through dict/JSON without
losing or mutating values (docs/EIIE_AGENT_PLAN.md Phase 1).

Run from project root:
    python tests/rl_agent/test_config.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.config import ExperimentConfig  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def main():
    print_header("test_config")
    passed = failed = 0

    default = ExperimentConfig()

    ok = default.data.window == 50 and default.data.features == ("close", "high", "low")
    print_check("defaults: window=50, features=(close,high,low)", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = default.costs.c_sell == 0.0003 and default.costs.c_buy == 0.0003
    print_check("defaults: B3 transaction cost 0.03%", ok, f"got {default.costs.c_sell}")
    passed, failed = passed + ok, failed + (not ok)

    # --- dict round-trip ---
    d = default.to_dict()
    restored = ExperimentConfig.from_dict(d)
    ok = restored == default
    print_check("dict round-trip preserves equality", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = isinstance(restored.data.features, tuple) and isinstance(restored.eval.baselines, tuple)
    print_check("tuple fields stay tuples after from_dict", ok,
                f"features={type(restored.data.features).__name__}, baselines={type(restored.eval.baselines).__name__}")
    passed, failed = passed + ok, failed + (not ok)

    # --- override via dict ---
    d2 = default.to_dict()
    d2["train"]["seed"] = 7
    d2["data"]["window_start"] = "2015-01-01"
    overridden = ExperimentConfig.from_dict(d2)
    ok = overridden.train.seed == 7 and overridden.data.window_start == "2015-01-01"
    print_check("overrides via dict apply correctly", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = overridden.costs.c_sell == default.costs.c_sell
    print_check("unrelated fields unaffected by a partial override", ok)
    passed, failed = passed + ok, failed + (not ok)

    # --- JSON file round-trip (save + from_json) ---
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        default.save(path)
        loaded = ExperimentConfig.from_json(path)
        ok = loaded == default
        print_check("save() -> from_json() round-trip preserves equality", ok)
        passed, failed = passed + ok, failed + (not ok)

        ok = json.loads(path.read_text())["data"]["window"] == 50
        print_check("saved JSON is human-readable and matches defaults", ok)
        passed, failed = passed + ok, failed + (not ok)

    # --- the actual baseline config file loads and matches the dataclass defaults ---
    baseline_path = ROOT / "configs" / "eiie_baseline.json"
    ok = baseline_path.exists()
    print_check("configs/eiie_baseline.json exists", ok, str(baseline_path))
    passed, failed = passed + ok, failed + (not ok)

    if ok:
        baseline_cfg = ExperimentConfig.from_json(baseline_path)
        ok = baseline_cfg == default
        print_check("configs/eiie_baseline.json matches ExperimentConfig defaults exactly", ok)
        passed, failed = passed + ok, failed + (not ok)

    # --- immutability (frozen dataclasses) ---
    try:
        default.train.seed = 99
        ok = False
    except (AttributeError, TypeError):
        ok = True
    print_check("ExperimentConfig sub-configs are frozen (immutable)", ok)
    passed, failed = passed + ok, failed + (not ok)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
