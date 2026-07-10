"""
Data-group tests for rolling_eval.py (needs the real dataset — AgentConfig validates
dataset_path.exists() at construction — run via `python tests/run_all.py --group data`).

Wraps the module's own self-check functions (previously dead code: only reachable via
`python -m src.agent.rolling_eval` with no CLI mode, never called from any test runner) so
they run automatically. This is exactly the gap that let a real bug ship: a smoke training
run with --no-promote silently overwrote artifacts/models/rolling_eval_results.json (fixed
2026-07-10, see _selfcheck_no_promote_protects_rolling_eval_results) because no test ever
exercised the promote=False path for that file, only for save_walkforward's paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.rolling_eval import (
    _selfcheck_no_promote_protects_rolling_eval_results,
    _selfcheck_online,
    _selfcheck_online_checkpoint,
    _selfcheck_promote_to_production,
    _selfcheck_stitch,
)


def main():
    _selfcheck_online()
    _selfcheck_stitch()
    _selfcheck_online_checkpoint()
    _selfcheck_promote_to_production()
    _selfcheck_no_promote_protects_rolling_eval_results()
    print("\n✓ All rolling_eval promote/selfcheck tests passed")


if __name__ == "__main__":
    main()
