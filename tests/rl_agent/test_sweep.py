"""
Test: sweep.py's bounded-concurrency job runner (TRAINING_SPEEDUP_PLAN.md S5)
-- fake subprocesses only, no real experiment runs.

Run from project root:
    python tests/rl_agent/test_sweep.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.sweep import _clamp_jobs, _job_env, run_jobs  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_clamp_jobs(passed, failed):
    ok = _clamp_jobs(4, None) == 4
    print_check("_clamp_jobs: no clamp when RAM is unreadable (None)", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = _clamp_jobs(4, available_mb=1500, per_job_mb=700) == 2
    print_check("_clamp_jobs: clamps down to what available RAM fits", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = _clamp_jobs(2, available_mb=100, per_job_mb=700) == 1
    print_check("_clamp_jobs: never clamps below 1 even under extreme pressure", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = _clamp_jobs(2, available_mb=100_000, per_job_mb=700) == 2
    print_check("_clamp_jobs: leaves -j alone when RAM is ample", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_job_env(passed, failed):
    env = _job_env(max_parallel=4)
    ok = "OMP_NUM_THREADS" in env and "MKL_NUM_THREADS" in env
    print_check("_job_env: sets OMP/MKL thread-count env vars", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] and int(env["OMP_NUM_THREADS"]) >= 1
    print_check("_job_env: thread count is a positive integer, consistent across both vars", ok,
                env.get("OMP_NUM_THREADS"))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_run_jobs(passed, failed):
    py = sys.executable
    jobs = [
        ("ok_a", [py, "-c", "print('hello a')"]),
        ("ok_b", [py, "-c", "print('hello b')"]),
        ("bad", [py, "-c", "import sys; print('boom'); sys.exit(3)"]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "logs"
        failures = run_jobs(jobs, max_parallel=2, log_dir=log_dir, poll_s=0.05)

        ok = failures == ["bad"]
        print_check("run_jobs: reports exactly the nonzero-exit job as failed", ok, str(failures))
        passed, failed = passed + ok, failed + (not ok)

        ok = all((log_dir / f"{label}.log").exists() for label, _ in jobs)
        print_check("run_jobs: writes one log file per job", ok)
        passed, failed = passed + ok, failed + (not ok)

        ok = "hello a" in (log_dir / "ok_a.log").read_text()
        print_check("run_jobs: job stdout lands in its log file", ok)
        passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_sweep")
    passed = failed = 0
    passed, failed = test_clamp_jobs(passed, failed)
    passed, failed = test_job_env(passed, failed)
    passed, failed = test_run_jobs(passed, failed)
    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
