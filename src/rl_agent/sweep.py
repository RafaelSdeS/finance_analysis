"""
sweep.py — run several experiment processes concurrently (TRAINING_SPEEDUP_PLAN.md S5).

The EIIE network is tiny (~3k params, ~0.5 GB GPU per process including its
feature store), so one GPU comfortably hosts several independent runs at once —
seed ensembles and hyperparameter/config sweeps are wall-clock-divided by the
number of parallel jobs. Each job is a plain `python -m src.rl_agent.experiment`
subprocess; its interleaved output goes to a per-job log file, and each run
still produces its own self-contained experiments/{name}_{timestamp}_{pid}/
dir (experiment.py appends the PID, not just a microsecond timestamp, since
concurrent same-batch launches were observed colliding on timestamp alone).

Usage:
    # seed ensemble: one config, many seeds, 4 at a time
    python -m src.rl_agent.sweep --config configs/eiie_baseline.json --seeds 1 2 3 4 5 -j 4

    # config sweep: several hyperparameter variants (optionally x seeds)
    python -m src.rl_agent.sweep --config configs/a.json configs/b.json --eval-split val
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from .paths import ROOT

# ponytail: rough per-process floor (python+torch+cuda-context import, plus the
# _PanelStore's GPU-resident tensors mirrored host-side). Raised 2026-07-18 after the E1
# capacity sweep (11 feature channels, conv2_out_channels=64) measured ~2.3-2.4 GB RSS
# per job via /proc/<pid>/status -- the old 700 MB estimate (from the original ~3k-param,
# 2/20-channel network) under-clamped -j and directly caused two VS Code OOM-kills that
# session. Not a tight bound, just enough to stop -j from blindly exceeding available RAM.
EST_RAM_PER_JOB_MB = 2500


def _available_ram_mb() -> float | None:
    """MemAvailable from /proc/meminfo (Linux only). None if unreadable, so callers
    degrade to "don't clamp" rather than block on an unsupported platform."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return None


def _clamp_jobs(requested: int, available_mb: float | None, per_job_mb: float = EST_RAM_PER_JOB_MB) -> int:
    """Pure decision (testable without touching /proc/meminfo): don't let -j promise
    more concurrent jobs than available RAM can plausibly hold. None (unreadable
    /proc/meminfo, e.g. non-Linux) means "don't clamp" rather than block."""
    if available_mb is None:
        return requested
    return min(requested, max(1, int(available_mb // per_job_mb)))


def _job_env(max_parallel: int) -> dict:
    """Divide CPU intra-op threads across concurrent jobs so N torch processes don't
    each default to all cores and thrash each other -- the hot loop is GPU tensor ops,
    so a job only needs a sliver of CPU for host-side dispatch and its one-time panel
    load."""
    threads = max(1, (os.cpu_count() or 1) // max_parallel)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    return env


def run_jobs(jobs: list, max_parallel: int, log_dir: Path, poll_s: float = 2.0,
             retry_failed: bool = True) -> list:
    """jobs: [(label, cmd_list)]. Runs at most max_parallel at once, each with
    stdout+stderr redirected to log_dir/{label}.log. A job that exits nonzero
    is retried once (retry_failed=True, the default) before being counted as
    a real failure -- covers transient flakes like the cuDNN sanity-gate
    nondeterminism or an incidental OOM under load, both observed in
    practice; a job that fails twice is a real failure, not retried further.
    Returns labels that ultimately failed. Ctrl-C terminates every child
    before re-raising."""
    log_dir.mkdir(parents=True, exist_ok=True)
    job_env = _job_env(max_parallel)
    queue = deque(jobs)
    running, failures, retried = {}, [], set()
    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                label, cmd = queue.popleft()
                log_path = log_dir / f"{label}.log"
                log_f = open(log_path, "w")
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=ROOT, env=job_env)
                print(f"[sweep] started {label} (pid {proc.pid}) -> {log_path}", flush=True)
                running[proc] = (label, cmd, log_path, log_f)
            time.sleep(poll_s)
            for proc in [p for p in running if p.poll() is not None]:
                label, cmd, log_path, log_f = running.pop(proc)
                log_f.close()
                ok = proc.returncode == 0
                if not ok and retry_failed and label not in retried:
                    retried.add(label)
                    print(f"[sweep] {label}: FAILED rc={proc.returncode}, retrying once "
                          f"(log: {log_path})", flush=True)
                    queue.append((label, cmd))
                    continue
                if not ok:
                    failures.append(label)
                print(f"[sweep] {label}: {'OK' if ok else f'FAILED rc={proc.returncode}'}"
                      f" (log: {log_path})", flush=True)
    except KeyboardInterrupt:
        for proc, (label, _, _, log_f) in running.items():
            proc.terminate()
            log_f.close()
            print(f"[sweep] terminated {label}", flush=True)
        raise
    return failures


def _find_artifact_dir(log_path: Path) -> Path | None:
    """experiment.py prints 'Experiment complete. Artifacts in <dir>' as its last
    line on success (or '--dry-run: stopping ... Artifacts in <dir>'). Parsing the
    job's own log is simpler and race-free compared to scanning experiments/ for
    the newest matching directory, which is ambiguous when several seeds of the
    same config finish close together."""
    text = log_path.read_text(errors="replace")
    for line in reversed(text.splitlines()):
        if "Artifacts in " in line:
            return Path(line.rsplit("Artifacts in ", 1)[1].strip())
    return None


def _write_sweep_summary(jobs: list, failures: list, log_dir: Path) -> Path:
    """Aggregates each job's metrics_summary.json (if it produced one) into a
    single sweep_summary.json next to the per-job logs -- replaces hand-copying
    per-seed Spearman/cash/entropy/return tables into chat by hand, done three
    times over the course of this investigation."""
    summary = {}
    for label, _ in jobs:
        entry = {"status": "failed" if label in failures else "ok"}
        artifact_dir = _find_artifact_dir(log_dir / f"{label}.log")
        if artifact_dir is not None:
            entry["artifact_dir"] = str(artifact_dir)
            metrics_path = artifact_dir / "metrics_summary.json"
            if metrics_path.exists():
                entry["metrics"] = json.loads(metrics_path.read_text()).get("agent", {})
        summary[label] = entry
    out_path = log_dir / "sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run experiment configs/seeds concurrently")
    parser.add_argument("--config", nargs="+", required=True, help="one or more ExperimentConfig JSONs")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="run each config once per seed (omit to use each config's own seed)")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    # ponytail: 4 fits a GPU with a few GB headroom (~2.5 GB/job, see EST_RAM_PER_JOB_MB);
    # -j is clamped automatically below if available RAM can't fit the request
    parser.add_argument("-j", "--jobs", type=int, default=4, help="max parallel runs (default 4)")
    args = parser.parse_args()

    available_mb = _available_ram_mb()
    clamped = _clamp_jobs(args.jobs, available_mb)
    if clamped < args.jobs:
        print(f"[sweep] WARNING: -j {args.jobs} requested but only {available_mb:.0f} MB RAM "
              f"available (~{EST_RAM_PER_JOB_MB} MB/job est.) -- clamping to -j {clamped}", flush=True)
        args.jobs = clamped

    jobs = []
    for cfg_path in args.config:
        for seed in (args.seeds or [None]):
            label = Path(cfg_path).stem + (f"_seed{seed}" if seed is not None else "")
            cmd = [sys.executable, "-m", "src.rl_agent.experiment",
                   "--config", cfg_path, "--eval-split", args.eval_split]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            jobs.append((label, cmd))

    log_dir = ROOT / "experiments" / "sweep_logs" / f"{datetime.now():%Y%m%dT%H%M%S}"
    print(f"[sweep] {len(jobs)} job(s), {args.jobs} at a time; logs in {log_dir}")
    failures = run_jobs(jobs, args.jobs, log_dir)
    print(f"[sweep] done: {len(jobs) - len(failures)}/{len(jobs)} OK"
          + (f", FAILED: {', '.join(failures)}" if failures else ""))
    summary_path = _write_sweep_summary(jobs, failures, log_dir)
    print(f"[sweep] summary written to {summary_path}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
