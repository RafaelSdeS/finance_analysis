"""
sweep.py — run several experiment processes concurrently (TRAINING_SPEEDUP_PLAN.md S5).

The EIIE network is tiny (~3k params, ~0.5 GB GPU per process including its
feature store), so one GPU comfortably hosts several independent runs at once —
seed ensembles and hyperparameter/config sweeps are wall-clock-divided by the
number of parallel jobs. Each job is a plain `python -m src.rl_agent.experiment`
subprocess; its interleaved output goes to a per-job log file, and each run
still produces its own self-contained experiments/{name}_{timestamp}/ dir
(timestamps carry microseconds, so same-second launches can't collide).

Usage:
    # seed ensemble: one config, many seeds, 4 at a time
    python -m src.rl_agent.sweep --config configs/eiie_baseline.json --seeds 1 2 3 4 5 -j 4

    # config sweep: several hyperparameter variants (optionally x seeds)
    python -m src.rl_agent.sweep --config configs/a.json configs/b.json --eval-split val
"""

import argparse
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from .paths import ROOT


def run_jobs(jobs: list, max_parallel: int, log_dir: Path, poll_s: float = 2.0) -> list:
    """jobs: [(label, cmd_list)]. Runs at most max_parallel at once, each with
    stdout+stderr redirected to log_dir/{label}.log. Returns labels that
    exited nonzero. Ctrl-C terminates every child before re-raising."""
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = deque(jobs)
    running, failures = {}, []
    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                label, cmd = queue.popleft()
                log_path = log_dir / f"{label}.log"
                log_f = open(log_path, "w")
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=ROOT)
                print(f"[sweep] started {label} (pid {proc.pid}) -> {log_path}", flush=True)
                running[proc] = (label, log_path, log_f)
            time.sleep(poll_s)
            for proc in [p for p in running if p.poll() is not None]:
                label, log_path, log_f = running.pop(proc)
                log_f.close()
                ok = proc.returncode == 0
                if not ok:
                    failures.append(label)
                print(f"[sweep] {label}: {'OK' if ok else f'FAILED rc={proc.returncode}'}"
                      f" (log: {log_path})", flush=True)
    except KeyboardInterrupt:
        for proc, (label, _, log_f) in running.items():
            proc.terminate()
            log_f.close()
            print(f"[sweep] terminated {label}", flush=True)
        raise
    return failures


def main():
    parser = argparse.ArgumentParser(description="Run experiment configs/seeds concurrently")
    parser.add_argument("--config", nargs="+", required=True, help="one or more ExperimentConfig JSONs")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="run each config once per seed (omit to use each config's own seed)")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    # ponytail: 4 fits a small GPU (~0.5 GB/job); raise if nvidia-smi shows headroom
    parser.add_argument("-j", "--jobs", type=int, default=4, help="max parallel runs (default 4)")
    args = parser.parse_args()

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
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
