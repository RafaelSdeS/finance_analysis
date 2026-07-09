#!/usr/bin/env python
"""Throughput + memory benchmark: vec-env backend / device / batch_size / n_envs.

Each variant runs as its own subprocess (OOM-isolated — a killed variant is recorded
and skipped, not fatal to the run) that imports trainer.train() directly (never the
CLI, so scratch runs never touch artifacts/models/agent_{best,final}.zip). Prints each
variant's row as it completes, then a final table sorted by fps.

Measures:
  - fps: timesteps per second across the rollout+train loop.
  - peak_rss_mb: maximum resident set size (variant's process tree) during the run.
  - swap_delta_mb: change in system swap used (isolates this variant's contribution).
  - status: "ok", "OOM (signal N)", "FAILED ...", or "SKIPPED ...".

Usage: python tools/bench_training_speed.py

Note: variants run sequentially in separate OS processes. Full run ~15–30 min depending
on hardware and which variants thrash or OOM.
"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.config import DEFAULT_CONFIG
from src.agent.trainer import train

SCRATCH = Path("/tmp/bench_training_speed")
TOTAL_TIMESTEPS = 65_536

VARIANTS = [
    dict(label="dummy_cuda_8_64", device="cuda", n_envs=8, batch_size=64, use_subprocess=False),
    dict(label="dummy_cuda_8_512", device="cuda", n_envs=8, batch_size=512, use_subprocess=False),
    dict(label="dummy_cuda_16_512", device="cuda", n_envs=16, batch_size=512, use_subprocess=False),
    dict(label="dummy_cpu_8_64", device="cpu", n_envs=8, batch_size=64, use_subprocess=False),
    dict(label="dummy_cpu_16_64", device="cpu", n_envs=16, batch_size=64, use_subprocess=False),
    dict(label="subproc_cuda_12_512", device="cuda", n_envs=12, batch_size=512, use_subprocess=True),
    dict(label="subproc_cuda_16_512", device="cuda", n_envs=16, batch_size=512, use_subprocess=True),
    dict(label="subproc_cpu_12_512", device="cpu", n_envs=12, batch_size=512, use_subprocess=True),
    dict(label="subproc_cpu_16_512", device="cpu", n_envs=16, batch_size=512, use_subprocess=True), 
]


def _get_swap_info() -> tuple[int, int]:
    """Return (swap_total_mb, swap_free_mb) from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            lines = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    lines[key] = int(parts[1])
        return lines.get("SwapTotal", 0) // 1024, lines.get("SwapFree", 0) // 1024
    except Exception:
        return 0, 0


def _get_process_tree_rss(pid: int) -> int:
    """Return total RSS in MB for pid and all descendants."""
    try:
        lines = subprocess.check_output(["ps", "-eo", "pid,ppid,rss"], text=True).strip().split("\n")
        processes = {}
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    processes[int(parts[0])] = (int(parts[1]), int(parts[2]))
                except ValueError:
                    pass

        # Walk descendants of pid
        def rss_sum(p):
            _, rss = processes.get(p, (None, 0))
            total = rss
            for child_pid, (ppid, _) in processes.items():
                if ppid == p:
                    total += rss_sum(child_pid)
            return total

        return rss_sum(pid) // 1024
    except Exception:
        return 0


def poll_memory(stop_event: threading.Event, result_dict: dict) -> None:
    """Poll RSS + swap every ~1s until stop_event is set. Store (peak_rss_mb, peak_swap_delta_mb) in result_dict."""
    pid = os.getpid()
    swap_total, swap_free_start = _get_swap_info()
    swap_used_start = swap_total - swap_free_start

    peak_rss = 0
    peak_swap_delta = 0

    while not stop_event.is_set():
        rss = _get_process_tree_rss(pid)
        peak_rss = max(peak_rss, rss)

        _, swap_free = _get_swap_info()
        swap_used = swap_total - swap_free
        swap_delta = swap_used - swap_used_start
        peak_swap_delta = max(peak_swap_delta, swap_delta)

        time.sleep(1.0)

    result_dict["peak_rss_mb"] = peak_rss
    result_dict["peak_swap_delta_mb"] = peak_swap_delta


def run_one_variant(variant: dict) -> None:
    """Run a single variant: config + train + measure. Print JSON result line to stdout."""
    label = variant.pop("label")
    use_subprocess = variant.pop("use_subprocess")

    # Guard: check available memory before launching
    _, swap_free = _get_swap_info()
    with open("/proc/meminfo") as f:
        lines = {}
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(':')
                lines[key] = int(parts[1])
    mem_available = lines.get("MemAvailable", 0) // 1024  # MB
    n_envs = variant.get("n_envs", 8)
    required_mb = n_envs * 500  # conservative estimate per worker

    if required_mb > mem_available and use_subprocess:
        result = {
            "label": label,
            "status": f"SKIPPED (need {required_mb}MB, have {mem_available}MB)",
            "fps": None,
            "peak_rss_mb": None,
            "swap_delta_mb": None,
        }
        print(json.dumps(result, sort_keys=True))
        return

    # Build config (variant dict now has only config fields, not train() kwargs)
    variant_dir = SCRATCH / label
    variant_dir.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(
        DEFAULT_CONFIG,
        universe_size=50,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=10_000,  # threshold >> total_timesteps -> no val eval fires
        model_dir=variant_dir,
        log_dir=variant_dir,
        **variant,
    )

    # Start memory poller
    stop_event = threading.Event()
    memory_result = {}
    poll_thread = threading.Thread(
        target=poll_memory,
        args=(stop_event, memory_result),
        daemon=True,
    )
    poll_thread.start()

    # Train and time
    t0 = time.time()
    try:
        train(cfg, model_tag="bench", use_subprocess=use_subprocess)
        elapsed = time.time() - t0
        fps = TOTAL_TIMESTEPS / elapsed if elapsed > 0 else 0
        status = "ok"
    except Exception as e:
        elapsed = time.time() - t0
        fps = None
        status = f"FAILED ({type(e).__name__})"
    finally:
        stop_event.set()
        poll_thread.join(timeout=2)

    result = {
        "label": label,
        "status": status,
        "fps": round(fps, 0) if fps else None,
        "peak_rss_mb": memory_result.get("peak_rss_mb"),
        "swap_delta_mb": memory_result.get("peak_swap_delta_mb"),
    }
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    """Orchestrate all variants: subprocess per variant, OOM-isolation."""
    # Check if we're running as a variant subprocess
    if "--variant" in sys.argv:
        idx = sys.argv.index("--variant")
        variant_json = sys.argv[idx + 1]
        variant = json.loads(variant_json)
        run_one_variant(variant)
        return

    # Orchestrator: run each variant as a subprocess
    results = []
    for v in VARIANTS:
        label = v["label"]
        print(f"\n[{len(results) + 1}/{len(VARIANTS)}] Running {label}...", file=sys.stderr, flush=True)

        proc = subprocess.run(
            [sys.executable, __file__, "--variant", json.dumps(v)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per variant, safety valve
        )

        # Parse result
        try:
            if proc.returncode < 0:
                # Killed by signal (e.g. OOM = -9)
                row = {
                    "label": label,
                    "status": f"OOM (signal {-proc.returncode})",
                    "fps": None,
                    "peak_rss_mb": None,
                    "swap_delta_mb": None,
                }
            elif proc.returncode != 0:
                # Non-zero exit (exception inside child)
                stderr_tail = proc.stderr.strip().split('\n')[-3:] if proc.stderr else []
                row = {
                    "label": label,
                    "status": f"FAILED (returncode {proc.returncode})",
                    "stderr": " | ".join(stderr_tail),
                    "fps": None,
                    "peak_rss_mb": None,
                    "swap_delta_mb": None,
                }
            else:
                # Success: parse JSON from last line of stdout
                lines = proc.stdout.strip().split('\n')
                last_json = next((l for l in reversed(lines) if l.startswith('{')), None)
                row = json.loads(last_json) if last_json else {"label": label, "status": "FAILED (no JSON)"}
        except Exception as e:
            row = {"label": label, "status": f"FAILED (parse error: {e})"}

        print(json.dumps(row, sort_keys=True))
        results.append(row)

    # Summary table
    print("\n" + "=" * 100, file=sys.stderr)
    print("Summary (sorted by fps):", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    # Filter successful runs for sorting
    successful = [r for r in results if r.get("fps") is not None]
    successful.sort(key=lambda r: r["fps"], reverse=True)

    for r in successful:
        fps = r.get("fps", "N/A")
        rss = r.get("peak_rss_mb", "N/A")
        swap = r.get("swap_delta_mb", "N/A")
        thrashed = "THRASHED" if (r.get("swap_delta_mb", 0) or 0) > 50 else ""
        print(
            f"{r['label']:30s} {fps:10,.0f} steps/s  {rss:6}MB RSS  {swap:6}MB swap {thrashed}",
            file=sys.stderr,
        )

    if any(r.get("fps") is None for r in results):
        print("\nFailed/skipped runs:", file=sys.stderr)
        for r in results:
            if r.get("fps") is None:
                print(f"  {r['label']:30s} {r.get('status', 'unknown')}", file=sys.stderr)

    # Cleanup
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"\nCleaned up {SCRATCH}", file=sys.stderr)


if __name__ == "__main__":
    main()
