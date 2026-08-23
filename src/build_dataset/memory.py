"""memory.py — one place that decides how much RAM a build may use.

These builds run on a workstation that is also doing other things (an editor, a
browser, another notebook). Before this module the builders simply allocated
whatever they needed and let the kernel referee: a full-scale US run was
OOM-killed three separate times (docs/US_DATASET_BUILD_PLAN.md §8.0.1/§8.0.3,
plus 2026-08-16 and 2026-08-23 in Pass 2), and a kernel OOM-kill does not
politely pick the build — it picks whatever the heuristic likes, which is how
a VS Code session died mid-build (§8.0 verification note).

Two things are needed for "never take the machine down", and they're different:

1. A BUDGET the build sizes itself against, so it fits by construction.
   `budget_bytes()` reads the real MemAvailable and hands back what's free
   *minus a reserve left for everything else on the machine*. Callers turn
   that into a chunk size, so the same code adapts to an idle 15GB box and to
   one with 6GB already spoken for, instead of a hardcoded chunk_size=150.

2. A CEILING the kernel enforces, so a mis-estimate can't escalate into a
   machine-wide OOM. `apply_limit()` sets RLIMIT_DATA, which since Linux 4.7
   covers anonymous mmaps — i.e. essentially everything pandas/numpy/pyarrow
   allocate. Crossing it raises MemoryError *inside this process* and nothing
   else on the machine is touched. Failing the build loudly beats killing the
   user's editor, and the traceback names the exact allocation that overran.
   The build also can't silently disappear into the 3GB swap partition and
   thrash for an hour instead.

Both are overridable from the environment (see the constants below) — the
reserve is a calibration knob, not a law of nature: what "enough headroom"
means depends on what else is actually running.
"""

import os
import resource

# Left for the rest of the machine, never handed to the build. Sized so an
# editor + browser + language server survive a full-scale run alongside it.
DEFAULT_RESERVE_GB = 4.0

# Fallback when MemAvailable can't be read (non-Linux). Deliberately small:
# guessing low costs some compression, guessing high costs an OOM.
FALLBACK_AVAILABLE_GB = 4.0

# Peak working-set cost of one ticker in Pass 1, measured on the real builds:
# the merged panel runs ~1.1 kB/row and each of the ~6 feature stages holds a
# transient copy or two, so it's roughly (rows per ticker x 1.1kB x 3). US
# averages ~5,300 rows/ticker -> ~20MB; BR ~3,000 -> ~10MB. Calibration knobs,
# not constants of nature — re-measure if the panel gets much wider.
US_BYTES_PER_TICKER = 20_000_000
BR_BYTES_PER_TICKER = 10_000_000

# chunk_size doubles as the parquet row-group size, so it can't shrink without
# limit: tiny row groups reset dictionary/RLE encoding and wreck compression
# (25 tickers/batch measured at ~4% size reduction vs ~75% for one row group).
# Below MIN_CHUNK the right answer is "get more RAM", not "write a 4x bigger file".
MIN_CHUNK = 25
MAX_CHUNK = 500

ENV_BUDGET = "BUILD_MEM_BUDGET_GB"    # hard override of the computed budget
ENV_RESERVE = "BUILD_MEM_RESERVE_GB"  # how much to leave for the rest of the machine
ENV_NO_LIMIT = "BUILD_MEM_NO_RLIMIT"  # set to disable the RLIMIT_DATA ceiling

GB = 1024 ** 3


def available_gb():
    """Free memory the kernel believes is actually claimable right now.

    MemAvailable, not MemFree: MemFree ignores the reclaimable page cache and
    would read as near-zero on a machine that has simply been up a while
    (this one shows 0GB free / 8GB available), which would starve the build
    for no reason.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    except OSError:
        pass
    return FALLBACK_AVAILABLE_GB


def budget_gb():
    """How many GiB this build may use. Never less than a floor that keeps
    MIN_CHUNK viable — if the machine is genuinely that loaded, the build
    should fail on the rlimit with a clear MemoryError rather than silently
    grind through swap."""
    override = os.environ.get(ENV_BUDGET)
    if override:
        return float(override)

    reserve = float(os.environ.get(ENV_RESERVE, DEFAULT_RESERVE_GB))
    return max(available_gb() - reserve, 1.5)


def chunk_size_for(bytes_per_ticker, budget=None):
    """Tickers per Pass-1 batch that fit the budget, clamped to a row-group
    size that still compresses."""
    budget = budget_gb() if budget is None else budget
    n = int(budget * GB // bytes_per_ticker)
    return max(MIN_CHUNK, min(MAX_CHUNK, n))


def apply_limit(budget=None):
    """Cap this process's allocations so an overrun can never reach the kernel
    OOM killer. Returns the cap in GiB, or None if not applied.

    Headroom over the budget is deliberate: the budget sizes the *steady*
    working set, while the cap only has to catch a genuine runaway. Setting
    them equal would fail builds on ordinary transient spikes.
    """
    if os.environ.get(ENV_NO_LIMIT):
        return None

    budget = budget_gb() if budget is None else budget
    cap = budget * 1.25
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        limit = int(cap * GB)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    except (ValueError, OSError, AttributeError):
        return None  # non-Linux, or a hard limit already below what we asked
    return cap


def report(label, bytes_per_ticker):
    """Apply the limit and print what was decided. Returns the chunk size."""
    budget = budget_gb()
    chunk = chunk_size_for(bytes_per_ticker, budget)
    cap = apply_limit(budget)

    print()
    print("=" * 80)
    print(f"MEMORY BUDGET ({label})")
    print("=" * 80)
    print(f"Available:   {available_gb():.1f} GiB")
    print(f"Budget:      {budget:.1f} GiB  (reserved for the rest of the machine: "
          f"{os.environ.get(ENV_RESERVE, DEFAULT_RESERVE_GB)} GiB — "
          f"override with {ENV_RESERVE} / {ENV_BUDGET})")
    print(f"Hard cap:    {f'{cap:.1f} GiB (RLIMIT_DATA)' if cap else 'not applied'}")
    print(f"chunk_size:  {chunk} tickers/batch")
    return chunk


def demo():
    """Self-check: the knobs actually move the numbers, and the clamps hold."""
    assert available_gb() > 0

    os.environ[ENV_BUDGET] = "8"
    assert abs(budget_gb() - 8.0) < 1e-9
    # 8 GiB / 20MB per ticker = ~429, inside [25, 500]
    assert chunk_size_for(US_BYTES_PER_TICKER) == int(8 * GB // US_BYTES_PER_TICKER)
    # A tiny budget clamps up to MIN_CHUNK rather than producing a 1-ticker row group
    os.environ[ENV_BUDGET] = "0.1"
    assert chunk_size_for(US_BYTES_PER_TICKER) == MIN_CHUNK
    # A huge budget clamps down to MAX_CHUNK
    os.environ[ENV_BUDGET] = "1000"
    assert chunk_size_for(BR_BYTES_PER_TICKER) == MAX_CHUNK
    del os.environ[ENV_BUDGET]

    # Reserve is honoured
    os.environ[ENV_RESERVE] = "0"
    unreserved = budget_gb()
    os.environ[ENV_RESERVE] = "2"
    assert budget_gb() < unreserved
    del os.environ[ENV_RESERVE]

    print("memory.py self-check OK")


if __name__ == "__main__":
    demo()
