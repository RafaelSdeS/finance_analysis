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

# Peak working-set cost of one ticker through merge + all of Pass 1.
#
# MEASURED against VmData, not VmHWM (BR, 2026-08-23, by running the real merge
# + all six feature stages over a ticker subset while polling /proc/self/status):
#
#     n=70, 294,443 rows:  peak VmData 3.98 GiB, peak VmRSS 2.07 GiB
#                          baseline 2.41 GiB -> 1.57 GiB attributable
#                          = 24.1 MB/ticker = 5,719 B/row
#
# VmData is the right meter because it is exactly what RLIMIT_DATA caps, and it
# runs ~1.9x peak RSS here: numpy/pandas back large arrays with anonymous mmaps,
# which count against the data segment in full whether or not every page is
# resident. An earlier calibration used VmHWM (RSS), and the build still died on
# MemoryError — it was over the *virtual* ceiling while comfortably under the
# physical one. Do not re-derive these from RSS. (Fragmentation is not the
# story either: an explicit malloc_trim(0) at the same point moved the baseline
# only 2.42 -> 2.39 GiB.)
#
# Per ROW is the stable unit; per-ticker figures move with whichever tickers a
# subset happens to contain. BR: 4,587 rows/ticker x 5.7 kB = ~26MB, rounded up.
#
# The previous 10MB was calibrated against a stale ~3,000 rows/ticker AND
# ignored BASELINE_BYTES below. It didn't merely under-budget — it made BR skip
# chunking ENTIRELY: 10MB/ticker let any ordinary budget ask for >500 tickers,
# MAX_CHUNK clamped that to 500, and BR's whole universe is 383, so every run
# was one batch of everything.
#
# REJECTED, don't re-attempt without measuring: rewriting features.py's six
# per-ticker loops to accumulate only their new columns (instead of appending a
# full-width copy per ticker and concatenating) looks like the obvious way to
# cut this number. It was implemented, verified output-identical on real data
# (159 columns, atol=0), and measured WORSE — 4.45 GiB peak vs 3.98 at n=70,
# 31.1 vs 24.1 MB/ticker. Sorting the frame once up front to feed the loops
# costs a full-frame copy the per-group sorts never made, and assigning a block
# of new columns back forces a consolidation. Reverted.
US_BYTES_PER_TICKER = 20_000_000
BR_BYTES_PER_TICKER = 28_000_000

# Fixed cost held for the WHOLE build regardless of chunk_size, so it has to
# come off the top before the rest is divided into batches. BR keeps prices +
# fundamentals + dividends + company_info in memory across every batch (its
# split-repair/continuity/BOVA11 stages are whole-universe, so batches are
# sliced from RAM rather than reloaded from disk): measured 2.39-2.42 GiB of
# VmData. Note how badly RSS understates this — the same inputs are only
# ~0.6 GiB resident, so an RSS-based reading of this number is 4x too small.
#
# Leaving it out entirely is what killed the run before last: chunk_size_for
# handed the ENTIRE budget to per-ticker cost, so a 4.1 GiB budget picked 246
# tickers and real peak came to ~7 GiB against a 5.2 GiB cap.
#
# US loads each batch's prices/fundamentals from disk instead (see
# build_us_dataset._load_batch_from_disk), so only its small join-side
# reference tables stay resident. NOT independently measured — set well below
# BR's on that structural argument, and low rather than high because
# overshooting here only shrinks batches.
BR_BASELINE_BYTES = 2_600_000_000
US_BASELINE_BYTES = 500_000_000

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


# NOTE if VmData below turns out to ratchet across passes: glibc raises its
# mmap threshold dynamically (up to 32MB) the first time it frees an mmap'd
# block, after which every ~2-30MB column and label array comes from the heap
# arena instead -- and malloc_trim can only madvise those pages away, which
# drops RSS but NOT VmData. `MALLOC_MMAP_THRESHOLD_=131072 python -m ...` pins
# it and needs no code here. Setting it via mallopt() was tried and measured
# no different on a synthetic alloc/free loop, so it is NOT applied by default;
# the per-pass VmData print is what should decide whether it's worth trying.


def vmdata_gb():
    """Current VmData -- the process's data segment + anonymous mappings, i.e.
    exactly what RLIMIT_DATA caps. Not RSS: it runs ~1.9x peak RSS here, and
    it's the number the build actually dies against."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmData:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    except OSError:
        pass
    return 0.0


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


def chunk_size_for(bytes_per_ticker, budget=None, baseline_bytes=0):
    """Tickers per Pass-1 batch that fit the budget, clamped to a row-group
    size that still compresses.

    `baseline_bytes` is the build's fixed resident cost (see BR_BASELINE_BYTES)
    and comes off the top: only what's left after it can be spent on the batch.
    Dividing the whole budget by bytes_per_ticker instead treats the resident
    inputs as free and overshoots by exactly their size — how the BR build
    picked 246 tickers for a 4.1 GiB budget and then peaked at 7.1 GiB.
    """
    budget = budget_gb() if budget is None else budget
    spendable = budget * GB - baseline_bytes
    n = int(spendable // bytes_per_ticker)
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


def report(label, bytes_per_ticker, baseline_bytes=0, n_tickers=None):
    """Apply the limit and print what was decided. Returns the chunk size."""
    budget = budget_gb()
    chunk = chunk_size_for(bytes_per_ticker, budget, baseline_bytes)
    cap = apply_limit(budget)
    peak = (baseline_bytes + chunk * bytes_per_ticker) / GB

    print()
    print("=" * 80)
    print(f"MEMORY BUDGET ({label})")
    print("=" * 80)
    print(f"Available:   {available_gb():.1f} GiB")
    print(f"Budget:      {budget:.1f} GiB  (reserved for the rest of the machine: "
          f"{os.environ.get(ENV_RESERVE, DEFAULT_RESERVE_GB)} GiB — "
          f"override with {ENV_RESERVE} / {ENV_BUDGET})")
    print(f"Baseline:    {baseline_bytes / GB:.1f} GiB resident for the whole build")
    print(f"Hard cap:    {f'{cap:.1f} GiB (RLIMIT_DATA)' if cap else 'not applied'}")
    n_batches = f", {-(-n_tickers // chunk)} batches for {n_tickers} tickers" if n_tickers else ""
    print(f"chunk_size:  {chunk} tickers/batch{n_batches}")
    print(f"Est. peak:   {peak:.1f} GiB  (baseline + chunk x "
          f"{bytes_per_ticker / 1e6:.0f}MB/ticker)")

    # Fail HERE, not 20 minutes into Pass 1. Only reachable via the MIN_CHUNK
    # clamp -- above that floor chunk_size_for fits the budget by construction,
    # so this means the machine genuinely cannot host the build right now (the
    # baseline alone may already exceed the budget). Same information the run
    # would eventually produce as a MemoryError, delivered before any work.
    if cap and peak > cap:
        raise MemoryError(
            f"{label}: projected peak {peak:.1f} GiB exceeds the {cap:.1f} GiB hard cap "
            f"even at the minimum batch size ({MIN_CHUNK} tickers).\n"
            f"  Free memory: {available_gb():.1f} GiB available, "
            f"{baseline_bytes / GB:.1f} GiB of that goes to inputs the build holds throughout.\n"
            f"  Fixes: close something, or lower the reserve "
            f"({ENV_RESERVE}=2 python -m ...), or set {ENV_BUDGET} explicitly, "
            f"or drop the ceiling with {ENV_NO_LIMIT}=1 (risks a kernel OOM kill)."
        )
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

    # baseline_bytes comes off the top: the same budget buys strictly fewer
    # tickers once a fixed resident cost is declared. Ignoring it is what made
    # the BR build pick 246 tickers for a 4.1 GiB budget and then peak at 7.1.
    free = chunk_size_for(BR_BYTES_PER_TICKER, budget=4.0)
    withbase = chunk_size_for(BR_BYTES_PER_TICKER, budget=4.0, baseline_bytes=BR_BASELINE_BYTES)
    assert withbase < free, (free, withbase)
    assert withbase == int((4.0 * GB - BR_BASELINE_BYTES) // BR_BYTES_PER_TICKER)

    # ...and the point of all of it: projected peak must stay under the cap
    # that apply_limit() would install for that same budget.
    for budget in (2.0, 4.0, 6.0, 8.0):
        chunk = chunk_size_for(BR_BYTES_PER_TICKER, budget, BR_BASELINE_BYTES)
        peak = (BR_BASELINE_BYTES + chunk * BR_BYTES_PER_TICKER) / GB
        # MIN_CHUNK can force an overrun on a genuinely tiny budget -- that's
        # the documented "get more RAM" floor, not a sizing error.
        if chunk > MIN_CHUNK:
            assert peak <= budget * 1.25, (budget, chunk, peak)

    print("memory.py self-check OK")


if __name__ == "__main__":
    demo()
