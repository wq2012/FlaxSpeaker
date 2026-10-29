"""Polite CPU, thread-pool, and process-priority resource limiter for shared machines."""

import os
from typing import Any


def configure_polite_resources(
    max_threads: int = 32,
    nice_level: int = 10,
) -> dict[str, Any]:
    """Configures polite CPU thread caps, XLA parallelism, niceness, and CPU affinity.

    Safe and idempotent to call multiple times in training, sharding, or evaluation
    entry points so no single process monopolizes a shared multi-core host.

    Args:
        max_threads: Maximum number of threads / logical vCPUs to use (clamped >= 1).
        nice_level: Target POSIX niceness value (only increases niceness up to target).

    Returns:
        Dictionary summarizing the applied thread limits, niceness, XLA flags, and
        pinned logical CPU IDs.
    """
    max_t = max(1, int(max_threads))
    target_nice = int(nice_level)

    applied_nice = 0
    if hasattr(os, "nice"):
        try:
            current_nice = os.nice(0)
            delta = max(0, target_nice - current_nice)
            applied_nice = os.nice(delta) if delta > 0 else current_nice
        except OSError:
            applied_nice = 0

    t_str = str(max_t)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = t_str

    existing_xla = os.environ.get("XLA_FLAGS", "").split()
    cleaned_xla = [
        tok
        for tok in existing_xla
        if not tok.startswith("--xla_cpu_multi_thread_eigen")
        and not tok.startswith("intra_op_parallelism_threads")
        and not tok.startswith("--xla_force_host_platform_device_count")
    ]
    cleaned_xla.append("--xla_cpu_multi_thread_eigen=true")
    cleaned_xla.append(f"intra_op_parallelism_threads={max_t}")
    xla_flags = " ".join(cleaned_xla)
    os.environ["XLA_FLAGS"] = xla_flags

    avail_cpus: list[int] = []
    pinned_cpus: list[int] = []
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        try:
            avail_cpus = sorted(os.sched_getaffinity(0))
            if len(avail_cpus) > max_t:
                pinned_cpus = avail_cpus[:max_t]
                os.sched_setaffinity(0, set(pinned_cpus))
            else:
                pinned_cpus = list(avail_cpus)
        except OSError:
            pinned_cpus = list(avail_cpus)

    return {
        "max_threads": max_t,
        "nice_level": applied_nice,
        "cpu_count_available": len(avail_cpus) if avail_cpus else (os.cpu_count() or 1),
        "pinned_cpus": pinned_cpus,
        "xla_flags": xla_flags,
    }
