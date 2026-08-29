"""Per-seed cost ledger: how much target time each seed has consumed.

``meta["total_time"]`` accumulates target execution time per seed, written in
:meth:`Fuzzer.fuzz_one` below ``t_start`` so it measures target execution and
not mutation.  Every reader of it divides by ``fuzz_count`` to recover a mean
``exec_us``.  Two things make ``fuzz_count`` the wrong denominator:

* ``Fuzzer.run``'s initial seed replay increments ``fuzz_count`` without
  crediting any time, so the ratio is biased low by the number of replays.
* ``fuzz_count`` is persisted in the corpus state and ``total_time`` was not,
  so after ``--resume`` a seed carried a large count against a zero numerator
  and read as the cheapest possible seed for the rest of the campaign.

The fix is a denominator that counts exactly the executions whose time is in
the numerator.  ``cost_samples`` is that counter; it is incremented in lockstep
with ``total_time`` and persisted with it.  A seed with no samples has *no
measurement*, which is a different state from *measured as free* — the readers
substitute the corpus mean rather than the 1 microsecond floor they used to
land on.

:func:`effective_fuzz_count` expresses the ledger back in units of executions:
"how many average-cost executions has this seed consumed".  Under a target
whose per-execution cost does not vary it equals the sample count, so any
consumer written against it is a no-op exactly where the cost signal carries
nothing, and diverges only where it does.  That is the falsification condition
from ``docs/handover/handover_persistence_mechanics_2026-08-29.md`` expressed
in the code instead of in a comment.
"""

from __future__ import annotations

__all__ = [
    "cost_samples",
    "effective_fuzz_count",
    "seed_exec_time",
    "seed_exec_us",
]


def cost_samples(meta: dict) -> int:
    """Number of executions whose time is included in ``meta["total_time"]``.

    Zero means the seed has no cost measurement — either it has never been
    fuzzed, or its metadata was restored from a state file written before the
    ledger was persisted.  Callers must treat that as "unknown", never as
    "free".
    """
    try:
        n = int(meta.get("cost_samples", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def seed_exec_time(meta: dict, fallback: float) -> float:
    """Mean target time per execution for one seed, in seconds.

    Returns ``fallback`` when the seed has no cost samples.  ``fallback`` is
    normally the corpus-wide mean, which is the least-assuming stand-in for a
    seed we have not timed.
    """
    n = cost_samples(meta)
    if n <= 0:
        return fallback
    try:
        total = float(meta.get("total_time", 0.0) or 0.0)
    except (TypeError, ValueError):
        return fallback
    if total <= 0.0:
        return fallback
    return total / n


def seed_exec_us(meta: dict, fallback_us: float) -> float:
    """:func:`seed_exec_time` in microseconds, floored at 1.0.

    The floor matches what the AFL-style cost consumers already applied; it is
    kept so a pathologically fast seed cannot drive ``cost`` to zero and win
    every edge outright.
    """
    fallback_s = max(fallback_us, 1.0) / 1_000_000.0
    return max(1.0, seed_exec_time(meta, fallback_s) * 1_000_000.0)


def effective_fuzz_count(meta: dict, mean_exec_time: float) -> float:
    """Cost consumed by this seed, expressed in average-cost executions.

    Falls back to the raw ``fuzz_count`` when the seed has no samples or the
    corpus has no mean yet, so a fresh or legacy-restored seed keeps its old
    reading instead of dropping to zero and looking untouched.
    """
    n = cost_samples(meta)
    if n <= 0 or mean_exec_time <= 0.0:
        try:
            return float(meta.get("fuzz_count", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    try:
        total = float(meta.get("total_time", 0.0) or 0.0)
    except (TypeError, ValueError):
        return float(n)
    return total / mean_exec_time
