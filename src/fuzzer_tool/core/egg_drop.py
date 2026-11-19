"""Egg-drop threshold probe: find where a monotone predicate starts failing.

Some limits cost a failure to observe: the input size or repeat count at
which a target OOMs or times out. Each failing probe burns a full timeout
(or an OOM kill), so the budget is *failures*, not probes. With k allowed
failures and m probes the largest searchable range is

    f(k, m) = f(k-1, m-1) + f(k, m-1) + 1 = sum_{i=1..k} C(m, i)

(probe at offset f(k-1, m-1): a failure leaves k-1 failures for the floors
below, a pass leaves k for the floors above). Probing that offset at every
step finds the threshold with the minimal worst-case probe count, never
exceeding k failures.

    k=1  linear scan           n probes
    k=2  ~sqrt(2n) probes      100 floors -> 14
    k>=log2(n+1)  bisection    log2(n+1) probes
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of :func:`find_threshold`.

    Attributes:
        threshold: Smallest failing value, or None if none failed.
        probes: Predicate calls made.
        failures: Calls that failed (<= the egg budget).
    """

    threshold: int | None
    probes: int
    failures: int


def reach(eggs: int, drops: int) -> int:
    """Floors f(eggs, drops) searchable with *eggs* failures and *drops* probes.

    Exact (Python ints): sum_{i=1..min(eggs, drops)} C(drops, i), O(eggs).
    A numpy cumsum table over m (one row per egg) was measured slower for
    k <= 2 at every m and only won for k >= 3, m >= 256 (> 2.7M floors), at
    O(k*m) memory; planning cost is microseconds against timeout-long execs.
    """
    total = 0
    term = 1
    for i in range(1, min(eggs, drops) + 1):
        term = term * (drops - i + 1) // i  # C(drops, i)
        total += term
    return total


def min_drops(eggs: int, floors: int) -> int:
    """Fewest probes that always pin the threshold among *floors* values.

    Exponential then binary search on m, since f(eggs, m) grows with m.
    """
    if floors <= 0:
        return 0

    hi = 1
    while reach(eggs, hi) < floors:
        hi *= 2
    lo = hi // 2  # reach(eggs, lo) < floors, or lo == 0

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if reach(eggs, mid) >= floors:
            hi = mid
        else:
            lo = mid
    return hi


def find_threshold(fails: Callable[[int], bool], lo: int, hi: int, eggs: int) -> ProbeResult:
    """Smallest x in [lo, hi] with ``fails(x)``, spending <= *eggs* failures.

    Assumes *fails* is monotone (False ... False True ... True). Probe count
    is at most :func:`min_drops` (eggs, hi - lo + 1), which is optimal. A
    non-monotone predicate still respects both bounds; only the answer is
    then meaningless.

    Raises:
        ValueError: ``eggs < 1`` or ``hi < lo``.
    """
    if eggs < 1:
        raise ValueError(f"need >= 1 allowed failure, got {eggs}")
    if hi < lo:
        raise ValueError(f"empty range [{lo}, {hi}]")

    k = eggs
    m = min_drops(eggs, hi - lo + 1)
    probes = failures = 0
    last_fail: int | None = None

    # Invariant: reach(k, m) >= hi - lo + 1; the threshold is in [lo, hi + 1].
    while lo <= hi:
        x = min(lo + reach(k - 1, m - 1), hi)
        probes += 1
        m -= 1
        if fails(x):
            failures += 1
            k -= 1
            hi = x - 1
            last_fail = x
            continue
        lo = x + 1

    return ProbeResult(last_fail, probes, failures)
