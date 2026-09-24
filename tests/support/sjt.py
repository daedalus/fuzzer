"""Steinhaus-Johnson-Trotter brute-force oracle for single-machine sequencing.

SJT orders all n! permutations so consecutive ones differ by one adjacent
swap. Swapping jobs a, b at positions (i, i+1) moves only their completion
times: b now ends at C[i-1] + p_b, a at the old C[i+1]. So each step
recosts two jobs, and precedence feasibility is an O(1) counter update::

    ... x | a  b | y ...   ->   ... x | b  a | y ...
            C_i C_i+1                  C_i' C_i+1 (unchanged)

Oracle for tests only (FINDINGS P4-14): n <= 8 keeps n! tractable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from itertools import accumulate

from fuzzer_tool.core.job_scheduling import Job

LEFT = -1


def sjt_swaps(n: int) -> Iterator[int]:
    """Yield i for each adjacent swap (i, i+1) of an SJT walk (Even's rule).

    Starting from the identity, the n! - 1 swaps visit every permutation
    once. E.g. n=3: 012 -> 021 -> 201 -> 210 -> 120 -> 102.
    """
    perm = list(range(n))
    pos = list(range(n))
    dirs = [LEFT] * n
    while True:
        # Largest mobile value: its neighbour in its direction is smaller.
        mobile = -1
        for v in range(n - 1, -1, -1):
            q = pos[v] + dirs[v]
            if 0 <= q < n and perm[q] < v:
                mobile = v
                break
        if mobile < 0:
            return

        p = pos[mobile]
        q = p + dirs[mobile]
        other = perm[q]
        perm[p], perm[q] = other, mobile
        pos[mobile], pos[other] = q, p
        yield min(p, q)

        for v in range(mobile + 1, n):
            dirs[v] = -dirs[v]


def _violations(seq: list[Job], prec: dict[object, set[object]]) -> int:
    """Count direct precedence pairs (pred, job) with pred placed after job."""
    where = {j.id: i for i, j in enumerate(seq)}
    return sum(1 for i, j in enumerate(seq) for p in prec.get(j.id, ()) if where[p] > i)


def sjt_min_fmax(
    jobs: list[Job],
    prec: dict[object, set[object]],
    cost: Callable[[Job, float], float],
) -> tuple[float, list[Job]]:
    """Exact min over feasible orders of max_j cost(j, C_j), by SJT walk.

    Returns ``(best, order)``; ``(inf, [])`` if no order respects *prec*,
    ``(-inf, [])`` for no jobs.
    """
    n = len(jobs)
    if n == 0:
        return -math.inf, []

    seq = list(jobs)
    comp = list(accumulate(float(j.processing_time) for j in seq))
    costs = [cost(j, c) for j, c in zip(seq, comp, strict=True)]
    bad = _violations(seq, prec)

    best, order = math.inf, []
    if bad == 0:
        best, order = max(costs), list(seq)

    for i in sjt_swaps(n):
        a, b = seq[i], seq[i + 1]
        seq[i], seq[i + 1] = b, a

        # Only the swapped pair's relative order changed.
        bad += a.id in prec.get(b.id, ())
        bad -= b.id in prec.get(a.id, ())

        comp[i] = (comp[i - 1] if i else 0.0) + b.processing_time
        costs[i] = cost(b, comp[i])
        costs[i + 1] = cost(a, comp[i + 1])
        if bad:
            continue

        value = max(costs)
        if value < best:
            best, order = value, list(seq)
    return best, order
