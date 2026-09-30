"""Max-weight b-matching: Bertsekas auction with epsilon-scaling.

Pure function, no fuzzer state (same split as ``core/stable_matching.py``).
Seeds are persons, each target's quota is that many seats; the result
maximizes the summed weight, where Gale-Shapley only finds a stable match.

    seeds   s0 s1 s2          weights[s][t] = value of s on t
    seats   t0 t0 | t1 t1     quotas = [2, 2]: one spare seat
    dummy   d0                -> a dummy seed worth 0 everywhere

The problem is squared (spare seats get dummy seeds, missing seats become
``UNMATCHED`` seats worth 0) so forward auction stays valid with prices
kept across scaling phases. Seats of one target are alike, so a bid scans
targets, not seats: each target keeps a min-heap of its seat prices.

Weights are quantized to ``1/QUANT``; the last phase runs at eps = 1, so
the total is within ``(m + k) / QUANT`` of the optimum (m seats, k seeds):
< 0.01 per seed, below Thompson-draw noise. Scaling by ``m + 1`` for exact
optimality measured 1.9x slower. At 32 seeds, 4-8 targets: ~0.45 ms per
call. Without eps-scaling near-tied weights run bidding wars of seconds;
a numpy per-bid scan was 5x slower (few targets: call overhead dominates);
keeping eps-happy holders across phases saved nothing (wars, not restarts).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Sequence

UNMATCHED = -1

# Weight resolution. Finer costs phases (2^10: 1.5x slower) for precision
# Thompson draws do not have.
QUANT = 1 << 8

# Epsilon shrinks by this factor per scaling phase (4-8 measured fastest).
EPS_STEP = 8

# Seat heap entry: [price, seat, owner].
_OWNER = 2


def auction(weights: Sequence[Sequence[float]], quotas: Sequence[int]) -> list[int]:
    """Max-weight match of seeds to targets under seat quotas.

    Args:
        weights: ``weights[s][t]``; non-finite values count as 0.
        quotas: seats per target; ``<= 0`` admits nobody.

    Returns:
        ``match[s]`` = target index, or ``UNMATCHED`` when seats run out.
    """
    k = len(weights)
    if k == 0:
        return []

    # Seat groups: targets with seats, plus UNMATCHED seats if short.
    groups = [t for t, q in enumerate(quotas) if q > 0]
    sizes = [quotas[t] for t in groups]
    short = k - sum(sizes)
    if short > 0:
        groups.append(UNMATCHED)
        sizes.append(short)

    m = sum(sizes)
    heaps = _bid(_benefits(weights, groups, k, m), sizes, m)

    match = [UNMATCHED] * k
    for g, heap in enumerate(heaps):
        for seat in heap:
            p = seat[_OWNER]
            if p < k:
                match[p] = groups[g]
    return match


def _benefits(
    weights: Sequence[Sequence[float]], groups: list[int], k: int, m: int
) -> list[list[int]]:
    """Integer benefit per (person, group); dummy rows and UNMATCHED are 0."""
    out = [[0] * len(groups) for _ in range(m)]
    for p in range(k):
        row, w = out[p], weights[p]
        for g, t in enumerate(groups):
            if t == UNMATCHED or not math.isfinite(w[t]):
                continue
            row[g] = round(w[t] * QUANT)
    return out


def _bid(benefit: list[list[int]], sizes: list[int], m: int) -> list[list[list[int]]]:
    """Forward auction with eps-scaling; returns per-group seat heaps."""
    prices = [[0] * s for s in sizes]
    span = max(map(max, benefit)) - min(map(min, benefit))
    eps = max(span // EPS_STEP, 1)

    while True:
        heaps = [[[pr, i, -1] for i, pr in enumerate(ps)] for ps in prices]
        for heap in heaps:
            heapq.heapify(heap)
        _phase(benefit, heaps, m, eps)

        if eps == 1:
            return heaps

        # Next phase re-bids everyone at a finer eps, keeping the prices.
        for g, heap in enumerate(heaps):
            for pr, i, _ in heap:
                prices[g][i] = pr
        eps = max(eps // EPS_STEP, 1)


def _phase(benefit: list[list[int]], heaps: list[list[list[int]]], m: int, eps: int) -> None:
    """Gauss-Seidel bidding until every person holds a seat."""
    free = deque(range(m))
    groups = range(len(heaps))
    while free:
        p = free.popleft()
        row = benefit[p]

        # Best and runner-up net value over each group's cheapest seat.
        # Hot loop: literal [0] (price) and range indexing, measured 1.4x
        # faster than enumerate + a module constant.
        best = second = -math.inf
        j = 0
        for g in groups:
            v = row[g] - heaps[g][0][0]
            if v > best:
                best, second, j = v, best, g
            elif v > second:
                second = v

        # The runner-up may be the winning group's next-cheapest seat.
        heap = heaps[j]
        n = len(heap)
        if n > 1:
            nxt = heap[1][0] if n == 2 else min(heap[1][0], heap[2][0])
            if row[j] - nxt > second:
                second = row[j] - nxt
        if second == -math.inf:
            second = best

        # Raise the cheapest seat to the indifference point (+eps), evict its holder.
        top = heap[0]
        if top[2] >= 0:  # owner
            free.append(top[2])
        bump = int(best - second) + eps  # finite: every seat group is non-empty
        heapq.heapreplace(heap, [top[0] + bump, top[1], p])
