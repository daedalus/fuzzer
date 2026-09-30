"""Many-to-one stable matching: Gale-Shapley deferred acceptance with quotas.

Pure function, no fuzzer state (same split as ``core/fair_queue.py``).
Proposers walk their preference list; each reviewer holds its ``quota``
best-scored proposers so far and bumps the worst when a better one arrives.
The result is stable (no proposer/reviewer pair both prefer each other to
their match) and proposer-optimal.

    proposers  p0 p1 p2        prefs[p]     = reviewers, best first
    reviewers  r0(q=2) r1(q=1) scores[r][p] = r's taste for p, higher wins

O(P * R log Q). The proposal loop is sequential by nature (each step
depends on who is held), so there is nothing to vectorize.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

UNMATCHED = -1


def deferred_acceptance(
    prefs: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    quotas: Sequence[int],
) -> list[int]:
    """Stable match of proposers to reviewers.

    Args:
        prefs: ``prefs[p]`` lists reviewer indices, most preferred first.
            Reviewers left off are unacceptable to ``p``.
        scores: ``scores[r][p]``; a reviewer keeps higher scores. Ties keep
            the lower proposer index, so the result is deterministic.
        quotas: seats per reviewer; ``<= 0`` admits nobody.

    Returns:
        ``match[p]`` = reviewer index, or ``UNMATCHED``.
    """
    n = len(prefs)
    match = [UNMATCHED] * n
    nxt = [0] * n

    # Min-heap per reviewer of (score, -p, p): root is the worst held seat.
    held: list[list[tuple[float, int, int]]] = [[] for _ in quotas]
    free = list(range(n - 1, -1, -1))

    while free:
        p = free.pop()
        row = prefs[p]
        if nxt[p] >= len(row):
            continue  # exhausted its list: stays unmatched

        r = row[nxt[p]]
        nxt[p] += 1
        seat = (scores[r][p], -p, p)
        heap = held[r]

        # Room left: admit.
        if len(heap) < quotas[r]:
            heapq.heappush(heap, seat)
            match[p] = r
            continue

        # Full (or zero quota): bump the worst held only if p beats it.
        if not heap or seat <= heap[0]:
            free.append(p)
            continue
        loser = heapq.heapreplace(heap, seat)[2]
        match[loser] = UNMATCHED
        match[p] = r
        free.append(loser)

    return match
