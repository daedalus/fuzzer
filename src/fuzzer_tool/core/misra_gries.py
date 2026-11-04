"""Misra–Gries heavy-hitter summary: bounded approximate frequency counts.

Keeps at most *k* counters. A newcomer arriving at a full table is not
stored; every counter is decremented instead and zeros are dropped.

    stream a a b c a d     k=2
    a:1  a:2  a:2,b:1  (c: full -> a:1)  a:2  (d: full -> a:1)

Guarantee: any key seen more than ``n / (k + 1)`` times survives, and a
stored count underestimates the true one by at most ``n / (k + 1)``. With
``k`` at or above the number of distinct keys the counts are exact.

``reduce_counts`` is the batch (mergeable) form, for callers that count a
whole batch with C-level ``Counter.update`` first: subtract the (k+1)-th
largest count from every counter and drop the non-positive. Same guarantee
(Agarwal et al., "Mergeable summaries", 2012), one O(n) pass per batch
instead of one per newcomer.

    counts a:5 b:1 c:3 d:1 e:2   k=2   (k+1)-th largest = 2
    ->     a:3     c:1                 evicted b d e
"""

from collections import deque
from collections.abc import Hashable, Iterator
from itertools import compress

import numpy as np


class MisraGries:
    """Bounded counter over a stream of hashable keys."""

    __slots__ = ("_counts", "_k")

    def __init__(self, k: int):
        self._k = max(1, k)
        self._counts: dict[Hashable, int] = {}

    def add(self, key: Hashable) -> int:
        """Count one occurrence of *key*; 0 when overflow dropped it."""
        counts = self._counts
        if key in counts:
            counts[key] += 1
            return counts[key]

        if len(counts) < self._k:
            counts[key] = 1
            return 1

        # Full: decrement all, rebuild without zeros (one dict pass).
        self._counts = {k: c - 1 for k, c in counts.items() if c > 1}
        return 0

    def get(self, key: Hashable) -> int:
        return self._counts.get(key, 0)

    def items(self):
        return self._counts.items()

    def most_common(self, n: int) -> list[tuple[Hashable, int]]:
        """Top *n* keys by stored count, highest first."""
        return sorted(self._counts.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def __contains__(self, key: Hashable) -> bool:
        return key in self._counts

    def __len__(self) -> int:
        return len(self._counts)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._counts)


def reduce_counts(counts: dict[Hashable, int], k: int) -> list[Hashable]:
    """Shrink *counts* to at most *k* keys in place; return the evicted keys.

    No-op at or under *k* keys. Survivors keep their insertion order.
    """
    n = len(counts)
    if n <= k:
        return []

    keys = list(counts)
    vals = np.fromiter(counts.values(), dtype=np.int64, count=n)
    cut = np.partition(vals, n - k - 1)[n - k - 1]  # (k+1)-th largest

    keep = vals > cut
    evicted = list(compress(keys, (~keep).tolist()))
    # dict.pop is C-level (Counter.__delitem__ is a Python override); values
    # then change in place, so survivors keep their order.
    deque(map(counts.pop, evicted), maxlen=0)
    dict.update(
        counts, zip(compress(keys, keep.tolist()), (vals[keep] - cut).tolist(), strict=True)
    )
    return evicted
