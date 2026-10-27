"""Weighted sampler over a Fenwick (binary indexed) tree: O(log n) update and draw.

``core/seed_picker._cdf_pick`` caches a prefix-sum CDF, which is right while the
weights hold still for a few hundred draws but costs O(n) to rebuild after any
change. A lottery over a corpus that gains seeds all the time wants the other
trade: insert, remove and reweight one key in O(log n), draw in O(log n), no
rebuild. That is this tree (Waldspurger & Weihl 1994, section 4.2 uses a tree
of partial ticket sums for the same reason)::

    weights a=3 b=1 c=2:   tree sums 3 4 2 | draw u*6 -> a for u<0.5, b <2/3, c

Slots are reused: ``remove`` zeroes a key's slot and parks it on a free list,
and the tree is rebuilt compactly once dead slots outnumber live ones. Updates
add a delta, so float error drifts; ``sample`` therefore lands on the nearest
positive slot if rounding puts it on a zero one, and ``rebuild`` recomputes
every partial sum exactly (it runs on compaction, and callers may force it).

Weights that are NaN, infinite or non-positive are rejected by ``set`` (the
key is removed): a zero-ticket key can never win a lottery.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterator

#: Rebuild once dead slots exceed this multiple of live ones (plus slack).
COMPACT_FACTOR = 2
COMPACT_SLACK = 8


class FenwickSampler:
    """Keys with positive weights; ``sample(u)`` picks one with P ~ weight."""

    def __init__(self) -> None:
        self._keys: list[Hashable | None] = []  # slot -> key (None when free)
        self._w: list[float] = []  # slot -> weight (0.0 when free)
        self._tree: list[float] = [0.0]  # 1-based Fenwick array
        self._slot: dict[Hashable, int] = {}
        self._free: list[int] = []

    def __len__(self) -> int:
        return len(self._slot)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._slot

    def keys(self) -> Iterator[Hashable]:
        return iter(self._slot)

    def weight(self, key: Hashable) -> float:
        slot = self._slot.get(key)
        return self._w[slot] if slot is not None else 0.0

    @property
    def total(self) -> float:
        """Sum of live weights, from the tree (O(log n))."""
        return self._prefix(len(self._w))

    def set(self, key: Hashable, weight: float) -> None:
        """Insert *key* or change its weight; a non-positive weight removes it."""
        if not (math.isfinite(weight) and weight > 0.0):
            self.remove(key)
            return
        slot = self._slot.get(key)
        if slot is None:
            slot = self._free.pop() if self._free else self._append_slot()
            self._slot[key] = slot
            self._keys[slot] = key
        self._add(slot, float(weight) - self._w[slot])
        self._w[slot] = float(weight)

    def remove(self, key: Hashable) -> None:
        slot = self._slot.pop(key, None)
        if slot is None:
            return
        self._add(slot, -self._w[slot])
        self._w[slot] = 0.0
        self._keys[slot] = None
        self._free.append(slot)
        if len(self._free) > COMPACT_FACTOR * len(self._slot) + COMPACT_SLACK:
            self._compact()

    def sample(self, u: float):
        """Key whose cumulative-weight interval holds ``u * total``; None if empty.

        *u* is clamped into [0, 1). The descent is O(log n).
        """
        if not self._slot:
            return None
        n = len(self._w)
        u = min(max(float(u), 0.0), math.nextafter(1.0, 0.0))
        rem = u * self._prefix(n)
        pos = 0
        step = 1 << (n.bit_length() - 1)
        while step:
            nxt = pos + step
            if nxt <= n and self._tree[nxt] <= rem:
                pos = nxt
                rem -= self._tree[nxt]
            step >>= 1
        slot = min(pos, n - 1)
        if self._w[slot] <= 0.0:
            slot = self._nearest_live(slot)
        return self._keys[slot]

    def rebuild(self) -> None:
        """Recompute every partial sum from the weights (drops accumulated float drift)."""
        n = len(self._w)
        tree = [0.0] * (n + 1)
        for i, w in enumerate(self._w, start=1):
            tree[i] += w
            parent = i + (i & -i)
            if parent <= n:
                tree[parent] += tree[i]
        self._tree = tree

    # -- internals -----------------------------------------------------------

    def _append_slot(self) -> int:
        slot = len(self._w)
        i = slot + 1
        self._w.append(0.0)
        self._keys.append(None)
        # tree[i] covers (i - lowbit(i), i]; with weight 0 here it is the sum of
        # the children already in the tree.
        self._tree.append(self._prefix(i - 1) - self._prefix(i - (i & -i)))
        return slot

    def _add(self, slot: int, delta: float) -> None:
        n = len(self._w)
        i = slot + 1
        while i <= n:
            self._tree[i] += delta
            i += i & -i

    def _prefix(self, i: int) -> float:
        s = 0.0
        while i > 0:
            s += self._tree[i]
            i -= i & -i
        return s

    def _nearest_live(self, slot: int) -> int:
        n = len(self._w)
        for d in range(1, n):
            for cand in (slot + d, slot - d):
                if 0 <= cand < n and self._w[cand] > 0.0:
                    return cand
        return slot

    def _compact(self) -> None:
        live = [
            (self._keys[s], self._w[s]) for s in range(len(self._w)) if self._keys[s] is not None
        ]
        self._keys = [k for k, _ in live]
        self._w = [w for _, w in live]
        self._slot = {k: i for i, (k, _) in enumerate(live)}
        self._free = []
        self.rebuild()
