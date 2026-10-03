"""Compression-based novelty on path traces (handover 7.4).

An edge *set* is order-blind. Two executions can hit identical edges in a
different order (a different state-machine walk) with no new edge, which the
map cannot see. This module scores that.

* ``lz76_complexity``: Lempel-Ziv 1976 phrase count (Kaspar-Schuster), plus a
  normalised form.
* ``ncd``: normalised compression distance via zlib on packed edge ids.
* ``PathNovelty``: keeps a bounded set of reference traces; ``order_novelty``
  is the minimum NCD to any reference with the **same edge set**, so a nonzero
  value means "new ordering, no new edge". ``novelty`` is the plain minimum
  NCD to any reference.

Pure stdlib, deterministic. Not wired; candidate admission signal next to
``--pool-drift`` once calibrated on a real clang build.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections import OrderedDict
from collections.abc import Hashable, Sequence


def lz76_complexity(seq: Sequence[Hashable]) -> int:
    """Number of distinct phrases in the LZ76 parsing (Kaspar-Schuster)."""
    n = len(seq)
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, k, kmax, l, c = 0, 1, 1, 1, 1
    while True:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            kmax = max(k, kmax)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return c


def lz76_normalised(seq: Sequence[Hashable]) -> float:
    """``c * log_a(n) / n`` with ``a`` the alphabet size; ~1 for random."""
    n = len(seq)
    a = len(set(seq))
    if n < 2 or a < 2:
        return 0.0
    return lz76_complexity(seq) * math.log(n, a) / n


def _pack(trace: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(trace)}I", *(e & 0xFFFFFFFF for e in trace))


def _c(b: bytes) -> int:
    return len(zlib.compress(b, 9))


def ncd(a: Sequence[int], b: Sequence[int]) -> float:
    """Normalised compression distance in [0, ~1]; 0 for identical traces."""
    pa, pb = _pack(a), _pack(b)
    ca, cb = _c(pa), _c(pb)
    lo, hi = min(ca, cb), max(ca, cb)
    if hi == 0:
        return 0.0
    return max(0.0, (_c(pa + pb) - lo) / hi)


class PathNovelty:
    """Bounded reference set of ordered edge traces."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(1, capacity)
        self._refs: OrderedDict[int, tuple[tuple[int, ...], frozenset[int]]] = (
            OrderedDict()
        )

    def __len__(self) -> int:
        return len(self._refs)

    def observe(self, trace: Sequence[int]) -> bool:
        """Add a trace; returns False if the identical ordering is held."""
        t = tuple(trace)
        key = hash(t)
        if key in self._refs:
            self._refs.move_to_end(key)
            return False
        self._refs[key] = (t, frozenset(t))
        while len(self._refs) > self.capacity:
            self._refs.popitem(last=False)
        return True

    def novelty(self, trace: Sequence[int]) -> float:
        """Min NCD to any reference; 1.0 when there are none."""
        if not self._refs:
            return 1.0
        return min(ncd(trace, t) for t, _ in self._refs.values())

    def order_novelty(self, trace: Sequence[int]) -> float | None:
        """Min NCD to references with the same edge set.

        ``None`` if no reference shares the set (the map already sees this as
        different, so ordering adds nothing); ``0.0`` if the ordering is held.
        """
        s = frozenset(trace)
        same = [t for t, fs in self._refs.values() if fs == s]
        if not same:
            return None
        return min(ncd(trace, t) for t in same)
