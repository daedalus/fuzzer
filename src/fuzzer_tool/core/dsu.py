"""Disjoint-set forest (union-find) over ``0..n-1``: union by size + path halving.

One implementation for every "which items end up connected" question:
crash clustering, LSH seed clusters, token chains. Amortized O(alpha(n)).

    union(0, 1); union(2, 3); union(1, 3)
    parent  0 0 2 0      groups() -> [[0, 1, 2, 3]]
    size    4 . . .      (sizes are kept on roots only)
"""

from __future__ import annotations

import numpy as np

# labels() crossover: below this a Python find() loop beats numpy pointer
# jumping (n=100: 9.4 vs 10.2 us; n=1000: 114 vs 59 us).
NP_LABELS_MIN = 128


class DisjointSet:
    """Disjoint-set forest over ``0..n-1``."""

    __slots__ = ("_parent", "_size")

    def __init__(self, n: int = 0) -> None:
        # Lists, not array('i'): 1.46x faster find()/union() up to 10k
        # elements; array only wins near 200k.
        self._parent = list(range(n))
        self._size = [1] * n

    def __len__(self) -> int:
        return len(self._parent)

    def find(self, x: int) -> int:
        """Root of *x*; halves the walked path."""
        parent = self._parent
        while parent[x] != x:
            parent[x] = x = parent[parent[x]]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of *a* and *b*; True when they were apart."""
        # find() inlined twice: two method calls cost more than the walks.
        parent = self._parent
        while parent[a] != a:
            parent[a] = a = parent[parent[a]]
        while parent[b] != b:
            parent[b] = b = parent[parent[b]]
        if a == b:
            return False

        # Smaller tree under the larger root keeps depth O(log n).
        size = self._size
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        return True

    def size(self, x: int) -> int:
        """Members in *x*'s set."""
        return self._size[self.find(x)]

    def labels(self) -> list[int]:
        """Root of every element, by index."""
        n = len(self._parent)
        if n < NP_LABELS_MIN:
            parent = self._parent
            out = []
            for x in range(n):
                while parent[x] != x:
                    parent[x] = x = parent[parent[x]]
                out.append(x)
            return out

        # Pointer jumping: p <- p[p] until every entry is a root.
        p = np.array(self._parent)
        while True:
            q = p[p]
            if np.array_equal(q, p):
                return q.tolist()
            p = q

    def groups(self) -> list[list[int]]:
        """Sets ordered by first member, members ascending."""
        # Membership test beats setdefault: no throwaway [] per element.
        out: dict[int, list[int]] = {}
        for i, root in enumerate(self.labels()):
            if root in out:
                out[root].append(i)
            else:
                out[root] = [i]
        return list(out.values())

    def depth(self, x: int) -> int:
        """Hops from *x* to its root, without compressing (diagnostics)."""
        parent = self._parent
        d = 0
        while parent[x] != x:
            x = parent[x]
            d += 1
        return d
