"""Set cover for corpus minimization: reductions, lazy greedy, redundancy pass.

Pick few seeds that together hit every edge any seed hits.

Greedy is a heuristic: breaking ties by file size usually saves bytes but
can steer it to a worse path (measured: 9 -> 11 seeds on one corpus). So
min_cover runs every Tie x Reduce combination and keeps the best by
(seed count, bytes). One combination is plain first-wins greedy, so the
count never exceeds it.

Why reductions: plain greedy takes the biggest gain first and can end up
holding a seed that later picks made redundant.

    A = {1,2,3,4}   B = {1,2,5}   C = {3,4,6}

    greedy: A, B, C   (A redundant)
    forced: edge 5 -> B only, edge 6 -> C only  =>  B, C   (optimal)

Pipeline: collapse identical edge sets -> [force sole coverers, drop
dominated seeds] x MAX_ROUNDS -> lazy greedy -> drop redundant picks.
Memory is O(sum of edge-set sizes). Reductions are optional: capped
rounds and scans only cost optimality, never correctness.
"""

import heapq
from collections.abc import Collection, Hashable, Mapping
from enum import Enum
from itertools import product
from typing import TypeVar

K = TypeVar("K", bound=Hashable)

# Reduction rounds; each is O(sum |edges|). Cascades beyond this are rare.
MAX_ROUNDS = 8

# Dominators examined per seed. Bounds the pass on corpora where one rare
# edge is shared by thousands of seeds.
DOMINATOR_SCAN_CAP = 256


class Tie(Enum):
    """Which seed wins among equal-gain candidates."""

    SIZE = "size"  # smaller file, then earlier
    ORDER = "order"  # earlier in input order


class Reduce(Enum):
    """Whether to run forced-seed and dominance reductions first."""

    ON = "on"
    OFF = "off"


def min_cover(seed_edges: Mapping[K, Collection[int]], size: Mapping[K, int]) -> list[K]:
    """Return keys of a small cover of every edge in *seed_edges*.

    *size* must hold every seed that has edges (KeyError otherwise).
    Result order is selection order; deterministic for a given input order.
    """
    order = {k: i for i, k in enumerate(seed_edges)}
    full = {k: frozenset(e) for k, e in seed_edges.items() if e}
    sizes = {k: size[k] for k in full}
    runs = [_Cover(order, full, sizes, tie, red).solve() for tie, red in product(Tie, Reduce)]
    return min(runs, key=lambda keys: (len(keys), sum(sizes[k] for k in keys)))


class _Cover:
    def __init__(
        self,
        order: Mapping,
        full: Mapping[K, frozenset],
        size: Mapping[K, int],
        tie: Tie,
        reduce: Reduce,
    ) -> None:
        self._tie = tie
        self._reduce = reduce
        self._order = order
        self._full = dict(full)  # frozensets shared, dict private: dominance deletes
        self._size = size
        self._live: dict = {}
        self._holders: dict[int, list] = {}
        self._uncovered: set[int] = set().union(*full.values()) if full else set()
        self._chosen: list = []

    def solve(self) -> list:
        if self._reduce is Reduce.ON:
            self._reduce_all()
        self._restrict()
        self._greedy()
        return self._drop_redundant()

    def _reduce_all(self) -> None:
        self._collapse_equal()
        for _ in range(MAX_ROUNDS):
            self._index()
            forced = self._force()
            if forced:
                self._index()
            dropped = self._dominate()
            if not (forced or dropped):
                break

    def _rank(self, key) -> tuple[int, int]:
        if self._tie is Tie.ORDER:
            return (0, self._order[key])
        return (self._size[key], self._order[key])

    def _collapse_equal(self) -> None:
        """Keep one seed per identical edge set: the smallest, then earliest."""
        best: dict[frozenset, object] = {}
        for key, edges in self._full.items():
            kept = best.get(edges)
            if kept is None or self._rank(key) < self._rank(kept):
                best[edges] = key
        keep = set(best.values())
        self._full = {k: e for k, e in self._full.items() if k in keep}

    def _restrict(self) -> None:
        """Restrict seeds to their uncovered edges; drop seeds left with none."""
        self._live = {}
        for key, edges in self._full.items():
            rest = edges & self._uncovered
            if rest:
                self._live[key] = rest

    def _index(self) -> None:
        """Restrict, then index which seeds hold each uncovered edge."""
        self._restrict()
        self._holders = {}
        for key, edges in self._live.items():
            for edge in edges:
                self._holders.setdefault(edge, []).append(key)

    def _take(self, key) -> None:
        self._chosen.append(key)
        self._uncovered -= self._full[key]

    def _force(self) -> bool:
        """Take every seed that is the only holder of some uncovered edge."""
        sole = {h[0] for h in self._holders.values() if len(h) == 1}
        for key in sorted(sole, key=self._order.__getitem__):
            self._take(key)
        return bool(sole)

    def _dominate(self) -> bool:
        """Drop seeds whose uncovered edges sit inside another seed's."""
        dropped = False
        for key in list(self._live):
            if self._dominated(key):
                del self._full[key]
                dropped = True
        return dropped

    def _dominated(self, key) -> bool:
        edges = self._live.get(key)
        if edges is None:
            return False
        rarest = min(edges, key=lambda e: len(self._holders[e]))
        rivals = self._holders[rarest]
        if len(rivals) > DOMINATOR_SCAN_CAP:
            return False
        return any(self._beats(rival, key, edges) for rival in rivals)

    def _beats(self, rival, key, edges: frozenset) -> bool:
        if rival == key:
            return False
        other = self._live.get(rival)
        if other is None or not edges <= other:
            return False
        return len(other) > len(edges) or self._rank(rival) < self._rank(key)

    def _greedy(self) -> None:
        """Lazy greedy: gains only shrink, so a stale heap top is re-scored."""
        heap = [(-len(e), *self._rank(k), k) for k, e in self._live.items()]
        heapq.heapify(heap)
        while heap and self._uncovered:
            neg_gain, size, order, key = heapq.heappop(heap)
            gain = len(self._live[key] & self._uncovered)
            if gain == 0:
                continue
            if gain != -neg_gain:
                heapq.heappush(heap, (-gain, size, order, key))
                continue
            self._take(key)

    def _drop_redundant(self) -> list:
        """Drop picks whose edges are all held by other picks, largest first."""
        count: dict[int, int] = {}
        for key in self._chosen:
            for edge in self._full[key]:
                count[edge] = count.get(edge, 0) + 1
        keep = set(self._chosen)
        for key in sorted(self._chosen, key=self._rank, reverse=True):
            if any(count[e] == 1 for e in self._full[key]):
                continue
            keep.discard(key)
            for edge in self._full[key]:
                count[edge] -= 1
        return [k for k in self._chosen if k in keep]
