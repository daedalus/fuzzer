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
Reductions are optional: capped rounds and scans only cost optimality,
never correctness.

Representation: each seed's edges are one int bitmask over a dense edge
index, so restrict / take / subset are big-int ops. Holder counts per edge
are a bit-sliced counter (one int per count bit), which yields the
sole-holder and over-cap edge masks without a per-edge dict. ffmpeg
(~1,000 seeds x 2,400 edges): 4.9 s on frozensets.
"""

import heapq
from collections.abc import Collection, Hashable, Mapping
from enum import Enum
from itertools import product
from typing import TypeVar

import numpy as np

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
    full = _to_masks({k: e for k, e in seed_edges.items() if e})
    sizes = {k: size[k] for k in full}
    runs = [_Cover(order, full, sizes, tie, red).solve() for tie, red in product(Tie, Reduce)]
    return min(runs, key=lambda keys: (len(keys), sum(sizes[k] for k in keys)))


def _to_masks(full: Mapping[K, Collection[int]]) -> dict[K, int]:
    """Edge sets -> int bitmasks over a dense index (bit i = i-th smallest edge id).

    Bit order never changes a result: every tie-break is on seed rank.
    """
    if not full:
        return {}

    sets = list(full.values())
    flat = np.concatenate([np.fromiter(e, dtype=np.int64, count=len(e)) for e in sets])
    ids = np.unique(flat)
    row = np.zeros(-(-ids.size // 8) * 8, dtype=np.uint8)

    masks: dict[K, int] = {}
    for key, edges in zip(full, sets, strict=True):
        bits = np.searchsorted(ids, np.fromiter(edges, dtype=np.int64, count=len(edges)))
        row[bits] = 1
        masks[key] = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        row[bits] = 0
    return masks


def _holder_planes(masks: Collection[int], width: int) -> tuple[list[int], int]:
    """Bit-sliced per-edge holder count: plane p holds bit p of each edge's count.

    Returns (planes, saturated): *saturated* marks edges whose count reached
    2**width. Example, masks 0b011 and 0b110: planes [0b101, 0b010], i.e.
    edge 0 -> 1, edge 1 -> 2, edge 2 -> 1.
    """
    planes = [0] * width
    saturated = 0
    for mask in masks:
        carry = mask
        for p in range(width):
            if not carry:
                break
            planes[p], carry = planes[p] ^ carry, planes[p] & carry
        saturated |= carry
    return planes, saturated


def _count_above(planes: list[int], saturated: int, bound: int) -> int:
    """Mask of edges whose bit-sliced count exceeds *bound* (MSB-first compare)."""
    above = 0
    equal = -1  # all ones: every edge ties until a plane separates it
    for p in reversed(range(len(planes))):
        plane = planes[p]
        if (bound >> p) & 1:
            equal &= plane
            continue
        above |= equal & plane
        equal &= ~plane
    return above | saturated


def _bit_indices(mask: int, nbytes: int) -> np.ndarray:
    """Set-bit positions of *mask*, ascending."""
    raw = np.frombuffer(mask.to_bytes(nbytes, "little"), dtype=np.uint8)
    return np.flatnonzero(np.unpackbits(raw, bitorder="little"))


class _Cover:
    def __init__(
        self,
        order: Mapping,
        full: Mapping[K, int],
        size: Mapping[K, int],
        tie: Tie,
        reduce: Reduce,
    ) -> None:
        self._tie = tie
        self._reduce = reduce
        self._order = order
        self._full = dict(full)  # private: dominance deletes
        self._size = size
        self._live: dict = {}
        self._uncovered = 0
        for mask in full.values():
            self._uncovered |= mask
        self._nbytes = max(1, -(-self._uncovered.bit_length() // 8))
        self._chosen: list = []

        # Holder index of the last _index(): live keys in row order, their
        # packed masks (one row per key), and the sole / over-cap edge masks.
        self._rows: list = []
        self._matrix = np.zeros((0, self._nbytes), dtype=np.uint8)
        self._sole = 0
        self._crowded = 0

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
        best: dict[int, object] = {}
        for key, edges in self._full.items():
            kept = best.get(edges)
            if kept is None or self._rank(key) < self._rank(kept):
                best[edges] = key
        keep = set(best.values())
        self._full = {k: e for k, e in self._full.items() if k in keep}

    def _restrict(self) -> None:
        """Restrict seeds to their uncovered edges; drop seeds left with none."""
        self._live = {}
        uncovered = self._uncovered
        for key, edges in self._full.items():
            rest = edges & uncovered
            if rest:
                self._live[key] = rest

    def _index(self) -> None:
        """Restrict, then index which seeds hold each uncovered edge.

        Counts saturate past DOMINATOR_SCAN_CAP, which is all _dominated
        reads; sole holders are count == 1.
        """
        self._restrict()
        self._rows = list(self._live)
        nbytes = self._nbytes
        packed = b"".join(m.to_bytes(nbytes, "little") for m in self._live.values())
        self._matrix = np.frombuffer(packed, dtype=np.uint8).reshape(len(self._rows), nbytes)

        planes, saturated = _holder_planes(self._live.values(), DOMINATOR_SCAN_CAP.bit_length())
        multi = saturated
        for plane in planes[1:]:
            multi |= plane
        self._sole = planes[0] & ~multi
        self._crowded = _count_above(planes, saturated, DOMINATOR_SCAN_CAP)

    def _take(self, key) -> None:
        self._chosen.append(key)
        self._uncovered &= ~self._full[key]

    def _force(self) -> bool:
        """Take every seed that is the only holder of some uncovered edge."""
        sole_edges = self._sole
        sole = [k for k, edges in self._live.items() if edges & sole_edges]
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
        """A dominator holds every edge of *key*, so the holders of any one
        edge are a complete rival list; past the scan cap on all of them,
        give up (the set version scanned the rarest edge's holders)."""
        edges = self._live.get(key)
        if edges is None:
            return False
        scannable = edges & ~self._crowded
        if not scannable:
            return False
        edge = (scannable & -scannable).bit_length() - 1
        column = (self._matrix[:, edge >> 3] >> (edge & 7)) & 1
        rows = self._rows
        return any(self._beats(rows[i], key, edges) for i in np.flatnonzero(column).tolist())

    def _beats(self, rival, key, edges: int) -> bool:
        if rival == key:
            return False
        other = self._live.get(rival)
        if other is None or edges & ~other:
            return False
        return other.bit_count() > edges.bit_count() or self._rank(rival) < self._rank(key)

    def _greedy(self) -> None:
        """Lazy greedy: gains only shrink, so a stale heap top is re-scored."""
        heap = [(-e.bit_count(), *self._rank(k), k) for k, e in self._live.items()]
        heapq.heapify(heap)
        while heap and self._uncovered:
            neg_gain, size, order, key = heapq.heappop(heap)
            gain = (self._live[key] & self._uncovered).bit_count()
            if gain == 0:
                continue
            if gain != -neg_gain:
                heapq.heappush(heap, (-gain, size, order, key))
                continue
            self._take(key)

    def _drop_redundant(self) -> list:
        """Drop picks whose edges are all held by other picks, largest first."""
        if not self._chosen:
            return []
        bits = {key: _bit_indices(self._full[key], self._nbytes) for key in self._chosen}
        count = np.bincount(np.concatenate(list(bits.values())), minlength=self._nbytes * 8)
        keep = set(self._chosen)
        for key in sorted(self._chosen, key=self._rank, reverse=True):
            idx = bits[key]
            if (count[idx] == 1).any():
                continue
            keep.discard(key)
            count[idx] -= 1
        return [k for k in self._chosen if k in keep]
