"""Compact per-seed edge set: sorted unique uint32 in an ``array``.

``EdgeTracker.seed_edges`` held one builtin set per tracked seed. On ffmpeg
(944 seeds, median 3,081 edges) that was 170-260 MB; the same edges as
sorted uint32 take 13 MB.

    set[int]   ~56-90 B/edge   hash slot + int object
    EdgeSet         4 B/edge   array('I'), sorted, unique

Reads match a builtin set: membership bisects; set algebra runs on a
materialized builtin set cached for the last few EdgeSets used, and returns a
new builtin ``set``, so callers that mutate a result keep working.
"""

from array import array
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from itertools import chain

import numpy as np

_TYPECODE = "I"
_DTYPE = np.uint32
_MAX_EDGE = int(np.iinfo(_DTYPE).max)
_BUILTIN_SETS = (set, frozenset)
# Materialized sets kept for repeated algebra (see EdgeSet._hot): bounded,
# 4 x ~200 KB at ffmpeg's 3k edges/seed.
_HOT_SLOTS = 4
_HOT: OrderedDict[int, tuple["EdgeSet", array, set[int]]] = OrderedDict()

if array(_TYPECODE).itemsize != np.dtype(_DTYPE).itemsize:
    raise ImportError(f"array('{_TYPECODE}') is not 32-bit on this platform")


def _pack(edges: Iterable[int]) -> array:
    """Sort, dedup and range-check *edges* into a uint32 array."""
    if isinstance(edges, EdgeSet):
        return array(_TYPECODE, edges._arr)

    raw = edges if isinstance(edges, np.ndarray) else np.fromiter(edges, dtype=np.int64)
    uniq = np.unique(raw)
    if uniq.size and (uniq[0] < 0 or uniq[-1] > _MAX_EDGE):
        raise OverflowError(f"edge id outside uint32: {uniq[0]}..{uniq[-1]}")
    return array(_TYPECODE, uniq.astype(_DTYPE).tobytes())


class EdgeSet(AbstractSet[int]):
    """Set of uint32 edge ids stored as one sorted array.

    Not hashable (it grows through :meth:`update`, like ``set``). Membership
    of a non-integer is False rather than a TypeError.
    """

    __slots__ = ("_arr",)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, edges: Iterable[int] = ()) -> None:
        self._arr = _pack(edges)

    @classmethod
    def _from_iterable(cls, it: Iterable[int]) -> set[int]:
        # Mixin results (``>=`` helpers etc.) stay builtin sets.
        return set(it)

    # ── Sized / Iterable / Container ───────────────────────────────────

    def __len__(self) -> int:
        return len(self._arr)

    def __iter__(self) -> Iterator[int]:
        return iter(self._arr)

    def __contains__(self, x: object) -> bool:
        if not isinstance(x, int | np.integer):
            return False

        a = self._arr
        i = bisect_left(a, x)
        return i < len(a) and a[i] == x

    def __repr__(self) -> str:
        return f"EdgeSet({list(self._arr)!r})"

    # ── Comparison ─────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EdgeSet):
            return self._arr == other._arr
        if not isinstance(other, AbstractSet):
            return NotImplemented
        if len(other) != len(self._arr):
            return False
        if isinstance(other, _BUILTIN_SETS):
            return other.issuperset(self._arr)
        return all(x in other for x in self._arr)

    def __le__(self, other: AbstractSet[object]) -> bool:
        if not isinstance(other, AbstractSet):
            return NotImplemented
        if len(self._arr) > len(other):
            return False
        if isinstance(other, _BUILTIN_SETS):
            return other.issuperset(self._arr)
        return all(x in other for x in self._arr)

    def isdisjoint(self, other: Iterable[object]) -> bool:
        """True when no element of *other* is in this set."""
        if isinstance(other, _BUILTIN_SETS):
            return other.isdisjoint(self._arr)
        return not any(x in self for x in other)

    # ── Algebra: builtin set results via a cached materialized set ─────

    def _hot(self) -> set[int]:
        """This set as a builtin set, cached for the last few EdgeSets used.

        Hot paths diff one parent against each mutant, exec after exec
        (``parent ^ mutant``, ``mutant & parent``). Hashing ~3k fresh ints
        off the array costs ~3x a set-to-set op, so the parent's set is
        built once and reused. Never handed to callers (they may mutate
        results). Revalidated by identity of ``_arr``, which update() rebinds.
        """
        key = id(self)
        hit = _HOT.get(key)
        if hit is not None and hit[0] is self and hit[1] is self._arr:
            _HOT.move_to_end(key)
            return hit[2]

        out = set(self._arr)
        _HOT[key] = (self, self._arr, out)
        _HOT.move_to_end(key)
        if len(_HOT) > _HOT_SLOTS:
            _HOT.popitem(last=False)
        return out

    def __and__(self, other: Iterable[object]) -> set[int]:  # type: ignore[override]
        if isinstance(other, _BUILTIN_SETS):
            return self._hot() & other
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._hot().intersection(other)

    __rand__ = __and__  # type: ignore[assignment]

    def __or__(self, other: Iterable[int]) -> set[int]:  # type: ignore[override]
        if isinstance(other, _BUILTIN_SETS):
            return self._hot() | other
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._hot().union(other)

    __ror__ = __or__  # type: ignore[assignment]

    def __sub__(self, other: Iterable[object]) -> set[int]:  # type: ignore[override]
        if isinstance(other, _BUILTIN_SETS):
            return self._hot() - other
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._hot().difference(other)

    def __rsub__(self, other: Iterable[int]) -> set[int]:
        if type(other) is set:
            return other - self._hot()
        if not isinstance(other, Iterable):
            return NotImplemented
        return set(other).difference(self._hot())

    def __xor__(self, other: Iterable[int]) -> set[int]:  # type: ignore[override]
        if isinstance(other, _BUILTIN_SETS):
            return self._hot() ^ other
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._hot().symmetric_difference(other)

    __rxor__ = __xor__  # type: ignore[assignment]

    # ── Growth / export ────────────────────────────────────────────────

    def update(self, *others: Iterable[int]) -> None:
        """Merge *others* in, keeping the array sorted and unique."""
        extra = np.fromiter(chain(*others), dtype=np.int64)
        if not extra.size:
            return

        self._arr = _pack(np.concatenate((self.as_numpy().astype(np.int64), extra)))

    def copy(self) -> "EdgeSet":
        """Independent copy."""
        return EdgeSet(self)

    def as_numpy(self) -> np.ndarray:
        """Read-only uint32 view, sorted ascending (no copy)."""
        view = np.frombuffer(self._arr, dtype=_DTYPE)
        view.flags.writeable = False
        return view
