"""Marginal contribution of operators to union edge coverage.

Game: ``v(S) = |union of edge sets of the operators in S|``. Per operator:

- ``size``: edges seen with it (what a raw count ranks by).
- ``gain``: edges it adds in a greedy cover order, so a copy of a better
  operator scores 0 where the raw count ties them.
- ``sole``: edges no other operator has (what pruning it would lose).
- ``shapley``: exact Shapley value; an edge held by k operators gives each 1/k.

``sole <= gain <= size`` and ``sole <= shapley <= size``; ``gain`` and
``shapley`` each sum to the union size.
"""

import heapq
import itertools
from typing import NamedTuple

import numpy as np

# Edge ids below this index a count table directly (no sort); 2**22 ids = 32 MB
# of int64 counts at worst. Ids are prev^cur hashes bounded by the map size.
DIRECT_MAX_ID = 1 << 22


class MarginalRow(NamedTuple):
    op: str
    size: int
    gain: int
    sole: int
    shapley: float


def _dense(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense column per entry, and how many entries share each column's edge.

    Small non-negative ids skip the sort: count ids, then rank the present ones.
    """
    if flat.size and flat.min() >= 0 and flat.max() < DIRECT_MAX_ID:
        counts = np.bincount(flat)
        present = counts > 0
        return (np.cumsum(present) - 1)[flat], counts[present]

    _, col, holders = np.unique(flat, return_inverse=True, return_counts=True)
    return col, holders


def _greedy(masks: dict[str, int]) -> list[tuple[str, int]]:
    """Greedy cover order: largest marginal gain first, ties by name.

    Lazy: gains only shrink, so a stale heap key bounds the true one and an
    entry whose fresh key still leads the heap is exactly the full-scan pick.
    """
    heap = [(-m.bit_count(), op) for op, m in masks.items()]
    heapq.heapify(heap)
    covered = 0
    order = []
    while heap:
        _, op = heapq.heappop(heap)
        fresh = (-(masks[op] & ~covered).bit_count(), op)
        if heap and fresh > heap[0]:
            heapq.heappush(heap, fresh)
            continue
        covered |= masks[op]
        order.append((op, -fresh[0]))
    return order


def marginal_table(sets: dict[str, set[int]]) -> list[MarginalRow]:
    """Rows in greedy cover order for ``{operator: edge set}``."""
    ops = list(sets)
    sizes = np.fromiter((len(sets[op]) for op in ops), dtype=np.int64, count=len(ops))
    flat = np.fromiter(
        itertools.chain.from_iterable(sets[op] for op in ops),
        dtype=np.int64,
        count=int(sizes.sum()),
    )
    owner = np.repeat(np.arange(len(ops)), sizes)

    # Dense edge index, and how many operators hold each edge.
    col, holders = _dense(flat)
    held = holders[col]
    share = np.bincount(owner, weights=1.0 / held, minlength=len(ops))
    sole = np.bincount(owner, weights=held == 1, minlength=len(ops)).astype(np.int64)

    # One bit per edge per operator, packed into a Python int for popcount.
    grid = np.zeros((len(ops), max(len(holders), 1)), dtype=bool)
    grid[owner, col] = True
    packed = np.packbits(grid, axis=1, bitorder="little")
    masks = {op: int.from_bytes(packed[i].tobytes(), "little") for i, op in enumerate(ops)}

    at = {op: i for i, op in enumerate(ops)}
    return [
        MarginalRow(op, int(sizes[at[op]]), gain, int(sole[at[op]]), float(share[at[op]]))
        for op, gain in _greedy(masks)
    ]
