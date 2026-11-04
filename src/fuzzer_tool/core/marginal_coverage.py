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
    # bincount costs O(max id), unique O(n log n): few entries with big ids sort instead.
    if flat.size and flat.min() >= 0 and flat.max() < min(DIRECT_MAX_ID, 64 * flat.size + 4096):
        counts = np.bincount(flat)
        present = counts > 0
        return (np.cumsum(present) - 1)[flat], counts[present]

    _, col, holders = np.unique(flat, return_inverse=True, return_counts=True)
    return col, holders


def _greedy(
    ops: list[str], sizes: np.ndarray, col: np.ndarray, holders: np.ndarray, owner: np.ndarray
) -> list[tuple[str, int]]:
    """Greedy cover order: largest marginal gain first, ties by name.

    Incremental and exact: picking an operator marks its newly covered edges,
    and each such edge costs one gain from every operator holding it, so the
    total work is one pass over the (operator, edge) entries.
    """
    n = len(ops)
    rank = np.argsort(np.array(ops, dtype=object), kind="stable")  # name order
    gains = sizes[rank].astype(np.int64)
    # Edge -> holders (CSR by dense column); entries are grouped by operator.
    # Stable sort on 16-bit keys is a radix sort (much faster than on int64).
    key = col.astype(np.uint16) if len(holders) <= 1 << 16 else col
    by_col = np.argsort(key, kind="stable")
    holder_ops = np.argsort(rank, kind="stable")[owner[by_col]]  # operator -> name rank
    starts = np.concatenate(([0], np.cumsum(holders)))
    offs = np.concatenate(([0], np.cumsum(sizes)))
    cols_of = [col[offs[i] : offs[i + 1]] for i in rank.tolist()]
    covered = np.zeros(len(holders), dtype=bool)
    order = []
    for _ in range(n):
        p = int(gains.argmax())  # first max == smallest name
        order.append((ops[rank[p]], int(gains[p])))
        gains[p] = -1
        cols = cols_of[p]
        new = cols[~covered[cols]]
        if new.size:
            covered[new] = True
            lens = holders[new]
            idx = np.repeat(starts[new] - (np.cumsum(lens) - lens), lens) + np.arange(
                int(lens.sum())
            )
            gains -= np.bincount(holder_ops[idx], minlength=n)
            gains[p] = -1
    return order


def marginal_table(sets: dict[str, set[int]]) -> list[MarginalRow]:
    """Rows in greedy cover order for ``{operator: edge set}``."""
    ops = list(sets)
    sizes = np.fromiter((len(sets[op]) for op in ops), dtype=np.int64, count=len(ops))
    chained = itertools.chain.from_iterable(sets[op] for op in ops)
    try:
        flat = np.fromiter(chained, dtype=np.int64, count=int(sizes.sum()))
    except OverflowError:
        # Ids outside int64: only identity matters, so factorize them.
        index: dict[int, int] = {}
        flat = np.fromiter(
            (
                index.setdefault(e, len(index))
                for e in itertools.chain.from_iterable(sets[op] for op in ops)
            ),
            dtype=np.int64,
            count=int(sizes.sum()),
        )
    return marginal_rows(ops, sizes, flat)


def marginal_rows(ops: list[str], sizes: np.ndarray, flat: np.ndarray) -> list[MarginalRow]:
    """Rows in greedy cover order; ops[i] holds the next sizes[i] distinct ids of *flat*."""
    owner = np.repeat(np.arange(len(ops)), sizes)

    # Dense edge index, and how many operators hold each edge.
    col, holders = _dense(flat)
    held = holders[col]
    share = np.bincount(owner, weights=1.0 / held, minlength=len(ops))
    sole = np.bincount(owner, weights=held == 1, minlength=len(ops)).astype(np.int64)

    at = {op: i for i, op in enumerate(ops)}
    return [
        MarginalRow(op, int(sizes[at[op]]), gain, int(sole[at[op]]), float(share[at[op]]))
        for op, gain in _greedy(ops, sizes, col, holders, owner)
    ]
