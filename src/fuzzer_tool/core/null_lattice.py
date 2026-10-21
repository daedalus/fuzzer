"""Short integer relations among the columns of an integer matrix.

Given ``A`` (S x n, exact ints), find short ``z in Z^n`` with ``A z = 0``.
This is the matrix form of ``core/integer_relation.py``, which takes one
real vector.  Spectral-Guided Lattice Discovery (SGLD) does the same on
float matrices: SVD for the near-null space, a quantised embedding
``[I | scale*A^T]``, LLL, then check on ``A``.  The checksum solver's data
is already integer, so the SVD and the quantisation drop out and the kernel
is computed exactly:

    A --column ops (unimodular U)--> A U = [pivots | 0] --> kernel = U's tail
      --lll_reduce (repeated)--> short, near-orthogonal kernel basis

Why not the embedding: with the S equations as extra lattice columns the LLL
dimension is n + S; the kernel has rank n - S, so reducing it directly is 2-5x
smaller in dimension and ~5x faster here (modulus relations, n = 19).

Rank is exact (``len(result) == n - rank``), so no SVD threshold is needed.
Every vector is built from integer operations only: it satisfies ``A z = 0``
exactly.  ``lll_reduce`` keeps float Gram-Schmidt, so a weak reduction can
leave a valid but long vector; callers must treat "no short vector" as "none
found", never as "none exists".

    >>> null_relations([[1, 2, 3], [4, 5, 6]])
    [(1, -2, 1)]
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

from fuzzer_tool.core.lattice import lll_reduce

# LLL passes over the kernel basis.  The elimination leaves entries far larger
# than the answer, and one float-GS pass then under-reduces (8-word relations
# at modulus 2^32: 1 of 20 found after one pass, 20 of 20 after four).  Each
# pass is exact on the integers, so extra passes only tighten the basis;
# the loop exits at the first fixed point.
_MAX_PASSES = 8

Matrix = Sequence[Sequence[int]]


def null_relations(rows: Matrix) -> list[tuple[int, ...]]:
    """LLL-reduced basis of ``{z in Z^n : rows . z = 0}``, shortest first.

    Args:
        rows: S x n integer matrix (S >= 1, n >= 1, rectangular).

    Returns:
        ``n - rank`` primitive vectors, first nonzero entry positive, sorted
        by squared norm then lexicographically.  Empty when the columns are
        independent.

    Raises:
        ValueError: *rows* is empty or ragged.
        TypeError: an entry is not an integer.
    """
    matrix = _checked(rows)
    kernel = _int_kernel(matrix)
    return _canonical(_reduce(kernel))


def _checked(rows: Matrix) -> list[list[int]]:
    """Copy of *rows* as Python ints, rejecting empty, ragged and float input."""
    if len(rows) == 0 or len(rows[0]) == 0:
        raise ValueError("matrix must have at least one row and one column")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("matrix rows must have equal length")

    return [[operator.index(x) for x in r] for r in rows]


def _int_kernel(matrix: list[list[int]]) -> list[list[int]]:
    """Basis of the integer right kernel by unimodular column operations.

    Row by row, Euclid over the not-yet-pivoted columns leaves one nonzero
    entry in that row; its column becomes a pivot.  The same operations on
    ``U`` keep ``A U = [pivots | 0]``, so the columns of ``U`` past the last
    pivot span the kernel lattice, and span it fully: ``U`` is unimodular.
    """
    n = len(matrix[0])
    cols = [[r[j] for r in matrix] for j in range(n)]
    unit = [[int(i == j) for i in range(n)] for j in range(n)]

    top = 0
    for s in range(len(matrix)):
        _euclid_row(cols, unit, s, top)
        pivot = next((j for j in range(top, n) if cols[j][s]), None)
        if pivot is None:
            continue

        cols[top], cols[pivot] = cols[pivot], cols[top]
        unit[top], unit[pivot] = unit[pivot], unit[top]
        top += 1
    return unit[top:]


def _euclid_row(cols: list[list[int]], unit: list[list[int]], s: int, top: int) -> None:
    """Zero row *s* in every column from *top* on except one (gcd survivor)."""
    n = len(cols)
    while True:
        live = [j for j in range(top, n) if cols[j][s]]
        if len(live) < 2:
            return

        # Smallest magnitude divides the others down: the sizes shrink like Euclid.
        p = min(live, key=lambda j: abs(cols[j][s]))
        for j in live:
            if j == p:
                continue
            _sub_col(cols, unit, j, p, cols[j][s] // cols[p][s], s)


def _sub_col(cols: list[list[int]], unit: list[list[int]], j: int, p: int, q: int, s: int) -> None:
    """Column ``j -= q * column p`` in A (rows from *s*; earlier rows are zero) and U."""
    if not q:
        return

    cj, cp = cols[j], cols[p]
    for r in range(s, len(cj)):
        cj[r] -= q * cp[r]

    uj, up = unit[j], unit[p]
    for r in range(len(uj)):
        uj[r] -= q * up[r]


def _reduce(kernel: list[list[int]]) -> list[list[int]]:
    """Repeat LLL until the basis stops changing, at most ``_MAX_PASSES`` times."""
    basis = kernel
    for _ in range(_MAX_PASSES):
        nxt = lll_reduce(basis)
        if nxt == basis:
            break
        basis = nxt
    return basis


def _canonical(basis: list[list[int]]) -> list[tuple[int, ...]]:
    """Sign-fix each vector (first nonzero positive) and sort short to long."""
    out = []
    for v in basis:
        lead = next((x for x in v if x), 0)
        out.append(tuple(-x for x in v) if lead < 0 else tuple(v))
    return sorted(out, key=lambda v: (sum(x * x for x in v), v))
