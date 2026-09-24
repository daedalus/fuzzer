"""Integer lattice reduction: LLL and Babai rounding.

Moved from ``tools/edge_diagnostic.py`` so core callers (``lcg_recovery``)
share one routine with the tool. Hard Rule 51: no sympy/fpylll.

    basis --lll_reduce--> short, near-orthogonal rows --babai_round--> lattice point near target
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import numpy as np


def lll_reduce(basis: Sequence[Sequence[int]], delta: float = 0.75) -> list[list[int]]:
    """LLL-reduce *basis* (a list of integer row vectors) in place, returning it.

    Written out rather than imported: Hard Rule 51, and the only outside
    implementation that would fit here is sympy's, which is a dependency this
    repo does not carry.

    The basis vectors stay exact Python integers -- every subtraction and
    swap below is integer arithmetic -- while the Gram-Schmidt coefficients
    are float.  That split is the usual engineering compromise and it is safe
    for callers that check results exactly: float error in ``mu`` can only
    make the reduction *weaker* (a size reduction skipped, a swap not taken),
    never produce a vector outside the lattice.  ``edge_diagnostic`` tests
    exact integer entries for zero; ``lcg_recovery`` replays the stream.

    The Gram-Schmidt row for ``k`` is recomputed from the orthogonalised rows
    below it whenever ``B[k]`` changes, and both affected rows are refreshed
    after a swap.  Rows above ``k`` are never read before ``k`` reaches them,
    so nothing stale is ever used.
    """
    rows = [list(map(int, r)) for r in basis]
    n = len(rows)
    if n < 2:
        return rows
    dim = len(rows[0])
    ortho = np.zeros((n, dim))
    mu = np.zeros((n, n))
    norms = np.zeros(n)

    def orthogonalise(k):
        v = np.array(rows[k], dtype=float)
        for j in range(k):
            if norms[j] > 0.0:
                mu[k, j] = float(np.dot(v, ortho[j]) / norms[j])
                v = v - mu[k, j] * ortho[j]
            else:
                mu[k, j] = 0.0
        ortho[k] = v
        norms[k] = float(np.dot(v, v))

    orthogonalise(0)
    k = 1
    while k < n:
        orthogonalise(k)
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k, j]))
            if q:
                rows[k] = [a - q * b for a, b in zip(rows[k], rows[j], strict=True)]
                orthogonalise(k)
        if norms[k] >= (delta - mu[k, k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            rows[k], rows[k - 1] = rows[k - 1], rows[k]
            orthogonalise(k - 1)
            orthogonalise(k)
            k = max(k - 1, 1)
    return rows


def solve_rows(rows: Sequence[Sequence[int]], target: Sequence[int]) -> list[Fraction]:
    """Exact coordinates ``c`` with ``sum_j c[j] * rows[j] == target``.

    Fraction Gauss-Jordan on the transposed system; square, full-rank *rows* only.

    Raises:
        ValueError: *rows* is singular or not square.
    """
    n = len(rows)
    if any(len(r) != n for r in rows) or len(target) != n:
        raise ValueError("solve_rows needs a square basis and a matching target")

    # Augmented [rows^T | target].
    m = [[Fraction(rows[j][i]) for j in range(n)] + [Fraction(target[i])] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if m[r][col] != 0), None)
        if piv is None:
            raise ValueError("singular basis")
        m[col], m[piv] = m[piv], m[col]
        pivot_row = m[col]
        for r in range(n):
            f = m[r][col]
            if r == col or f == 0:
                continue
            f /= pivot_row[col]
            m[r] = [a - f * b for a, b in zip(m[r], pivot_row, strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def babai_round(rows: Sequence[Sequence[int]], target: Sequence[int]) -> list[int]:
    """Babai rounding: the lattice point with coordinates ``round(solve_rows(rows, target))``.

    Close to the true closest vector when *rows* is LLL-reduced.
    """
    coords = [round(c) for c in solve_rows(rows, target)]
    n = len(target)
    return [sum(c * r[i] for c, r in zip(coords, rows, strict=True)) for i in range(n)]
