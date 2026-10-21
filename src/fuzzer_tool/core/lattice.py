"""Integer lattice reduction: LLL (float and integral) and Babai rounding.

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


# Lovász parameter of the integral reduction, as num/den.
_DELTA_NUM, _DELTA_DEN = 3, 4

# Tolerated range for a caller-supplied delta: LLL terminates for 1/4 < delta < 1.
_DELTA_MIN, _DELTA_MAX = Fraction(1, 4), Fraction(1)


class _IntegralLLL:
    """Cohen's integral LLL (Algorithm 2.6.7): exact, no Fractions, no floats.

    Rows are 1-based inside (index 0 is padding) so the formulas read as in the
    book.  With ``d_i = prod_{j<=i} |b*_j|^2`` and ``lam[i][j] = d_j * mu[i][j]``
    all integers, every update is an exact integer division::

        extend(k)  Gram-Schmidt data for row k
        red(k, l)  size-reduce b_k by b_l   (|mu[k][l]| <= 1/2)
        swap(k)    exchange b_k, b_{k-1} and patch d, lam
    """

    def __init__(self, basis: Sequence[Sequence[int]], delta: Fraction) -> None:
        self.n = len(basis)
        self.b: list[list[int]] = [[], *[list(map(int, r)) for r in basis]]
        self.d = [1] + [0] * self.n
        self.lam = [[0] * (self.n + 1) for _ in range(self.n + 1)]
        self.kmax = 0
        self.num, self.den = delta.numerator, delta.denominator

    def extend(self, k: int) -> None:
        """Fill ``lam[k][1..k-1]`` and ``d[k]`` from the rows above."""
        b, d, lam = self.b, self.d, self.lam
        for j in range(1, k + 1):
            u = sum(x * y for x, y in zip(b[k], b[j], strict=True))
            for i in range(1, j):
                u = (d[i] * u - lam[k][i] * lam[j][i]) // d[i - 1]
            if j < k:
                lam[k][j] = u
                continue
            if u <= 0:
                raise ValueError("basis rows are linearly dependent")
            d[k] = u
        self.kmax = max(self.kmax, k)

    def red(self, k: int, j: int) -> None:
        """``b_k -= round(mu[k][j]) * b_j`` when ``|mu[k][j]| > 1/2``."""
        b, d, lam = self.b, self.d, self.lam
        if 2 * abs(lam[k][j]) <= d[j]:
            return

        # Nearest integer to lam/d, exact: floor((2 lam + d) / 2d).
        q = (2 * lam[k][j] + d[j]) // (2 * d[j])
        b[k] = [x - q * y for x, y in zip(b[k], b[j], strict=True)]
        lam[k][j] -= q * d[j]
        for i in range(1, j):
            lam[k][i] -= q * lam[j][i]

    def swap(self, k: int) -> None:
        """Exchange ``b_k`` and ``b_{k-1}``; update ``d`` and ``lam`` for rows >= k."""
        b, d, lam = self.b, self.d, self.lam
        b[k], b[k - 1] = b[k - 1], b[k]
        for j in range(1, k - 1):
            lam[k][j], lam[k - 1][j] = lam[k - 1][j], lam[k][j]

        mu = lam[k][k - 1]
        new_d = (d[k - 2] * d[k] + mu * mu) // d[k - 1]
        for i in range(k + 1, self.kmax + 1):
            t = lam[i][k]
            lam[i][k] = (d[k] * lam[i][k - 1] - mu * t) // d[k - 1]
            lam[i][k - 1] = (new_d * t + mu * lam[i][k]) // d[k]
        d[k - 1] = new_d

    def lovasz_fails(self, k: int) -> bool:
        """``d_k d_{k-2} < delta d_{k-1}^2 - lam[k][k-1]^2``, scaled to integers."""
        d, mu = self.d, self.lam[k][k - 1]
        return self.den * d[k] * d[k - 2] < self.num * d[k - 1] ** 2 - self.den * mu * mu

    def run(self) -> list[list[int]]:
        self.extend(1)
        k = 2
        while k <= self.n:
            if k > self.kmax:
                self.extend(k)
            self.red(k, k - 1)
            if self.lovasz_fails(k):
                self.swap(k)
                k = max(2, k - 1)
                continue

            for j in range(k - 2, 0, -1):
                self.red(k, j)
            k += 1
        return self.b[1:]


def lll_integral(
    basis: Sequence[Sequence[int]], delta: Fraction = Fraction(_DELTA_NUM, _DELTA_DEN)
) -> list[list[int]]:
    """LLL-reduce *basis* in exact integer arithmetic (Cohen 2.6.7), returning new rows.

    Unlike :func:`lll_reduce` nothing here is approximate: the result is size
    reduced (``|mu| <= 1/2``) and satisfies the Lovász condition for *delta*,
    so a single call suffices and no float error can weaken it.  Cost is bigint
    arithmetic on integers the size of the Gram determinants; on the kernel
    bases of ``core/null_lattice`` it ran 7-10x faster than repeated
    :func:`lll_reduce` passes (8 words, N = 2^16: 24 -> 3 ms).

    Args:
        basis: Linearly independent integer rows.
        delta: Lovász parameter, 1/4 < delta < 1 (default 3/4).

    Raises:
        ValueError: *delta* out of range, or the rows are linearly dependent
            or ragged.
    """
    if not _DELTA_MIN < delta < _DELTA_MAX:
        raise ValueError("delta must satisfy 1/4 < delta < 1")
    if len(basis) == 0:
        return []
    if len({len(r) for r in basis}) != 1:
        raise ValueError("basis rows must have equal length")
    if len(basis) == 1:
        if not any(basis[0]):
            raise ValueError("basis rows are linearly dependent")
        return [list(map(int, basis[0]))]

    return _IntegralLLL(basis, Fraction(delta)).run()


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
