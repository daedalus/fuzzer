"""Divide-and-conquer symmetric eigensolver (Cuppen 1981), pure Python.

Pipeline::

    A ──Householder──► Q T Qᵀ          (T tridiagonal: d, e)
    T = diag(T1', T2') + ρ u uᵀ        (split at every boundary, ρ = |e|)
    leaves (1x1) ──merge──► ... ──merge──► (w, Z)    bottom-up, no recursion
    eigvecs = Q Z

Each merge solves diag(D) + ρ z zᵀ: deflate tiny z_i and near-equal d_i
(Givens), root the secular equation 1 + ρ Σ z_i²/(d_i - λ) = 0 by bisection
on an offset from the nearest pole, then rebuild z (Gu-Eisenstat) so the
eigenvectors stay orthogonal.

numpy users get the same algorithm from LAPACK (``np.linalg.eigh`` →
``?syevd``); this module serves dependency-free paths.
"""

from __future__ import annotations

import math
import sys
from operator import mul, truediv

EPS = sys.float_info.epsilon
DEFLATE_FACTOR = 8.0  # LAPACK dlaed2 deflation tolerance: 8 * eps * scale
BISECT_MAX = 1200  # halvings to reach any double from the widest bracket

Matrix = list[list[float]]
Block = tuple[int, list[float], Matrix]  # (start row, eigenvalues, eigvec rows)
Pair = tuple[float, list[float]]  # (eigenvalue, eigenvector)


def eigh(a: Matrix) -> tuple[list[float], Matrix]:
    """Eigen-decompose a real symmetric matrix.

    Args:
        a: Square symmetric matrix as row lists (lower triangle is read).

    Returns:
        ``(w, v)``: ascending eigenvalues and ``v[i][j]`` = component i of
        eigenvector j, matching ``np.linalg.eigh``.

    Raises:
        ValueError: ``a`` is not square.
    """
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("eigh: matrix must be square")
    if n == 0:
        return [], []

    d, e, q = _tridiag(a)
    w, z = _dc_tridiag(d, e)
    return w, _matmul(q, z)


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    cols = list(zip(*b, strict=True))
    return [[sum(map(mul, row, col)) for col in cols] for row in a]


def _tridiag(a: Matrix) -> tuple[list[float], list[float], Matrix]:
    """Householder reduction ``a = Q T Qᵀ``; returns (diag, offdiag, Q)."""
    n = len(a)
    m = [[float(a[max(i, j)][min(i, j)]) for j in range(n)] for i in range(n)]
    q = [[float(i == j) for j in range(n)] for i in range(n)]

    for k in range(n - 2):
        _reflect(m, q, k)

    d = [m[i][i] for i in range(n)]
    e = [m[i + 1][i] for i in range(n - 1)]
    return d, e, q


def _reflect(m: Matrix, q: Matrix, k: int) -> None:
    """Zero column k below the subdiagonal with H = I - 2 v vᵀ."""
    n = len(m)
    x = [m[i][k] for i in range(k + 1, n)]
    if math.hypot(*x[1:]) == 0.0:
        return

    # v = x - alpha e1, alpha signed away from x0 to avoid cancellation.
    alpha = -math.copysign(math.hypot(*x), x[0])
    v = x[:]
    v[0] -= alpha
    vn = math.hypot(*v)
    v = [t / vn for t in v]
    idx = range(k + 1, n)

    # H M H = M - v uᵀ - u vᵀ with w = M v, c = vᵀw, u = 2(w - c v).
    w = [math.fsum(m[i][j] * v[j - k - 1] for j in idx) for i in range(n)]
    c = math.fsum(v[i - k - 1] * w[i] for i in idx)
    u = [0.0] * n
    for i in idx:
        u[i] = 2.0 * (w[i] - c * v[i - k - 1])
    for i in idx:
        vi, ui = v[i - k - 1], u[i]
        row = m[i]
        for j in idx:
            row[j] -= vi * u[j] + ui * v[j - k - 1]
    for i in idx:
        m[i][k] = m[k][i] = 0.0
    m[k + 1][k] = m[k][k + 1] = alpha

    # Q ← Q H.
    for row in q:
        s = 2.0 * math.fsum(row[j] * v[j - k - 1] for j in idx)
        for j in idx:
            row[j] -= s * v[j - k - 1]


def _dc_tridiag(d: list[float], e: list[float]) -> tuple[list[float], Matrix]:
    """Eigenpairs of tridiagonal (d, e) by bottom-up pairwise merging."""
    n = len(d)

    # Every boundary is split exactly once, so all corrections apply up front.
    dd = d[:]
    for i, b in enumerate(e):
        dd[i] -= abs(b)
        dd[i + 1] -= abs(b)

    blocks: list[Block] = [(i, [dd[i]], [[1.0]]) for i in range(n)]
    while len(blocks) > 1:
        merged = [_merge(blocks[i], blocks[i + 1], e) for i in range(0, len(blocks) - 1, 2)]
        if len(blocks) % 2:
            merged.append(blocks[-1])
        blocks = merged
    return blocks[0][1], blocks[0][2]


def _merge(lo: Block, hi: Block, e: list[float]) -> Block:
    """Glue adjacent blocks across boundary e[start_hi - 1]."""
    start, wa, qa = lo
    start_b, wb, qb = hi
    na = len(wa)
    b = e[start_b - 1]
    sign = 1.0 if b >= 0.0 else -1.0

    # u = e_last(lo) + sign e_first(hi), seen through each block's Q.
    z = qa[-1] + [sign * t for t in qb[0]]
    w, u = _rank_one(wa + wb, z, abs(b))

    q = _matmul(qa, u[:na]) + _matmul(qb, u[na:])
    return start, w, q


def _rank_one(dv: list[float], z: list[float], rho: float) -> tuple[list[float], Matrix]:
    """Eigenpairs of diag(dv) + rho z zᵀ (rho >= 0); columns of the result."""
    n = len(dv)
    order = sorted(range(n), key=dv.__getitem__)
    d = [dv[i] for i in order]
    zs = [z[i] for i in order]

    # Basis columns in original coordinates: starts as the sort permutation.
    cols = [[0.0] * n for _ in range(n)]
    for j, i in enumerate(order):
        cols[j][i] = 1.0

    zn = math.hypot(*zs)
    if zn > 0.0:
        rho *= zn * zn
        zs = [t / zn for t in zs]
    tol = DEFLATE_FACTOR * EPS * max(max(abs(t) for t in d), rho)

    keep = _deflate(d, zs, cols, rho, tol)
    pairs: list[Pair] = [(d[j], cols[j]) for j in range(n) if j not in set(keep)]
    if keep:
        pairs += _secular_pairs(d, zs, cols, keep, rho)

    pairs.sort(key=lambda p: p[0])
    w = [p[0] for p in pairs]
    return w, [list(r) for r in zip(*(p[1] for p in pairs), strict=True)]


def _deflate(d: list[float], zs: list[float], cols: Matrix, rho: float, tol: float) -> list[int]:
    """Return indices that still need the secular equation; rotate the rest out."""
    keep: list[int] = []
    for j in range(len(d)):
        if rho * abs(zs[j]) <= tol:
            continue
        if not keep or d[j] - d[keep[-1]] > tol:
            keep.append(j)
            continue

        # Near-equal poles: Givens folds z_p into z_j, so p decouples.
        p = keep.pop()
        r = math.hypot(zs[p], zs[j])
        c, s = zs[j] / r, zs[p] / r
        zs[p], zs[j] = 0.0, r
        cp, cj = cols[p], cols[j]
        cols[p] = [c * x - s * y for x, y in zip(cp, cj, strict=True)]
        cols[j] = [s * x + c * y for x, y in zip(cp, cj, strict=True)]
        keep.append(j)
    return keep


def _secular_pairs(
    d: list[float], zs: list[float], cols: Matrix, keep: list[int], rho: float
) -> list[Pair]:
    """Roots and vectors for the non-deflated subproblem."""
    dk = [d[j] for j in keep]
    zk = [zs[j] for j in keep]
    k = len(dk)

    # diffs[i][t] = dk[t] - λ_i, carried exactly from the root's pole offset.
    roots, diffs = [], []
    for i in range(k):
        lam, diff = _root(dk, zk, rho, i)
        roots.append(lam)
        diffs.append(diff)

    zhat = _gu_eisenstat(dk, zk, rho, diffs)
    rows = list(zip(*(cols[j] for j in keep), strict=True))
    out = []
    for i in range(k):
        v = list(map(truediv, zhat, diffs[i]))
        vn = math.hypot(*v)
        out.append((roots[i], [sum(map(mul, v, row)) / vn for row in rows]))
    return out


def _root(dk: list[float], zk: list[float], rho: float, i: int) -> tuple[float, list[float]]:
    """i-th secular root as (λ, [dk[t] - λ]), bisected relative to its nearest pole."""
    k = len(dk)
    z2 = [rho * t * t for t in zk]
    upper = dk[i + 1] if i + 1 < k else dk[i] + math.fsum(z2)

    def g(origin: float, tau: float) -> float:
        return 1.0 + math.fsum(map(truediv, z2, [(t - origin) - tau for t in dk]))

    # f(λ) increases between poles; its sign at the midpoint picks the closer pole.
    half = (upper - dk[i]) / 2.0
    if i + 1 == k or g(dk[i], half) >= 0.0:
        origin, lo, hi = dk[i], 0.0, half if i + 1 < k else 2.0 * half
    else:
        origin, lo, hi = dk[i + 1], -half, 0.0

    for _ in range(BISECT_MAX):
        mid = (lo + hi) / 2.0
        if mid in (lo, hi):
            break
        if g(origin, mid) > 0.0:
            hi = mid
        else:
            lo = mid

    tau = hi if origin == dk[i] else lo
    return origin + tau, [(t - origin) - tau for t in dk]


def _gu_eisenstat(dk: list[float], zk: list[float], rho: float, diffs: Matrix) -> list[float]:
    """z consistent with the computed roots, so (D - λ)⁻¹ ẑ are orthogonal.

    ẑ_t² = Π_i (λ_i - d_t) / (ρ Π_{i≠t} (d_i - d_t)), paired term by term to
    keep every factor O(1) and avoid overflow.
    """
    k = len(dk)
    zhat = []
    for t in range(k):
        val = -diffs[k - 1][t] / rho
        for i in range(t):
            val *= -diffs[i][t] / (dk[i] - dk[t])
        for i in range(t, k - 1):
            val *= -diffs[i][t] / (dk[i + 1] - dk[t])
        zhat.append(math.copysign(math.sqrt(max(val, 0.0)), zk[t]))
    return zhat
