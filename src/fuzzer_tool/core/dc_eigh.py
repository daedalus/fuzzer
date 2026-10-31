"""Divide-and-conquer symmetric eigensolver (Cuppen 1981), pure Python.

Pipeline::

    A ──Householder──► Q T Qᵀ          (T tridiagonal: d, e)
    T = diag(T1', T2') + ρ u uᵀ        (split at every boundary, ρ = |e|)
    leaves (1x1) ──merge──► ... ──merge──► (w, Z)    bottom-up, no recursion
    eigvecs = Q Z

Each merge solves diag(D) + ρ z zᵀ: deflate tiny z_i and near-equal d_i
(Givens), root the secular equation 1 + ρ Σ z_i²/(d_i - λ) = 0 as an offset
from the nearest pole (two-pole rational steps inside a bisection bracket,
~3.4 evaluations per root), then rebuild z (Gu-Eisenstat) so the
eigenvectors stay orthogonal. Householder reflectors are applied straight to
the eigenvectors and merges rotate the block columns in place, so Q is never
formed.

numpy users get the same algorithm from LAPACK (``np.linalg.eigh`` →
``?syevd``); this module serves dependency-free paths.
"""

from __future__ import annotations

import math
import sys
from operator import mul, truediv

EPS = sys.float_info.epsilon
DEFLATE_FACTOR = 8.0  # LAPACK dlaed2 deflation tolerance: 8 * eps * scale
BISECT_MAX = 1200  # iteration cap: enough halvings to reach any double
NEWTON_RTOL = 2.0 * EPS  # stop once a step moves τ by ~1 ulp
SECULAR_ERR = 8.0  # |g| below 8 eps (1 + Σ|terms|) is rounding noise (LAPACK dlaed4)

Matrix = list[list[float]]
Block = tuple[int, list[float], Matrix]  # (start row, eigenvalues, eigvec columns)
Pair = tuple[float, list[float]]  # (eigenvalue, eigenvector)
Reflector = tuple[int, list[float]]  # (first row lo, unit v): H = I - 2 v vᵀ on rows lo..


def eigh(a: Matrix, count: int | None = None) -> tuple[list[float], Matrix]:
    """Eigen-decompose a real symmetric matrix.

    Args:
        a: Square symmetric matrix as row lists (lower triangle is read).
        count: Return only the ``count`` smallest pairs (default: all). Skips
            back-transforming the rest, the last O(n³) step.

    Returns:
        ``(w, v)``: ascending eigenvalues and ``v[i][j]`` = component i of
        eigenvector j, matching ``np.linalg.eigh``.

    Raises:
        ValueError: ``a`` is not square, or ``count`` is negative.
    """
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("eigh: matrix must be square")
    if count is not None and count < 0:
        raise ValueError("eigh: count must be >= 0")
    if n == 0:
        return [], []

    d, e, refl = _tridiag(a)
    w, cols = _dc_tridiag(d, e)
    w, cols = w[:count], cols[:count]

    # eigvecs = H_0 H_1 ... z: apply reflectors last-first to each column.
    for lo, v in reversed(refl):
        for col in cols:
            s = 2.0 * sum(map(mul, col[lo:], v))
            col[lo:] = [x - s * vj for x, vj in zip(col[lo:], v, strict=True)]
    return w, [[c[i] for c in cols] for i in range(n)]


def _tridiag(a: Matrix) -> tuple[list[float], list[float], list[Reflector]]:
    """Householder reduction ``a = Q T Qᵀ``; returns (diag, offdiag, Q's reflectors)."""
    n = len(a)
    m = [[float(a[max(i, j)][min(i, j)]) for j in range(n)] for i in range(n)]
    refl = [r for k in range(n - 2) if (r := _reflect(m, k)) is not None]

    d = [m[i][i] for i in range(n)]
    e = [m[i + 1][i] for i in range(n - 1)]
    return d, e, refl


def _reflect(m: Matrix, k: int) -> Reflector | None:
    """Zero column k below the subdiagonal with H = I - 2 v vᵀ."""
    n = len(m)
    x = [m[i][k] for i in range(k + 1, n)]
    if math.hypot(*x[1:]) == 0.0:
        return None

    # v = x - alpha e1, alpha signed away from x0 to avoid cancellation.
    alpha = -math.copysign(math.hypot(*x), x[0])
    v = x[:]
    v[0] -= alpha
    vn = math.hypot(*v)
    v = [t / vn for t in v]
    lo = k + 1

    # H M H = M - v uᵀ - u vᵀ with w = M v, c = vᵀw, u = 2(w - c v); M[lo:, lo:] only.
    w = [sum(map(mul, m[i][lo:], v)) for i in range(lo, n)]
    c = sum(map(mul, v, w))
    u = [2.0 * (wi - c * vi) for wi, vi in zip(w, v, strict=True)]
    for i, vi, ui in zip(range(lo, n), v, u, strict=True):
        row = m[i]
        row[lo:] = [r - vi * uj - ui * vj for r, uj, vj in zip(row[lo:], u, v, strict=True)]
        row[k] = m[k][i] = 0.0
    m[lo][k] = m[k][lo] = alpha
    return lo, v


def _dc_tridiag(d: list[float], e: list[float]) -> tuple[list[float], Matrix]:
    """Eigenvalues and eigenvector columns of tridiagonal (d, e), merged bottom-up."""
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
    b = e[start_b - 1]
    sign = 1.0 if b >= 0.0 else -1.0

    # u = e_last(lo) + sign e_first(hi), seen through each block's Q.
    z = [c[-1] for c in qa] + [sign * c[0] for c in qb]

    # diag(Qa, Qb) as zero-padded columns: rotations then act on them directly.
    pad_a, pad_b = [0.0] * len(qb[0]), [0.0] * len(qa[0])
    basis = [c + pad_a for c in qa] + [pad_b + c for c in qb]
    w, cols = _rank_one(wa + wb, z, abs(b), basis)
    return start, w, cols


def _rank_one(
    dv: list[float], z: list[float], rho: float, basis: Matrix
) -> tuple[list[float], Matrix]:
    """Eigenpairs of diag(dv) + rho z zᵀ (rho >= 0), as columns over ``basis``."""
    n = len(dv)
    order = sorted(range(n), key=dv.__getitem__)
    d = [dv[i] for i in order]
    zs = [z[i] for i in order]
    cols = [basis[i] for i in order]

    zn = math.hypot(*zs)
    if zn > 0.0:
        rho *= zn * zn
        zs = [t / zn for t in zs]
    tol = DEFLATE_FACTOR * EPS * max(max(abs(t) for t in d), rho)

    keep = _deflate(d, zs, cols, rho, tol)
    kept = set(keep)
    pairs: list[Pair] = [(d[j], cols[j]) for j in range(n) if j not in kept]
    if keep:
        pairs += _secular_pairs(d, zs, cols, keep, rho)

    pairs.sort(key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


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
    """i-th secular root as (λ, [dk[t] - λ]), offset from its nearest pole."""
    k = len(dk)
    z2 = [rho * t * t for t in zk]
    upper = dk[i + 1] if i + 1 < k else dk[i] + math.fsum(z2)

    # f(λ) increases between poles; its sign at the midpoint picks the closer pole.
    half = (upper - dk[i]) / 2.0
    if i + 1 == k or _secular(z2, [t - dk[i] for t in dk], half) >= 0.0:
        origin, lo, hi = dk[i], 0.0, half if i + 1 < k else 2.0 * half
    else:
        origin, lo, hi = dk[i + 1], -half, 0.0

    delta = [t - origin for t in dk]
    tau = _bracket_root(z2, delta, i, lo, hi)
    return origin + tau, [t - tau for t in delta]


def _secular(z2: list[float], delta: list[float], tau: float) -> float:
    """g(τ) = 1 + Σ z2_t / (delta_t - τ)."""
    return 1.0 + math.fsum(map(truediv, z2, [t - tau for t in delta]))


def _bracket_root(z2: list[float], delta: list[float], i: int, lo: float, hi: float) -> float:
    """Root of increasing g in (lo, hi); bisection whenever a model step leaves the bracket.

    Steps solve the two-pole model (LAPACK dlaed4): ψ = Σ_{t<=i} and φ = Σ_{t>i}
    are each replaced by a + b/(delta_i - τ) matching value and slope at τ.
    One pole sits at τ = 0 (an endpoint); the returned τ is never 0.
    """
    p = delta[i]
    q = delta[i + 1] if i + 1 < len(delta) else math.inf
    tau = (lo + hi) / 2.0
    for _ in range(BISECT_MAX):
        r = [1.0 / (t - tau) for t in delta]
        terms = list(map(mul, z2, r))
        slopes = list(map(mul, terms, r))
        psi, dpsi = math.fsum(terms[: i + 1]), math.fsum(slopes[: i + 1])
        phi, dphi = math.fsum(terms[i + 1 :]), math.fsum(slopes[i + 1 :])
        g = 1.0 + psi + phi

        # ψ terms are negative, φ terms positive: φ - ψ bounds g's rounding error.
        if abs(g) <= SECULAR_ERR * EPS * (1.0 + phi - psi):
            return tau
        if g > 0.0:
            hi = tau
        else:
            lo = tau

        step = _model_step(tau, psi, dpsi, phi, dphi, p, q)
        if not lo < step < hi:
            step = (lo + hi) / 2.0
        if step in (lo, hi) or abs(step - tau) <= NEWTON_RTOL * abs(step):
            return step if step != 0.0 else (hi if hi != 0.0 else lo)
        tau = step
    return tau


def _model_step(
    tau: float, psi: float, dpsi: float, phi: float, dphi: float, p: float, q: float
) -> float:
    """Zero in (p, q) of c + b/(p - x) + B/(q - x) fitted at τ; nan if degenerate."""
    b = dpsi * (p - tau) ** 2
    c = 1.0 + psi - dpsi * (p - tau)
    if q == math.inf:
        return p + b / c if c > 0.0 else math.nan
    big_b = dphi * (q - tau) ** 2
    c += phi - dphi * (q - tau)

    # c(p - x)(q - x) + b(q - x) + B(p - x) = 0: +b(q-p) at p, -B(q-p) at q, one root between.
    beta = c * (p + q) + b + big_b
    gamma = c * p * q + b * q + big_b * p
    if c == 0.0:
        return gamma / beta
    sq = math.sqrt(max(beta * beta - 4.0 * c * gamma, 0.0))
    t = beta + math.copysign(sq, beta)
    if t == 0.0:
        return math.nan
    x1, x2 = t / (2.0 * c), 2.0 * gamma / t
    return x1 if p < x1 < q else x2


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
