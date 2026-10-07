"""Integer relation detection: HJLS, PSOS, PSLQ.

Find a nonzero integer vector m with ``sum(m_i * x_i) == 0`` for real x.
All three run the same iteration in different numerics (Ferguson, Bailey,
Arno 1999); HJLS is the gamma = sqrt(2) member of the family::

    x --init--> y = x, A = B = I, lower-trapezoidal H (n x n-1) spanning x-perp
      loop: m = argmax gamma^m |H_mm|, swap rows m, m+1, re-reduce (Hermite)
      stop: y_j ~ 0                 -> relation = column j of B
            1 / max |H_jj| > maxcoeff -> no relation with norm <= maxcoeff

    PSLQ  H kept directly; a Givens corner restores the trapezoid (sqrt).
    PSOS  H = L sqrt(D): unit-lower L, squared diagonal D from partial sums of
          squares, LLL swap formulas, no square roots.
    HJLS  L, D re-derived every step by classical Gram-Schmidt on the integer
          rows of A projected off x.

Decimal arithmetic at ``digits`` precision. Hard Rule 51: no mpmath/sympy.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, localcontext
from enum import Enum
from fractions import Fraction

Number = int | float | str | Decimal | Fraction

DEFAULT_DIGITS = 50
DEFAULT_MAXCOEFF = 1000
DEFAULT_MAXSTEPS = 1000

# tol = 10^-(digits * 4/5): leaves digits/5 for coefficient growth.
_TOL_NUM, _TOL_DEN = 4, 5

# PSLQ needs gamma > sqrt(4/3); Ferguson-Bailey use the bound itself.
_MIN_GAMMA2 = Fraction(4, 3)


class Algo(Enum):
    """Integer relation algorithm."""

    HJLS = "hjls"
    PSOS = "psos"
    PSLQ = "pslq"


# Squared gamma: HJLS selects on 2^m |b*_m|^2.
_GAMMA2 = {Algo.HJLS: Fraction(2), Algo.PSOS: _MIN_GAMMA2, Algo.PSLQ: _MIN_GAMMA2}


def _dec(v: Number) -> Decimal:
    """Round *v* into the active Decimal context."""
    if isinstance(v, Fraction):
        return Decimal(v.numerator) / Decimal(v.denominator)

    return +Decimal(v)


class _Search:
    """Shared state and loop; subclasses own the trapezoid H.

    y = x B and A = B^-1 stay exact-integer-linked: every row op on A is
    mirrored as a column op on B and on y.
    """

    def __init__(self, x: list[Decimal], gamma2: Decimal, tol2: Decimal, limit: Decimal) -> None:
        self.n = len(x)
        self.y = list(x)
        self.gamma2 = gamma2
        self.tol2 = tol2
        self.limit = limit
        self.a = [[int(i == j) for j in range(self.n)] for i in range(self.n)]
        self.b = [[int(i == j) for j in range(self.n)] for i in range(self.n)]

        # Partial sums of squares: s2[j] = sum_{k>=j} x_k^2.
        self.s2 = [Decimal(0)] * (self.n + 1)
        for j in range(self.n - 1, -1, -1):
            self.s2[j] = self.s2[j + 1] + x[j] * x[j]

    # Trapezoid interface.
    def _ratio(self, i: int, j: int) -> Decimal | None:
        raise NotImplementedError

    def _row_sub(self, i: int, j: int, t: int) -> None:
        raise NotImplementedError

    def _swap(self, m: int) -> None:
        raise NotImplementedError

    def _diag2(self) -> list[Decimal]:
        raise NotImplementedError

    def _reduce(self, i: int, jmax: int) -> None:
        """Hermite-reduce row i against rows jmax..0."""
        a, b, n = self.a, self.b, self.n
        for j in range(jmax, -1, -1):
            r = self._ratio(i, j)
            if r is None:
                continue

            t = int(r.to_integral_value())
            if not t:
                continue

            self.y[j] += t * self.y[i]
            self._row_sub(i, j, t)
            a[i] = [p - t * q for p, q in zip(a[i], a[j], strict=True)]
            for k in range(n):
                b[k][j] += t * b[k][i]

    def _hit(self) -> list[int] | None:
        """Column of B whose y entry vanished, sign-normalised."""
        j = min(range(self.n), key=lambda k: abs(self.y[k]))
        if self.y[j] * self.y[j] >= self.tol2 * self.s2[0]:
            return None

        col = [self.b[k][j] for k in range(self.n)]
        sign = 1 if next(c for c in col if c) > 0 else -1
        return [sign * c for c in col]

    def _pick(self, d2: list[Decimal]) -> int:
        """argmax gamma^(2(m+1)) D_m."""
        best, arg, g = Decimal(-1), 0, self.gamma2
        for m, d in enumerate(d2):
            if g * d > best:
                best, arg = g * d, m
            g *= self.gamma2
        return arg

    def _exchange(self, m: int) -> None:
        """Swap entries m, m+1 of y, rows of A, columns of B, then H."""
        self.y[m], self.y[m + 1] = self.y[m + 1], self.y[m]
        self.a[m], self.a[m + 1] = self.a[m + 1], self.a[m]
        for row in self.b:
            row[m], row[m + 1] = row[m + 1], row[m]
        self._swap(m)

    def _overflow(self) -> bool:
        """B entries too large for the precision to resolve y against tol."""
        return any(abs(c) > self.limit for row in self.b for c in row)

    def _reduce_all(self) -> None:
        for i in range(1, self.n):
            self._reduce(i, i - 1)

    def run(self, maxsteps: int, maxcoeff: int) -> list[int] | None:
        self._reduce_all()
        for _ in range(maxsteps):
            hit = self._hit()
            if hit is not None:
                return hit

            # Any relation has norm >= 1 / max |H_jj|.
            if max(self._diag2()) * maxcoeff * maxcoeff < 1:
                return None

            self._step()
            if self._overflow():
                return None

        return self._hit()

    def _step(self) -> int:
        """One iteration; returns the swapped row."""
        m = self._pick(self._diag2())
        self._exchange(m)
        for i in range(m + 1, self.n):
            self._reduce(i, min(i - 1, m + 1))
        return m


class _Pslq(_Search):
    """H held directly; Givens rotation fixes the corner after a swap."""

    def __init__(self, x: list[Decimal], gamma2: Decimal, tol2: Decimal, limit: Decimal) -> None:
        super().__init__(x, gamma2, tol2, limit)
        n, s2 = self.n, self.s2
        s = [v.sqrt() for v in s2]

        # H_jj = s_{j+1}/s_j, H_ij = -x_i x_j / (s_j s_{j+1}) below.
        self.h = [[Decimal(0)] * (n - 1) for _ in range(n)]
        for j in range(n - 1):
            self.h[j][j] = s[j + 1] / s[j]
            for i in range(j + 1, n):
                self.h[i][j] = -x[i] * x[j] / (s[j] * s[j + 1])

    def _ratio(self, i: int, j: int) -> Decimal | None:
        hjj = self.h[j][j]
        return self.h[i][j] / hjj if hjj else None

    def _row_sub(self, i: int, j: int, t: int) -> None:
        hi, hj = self.h[i], self.h[j]
        for k in range(j + 1):
            hi[k] -= t * hj[k]

    def _diag2(self) -> list[Decimal]:
        return [self.h[j][j] * self.h[j][j] for j in range(self.n - 1)]

    def _swap(self, m: int) -> None:
        h = self.h
        h[m], h[m + 1] = h[m + 1], h[m]
        if m >= self.n - 2:
            return

        # Rotate columns m, m+1 so H_{m,m+1} = 0 again.
        t0 = (h[m][m] * h[m][m] + h[m][m + 1] * h[m][m + 1]).sqrt()
        if not t0:
            return

        t1, t2 = h[m][m] / t0, h[m][m + 1] / t0
        for i in range(m, self.n):
            t3, t4 = h[i][m], h[i][m + 1]
            h[i][m] = t1 * t3 + t2 * t4
            h[i][m + 1] = -t2 * t3 + t1 * t4


class _Psos(_Search):
    """H = L sqrt(D) with unit-lower L; square-root free."""

    def __init__(self, x: list[Decimal], gamma2: Decimal, tol2: Decimal, limit: Decimal) -> None:
        super().__init__(x, gamma2, tol2, limit)
        n, s2 = self.n, self.s2
        self.c = n - 1

        # D_j = S_{j+1}/S_j, L_ij = -x_i x_j / S_{j+1}: partial sums only.
        self.d = [s2[j + 1] / s2[j] for j in range(self.c)]
        self.low = [[Decimal(int(i == j)) for j in range(self.c)] for i in range(n)]
        for j in range(self.c):
            for i in range(j + 1, n):
                self.low[i][j] = -x[i] * x[j] / s2[j + 1]

    def _ratio(self, i: int, j: int) -> Decimal | None:
        return self.low[i][j] if self.d[j] else None

    def _row_sub(self, i: int, j: int, t: int) -> None:
        li, lj = self.low[i], self.low[j]
        for k in range(j):
            li[k] -= t * lj[k]
        li[j] -= t

    def _diag2(self) -> list[Decimal]:
        return list(self.d)

    def _swap(self, m: int) -> None:
        """LLL exchange of rows m, m+1 (Cohen 2.6.3); row n-1 has D = 0."""
        lo, d, c = self.low, self.d, self.c
        last = m + 1 >= c
        mu, dm = lo[m + 1][m], d[m]
        dn = Decimal(0) if last else d[m + 1]
        big = dn + mu * mu * dm

        for k in range(m):
            lo[m][k], lo[m + 1][k] = lo[m + 1][k], lo[m][k]

        # Degenerate: new row m lies in the span below it.
        if not big:
            d[m] = Decimal(0)
            lo[m + 1][m] = Decimal(0)
            if not last:
                d[m + 1] = dm
            return

        nu = mu * dm / big
        d[m] = big
        lo[m + 1][m] = nu
        if last:
            return

        d[m + 1] = dm * dn / big
        for i in range(m + 2, self.n):
            t = lo[i][m + 1]
            lo[i][m + 1] = lo[i][m] - mu * t
            lo[i][m] = t + nu * lo[i][m + 1]


class _Hjls(_Psos):
    """L, D recomputed from A by classical Gram-Schmidt after every swap."""

    def __init__(self, x: list[Decimal], gamma2: Decimal, tol2: Decimal, limit: Decimal) -> None:
        super().__init__(x, gamma2, tol2, limit)
        self.x = list(x)
        self._gram()

    def _swap(self, m: int) -> None:
        self._gram()

    def _gram(self) -> None:
        """b*_i of the rows of A projected off x: p_i = a_i - (a_i.x / |x|^2) x."""
        n, c, x, s0 = self.n, self.c, self.x, self.s2[0]
        proj = []
        for row in self.a:
            f = sum((Decimal(v) * xi for v, xi in zip(row, x, strict=True)), Decimal(0)) / s0
            proj.append([v - f * xi for v, xi in zip(row, x, strict=True)])

        # Classical: mu_ij = <p_i, b*_j> / |b*_j|^2 with the unreduced p_i.
        star: list[list[Decimal]] = []
        for i in range(n):
            v = list(proj[i])
            for j in range(min(i, c)):
                mu = Decimal(0)
                if self.d[j]:
                    mu = _dot(proj[i], star[j]) / self.d[j]
                    v = [p - mu * q for p, q in zip(v, star[j], strict=True)]
                self.low[i][j] = mu
            if i < c:
                self.low[i][i] = Decimal(1)
                self.d[i] = _dot(v, v)
            star.append(v)


def _dot(u: list[Decimal], v: list[Decimal]) -> Decimal:
    return sum((p * q for p, q in zip(u, v, strict=True)), Decimal(0))


_IMPL: dict[Algo, type[_Search]] = {Algo.HJLS: _Hjls, Algo.PSOS: _Psos, Algo.PSLQ: _Pslq}


def find_relation(
    x: Sequence[Number],
    algo: Algo = Algo.PSLQ,
    *,
    digits: int = DEFAULT_DIGITS,
    tol: Number | None = None,
    maxcoeff: int = DEFAULT_MAXCOEFF,
    maxsteps: int = DEFAULT_MAXSTEPS,
    gamma: Number | None = None,
) -> list[int] | None:
    """Integer m != 0 with ``|sum m_i x_i| < tol * |x|``, or None.

    Args:
        x: At least two reals; give them at >= *digits* accuracy.
        algo: HJLS, PSOS or PSLQ.
        digits: Working Decimal precision.
        tol: Relative zero threshold; default ``10^-(4 digits / 5)``.
        maxcoeff: Give up once every relation must have norm > maxcoeff.
        maxsteps: Iteration budget.
        gamma: Selection parameter, >= sqrt(4/3); default per algorithm.

    Returns:
        Relation with positive first nonzero entry, e.g. ``[1, 1, -1]`` for
        ``(ln 2, ln 3, ln 6)``; None if none found within the bounds.
    """
    if len(x) < 2:
        raise ValueError("need at least two values")

    with localcontext() as ctx:
        ctx.prec = digits
        xs = [_dec(v) for v in x]
        t = _dec(tol) if tol is not None else Decimal(10) ** -(digits * _TOL_NUM // _TOL_DEN)
        g2 = _dec(gamma) ** 2 if gamma is not None else _dec(_GAMMA2[algo])
        if g2 < _dec(_MIN_GAMMA2) * (1 - t):
            raise ValueError("gamma must be >= sqrt(4/3)")

        # A vanishing entry is its own relation; also keeps every S_j > 0.
        s0 = sum((v * v for v in xs), Decimal(0))
        for i, v in enumerate(xs):
            if v * v < t * t * s0 or not s0:
                return [int(i == j) for j in range(len(xs))]

        limit = t * Decimal(10) ** digits
        return _IMPL[algo](xs, g2, t * t, limit).run(maxsteps, maxcoeff)


def hjls(x: Sequence[Number], **kw) -> list[int] | None:
    """HJLS (Hastad, Just, Lagarias, Schnorr 1989); see find_relation."""
    return find_relation(x, Algo.HJLS, **kw)


def psos(x: Sequence[Number], **kw) -> list[int] | None:
    """PSOS (Bailey, Ferguson 1988); see find_relation."""
    return find_relation(x, Algo.PSOS, **kw)


def pslq(x: Sequence[Number], **kw) -> list[int] | None:
    """PSLQ (Ferguson, Bailey 1992); see find_relation."""
    return find_relation(x, Algo.PSLQ, **kw)
