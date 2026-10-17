"""Polynomial compares mod 2^k: ``p(x) mod 2^k == t``.

Cmplog logs ``v = p(x)`` truncated to the operand width (``x*x + 3*x``, the
article's degree-4 loop). ``core.affine_mod`` fits degree 1 only. Forward
differences at one point pin any integer polynomial::

    v_i = v(xs + i), i < SAMPLES        c_j = Δ^j v(xs)  (mod 2^k)
    v(x) = sum_j c_j * C(x - xs, j)     Newton form, exact for every integer x

Degree ``d`` is the last nonzero ``c_j``; ``TAIL`` trailing zeros and
``CONFIRMS`` spread-out probes must agree, else fail closed (xor, hash
rounds, table lookups).

Roots lift one bit at a time (Hensel): ``p(x) mod 2^(j+1)`` depends only on
``x mod 2^(j+1)``, so a root mod 2^j extends to ``r`` and/or ``r + 2^j``::

    bit 0: {0, 1} ∩ roots ──▶ bit 1: r, r+2 ──▶ ... ──▶ roots mod 2^k

Wider fields repeat each root every ``2^k``. Following Leviathan, "Encoding
Bitwise Functions as Polynomials" (2024): root finding over Z misses
machine-integer wraparound; the ring is Z/2^k.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# x -> operand ``v`` from one run with the field set to x; None if not reached.
Observe = Callable[[int], int | None]

# Trailing zero differences required past the degree.
TAIL = 2

# Unit-step samples: degree <= MAX_DEGREE leaves TAIL zero differences.
SAMPLES = 10
MAX_DEGREE = SAMPLES - TAIL - 1

# Probes outside the sample window that must agree with the fit.
CONFIRMS = 3

# Roots kept per Hensel level; x*x << 30 would otherwise keep 2^31.
MAX_FRONTIER = 256

# Field values returned per solve, nearest to the current value first.
MAX_SOLUTIONS = 3


@dataclass(frozen=True)
class PolyMod:
    """``v(x) = sum_j coeffs[j] * C(x - x_ref, j) mod 2^bits``."""

    x_ref: int
    coeffs: tuple[int, ...]
    bits: int

    def __call__(self, x: int) -> int:
        t = x - self.x_ref
        v, binom = 0, 1
        for j, c in enumerate(self.coeffs):
            v += c * binom
            # C(t, j+1) = C(t, j) * (t - j) / (j + 1): exact, also for t < 0.
            binom = binom * (t - j) // (j + 1)
        return v & ((1 << self.bits) - 1)


def _newton(vs: list[int], mask: int) -> list[int]:
    """Forward differences ``Δ^j v(xs)`` mod 2^k, j = 0 .. len(vs) - 1."""
    coeffs: list[int] = []
    row = vs
    while row:
        coeffs.append(row[0])
        row = [(b - a) & mask for a, b in zip(row, row[1:], strict=False)]
    return coeffs


def _spots(xs: int, top: int) -> list[int]:
    """Confirm points outside the window, inside the field."""
    window = range(xs, xs + SAMPLES)
    spots = {top, top // 2 + 1, xs + 2 * SAMPLES}
    return sorted(x for x in spots if 0 <= x <= top and x not in window)[:CONFIRMS]


def fit(observe: Observe, x0: int, width_bits: int, bits: int) -> PolyMod | None:
    """Polynomial of degree 1..MAX_DEGREE behind *observe*, mod 2^bits; else None.

    ``x0`` is the field value the target reached the compare with; the
    window starts there, shifted down only to stay inside the field.
    """
    top = (1 << width_bits) - 1
    if top + 1 < SAMPLES:
        return None

    xs = max(0, min(x0, top - SAMPLES + 1))
    vs: list[int] = []
    for i in range(SAMPLES):
        v = observe(xs + i)
        if v is None:
            return None
        vs.append(v)

    coeffs = _newton(vs, (1 << bits) - 1)
    deg = max((j for j, c in enumerate(coeffs) if c), default=0)
    if not 1 <= deg <= MAX_DEGREE:
        return None

    model = PolyMod(xs, tuple(coeffs[: deg + 1]), bits)
    for x in _spots(xs, top):
        if observe(x) != model(x):
            return None

    return model


def _thin(frontier: list[int], width: int) -> list[int]:
    """At most MAX_FRONTIER roots, evenly spread over their low bits.

    Truncating in order keeps only the ``x = 0`` lineage (all even); x*x << 30
    then loses every odd root at bit 30. Sorting by bit-reversed value groups
    by bit 0, then bit 1, ...; even spacing keeps every low-bit class.
    """
    n = len(frontier)
    if n <= MAX_FRONTIER:
        return frontier

    frontier.sort(key=lambda r: int(f"{r:0{width}b}"[::-1], 2))
    return [frontier[i * n // MAX_FRONTIER] for i in range(MAX_FRONTIER)]


def roots(model: PolyMod, target: int, width_bits: int) -> list[int]:
    """Every ``x < 2^min(bits, width_bits)`` with ``model(x) == target``.

    The frontier is capped at MAX_FRONTIER per level, so a degenerate map
    returns a subset (still all true roots), never a wrong one.
    """
    if target < 0 or target >> model.bits:
        return []

    frontier = [0]
    for j in range(model.bits):
        bit = 1 << j
        sub = (bit << 1) - 1
        want = target & sub
        # Above the field's top bit x stays 0: only r survives.
        options = (0, bit) if j < width_bits else (0,)

        frontier = [r | o for r in frontier for o in options if model(r | o) & sub == want]
        frontier = _thin(frontier, j + 1)
        if not frontier:
            return []

    return frontier


def _nearest(r: int, period: int, top: int, near: int) -> int:
    """Member of ``r (mod period)`` in ``[0, top]`` closest to *near*."""
    k = min(max((near - r + period // 2) // period, 0), (top - r) // period)
    return r + k * period


def solve(model: PolyMod, target: int, width_bits: int, near: int = 0) -> list[int]:
    """Up to MAX_SOLUTIONS field values with ``model(x) == target``, nearest first."""
    top = (1 << width_bits) - 1
    period = 1 << model.bits
    found = {_nearest(r, period, top, near) for r in roots(model, target, width_bits)}
    return sorted(found, key=lambda x: (abs(x - near), x))[:MAX_SOLUTIONS]
