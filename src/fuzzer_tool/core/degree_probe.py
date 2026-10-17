"""Algebraic-degree probe for a single branch bit over GF(2).

Given ``exec_bit(x) -> 0|1`` (one target execution with input bits ``x`` on a
small field, observing one edge), classify how ``f`` depends on ``x`` and,
for the cheapest class, solve it exactly::

    f(x) = c ^ (w . x)          AFFINE      n+2 execs, exact
    degree <= 2                 QUADRATIC   solve_low: <= 1+n+C(n,2) execs, exact
    anything else               HIGH

Identities (all over GF(2), ``^`` is XOR):

    BLR        f(x)^f(y)^f(z)^f(x^y^z) == 0         for every affine f
    2nd deriv  f(x)^f(x^a)^f(x^b)^f(x^a^b)          constant in x iff deg <= 2

Limits, stated so callers do not over-trust a verdict:

- A verdict describes the *sampled* behaviour.  A rare-event function (an
  equality gate, ``f = [x == K]``) is indistinguishable from a constant by
  random probing and may land in any class (measured over 300 seeds,
  n=8: constant 208, quadratic 46, high 42, affine 4); that is cmplog's job,
  not this module's.  ``classify_exact`` reads all ``2^n`` points and has no
  blind spot, for small ``n`` only.  ``solve_affine``
  re-executes its answer, so a wrong verdict costs one exec, never a wrong
  result.
- A nondeterministic edge breaks every identity above.  It is reported as
  ``NOISY`` (re-reading a sample changed the bit) rather than guessed at.
- Each trial's false-pass probability for a degree-3 function is about 1/2,
  so ``TRIALS`` trials leave ~2**-TRIALS.

``solve_affine`` solves ``w . x == want ^ c`` directly from the unit vectors
instead of going through :class:`~fuzzer_tool.core.xor_map_solver.
IncrementalXorMapSolver`: a single output bit needs no elimination.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np

from fuzzer_tool.core import anf

# BLR and second-derivative trials per classification.
TRIALS = 24

# Random samples read twice for the stability check, besides 0 and all-ones.
STABILITY_SAMPLES = 4

BitFn = Callable[[int], int]

# Exact ANF degree -> verdict; anything above 2 is HIGH.
_BY_DEGREE = {0: "CONSTANT", 1: "AFFINE", 2: "QUADRATIC"}

# solve_low's layer: a nonconstant f of degree <= 2 flips at weight <= 2.
LOW_DEGREE = 2


class Degree(Enum):
    CONSTANT = "constant"
    AFFINE = "affine"
    QUADRATIC = "quadratic"
    HIGH = "high"
    NOISY = "noisy"


def blr_defect(f: BitFn, x: int, y: int, z: int) -> int:
    """0 for every affine ``f``; 1 is a proof that ``f`` is not affine."""
    return (f(x) ^ f(y) ^ f(z) ^ f(x ^ y ^ z)) & 1


def second_derivative(f: BitFn, x: int, a: int, b: int) -> int:
    """``D_a D_b f (x)``: constant over ``x`` iff ``deg f <= 2``."""
    return (f(x) ^ f(x ^ a) ^ f(x ^ b) ^ f(x ^ a ^ b)) & 1


def _stable(f: BitFn, n_bits: int, rng) -> bool:
    # Read each sample twice; a changed bit means the edge is noisy.
    mask = (1 << n_bits) - 1
    xs = [0, mask] + [rng.randrange(mask + 1) for _ in range(STABILITY_SAMPLES)]
    return all(f(x) & 1 == f(x) & 1 for x in xs)


def _is_constant(f: BitFn, n_bits: int) -> bool:
    # For affine f, w == 0 iff every unit vector leaves f(0) unchanged.
    c = f(0) & 1
    return all(f(1 << i) & 1 == c for i in range(n_bits))


def _is_affine(f: BitFn, n_bits: int, rng, trials: int) -> bool:
    # Any BLR defect refutes affinity.
    size = 1 << n_bits
    for _ in range(trials):
        x, y, z = rng.randrange(size), rng.randrange(size), rng.randrange(size)
        if blr_defect(f, x, y, z):
            return False
    return True


def _is_quadratic(f: BitFn, n_bits: int, rng, trials: int) -> bool:
    # D_a D_b f must agree at two random points for deg <= 2.
    mask = (1 << n_bits) - 1
    size = mask + 1
    if mask < 2:
        return True
    for _ in range(trials):
        a = 1 + rng.randrange(mask)
        b = a % mask + 1  # nonzero and != a
        x1, x2 = rng.randrange(size), rng.randrange(size)
        if second_derivative(f, x1, a, b) != second_derivative(f, x2, a, b):
            return False
    return True


def classify(f: BitFn, n_bits: int, rng, trials: int = TRIALS) -> Degree:
    """Classify ``f`` on ``n_bits`` input bits; ``rng`` is a ``RandPool``."""
    if n_bits < 1:
        raise ValueError(f"n_bits must be positive, got {n_bits}")

    if not _stable(f, n_bits, rng):
        return Degree.NOISY
    if _is_affine(f, n_bits, rng, trials):
        return Degree.CONSTANT if _is_constant(f, n_bits) else Degree.AFFINE
    if _is_quadratic(f, n_bits, rng, trials):
        return Degree.QUADRATIC
    return Degree.HIGH


def solve_affine(f: BitFn, n_bits: int, want: int) -> int | None:
    """Input ``x`` with ``f(x) == want`` for affine ``f``, else ``None``.

    ``c = f(0)``; ``w_i = f(e_i) ^ c``.  Cost: ``n_bits + 2`` execs (the last
    re-executes the answer, so a misclassified ``f`` returns ``None``).
    """
    want &= 1
    c = f(0) & 1
    if c == want:
        return 0
    for i in range(n_bits):
        if (f(1 << i) & 1) ^ c:
            x = 1 << i
            return x if f(x) & 1 == want else None
    return None


def classify_exact(f: BitFn, n_bits: int) -> Degree:
    """Exact verdict from the full truth table: ``2^n_bits`` execs, no noise check.

    Example: ``[x == K]`` is degree ``n`` (HIGH), never CONSTANT.
    """
    if not 1 <= n_bits <= anf.MAX_BITS:
        raise ValueError(f"n_bits must be in [1, {anf.MAX_BITS}], got {n_bits}")

    table = np.array([f(x) & 1 for x in range(1 << n_bits)], dtype=np.uint64)
    return Degree[_BY_DEGREE.get(anf.degree(table), "HIGH")]


def solve_low(f: BitFn, n_bits: int, want: int, degree: int = LOW_DEGREE) -> int | None:
    """Input ``x`` with ``f(x) == want`` for ``deg f <= degree``, else ``None``.

    Scans the weight ``<= degree`` layer, lowest weight first: complete for
    such ``f`` (minimal-monomial flip, ``core.anf``), and every return is a
    read value, so a higher-degree ``f`` costs execs, never a wrong answer.
    """
    want &= 1
    for x in anf.layer(n_bits, degree):
        if f(x) & 1 == want:
            return x
    return None
