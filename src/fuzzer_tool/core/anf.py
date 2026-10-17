"""Algebraic normal form (Zhegalkin polynomial) of maps over GF(2)^n.

Every ``f: {0,1}^n -> {0,1}`` is one XOR of AND-monomials::

    f(x) = XOR_S a_S * prod_{i in S} x_i        a_S = XOR_{T ⊆ S} f(T)

The Möbius transform computes every ``a_S`` from the truth table in
``n * 2^n`` XORs and is its own inverse. A packed table (one uint64 per
``x``) transforms all output bits at once. ``deg f`` is the largest ``|S|``
with ``a_S != 0``.

Low degree from few points: when ``deg f <= d``, each ``a_S`` with
``|S| <= d`` needs ``f`` only on points of weight ``<= d`` (``T ⊆ S``),
so ``sum_{j<=d} C(n, j)`` execs fix it (137 for ``n=16, d=2``). The inverse
transform then predicts ``f`` on all ``2^n`` points::

    f on layer ──moebius──▶ a_S, |S| <= d ──drop |S| > d──▶ moebius ──▶ f everywhere

Minimal monomial: for an inclusion-minimal ``S`` with ``a_S = 1``,
``f(1_S) = f(0) ^ 1``; a nonconstant ``f`` of degree ``d`` flips inside
the weight ``<= d`` layer.

Following Leviathan, "Encoding Bitwise Functions as Polynomials" (2024):
the article builds this form symbolically; here it is read from execs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np

# Largest field one table may cover: 2^16 uint64 = 512 KiB (Hard Rule 54).
MAX_BITS = 16

# x -> packed output bits; None when the observable was not reached.
Observe = Callable[[int], int | None]


def _bits_of(size: int) -> int:
    n = size.bit_length() - 1
    if size < 1 or 1 << n != size:
        raise ValueError(f"table length must be a power of two, got {size}")
    return n


def _check_bits(n: int) -> None:
    if not 0 <= n <= MAX_BITS:
        raise ValueError(f"n must be in [0, {MAX_BITS}], got {n}")


def moebius(table: np.ndarray) -> np.ndarray:
    """ANF coefficients of a packed truth table (and back: an involution).

    Butterfly on bit ``i``: ``a[x | 1<<i] ^= a[x]``, all x at once.
    """
    out = np.array(table, dtype=np.uint64)
    n = _bits_of(len(out))
    for i in range(n):
        view = out.reshape(-1, 2, 1 << i)
        view[:, 1, :] ^= view[:, 0, :]
    return out


def _weights(n: int) -> np.ndarray:
    return np.bitwise_count(np.arange(1 << n, dtype=np.uint32))


def degree(table: np.ndarray) -> int:
    """Exact algebraic degree over every packed output bit; 0 when constant."""
    coeffs = moebius(table)
    weights = _weights(_bits_of(len(coeffs)))[coeffs != 0]
    return int(weights.max()) if weights.size else 0


def layer(n: int, d: int) -> list[int]:
    """Every n-bit point of weight ``<= d``, lowest weight first."""
    return [sum(1 << i for i in c) for j in range(min(d, n) + 1) for c in combinations(range(n), j)]


def fit_low(f: Observe, n: int, d: int, confirms: Sequence[int]) -> np.ndarray | None:
    """Predicted packed table of ``f`` assuming ``deg f <= d``; None if refuted.

    Probes the weight ``<= d`` layer, then every point in *confirms*; one
    disagreement or unreached point fails closed.
    """
    _check_bits(n)
    table = np.zeros(1 << n, dtype=np.uint64)
    for x in layer(n, d):
        v = f(x)
        if v is None:
            return None
        table[x] = v

    # Coefficients of weight > d are artefacts of the zero fill: drop them.
    coeffs = moebius(table)
    coeffs[_weights(n) > d] = 0
    pred = moebius(coeffs)

    mask = (1 << n) - 1
    for x in confirms:
        if f(x & mask) != int(pred[x & mask]):
            return None

    return pred
