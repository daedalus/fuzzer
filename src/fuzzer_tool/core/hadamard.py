"""Hadamard matrices: Sylvester doubling and Paley I.

An order-n Hadamard matrix H has +/-1 entries and ``H H^T = n I``. Once
normalized (row 0 and column 0 all +1), columns 1..n-1 form a strength-2
orthogonal array: every column pair takes ++/+-/-+/-- exactly n/4 times.

Orders built: 1, ``q+1`` for prime ``q = 3 mod 4`` (Paley I: 4, 8, 12, 20,
24, 32, 44, 48, 60, 64, ...), and doublings of those (Sylvester). Others
(e.g. 28, 36) return None.

Uses: Plackett-Burman screening (``tools/lib/factorial_design.py``) and
``fxt.hadamard_oa`` flag-word rows.
"""

from __future__ import annotations

from functools import cache

from fuzzer_tool.core.numtheory import is_prime

_PALEY_MOD = 4
_PALEY_RESIDUE = 3


def _paley(q: int) -> list[list[int]]:
    """Paley I Hadamard matrix of order q+1 (q prime, q = 3 mod 4)."""
    squares = {(x * x) % q for x in range(1, q)}

    def chi(a: int) -> int:
        a %= q
        if a == 0:
            return 0
        return 1 if a in squares else -1

    # S = [[0, 1..1], [-1, Q]] is skew; H = S + I.
    h = [[1] + [1] * q]
    for i in range(q):
        h.append([-1] + [chi(j - i) + (1 if i == j else 0) for j in range(q)])
    return h


def _sylvester_double(h: list[list[int]]) -> list[list[int]]:
    """``[[H, H], [H, -H]]``: order 2n from order n."""
    return [r + r for r in h] + [r + [-v for v in r] for r in h]


def _base(n: int) -> list[list[int]] | None:
    """Order 1 or Paley I; None when *n* is neither."""
    if n == 1:
        return [[1]]
    if is_prime(n - 1) and (n - 1) % _PALEY_MOD == _PALEY_RESIDUE:
        return _paley(n - 1)
    return None


def hadamard(n: int) -> list[list[int]] | None:
    """Hadamard matrix of order *n*, or None if none is constructed here."""
    if n < 1:
        return None

    # Halve to a base order, then double back up: 16 -> 8 (Paley 7) -> 16.
    doublings = 0
    h = _base(n)
    while h is None:
        if n % 2:
            return None
        n //= 2
        doublings += 1
        h = _base(n)

    for _ in range(doublings):
        h = _sylvester_double(h)
    return h


@cache
def hadamard_rows(order: int) -> tuple[int, ...]:
    """Normalized Hadamard rows as ints, bit j set where entry j is -1.

    Row 0 is 0 (all +1) and bit 0 is clear in every row (column 0 all +1).
    E.g. order 4 -> ``(0b0000, 0b0110, 0b1100, 0b1010)``.

    Raises:
        ValueError: no matrix of that order is built here.
    """
    h = hadamard(order)
    if h is None:
        raise ValueError(f"no Hadamard matrix of order {order} built")

    # Flip rows so column 0 is +1, then columns so row 0 is +1.
    h = [[v * r[0] for v in r] for r in h]
    top = h[0]
    return tuple(
        sum(1 << j for j, (v, s) in enumerate(zip(r, top, strict=True)) if v * s < 0) for r in h
    )
