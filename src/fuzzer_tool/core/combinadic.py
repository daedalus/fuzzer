"""Rank/unrank of m-permutations and m-combinations (Lehmer code, combinadics).

An m-tuple swap (``_swap_tuple``) draws from ``n!/(n-m)!`` ordered tuples or
``C(n,m)`` index sets -- ~3e16 for n=2000, m=5. Ranking maps each tuple to an
integer in ``[0, total)`` and back, so callers can:

- enumerate without repeats by walking indices ``0..total-1``;
- sample uniformly *without replacement* in O(count) memory, never
  materialising the space (``sample_perms`` / ``sample_combs``).

Order is lexicographic, matching ``itertools.permutations`` /
``itertools.combinations`` over ``range(n)``.

Not wired into a mutator yet: whether m>2 swaps pay at all is unmeasured
(``docs/handover/handover_combinatorics_permutations_2026-09-02.md`` §1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

_LIMB_BITS = 32
_LIMB_MAX = (1 << _LIMB_BITS) - 1
# Extra random bits over the range's width: modulo bias <= 2**-64.
_BIAS_BITS = 64


def perm_count(n: int, m: int) -> int:
    """Number of ordered m-tuples of distinct elements of ``range(n)``."""
    if m < 0 or m > n:
        return 0
    return math.perm(n, m)


def comb_count(n: int, m: int) -> int:
    """Number of m-element subsets of ``range(n)``."""
    if m < 0 or m > n:
        return 0
    return math.comb(n, m)


def _check_shape(n: int, m: int) -> None:
    if m < 0 or m > n:
        raise ValueError(f"need 0 <= m <= n, got n={n} m={m}")


def unrank_perm(index: int, n: int, m: int) -> tuple[int, ...]:
    """The *index*-th m-permutation of ``range(n)`` in lexicographic order."""
    _check_shape(n, m)
    if not 0 <= index < perm_count(n, m):
        raise ValueError(f"index {index} outside [0, {perm_count(n, m)})")

    # Mixed radix: digit i has radix n-i and weight perm(n-i-1, m-i-1).
    avail = list(range(n))
    out: list[int] = []
    for i in range(m):
        weight = math.perm(n - i - 1, m - i - 1)
        digit, index = divmod(index, weight)
        out.append(avail.pop(digit))
    return tuple(out)


def rank_perm(perm: Sequence[int], n: int) -> int:
    """Inverse of ``unrank_perm``; ``m`` is ``len(perm)``."""
    m = len(perm)
    _check_shape(n, m)
    if len(set(perm)) != m or any(not 0 <= x < n for x in perm):
        raise ValueError(f"not distinct elements of range({n}): {tuple(perm)}")

    avail = list(range(n))
    index = 0
    for i, x in enumerate(perm):
        pos = avail.index(x)
        index += pos * math.perm(n - i - 1, m - i - 1)
        avail.pop(pos)
    return index


def unrank_comb(index: int, n: int, m: int) -> tuple[int, ...]:
    """The *index*-th m-combination of ``range(n)`` in lexicographic order."""
    _check_shape(n, m)
    total = comb_count(n, m)
    if not 0 <= index < total:
        raise ValueError(f"index {index} outside [0, {total})")

    out: list[int] = []
    c = 0
    for i in range(m):
        # Skip every block of combinations that start with a smaller element.
        while True:
            block = math.comb(n - c - 1, m - i - 1)
            if index < block:
                break
            index -= block
            c += 1
        out.append(c)
        c += 1
    return tuple(out)


def rank_comb(comb: Sequence[int], n: int) -> int:
    """Inverse of ``unrank_comb``; ``m`` is ``len(comb)``."""
    m = len(comb)
    _check_shape(n, m)
    if any(not 0 <= x < n for x in comb) or any(
        a >= b for a, b in zip(comb, comb[1:], strict=False)
    ):
        raise ValueError(f"not strictly increasing in range({n}): {tuple(comb)}")

    index = 0
    prev = -1
    for i, x in enumerate(comb):
        for skipped in range(prev + 1, x):
            index += math.comb(n - skipped - 1, m - i - 1)
        prev = x
    return index


def _below(rng: Any, bound: int) -> int:
    """Near-uniform int in ``[0, bound)`` for any *bound*, via 32-bit limbs.

    ``RandPool.randrange`` is one 32-bit draw modulo *bound*: for a bound past
    2**32 it can never return most of the range. Stacking limbs (plus
    ``_BIAS_BITS`` spare bits) keeps the modulo bias negligible.
    """
    limbs = (bound.bit_length() + _BIAS_BITS + _LIMB_BITS - 1) // _LIMB_BITS
    acc = 0
    for _ in range(limbs):
        acc = (acc << _LIMB_BITS) | rng.randint(0, _LIMB_MAX)
    return acc % bound


def sample_indices(total: int, count: int, rng: Any) -> list[int]:
    """*count* distinct uniform indices in ``[0, total)``; O(count) memory.

    Floyd's algorithm over Python ints, so *total* may exceed any machine word.
    """
    if count < 0 or count > total:
        raise ValueError(f"cannot draw {count} distinct indices from {total}")

    chosen: set[int] = set()
    for j in range(total - count, total):
        t = _below(rng, j + 1)
        chosen.add(j if t in chosen else t)

    # Floyd yields an unordered set; shuffle so prefixes are unbiased too.
    out = sorted(chosen)
    for i in range(len(out) - 1, 0, -1):
        k = _below(rng, i + 1)
        out[i], out[k] = out[k], out[i]
    return out


def sample_perms(n: int, m: int, count: int, rng: Any) -> list[tuple[int, ...]]:
    """*count* distinct uniform m-permutations of ``range(n)``."""
    _check_shape(n, m)
    return [unrank_perm(i, n, m) for i in sample_indices(perm_count(n, m), count, rng)]


def sample_combs(n: int, m: int, count: int, rng: Any) -> list[tuple[int, ...]]:
    """*count* distinct uniform m-combinations of ``range(n)``."""
    _check_shape(n, m)
    return [unrank_comb(i, n, m) for i in sample_indices(comb_count(n, m), count, rng)]
