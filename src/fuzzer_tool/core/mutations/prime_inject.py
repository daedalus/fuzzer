"""Number-theoretic boundary constants for primality / modular-arithmetic code.

Targets that test primality (RSA/DH parameter checks, ``BN_is_prime``-style
code, custom Miller-Rabin) or reduce modulo a prime rarely see inputs that
sit on the edges of those routines under flat-byte mutation.  This operator
overwrites an aligned 1/2/4/8-byte slot with one of:

- the largest primes at and just above each power-of-two boundary,
- Carmichael numbers and strong pseudoprimes (composites that fool Fermat
  or fixed-base Miller-Rabin tests),
- Mersenne and Fermat primes,
- the square and a semiprime of the largest primes below 2^(bits/2)
  (composites a trial-division bound or rho step must still catch).
- the fast-inverse-square-root magic constants (``0x5F3759DF`` for float32,
  ``0x5FE6EB50C7B537A9`` for float64): not number-theoretic, but they are
  word-size magic numbers a target that parses one from its input compares
  against, and this table is where aligned 4/8-byte constants live.

The table is built once from :mod:`fuzzer_tool.core.numtheory` (Hard Rule 51).
"""

from __future__ import annotations

from functools import cache

from fuzzer_tool.core import numtheory as nt
from fuzzer_tool.core.mutations.structured import _region, _splice

#: Slot widths in bytes, smallest first.
_WIDTHS = (1, 2, 4, 8)
_BITS_PER_BYTE = 8
#: Width used for an empty buffer.
_EMPTY_WIDTH = 4

#: Carmichael numbers.
_CARMICHAEL = (561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 825265, 321197185)
#: Strong pseudoprimes: 2047 (base 2), 3215031751 (bases 2,3,5,7),
#: 3825123056546413051 (bases 2..23).
_STRONG_PSEUDOPRIMES = (2047, 3215031751, 3825123056546413051)
#: Mersenne exponents (2^k - 1 prime) and Fermat primes.
_MERSENNE_EXPONENTS = (3, 5, 7, 13, 17, 19, 31, 61)
_FERMAT_PRIMES = (3, 5, 17, 257, 65537)
#: Quake III fast inverse square root: ``i = MAGIC - (i >> 1)`` on the float's bits.
_RSQRT_MAGIC = (0x5F3759DF, 0x5FE6EB50C7B537A9)


def _boundary(bits: int) -> list[int]:
    """Primes on the 2^bits and 2^(bits-1) edges, below 2^bits."""
    top, half = 1 << bits, 1 << (bits - 1)
    found = [nt.prev_prime(top), nt.prev_prime(half), nt.next_prime(half)]
    return [v for v in found if v is not None]


def _composites(bits: int) -> list[int]:
    """Square and semiprime of the two largest primes below 2^(bits/2)."""
    p = nt.prev_prime(1 << (bits // 2))
    q = nt.prev_prime(p) if p else None
    if p is None or q is None:
        return []
    return [p * p, p * q]


@cache
def values_for(width: int) -> tuple[int, ...]:
    """Distinct table values that fit in *width* bytes, stable order."""
    bits = width * _BITS_PER_BYTE
    limit = 1 << bits
    vals = _boundary(bits) + _composites(bits)
    vals += [(1 << k) - 1 for k in _MERSENNE_EXPONENTS]
    vals += list(_FERMAT_PRIMES) + list(_CARMICHAEL) + list(_STRONG_PSEUDOPRIMES)
    vals += list(_RSQRT_MAGIC)
    return tuple(dict.fromkeys(v for v in vals if 0 < v < limit))


def prime_inject(data: bytes, rng) -> bytes:
    """Overwrite an aligned slot of *data* with a number-theory constant.

    Args:
        data: Input buffer.
        rng: ``RandPool`` (or ScriptedRng in tests).

    Returns:
        Buffer of the same length (empty input yields one 4-byte constant).
    """
    if not data:
        return values_for(_EMPTY_WIDTH)[0].to_bytes(_EMPTY_WIDTH, "big")

    widths = [w for w in _WIDTHS if w <= len(data)]
    width = widths[rng.randint(0, len(widths) - 1)]
    order = "big" if rng.randint(0, 1) else "little"
    table = values_for(width)
    value = table[rng.randint(0, len(table) - 1)]
    offset, length = _region(len(data), rng, min_len=width, align=width)
    if length < width:
        return data
    return _splice(data, offset, value.to_bytes(width, order))
