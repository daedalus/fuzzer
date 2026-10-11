"""Inverses of bijective 32-bit integer mixers (Murmur3 fmix32, lowbias32, xxh32 avalanche).

Each mixer is xorshift-right and odd-constant multiply steps, so each step
inverts exactly and the whole map has one preimage per value (no search):

    y = x ^ (x >> s)   <->   x = y ^ (y >> s) ^ (y >> 2s) ^ ...
    y = x * c mod 2^32 <->   x = y * c^-1 mod 2^32        (c odd)

Used by ``rq_encodings.MixerEncoder``: ``if (mix(field) == CONST)`` becomes
``field = inverse(CONST)``.  Hard Rule 51: stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable

_BITS = 32
_MASK = (1 << _BITS) - 1

# Murmur3 fmix32: x^=x>>16; x*=C1; x^=x>>13; x*=C2; x^=x>>16
_FMIX_C1 = 0x85EBCA6B
_FMIX_C2 = 0xC2B2AE35
_FMIX_SHIFTS = (16, 13, 16)

# lowbias32 (C. Wellons): x^=x>>16; x*=C1; x^=x>>15; x*=C2; x^=x>>16
_LOWBIAS_C1 = 0x7FEB352D
_LOWBIAS_C2 = 0x846CA68B
_LOWBIAS_SHIFTS = (16, 15, 16)

# xxh32 avalanche (PRIME32_2, PRIME32_3): x^=x>>15; x*=P2; x^=x>>13; x*=P3; x^=x>>16
_XXH_C1 = 0x85EBCA77
_XXH_C2 = 0xC2B2AE3D
_XXH_SHIFTS = (15, 13, 16)


def unxorshift_r(y: int, shift: int) -> int:
    """Invert ``x ^ (x >> shift)`` on a 32-bit value."""
    x = y
    t = y >> shift
    while t:
        x ^= t
        t >>= shift
    return x


def mul_inv(c: int) -> int:
    """Inverse of odd *c* mod 2^32."""
    return pow(c, -1, 1 << _BITS)


def _inv_mixer(y: int, inv1: int, inv2: int, shifts: tuple[int, int, int]) -> int:
    """Undo ``x^=x>>s0; x*=c1; x^=x>>s1; x*=c2; x^=x>>s2`` (last step first).

    Takes the constants' inverses, computed once at import: ``pow`` per call
    dominated the cost of a pair's first touch.
    """
    s0, s1, s2 = shifts
    x = unxorshift_r(y & _MASK, s2)
    x = (x * inv2) & _MASK
    x = unxorshift_r(x, s1)
    x = (x * inv1) & _MASK
    return unxorshift_r(x, s0)


_FMIX_INV = (mul_inv(_FMIX_C1), mul_inv(_FMIX_C2))
_LOWBIAS_INV = (mul_inv(_LOWBIAS_C1), mul_inv(_LOWBIAS_C2))
_XXH_INV = (mul_inv(_XXH_C1), mul_inv(_XXH_C2))


def fmix32_inv(h: int) -> int:
    return _inv_mixer(h, *_FMIX_INV, _FMIX_SHIFTS)


def lowbias32_inv(h: int) -> int:
    return _inv_mixer(h, *_LOWBIAS_INV, _LOWBIAS_SHIFTS)


def xxh32_aval_inv(h: int) -> int:
    return _inv_mixer(h, *_XXH_INV, _XXH_SHIFTS)


#: name -> inverse, in registration order.
MIXERS: dict[str, Callable[[int], int]] = {
    "fmix32": fmix32_inv,
    "lowbias32": lowbias32_inv,
    "xxh32_aval": xxh32_aval_inv,
}
