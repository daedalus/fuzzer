"""Inverses of bijective integer mixers (Murmur3 fmix32, lowbias32, xxh32 avalanche, XXH64 lanes).

Each mixer is xorshift-right and odd-constant multiply steps, so each step
inverts exactly and the whole map has one preimage per value (no search):

    y = x ^ (x >> s)   <->   x = y ^ (y >> s) ^ (y >> 2s) ^ ...
    y = x * c mod 2^32 <->   x = y * c^-1 mod 2^32        (c odd)

Used by ``rq_encodings.MixerEncoder``: ``if (mix(field) == CONST)`` becomes
``field = inverse(CONST)``.  Hard Rule 51: stdlib only.

XXH64 second preimage (orlp.net, "Breaking CityHash64, MurmurHash2/3, ..."):
each lane round ``acc' = rotl(acc + lane * P2, 31) * P1`` inverts, so after
any first stripe a solved second stripe drives all four lanes back to the
reference state::

    stripe 1: arbitrary prefix    -> lanes diverge
    stripe 2: solved per lane     -> lanes == reference, digest equal
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence

from fuzzer_tool.core.hashing import _LANE_FMT, _P1, _P2, _STRIPE, _U64

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


def mul_inv(c: int, bits: int = _BITS) -> int:
    """Inverse of odd *c* mod 2^bits (default 32)."""
    return pow(c, -1, 1 << bits)


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


# ── XXH64 lanes ──────────────────────────────────────────────────────────
_BITS64 = 64
_ROT = 31
_P1_INV = mul_inv(_P1, _BITS64)
_P2_INV = mul_inv(_P2, _BITS64)
_REF_STRIPES = 2


def _round64(acc: int, lane: int) -> int:
    x = (acc + lane * _P2) & _U64
    return (((x << _ROT) | (x >> (_BITS64 - _ROT))) * _P1) & _U64


def xxh64_unround(acc: int, out: int) -> int:
    """The lane with ``round(acc, lane) == out``: undo *P1, rotl, +acc, *P2."""
    x = (out * _P1_INV) & _U64
    x = ((x >> _ROT) | (x << (_BITS64 - _ROT))) & _U64
    return ((x - acc) * _P2_INV) & _U64


def xxh64_collide(ref: bytes, prefixes: Sequence[bytes]) -> list[bytes]:
    """*ref* plus one forged input per 32-byte prefix, all with ``xxh64(ref)``.

    Forged inputs share *ref*'s length and every byte past its first two
    stripes; seed 0 only (``key16``).

    Raises:
        ValueError: *ref* is shorter than two stripes, or a prefix is not one.
    """
    if len(ref) < _REF_STRIPES * _STRIPE:
        raise ValueError(f"ref needs >= {_REF_STRIPES * _STRIPE} bytes, got {len(ref)}")
    bad = [len(p) for p in prefixes if len(p) != _STRIPE]
    if bad:
        raise ValueError(f"prefixes must be {_STRIPE} bytes, got {bad}")

    # Seed-0 lanes, then the reference state after two stripes.
    v0 = ((_P1 + _P2) & _U64, _P2, 0, (-_P1) & _U64)
    l1 = struct.unpack(_LANE_FMT, ref[:_STRIPE])
    l2 = struct.unpack(_LANE_FMT, ref[_STRIPE : _REF_STRIPES * _STRIPE])
    target = [_round64(_round64(v, a), b) for v, a, b in zip(v0, l1, l2, strict=True)]
    tail = ref[_REF_STRIPES * _STRIPE :]

    out = [ref]
    for p in prefixes:
        mid = [_round64(v, a) for v, a in zip(v0, struct.unpack(_LANE_FMT, p), strict=True)]
        s2 = struct.pack(_LANE_FMT, *map(xxh64_unround, mid, target))
        out.append(p + s2 + tail)
    return out
