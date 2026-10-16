"""Forge distinct inputs with one XXH64 (``key16``) digest.

Independent of ``core/hashing.py``: constants from the XXH64 spec. Each lane
round ``acc' = rotl(acc + lane * P2, 31) * P1`` inverts per lane, so after
any first stripe the second stripe can drive all four lanes back to the
reference state (orlp.net, "Breaking CityHash64, MurmurHash2/3, ...")::

    stripe 1: arbitrary prefix    -> lanes diverge
    stripe 2: solved per lane     -> lanes == reference, digest equal
"""

from __future__ import annotations

import struct

_U64 = (1 << 64) - 1
_P1 = 0x9E3779B185EBCA87
_P2 = 0xC2B2AE3D27D4EB4F
_P1_INV = pow(_P1, -1, 1 << 64)
_P2_INV = pow(_P2, -1, 1 << 64)
_STRIPE = 32
_LANES = "<4Q"
_ROT = 31


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _U64


def _round(acc: int, lane: int) -> int:
    return (_rotl((acc + lane * _P2) & _U64, _ROT) * _P1) & _U64


def _unround(acc: int, out: int) -> int:
    """The lane with ``_round(acc, lane) == out``."""
    return ((_rotl((out * _P1_INV) & _U64, 64 - _ROT) - acc) * _P2_INV) & _U64


def _init_lanes(seed: int = 0) -> list[int]:
    return [(seed + _P1 + _P2) & _U64, (seed + _P2) & _U64, seed, (seed - _P1) & _U64]


def colliding(ref: bytes, prefixes: list[bytes]) -> list[bytes]:
    """*ref* plus one forged input per 32-byte prefix, all with ``key16(ref)``.

    *ref* must be >= 64 bytes; bytes past the first two stripes are shared.
    """
    assert len(ref) >= 2 * _STRIPE
    v0 = _init_lanes()
    l1 = struct.unpack(_LANES, ref[:_STRIPE])
    l2 = struct.unpack(_LANES, ref[_STRIPE : 2 * _STRIPE])
    target = [_round(_round(v, a), b) for v, a, b in zip(v0, l1, l2, strict=True)]

    out = [ref]
    for p in prefixes:
        assert len(p) == _STRIPE
        mid = [_round(v, a) for v, a in zip(v0, struct.unpack(_LANES, p), strict=True)]
        s2 = struct.pack(_LANES, *(_unround(m, t) for m, t in zip(mid, target, strict=True)))
        out.append(p + s2 + ref[2 * _STRIPE :])
    return out
