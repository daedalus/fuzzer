"""Meet-in-the-middle preimages of 4-byte inputs under 32-bit byte-wise hashes.

A byte hash updates a 32-bit state per byte, ``h_i = step(h_{i-1}, c_i)``, with
``step(., c)`` bijective for every byte ``c``.  Four bytes then split in two::

    forward   h2 = step(step(init, a), b)        2^16 (a, b), tabulated
    backward  h2 = step_inv(step_inv(final_inv(t), d), c)   2^16 (c, d)

and a preimage of ``t`` is every ``(a, b, c, d)`` whose ``h2`` agree.  Cost is
2^16 per side, vectorised, and the result is cached per (family, target).

Families (all 32-bit): FNV-1, djb2, sdbm and Jenkins one-at-a-time with its
final avalanche.  FNV-1a already has its own encoder (``rq_encodings``).  4 bytes
-> 32 bits is not a bijection: ~1 preimage on average, some targets have none.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from fuzzer_tool.core.hash_inverse import mul_inv, unxorshift_r
from fuzzer_tool.core.lru import LRUCache

FIELD_BYTES = 4
HALF = 1 << 16
_MASK = 0xFFFFFFFF
# uint32 arrays wrap mod 2^32 on their own: no masking, ~10x faster than uint64.
_U32 = np.uint32
_BYTE = _U32(0xFF)
_BYTE_SHIFT = _U32(8)
# Bounded like the FNV-1a cache so encode() stays O(1) after the first inversion.
_CACHE_MAX = 4096
# Prefilter: one byte per low-22-bit residue of the forward keys (4 MB per family).
# 2^16 keys fill ~1.6% of it, so ~98% of backward candidates skip the binary search.
_PRE_BITS = 22
_PRE_MASK = _U32((1 << _PRE_BITS) - 1)

_FNV_INIT = 2166136261
_FNV_PRIME = 16777619
_DJB2_INIT = 5381
_DJB2_MUL = 33
_SDBM_MUL = 65599
# Jenkins OAAT: h += h << 10; h ^= h >> 6 per byte; final h += h << 3; h ^= h >> 11; h += h << 15.
_OAAT_ADD = 1 + (1 << 10)
_OAAT_XS = 6
_OAAT_FIN = ((1 + (1 << 3)), 11, (1 + (1 << 15)))

Arr = np.ndarray


class ByteHashSpec(NamedTuple):
    """One hash family: init state, vectorised step and its inverse, final inverse."""

    name: str
    init: int
    step: Callable[[Arr, Arr], Arr]  # (state, byte) -> state, uint32 arrays
    step_inv: Callable[[Arr, Arr], Arr]
    final_inv: Callable[[int], int]


def _identity(h: int) -> int:
    return h


def _mul(h: Arr, k: int) -> Arr:
    return h * _U32(k)


def _unxs_arr(h: Arr, shift: int) -> Arr:
    # Inverse of h ^= h >> shift on 32 bits (shifts accumulate until they pass the width).
    out, s = h.copy(), shift
    while s < 32:
        out ^= h >> _U32(s)
        s += shift
    return out


_FNV1_INV = pow(_FNV_PRIME, -1, 1 << 32)
_DJB2_INV = pow(_DJB2_MUL, -1, 1 << 32)
_SDBM_INV = pow(_SDBM_MUL, -1, 1 << 32)
_OAAT_ADD_INV = mul_inv(_OAAT_ADD)


def _oaat_step(h: Arr, c: Arr) -> Arr:
    h = _mul(h + c, _OAAT_ADD)
    return h ^ (h >> _U32(_OAAT_XS))


def _oaat_step_inv(h: Arr, c: Arr) -> Arr:
    h = _unxs_arr(h, _OAAT_XS)
    h = _mul(h, _OAAT_ADD_INV)
    return h - c


def _oaat_final_inv(h: int) -> int:
    m1, sh, m2 = _OAAT_FIN
    h = (h * mul_inv(m2)) & _MASK
    h = unxorshift_r(h, sh)
    return (h * mul_inv(m1)) & _MASK


SPECS: dict[str, ByteHashSpec] = {
    "fnv1": ByteHashSpec(
        "fnv1",
        _FNV_INIT,
        lambda h, c: _mul(h, _FNV_PRIME) ^ c,
        lambda h, c: _mul(h ^ c, _FNV1_INV),
        _identity,
    ),
    "djb2": ByteHashSpec(
        "djb2",
        _DJB2_INIT,
        lambda h, c: _mul(h, _DJB2_MUL) + c,
        lambda h, c: _mul(h - c, _DJB2_INV),
        _identity,
    ),
    "sdbm": ByteHashSpec(
        "sdbm",
        0,
        lambda h, c: _mul(h, _SDBM_MUL) + c,
        lambda h, c: _mul(h - c, _SDBM_INV),
        _identity,
    ),
    "oaat": ByteHashSpec("oaat", 0, _oaat_step, _oaat_step_inv, _oaat_final_inv),
}

_cache: LRUCache = LRUCache(_CACHE_MAX)


@functools.cache
def _forward_table(spec: ByteHashSpec) -> tuple[Arr, Arr, Arr]:
    # Sorted h2 keys over all (a, b), the packed a | b << 8, and the residue prefilter.
    ab = np.arange(HALF, dtype=_U32)
    h = spec.step(np.full(HALF, spec.init, dtype=_U32), ab & _BYTE)
    h = spec.step(h, ab >> _BYTE_SHIFT)
    order = np.argsort(h, kind="stable")
    pre = np.zeros(1 << _PRE_BITS, dtype=np.uint8)
    pre[h & _PRE_MASK] = 1
    return h[order], ab[order], pre


def preimages(spec: ByteHashSpec, target: int) -> list[bytes]:
    """All 4-byte inputs ``x`` with ``hash(x) == target`` (possibly none)."""
    key = (spec.name, target)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    keys, packed, pre = _forward_table(spec)
    cd = np.arange(HALF, dtype=_U32)
    c, d = cd & _BYTE, cd >> _BYTE_SHIFT
    h4 = np.full(HALF, spec.final_inv(target & _MASK), dtype=_U32)
    h2 = spec.step_inv(spec.step_inv(h4, d), c)

    # Prefilter, then one searchsorted on the survivors; equal h2 can come from
    # several (a, b), so walk each run.
    cand = np.nonzero(pre[h2 & _PRE_MASK])[0]
    lo = np.searchsorted(keys, h2[cand])
    at = np.minimum(lo, HALF - 1)
    out: list[bytes] = []
    for k in np.nonzero(keys[at] == h2[cand])[0].tolist():
        i, j = int(cand[k]), int(lo[k])
        while j < HALF and keys[j] == h2[i]:
            ab = int(packed[j])
            out.append(bytes((ab & 0xFF, ab >> 8, int(c[i]), int(d[i]))))
            j += 1
    _cache[key] = out
    return out
