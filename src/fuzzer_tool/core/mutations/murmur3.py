"""MurmurHash3 (x86, 32-bit) and the block chain the ``murmurhash3`` op writes.

The chain fills a region one 4-byte block at a time, each block the hash of
everything before it::

    b0 = murmur3_32(prefix, seed)
    bt = murmur3_32(b0 .. b(t-1), seed ^ 4t)      t >= 1

Hashed directly that is O(T^2) Python block steps (T = length / 4): ~500k
for a 4 KB region. ``murmur3_chain`` runs the T hashes side by side instead:
once block j is known, every hash still pending absorbs it in one in-place
numpy step over a uint32 state vector, and the hash that just consumed its
last block is finalized into block j + 1::

    state[t] after j steps = h of seed ^ 4t over blocks 0..j-1
    step j:  state[j+1:] <- mix(state[j+1:], b_j);  b(j+1) = fmix(state[j+1])
"""

import struct

import numpy as np

_MASK = 0xFFFFFFFF
_C1 = 0xCC9E2D51
_C2 = 0x1B873593
_ROUND_ADD = 0xE6546B64
_FMIX1 = 0x85EBCA6B
_FMIX2 = 0xC2B2AE35
_BLOCK = 4

# Below this many blocks the per-step numpy calls lose to plain Python over
# pre-mixed blocks. Measured, with and without the in-process ASAN preload:
# 32 bytes scalar 0.01 ms vs vector 0.05 ms; even at 256 bytes (64 blocks).
_VECTOR_MIN_BLOCKS = 64


def _mix_k(k: int) -> int:
    """Scramble one block before it is folded into the state."""
    k = (k * _C1) & _MASK
    k = ((k << 15) | (k >> 17)) & _MASK
    return (k * _C2) & _MASK


def _fmix(h: int, length: int) -> int:
    """Length fold and final avalanche."""
    h ^= length
    h ^= h >> 16
    h = (h * _FMIX1) & _MASK
    h ^= h >> 13
    h = (h * _FMIX2) & _MASK
    return h ^ (h >> 16)


def murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3_x86_32 of *data*."""
    h = seed & _MASK
    nblocks = len(data) // _BLOCK
    for (k,) in struct.iter_unpack("<I", data[: nblocks * _BLOCK]):
        h ^= _mix_k(k)
        h = ((h << 13) | (h >> 19)) & _MASK
        h = (h * 5 + _ROUND_ADD) & _MASK

    # Tail: 1-3 trailing bytes, little-endian, no rotate/add round.
    tail = int.from_bytes(data[nblocks * _BLOCK :], "little")
    if tail:
        h ^= _mix_k(tail)
    return _fmix(h, len(data))


def murmur3_chain(prefix: bytes, length: int, seed: int) -> bytes:
    """*length* bytes of chained MurmurHash3 blocks (see module docstring)."""
    n_blocks = -(-length // _BLOCK)
    if n_blocks == 0:
        return b""

    blocks = [murmur3_32(prefix, seed)]
    if n_blocks < _VECTOR_MIN_BLOCKS:
        _chain_scalar(blocks, n_blocks, seed)
    else:
        _chain_vector(blocks, n_blocks, seed)
    return struct.pack(f"<{n_blocks}I", *blocks)[:length]


def _chain_scalar(blocks: list[int], n_blocks: int, seed: int) -> None:
    """Append blocks 1..T-1, re-hashing the mixed-block prefix each time."""
    mixed: list[int] = []
    for t in range(1, n_blocks):
        mixed.append(_mix_k(blocks[t - 1]))
        h = (seed ^ (t * _BLOCK)) & _MASK
        for k in mixed:
            h ^= k
            h = ((h << 13) | (h >> 19)) & _MASK
            h = (h * 5 + _ROUND_ADD) & _MASK
        blocks.append(_fmix(h, t * _BLOCK))


def _chain_vector(blocks: list[int], n_blocks: int, seed: int) -> None:
    """Append blocks 1..T-1, advancing every pending hash per known block."""
    # Pending hashes t = 1..T-1, seeded seed ^ 4t; tmp holds the rotate.
    state = (seed ^ (np.arange(1, n_blocks, dtype=np.int64) * _BLOCK)).astype(np.uint32)
    tmp = np.empty_like(state)
    for j in range(n_blocks - 1):
        live, scratch = state[j:], tmp[j:]
        live ^= np.uint32(_mix_k(blocks[j]))
        np.left_shift(live, 13, out=scratch)
        live >>= 19
        live |= scratch
        live *= np.uint32(5)
        live += np.uint32(_ROUND_ADD)
        blocks.append(_fmix(int(live[0]), (j + 1) * _BLOCK))
