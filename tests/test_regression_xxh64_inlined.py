"""XXH64 rounds, rotates and merges inlined (no helper call per word).

Persisted corpus names depend on these bits, so the inlined loops must
match the pre-change helpers exactly. Oracle: the pre-change ``_bulk`` and
``_tail``, frozen here; ``tests/test_hashing.py`` holds the reference
vectors.
"""

import struct

from fuzzer_tool.core import hashing as h
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 0x5EED
# Seeds that wrap: 0, 2**64 - 1 (v4 = seed - P1 underflows at small seeds).
_SEEDS = (0, 1, h._P1 - 1, h._U64, 0x9E3779B1)
_LENS = (0, 1, 3, 4, 7, 8, 31, 32, 33, 63, 64, 95, 96, 97, 200, 4096 + 13)
_ROT_LANES = (1, 7, 12, 18)


def _rotl(x, r):
    """Pre-change helper, verbatim (removed from hashing.py)."""
    return ((x << r) | (x >> (64 - r))) & h._U64


def _round(acc, lane):
    return (_rotl((acc + lane * h._P2) & h._U64, 31) * h._P1) & h._U64


def _merge(acc, lane):
    return (((acc ^ _round(0, lane)) * h._P1) + h._P4) & h._U64


def _oracle_bulk(data, seed):
    """Pre-change _bulk, verbatim."""
    v1 = (seed + h._P1 + h._P2) & h._U64
    v2 = (seed + h._P2) & h._U64
    v3 = seed & h._U64
    v4 = (seed - h._P1) & h._U64
    used = len(data) - len(data) % h._STRIPE
    for a, b, c, d in struct.iter_unpack(h._LANE_FMT, data[:used]):
        v1 = (_rotl((v1 + a * h._P2) & h._U64, 31) * h._P1) & h._U64
        v2 = (_rotl((v2 + b * h._P2) & h._U64, 31) * h._P1) & h._U64
        v3 = (_rotl((v3 + c * h._P2) & h._U64, 31) * h._P1) & h._U64
        v4 = (_rotl((v4 + d * h._P2) & h._U64, 31) * h._P1) & h._U64
    acc = 0
    for lane, rot in zip((v1, v2, v3, v4), _ROT_LANES, strict=True):
        acc += _rotl(lane, rot)
    acc &= h._U64
    for lane in (v1, v2, v3, v4):
        acc = _merge(acc, lane)
    return acc, used


def _oracle_tail(acc, data, pos):
    """Pre-change _tail, verbatim."""
    n = len(data)
    while pos + h._WORD64 <= n:
        word = int.from_bytes(data[pos : pos + h._WORD64], "little")
        acc ^= _round(0, word)
        acc = (_rotl(acc, 27) * h._P1 + h._P4) & h._U64
        pos += h._WORD64
    if pos + h._WORD32 <= n:
        word = int.from_bytes(data[pos : pos + h._WORD32], "little")
        acc ^= (word * h._P1) & h._U64
        acc = (_rotl(acc, 23) * h._P2 + h._P3) & h._U64
        pos += h._WORD32
    while pos < n:
        acc ^= (data[pos] * h._P5) & h._U64
        acc = (_rotl(acc, 11) * h._P1) & h._U64
        pos += 1
    return acc


def _cases():
    rng = RandPool(seed=_SEED)
    # All-0xFF words push every product to its widest.
    for n in _LENS:
        for data in (rng.randbytes(n), b"\xff" * n):
            for seed in _SEEDS:
                yield data, seed


def test_control_oracle_matches_itself():
    """Rule 46: the frozen helpers agree with themselves."""
    for data, seed in _cases():
        assert _oracle_bulk(data, seed) == _oracle_bulk(bytes(data), seed)


def test_regression_bulk_matches_oracle():
    """Falsification: same state and consumed length for every case."""
    for data, seed in _cases():
        assert h._bulk(data, seed) == _oracle_bulk(data, seed)


def test_regression_tail_matches_oracle():
    """Adversarial: every tail start (word, half-word, byte paths) and the
    widest accumulator."""
    for data, _seed in _cases():
        for pos in range(min(len(data), 40)):
            for acc in (0, h._U64, h._P5):
                assert h._tail(acc, data, pos) == _oracle_tail(acc, data, pos)
