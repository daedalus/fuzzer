"""LZ4 ``_xxh32_py`` unpacks stripes with ``struct.iter_unpack``.

Four tuple indexings per 16-byte stripe cost 5-10%. Oracles: published
XXH32 vectors, and the pre-change function, frozen here.
"""

import struct

from fuzzer_tool.core.mutations import lz4 as m
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 0x32
_SEEDS = (0, 1, m._P1 - 1, 0xFFFFFFFF)
# Published XXH32 (seed 0) vectors.
_VECTORS = {
    b"": 0x02CC5D05,
    b"a": 0x550D7456,
    b"abc": 0x32D153FF,
    b"Nobody inspects the spammish repetition": 0xE2293B2F,
}


def _oracle(data, seed=0):
    """Pre-change _xxh32_py, verbatim."""
    n = len(data)
    pos = 0
    if n >= m._STRIPE:
        v1 = (seed + m._P1 + m._P2) & m._U32
        v2 = (seed + m._P2) & m._U32
        v3 = seed & m._U32
        v4 = (seed - m._P1) & m._U32
        nw = (n // m._STRIPE) * 4
        words = struct.unpack_from(f"<{nw}I", data)
        for i in range(0, nw, 4):
            v1 = (v1 + words[i] * m._P2) & m._U32
            v1 = (((v1 << 13) | (v1 >> 19)) * m._P1) & m._U32
            v2 = (v2 + words[i + 1] * m._P2) & m._U32
            v2 = (((v2 << 13) | (v2 >> 19)) * m._P1) & m._U32
            v3 = (v3 + words[i + 2] * m._P2) & m._U32
            v3 = (((v3 << 13) | (v3 >> 19)) * m._P1) & m._U32
            v4 = (v4 + words[i + 3] * m._P2) & m._U32
            v4 = (((v4 << 13) | (v4 >> 19)) * m._P1) & m._U32
        h = (
            ((v1 << 1) | (v1 >> 31))
            + ((v2 << 7) | (v2 >> 25))
            + ((v3 << 12) | (v3 >> 20))
            + ((v4 << 18) | (v4 >> 14))
        )
        pos = nw * m._WORD
    else:
        h = seed + m._P5
    h = (h + n) & m._U32
    while pos + m._WORD <= n:
        h = (h + struct.unpack_from("<I", data, pos)[0] * m._P3) & m._U32
        h = (((h << 17) | (h >> 15)) * m._P4) & m._U32
        pos += m._WORD
    while pos < n:
        h = (h + data[pos] * m._P5) & m._U32
        h = (((h << 11) | (h >> 21)) * m._P1) & m._U32
        pos += 1
    h ^= h >> 15
    h = (h * m._P2) & m._U32
    h ^= h >> 13
    h = (h * m._P3) & m._U32
    return h ^ (h >> 16)


def test_control_oracle_matches_vectors():
    """Rule 46: the frozen oracle reproduces the published vectors."""
    for data, want in _VECTORS.items():
        assert _oracle(data) == want


def test_regression_xxh32_vectors():
    """Falsification: published vectors."""
    for data, want in _VECTORS.items():
        assert m._xxh32_py(data) == want


def test_regression_xxh32_matches_oracle():
    """Adversarial: every length across the 16-byte stripe and 4-byte word
    boundaries, wrapping seeds, all-0xFF input, bytearray input."""
    rng = RandPool(seed=_SEED)
    for n in range(0, 200):
        for data in (rng.randbytes(n), b"\xff" * n):
            for seed in _SEEDS:
                assert m._xxh32_py(data, seed) == _oracle(data, seed)
    blob = bytearray(rng.randbytes(77))
    assert m._xxh32_py(blob) == _oracle(bytes(blob))
