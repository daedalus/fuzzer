"""MurmurHash3 chain behind the ``murmurhash3`` operator.

The operator fills a region block by block, each block the hash of every
block before it. ``murmur3_chain`` computes that without the O(L^2) Python
loop; the oracle below is the operator's original code, copied verbatim.
"""

import struct

import pytest

from fuzzer_tool.core.mutations.murmur3 import murmur3_32, murmur3_chain


def _ref_murmur3_32(data, seed=0):
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed
    nblocks = len(data) // 4
    for i in range(nblocks):
        k = struct.unpack_from("<I", data, i * 4)[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail = data[nblocks * 4 :]
    k = 0
    for i, b in enumerate(tail):
        k ^= b << (i * 8)
    if k:
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def _ref_chain(prefix: bytes, length: int, seed: int) -> bytes:
    """The operator's original fill loop over a region of *length* bytes."""
    out = bytearray(length + 4)
    hash_input = prefix
    for i in range(0, length, 4):
        h = _ref_murmur3_32(hash_input, seed ^ i)
        block = struct.pack("<I", h)
        for j in range(min(4, length - i)):
            out[i + j] = block[j]
        hash_input = bytes(out[: i + 4])
    return bytes(out[:length])


CASES = [
    (b"\x00", 4, 0),
    (b"\x00", 5, 1),
    (b"RIFF\x24\x08\x00\x00WAVE", 64, 0xDEADBEEF),
    (b"abc", 7, 0xFFFFFFFF),
    (bytes(range(256)) * 3, 1023, 12345),
    (b"\x01\x02", 4096, 0x9747B28C),
    # Either side of the scalar/vector switch (64 blocks).
    (b"edge", 252, 11),
    (b"edge", 256, 11),
    (b"edge", 257, 11),
]


class TestMurmur3:
    @pytest.mark.parametrize(
        "data,seed,want",
        [
            (b"", 0, 0x00000000),
            (b"", 1, 0x514E28B7),
            (b"", 0xFFFFFFFF, 0x81F16F39),
            (b"test", 0, 0xBA6BD213),
            (b"Hello, world!", 1234, 0xFAF6CDB3),
            (b"The quick brown fox jumps over the lazy dog", 0x9747B28C, 0x2FA826CD),
        ],
    )
    def test_published_vectors(self, data, seed, want):
        assert murmur3_32(data, seed) == want

    @pytest.mark.parametrize("prefix,length,seed", CASES)
    def test_chain_matches_original_operator(self, prefix, length, seed):
        assert murmur3_chain(prefix, length, seed) == _ref_chain(prefix, length, seed)

    @pytest.mark.parametrize("prefix,length,seed", CASES)
    def test_control_reference_against_itself(self, prefix, length, seed):
        """Hard Rule 46: the oracle must reproduce itself before it judges."""
        assert _ref_chain(prefix, length, seed) == _ref_chain(bytes(prefix), length, seed)


class TestMurmur3Falsification:
    def test_seed_changes_every_block(self):
        a = murmur3_chain(b"\x00", 64, 1)
        b = murmur3_chain(b"\x00", 64, 2)
        assert all(a[i : i + 4] != b[i : i + 4] for i in range(0, 64, 4))

    def test_blocks_depend_on_prefix(self):
        assert murmur3_chain(b"\x00", 16, 7) != murmur3_chain(b"\x01", 16, 7)


class TestMurmur3Adversarial:
    @pytest.mark.parametrize("length", [0, 1, 2, 3])
    def test_short_lengths(self, length):
        assert murmur3_chain(b"\x00", length, 5) == _ref_chain(b"\x00", length, 5)

    def test_seed_high_bits_do_not_overflow(self):
        # seed ^ i stays within 32 bits; seeds near 2^32 exercise wraparound.
        for seed in (0xFFFFFFFC, 0x80000000):
            assert murmur3_chain(b"x", 40, seed) == _ref_chain(b"x", 40, seed)

    def test_long_prefix_with_tail(self):
        prefix = bytes(range(251)) * 4  # 1004 bytes: 251 blocks, no tail
        assert murmur3_chain(prefix + b"\xff", 32, 3) == _ref_chain(prefix + b"\xff", 32, 3)
