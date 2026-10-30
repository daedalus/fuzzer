"""Tests for core/hash_inverse XXH64 lane inversion and second preimages."""

import struct

import pytest

from fuzzer_tool.core.hash_inverse import mul_inv, xxh64_collide, xxh64_unround
from fuzzer_tool.core.hashing import key16, xxh64
from fuzzer_tool.core.rand_pool import RandPool

_U64 = (1 << 64) - 1
_P1 = 0x9E3779B185EBCA87  # XXH64 spec constants, independent of hashing.py
_P2 = 0xC2B2AE3D27D4EB4F
_REF = b"R" * 64 + b"shared-tail"
_PREFIXES = [bytes([i]) * 32 for i in range(1, 4)]


def _round(acc: int, lane: int) -> int:
    x = (acc + lane * _P2) & _U64
    return (((x << 31) | (x >> 33)) * _P1) & _U64


class TestUnround:
    def test_unround_inverts_round(self):
        rng = RandPool(seed=5)
        for _ in range(200):
            acc = rng.randint(0, _U64)
            lane = rng.randint(0, _U64)
            assert xxh64_unround(acc, _round(acc, lane)) == lane

    def test_unround_oracle_can_fail(self):
        # Control: a wrong accumulator does not recover the lane.
        assert xxh64_unround(1, _round(0, 42)) != 42

    def test_mul_inv_64(self):
        assert (_P1 * mul_inv(_P1, 64)) & _U64 == 1

    def test_mul_inv_default_stays_32(self):
        c = 0x85EBCA6B
        assert mul_inv(c) == pow(c, -1, 1 << 32)


class TestCollide:
    def test_forged_inputs_hash_equal(self):
        forged = xxh64_collide(_REF, _PREFIXES)
        assert len(set(forged)) == len(_PREFIXES) + 1
        assert len({xxh64(f) for f in forged}) == 1
        assert len({key16(f) for f in forged}) == 1

    def test_digest_oracle_tells_distinct_inputs(self):
        # Control: the prefixes alone, unforged, hash differently.
        assert len({xxh64(p + _REF[32:]) for p in _PREFIXES}) == len(_PREFIXES)

    def test_forged_keep_length_and_tail(self):
        for f in xxh64_collide(_REF, _PREFIXES):
            assert len(f) == len(_REF) and f.endswith(b"shared-tail")

    def test_two_stripe_ref(self):
        ref = struct.pack("<8Q", *range(8))
        forged = xxh64_collide(ref, _PREFIXES[:1])
        assert xxh64(forged[0]) == xxh64(forged[1]) and forged[0] != forged[1]

    @pytest.mark.parametrize("n", (0, 31, 63))
    def test_adversarial_short_ref_raises(self, n):
        with pytest.raises(ValueError):
            xxh64_collide(b"x" * n, _PREFIXES)

    @pytest.mark.parametrize("p", (b"", b"x" * 31, b"x" * 33))
    def test_adversarial_bad_prefix_raises(self, p):
        with pytest.raises(ValueError):
            xxh64_collide(_REF, [p])
