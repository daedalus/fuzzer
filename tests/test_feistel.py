"""Tests for the minimal Feistel network (core/feistel.py)."""

import random

import pytest

from fuzzer_tool.core.feistel import (
    feistel_decrypt,
    feistel_encrypt,
    feistel_permute,
    feistel_round_function,
    feistel_scramble,
    feistel_unpermute,
)

# ── round function ──────────────────────────────────────────────────────


def test_round_function_deterministic():
    a = feistel_round_function(b"abcd", 0x1234)
    b = feistel_round_function(b"abcd", 0x1234)
    assert a == b


def test_round_function_key_sensitive():
    a = feistel_round_function(b"abcd", 0x1234)
    b = feistel_round_function(b"abcd", 0x1235)
    assert a != b


def test_round_function_returns_32_bits():
    out = feistel_round_function(b"abcd", 0)
    assert 0 <= out <= 0xFFFFFFFF


# ── encrypt / decrypt ────────────────────────────────────────────────────


def test_encrypt_decrypt_round_trip():
    block = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    keys = [0xDEADBEEF, 0xCAFEBABE]
    enc = feistel_encrypt(block, keys)
    dec = feistel_decrypt(enc, keys)
    assert dec == block


def test_encrypt_changes_block():
    block = b"\x00" * 8
    keys = [1, 2, 3]
    enc = feistel_encrypt(block, keys)
    assert enc != block


def test_encrypt_rejects_wrong_length():
    with pytest.raises(ValueError):
        feistel_encrypt(b"short", [1])
    with pytest.raises(ValueError):
        feistel_encrypt(b"toolongblock", [1])


def test_encrypt_rejects_empty_keys():
    with pytest.raises(ValueError):
        feistel_encrypt(b"\x00" * 8, [])


def test_round_trip_single_round():
    block = b"\xff" * 8
    keys = [42]
    enc = feistel_encrypt(block, keys, rounds=1)
    dec = feistel_decrypt(enc, keys, rounds=1)
    assert dec == block


def test_round_trip_many_rounds_key_cycling():
    # rounds > len(keys) forces key cycling on both encrypt and decrypt.
    block = b"\x10\x20\x30\x40\x50\x60\x70\x80"
    keys = [1, 2, 3]
    enc = feistel_encrypt(block, keys, rounds=9)
    dec = feistel_decrypt(enc, keys, rounds=9)
    assert dec == block


def test_round_trip_property_random_blocks():
    rng = random.Random(1234)
    for _ in range(50):
        block = bytes(rng.randrange(256) for _ in range(8))
        keys = [rng.randrange(2**32) for _ in range(rng.randint(1, 5))]
        rounds = rng.randint(1, 8)
        enc = feistel_encrypt(block, keys, rounds=rounds)
        assert feistel_decrypt(enc, keys, rounds=rounds) == block


# ── permute / unpermute (multi-block, buffer-level) ─────────────────────


def test_permute_unpermute_round_trip_exact_multiple():
    data = bytes(range(16))  # exactly two 8-byte blocks
    keys = [7, 8, 9]
    permuted = feistel_permute(data, keys)
    assert feistel_unpermute(permuted, keys) == data


def test_permute_leaves_trailing_bytes_unchanged():
    data = bytes(range(19))  # two full blocks + 3 trailing bytes
    keys = [7]
    permuted = feistel_permute(data, keys)
    assert permuted[16:] == data[16:]
    assert feistel_unpermute(permuted, keys) == data


def test_permute_empty_data():
    assert feistel_permute(b"", [1]) == b""
    assert feistel_unpermute(b"", [1]) == b""


def test_permute_sub_block_data_untouched():
    data = b"\x01\x02\x03"  # shorter than one block
    permuted = feistel_permute(data, [1])
    assert permuted == data


def test_permute_changes_full_blocks():
    data = bytes(range(8))
    permuted = feistel_permute(data, [123])
    assert permuted != data
    assert len(permuted) == len(data)


# ── feistel_scramble (mutation-operator entry point) ────────────────────


def test_scramble_is_length_preserving():
    rng = random.Random(0)
    data = bytes(range(37))
    out = feistel_scramble(data, rng=rng)
    assert len(out) == len(data)


def test_scramble_short_data_returned_unchanged():
    rng = random.Random(0)
    data = b"\x01\x02\x03"
    assert feistel_scramble(data, rng=rng) == data


def test_scramble_empty_data():
    rng = random.Random(0)
    assert feistel_scramble(b"", rng=rng) == b""


def test_scramble_changes_full_block_data():
    rng = random.Random(42)
    data = bytes(range(16))
    out = feistel_scramble(data, rng=rng)
    assert out != data


def test_scramble_deterministic_given_seeded_rng():
    data = bytes(range(24))
    out_a = feistel_scramble(data, rng=random.Random(99))
    out_b = feistel_scramble(data, rng=random.Random(99))
    assert out_a == out_b


def test_scramble_without_rng_uses_module_random():
    # No rng passed -> falls back to the stdlib `random` module; just check
    # it runs, preserves length, and doesn't raise.
    data = bytes(range(8))
    out = feistel_scramble(data)
    assert len(out) == len(data)


def test_scramble_respects_rounds_upper_bound():
    class _CountingRng(random.Random):
        def __init__(self, seed):
            super().__init__(seed)
            self.randint_calls = []

        def randint(self, a, b):
            self.randint_calls.append((a, b))
            return super().randint(a, b)

    rng = _CountingRng(1)
    feistel_scramble(bytes(range(8)), rng=rng, rounds=2)
    # First randint call picks key_count in [1, rounds].
    assert rng.randint_calls[0] == (1, 2)
