"""Regression tests for BloomFilter analysis findings.

Covers: double hashing for tight error rates, argument validation,
overfill behaviour, load_factor, and add_bytes/update_bytes consistency.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from fuzzer_tool.core.bloom import BloomFilter


def _keys(n: int, salt: int = 0) -> list[bytes]:
    r = random.Random(9000 + salt)
    return [r.randbytes(12) + i.to_bytes(4, "big") for i in range(n)]


def _digest(k: bytes) -> int:
    return int.from_bytes(hashlib.sha256(k).digest(), "big")


# --- tight error rates -----------------------------------------------------


def test_tight_error_rate_uses_double_hashing_and_keeps_k():
    bf = BloomFilter(capacity=50_000, error_rate=1e-8)
    assert bf.digest_limited and bf._k == bf._k_ideal
    assert bf._k * bf._bits_per_slice > BloomFilter.DIGEST_BITS


def test_tight_error_rate_is_honoured_by_fill():
    # Realised rate = fill**k; must not exceed the requested rate (the old
    # k-clamp made this ~10x+ worse for 1e-9).
    bf = BloomFilter(capacity=50_000, error_rate=1e-8)
    for k in _keys(50_000):
        bf.update_bytes(k)
    assert bf.load_factor**bf._k <= 1e-8 * 2
    assert bf.expected_fpr <= 1e-8 * 2


def test_double_hashing_has_no_false_negatives():
    bf = BloomFilter(capacity=20_000, error_rate=1e-9)
    ks = _keys(20_000)
    for k in ks:
        bf.update_bytes(k)
    assert all(bf._check(_digest(k)) for k in ks)


def test_double_hash_probes_are_distinct():
    bf = BloomFilter(capacity=50_000, error_rate=1e-8)
    for k in _keys(300):
        v = _digest(k)
        h, step, pos = v & ((1 << 128) - 1), (v >> 128) | 1, []
        for _ in range(bf._k):
            pos.append(h & bf._mask)
            h += step
        assert len(set(pos)) == len(pos)


def test_sliced_mode_positions_unchanged():
    # Non-limited configs keep the historic independent-slice positions.
    bf = BloomFilter(capacity=100_000, error_rate=0.01)
    assert not bf.digest_limited
    k = b"pin-me"
    bf.update_bytes(k)
    v, expect = _digest(k), set()
    for _ in range(bf._k):
        expect.add(v & bf._mask)
        v >>= bf._bits_per_slice
    got = {i * 8 + b for i, byte in enumerate(bf._bits) for b in range(8) if byte >> b & 1}
    assert got == expect


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize("cap", [0, -5])
def test_bad_capacity_raises(cap):
    with pytest.raises(ValueError):
        BloomFilter(capacity=cap)


@pytest.mark.parametrize("er", [0.0, 1.0, 1.5, -0.1])
def test_bad_error_rate_raises(er):
    with pytest.raises(ValueError):
        BloomFilter(capacity=10, error_rate=er)


# --- overfill / introspection ---------------------------------------------


def test_overfill_never_loses_keys_and_is_reported():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    ks = _keys(5000)
    for k in ks:
        bf.update_bytes(k)
    assert bf.over_capacity
    assert all(bf._check(_digest(k)) for k in ks)
    assert bf.expected_fpr > 0.3  # 5x load; pow2 rounding gives extra room


def test_expected_fpr_tracks_measurement():
    bf = BloomFilter(capacity=20_000, error_rate=1e-2)
    for k in _keys(20_000):
        bf.update_bytes(k)
    probes = _keys(100_000, salt=1)
    measured = sum(bf._check(_digest(k)) for k in probes) / len(probes)
    assert measured == pytest.approx(bf.expected_fpr, rel=0.5)
    assert not bf.over_capacity


def test_load_factor_matches_naive_count():
    bf = BloomFilter(capacity=5000, error_rate=1e-3)
    for k in _keys(3000):
        bf.update_bytes(k)
    naive = sum(b.bit_count() for b in bf._bits) / bf.m
    assert bf.load_factor == naive


# --- add_bytes -------------------------------------------------------------


def test_add_bytes_dedups_without_init_fuzzy():
    bf = BloomFilter(100)
    assert bf.add_bytes(b"hello") is False
    assert bf.add_bytes(b"hello") is True  # used to be False forever


def test_add_bytes_and_update_bytes_share_keyspace():
    bf = BloomFilter(1000)
    bf.add_bytes(b"k")
    assert bf.update_bytes(b"k") is True
    bf2 = BloomFilter(1000)
    bf2.update_bytes(b"k")
    assert bf2.add_bytes(b"k") is True


def test_add_bytes_auto_inits_fuzzy_buffer():
    bf = BloomFilter(1000)
    assert bf.add_bytes(b"hello", max_hamming=1) is False
    assert bf.add_bytes(b"hellx", max_hamming=1) is True


def test_add_bytes_skips_other_lengths_without_exceptions(monkeypatch):
    import fuzzer_tool.core.bloom as mod

    calls = []
    real = mod.hamming_distance
    monkeypatch.setattr(mod, "hamming_distance", lambda a, b: calls.append(1) or real(a, b))
    bf = BloomFilter(1000)
    bf.init_fuzzy()
    for i in range(50):
        bf.add_bytes(b"x" * (i + 1), max_hamming=2)
    assert calls == []  # every recent key had a different length


def test_clear_drops_recent_keys():
    bf = BloomFilter(1000)
    bf.init_fuzzy()
    bf.add_bytes(b"hello", max_hamming=1)
    bf.clear()
    assert bf.add_bytes(b"hellx", max_hamming=1) is False
