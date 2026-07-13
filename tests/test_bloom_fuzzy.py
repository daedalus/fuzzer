"""Tests for core/bloom.py — Bloom filter with fuzzy dedup."""

import pytest

from fuzzer_tool.core.bloom import BloomFilter


class TestBloomFilter:
    def test_add_and_query(self):
        bf = BloomFilter(100)
        bf.add("hello")
        assert bf.query("hello") is True

    def test_query_absent(self):
        bf = BloomFilter(100)
        assert bf.query("hello") is False

    def test_update_new(self):
        bf = BloomFilter(100)
        assert bf.update("hello") is False  # was not present
        assert bf.query("hello") is True

    def test_update_existing(self):
        bf = BloomFilter(100)
        bf.add("hello")
        assert bf.update("hello") is True  # was already present

    def test_load_factor(self):
        bf = BloomFilter(100)
        assert bf.load_factor == 0.0
        bf.add("test")
        assert bf.load_factor > 0.0

    def test_clear(self):
        bf = BloomFilter(100)
        bf.add("hello")
        bf.clear()
        assert bf.query("hello") is False
        assert bf.load_factor == 0.0

    def test_false_positive_possible(self):
        bf = BloomFilter(1000, error_rate=0.01)
        for i in range(500):
            bf.add(f"key_{i}")
        # Some absent keys may match (false positives)
        matches = sum(1 for i in range(500, 600) if bf.query(f"key_{i}"))
        # With capacity=1000 and 500 inserts, FP rate should be low
        assert matches < 30

    def test_capacity_one(self):
        bf = BloomFilter(1)
        bf.add("x")
        assert bf.query("x") is True

    def test_many_inserts(self):
        bf = BloomFilter(1000)
        for i in range(500):
            bf.add(f"item_{i}")
        # All should be present
        for i in range(500):
            assert bf.query(f"item_{i}") is True


class TestBloomFuzzy:
    def test_init_fuzzy(self):
        bf = BloomFilter(100)
        bf.init_fuzzy()
        assert hasattr(bf, "_recent_keys")

    def test_add_bytes_exact_only(self):
        bf = BloomFilter(100)
        bf.init_fuzzy()
        result = bf.add_bytes(b"hello", max_hamming=0)
        assert result is False  # unique, added

    def test_add_bytes_near_duplicate(self):
        bf = BloomFilter(100)
        bf.init_fuzzy()
        bf.add_bytes(b"hello", max_hamming=0)
        # Very similar bytes → near-duplicate
        result = bf.add_bytes(b"hellx", max_hamming=1)
        assert result is True  # near-duplicate detected

    def test_add_bytes_unique(self):
        bf = BloomFilter(100)
        bf.init_fuzzy()
        bf.add_bytes(b"hello", max_hamming=0)
        result = bf.add_bytes(b"world", max_hamming=1)
        assert result is False  # unique

    def test_add_bytes_exact_match(self):
        bf = BloomFilter(100)
        bf.init_fuzzy()
        bf.add_bytes(b"hello", max_hamming=0)
        result = bf.add_bytes(b"hello", max_hamming=0)
        assert result is True  # exact match

    def test_add_bytes_without_init_fuzzy(self):
        bf = BloomFilter(100)
        # No init_fuzzy — falls back to exact add
        result = bf.add_bytes(b"hello", max_hamming=5)
        assert result is False

    def test_recent_keys_bounded(self):
        bf = BloomFilter(100)
        bf.init_fuzzy(max_recent=5)
        for i in range(10):
            bf.add_bytes(bytes([i] * 8), max_hamming=0)
        assert len(bf._recent_keys) <= 5

    def test_add_bytes_different_lengths(self):
        bf = BloomFilter(100)
        bf.init_fuzzy()
        bf.add_bytes(b"short", max_hamming=0)
        # Different length → hamming_distance raises ValueError → skip
        result = bf.add_bytes(b"much longer bytes here", max_hamming=5)
        assert result is False  # no crash, treated as unique
