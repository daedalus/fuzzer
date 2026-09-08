"""Unit tests for the CuckooFilter (core/cuckoo.py).

Mirrors the style of ``tests/test_bloom.py``: a single ``TestCuckooFilter``
class with focused methods, bare ``assert`` and ``pytest.approx``.
"""

import random

import pytest

from fuzzer_tool.core.cuckoo import CuckooFilter


def _k(i: int) -> bytes:
    """Deterministic bytes key for an integer index."""
    return i.to_bytes((i.bit_length() + 7) // 8 or 1, "big")


class TestCuckooFilter:
    def test_add_and_query_round_trip(self):
        cf = CuckooFilter(capacity=1000)
        for key in ("alpha", "beta", "gamma"):
            cf.add(key)
            assert cf.contains(key) is True
            assert cf.query(key) is True

    def test_query_unknown_returns_false(self):
        cf = CuckooFilter(capacity=1000)
        assert cf.contains("missing") is False
        assert cf.query("missing") is False

    def test_no_false_negatives_after_insert(self):
        cf = CuckooFilter(capacity=1000)
        for i in range(100):
            cf.add(_k(i))
        for i in range(100):
            assert cf.contains(_k(i)) is True

    def test_add_bytes_and_query(self):
        cf = CuckooFilter(capacity=1000)
        cf.add(b"\x00\x01\x02")
        assert cf.contains(b"\x00\x01\x02") is True
        assert cf.contains(b"\x00\x01\x03") is False

    def test_remove_decrements_count_and_membership(self):
        cf = CuckooFilter(capacity=1000)
        cf.add("alpha")
        assert cf.count == 1
        cf.remove("alpha")
        assert cf.count == 0
        assert cf.contains("alpha") is False

    def test_remove_unknown_is_noop(self):
        cf = CuckooFilter(capacity=1000)
        cf.remove("missing")  # must not raise
        assert cf.count == 0

    def test_update_check_then_add_new_key(self):
        cf = CuckooFilter(capacity=1000)
        assert cf.update("alpha") is False  # newly inserted
        assert cf.update("alpha") is True  # already present
        assert cf.count == 1

    def test_clear_resets_state(self):
        cf = CuckooFilter(capacity=1000)
        for i in range(10):
            cf.add(_k(i))
        assert cf.count == 10
        cf.clear()
        assert cf.count == 0
        assert cf.n_added == 0
        assert cf.load_factor == 0.0
        for i in range(10):
            assert cf.contains(_k(i)) is False

    def test_load_factor_bounded(self):
        cf = CuckooFilter(capacity=1000)
        assert cf.load_factor == 0.0
        for i in range(100):
            cf.add(_k(i))
        assert 0.0 <= cf.load_factor <= 1.0

    def test_load_factor_increases_with_inserts(self):
        cf = CuckooFilter(capacity=1000)
        cf.add("a")
        after_one = cf.load_factor
        for i in range(200):
            cf.add(_k(i))
        assert cf.load_factor >= after_one

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            CuckooFilter(capacity=0)
        with pytest.raises(ValueError):
            CuckooFilter(capacity=-5)

    def test_invalid_fingerprint_size_raises(self):
        with pytest.raises(ValueError):
            CuckooFilter(capacity=100, fingerprint_size=0)
        with pytest.raises(ValueError):
            CuckooFilter(capacity=100, fingerprint_size=-1)

    def test_invalid_bucket_size_raises(self):
        with pytest.raises(ValueError):
            CuckooFilter(capacity=100, bucket_size=0)

    def test_cuckoo_kicking_preserves_all_inserts(self):
        # Fill a comfortably-sized filter; every inserted key must still be
        # queryable even after cuckoo kicking relocates fingerprints.
        cf = CuckooFilter(capacity=200, bucket_size=4)
        keys = [_k(i) for i in range(200)]
        rng = random.Random(20260821)
        rng.shuffle(keys)
        for k in keys:
            cf.add(k)
        for i in range(200):
            assert cf.contains(_k(i)) is True

    def test_alt_index_consistency(self):
        # contains() must check both i1 and i2, so a key inserted via one
        # index is found even after the other index is probed.
        cf = CuckooFilter(capacity=1000)
        cf.add("alpha")
        i1 = cf._get_index("alpha")
        i2 = cf._get_alt_index(cf._get_fingerprint("alpha"), i1)
        assert i1 != i2
        assert cf.buckets[i1] or cf.buckets[i2]  # fingerprint landed in one

    def test_n_added_tracks_insertions(self):
        cf = CuckooFilter(capacity=1000)
        for i in range(5):
            cf.add(_k(i))
        assert cf.n_added == 5
        # remove() does not decrement n_added (mirrors BloomFilter)
        cf.remove(_k(0))
        assert cf.n_added == 5


class TestUpdateBytes:
    def test_first_insert_reports_unseen(self):
        cf = CuckooFilter(capacity=1000)
        assert cf.update_bytes(b"alpha") is False
        assert cf.update_bytes(b"alpha") is True

    def test_distinct_keys_are_independent(self):
        cf = CuckooFilter(capacity=1000)
        assert cf.update_bytes(b"a") is False
        assert cf.update_bytes(b"b") is False
        assert cf.update_bytes(b"a") is True
        assert cf.update_bytes(b"b") is True

    def test_empty_key_is_tracked(self):
        cf = CuckooFilter(capacity=1000)
        assert cf.update_bytes(b"") is False
        assert cf.update_bytes(b"") is True

    def test_counts_insertions(self):
        cf = CuckooFilter(capacity=1000)
        for i in range(10):
            cf.update_bytes(bytes([i]))
        assert cf.n_added == 10

    def test_reset_on_full_wipes_at_capacity(self):
        # capacity=100, bucket_size=4 -> 32 buckets / 128 slots, so all 100
        # distinct keys are admitted; n_added reaches capacity and the 101st
        # admission triggers the generational reset, accepting the key as
        # unseen (the filter was wiped first).
        cf = CuckooFilter(capacity=100, bucket_size=4)
        for i in range(100):
            cf.update_bytes(_k(i), reset_on_full=True)
        assert cf.n_added == 100
        assert cf.update_bytes(b"overflow", reset_on_full=True) is False
        assert cf.update_bytes(b"overflow", reset_on_full=True) is True

    def test_reset_off_by_default(self):
        # capacity=1000 is comfortably larger than the 10 inserts, so no
        # eviction can lose a key and every insert stays queryable.
        cf = CuckooFilter(capacity=1000, bucket_size=4)
        for i in range(10):
            cf.update_bytes(bytes([i]))
        for i in range(10):
            assert cf.contains(bytes([i])) is True

    def test_clear_resets_n_added(self):
        cf = CuckooFilter(capacity=1000)
        for i in range(5):
            cf.update_bytes(bytes([i]))
        cf.clear()
        assert cf.n_added == 0
        assert cf.count == 0
