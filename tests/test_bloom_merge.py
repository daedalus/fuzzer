"""Tests for BloomFilter.merge: in-place union of two same-layout filters."""

import pytest

from fuzzer_tool.core.bloom import BloomFilter

KEYS_A = [b"alpha-%d" % i for i in range(200)]
KEYS_B = [b"beta-%d" % i for i in range(200)]


def _filled(keys, capacity=1000):
    bf = BloomFilter(capacity=capacity, error_rate=1e-3)
    for k in keys:
        bf.update_bytes(k)
    return bf


class TestMerge:
    def test_union_holds_both_sides(self):
        a, b = _filled(KEYS_A), _filled(KEYS_B)
        assert a.merge(b) is True
        assert all(a.query_bytes(k) for k in KEYS_A + KEYS_B)

    def test_subset_reports_no_change(self):
        # Falsification: merging what is already there changes nothing.
        a, sub = _filled(KEYS_A), _filled(KEYS_A[:50])
        before = a.to_dict()["bits"]
        assert a.merge(sub) is False
        assert a.to_dict()["bits"] == before

    def test_self_merge_is_no_change(self):
        # Control (Rule 46): a filter merged with an identical copy of itself.
        a = _filled(KEYS_A)
        twin = BloomFilter.from_dict(a.to_dict())
        assert a.merge(twin) is False

    def test_other_left_untouched_and_counters_kept(self):
        a, b = _filled(KEYS_A), _filled(KEYS_B)
        b_bits, a_added = b.to_dict()["bits"], a.n_added
        a.merge(b)
        assert b.to_dict()["bits"] == b_bits
        assert a.n_added == a_added  # caller decides how counts combine

    def test_bits_equal_or_of_both(self):
        # Independent reference: bytewise OR computed here, not by merge().
        a, b = _filled(KEYS_A), _filled(KEYS_B)
        expected = bytes(
            x | y for x, y in zip(a.to_dict()["bits"], b.to_dict()["bits"], strict=True)
        )
        a.merge(b)
        assert a.to_dict()["bits"] == expected

    def test_layout_mismatch_rejected(self):
        # Adversarial: ORing filters of different size or k corrupts both.
        a = _filled(KEYS_A, capacity=1000)
        other = _filled(KEYS_B, capacity=50_000)
        before = a.to_dict()["bits"]
        with pytest.raises(ValueError):
            a.merge(other)
        assert a.to_dict()["bits"] == before
