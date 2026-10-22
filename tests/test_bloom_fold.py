"""In-place folding of the flat BloomFilter (half-overlay, slice width kept)."""

import pytest

from fuzzer_tool.core.bloom import BloomFilter

# (capacity, error_rate) -> sliced and double-hashing modes.
SLICED = (100_000, 0.01)
DOUBLE = (500_000, 1e-3)


def _keys(n, tag=b"k"):
    return [tag + i.to_bytes(4, "big") for i in range(n)]


def _filled(cfg, n=20_000):
    bf = BloomFilter(*cfg)
    keys = _keys(n)
    for k in keys:
        bf.update_bytes(k)
    return bf, keys


def _halves_or(bits):
    """Reference fold, independent of BloomFilter: OR top half into bottom."""
    half = len(bits) // 2
    return bytearray(a | b for a, b in zip(bits[:half], bits[half:], strict=True))


def _fill(bits):
    return int.from_bytes(bits, "little").bit_count() / (len(bits) * 8)


class TestFold:
    def test_modes_under_test(self):
        assert not BloomFilter(*SLICED).digest_limited
        assert BloomFilter(*DOUBLE).digest_limited

    @pytest.mark.parametrize("cfg", [SLICED, DOUBLE])
    def test_no_false_negatives_after_each_fold(self, cfg):
        bf, keys = _filled(cfg)
        for _ in range(4):
            bf.fold()
            assert all(bf.query_bytes(k) for k in keys)

    @pytest.mark.parametrize("cfg", [SLICED, DOUBLE])
    def test_fold_is_half_overlay(self, cfg):
        # Falsification: bits match an OR of halves computed outside the class.
        bf, _ = _filled(cfg)
        m, expected = bf.m, _halves_or(bf._bits)
        bf.fold()
        assert bf.m == m // 2
        assert bf._mask == bf.m - 1
        assert bf._bits == expected
        assert bf.memory_bytes == len(expected)

    def test_slice_width_survives_fold(self):
        bf, _ = _filled(SLICED)
        width = bf._bits_per_slice
        bf.fold()
        assert bf._bits_per_slice == width

    def test_oracle_catches_recomputed_slice_width(self):
        # Control (Rule 46): the FN oracle above must be able to fail.  A fold
        # that re-derives the slice width from the new m loses nearly every key.
        bf, keys = _filled(SLICED)
        bf.fold()
        bf._bits_per_slice = bf.m.bit_length() - 1
        assert sum(not bf.query_bytes(k) for k in keys) > len(keys) // 2

    @pytest.mark.parametrize("cfg", [SLICED, DOUBLE])
    def test_inserts_after_fold(self, cfg):
        bf, old = _filled(cfg)
        bf.fold()
        new = _keys(5_000, b"n")
        for k in new:
            bf.update_bytes(k)
        assert all(bf.query_bytes(k) for k in old + new)

    def test_counters_kept_capacity_halved(self):
        bf, keys = _filled(SLICED)
        cap, k = bf.capacity, bf._k
        bf.fold()
        assert bf.n_added == len(keys)
        assert bf.capacity == cap // 2
        assert bf._k == k

    def test_min_size_raises(self):
        bf = BloomFilter(capacity=1_000, error_rate=0.01)
        while bf.m > BloomFilter.MIN_FOLD_BITS // 2:
            bf.fold()
        with pytest.raises(ValueError):
            bf.fold()
        assert bf.m == BloomFilter.MIN_FOLD_BITS // 2
        assert len(bf._bits) * 8 == bf.m

    def test_capacity_never_below_one(self):
        bf = BloomFilter(capacity=1, error_rate=0.01)
        while bf.m > BloomFilter.MIN_FOLD_BITS // 2:
            bf.fold()
        assert bf.capacity >= 1


class TestFoldTo:
    @pytest.mark.parametrize("cfg", [SLICED, DOUBLE])
    def test_stops_at_last_fold_within_target(self, cfg):
        bf, keys = _filled(cfg, n=5_000)
        target = cfg[1]
        k = bf._k
        folds = bf.fold_to(target)

        assert folds >= 1
        assert _fill(bf._bits) ** k <= target
        if bf.m >= BloomFilter.MIN_FOLD_BITS:
            assert _fill(_halves_or(bf._bits)) ** k > target
        assert all(bf.query_bytes(x) for x in keys)

    def test_observed_fpr_tracks_estimate(self):
        bf, _ = _filled(DOUBLE, n=5_000)
        bf.fold_to(1e-2)
        est = _fill(bf._bits) ** bf._k
        probes = _keys(100_000, b"absent")
        observed = sum(bf.query_bytes(p) for p in probes) / len(probes)
        assert observed < 2 * est + 1e-3

    def test_saturated_filter_does_not_fold(self):
        bf, _ = _filled(SLICED, n=200_000)
        m = bf.m
        assert bf.fold_to(1e-3) == 0
        assert bf.m == m

    def test_empty_filter_folds_to_minimum(self):
        bf = BloomFilter(*SLICED)
        bf.fold_to(0.01)
        assert bf.m < BloomFilter.MIN_FOLD_BITS

    @pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 2.0])
    def test_rejects_bad_target(self, target):
        with pytest.raises(ValueError):
            BloomFilter(*SLICED).fold_to(target)


class TestFoldPersistence:
    @pytest.mark.parametrize("cfg", [SLICED, DOUBLE])
    def test_roundtrip_after_fold(self, cfg):
        bf, keys = _filled(cfg)
        bf.fold_to(cfg[1])
        back = BloomFilter.from_dict(bf.to_dict())
        assert back._bits == bf._bits
        assert back._bits_per_slice == bf._bits_per_slice
        assert all(back.query_bytes(k) for k in keys)

    def test_v1_payload_still_loads(self):
        bf, keys = _filled(SLICED, n=1_000)
        state = bf.to_dict()
        state["version"] = 1
        del state["slice_bits"]
        back = BloomFilter.from_dict(state)
        assert all(back.query_bytes(k) for k in keys)

    def test_v2_payload_requires_slice_bits(self):
        # A folded filter needs slice_bits: dropping it must not load silently.
        bf, _ = _filled(SLICED, n=1_000)
        bf.fold()
        state = bf.to_dict()
        del state["slice_bits"]
        with pytest.raises(ValueError):
            BloomFilter.from_dict(state)

    @pytest.mark.parametrize("width", [0, -1, 300, "20"])
    def test_rejects_bad_slice_bits(self, width):
        state = BloomFilter(*SLICED).to_dict()
        state["slice_bits"] = width
        with pytest.raises(ValueError):
            BloomFilter.from_dict(state)
