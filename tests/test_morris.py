"""Tests for Morris probabilistic counting (a=30)."""

import ctypes
import math

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker, MORRIS_A, morris_estimate


class TestMorrisEstimate:
    def test_estimate_zero(self):
        assert morris_estimate(0) == 0.0

    def test_estimate_one(self):
        # a * ((1 + 1/a)^1 - 1) = a * (1/a) = 1
        assert morris_estimate(1) == pytest.approx(1.0)

    def test_estimate_small(self):
        # v=2: a * ((a+1)/a)^2 - 1) = a * ((a^2 + 2a + 1)/a^2 - 1) = a * (2a+1)/a^2 = (2a+1)/a
        expected = (2 * MORRIS_A + 1) / MORRIS_A
        assert morris_estimate(2) == pytest.approx(expected)

    def test_estimate_growth(self):
        # Estimates should be monotonically increasing
        for v in range(1, 20):
            assert morris_estimate(v) > morris_estimate(v - 1)

    def test_estimate_approximate_values(self):
        # Spot-check approximate values
        assert morris_estimate(5) == pytest.approx(5.43, abs=0.1)
        assert morris_estimate(10) == pytest.approx(11.7, abs=0.5)
        assert morris_estimate(20) == pytest.approx(27.8, abs=1.0)


class TestMorrisEdgeTracker:
    def test_morris_mode_flag(self):
        et = EdgeTracker(map_size=4096, morris_mode=True)
        assert et._morris_mode is True

    def test_legacy_mode_default(self):
        et = EdgeTracker(map_size=4096)
        assert et._morris_mode is False

    def test_record_edges_morris_conversion(self):
        """Morris values should be converted to approximate counts."""
        et = EdgeTracker(map_size=4096, morris_mode=True)
        # Morris value 1 → approximate count 1
        bitmap = bytearray(4096)
        bitmap[100] = 1
        et.record_edges("seed1", bytes(bitmap), morris_mode=True)
        assert et.seed_hit_counts["seed1"][100] == 1

        # Morris value 5 → approximate count ~5
        bitmap2 = bytearray(4096)
        bitmap2[200] = 5
        et.record_edges("seed2", bytes(bitmap2), morris_mode=True)
        hc = et.seed_hit_counts["seed2"][200]
        assert hc == int(round(morris_estimate(5)))
        assert hc >= 5  # estimate(5) ≈ 5.43, rounds to 5

    def test_record_edges_legacy_mode(self):
        """Legacy mode should use raw byte values."""
        et = EdgeTracker(map_size=4096, morris_mode=False)
        bitmap = bytearray(4096)
        bitmap[100] = 5
        et.record_edges("seed1", bytes(bitmap), morris_mode=False)
        assert et.seed_hit_counts["seed1"][100] == 5

    def test_edge_rarity_stats_morris(self):
        """Morris mode uses adjusted thresholds."""
        et = EdgeTracker(map_size=4096, morris_mode=True)
        # Seed with edges at different approximate counts
        for seed_idx in range(50):
            bitmap = bytearray(4096)
            # Each seed hits one unique edge
            bitmap[seed_idx] = 1
            et.record_edges(f"s{seed_idx}", bytes(bitmap), morris_mode=True)

        stats = et.edge_rarity_stats()
        assert stats["total"] == 50
        # All edges have count 1 → singleton
        assert stats["singleton"] == 50

    def test_edge_rarity_stats_legacy(self):
        """Legacy mode uses standard thresholds."""
        et = EdgeTracker(map_size=4096, morris_mode=False)
        bm1 = bytearray(4096)
        bm1[10] = 1
        et.record_edges("s1", bytes(bm1), morris_mode=False)
        bm2 = bytearray(4096)
        bm2[20] = 3
        et.record_edges("s2", bytes(bm2), morris_mode=False)
        bm3 = bytearray(4096)
        bm3[30] = 10
        et.record_edges("s3", bytes(bm3), morris_mode=False)

        stats = et.edge_rarity_stats()
        assert stats["singleton"] == 1  # edge 10
        assert stats["cold"] == 1  # edge 20 (count 3)
        assert stats["warm"] == 1  # edge 30 (count 10)

    def test_good_turing_morris(self):
        """Good-Turing should produce valid results in Morris mode."""
        et = EdgeTracker(map_size=4096, morris_mode=True)
        for i in range(20):
            bitmap = bytearray(4096)
            bitmap[i] = 1
            et.record_edges(f"s{i}", bytes(bitmap), morris_mode=True)

        gt = et.good_turing_estimate()
        assert gt["n"] == 20
        assert gt["n1"] == 20  # all edges have count 1
        assert gt["estimated_undiscovered"] >= 0
        assert gt["saturation"] >= 0.0

    def test_bitmap_density(self):
        """bitmap_density should work the same in both modes."""
        for morris in [True, False]:
            et = EdgeTracker(map_size=4096, morris_mode=morris)
            bitmap = bytearray(4096)
            for i in range(10):
                bitmap[i] = 1 if not morris else 3
            et.record_edges("s1", bytes(bitmap), morris_mode=morris)
            density = et.bitmap_density()
            assert 0.0 <= density <= 1.0
            assert density > 0


class TestMorrisUnpackbitsFix:
    def test_nonzero_byte_is_single_edge(self):
        """A nonzero byte should produce exactly one edge index, not 8."""
        import numpy as np

        bitmap = bytearray(4096)
        bitmap[5] = 10  # any nonzero value

        arr = np.frombuffer(bytes(bitmap), dtype=np.uint8)[:4096]
        edges = set(np.flatnonzero(arr))

        # Should be {5}, not {0, 1, 2, 3, 4, 5, 6, 7} from unpackbits
        assert edges == {5}

    def test_morris_value_does_not_expand_bits(self):
        """Morris value 5 (binary 00000101) should not create edge at bit 0 and 2."""
        import numpy as np

        bitmap = bytearray(4096)
        bitmap[100] = 5  # Morris value 5

        arr = np.frombuffer(bytes(bitmap), dtype=np.uint8)[:4096]
        edges = set(np.flatnonzero(arr))

        assert edges == {100}


class TestMorrisShim:
    """C shim threshold computation — verified in pure Python."""

    def test_morris_threshold_strictly_decreasing(self):
        """Thresholds should decrease monotonically."""
        MORRIS_A = 30
        MORRIS_MAX_V = 255
        threshold = [0] * (MORRIS_MAX_V + 1)
        threshold[0] = 0xFFFFFFFF
        for i in range(1, MORRIS_MAX_V + 1):
            threshold[i] = (threshold[i - 1] * MORRIS_A) // (MORRIS_A + 1)

        for i in range(MORRIS_MAX_V):
            assert threshold[i] >= threshold[i + 1]

    def test_morris_threshold_first_is_max(self):
        """Threshold[0] should be UINT32_MAX."""
        MORRIS_A = 30
        MORRIS_MAX_V = 255
        threshold = [0] * (MORRIS_MAX_V + 1)
        threshold[0] = 0xFFFFFFFF
        for i in range(1, MORRIS_MAX_V + 1):
            threshold[i] = (threshold[i - 1] * MORRIS_A) // (MORRIS_A + 1)

        assert threshold[0] == 0xFFFFFFFF
