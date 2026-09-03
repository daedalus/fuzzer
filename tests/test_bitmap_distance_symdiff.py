"""Bitmap distance is a bit count, and the symmetric difference computes it.

``compute_hamming_bitmap_distance`` divides by the bitmap width in *bits*.
The numerator used to come from ``similarity.hamming_distance``, which
counts differing *byte* positions, so the ratio was mixed-unit and
undercounted by up to 8x.

The oracle here builds the two bitmaps explicitly and counts differing
bits by popcount of the XOR -- the thing the function claims to do -- and
the implementation must agree with it. The identity that replaces the
bitmaps is inclusion-exclusion on the indicator vectors:

    |A XOR B| = |A| + |B| - 2|A AND B|
"""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker


def bitmap_bit_distance(edges_a: set[int], edges_b: set[int]) -> float:
    """Explicit-bitmap oracle: popcount of the XOR over the bitmap width."""
    if not edges_a and not edges_b:
        return 0.0
    if not edges_a or not edges_b:
        return 1.0
    max_edge = max(max(edges_a), max(edges_b)) + 1
    size = (max_edge + 7) // 8
    bm_a = bytearray(size)
    bm_b = bytearray(size)
    for e in edges_a:
        bm_a[e >> 3] |= 1 << (e & 7)
    for e in edges_b:
        bm_b[e >> 3] |= 1 << (e & 7)
    xor = int.from_bytes(bytes(bm_a), "big") ^ int.from_bytes(bytes(bm_b), "big")
    return xor.bit_count() / (size * 8)


def tracker(edges_a, edges_b) -> EdgeTracker:
    et = EdgeTracker.__new__(EdgeTracker)
    et.seed_edges = {"a": set(edges_a), "b": set(edges_b)}
    return et


def distance(edges_a, edges_b) -> float:
    return tracker(edges_a, edges_b).compute_hamming_bitmap_distance("a", "b")


class TestBitsNotBytes:
    """The three worked cases from the report, pinned as regressions."""

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            # Bitmap is 13 bytes = 104 bits in all three, because edge 100
            # is present on both sides and fixes the width.
            (range(0, 8), 8 / 104),  # eight bits inside one byte
            (range(0, 1), 1 / 104),  # a single bit
            (range(0, 64), 64 / 104),  # eight whole bytes
        ],
        ids=["eight-bits-one-byte", "one-bit", "sixty-four-bits"],
    )
    def test_differing_bits_are_counted(self, extra, expected):
        assert distance([100, *extra], [100]) == pytest.approx(expected)

    def test_eight_bits_in_one_byte_outscore_one_bit(self):
        """The distinction the byte-counting numerator collapsed."""
        eight = distance([100, *range(0, 8)], [100])
        one = distance([100, 0], [100])
        assert eight == pytest.approx(8 * one)


class TestMatchesExplicitBitmapOracle:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_sets(self, seed):
        rng = random.Random(seed)
        for _ in range(60):
            universe = rng.choice([64, 1024, 65536])
            a = set(rng.sample(range(universe), rng.randrange(1, 60)))
            b = set(rng.sample(range(universe), rng.randrange(1, 60)))
            assert distance(a, b) == pytest.approx(bitmap_bit_distance(a, b))

    def test_dense_overlapping_sets(self):
        rng = random.Random(9)
        a = set(rng.sample(range(4096), 2000))
        b = set(rng.sample(range(4096), 2000))
        assert distance(a, b) == pytest.approx(bitmap_bit_distance(a, b))


class TestInclusionExclusion:
    def test_symmetric_difference_identity(self):
        """|A xor B| == |A| + |B| - 2|A and B| is what the numerator is."""
        rng = random.Random(11)
        for _ in range(100):
            a = set(rng.sample(range(512), rng.randrange(1, 80)))
            b = set(rng.sample(range(512), rng.randrange(1, 80)))
            assert len(a ^ b) == len(a) + len(b) - 2 * len(a & b)

    def test_disjoint_sets_sum_their_sizes(self):
        a, b = {0, 2, 4, 6}, {1, 3, 5, 7}
        # width fixed at one byte by max edge 7
        assert distance(a, b) == pytest.approx(8 / 8)

    def test_identical_sets_are_zero(self):
        s = {3, 17, 91}
        assert distance(s, s) == 0.0

    def test_symmetry(self):
        rng = random.Random(13)
        for _ in range(50):
            a = set(rng.sample(range(1024), 30))
            b = set(rng.sample(range(1024), 30))
            assert distance(a, b) == distance(b, a)

    def test_bounded_in_unit_interval(self):
        rng = random.Random(17)
        for _ in range(200):
            a = set(rng.sample(range(256), rng.randrange(1, 200)))
            b = set(rng.sample(range(256), rng.randrange(1, 200)))
            assert 0.0 <= distance(a, b) <= 1.0


class TestEmptyEdgeCases:
    def test_both_empty_is_zero(self):
        assert distance(set(), set()) == 0.0

    def test_one_empty_is_one(self):
        assert distance({1, 2, 3}, set()) == 1.0
        assert distance(set(), {1, 2, 3}) == 1.0

    def test_missing_seed_key_behaves_as_empty(self):
        et = tracker({1, 2}, {1, 2})
        assert et.compute_hamming_bitmap_distance("a", "absent") == 1.0
        assert et.compute_hamming_bitmap_distance("gone", "absent") == 0.0
