"""Tests for core/combinadic.py -- rank/unrank of m-permutations and m-combinations."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from fuzzer_tool.core import combinadic as cb


class TestCounts:
    def test_perm_count_falling_factorial(self):
        assert cb.perm_count(5, 3) == 60
        assert cb.perm_count(5, 5) == math.factorial(5)
        assert cb.perm_count(5, 0) == 1

    def test_perm_count_m_exceeds_n_is_zero(self):
        assert cb.perm_count(3, 4) == 0

    def test_comb_count(self):
        assert cb.comb_count(6, 2) == 15
        assert cb.comb_count(3, 4) == 0


class TestPermRanking:
    @pytest.mark.parametrize(("n", "m"), [(4, 2), (5, 3), (5, 5), (6, 1), (4, 0)])
    def test_unrank_matches_lexicographic_enumeration(self, n, m):
        expected = list(itertools.permutations(range(n), m))
        got = [cb.unrank_perm(i, n, m) for i in range(cb.perm_count(n, m))]
        assert got == expected

    @pytest.mark.parametrize(("n", "m"), [(4, 2), (5, 3), (6, 6)])
    def test_rank_inverts_unrank(self, n, m):
        for i in range(cb.perm_count(n, m)):
            assert cb.rank_perm(cb.unrank_perm(i, n, m), n) == i

    def test_large_index_roundtrip(self):
        # n=2000, m=5: ~3.2e16 tuples, far beyond enumeration.
        n, m = 2000, 5
        rng = random.Random(7)
        for _ in range(50):
            i = rng.randrange(cb.perm_count(n, m))
            assert cb.rank_perm(cb.unrank_perm(i, n, m), n) == i

    def test_unrank_out_of_range_raises(self):
        with pytest.raises(ValueError):
            cb.unrank_perm(cb.perm_count(4, 2), 4, 2)
        with pytest.raises(ValueError):
            cb.unrank_perm(-1, 4, 2)

    def test_unrank_m_exceeds_n_raises(self):
        with pytest.raises(ValueError):
            cb.unrank_perm(0, 3, 4)

    def test_rank_rejects_repeats_and_out_of_domain(self):
        with pytest.raises(ValueError):
            cb.rank_perm((1, 1), 4)
        with pytest.raises(ValueError):
            cb.rank_perm((0, 4), 4)


class TestCombRanking:
    @pytest.mark.parametrize(("n", "m"), [(5, 2), (6, 3), (6, 6), (7, 1), (4, 0)])
    def test_unrank_matches_lexicographic_enumeration(self, n, m):
        expected = list(itertools.combinations(range(n), m))
        got = [cb.unrank_comb(i, n, m) for i in range(cb.comb_count(n, m))]
        assert got == expected

    @pytest.mark.parametrize(("n", "m"), [(5, 2), (8, 3), (7, 7)])
    def test_rank_inverts_unrank(self, n, m):
        for i in range(cb.comb_count(n, m)):
            assert cb.rank_comb(cb.unrank_comb(i, n, m), n) == i

    def test_large_index_roundtrip(self):
        n, m = 5000, 6
        rng = random.Random(11)
        for _ in range(50):
            i = rng.randrange(cb.comb_count(n, m))
            assert cb.rank_comb(cb.unrank_comb(i, n, m), n) == i

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            cb.unrank_comb(cb.comb_count(5, 2), 5, 2)

    def test_rank_rejects_unsorted(self):
        with pytest.raises(ValueError):
            cb.rank_comb((3, 1), 5)


class TestSampleIndices:
    def test_distinct_and_in_range(self):
        got = cb.sample_indices(1000, 100, random.Random(3))
        assert len(got) == 100
        assert len(set(got)) == 100
        assert all(0 <= i < 1000 for i in got)

    def test_full_range_is_a_permutation(self):
        got = cb.sample_indices(50, 50, random.Random(4))
        assert sorted(got) == list(range(50))

    def test_huge_total_bounded_memory(self):
        # total ~3e16: must not materialise the range.
        total = cb.perm_count(2000, 5)
        got = cb.sample_indices(total, 64, random.Random(5))
        assert len(set(got)) == 64
        assert all(0 <= i < total for i in got)

    def test_count_exceeds_total_raises(self):
        with pytest.raises(ValueError):
            cb.sample_indices(3, 4, random.Random(0))

    def test_zero_count(self):
        assert cb.sample_indices(10, 0, random.Random(0)) == []

    def test_uniform_control_vs_self(self):
        # Hard Rule 46: control (sampler vs itself) must pass before comparing.
        # Each of 10 indices should be picked ~equally over many draws of 3.
        def freq(seed: int) -> list[int]:
            rng = random.Random(seed)
            counts = [0] * 10
            for _ in range(3000):
                for i in cb.sample_indices(10, 3, rng):
                    counts[i] += 1
            return counts

        a, b = freq(1), freq(2)
        expected = 3000 * 3 / 10
        # 4-sigma on Binomial(3000, 0.3): sqrt(3000*.3*.7)=25 -> 100.
        for c in (*a, *b):
            assert abs(c - expected) < 100


class TestSampleTuples:
    def test_perm_samples_are_valid_and_distinct(self):
        got = cb.sample_perms(20, 5, 200, random.Random(9))
        assert len(set(got)) == 200
        for p in got:
            assert len(set(p)) == 5
            assert all(0 <= x < 20 for x in p)

    def test_comb_samples_are_sorted_and_distinct(self):
        got = cb.sample_combs(30, 4, 100, random.Random(9))
        assert len(set(got)) == 100
        for c in got:
            assert list(c) == sorted(c)

    def test_exhausting_small_space_covers_everything(self):
        got = cb.sample_perms(4, 2, cb.perm_count(4, 2), random.Random(2))
        assert set(got) == set(itertools.permutations(range(4), 2))


class TestRandPoolCompat:
    def test_sample_perms_with_randpool_beyond_32_bits(self):
        from fuzzer_tool.core.rand_pool import RandPool

        total = cb.perm_count(2000, 5)
        assert total > 2**32
        got = cb.sample_indices(total, 200, RandPool(seed=1))
        assert len(set(got)) == 200
        # A single 32-bit draw modulo `total` could never exceed 2**32.
        assert max(got) > 2**32
