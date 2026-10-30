"""Tests for core/permutation: cycle type, sign, SJT adjacent-swap walk."""

import itertools
import math

import pytest

from fuzzer_tool.core.permutation import cycle_type, perm_sign, sjt_swaps


def _inversions(perm) -> int:
    """Independent sign oracle: parity of the inversion count."""
    return sum(1 for i, j in itertools.combinations(range(len(perm)), 2) if perm[i] > perm[j])


class TestCycleType:
    def test_known_decomposition(self):
        # 0->1->2->0, 3->4->3, 5 fixed.
        assert sorted(cycle_type([1, 2, 0, 4, 3, 5])) == [1, 2, 3]

    def test_identity_is_all_fixed_points(self):
        assert cycle_type(list(range(5))) == [1] * 5

    def test_lengths_sum_to_n(self):
        for p in itertools.permutations(range(5)):
            assert sum(cycle_type(p)) == 5

    def test_empty(self):
        assert cycle_type([]) == []

    def test_adversarial_non_permutation_raises(self):
        with pytest.raises(ValueError):
            cycle_type([0, 0, 1])
        with pytest.raises(ValueError):
            cycle_type([0, 3])


class TestPermSign:
    @pytest.mark.parametrize("n", range(1, 6))
    def test_matches_inversion_parity(self, n):
        for p in itertools.permutations(range(n)):
            assert perm_sign(p) == (-1) ** _inversions(p)

    def test_inversion_oracle_can_fail(self):
        # Control: a transposition has odd inversion count.
        assert (-1) ** _inversions([1, 0, 2]) == -1

    def test_even_cycle_is_odd(self):
        assert perm_sign([1, 2, 3, 0]) == -1
        assert perm_sign([1, 2, 0]) == 1

    def test_adversarial_non_permutation_raises(self):
        with pytest.raises(ValueError):
            perm_sign([1, 1])


class TestSjtSwaps:
    @pytest.mark.parametrize("n", range(1, 6))
    def test_visits_every_permutation_once(self, n):
        perm = list(range(n))
        seen = {tuple(perm)}
        for i in sjt_swaps(n):
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
            seen.add(tuple(perm))
        assert len(seen) == math.factorial(n)

    def test_known_order_n3(self):
        assert list(sjt_swaps(3)) == [1, 0, 1, 0, 1]

    def test_each_swap_flips_sign(self):
        perm = list(range(4))
        sign = perm_sign(perm)
        for i in sjt_swaps(4):
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
            assert perm_sign(perm) == -sign
            sign = -sign

    @pytest.mark.parametrize("n", (0, -1))
    def test_adversarial_degenerate_n_yields_nothing(self, n):
        assert list(sjt_swaps(n)) == []
