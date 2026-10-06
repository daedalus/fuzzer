"""Maximum-sum contiguous run, vectorized.

Covers core/kadane.py.
"""

import numpy as np
import pytest

from fuzzer_tool.core.kadane import max_subarray


def _brute(x):
    n = len(x)
    return max(float(x[i:j].sum()) for i in range(n) for j in range(i + 1, n + 1))


def test_empty_is_none():
    assert max_subarray(np.array([])) is None


def test_single_element_even_when_negative():
    assert max_subarray(np.array([-3.0])) == (0, 1, -3.0)


def test_all_negative_picks_the_largest_element():
    assert max_subarray(np.array([-5.0, -2.0, -9.0])) == (1, 2, -2.0)


def test_classic_example():
    x = np.array([-2, 1, -3, 4, -1, 2, 1, -5, 4], dtype=float)
    assert max_subarray(x) == (3, 7, 6.0)


def test_zero_edges_are_left_out_but_inner_zero_gaps_are_bridged():
    assert max_subarray(np.array([0, 0, 3, 0, 0, 4, 0, 0.0])) == (2, 6, 7.0)


def test_negative_gap_splits_two_runs_when_it_costs_more_than_it_earns():
    assert max_subarray(np.array([5.0, -9.0, 4.0]))[:2] == (0, 1)
    assert max_subarray(np.array([5.0, -1.0, 4.0]))[:2] == (0, 3)


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force_and_total_is_the_window_sum(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(-6, 7, size=int(rng.integers(1, 40))).astype(float)
    start, end, total = max_subarray(x)
    assert 0 <= start < end <= len(x)
    assert total == pytest.approx(_brute(x))
    assert float(x[start:end].sum()) == pytest.approx(total)
