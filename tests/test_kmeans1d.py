"""Exact 1-D k-means by DP vs brute force over every contiguous partition."""

from itertools import combinations

import numpy as np
import pytest

from fuzzer_tool.core.kmeans1d import (
    MAX_POINTS,
    Scale,
    optimal_buckets,
    partition_cost,
    to_scale,
)


def _brute(values, weights, k):
    """Min cost over all C(n-1, k-1) contiguous partitions, no DP."""
    n = len(values)
    best = np.inf
    for cuts in combinations(range(1, n), k - 1):
        starts = np.array([0, *cuts])
        best = min(best, partition_cost(values, weights, starts))
    return best


V = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 20.0, 21.0, 22.0, 50.0])
W = np.array([5.0, 1.0, 1.0, 9.0, 2.0, 1.0, 4.0, 1.0, 3.0])


class TestOptimal:
    @pytest.mark.parametrize("k", range(1, len(V) + 1))
    def test_matches_brute_force(self, k):
        starts, cost = optimal_buckets(V, W, k)
        assert cost == pytest.approx(_brute(V, W, k), abs=1e-9)
        assert len(starts) == k

    @pytest.mark.parametrize("k", range(1, len(V) + 1))
    def test_cost_matches_partition_cost(self, k):
        starts, cost = optimal_buckets(V, W, k)
        assert partition_cost(V, W, starts) == pytest.approx(cost, abs=1e-9)

    def test_known_split(self):
        starts, cost = optimal_buckets(np.array([1.0, 2.0, 10.0, 11.0]), np.ones(4), 2)
        assert starts.tolist() == [0, 2]
        assert cost == pytest.approx(1.0)

    def test_weight_moves_the_cut(self):
        v = np.array([0.0, 2.0, 3.0, 5.0])
        even, _ = optimal_buckets(v, np.ones(4), 2)
        heavy, _ = optimal_buckets(v, np.array([1.0, 1.0, 1.0, 1000.0]), 2)
        assert even.tolist() == [0, 2]
        assert heavy.tolist() == [0, 3]

    def test_k1_is_total_variance(self):
        _, cost = optimal_buckets(V, W, 1)
        mean = (V * W).sum() / W.sum()
        assert cost == pytest.approx((W * (V - mean) ** 2).sum())

    def test_cost_monotone_in_k(self):
        costs = [optimal_buckets(V, W, k)[1] for k in range(1, len(V) + 1)]
        assert all(a >= b - 1e-9 for a, b in zip(costs, costs[1:], strict=False))

    def test_k_ge_n_is_zero(self):
        starts, cost = optimal_buckets(V, W, len(V) + 5)
        assert starts.tolist() == list(range(len(V)))
        assert cost == 0.0

    def test_single_point(self):
        starts, cost = optimal_buckets(np.array([4.0]), np.array([3.0]), 1)
        assert starts.tolist() == [0]
        assert cost == 0.0

    def test_control_repeat_identical(self):
        # Rule 46: the reference run against itself must agree exactly.
        a = optimal_buckets(V, W, 4)
        b = optimal_buckets(V, W, 4)
        assert a[0].tolist() == b[0].tolist()
        assert a[1] == b[1]

    def test_offset_invariant(self):
        base, _ = optimal_buckets(V, W, 3)
        shifted, _ = optimal_buckets(V + 1e9, W, 3)
        assert shifted.tolist() == base.tolist()


class TestPartitionCost:
    def test_worse_partition_costs_more(self):
        _, best = optimal_buckets(V, W, 3)
        assert partition_cost(V, W, np.array([0, 1, 2])) > best

    def test_rejects_bad_starts(self):
        with pytest.raises(ValueError):
            partition_cost(V, W, np.array([1, 3]))  # must start at 0
        with pytest.raises(ValueError):
            partition_cost(V, W, np.array([0, 3, 3]))  # strictly increasing
        with pytest.raises(ValueError):
            partition_cost(V, W, np.array([0, len(V)]))  # in range


class TestAdversarial:
    def test_rejects_k_zero(self):
        with pytest.raises(ValueError):
            optimal_buckets(V, W, 0)

    def test_rejects_unsorted(self):
        with pytest.raises(ValueError):
            optimal_buckets(V[::-1], W, 2)

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError):
            optimal_buckets(np.array([1.0, 1.0, 2.0]), np.ones(3), 2)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_bad_weight(self, bad):
        w = W.copy()
        w[2] = bad
        with pytest.raises(ValueError):
            optimal_buckets(V, w, 2)

    def test_rejects_nan_value(self):
        v = V.copy()
        v[1] = np.nan
        with pytest.raises(ValueError):
            optimal_buckets(v, W, 2)

    def test_rejects_shape_mismatch_and_empty(self):
        with pytest.raises(ValueError):
            optimal_buckets(V, W[:-1], 2)
        with pytest.raises(ValueError):
            optimal_buckets(np.array([]), np.array([]), 1)

    def test_rejects_over_cap(self):
        n = MAX_POINTS + 1
        with pytest.raises(ValueError):
            optimal_buckets(np.arange(n, dtype=float), np.ones(n), 2)

    def test_extreme_weight_ratio(self):
        v = np.array([1.0, 2.0, 3.0])
        starts, _ = optimal_buckets(v, np.array([1e9, 1.0, 1e3]), 2)
        assert starts.tolist() == [0, 1]


class TestScale:
    def test_linear_is_identity(self):
        assert to_scale(np.array([1, 2, 255]), Scale.LINEAR).tolist() == [1.0, 2.0, 255.0]

    def test_log_is_log2(self):
        out = to_scale(np.array([1, 2, 4, 128]), Scale.LOG)
        assert out.tolist() == [0.0, 1.0, 2.0, 7.0]

    def test_log_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            to_scale(np.array([0, 1]), Scale.LOG)
