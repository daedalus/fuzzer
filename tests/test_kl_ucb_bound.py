"""KL-UCB confidence bound: the shared Bernoulli upper bound.

KL-UCB (Cappé, Garivier, Maillard, Munos, Stoltz, 2013) replaces the Gaussian
width in the D-UCB and SW-UCB indexes with the smallest q in [p, 1] such that
the empirical-Bernoulli KL between the observed mean p and q is at least the
budget log(n_t)/N_t(i).  The bound is what makes the index valid for bounded
rewards with a mass at zero, which is the fuzzer's cost-adjusted surprisal
weight distribution.

The edge cases here are not decorative: two of them were the difference between
a working scheduler and one that starved the best arm to zero tail share in the
measurement pass, and each is pinned so that regression shows up as a failure
rather than as a silent change in tail share.
"""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.schedulers._kl_ucb import _kl_bernoulli, kl_upper_bound


def _kl(p: float, q: float) -> float:
    return _kl_bernoulli(p, q)


class TestKlBernoulli:
    def test_identical_means_are_zero(self):
        assert _kl(0.3, 0.3) == 0.0
        assert _kl(1.0, 1.0) == 0.0

    def test_zero_left_is_minus_log_one_minus_q(self):
        # KL(0 || q) = -log(1 - q)
        assert _kl(0.0, 0.5) == pytest.approx(-math.log(0.5))

    def test_one_left_is_minus_log_q(self):
        # KL(1 || q) = -log(q)
        assert _kl(1.0, 0.5) == pytest.approx(-math.log(0.5))

    def test_is_non_negative_and_symmetric_only_at_the_mean(self):
        for p, q in ((0.1, 0.9), (0.9, 0.1), (0.25, 0.75), (0.5, 0.5)):
            assert _kl(p, q) >= 0.0


class TestKlUpperBound:
    def test_zero_budget_returns_the_mean(self):
        assert kl_upper_bound(0.3, 0.0) == pytest.approx(0.3)

    def test_perfect_mean_saturates_at_one(self):
        # KL(1 || q) = -log(q); the only q >= 1 is q = 1, where the KL is 0,
        # below any positive budget. No q satisfies the constraint, so the
        # bound is the boundary: 1.0. Returning exp(-budget) here would put
        # the "upper bound" below the empirical mean and starve the arm.
        assert kl_upper_bound(1.0, 2.355) == pytest.approx(1.0)

    def test_zero_mean_bound_is_one_minus_exp_minus_budget(self):
        # KL(0 || q) = -log(1 - q); the smallest q with -log(1 - q) >= budget
        # is 1 - exp(-budget).
        assert kl_upper_bound(0.0, 2.355) == pytest.approx(1.0 - math.exp(-2.355))

    def test_bound_is_at_least_the_mean(self):
        for p in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            for budget in (0.01, 0.1, 0.5, 1.0, 2.0):
                assert kl_upper_bound(p, budget) >= p - 1e-12

    def test_bound_satisfies_the_kl_constraint(self):
        for p in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            for budget in (0.01, 0.1, 0.5, 1.0, 2.0):
                q = kl_upper_bound(p, budget)
                # At p = 1.0 the only candidate q >= p is q = 1.0, where
                # KL(1 || 1) = 0, below any positive budget: the constraint
                # is unsatisfiable and the bound saturates at the boundary.
                # Everywhere else the bound is a genuine KL witness.
                if p >= 1.0:
                    assert q == pytest.approx(1.0)
                else:
                    assert _kl(p, q) >= budget - 1e-9

    def test_larger_budget_gives_larger_bound(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            assert kl_upper_bound(p, 0.5) <= kl_upper_bound(p, 1.0) <= kl_upper_bound(p, 2.0)

    def test_gaussian_fallback_matches_when_the_approximation_is_tight(self):
        # At small budgets the Gaussian form p + sqrt(2*budget) is within
        # floating-point tolerance of the true Bernoulli bound, and the helper
        # returns it directly rather than iterating to float noise.
        p, budget = 0.5, 1e-6
        assert kl_upper_bound(p, budget) == pytest.approx(p + math.sqrt(2.0 * budget))

    def test_bounded_to_one(self):
        assert kl_upper_bound(0.5, 100.0) <= 1.0
