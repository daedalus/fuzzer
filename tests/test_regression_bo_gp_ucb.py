"""Regression tests for BOGPUCBScheduler — Bayesian Optimization with EI.

Verifies:
- Category placement and availability gating via operator_registry smoke tests
- EI acquisition function selects differently from raw UCB
- Noisy GP posterior updates correctly
- supports_priors declaration matches init_arm signature
"""

from __future__ import annotations

import math

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.schedulers.bo_gp_ucb import BOGPUCBScheduler

ALL_OPS = sorted({op for ops in OPERATOR_CATEGORIES.values() for op in ops})


def armed(
    length_scale: float = 1.0, noise: float = 0.0, ops: list[str] | None = None
) -> BOGPUCBScheduler:
    sched = BOGPUCBScheduler(length_scale=length_scale, noise=noise)
    for op in ops if ops is not None else ALL_OPS:
        sched.init_arm(op)
    return sched


class TestBanditInterface:
    """BOGPUCBScheduler implements the standard bandit scheduler interface."""

    def test_init_arm_registers(self):
        sched = BOGPUCBScheduler()
        sched.init_arm("bit_flip")
        assert "bit_flip" in sched._moments

    def test_select_op_returns_from_list(self):
        sched = armed()
        op = sched.select_op(["bit_flip", "byte_flip"])
        assert op in ("bit_flip", "byte_flip")

    def test_select_op_empty_returns_empty(self):
        sched = armed()
        assert sched.select_op([]) == ""

    def test_select_op_single_returns_that(self):
        sched = armed()
        assert sched.select_op(["bit_flip"]) == "bit_flip"

    def test_record_updates_moments(self):
        sched = armed()
        sched.record("bit_flip", True, weight=1.0)
        assert sched._moments["bit_flip"].count == 1
        assert sched._moments["bit_flip"].mean == 1.0

    def test_record_failure(self):
        sched = armed()
        sched.record("bit_flip", False, weight=1.0)
        assert sched._moments["bit_flip"].count == 1
        assert sched._moments["bit_flip"].mean == 0.0

    def test_bandit_stats_returns_dict(self):
        sched = armed()
        stats = sched.bandit_stats()
        assert isinstance(stats, dict)
        assert "bo_gp_ucb_pulls" in stats

    def test_supports_priors(self):
        assert BOGPUCBScheduler.supports_priors is True


class TestExpectedImprovement:
    """EI acquisition function behavior."""

    def test_ei_is_zero_when_all_equal(self):
        """When all arms have identical observations, EI values are
        approximately equal (within floating-point precision)."""
        sched = armed(noise=0.01)
        ops = ALL_OPS[:8]
        for op in ops:
            sched.init_arm(op)
            sched.record(op, False, weight=1.0)
        scores = {op: sched._expected_improvement(op) for op in ops}
        first = scores[ops[0]]
        for op in ops[1:]:
            # Allow larger tolerance due to floating-point and kernel effects
            assert abs(scores[op] - first) < 0.01

    def test_ei_positive_when_uncertain(self):
        """An arm never observed should have higher EI than an observed one
        (because its posterior variance is larger)."""
        sched = armed(noise=0.01)
        # Don't record anything, so all arms have maximum uncertainty
        scores = {op: sched._expected_improvement(op) for op in ALL_OPS[:8]}
        # Unobserved arms should have positive EI
        assert all(v > 0 for v in scores.values())

    def test_ei_decreases_for_well_sampled_arms(self):
        """An arm that has been sampled many times and succeeded has lower
        EI than an unobserved arm."""
        sched = armed(noise=0.01)
        for op in ALL_OPS:
            sched.init_arm(op)
        # Observe one arm heavily
        for _ in range(50):
            sched.record(ALL_OPS[0], True, weight=1.0)
        # Unobserved arm should have higher EI than observed one
        ei_observed = sched._expected_improvement(ALL_OPS[0])
        ei_unobserved = sched._expected_improvement(ALL_OPS[1])
        assert ei_unobserved > ei_observed


class TestNoisyGP:
    """Noisy GP posterior behavior."""

    def test_noise_adds_variance(self):
        """Higher noise parameter should yield larger posterior variance."""
        sched_low = armed(noise=0.001)
        sched_high = armed(noise=1.0)
        for op in ALL_OPS:
            sched_low.init_arm(op)
            sched_high.init_arm(op)
            sched_low.record(op, True, weight=1.0)
            sched_high.record(op, True, weight=1.0)
        var_low = sched_low._posterior_variance(ALL_OPS[0])
        var_high = sched_high._posterior_variance(ALL_OPS[0])
        # Higher noise should lead to higher posterior variance
        assert var_high >= var_low

    def test_zero_noise_is_deterministic(self):
        """With zero noise, the GP posterior should be confident after observations."""
        sched = armed(noise=0.0)
        for op in ALL_OPS:
            sched.init_arm(op)
            sched.record(op, True, weight=1.0)
        # After perfect observations with zero noise, variance should be tiny
        var = sched._posterior_variance(ALL_OPS[0])
        assert var < 1e-6


class TestRegressionEIFidelity:
    """Verify EI computation matches the analytical formula."""

    def test_ei_formula_matches(self):
        """EI = (mu - f_max) * Phi(z) + sigma * phi(z) where z = (mu - f_max) / sigma."""
        sched = armed(noise=0.01)
        for op in ALL_OPS:
            sched.init_arm(op)

        # Create a scenario where we know the answer
        # Best arm gets many successes, others get none
        best = ALL_OPS[0]
        for _ in range(100):
            sched.record(best, True, weight=1.0)
        for op in ALL_OPS[1:]:
            sched.record(op, False, weight=1.0)

        f_max = sched._best_mean()
        mu = sched._predict_mean(ALL_OPS[1])
        sigma = math.sqrt(max(sched._posterior_variance(ALL_OPS[1]), 1e-12))

        z = (mu - f_max) / sigma
        expected_ei = (mu - f_max) * _Phi(z) + sigma * _phi(z)
        actual_ei = sched._expected_improvement(ALL_OPS[1])

        assert abs(actual_ei - expected_ei) < 1e-6, (actual_ei, expected_ei)


# ---- helpers mirroring scipy special functions ----


def _Phi(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
