"""Regression: MonteCarloScheduler's Beta posterior estimated the wrong mean.

1. A success added ``w <= 1`` to alpha but a failure added 1 to beta, so the
   posterior mean converged to ``p*E[w] / (p*E[w] + 1 - p)`` -- neither the
   success rate nor the expected reward. A success now also adds ``1 - w`` to
   beta (fractional Bernoulli), so the mean converges to ``E[reward]``.
2. Periodic decay multiplied alpha and beta toward 0, so forgotten evidence
   left Beta(eps, eps) -- a near-degenerate coin flip -- instead of the prior.
   Decay now pulls toward each arm's prior, as ``seed_quality`` already does.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

NO_DECAY = 1.0


def _mean(mc: MonteCarloScheduler, name: str) -> float:
    a, b = mc.arm_alpha[name], mc.arm_beta[name]
    return a / (a + b)


class TestFractionalReward:
    def test_regression_success_weight_adds_complement_to_beta(self):
        mc = MonteCarloScheduler(arm_decay=NO_DECAY)
        mc.init_arm("A")
        w = 0.3
        mc.record("A", success=True, weight=w)
        assert mc.arm_alpha["A"] == pytest.approx(1.0 + w)
        assert mc.arm_beta["A"] == pytest.approx(1.0 + (1.0 - w))

    def test_posterior_mean_converges_to_expected_reward(self):
        """Falsification: alternate success(w)/failure; the LLN target is
        E[reward] = 0.5 * w. The old update converged to w / (w + 1)."""
        mc = MonteCarloScheduler(arm_decay=NO_DECAY)
        mc.init_arm("A")
        w, n = 0.4, 4000
        for i in range(n):
            mc.record("A", success=i % 2 == 0, weight=w)
        expected_reward = 0.5 * w
        assert _mean(mc, "A") == pytest.approx(expected_reward, abs=1e-3)

    def test_overweight_success_never_shrinks_beta(self):
        """Adversarial: weight > 1 must not subtract from beta."""
        mc = MonteCarloScheduler(arm_decay=NO_DECAY)
        mc.init_arm("A")
        mc.record("A", success=True, weight=15.0)
        assert mc.arm_beta["A"] == 1.0
        assert mc.arm_alpha["A"] == 16.0

    def test_zero_weight_success_counts_as_failure(self):
        """Adversarial: a success worth nothing carries the evidence of a miss."""
        hit, miss = MonteCarloScheduler(arm_decay=NO_DECAY), MonteCarloScheduler(arm_decay=NO_DECAY)
        for mc in (hit, miss):
            mc.init_arm("A")
        hit.record("A", success=True, weight=0.0)
        miss.record("A", success=False)
        assert (hit.arm_alpha["A"], hit.arm_beta["A"]) == (miss.arm_alpha["A"], miss.arm_beta["A"])

    def test_pooled_counts_match_arm_increments(self):
        mc = MonteCarloScheduler(arm_decay=NO_DECAY)
        mc.init_arm("A")
        mc.record("A", success=True, weight=0.25)
        mc.record("A", success=False)
        assert mc._pooled_successes == pytest.approx(0.25)
        assert mc._pooled_failures == pytest.approx(0.75 + 1.0)


class TestDecayTowardPrior:
    def test_regression_idle_arm_decays_to_its_prior(self):
        """Falsification: 200 decays with no evidence for B leave B at its
        prior. The old multiplicative decay drove it to ~0.5**200."""
        mc = MonteCarloScheduler(arm_decay=0.5, decay_interval=1)
        mc.init_arm("A")
        mc.init_arm("B", prior_alpha=3.0, prior_beta=2.0)
        for _ in range(200):
            mc.record("A", success=False)
        assert mc.arm_alpha["B"] == pytest.approx(3.0)
        assert mc.arm_beta["B"] == pytest.approx(2.0)

    def test_evidence_shrinks_geometrically_toward_prior(self):
        mc = MonteCarloScheduler(arm_decay=0.5, decay_interval=10)
        pa, pb = 2.0, 5.0
        mc.init_arm("A", prior_alpha=pa, prior_beta=pb)
        n = 95
        for _ in range(n):
            mc.record("A", success=True, weight=1.0)
        # Independent derivation: decay fires before the update on every
        # 10th record and pulls the excess over the prior by half.
        alpha = pa
        for k in range(1, n + 1):
            if k % 10 == 0:
                alpha = pa + (alpha - pa) * 0.5
            alpha += 1.0
        assert mc.arm_alpha["A"] == pytest.approx(alpha)
        assert mc.arm_beta["A"] == pytest.approx(pb)

    def test_unregistered_arm_decays_to_uniform_prior(self):
        """Adversarial: an arm first seen in record() (never init_arm'd)
        has no stored prior; it must fall back to Beta(1, 1), not 0."""
        mc = MonteCarloScheduler(arm_decay=0.5, decay_interval=1)
        mc.record("ghost", success=True, weight=1.0)
        for _ in range(100):
            mc.record("other", success=False)
        assert mc.arm_alpha["ghost"] == pytest.approx(1.0)
