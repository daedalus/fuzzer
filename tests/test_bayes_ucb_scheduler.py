"""Falsification and adversarial tests for Bayes-UCB scheduler."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.bayes_ucb import (
    BayesUCBScheduler,
    _betai,
    beta_quantile,
)


class TestIncompleteBeta:
    """The regularized incomplete beta function I_x(a,b), the Beta CDF."""

    def test_betai_uniform_is_identity(self):
        """Beta(1,1) is Uniform(0,1): I_x(1,1) == x exactly."""
        for x in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            assert _betai(1.0, 1.0, x) == pytest.approx(x, abs=1e-9)

    def test_betai_symmetric_at_half_for_symmetric_beta(self):
        """Beta(a,a) is symmetric about 0.5: I_0.5(a,a) == 0.5."""
        for a in (0.5, 1.0, 2.0, 10.0):
            assert _betai(a, a, 0.5) == pytest.approx(0.5, abs=1e-6)

    def test_betai_boundary_values(self):
        assert _betai(2.0, 3.0, 0.0) == 0.0
        assert _betai(2.0, 3.0, 1.0) == 1.0

    def test_betai_monotonic_in_x(self):
        """The CDF must be non-decreasing in x for fixed (a, b)."""
        a, b = 3.0, 7.0
        xs = [i / 20.0 for i in range(21)]
        vals = [_betai(a, b, x) for x in xs]
        # Not strict=True: `vals[1:]` is one shorter than `vals` by
        # construction, so the pairwise-adjacent idiom raises ValueError
        # before it ever evaluates the comparison. Both monotonicity tests
        # here were unconditionally erroring rather than checking anything.
        assert all(v2 >= v1 - 1e-12 for v1, v2 in zip(vals, vals[1:], strict=False))


class TestBetaQuantile:
    def test_quantile_uniform_is_identity(self):
        """Beta(1,1) quantile function is the identity, within bisection tol."""
        for p in (0.1, 0.5, 0.9, 0.995):
            assert beta_quantile(p, 1.0, 1.0) == pytest.approx(p, abs=2e-3)

    def test_quantile_is_cdf_inverse(self):
        """betai(quantile(p)) recovers p, within the documented tolerance."""
        for a, b, p in [(2.0, 5.0, 0.9), (0.5, 0.5, 0.5), (50.0, 3.0, 0.995)]:
            x = beta_quantile(p, a, b)
            assert _betai(a, b, x) == pytest.approx(p, abs=5e-3)

    def test_quantile_boundary_probabilities(self):
        assert beta_quantile(0.0, 2.0, 3.0) == 0.0
        assert beta_quantile(1.0, 2.0, 3.0) == 1.0

    def test_quantile_monotonic_in_p(self):
        a, b = 4.0, 9.0
        ps = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        qs = [beta_quantile(p, a, b) for p in ps]
        assert all(q2 >= q1 for q1, q2 in zip(qs, qs[1:], strict=False))

    def test_quantile_shifts_right_with_more_evidence_of_success(self):
        """More observed successes at fixed n shifts the quantile up."""
        p = 0.9
        low = beta_quantile(p, 0.5 + 2, 0.5 + 18)  # 2/20 successes
        high = beta_quantile(p, 0.5 + 18, 0.5 + 2)  # 18/20 successes
        assert high > low


class TestConstructorValidation:
    def test_rejects_non_positive_prior_alpha(self):
        with pytest.raises(ValueError):
            BayesUCBScheduler(prior_alpha=0.0)

    def test_rejects_non_positive_prior_beta(self):
        with pytest.raises(ValueError):
            BayesUCBScheduler(prior_beta=-1.0)

    def test_rejects_negative_c(self):
        with pytest.raises(ValueError):
            BayesUCBScheduler(c=-0.1)

    def test_accepts_zero_c(self):
        BayesUCBScheduler(c=0.0)  # must not raise


class TestSharedContract:
    def test_declares_supports_priors(self):
        assert BayesUCBScheduler.supports_priors is True

    def test_empty_candidate_list(self):
        s = BayesUCBScheduler()
        assert s.select_op([]) == ""

    def test_single_candidate(self):
        s = BayesUCBScheduler()
        assert s.select_op(["only"]) == "only"

    def test_record_auto_registers_unknown_arm(self):
        s = BayesUCBScheduler()
        s.record("never_initialized", True)
        assert s.bandit_stats()["bayes_ucb_arms"] == 1

    def test_unpulled_arms_take_priority(self):
        """An arm with zero pulls must be picked over a heavily-pulled one."""
        s = BayesUCBScheduler()
        s.init_arm("seasoned")
        for _ in range(200):
            s.record("seasoned", True)  # near-certain high success rate
        s.init_arm("fresh")
        assert s.select_op(["seasoned", "fresh"]) == "fresh"


class TestPerArmPriors:
    """The reason this scheduler isn't a UCBBase subclass — see module docstring."""

    def test_init_arm_accepts_a_per_arm_prior_override(self):
        s = BayesUCBScheduler(prior_alpha=0.5, prior_beta=0.5)
        s.init_arm("informed", prior_alpha=40.0, prior_beta=2.0)
        s.init_arm("uninformed")
        # Before any evidence, the informed arm's prior alone should win —
        # its prior mean (40/42 ≈ 0.95) dwarfs the Jeffreys-prior arm's.
        assert s.select_op(["informed", "uninformed"]) == "informed"

    def test_prior_override_is_idempotent(self):
        """A second init_arm() call must not overwrite the first prior."""
        s = BayesUCBScheduler()
        s.init_arm("a", prior_alpha=10.0, prior_beta=1.0)
        s.init_arm("a", prior_alpha=1.0, prior_beta=10.0)  # must be ignored
        assert s._prior_alpha["a"] == 10.0
        assert s._prior_beta["a"] == 1.0

    def test_prior_alpha_and_beta_must_be_positive_after_clamping(self):
        """A degenerate override is clamped, not allowed to zero out the prior."""
        s = BayesUCBScheduler()
        s.init_arm("a", prior_alpha=0.0, prior_beta=-5.0)
        assert s._prior_alpha["a"] > 0.0
        assert s._prior_beta["a"] > 0.0


class TestAdversarialConvergence:
    """The falsification test: a scheduler with no mechanism, or one that
    ignored evidence, would not separate these arms. This one must.
    """

    def test_adversarial_converges_to_the_better_arm(self):
        s = BayesUCBScheduler()
        ops = ["good", "bad"]
        for op in ops:
            s.init_arm(op)
        rng = RandPool(seed=42)
        counts = {"good": 0, "bad": 0}
        for _ in range(3000):
            op = s.select_op(ops)
            counts[op] += 1
            ok = rng.random() < (0.35 if op == "good" else 0.05)
            s.record(op, ok)
        assert counts["good"] > counts["bad"] * 3, (
            f"expected 'good' to dominate selections, got {counts}"
        )

    def test_identical_priors_are_an_exact_tie_broken_by_ops_order(self):
        """Two arms with identical priors must score identically.

        This asserted the opposite -- that both arms get selected across 20
        seeds -- which only held because of the zero-evidence branch that
        picked uniformly at random, the branch the module docstring says
        must not exist. With it gone the scheduler is deterministic and an
        exact tie falls to the first candidate in `ops` order, the same
        convention every UCBBase score loop uses. What still needs pinning
        is that nothing *else* decides it: not dict insertion order, not
        alphabetical order, only the caller's list order.
        """
        s = BayesUCBScheduler()
        s.init_arm("y")  # registered first, deliberately out of alphabetical order
        s.init_arm("x")

        assert s.select_op(["x", "y"]) == "x"
        assert s.select_op(["y", "x"]) == "y", (
            "the tie-break must follow ops order, not registration or sort order"
        )


class TestQuantileOrderParameter:
    def test_higher_c_raises_the_quantile_order_for_large_t(self):
        """q_t = 1 - 1/(t*(log t)^c) increases with c for t > e (log t > 1)."""
        t = 1000.0
        log_t = math.log(t)
        q_c0 = 1.0 - 1.0 / (t * (log_t**0.0))
        q_c1 = 1.0 - 1.0 / (t * (log_t**1.0))
        assert q_c1 > q_c0

    def test_scheduler_does_not_crash_at_t_equals_one(self):
        """The first real (non-unpulled-branch) score computation guards t>=2."""
        s = BayesUCBScheduler(c=2.0)
        s.init_arm("a")
        s.init_arm("b")
        s.select_op(["a", "b"])  # unpulled branch
        s.record("a", True)
        s.record("b", False)
        # Both now have 1 pull; the next call must exercise the quantile
        # path with self._total_pulls == 2, and must not raise.
        s.select_op(["a", "b"])


class TestBanditStats:
    def test_bandit_stats_shape(self):
        s = BayesUCBScheduler(prior_alpha=1.0, prior_beta=2.0, c=1.5)
        s.init_arm("a")
        s.record("a", True)
        stats = s.bandit_stats()
        assert stats["bayes_ucb_pulls"] == 1
        assert stats["bayes_ucb_arms"] == 1
        assert stats["bayes_ucb_prior_alpha"] == 1.0
        assert stats["bayes_ucb_prior_beta"] == 2.0
        assert stats["bayes_ucb_c"] == 1.5

    def test_bandit_stats_are_json_safe(self):
        import json

        s = BayesUCBScheduler()
        s.init_arm("a")
        s.record("a", True)
        json.dumps(s.bandit_stats())  # must not raise
