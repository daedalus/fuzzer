"""The probit (``core.gaussian.norm_ppf``) and Bayes-UCB's shortlist.

``select_op`` used to bisect the Beta posterior of every candidate arm to
find one argmax -- ~27us per arm, ~4ms for a realistic 150-operator
registry. The shortlist ranks well-evidenced arms with a closed-form
Cornish-Fisher estimate first and bisects only the top few, leaving
low-evidence arms (where the normal approximation is unreliable, and
where this scheduler's per-arm priors live) on the exact path.

The property that matters is not that the approximation is accurate --
it is deliberately not, see ``approx_beta_quantile`` -- but that the
shortlist never excludes the arm the exact pass would have chosen.
"""

from __future__ import annotations

import math
import random

import pytest

from fuzzer_tool.core.gaussian import norm_cdf, norm_ppf
from fuzzer_tool.core.schedulers.op_bayes_ucb import (
    SHORTLIST_EXACT_BELOW,
    SHORTLIST_K,
    BayesUCBScheduler,
    approx_beta_quantile,
    beta_quantile,
)


class TestNormPpf:
    def test_known_quantiles(self):
        assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
        assert norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-9)
        assert norm_ppf(0.025) == pytest.approx(-1.959963984540054, abs=1e-9)
        assert norm_ppf(0.995) == pytest.approx(2.5758293035489004, abs=1e-9)

    def test_round_trips_through_norm_cdf(self):
        """Phi(Phi^-1(p)) == p to double precision over the usable range."""
        for i in range(1, 1000):
            p = i / 1000.0
            assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-12, abs=1e-15)

    def test_round_trips_in_the_upper_tail(self):
        """Where Bayes-UCB actually evaluates: p = 1 - 1/t for large t."""
        for t in (10, 1000, 100_000, 10_000_000):
            p = 1.0 - 1.0 / t
            assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-12)

    def test_symmetry(self):
        for p in (0.001, 0.02, 0.1, 0.3, 0.49):
            assert norm_ppf(p) == pytest.approx(-norm_ppf(1.0 - p), rel=1e-9)

    def test_strictly_increasing(self):
        vals = [norm_ppf(i / 500.0) for i in range(1, 500)]
        assert all(b > a for a, b in zip(vals, vals[1:], strict=False))

    def test_open_endpoints_are_infinite(self):
        assert norm_ppf(0.0) == -math.inf
        assert norm_ppf(1.0) == math.inf
        assert norm_ppf(-0.5) == -math.inf
        assert norm_ppf(1.5) == math.inf


class TestApproxBetaQuantile:
    def test_bounded_to_the_unit_interval(self):
        rng = random.Random(1)
        for _ in range(2000):
            a = rng.uniform(0.01, 500.0)
            b = rng.uniform(0.01, 500.0)
            p = rng.uniform(1e-6, 1.0 - 1e-6)
            assert 0.0 <= approx_beta_quantile(p, a, b) <= 1.0

    def test_close_to_exact_on_concentrated_posteriors(self):
        """Close enough to rank with, nowhere near close enough to score."""
        rng = random.Random(2)
        errs = []
        for _ in range(500):
            n = rng.randint(60, 3000)
            k = rng.randint(0, n)
            p = 1.0 - 1.0 / rng.randint(100, 200000)
            a, b = 0.5 + k, 0.5 + n - k
            errs.append(abs(approx_beta_quantile(p, a, b) - beta_quantile(p, a, b)))
        assert sum(errs) / len(errs) < 0.01
        # And explicitly NOT accurate enough to be the index itself.
        assert max(errs) > 1e-3

    def test_never_under_estimates_a_cold_arm(self):
        """The error direction that makes the shortlist safe: an arm with
        little evidence must not be scored low and silently dropped.

        (40, 2) is deliberately absent -- its posterior mass is 42, above
        SHORTLIST_EXACT_BELOW, so it takes the exact path and the
        approximation's sign there is irrelevant. It does under-estimate
        such arms, which is exactly why the threshold exists.
        """
        p = 0.999
        for prior in ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 3.0), (10.0, 10.0)):
            assert sum(prior) < SHORTLIST_EXACT_BELOW
            assert approx_beta_quantile(p, *prior) >= beta_quantile(p, *prior) - 1e-9


def _loaded_scheduler(n_arms, cold_frac, rng):
    sched = BayesUCBScheduler()
    ops = [f"op{i}" for i in range(n_arms)]
    for op in ops:
        sched.init_arm(op)
        n = rng.randint(0, 25) if rng.random() < cold_frac else rng.randint(30, 3000)
        k = int(n * rng.betavariate(2, 8))
        sched._counts[op] = n
        sched._sums[op] = float(k)
        sched._total_pulls += n
    return sched, ops


def _exact_pick(sched, ops):
    """The pre-shortlist selection loop, verbatim, as an oracle."""
    t = max(sched._total_pulls, 2)
    q_t = 1.0 - 1.0 / (t * (math.log(t) ** sched.c))
    q_t = min(max(q_t, 1e-9), 1.0 - 1e-9)
    best, best_score, best_pulls = ops[0], -math.inf, math.inf
    for op in ops:
        n = sched._counts.get(op, 0)
        s = sched._sums.get(op, 0.0)
        a = sched._prior_alpha.get(op, sched.prior_alpha) + s
        b = sched._prior_beta.get(op, sched.prior_beta) + (n - s)
        score = beta_quantile(q_t, a, b)
        if score > best_score or (score == best_score and n < best_pulls):
            best_score, best_pulls, best = score, n, op
    return best


class TestShortlistAgreesWithExactSelection:
    """Hard Rule 46: the shortlisted selection against the loop it replaced."""

    @pytest.mark.parametrize("cold_frac", [0.0, 0.3, 1.0])
    def test_argmax_matches(self, cold_frac):
        rng = random.Random(int(cold_frac * 100) + 7)
        for _ in range(60):
            sched, ops = _loaded_scheduler(150, cold_frac, rng)
            assert sched.select_op(ops) == _exact_pick(sched, ops)

    def test_matches_on_small_registries(self):
        rng = random.Random(21)
        for n_arms in (2, 5, SHORTLIST_K, SHORTLIST_K + 1, 40):
            for _ in range(25):
                sched, ops = _loaded_scheduler(n_arms, 0.2, rng)
                assert sched.select_op(ops) == _exact_pick(sched, ops)

    def test_single_dominant_arm_is_found_from_the_back_of_the_list(self):
        """A shortlist that silently kept the first K entries would pass
        most random cases; put the winner last."""
        sched = BayesUCBScheduler()
        ops = [f"op{i}" for i in range(150)]
        for op in ops:
            sched.init_arm(op)
            sched._counts[op] = 500
            sched._sums[op] = 5.0
            sched._total_pulls += 500
        sched._sums[ops[-1]] = 480.0
        assert sched.select_op(ops) == ops[-1]


class TestShortlistMechanics:
    def test_low_evidence_arms_always_survive(self):
        """Below SHORTLIST_EXACT_BELOW an arm skips the approximation."""
        sched = BayesUCBScheduler()
        ops = [f"op{i}" for i in range(150)]
        for op in ops:
            sched.init_arm(op)
            sched._counts[op] = 2000
            sched._sums[op] = 1800.0
            sched._total_pulls += 2000
        cold = ops[100]
        sched._counts[cold] = 0
        sched._sums[cold] = 0.0

        t = max(sched._total_pulls, 2)
        q_t = min(max(1.0 - 1.0 / (t * (math.log(t) ** sched.c)), 1e-9), 1.0 - 1e-9)
        shortlist = sched._shortlist(ops, q_t)
        assert cold in shortlist
        assert len(shortlist) < len(ops)

    def test_shortlist_preserves_caller_order(self):
        """The exact loop's final tie-break is ops order; it must see the
        same order it would have seen."""
        rng = random.Random(33)
        sched, ops = _loaded_scheduler(150, 0.1, rng)
        t = max(sched._total_pulls, 2)
        q_t = min(max(1.0 - 1.0 / (t * (math.log(t) ** sched.c)), 1e-9), 1.0 - 1e-9)
        shortlist = sched._shortlist(ops, q_t)
        assert shortlist == [op for op in ops if op in set(shortlist)]

    def test_shortlist_is_a_subset_and_non_empty(self):
        rng = random.Random(34)
        for cold in (0.0, 0.5, 1.0):
            sched, ops = _loaded_scheduler(150, cold, rng)
            t = max(sched._total_pulls, 2)
            q_t = min(max(1.0 - 1.0 / (t * (math.log(t) ** sched.c)), 1e-9), 1.0 - 1e-9)
            shortlist = sched._shortlist(ops, q_t)
            assert shortlist
            assert set(shortlist) <= set(ops)

    def test_no_shortlisting_below_the_threshold_size(self):
        rng = random.Random(35)
        sched, ops = _loaded_scheduler(SHORTLIST_K, 0.0, rng)
        t = max(sched._total_pulls, 2)
        q_t = min(max(1.0 - 1.0 / (t * (math.log(t) ** sched.c)), 1e-9), 1.0 - 1e-9)
        assert sched._shortlist(ops, q_t) == ops

    def test_all_cold_degenerates_to_the_exact_path(self):
        """At true cold start every arm is below the threshold, so the
        shortlist is the identity -- no speedup, no semantic change."""
        sched = BayesUCBScheduler()
        ops = [f"op{i}" for i in range(150)]
        for op in ops:
            sched.init_arm(op)
        t = max(sched._total_pulls, 2)
        q_t = min(max(1.0 - 1.0 / (t * (math.log(t) ** sched.c)), 1e-9), 1.0 - 1e-9)
        assert sched._shortlist(ops, q_t) == ops
        assert sum(sched._posterior(op)) < SHORTLIST_EXACT_BELOW

    def test_per_arm_priors_still_decide_at_cold_start(self):
        """The reason low-evidence arms bypass the approximation: it clamps
        several distinct cold posteriors to 1.0 and would lose them."""
        sched = BayesUCBScheduler()
        ops = [f"op{i}" for i in range(150)]
        for op in ops:
            sched.init_arm(op, prior_alpha=1.0, prior_beta=5.0)
        favoured = ops[-1]
        sched._prior_alpha[favoured] = 5.0
        sched._prior_beta[favoured] = 1.0
        assert sched.select_op(ops) == favoured

    def test_is_faster_than_exact_when_warm(self):
        import time

        rng = random.Random(41)
        scheds = [_loaded_scheduler(150, 0.0, rng) for _ in range(12)]
        t0 = time.perf_counter()
        for sched, ops in scheds:
            _exact_pick(sched, ops)
        exact = time.perf_counter() - t0
        t0 = time.perf_counter()
        for sched, ops in scheds:
            sched.select_op(ops)
        shortlisted = time.perf_counter() - t0
        # Measured ~7x; assert a loose 2x so the test is not a
        # machine-speed flake while still catching a reverted shortlist.
        assert shortlisted * 2 < exact, (shortlisted, exact)
