"""Tests for core/pool_posterior.py -- exact posterior over pooled-test constraints."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from fuzzer_tool.core import pool_posterior as pp

PRIOR = 0.2


def _brute(n: int, obs: list[pp.Obs], prior: float) -> np.ndarray:
    """Marginals by enumerating all 2^n assignments."""
    mass = np.zeros(n)
    total = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        if not all(o.lo <= sum(bits[i] for i in o.pool) <= o.hi for o in obs):
            continue
        w = (prior / (1 - prior)) ** sum(bits)
        total += w
        mass += w * np.array(bits)
    return mass / total


class TestPosterior:
    def test_matches_brute_force_or(self):
        rng = random.Random(3)
        n = 9
        for _ in range(20):
            pools = [frozenset(rng.sample(range(n), rng.randint(1, 5))) for _ in range(4)]
            truth = set(rng.sample(range(n), 2))
            obs = [pp.or_obs(p, bool(p & truth)) for p in pools]
            post = pp.posterior(n, obs, prior=PRIOR, max_def=n)
            assert post.exact
            assert np.allclose(post.marginals(), _brute(n, obs, PRIOR))

    def test_matches_brute_force_counts(self):
        obs = [pp.Obs(frozenset({0, 1, 2}), 2, 2), pp.Obs(frozenset({2, 3}), 1, 1)]
        post = pp.posterior(5, obs, prior=PRIOR, max_def=5)
        assert np.allclose(post.marginals(), _brute(5, obs, PRIOR))

    def test_p_clear_matches_brute_force(self):
        obs = [pp.or_obs(frozenset({0, 1, 2}), True), pp.or_obs(frozenset({2, 3}), False)]
        post = pp.posterior(6, obs, prior=PRIOR, max_def=6)
        pool = frozenset({0, 4})
        p_hit = _brute(6, obs + [pp.Obs(pool, 1, 2)], PRIOR)  # shape check only
        assert p_hit.shape == (6,)
        # P(no defective in pool) = P(0 and 4 both clear)
        want = _both_clear(6, obs, pool)
        assert post.p_clear(pool) == pytest.approx(want)

    def test_lone_candidate_is_certain(self):
        obs = [pp.or_obs(frozenset({0, 1}), True), pp.or_obs(frozenset({1}), False)]
        post = pp.posterior(4, obs, prior=PRIOR, max_def=4)
        assert post.marginals()[0] == pytest.approx(1.0)
        assert post.marginals()[1] == 0.0

    def test_unconstrained_items_keep_prior(self):
        post = pp.posterior(5, [pp.or_obs(frozenset({0, 1}), True)], prior=PRIOR, max_def=5)
        assert post.marginals()[4] == pytest.approx(PRIOR)


def _both_clear(n: int, obs: list[pp.Obs], pool: frozenset[int]) -> float:
    ok = total = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        if not all(o.lo <= sum(bits[i] for i in o.pool) <= o.hi for o in obs):
            continue
        w = (PRIOR / (1 - PRIOR)) ** sum(bits)
        total += w
        if not any(bits[i] for i in pool):
            ok += w
    return ok / total


class TestIdentify:
    @pytest.mark.parametrize("n,d", [(16, 1), (40, 2), (64, 3)])
    def test_exact(self, n, d):
        rng = random.Random(n * d)
        for _ in range(10):
            truth = set(rng.sample(range(n), d))
            res = pp.identify(n, lambda p, t=truth: bool(p & t), d=d)
            assert res.defective == truth
            assert not res.truncated

    def test_beats_split_search_in_tests(self):
        from fuzzer_tool.core import group_testing as gt

        rng = random.Random(5)
        n, d = 64, 2
        mine = base = 0
        for _ in range(15):
            truth = set(rng.sample(range(n), d))
            oracle = lambda p, t=truth: bool(p & t)  # noqa: E731
            mine += pp.identify(n, oracle, d=d).tests
            base += gt.split_search(n, oracle).tests
        assert mine < base

    def test_no_defectives_is_cheap(self):
        res = pp.identify(32, lambda p: False, d=1)
        assert res.defective == set()
        assert res.tests <= 3

    def test_underestimated_d_still_exact(self):
        truth = {1, 7, 20}
        res = pp.identify(32, lambda p: bool(p & truth), d=1)
        assert res.defective == truth


class TestAdversarial:
    def test_contradiction_raises(self):
        pool = frozenset({0, 1})
        with pytest.raises(ValueError):
            pp.posterior(3, [pp.or_obs(pool, True), pp.or_obs(pool, False)], prior=PRIOR, max_def=3)

    def test_max_def_too_low_raises_no_solution(self):
        obs = [pp.Obs(frozenset({0, 1}), 2, 2)]
        with pytest.raises(pp.NoSolution):
            pp.posterior(3, obs, prior=PRIOR, max_def=1)

    @pytest.mark.parametrize(
        "kw", [{"n": 0, "prior": 0.1}, {"n": 3, "prior": 0.0}, {"n": 3, "prior": 1.0}]
    )
    def test_bad_arguments(self, kw):
        with pytest.raises(ValueError):
            pp.posterior(kw["n"], [], prior=kw["prior"], max_def=1)

    def test_out_of_range_pool_rejected(self):
        with pytest.raises(ValueError):
            pp.posterior(3, [pp.or_obs(frozenset({5}), True)], prior=PRIOR, max_def=2)

    def test_budget_overflow_flags_inexact_and_stays_sound(self):
        # Big positive pool + tiny budget: not enumerated, never claims certainty.
        n = 60
        obs = [pp.or_obs(frozenset(range(n)), True)]
        post = pp.posterior(n, obs, prior=0.05, max_def=3, max_sol=50)
        assert not post.exact
        assert (post.marginals() < 1.0).all()

    def test_identify_reports_truncation_when_inexact(self):
        truth = {3, 50}
        res = pp.identify(64, lambda p: bool(p & truth), d=2, max_sol=10)
        assert res.truncated
        assert not res.exact

    def test_max_tests_truncates(self):
        truth = {3, 9}
        res = pp.identify(64, lambda p: bool(p & truth), d=2, max_tests=2)
        assert res.truncated and res.tests == 2
