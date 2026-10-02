"""Tests for ``core/schedulers/op_regret_matching.py`` -- bandit regret matching+.

Read first:

- ``test_update_matches_the_rm_plus_rule`` -- one success moves the drawn
  arm's clipped regret by r/p - r and charges the rest of the ballot r.
- ``test_failures_only_stays_uniform`` (falsification) -- reward 0 is no
  regret for anyone; the strategy must not move.
- ``test_regret_is_never_negative`` (adversarial) -- RM+ clipping holds
  under a long stream of large rewards.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_regret_matching import RegretMatchingScheduler
from tests.support.scripted_rng import ScriptedRng


def _with_arms(n: int, **kw) -> tuple[RegretMatchingScheduler, list[str]]:
    kw.setdefault("rng", RandPool(seed=8))
    s = RegretMatchingScheduler(**kw)
    arms = [f"op{i}" for i in range(n)]
    for a in arms:
        s.init_arm(a)
    return s, arms


def test_declares_no_prior_support():
    assert RegretMatchingScheduler.supports_priors is False


def test_has_the_operator_scheduler_interface():
    s, _ = _with_arms(1)
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize("mix", [-0.1, 0.5, math.nan])
def test_rejects_bad_mix(mix):
    with pytest.raises(ValueError):
        RegretMatchingScheduler(mix=mix)


def test_empty_candidate_list_is_not_a_crash():
    s, _ = _with_arms(0)
    assert s.select_op([]) == ""


def test_zero_regret_is_uniform():
    s, arms = _with_arms(5)
    assert all(p == pytest.approx(0.2) for p in s.probabilities(arms).values())


def test_probabilities_match_the_formula():
    mix = 0.1
    s, arms = _with_arms(4, mix=mix)
    s._regv[:] = [0.0, 3.0, 1.0, 0.0]
    probs = s.probabilities(arms)
    want = [(1 - mix) * r / 4.0 + mix / 4 for r in (0.0, 3.0, 1.0, 0.0)]
    assert [probs[a] for a in arms] == pytest.approx(want)


def test_update_matches_the_rm_plus_rule():
    """mix = 0, 3 equal arms: u = 0.1 draws a at p = 1/3. A success worth
    r = 0.6 gives a regret r/p - r and b, c max(0, -r) = 0. Then a draw of
    b at its new p charges a by r: max(0, R_a - r)."""
    r = 0.6
    s = RegretMatchingScheduler(mix=0.0, rng=ScriptedRng(randoms=[0.1, 0.9]))
    ops = ["a", "b", "c"]
    for _ in ops:  # warm start: one failure each moves no regret
        s.record(s.select_op(ops), False)
    s.record(s.select_op(ops), True, weight=r)
    ra = r / (1.0 / 3.0) - r
    assert s._regv[s._idx["a"]] == pytest.approx(ra)
    assert s._regv[s._idx["b"]] == 0.0

    # a now holds all the positive regret, so p = 1 on a: 0.9 still draws a.
    s.record(s.select_op(ops), True, weight=r)
    assert s._regv[s._idx["a"]] == pytest.approx(ra + (r / 1.0 - r))


def test_offered_ballot_only_is_charged():
    """An arm not offered in the round is not charged the obtained reward."""
    s = RegretMatchingScheduler(rng=ScriptedRng(randoms=[0.1, 0.1]))
    s.init_arm("z")
    s._regv[s._idx["z"]] = 2.0
    s.record(s.select_op(["a", "b"]), True, weight=0.5)
    assert s._regv[s._idx["z"]] == 2.0


def test_failures_only_stays_uniform():
    s, arms = _with_arms(6)
    for _ in range(500):
        s.record(s.select_op(arms), False)
    assert np.all(s._regv == 0.0)


def test_regret_is_never_negative():
    s, arms = _with_arms(5)
    env = random.Random(2)
    for _ in range(2000):
        s.record(s.select_op(arms), env.random() < 0.5, weight=1.0)
        assert s._regv.min() >= 0.0


def test_record_is_on_policy():
    s, arms = _with_arms(3)
    s.record(arms[0], True)
    assert s.bandit_stats()["regret_matching_orphans"] == 1
    assert np.all(s._regv == 0.0)


def test_unconsumed_draws_do_not_grow_without_bound():
    s, _ = _with_arms(0)
    for i in range(1000):
        s.select_op([f"a{i}", f"b{i}"])
    assert len(s._pending) <= 256


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e300])
def test_adversarial_weights_are_clamped(weight):
    s, arms = _with_arms(3)
    for _ in range(50):
        s.record(s.select_op(arms), True, weight=weight)
    assert np.all(np.isfinite(s._regv))
    assert sum(s.probabilities(arms).values()) == pytest.approx(1.0)


def test_a_productive_operator_gains_probability():
    s, arms = _with_arms(12)
    env = random.Random(7)
    for _ in range(3000):
        op = s.select_op(arms)
        s.record(op, env.random() < (0.3 if op == arms[5] else 0.03))
    probs = s.probabilities(arms)
    assert max(probs, key=probs.get) == arms[5]


def test_bandit_stats_reports_pulls():
    s, arms = _with_arms(2)
    s.record(s.select_op(arms), True)
    stats = s.bandit_stats()
    assert stats["regret_matching_pulls"] == 1
    assert stats["regret_matching_arms"] == 2


def test_warm_start_draws_every_offered_arm_once():
    """Adversarial for reach: no RNG draw is scripted, so sampling would
    raise. Each never-drawn offered arm is played first, in order."""
    s = RegretMatchingScheduler(rng=ScriptedRng())
    ops = ["a", "b", "c"]
    assert [s.select_op(ops) for _ in ops] == ops


def test_regression_equal_arms_are_all_reached():
    """252 equal 5% arms over 20k rounds left 1-2 never drawn on 3-4 of 20
    seeds before the warm start."""
    arms = [f"op{i}" for i in range(252)]
    env = random.Random(1234)
    for seed in range(3):
        s = RegretMatchingScheduler(rng=RandPool(seed))
        seen = set()
        for _ in range(len(arms)):
            op = s.select_op(arms)
            seen.add(op)
            s.record(op, env.random() < 0.05)
        assert seen == set(arms)
