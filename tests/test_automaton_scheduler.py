"""Tests for ``core/schedulers/op_automaton.py`` -- linear reward-inaction automaton.

Read first:

- ``test_success_applies_the_lri_update`` -- p_j += l r (1 - p_j), every
  other p_i scales by 1 - l r.
- ``test_failure_is_inaction`` (falsification) -- the "I" in L_R-I: no
  amount of failure may move the vector.
- ``test_mix_floor_survives_absorption`` (adversarial) -- after the vector
  collapses onto one arm, every offered arm keeps mix / n of the draw.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_automaton import LearningAutomatonScheduler
from tests.support.scripted_rng import ScriptedRng


def _with_arms(n: int, **kw) -> tuple[LearningAutomatonScheduler, list[str]]:
    kw.setdefault("rng", RandPool(seed=12))
    s = LearningAutomatonScheduler(**kw)
    arms = [f"op{i}" for i in range(n)]
    for a in arms:
        s.init_arm(a)
    return s, arms


def test_declares_no_prior_support():
    assert LearningAutomatonScheduler.supports_priors is False


def test_has_the_operator_scheduler_interface():
    s, _ = _with_arms(1)
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize(
    "kwargs", [{"rate": 0.0}, {"rate": 1.5}, {"rate": math.nan}, {"mix": -0.1}, {"mix": 0.5}]
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        LearningAutomatonScheduler(**kwargs)


def test_empty_candidate_list_is_not_a_crash():
    s, _ = _with_arms(0)
    assert s.select_op([]) == ""


def test_init_arm_keeps_the_simplex():
    s, _ = _with_arms(7)
    assert s._pv.sum() == pytest.approx(1.0)
    assert np.allclose(s._pv, 1.0 / 7)


def test_success_applies_the_lri_update():
    rate, r = 0.1, 0.5
    s = LearningAutomatonScheduler(rate=rate, mix=0.0, rng=ScriptedRng(randoms=[0.1]))
    ops = ["a", "b", "c", "d"]
    s.record(s.select_op(ops), True, weight=r)
    step = rate * r
    assert s._pv[s._idx["a"]] == pytest.approx(0.25 + step * (1 - 0.25))
    assert s._pv[s._idx["b"]] == pytest.approx(0.25 * (1 - step))
    assert s._pv.sum() == pytest.approx(1.0)


def test_failure_is_inaction():
    s, arms = _with_arms(5)
    before = s._pv.copy()
    for _ in range(500):
        s.record(s.select_op(arms), False)
    assert np.array_equal(s._pv, before)


def test_mix_floor_survives_absorption():
    mix = 0.1
    s, arms = _with_arms(4, rate=0.5, mix=mix)
    s._pv[:] = [1.0, 0.0, 0.0, 0.0]
    probs = s.probabilities(arms)
    assert min(probs.values()) >= mix / 4 - 1e-15
    assert sum(probs.values()) == pytest.approx(1.0)


def test_without_mix_it_absorbs():
    """L_R-I converges to a unit vector: rate 0.2, only arm a ever pays."""
    s, arms = _with_arms(3, rate=0.2, mix=0.0)
    for _ in range(400):
        op = s.select_op(arms)
        s.record(op, op == arms[0])
    assert s._pv[0] > 0.999


def test_record_is_on_policy():
    s, arms = _with_arms(3)
    before = s._pv.copy()
    s.record(arms[0], True)
    assert s.bandit_stats()["automaton_orphans"] == 1
    assert np.array_equal(s._pv, before)


def test_unconsumed_draws_do_not_grow_without_bound():
    s, _ = _with_arms(0)
    for i in range(1000):
        s.select_op([f"a{i}", f"b{i}"])
    assert len(s._pending) <= 256


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e300])
def test_adversarial_weights_keep_the_simplex(weight):
    s, arms = _with_arms(3, rate=1.0)
    for _ in range(50):
        s.record(s.select_op(arms), True, weight=weight)
    assert np.all(s._pv >= 0.0)
    assert s._pv.sum() == pytest.approx(1.0)


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
    assert stats["automaton_pulls"] == 1
    assert 0.0 < stats["automaton_concentration"] <= 1.0


def test_warm_start_draws_every_offered_arm_once():
    """Adversarial for reach: no RNG draw is scripted, so sampling would
    raise. Each never-drawn offered arm is played first, in order."""
    s = LearningAutomatonScheduler(rng=ScriptedRng())
    ops = ["a", "b", "c"]
    assert [s.select_op(ops) for _ in ops] == ops


def test_regression_equal_arms_are_all_reached():
    """252 equal 5% arms over 20k rounds left 1-2 never drawn on 3-4 of 20
    seeds before the warm start."""
    arms = [f"op{i}" for i in range(252)]
    env = random.Random(1234)
    for seed in range(3):
        s = LearningAutomatonScheduler(rng=RandPool(seed))
        seen = set()
        for _ in range(len(arms)):
            op = s.select_op(arms)
            seen.add(op)
            s.record(op, env.random() < 0.05)
        assert seen == set(arms)
