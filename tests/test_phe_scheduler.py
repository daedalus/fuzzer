"""Tests for ``core/schedulers/op_phe.py`` -- Perturbed-History Exploration.

Read first:

- ``test_index_matches_the_formula`` -- (V + U) / (s + ceil(a s)) with U
  scripted, against the formula written out.
- ``test_without_perturbation_it_is_greedy`` (falsification) -- pin U at its
  mean and PHE never re-tries a worse arm: the Binomial noise is the only
  exploration.
- ``test_unpulled_arms_draw_nothing`` (adversarial) -- an unpulled arm is
  played before any pseudo-reward is drawn.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_phe import PHEScheduler
from tests.support.scripted_rng import ScriptedRng


def _sched(**kw) -> PHEScheduler:
    kw.setdefault("rng", RandPool(seed=21))
    return PHEScheduler(**kw)


class _RecordingRng:
    """Captures the pseudo-reward counts and returns their means."""

    def __init__(self) -> None:
        self.counts: list[list[int]] = []

    def binomial_array(self, counts, p):
        self.counts.append([int(c) for c in counts])
        return np.floor(np.asarray(counts) * p)


# --------------------------------------------------------------------------
# Contract (Hard Rules 1 and 40)
# --------------------------------------------------------------------------


def test_declares_prior_support():
    assert PHEScheduler.supports_priors is True


def test_has_the_operator_scheduler_interface():
    s = _sched()
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize("a", [0.0, -1.0, math.nan, math.inf])
def test_rejects_bad_perturbation_scale(a):
    with pytest.raises(ValueError):
        PHEScheduler(a=a)


def test_prior_becomes_pseudo_history():
    """Beta(3, 5) is 2 successes in 6 pulls of history."""
    s = _sched()
    s.init_arm("x", 3.0, 5.0)
    s.init_arm("x", 1.0, 1.0)  # idempotent: first registration wins
    assert s.history("x") == (6.0, 2.0)


def test_empty_candidate_list_is_not_a_crash():
    assert _sched().select_op([]) == ""


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------


def test_index_matches_the_formula():
    a = 1.5
    u = [3, 0, 7]
    s = PHEScheduler(a=a, rng=ScriptedRng(binomial_arrays=[np.array(u)]))
    ops = ["x", "y", "z"]
    hist = {"x": (4, 2.0), "y": (2, 1.5), "z": (6, 1.0)}
    for op, (pulls, reward) in hist.items():
        for _ in range(pulls):
            s.record(op, True, weight=reward / pulls)
    want = []
    for op, uu in zip(ops, u, strict=True):
        pulls, reward = hist[op]
        m = math.ceil(a * pulls)
        want.append((reward + uu) / (pulls + m))
    assert s.select_op(ops) == ops[int(np.argmax(want))]


def test_pseudo_reward_count_is_ceil_a_times_pulls():
    rng = _RecordingRng()
    s = PHEScheduler(a=1.1, rng=rng)
    for op, pulls in (("x", 3), ("y", 10)):
        for _ in range(pulls):
            s.record(op, False)
    s.select_op(["x", "y"])
    assert rng.counts == [[math.ceil(1.1 * 3), math.ceil(1.1 * 10)]]


def test_unpulled_arms_draw_nothing():
    """Adversarial: ScriptedRng has no binomial arrays, so any draw raises.
    The first unpulled offered arm must be returned without one."""
    s = PHEScheduler(rng=ScriptedRng())
    s.record("x", True)
    assert s.select_op(["x", "y", "z"]) == "y"


def test_huge_histories_stay_finite():
    """Adversarial: 10^12 pulls must not overflow the pseudo-reward count."""
    s = _sched()
    s.init_arm("x")
    s.init_arm("y")
    s._pullv[:] = 1e12
    s._rewv[:] = [3e11, 1e11]
    assert s.select_op(["x", "y"]) == "x"


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e300])
def test_adversarial_weights_are_clamped(weight):
    s = _sched()
    for _ in range(10):
        s.record("x", True, weight=weight)
    pulls, reward = s.history("x")
    assert pulls == 10.0 and 0.0 <= reward <= 10.0


def test_failure_is_zero_reward_whatever_the_weight():
    s = _sched()
    s.record("x", False, weight=0.9)
    assert s.history("x") == (1.0, 0.0)


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


class _MeanRng:
    """Pseudo-rewards pinned at their mean: PHE with the noise removed."""

    def binomial_array(self, counts, p):
        return np.asarray(counts) * p


def test_without_perturbation_it_is_greedy():
    """Falsification: x leads after one round each; with U fixed at m/2 the
    index is a monotone shrink of the empirical mean, so y (and z) are never
    re-tried, whatever they would have paid."""
    s = PHEScheduler(rng=_MeanRng())
    ops = ["x", "y", "z"]
    s.record("x", True)
    s.record("y", False)
    s.record("z", False)
    picks = []
    for _ in range(200):
        op = s.select_op(ops)
        picks.append(op)
        s.record(op, op == "x")
    assert set(picks) == {"x"}


def test_perturbation_retries_an_early_loser():
    """Contrast: the same start with real Binomial noise re-tries y."""
    s = _sched()
    ops = ["x", "y"]
    s.record("x", True)
    s.record("y", False)
    picks = set()
    for _ in range(200):
        op = s.select_op(ops)
        picks.add(op)
        s.record(op, False)
    assert picks == {"x", "y"}


def test_a_productive_operator_dominates():
    s = _sched()
    arms = [f"op{i}" for i in range(12)]
    env = random.Random(7)
    picks = 0
    for t in range(3000):
        op = s.select_op(arms)
        s.record(op, env.random() < (0.3 if op == arms[5] else 0.03))
        picks += t >= 2000 and op == arms[5]
    assert picks > 600


def test_bandit_stats_reports_pulls():
    s = _sched()
    s.record(s.select_op(["x", "y"]), True)
    stats = s.bandit_stats()
    assert stats["phe_pulls"] == 1
    assert stats["phe_arms"] == 2
    assert stats["phe_unpulled"] == 1
