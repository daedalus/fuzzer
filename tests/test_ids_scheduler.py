"""Tests for ``core/schedulers/op_ids.py`` -- variance-based Information-Directed Sampling.

Read first:

- ``test_gap_and_gain_match_the_loop_oracle`` -- the vectorised regret gap
  and information gain against plain-Python loops over the sample matrix.
- ``test_pair_mixing_matches_a_grid_search`` -- the closed-form q* against a
  brute-force minimisation of the information ratio.
- ``test_never_plays_a_surely_worse_arm`` (falsification) and
  ``test_identical_posteriors_are_not_a_crash`` (adversarial).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_ids import (
    IDSScheduler,
    _pair_search,
    ids_gap_gain,
    ids_pair,
)
from tests.support.scripted_rng import ScriptedRng


def _sched(**kw) -> IDSScheduler:
    kw.setdefault("rng", RandPool(seed=17))
    return IDSScheduler(**kw)


class _CountingPool(RandPool):
    """RandPool that counts posterior-sample batches."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.batches = 0

    def betavariate_array(self, alphas, betas):
        self.batches += 1
        return super().betavariate_array(alphas, betas)


# --------------------------------------------------------------------------
# Contract (Hard Rules 1 and 40)
# --------------------------------------------------------------------------


def test_declares_prior_support():
    assert IDSScheduler.supports_priors is True


def test_has_the_operator_scheduler_interface():
    s = _sched()
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize("kwargs", [{"samples": 1}, {"refresh": 0}])
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        IDSScheduler(**kwargs)


def test_prior_is_kept_and_idempotent():
    s = _sched()
    s.init_arm("x", 4.0, 2.0)
    s.init_arm("x", 1.0, 1.0)
    assert s.posterior("x") == (4.0, 2.0)


def test_empty_candidate_list_is_not_a_crash():
    assert _sched().select_op([]) == ""


def test_select_op_registers_unseen_operators():
    s = _sched()
    assert s.select_op(["x", "y"]) in ("x", "y")
    assert s.bandit_stats()["ids_arms"] == 2


# --------------------------------------------------------------------------
# The information ratio
# --------------------------------------------------------------------------


def _loop_oracle(theta: list[list[float]]) -> tuple[list[float], list[float]]:
    m, k = len(theta), len(theta[0])
    best = [max(range(k), key=lambda j, r=row: r[j]) for row in theta]
    rho = sum(max(row) for row in theta) / m
    mean = [sum(row[j] for row in theta) / m for j in range(k)]
    gap = [rho - mean[j] for j in range(k)]
    gain = [0.0] * k
    for a in set(best):
        rows = [row for row, b in zip(theta, best, strict=True) if b == a]
        pa = len(rows) / m
        for j in range(k):
            cond = sum(row[j] for row in rows) / len(rows)
            gain[j] += pa * (cond - mean[j]) ** 2
    return gap, gain


@pytest.mark.parametrize(("m", "k", "seed"), [(8, 2, 0), (64, 5, 1), (128, 30, 2)])
def test_gap_and_gain_match_the_loop_oracle(m, k, seed):
    rnd = random.Random(seed)
    theta = [[rnd.random() for _ in range(k)] for _ in range(m)]
    gap, gain = ids_gap_gain(np.asarray(theta))
    want_gap, want_gain = _loop_oracle(theta)
    assert np.allclose(gap, want_gap, rtol=1e-12, atol=1e-15)
    assert np.allclose(gain, want_gain, rtol=1e-12, atol=1e-15)


def _ratio(gap, gain, i, j, q):
    num = (q * gap[i] + (1 - q) * gap[j]) ** 2
    den = q * gain[i] + (1 - q) * gain[j]
    if den <= 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


@pytest.mark.parametrize("seed", range(6))
def test_pair_mixing_matches_a_grid_search(seed):
    rnd = random.Random(seed)
    k = 7
    gap = np.array([rnd.uniform(0.0, 0.3) for _ in range(k)])
    gain = np.array([rnd.uniform(1e-4, 0.05) for _ in range(k)])
    i, j, q = ids_pair(gap, gain)
    got = _ratio(gap, gain, i, j, q)
    grid = [g / 2000 for g in range(2001)]
    best = min(_ratio(gap, gain, a, b, x) for a in range(k) for b in range(k) for x in grid)
    assert got <= best + 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_frontier_pruning_keeps_the_exhaustive_optimum(seed):
    """Posterior-shaped instances at K = 120: pruning to the Pareto frontier
    must not raise the ratio above the full K^2 search."""
    pool = RandPool(seed)
    k, m = 120, 128
    rs = np.random.default_rng(seed)
    a = np.broadcast_to(rs.uniform(0.5, 60.0, k), (m, k))
    b = np.broadcast_to(rs.uniform(1.0, 600.0, k), (m, k))
    gap, gain = ids_gap_gain(pool.betavariate_array(a, b))
    full = _ratio(gap, gain, *_pair_search(gap, gain))
    assert _ratio(gap, gain, *ids_pair(gap, gain)) <= full + 1e-15


def test_a_certain_best_arm_is_played_deterministically():
    """Gap 0 and gain 0: ratio 0/0 is the exploit case, ratio 0."""
    i, j, q = ids_pair(np.array([0.0, 0.2]), np.array([0.0, 0.01]))
    assert (i, q) == (0, 1.0) or (j, q) == (0, 0.0)


def test_never_plays_a_surely_worse_arm():
    """Falsification: arm a always samples above b, so A* = a in every row,
    b's gain is 0 and its gap positive. IDS must never draw b."""
    theta = np.tile([0.6, 0.1], (16, 1))
    rng = ScriptedRng(beta_arrays=[theta], randoms=[0.999] * 20)
    s = IDSScheduler(samples=16, refresh=1000, rng=rng)
    assert {s.select_op(["a", "b"]) for _ in range(20)} == {"a"}


def test_identical_posteriors_are_not_a_crash():
    """Adversarial: every sample equal, so gap = gain = 0 everywhere."""
    theta = np.full((16, 4), 0.25)
    rng = ScriptedRng(beta_arrays=[theta], randoms=[0.3])
    s = IDSScheduler(samples=16, rng=rng)
    assert s.select_op(["a", "b", "c", "d"]) in ("a", "b", "c", "d")


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e300])
def test_adversarial_weights_are_clamped(weight):
    s = _sched()
    for _ in range(20):
        s.record("a", True, weight=weight)
    a, b = s.posterior("a")
    assert math.isfinite(a) and math.isfinite(b)
    assert a + b == pytest.approx(2.0 + 20)


# --------------------------------------------------------------------------
# Amortisation
# --------------------------------------------------------------------------


def test_policy_is_resampled_only_every_refresh_records():
    pool = _CountingPool(5)
    s = IDSScheduler(samples=32, refresh=4, rng=pool)
    ops = ["a", "b", "c"]
    for _ in range(12):
        s.record(s.select_op(ops), False)
    # Fresh at round 0, then stale after 4 and 8 records.
    assert pool.batches == 3


def test_policy_cache_is_bounded():
    s = _sched()
    for i in range(100):
        s.select_op([f"a{i}", f"b{i}"])
    assert len(s._policies) <= 16


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


def test_record_is_a_fractional_bernoulli():
    s = _sched()
    s.record("a", True, weight=0.25)
    s.record("a", False, weight=0.9)
    assert s.posterior("a") == (1.0 + 0.25, 1.0 + 0.75 + 1.0)


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
    s.record(s.select_op(["a", "b"]), True)
    stats = s.bandit_stats()
    assert stats["ids_pulls"] == 1
    assert stats["ids_arms"] == 2
