"""Tests for ``core/schedulers/op_exp3_ix.py`` -- EXP3 with implicit exploration.

Read first:

- ``test_probabilities_match_the_softmax_oracle`` -- p = softmax(-eta L)
  against a plain-Python transcription.
- ``test_all_successes_stay_uniform`` (falsification) -- zero loss is no
  signal; any movement is estimator bias.
- ``test_huge_losses_stay_a_distribution`` (adversarial) -- the softmax
  must not overflow when cumulative losses reach 1e12.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_exp3_ix import EXP3IXScheduler
from tests.support.scripted_rng import ScriptedRng


def _with_arms(n: int, **kw) -> tuple[EXP3IXScheduler, list[str]]:
    kw.setdefault("rng", RandPool(seed=4))
    s = EXP3IXScheduler(**kw)
    arms = [f"op{i}" for i in range(n)]
    for a in arms:
        s.init_arm(a)
    return s, arms


def _softmax_oracle(losses: list[float], eta: float) -> list[float]:
    lo = min(losses)
    w = [math.exp(-eta * (x - lo)) for x in losses]
    total = sum(w)
    return [x / total for x in w]


# --------------------------------------------------------------------------
# Contract (Hard Rules 1 and 40)
# --------------------------------------------------------------------------


def test_declares_no_prior_support():
    assert EXP3IXScheduler.supports_priors is False


def test_has_the_operator_scheduler_interface():
    s, _ = _with_arms(1)
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf])
def test_rejects_bad_eta_scale(scale):
    with pytest.raises(ValueError):
        EXP3IXScheduler(eta_scale=scale)


def test_empty_candidate_list_is_not_a_crash():
    s, _ = _with_arms(0)
    assert s.select_op([]) == ""


# --------------------------------------------------------------------------
# Rates and the distribution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.5, 0.1, 0.0])
def test_rates_follow_neu_2015(ratio):
    """eta_t = scale sqrt(2 ln K / (K t)), gamma_t = ratio eta_t (Neu: 0.5)."""
    s, arms = _with_arms(5, eta_scale=1.5, gamma_ratio=ratio)
    for _ in range(9):
        s.record(s.select_op(arms), False)
    eta, gamma = s.rates(5)
    assert eta == pytest.approx(1.5 * math.sqrt(2.0 * math.log(5) / (5 * 10)))
    assert gamma == pytest.approx(ratio * eta)


@pytest.mark.parametrize("ratio", [-0.1, math.nan, math.inf])
def test_rejects_bad_gamma_ratio(ratio):
    with pytest.raises(ValueError):
        EXP3IXScheduler(gamma_ratio=ratio)


@pytest.mark.parametrize("seed", range(4))
def test_probabilities_match_the_softmax_oracle(seed):
    rnd = random.Random(seed)
    s, arms = _with_arms(9)
    losses = [rnd.uniform(0.0, 300.0) for _ in arms]
    s._lossv[:] = losses
    eta, _ = s.rates(len(arms))
    got = s.probabilities(arms)
    want = _softmax_oracle(losses, eta)
    assert np.allclose([got[a] for a in arms], want, rtol=1e-12)


def test_huge_losses_stay_a_distribution():
    s, arms = _with_arms(4)
    s._lossv[:] = [0.0, 1e12, -1e12, 5.0]
    probs = s.probabilities(arms)
    assert all(math.isfinite(p) for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


def test_failure_adds_loss_over_p_plus_gamma():
    """Two equal arms, u = 0.1 draws a at p = 1/2: L_a += 1 / (1/2 + gamma)."""
    s = EXP3IXScheduler(rng=ScriptedRng(randoms=[0.1]))
    ops = ["a", "b"]
    _, gamma = s.rates(2)
    s.record(s.select_op(ops), False)
    assert s._lossv[s._idx["a"]] == pytest.approx(1.0 / (0.5 + gamma))
    assert s._lossv[s._idx["b"]] == 0.0


def test_all_successes_stay_uniform():
    """Falsification: reward 1 is loss 0 every round."""
    s, arms = _with_arms(6)
    for _ in range(300):
        s.record(s.select_op(arms), True)
    probs = s.probabilities(arms)
    assert all(p == pytest.approx(1.0 / 6, rel=1e-12) for p in probs.values())


def test_record_is_on_policy():
    s, arms = _with_arms(3)
    s.record(arms[0], False)
    assert s.bandit_stats()["exp3_ix_orphans"] == 1
    assert np.all(s._lossv == 0.0)


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
    assert np.all(np.isfinite(s._lossv)) and np.all(s._lossv >= 0.0)


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


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
    assert stats["exp3_ix_pulls"] == 1
    assert stats["exp3_ix_arms"] == 2
