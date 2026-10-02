"""Tests for ``core/schedulers/op_tsallis.py`` -- 1/2-Tsallis-INF over operators.

Read first:

- ``test_solver_matches_the_bisection_oracle`` -- the vectorised Newton
  normaliser agrees with a plain-Python bisection written independently.
- ``test_centred_loss_stays_uniform`` (falsification) -- a loss equal to
  the estimator's shift carries no signal, so the distribution must not
  move; any drift is estimator bias.
- ``test_recovers_when_the_best_arm_switches`` (adversarial) -- the
  best-of-both-worlds claim: an arm that stops paying is abandoned.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_tsallis import TsallisINFScheduler, tsallis_probs
from tests.support.scripted_rng import ScriptedRng


def _with_arms(n: int, **kw) -> tuple[TsallisINFScheduler, list[str]]:
    kw.setdefault("rng", RandPool(seed=1234))
    s = TsallisINFScheduler(**kw)
    arms = [f"op{i}" for i in range(n)]
    for a in arms:
        s.init_arm(a)
    return s, arms


def _oracle(losses: list[float], eta: float) -> list[float]:
    """Bisection for x with sum 4 / (eta (L_i - x))^2 = 1, plain Python."""
    k = len(losses)
    lo = min(losses) - 2.0 * math.sqrt(k) / eta  # every term <= 1/k
    hi = min(losses) - 2.0 / eta  # the smallest term alone is 1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = sum(4.0 / (eta * (L - mid)) ** 2 for L in losses)
        if total > 1.0:
            hi = mid
        else:
            lo = mid
    w = [4.0 / (eta * (L - lo)) ** 2 for L in losses]
    s = sum(w)
    return [x / s for x in w]


# --------------------------------------------------------------------------
# Contract (Hard Rules 1 and 40)
# --------------------------------------------------------------------------


def test_declares_no_prior_support():
    assert TsallisINFScheduler.supports_priors is False


def test_has_the_operator_scheduler_interface():
    s, _ = _with_arms(1)
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize("kwargs", [{"eta": 0.0}, {"eta": -1.0}, {"mix": -0.1}, {"mix": 0.5}])
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        TsallisINFScheduler(**kwargs)


def test_empty_candidate_list_is_not_a_crash():
    s, _ = _with_arms(0)
    assert s.select_op([]) == ""


def test_select_op_registers_unseen_operators():
    s, _ = _with_arms(0)
    assert s.select_op(["x", "y"]) in ("x", "y")
    assert s.bandit_stats()["registered"] == 2


# --------------------------------------------------------------------------
# The normaliser
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "seed"), [(2, 0), (12, 1), (155, 2), (260, 3)])
def test_solver_matches_the_bisection_oracle(k, seed):
    rnd = random.Random(seed)
    losses = [rnd.uniform(-50.0, 50.0) for _ in range(k)]
    eta = rnd.uniform(0.01, 2.0)
    got = tsallis_probs(np.asarray(losses), eta)
    assert np.allclose(got, _oracle(losses, eta), rtol=1e-9, atol=1e-12)


def test_equal_losses_give_uniform():
    got = tsallis_probs(np.full(7, 3.5), 0.4)
    assert np.allclose(got, 1.0 / 7, rtol=1e-12)


def test_lower_loss_gets_more_mass():
    got = tsallis_probs(np.array([0.0, 5.0, 10.0]), 0.5)
    assert got[0] > got[1] > got[2] > 0.0


def test_mass_decays_polynomially_not_exponentially():
    """The 1/2-Tsallis signature: p ~ 4 / (eta * gap)^2 far from the leader,
    where EXP3 would give exp(-eta * gap) -- here ~1e-217, i.e. dead."""
    eta, gap = 1.0, 500.0
    got = tsallis_probs(np.array([0.0, gap]), eta)
    assert got[1] == pytest.approx(4.0 / (eta * gap) ** 2, rel=0.05)


def test_extreme_losses_stay_a_distribution():
    """Adversarial magnitudes: spread over 24 orders of magnitude."""
    losses = np.array([-1e12, 0.0, 1e-12, 1e12])
    got = tsallis_probs(losses, 1e-3)
    assert np.all(np.isfinite(got)) and np.all(got > 0.0)
    assert got.sum() == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_scripted_draw_inverts_the_cdf():
    """Uniform over 4 arms: u lands in arm floor(4u)."""
    u = 0.6
    s = TsallisINFScheduler(rng=ScriptedRng(randoms=[u]))
    ops = ["a", "b", "c", "d"]
    assert s.select_op(ops) == ops[math.floor(u * len(ops))]


def test_mixing_floor_guarantees_every_offered_arm_a_share():
    s, arms = _with_arms(3, mix=0.3)
    s._lossv[:] = [0.0, 1e6, 1e6]
    probs = s.probabilities(arms)
    assert min(probs.values()) >= 0.3 / 3 - 1e-12


def test_probabilities_restrict_to_the_offered_subset():
    s, arms = _with_arms(5)
    probs = s.probabilities(arms[:2])
    assert set(probs) == set(arms[:2])
    assert sum(probs.values()) == pytest.approx(1.0)


def test_learning_rate_anneals_as_inverse_sqrt():
    s, arms = _with_arms(2, eta=3.0)
    assert s.learning_rate() == pytest.approx(3.0)
    for _ in range(8):
        s.record(s.select_op(arms), False)
    assert s.learning_rate() == pytest.approx(3.0 / math.sqrt(9))


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


def test_record_is_on_policy():
    s, arms = _with_arms(3)
    s.record(arms[0], True)
    assert s.bandit_stats()["orphan_records"] == 1
    assert np.all(s._lossv == 0.0)


def test_a_second_reward_for_the_same_draw_is_dropped():
    s, arms = _with_arms(2)
    op = s.select_op(arms)
    s.record(op, True)
    s.record(op, True)
    assert s.bandit_stats()["orphan_records"] == 1


def test_unconsumed_draws_do_not_grow_without_bound():
    s, _ = _with_arms(0)
    for i in range(1000):
        s.select_op([f"a{i}", f"b{i}"])
    assert len(s._pending) <= 256
    assert s.bandit_stats()["expired_draws"] > 0


def test_estimate_is_the_shifted_importance_weight():
    """eta = 0.1 keeps p >= eta^2, so the shift is 1/2: a failure at
    p = 1/2 adds (1 - 1/2) / (1/2) = 1, then a success at the re-solved
    p adds (0 - 1/2) / p."""
    eta = 0.1
    s = TsallisINFScheduler(eta=eta, rng=ScriptedRng(randoms=[0.1, 0.1]))
    ops = ["a", "b"]
    s.record(s.select_op(ops), False)
    assert s._lossv[s._idx["a"]] == pytest.approx(1.0)

    p2 = tsallis_probs(np.array([1.0, 0.0]), eta / math.sqrt(2))[0]
    s.record(s.select_op(ops), True)
    assert s._lossv[s._idx["a"]] == pytest.approx(1.0 - 0.5 / p2)


def test_shift_drops_while_p_is_below_eta_squared():
    """Round 1 at eta = 2: p = 1/2 < eta^2 = 4, so no shift: a failure adds 1/p."""
    s = TsallisINFScheduler(eta=2.0, rng=ScriptedRng(randoms=[0.1]))
    s.record(s.select_op(["a", "b"]), False)
    assert s._lossv[s._idx["a"]] == pytest.approx(1.0 / 0.5)


def test_centred_loss_stays_uniform():
    """Falsification: reward 1/2 is loss 1/2, exactly the shift, so every
    estimate is zero and no arm may be preferred."""
    s, arms = _with_arms(6, eta=0.1)
    for _ in range(500):
        s.record(s.select_op(arms), True, weight=0.5)
    probs = s.probabilities(arms)
    assert all(p == pytest.approx(1.0 / 6, rel=1e-12) for p in probs.values())


def test_equal_operators_are_all_reached():
    """Adversarial for lock-in: 252 equal 5% arms over 20k rounds. The
    running-mean baseline this replaced left 94 never drawn."""
    s, arms = _with_arms(252)
    env = random.Random(1234)
    seen = set()
    for _ in range(20_000):
        op = s.select_op(arms)
        seen.add(op)
        s.record(op, env.random() < 0.05)
    assert seen == set(arms)


def test_a_productive_operator_gains_probability():
    s, arms = _with_arms(12)
    env = random.Random(7)
    for _ in range(3000):
        op = s.select_op(arms)
        rate = 0.3 if op == arms[5] else 0.03
        s.record(op, env.random() < rate)
    probs = s.probabilities(arms)
    assert max(probs, key=probs.get) == arms[5]
    assert probs[arms[5]] > 0.5


def test_recovers_when_the_best_arm_switches():
    """Adversarial: arm a pays for 3000 rounds then dies; b pays after.
    The last 1000 rounds must favour b."""
    s, _ = _with_arms(0)
    ops = ["a", "b", "c", "d"]
    env = random.Random(11)
    late_b = 0
    for t in range(6000):
        op = s.select_op(ops)
        best = "a" if t < 3000 else "b"
        s.record(op, env.random() < (0.3 if op == best else 0.02))
        late_b += t >= 5000 and op == "b"
    assert late_b > 500


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e300])
def test_adversarial_weights_are_clamped(weight):
    s, arms = _with_arms(3)
    for _ in range(50):
        s.record(s.select_op(arms), True, weight=weight)
    assert np.all(np.isfinite(s._lossv))
    assert sum(s.probabilities(arms).values()) == pytest.approx(1.0)


def test_failure_is_zero_reward_whatever_the_weight():
    a, arms = _with_arms(2, rng=RandPool(seed=5))
    b, _ = _with_arms(2, rng=RandPool(seed=5))
    for _ in range(20):
        a.record(a.select_op(arms), False, weight=0.9)
        b.record(b.select_op(arms), False)
    assert np.array_equal(a._lossv, b._lossv)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_bandit_stats_is_safe_before_any_round():
    s, _ = _with_arms(0)
    stats = s.bandit_stats()
    assert stats["rounds"] == 0 and stats["entropy"] == 0.0


def test_bandit_stats_counts_pulls_and_wins():
    s, arms = _with_arms(2)
    for _ in range(10):
        s.record(s.select_op(arms), True, weight=0.5)
    stats = s.bandit_stats()
    assert sum(stats["pulls"].values()) == 10
    assert sum(stats["wins"].values()) == pytest.approx(5.0)
    assert 0.0 <= stats["entropy"] <= 1.0
