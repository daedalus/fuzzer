"""Tests for ``core/schedulers/op_kalman_ts.py`` -- Thompson over Kalman-tracked arms.

Read first:

- ``test_update_matches_the_written_out_filter`` -- predict/update/drift
  adaptation against a scalar transcription of the equations.
- ``test_without_drift_unobserved_variance_is_frozen`` (falsification) --
  q = 0 must reduce to a static posterior; any growth is a leak.
- ``test_abandons_a_dead_arm_faster_than_a_beta_posterior`` (adversarial) --
  the reason this exists: a 200-success arm that dies is dropped in far
  fewer failures than Beta(201, 1) needs to cross 0.5.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_kalman_ts import (
    Q_MAX,
    Q_MIN,
    VAR_MAX,
    KalmanTSScheduler,
)
from tests.support.scripted_rng import ScriptedRng


def _sched(**kw) -> KalmanTSScheduler:
    kw.setdefault("rng", RandPool(seed=99))
    return KalmanTSScheduler(**kw)


# --------------------------------------------------------------------------
# Contract (Hard Rules 1 and 40)
# --------------------------------------------------------------------------


def test_declares_prior_support():
    assert KalmanTSScheduler.supports_priors is True


def test_has_the_operator_scheduler_interface():
    s = _sched()
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize(
    "kwargs", [{"q0": -1e-4}, {"drift_rate": -0.1}, {"obs_floor": 0.0}, {"obs_floor": 0.3}]
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        KalmanTSScheduler(**kwargs)


def test_prior_maps_to_beta_moments():
    s = _sched()
    a, b = 3.0, 7.0
    s.init_arm("x", a, b)
    mean, var = s.posterior("x")
    assert mean == pytest.approx(a / (a + b))
    assert var == pytest.approx(a * b / ((a + b) ** 2 * (a + b + 1.0)))


def test_init_arm_is_idempotent():
    s = _sched()
    s.init_arm("x", 9.0, 1.0)
    s.init_arm("x", 1.0, 9.0)
    assert s.posterior("x")[0] == pytest.approx(0.9)


def test_empty_candidate_list_is_not_a_crash():
    assert _sched().select_op([]) == ""


def test_select_op_registers_unseen_operators():
    s = _sched()
    assert s.select_op(["x", "y"]) in ("x", "y")
    assert s.bandit_stats()["kalman_ts_arms"] == 2


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------


def _reference(m, p, q, ys, dts, drift_rate, obs_floor):
    """Local-level Kalman filter with whiteness-matched drift, one arm, scalar."""
    prev = 0.0
    for y, dt in zip(ys, dts, strict=True):
        p = min(p + q * dt, VAR_MAX)
        s = p + max(m * (1.0 - m), obs_floor)
        innov = y - m
        gain = p / s
        m += gain * innov
        p *= 1.0 - gain
        e = innov / math.sqrt(s)
        q = min(max(q * math.exp(drift_rate * e * prev), Q_MIN), Q_MAX)
        prev = e
    return m, p, q


def test_update_matches_the_written_out_filter():
    """Arm a recorded on every third round; b fills the gaps, so a's
    elapsed time between updates is 3 and the drift term is exercised."""
    s = _sched(q0=1e-3, drift_rate=0.1, obs_floor=0.02)
    s.init_arm("a")
    s.init_arm("b")
    rnd = random.Random(3)
    ys = []
    for i in range(60):
        name = "a" if i % 3 == 2 else "b"
        y = rnd.random()
        s.record(name, True, weight=y)
        if name == "a":
            ys.append(y)
    m, p, q = _reference(0.5, 1.0 / 12.0, 1e-3, ys, [3] * len(ys), 0.1, 0.02)
    j = s._idx["a"]
    assert s._meanv[j] == pytest.approx(m, rel=1e-12)
    assert s._varv[j] == pytest.approx(p, rel=1e-12)
    assert s._qv[j] == pytest.approx(q, rel=1e-12)


def test_unobserved_variance_grows_with_elapsed_rounds():
    s = _sched(q0=1e-3)
    s.init_arm("a")
    s.init_arm("b")
    v0 = s.posterior("a")[1]
    for _ in range(10):
        s.record("b", False)
    assert s.posterior("a")[1] == pytest.approx(v0 + 10 * 1e-3)


def test_without_drift_unobserved_variance_is_frozen():
    """Falsification: q0 = 0 and no adaptation is a static model."""
    s = _sched(q0=0.0, drift_rate=0.0)
    s.init_arm("a")
    s.init_arm("b")
    v0 = s.posterior("a")[1]
    for _ in range(1000):
        s.record("b", True)
    assert s.posterior("a")[1] == v0


def test_variance_is_capped_after_an_enormous_gap():
    """Adversarial: a 1e12-round gap must not make the draw scale explode."""
    s = _sched(q0=1e-2)
    s.init_arm("a")
    s._t = 10**12
    assert s.posterior("a")[1] == VAR_MAX


def test_consistent_noise_shrinks_drift():
    """I.i.d. fair-coin outcomes: an over-reactive filter's innovations
    alternate in sign, so q drifts down. log q random-walks with step
    ~drift_rate, so the second-half geometric mean is asserted, not a
    snapshot."""
    q0, rounds = 1e-3, 20_000
    s = _sched(q0=q0)
    s.init_arm("a")
    coin = random.Random(1)
    log_q = 0.0
    for t in range(rounds):
        s.record("a", coin.random() < 0.5)
        log_q += math.log(s._qv[s._idx["a"]]) if t >= rounds // 2 else 0.0
    assert log_q / (rounds // 2) < math.log(q0 / 10)


def test_abrupt_change_raises_drift():
    s = _sched(q0=1e-3)
    s.init_arm("a")
    for _ in range(300):
        s.record("a", True)
    q_steady = s._qv[s._idx["a"]]
    for _ in range(10):
        s.record("a", False)
    assert s._qv[s._idx["a"]] > q_steady


def test_abandons_a_dead_arm_faster_than_a_beta_posterior():
    """200 successes, then failures. Beta(201, 1) needs 200 failures to
    reach mean 0.5; the filter must get there in under a quarter of that."""
    s = _sched()
    s.init_arm("a")
    for _ in range(200):
        s.record("a", True)
    beta_failures = 200  # 201 / (202 + n) < 0.5  <=>  n > 200
    # Deterministic (no RNG): bounded so a broken filter fails, not hangs.
    n = 0
    while s.posterior("a")[0] >= 0.5 and n < beta_failures:
        s.record("a", False)
        n += 1
    assert n < beta_failures / 4


def test_frozen_drift_does_not_abandon_a_dead_arm():
    """Contrast for the test above: with drift adaptation off, the same
    collapse is not tracked within the Beta posterior's 200 failures --
    so the adaptation, not the filter alone, is the mechanism."""
    s = _sched(drift_rate=0.0)
    s.init_arm("a")
    for _ in range(200):
        s.record("a", True)
    for _ in range(200):
        s.record("a", False)
    assert s.posterior("a")[0] >= 0.5


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e300])
def test_adversarial_weights_are_clamped(weight):
    s = _sched()
    for _ in range(50):
        s.record("a", True, weight=weight)
    mean, var = s.posterior("a")
    assert 0.0 <= mean <= 1.0 and math.isfinite(var)
    assert Q_MIN <= s._qv[s._idx["a"]] <= Q_MAX


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_scripted_draw_takes_the_argmax_sample():
    rng = ScriptedRng(gauss_lists=[[2.0, -1.0, 0.5]])
    s = KalmanTSScheduler(rng=rng)
    ops = ["a", "b", "c"]
    s.init_arm("a", 1.0, 9.0)
    s.init_arm("b", 9.0, 1.0)
    s.init_arm("c", 5.0, 5.0)
    draws = []
    for op, z in zip(ops, [2.0, -1.0, 0.5], strict=True):
        m, v = s.posterior(op)
        draws.append(m + math.sqrt(v) * z)
    assert s.select_op(ops) == ops[int(np.argmax(draws))]


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


def test_bandit_stats_reports_pulls_and_drift():
    s = _sched()
    s.record("a", True)
    s.record("b", False)
    stats = s.bandit_stats()
    assert stats["kalman_ts_pulls"] == 2
    assert stats["kalman_ts_arms"] == 2
    assert Q_MIN <= stats["kalman_ts_mean_q"] <= Q_MAX
