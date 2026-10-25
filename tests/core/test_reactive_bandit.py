"""Unit tests for ReactiveGradientScheduler.

Oracle: a plain GradientBanditScheduler driven through the same draws. Each
reactive add-on (success-clocked decay, spike guard) is checked as an exact
transform of the oracle's input, so the tests derive expected values without
re-implementing the update rule.
"""

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_gradient import GradientBanditScheduler
from fuzzer_tool.core.schedulers.op_reactive import ReactiveGradientScheduler

OPS = ["a", "b", "c"]
SEED = 5


def _pair(**kw):
    """Reactive scheduler and plain oracle on identical RNG streams."""
    r = ReactiveGradientScheduler(rng=RandPool(SEED), **kw)
    g = GradientBanditScheduler(rng=RandPool(SEED))
    for op in OPS:
        r.init_arm(op)
        g.init_arm(op)
    return r, g


def _step(s, success, weight=1.0):
    op = s.select_op(OPS)
    s.record(op, success, weight)
    return op


def test_supports_priors_declared():
    assert ReactiveGradientScheduler.supports_priors is False


def test_failure_matches_plain_gradient():
    """Failures carry no information to forget on: update equals the oracle's."""
    r, g = _pair()
    _step(r, True)
    _step(g, True)

    assert _step(r, False) == _step(g, False)
    assert r.preferences == pytest.approx(g.preferences)


def test_success_decays_then_steps():
    """A success shrinks every preference by (1 - forget) before the gradient."""
    r, g = _pair(forget=0.1)
    r.preferences = {"a": 2.0, "b": -1.0, "c": 0.5}
    g.preferences = dict(r.preferences)

    op = r.select_op(OPS)
    assert g.select_op(OPS) == op
    r.record(op, True)
    g.preferences = {k: h * 0.9 for k, h in g.preferences.items()}
    g.record(op, True)
    assert r.preferences == pytest.approx(g.preferences)


def test_zero_knobs_equal_plain_gradient():
    """Falsification: with both add-ons off it is exactly the static scheduler."""
    r, g = _pair(forget=0.0, spike=0.0)
    for i in range(200):
        success = i % 3 == 0
        weight = 50.0 if i == 99 else 1.0
        assert _step(r, success, weight) == _step(g, success, weight)
    assert r.preferences == g.preferences


def test_spike_clipped_to_ema_multiple():
    """A reward far above the success EMA is cut to spike * EMA."""
    r, g = _pair(forget=0.0, spike=3.0)
    for _ in range(5):
        _step(r, True, 1.0)
        _step(g, True, 1.0)

    _step(r, True, 1000.0)
    _step(g, True, 3.0)
    assert r.preferences == pytest.approx(g.preferences)
    assert r.bandit_stats()["reactive_spike_cuts"] == 1


def test_spike_does_not_inflate_ema():
    """Adversarial: repeated spikes stay clipped; the EMA tracks clipped values."""
    r, _ = _pair(forget=0.0, spike=3.0)
    _step(r, True, 1.0)
    for _ in range(50):
        _step(r, True, 1e6)
    cap = r.bandit_stats()["reactive_reward_ema"] * 3.0
    assert cap < 1e6
    assert r.bandit_stats()["reactive_spike_cuts"] == 50


def test_one_huge_reward_does_not_lock_in():
    """Adversarial: a single 1e9-weight fluke on a dead arm must not capture the policy."""
    r = ReactiveGradientScheduler(rng=RandPool(SEED))
    for op in OPS:
        r.init_arm(op)
    for _ in range(300):
        op = r.select_op(OPS)
        r.record(op, op == "a")
    r.select_op(OPS)
    r._last_op = "c"
    r.record("c", True, 1e9)

    assert r.preferences["c"] < r.preferences["a"]


def test_shadow_record_ignored():
    r, _ = _pair()
    r.select_op(OPS)
    before = dict(r.preferences)
    r.record("x", True, 1.0)
    assert r.preferences == before


@pytest.mark.parametrize("kw", [{"forget": -0.1}, {"forget": 1.0}, {"spike": -1.0}])
def test_rejects_bad_knobs(kw):
    with pytest.raises(ValueError):
        ReactiveGradientScheduler(**kw)
