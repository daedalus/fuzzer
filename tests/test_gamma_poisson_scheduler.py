"""Tests for ``core/schedulers/op_gamma_poisson.py`` -- Thompson over Gamma-Poisson rates.

Read first:

- ``test_update_matches_the_written_out_model`` -- discount/update against a
  scalar transcription of the equations.
- ``test_without_discount_posterior_is_conjugate`` (falsification) --
  discount = 1 must reduce to exact conjugate counts; any drift is a leak.
- ``test_underflowing_gap_floors_the_shape`` (adversarial) -- a gap where
  ``discount ** dt`` underflows to 0 must keep a finite, mean-preserving
  posterior instead of a zero-shape Gamma.
- ``test_rare_yield_beats_gaussian`` (``test_scheduler_convergence.py``) --
  the reason this exists.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_gamma_poisson import GammaPoissonScheduler
from tests.support.scripted_rng import ScriptedRng


def _sched(**kw) -> GammaPoissonScheduler:
    kw.setdefault("rng", RandPool(seed=99))
    return GammaPoissonScheduler(**kw)


# --------------------------------------------------------------------------
# Contract (Hard Rules 1 and 40)
# --------------------------------------------------------------------------


def test_declares_prior_support():
    assert GammaPoissonScheduler.supports_priors is True


def test_has_the_operator_scheduler_interface():
    s = _sched()
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize(
    "kwargs",
    [{"discount": 0.0}, {"discount": -0.1}, {"discount": 1.5}, {"shape_min": 0.0}],
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        GammaPoissonScheduler(**kwargs)


def test_prior_maps_to_matching_mean():
    """Beta(a, b) -> Gamma(a, a + b): same mean, a + b pseudo-pulls."""
    s = _sched()
    a, b = 3.0, 7.0
    s.init_arm("x", a, b)
    shape, rate = s.posterior("x")
    assert (shape, rate) == (a, a + b)
    assert shape / rate == pytest.approx(a / (a + b))


def test_init_arm_is_idempotent():
    s = _sched()
    s.init_arm("x", 9.0, 1.0)
    s.init_arm("x", 1.0, 9.0)
    assert s.posterior("x") == (9.0, 10.0)


def test_empty_candidate_list_is_not_a_crash():
    assert _sched().select_op([]) == ""


def test_select_op_registers_unseen_operators():
    s = _sched()
    assert s.select_op(["x", "y"]) in ("x", "y")
    assert s.bandit_stats()["gamma_poisson_arms"] == 2


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def _reference(shape, rate, ys, dts, discount, shape_min):
    """Discounted Gamma-Poisson update, one arm, scalar."""
    for y, dt in zip(ys, dts, strict=True):
        f = max(discount**dt, min(1.0, shape_min / shape))
        shape, rate = shape * f + y, rate * f + 1.0
    return shape, rate


def test_update_matches_the_written_out_model():
    """Arm a recorded on every third round; b fills the gaps, so a's
    elapsed time between updates is 3 and the discount is exercised."""
    s = _sched(discount=0.99, shape_min=1.0)
    s.init_arm("a", 4.0, 4.0)
    s.init_arm("b")
    rnd = random.Random(3)
    ys = []
    for i in range(60):
        name = "a" if i % 3 == 2 else "b"
        y = rnd.random()
        s.record(name, True, weight=y)
        if name == "a":
            ys.append(y)
    shape, rate = _reference(4.0, 8.0, ys, [3] * len(ys), 0.99, 1.0)
    got = s.posterior("a")
    assert got[0] == pytest.approx(shape, rel=1e-12)
    assert got[1] == pytest.approx(rate, rel=1e-12)


def test_without_discount_posterior_is_conjugate():
    """Falsification: discount = 1 is the static Gamma(1 + sum y, 2 + n)."""
    s = _sched(discount=1.0)
    s.init_arm("a")
    s.init_arm("idle")
    ys = [1.0 if i % 7 == 0 else 0.0 for i in range(700)]
    for y in ys:
        s.record("a", y > 0.0)
    assert s.posterior("a") == (1.0 + sum(ys), 2.0 + len(ys))
    assert s.posterior("idle") == (1.0, 2.0)


def test_discount_preserves_the_mean_and_shrinks_the_shape():
    """An unobserved arm loses evidence (shape) but keeps its rate estimate."""
    g = 0.95
    s = _sched(discount=g)
    s.init_arm("a", 50.0, 50.0)
    s.init_arm("b")
    for _ in range(10):
        s.record("b", False)
    shape, rate = s.posterior("a")
    assert shape == pytest.approx(50.0 * g**10)
    assert rate == pytest.approx(100.0 * g**10)
    assert shape / rate == pytest.approx(0.5)


def test_underflowing_gap_floors_the_shape():
    """Adversarial: 0.5 ** 2000 underflows to 0.0; the posterior stays finite."""
    s = _sched(discount=0.5, shape_min=1.0)
    s.init_arm("a", 50.0, 50.0)
    s.init_arm("b")
    for _ in range(2000):
        s.record("b", False)
    shape, rate = s.posterior("a")
    assert math.isfinite(shape) and math.isfinite(rate)
    assert shape == pytest.approx(1.0)
    assert shape / rate == pytest.approx(0.5)


def test_prior_below_the_floor_is_not_inflated():
    """A shape already under shape_min is left alone, not raised to it."""
    s = _sched(discount=0.5, shape_min=1.0)
    s.init_arm("a", 0.5, 9.5)
    s.init_arm("b")
    for _ in range(50):
        s.record("b", False)
    assert s.posterior("a") == (0.5, 10.0)


@pytest.mark.parametrize(
    ("success", "weight", "added"),
    [
        (True, 1.0, 1.0),
        (True, 0.25, 0.25),
        (True, math.inf, 1.0),
        (True, math.nan, 0.0),
        (True, -3.0, 0.0),
        (False, 1.0, 0.0),
    ],
)
def test_reward_is_clamped_to_unit(success, weight, added):
    """Adversarial weights must not push the shape negative or unbounded."""
    s = _sched(discount=1.0)
    s.init_arm("a")
    s.record("a", success, weight)
    assert s.posterior("a") == (1.0 + added, 3.0)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


class _SpyRng(ScriptedRng):
    """ScriptedRng that also records the Gamma parameters it was asked for."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.params: list[tuple[np.ndarray, np.ndarray]] = []

    def gammavariate_array(self, alphas, betas):
        self.params.append((np.array(alphas), np.array(betas)))
        return super().gammavariate_array(alphas, betas)


def test_select_is_argmax_of_the_draw():
    rng = ScriptedRng(gamma_arrays=[np.array([0.1, 0.7, 0.3])])
    s = GammaPoissonScheduler(rng=rng)
    assert s.select_op(["x", "y", "z"]) == "y"


def test_select_draws_from_the_discounted_posterior():
    """The draw must use the lazily discounted parameters, not the stored ones."""
    g = 0.9
    rng = _SpyRng(gamma_arrays=[np.array([0.0, 1.0])])
    s = GammaPoissonScheduler(discount=g, rng=rng)
    s.init_arm("a", 40.0, 60.0)
    s.init_arm("b", 40.0, 60.0)
    for _ in range(5):
        s.record("b", False)
    s.select_op(["a", "b"])
    shapes, rates = rng.params[0]
    assert shapes[0] == pytest.approx(40.0 * g**5)
    assert rates[0] == pytest.approx(100.0 * g**5)
    assert (shapes[1], rates[1]) == pytest.approx(s.posterior("b"))


def test_bandit_stats_reports_the_best_mean():
    s = _sched(discount=1.0)
    s.init_arm("a", 1.0, 1.0)
    s.init_arm("b", 9.0, 1.0)
    stats = s.bandit_stats()
    assert stats["gamma_poisson_max_mean"] == pytest.approx(0.9)
    assert stats["gamma_poisson_pulls"] == 0
