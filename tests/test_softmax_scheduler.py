"""Softmax operator scheduler behavior."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.schedulers.op_softmax import SoftmaxScheduler
from tests.support.scripted_rng import ScriptedRng


def test_initialises_arms_idempotently():
    scheduler = SoftmaxScheduler()

    scheduler.init_arm("bit_flip")
    scheduler.init_arm("bit_flip")
    scheduler.init_arm("byte_flip")

    assert scheduler.bandit_stats() == {"bit_flip": (0.0, 0), "byte_flip": (0.0, 0)}


def test_record_updates_weighted_mean_and_count():
    scheduler = SoftmaxScheduler()

    scheduler.record("bit_flip", True, weight=0.25)
    scheduler.record("bit_flip", True, weight=0.75)
    scheduler.record("byte_flip", False, weight=1.0)

    assert scheduler.bandit_stats()["bit_flip"] == pytest.approx((0.5, 2))
    assert scheduler.bandit_stats()["byte_flip"] == pytest.approx((0.0, 1))


def test_select_op_samples_the_softmax_distribution():
    scheduler = SoftmaxScheduler(tau=1.0, rng=ScriptedRng(randoms=[0.8]))
    scheduler.record("good", True)
    scheduler.record("bad", False)

    good_probability = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    assert scheduler.select_op(["good", "bad"]) == "bad"
    assert good_probability == pytest.approx(0.7310585786, rel=1e-9)


def test_select_op_uses_first_arm_for_equal_means():
    scheduler = SoftmaxScheduler(rng=ScriptedRng(randoms=[0.999]))
    scheduler.init_arm("first")
    scheduler.init_arm("second")

    assert scheduler.select_op(["first", "second"]) == "first"


def test_select_op_handles_empty_and_single_candidate_lists():
    scheduler = SoftmaxScheduler()

    assert scheduler.select_op([]) == ""
    assert scheduler.select_op(["only"]) == "only"


def test_select_op_is_numerically_stable():
    scheduler = SoftmaxScheduler(rng=ScriptedRng(randoms=[0.5]))
    scheduler.record("large", True, weight=1000.0)
    scheduler.record("small", False)

    assert scheduler.select_op(["large", "small"]) == "large"


def test_tau_must_be_positive():
    with pytest.raises(ValueError, match="tau must be positive"):
        SoftmaxScheduler(tau=0.0)


def test_bandit_stats_reports_registered_arms():
    scheduler = SoftmaxScheduler()
    scheduler.record("a", True, weight=0.5)
    scheduler.record("b", False)

    assert scheduler.bandit_stats() == {"a": pytest.approx((0.5, 1)), "b": pytest.approx((0.0, 1))}
