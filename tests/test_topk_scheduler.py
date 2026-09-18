"""Top-K operator scheduler behavior."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.schedulers.op_topk import TopKScheduler
from tests.support.scripted_rng import ScriptedRng


def test_initialises_arms_idempotently():
    scheduler = TopKScheduler()

    scheduler.init_arm("bit_flip")
    scheduler.init_arm("bit_flip")
    scheduler.init_arm("byte_flip")

    assert scheduler.bandit_stats() == {"bit_flip": (0.0, 0), "byte_flip": (0.0, 0)}


def test_record_updates_weighted_mean_and_count():
    scheduler = TopKScheduler()

    scheduler.record("bit_flip", True, weight=0.25)
    scheduler.record("bit_flip", True, weight=0.75)
    scheduler.record("byte_flip", False, weight=1.0)

    assert scheduler.bandit_stats()["bit_flip"] == pytest.approx((0.5, 2))
    assert scheduler.bandit_stats()["byte_flip"] == pytest.approx((0.0, 1))


def test_select_op_chooses_uniformly_from_top_k():
    scheduler = TopKScheduler(k=2, rng=ScriptedRng(choice_idxs=[1]))
    scheduler.record("best", True)
    scheduler.record("second", True, weight=0.5)
    scheduler.record("worst", False)

    assert scheduler.select_op(["best", "second", "worst"]) == "second"


def test_top_k_ties_keep_candidate_order_before_random_choice():
    scheduler = TopKScheduler(k=2, rng=ScriptedRng(choice_idxs=[1]))
    scheduler.init_arm("first")
    scheduler.init_arm("second")
    scheduler.init_arm("third")

    assert scheduler.select_op(["first", "second", "third"]) == "second"


def test_select_op_handles_empty_and_single_candidate_lists():
    scheduler = TopKScheduler()

    assert scheduler.select_op([]) == ""
    assert scheduler.select_op(["only"]) == "only"


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        TopKScheduler(k=0)


def test_bandit_stats_reports_registered_arms():
    scheduler = TopKScheduler()
    scheduler.record("a", True, weight=0.5)
    scheduler.record("b", False)

    assert scheduler.bandit_stats() == {"a": pytest.approx((0.5, 1)), "b": pytest.approx((0.0, 1))}
