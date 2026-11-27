"""Regression tests for PairCoverScheduler (2-way op-sequence coverage)."""

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_pair_cover import PairCoverScheduler

OPS = ["a", "b", "c", "d", "e"]


class ScriptedRng:
    """Deterministic RNG: first candidate, posterior mean (Hard Rule 39)."""

    def random(self):
        return 0.0

    def choice(self, seq):
        return seq[0]

    def betavariate(self, alpha, beta):
        return alpha / (alpha + beta)


def _drive(sched, prev, ops):
    """Select from row *prev*, record a failure, return the pick."""
    sched._prev = prev
    pick = sched.select_op(ops)
    sched.record(pick, False)
    return pick


def test_init_contract():
    sched = PairCoverScheduler(rng=RandPool(1))
    assert sched.supports_priors is False
    assert callable(sched.select_op)
    assert callable(sched.record)
    assert callable(sched.bandit_stats)


def test_init_rejects_bad_params():
    with pytest.raises(ValueError):
        PairCoverScheduler(rng=RandPool(1), cover_target=0)
    with pytest.raises(ValueError):
        PairCoverScheduler(rng=RandPool(1), prior_strength=0.0)


def test_empty_and_single():
    sched = PairCoverScheduler(rng=RandPool(1))
    assert sched.select_op([]) == ""
    assert sched.select_op(["only"]) == "only"


@pytest.mark.parametrize("seed", range(8))
def test_row_sweep_covers_every_pair(seed):
    """K visits to one row cover all K pairs, whatever the RNG."""
    sched = PairCoverScheduler(rng=RandPool(seed))
    picks = {_drive(sched, "a", OPS) for _ in OPS}
    assert picks == set(OPS)


def test_cover_target_two_needs_two_sweeps():
    sched = PairCoverScheduler(cover_target=2, rng=RandPool(3))
    picks = [_drive(sched, "a", OPS) for _ in range(2 * len(OPS))]
    assert all(picks.count(op) == 2 for op in OPS)


def test_exploit_follows_pair_not_marginal():
    """Falsification: b's marginal is low, yet row a picks b via pair (a, b)."""
    sched = PairCoverScheduler(cover_target=1, rng=ScriptedRng())
    ops = ["a", "b", "c"]
    for op in ops:
        sched.init_arm(op)
    # (prev, next, success, repeats): a->b wins once; c->b fails five times,
    # dragging b's marginal to 1/4 while pair (a, b) stays at 1/1.
    rows = [
        ("a", "a", False, 1),
        ("a", "b", True, 1),
        ("a", "c", False, 1),
        ("c", "a", False, 1),
        ("c", "b", False, 5),
        ("c", "c", False, 1),
    ]
    for prev, nxt, ok, reps in rows:
        for _ in range(reps):
            sched._sel_prev = prev
            sched.record(nxt, ok)
    sched._prev = "a"
    assert sched.select_op(ops) == "b"
    sched._prev = "c"
    assert sched.select_op(ops) != "b"


def test_control_identical_runs_match():
    """Control (Hard Rule 46): same seed, same script, same picks."""

    def run():
        sched = PairCoverScheduler(rng=RandPool(7))
        out = []
        for i in range(60):
            pick = sched.select_op(OPS)
            sched.record(pick, i % 3 == 0)
            out.append(pick)
        return out

    assert run() == run()


def test_adversarial_duplicate_and_unseen_ops():
    sched = PairCoverScheduler(rng=RandPool(5))
    for i in range(40):
        ops = ["a", "a", "b"] if i % 2 else ["b", "z", "", "z"]
        pick = sched.select_op(ops)
        assert pick in ops
        sched.record(pick, True)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -5.0, 1e308])
def test_adversarial_record_weight(weight):
    sched = PairCoverScheduler(rng=RandPool(5))
    for _ in range(10):
        pick = sched.select_op(OPS)
        sched.record(pick, True, weight=weight)
    for _ in range(10):
        assert sched.select_op(OPS) in OPS
    stats = sched.bandit_stats()
    assert all(math.isfinite(v) for v in stats["pc_success"].values())


def test_adversarial_record_without_select():
    sched = PairCoverScheduler(rng=RandPool(5))
    sched.record("never_selected", True)
    assert sched.bandit_stats()["pc_pulls"] == 1


def test_bandit_stats_coverage():
    sched = PairCoverScheduler(rng=RandPool(9))
    for _ in OPS:
        _drive(sched, "a", OPS)
    stats = sched.bandit_stats()
    assert stats["pc_pulls"] == len(OPS)
    assert stats["pc_pairs_covered"] == len(OPS)
    assert 0.0 < stats["pc_coverage"] <= 1.0


def test_memory_bounded_by_arm_count():
    """Pair table never exceeds K^2 arms ever registered."""
    sched = PairCoverScheduler(rng=RandPool(2))
    for _ in range(500):
        pick = sched.select_op(OPS)
        sched.record(pick, False)
    assert len(sched._pair_n) <= len(OPS) ** 2
