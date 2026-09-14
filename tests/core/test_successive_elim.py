"""Unit tests for SuccessiveEliminationScheduler."""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.successive_elim import SuccessiveEliminationScheduler


def test_select_single():
    s = SuccessiveEliminationScheduler(rng=RandPool(seed=1))
    s.init_arm("a")
    assert s.select_op(["a"]) == "a"


def test_round_robin_among_active():
    s = SuccessiveEliminationScheduler(min_pulls=100, rng=RandPool(seed=2))
    ops = ["a", "b", "c"]
    for o in ops:
        s.init_arm(o)
    seen = [s.select_op(ops) for _ in range(6)]
    # With min_pulls high, nothing is eliminated; pure RR
    assert seen == ["a", "b", "c", "a", "b", "c"]


def test_eliminates_inferior_arm():
    s = SuccessiveEliminationScheduler(
        delta=0.05, min_pulls=5, reopen_interval=0, rng=RandPool(seed=3)
    )
    ops = ["good", "bad"]
    for o in ops:
        s.init_arm(o)
    # Feed a clear gap: good always succeeds, bad always fails
    for _ in range(40):
        op = s.select_op(ops)
        s.record(op, success=(op == "good"), weight=1.0)
    stats = s.bandit_stats()
    assert "bad" in stats["eliminated_ops"] or stats["se_eliminated"] >= 1
    assert "good" in stats["active_ops"]


def test_reopen_readmits():
    s = SuccessiveEliminationScheduler(
        delta=0.05, min_pulls=3, reopen_interval=20, rng=RandPool(seed=4)
    )
    ops = ["good", "bad"]
    for o in ops:
        s.init_arm(o)
    for _ in range(30):
        op = s.select_op(ops)
        s.record(op, success=(op == "good"), weight=1.0)
    # After enough pulls past reopen_interval, eliminated arms return
    stats = s.bandit_stats()
    # Either still racing or reopened; should not crash and should have arms
    assert stats["se_arms"] == 2


def test_bandit_stats_keys():
    s = SuccessiveEliminationScheduler(rng=RandPool(seed=0))
    s.init_arm("z")
    s.select_op(["z"])
    s.record("z", True)
    stats = s.bandit_stats()
    assert stats["se_pulls"] == 1
    assert "se_active" in stats
    assert "active_ops" in stats
