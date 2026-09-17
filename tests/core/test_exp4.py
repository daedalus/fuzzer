"""Unit tests for Exp4Scheduler."""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_exp4 import Exp4Scheduler


def test_select_single():
    s = Exp4Scheduler(rng=RandPool(seed=1))
    s.init_arm("a")
    assert s.select_op(["a"]) == "a"


def test_select_runs():
    s = Exp4Scheduler(gamma=0.2, rng=RandPool(seed=2))
    ops = ["bit_flip", "byte_insert", "crossover"]
    for o in ops:
        s.init_arm(o)
    chosen = {s.select_op(ops) for _ in range(30)}
    assert chosen & set(ops)


def test_learns_category():
    s = Exp4Scheduler(gamma=0.15, rng=RandPool(seed=7))
    ops = ["bit_flip", "bit_rotate", "byte_insert", "crossover"]
    for o in ops:
        s.init_arm(o)
    for _ in range(80):
        op = s.select_op(ops)
        # reward only bit_* ops
        s.record(op, success=op.startswith("bit"), weight=1.0)
    stats = s.bandit_stats()
    assert stats["exp4_pulls"] == 80
    # bit expert should dominate or at least beat uniform baseline trend
    assert s.weights.get("bit", 0) >= s.weights.get("byte", 0)


def test_shadow_record_ignored():
    s = Exp4Scheduler(rng=RandPool(seed=3))
    ops = ["a", "b"]
    for o in ops:
        s.init_arm(o)
    s.select_op(ops)
    before = dict(s.weights)
    s.record("not_selected", success=True, weight=1.0)
    assert s.weights == before


def test_bandit_stats_keys():
    s = Exp4Scheduler(rng=RandPool(seed=0))
    s.init_arm("z")
    s.select_op(["z"])
    s.record("z", True)
    stats = s.bandit_stats()
    assert "exp4_pulls" in stats
    assert "exp4_experts" in stats
