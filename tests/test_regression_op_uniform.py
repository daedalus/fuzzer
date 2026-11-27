"""Regression tests for the op_uniform scheduler unit behavior."""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_uniform import UniformScheduler


class TestUniformScheduler:
    def test_uniform_scheduler_exists_and_init(self):
        scheduler = UniformScheduler()
        assert hasattr(scheduler, "select_op")
        assert hasattr(scheduler, "record")
        assert hasattr(scheduler, "bandit_stats")
        assert hasattr(scheduler, "init_arm")
        assert hasattr(scheduler, "supports_priors")
        assert scheduler.supports_priors is False

    def test_uniform_select_op_uniform_random(self):
        scheduler = UniformScheduler()
        scheduler.init_arm("op_a")
        scheduler.init_arm("op_b")
        scheduler.init_arm("op_c")

        scheduler._rng = RandPool(seed=42)
        ops = ["op_a", "op_b", "op_c"]
        results = [scheduler.select_op(ops) for _ in range(10)]
        unique_ops = set(results)
        assert "op_a" in unique_ops
        assert "op_b" in unique_ops
        assert "op_c" in unique_ops

    def test_uniform_record_and_stats(self):
        scheduler = UniformScheduler()
        scheduler.init_arm("op_a")
        scheduler.init_arm("op_b")

        scheduler.record("op_a", True, 1.0)
        scheduler.record("op_a", False, 1.0)
        scheduler.record("op_b", True, 1.0)

        stats = scheduler.bandit_stats()
        assert stats["op_a"] == (1.0, 1.0)
        assert stats["op_b"] == (1.0, 0)

    def test_uniform_empty_and_single(self):
        scheduler = UniformScheduler()
        assert scheduler.select_op([]) == ""
        assert scheduler.select_op(["only"]) == "only"
        assert scheduler.select_op(["only"]) == "only"

    def test_uniform_register_on_fly(self):
        scheduler = UniformScheduler()
        scheduler._rng = RandPool(seed=42)
        ops = ["op_x", "op_y"]
        result = scheduler.select_op(ops)
        assert result in ops
        assert "op_x" in scheduler._operator_counts
        assert "op_y" in scheduler._operator_counts
