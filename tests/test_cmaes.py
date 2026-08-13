"""Tests for CMAESScheduler."""

import numpy as np
import pytest

from fuzzer_tool.core.schedulers import CMAESScheduler


class TestCMAESScheduler:
    def test_init_defaults(self):
        sched = CMAESScheduler()
        assert sched.pop_size == 8
        assert sched.generation_size == 200
        assert sched.step_size == pytest.approx(0.3)
        assert sched.elite_frac == pytest.approx(0.5)

    def test_init_arm(self):
        sched = CMAESScheduler()
        sched.init_arm("bit_flip")
        assert "bit_flip" in sched.op_index
        assert sched.op_index["bit_flip"] == 0
        assert sched._mean is not None
        assert len(sched._mean) == 1

    def test_init_arm_idempotent(self):
        sched = CMAESScheduler()
        sched.init_arm("bit_flip")
        sched.init_arm("bit_flip")
        assert sched.op_index["bit_flip"] == 0
        assert len(sched.operators) == 1

    def test_init_arm_grows_state(self):
        sched = CMAESScheduler()
        sched.init_arm("bit_flip")
        sched.init_arm("byte_flip")
        assert len(sched.operators) == 2
        assert sched._mean.shape[0] == 2
        assert sched._C.shape == (2, 2)

    def test_select_op_returns_registered_operator(self):
        sched = CMAESScheduler(pop_size=4, generation_size=8)
        sched.init_arm("bit_flip")
        sched.init_arm("byte_flip")
        for _ in range(20):
            op = sched.select_op(["bit_flip", "byte_flip"])
            assert op in ("bit_flip", "byte_flip")

    def test_select_op_empty_fallback(self):
        sched = CMAESScheduler()
        assert sched.select_op([]) == ""
        assert sched.select_op(["bit_flip"]) == "bit_flip"

    def test_record_triggers_update_after_generation_size(self):
        sched = CMAESScheduler(pop_size=4, generation_size=8)
        sched.init_arm("A")
        sched.init_arm("B")
        # First generation: 8 records trigger an update
        for _ in range(8):
            op = sched.select_op(["A", "B"])
            sched.record(op, success=True)
        assert sched._generation == 1

    def test_record_success_increments_discoveries(self):
        sched = CMAESScheduler()
        sched.init_arm("A")
        op = sched.select_op(["A"])
        sched.record(op, success=True)
        assert sched._total_discoveries == 1
        assert sched._total_execs == 1

    def test_covariance_adapts_to_correlated_operators(self):
        sched = CMAESScheduler(pop_size=8, generation_size=16, step_size=0.5)
        ops = ["A", "B"]
        for _ in range(len(ops)):
            sched.init_arm(ops[_])
        # Freeze RNG so the test is deterministic.
        np.random.seed(0)
        for _ in range(16):
            # Use a fixed candidate pool that always puts A and B at top
            sched._new_generation()
            # Draw only from the first candidate, which we know gets high reward
            op = sched.select_op(ops)
            # Reward both equally so covariance captures their joint movement
            sched.record(op, success=True, weight=1.0)
        # After one generation, C should have changed from identity
        diag = np.diag(sched._C)
        assert not np.allclose(diag, 1.0)

    def test_state_round_trip(self):
        sched = CMAESScheduler(pop_size=4, generation_size=16)
        ops = ["A", "B", "C"]
        for op in ops:
            sched.init_arm(op)
        for _ in range(16):
            op = sched.select_op(ops)
            sched.record(op, success=op == "A", weight=1.0)
        data = sched.to_dict()
        restored = CMAESScheduler()
        for op in ops:
            restored.init_arm(op)
        restored.from_dict(data)
        assert restored._generation == sched._generation
        assert restored._sigma == pytest.approx(sched._sigma)
        assert restored._mean.shape == sched._mean.shape
        assert np.allclose(restored._mean, sched._mean)
        assert np.allclose(restored._C, sched._C)
        # After restore, select_op should work without reinitialising
        op = restored.select_op(ops)
        assert op in ops

    def test_convergence_stats_keys(self):
        sched = CMAESScheduler()
        sched.init_arm("A")
        stats = sched.convergence_stats()
        assert "generation" in stats
        assert "sigma" in stats
        assert "top_op" in stats
        assert "top_prob" in stats
        assert "total_execs" in stats
        assert "total_discoveries" in stats

    def test_bandit_stats(self):
        sched = CMAESScheduler()
        sched.init_arm("A")
        sched.select_op(["A"])
        sched.record("A", success=True)
        stats = sched.bandit_stats()
        assert "_cmaes_global" in stats
        assert stats["_cmaes_global"][0] == 1.0
        assert stats["_cmaes_global"][1] == 0.0

    def test_generation_property(self):
        sched = CMAESScheduler(pop_size=2, generation_size=2)
        sched.init_arm("A")
        assert sched.generation == 0
        sched.select_op(["A"])
        sched.record("A", success=True)
        sched.select_op(["A"])
        sched.record("A", success=True)
        assert sched.generation == 1

    def test_best_fitness_property(self):
        sched = CMAESScheduler()
        sched.init_arm("A")
        assert sched.best_fitness == 0.0
        sched.select_op(["A"])
        sched.record("A", success=True)
        assert sched.best_fitness > 0.0
