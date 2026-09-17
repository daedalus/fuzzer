"""Tests for OpTangScheduler / OperatorEdgeTracker.

The core claim under test is reuse, not new math: TangRecommendationScheduler
is exercised directly against an OperatorEdgeTracker to confirm the
duck-typing actually works (refit() reads seed_edges/seed_hit_counts and
does not care that the keys are operator names), then OpTangScheduler's
thin wiring (select_op/record/observe_new_edges/maybe_refit) is tested on
top of that.
"""

import pytest

from fuzzer_tool.core.op_edge_tracker import OperatorEdgeTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_tang import OpTangScheduler
from fuzzer_tool.core.schedulers.seed_tang import TangRecommendationScheduler


class TestOperatorEdgeTracker:
    def test_record_accumulates_edges_and_counts(self):
        t = OperatorEdgeTracker()
        t.record("havoc", [1, 2, 3])
        t.record("havoc", [3, 4])
        assert t.seed_edges["havoc"] == {1, 2, 3, 4}
        assert t.seed_hit_counts["havoc"][3] == 2
        assert t.seed_hit_counts["havoc"][1] == 1

    def test_record_empty_is_noop(self):
        t = OperatorEdgeTracker()
        t.record("havoc", [])
        assert "havoc" not in t.seed_edges

    def test_reset_clears_both_dicts(self):
        t = OperatorEdgeTracker()
        t.record("havoc", [1])
        t.reset()
        assert t.seed_edges == {}
        assert t.seed_hit_counts == {}

    def test_duck_types_for_tang_refit(self):
        """The actual reuse claim: TangRecommendationScheduler.refit() works
        unmodified against an OperatorEdgeTracker instance."""
        t = OperatorEdgeTracker()
        t.record("havoc", [1, 2, 3])
        t.record("splice", [1, 4])
        t.record("arith", [5])
        t.record("dict_insert", [1, 2, 3, 4, 5, 6])
        tang = TangRecommendationScheduler(rng=RandPool(seed=1), rank=2, refit_interval=1)
        assert tang.refit(t) is True
        assert tang.fitted
        assert tang.last_shape == (4, 6)
        for op in ("havoc", "splice", "arith", "dict_insert"):
            energy = tang.seed_energy(op)
            assert energy >= 0.0


class TestOpTangScheduler:
    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpTangScheduler(rng=None)

    def test_select_op_before_fit_is_random_but_valid(self):
        sched = OpTangScheduler(rng=RandPool(seed=1))
        ops = ["a", "b", "c"]
        for _ in range(10):
            assert sched.select_op(ops) in ops

    def test_select_op_single_arm_shortcut(self):
        sched = OpTangScheduler(rng=RandPool(seed=1))
        assert sched.select_op(["only"]) == "only"

    def test_select_op_empty(self):
        sched = OpTangScheduler(rng=RandPool(seed=1))
        assert sched.select_op([]) == ""

    def test_record_is_noop_and_does_not_raise(self):
        sched = OpTangScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True, weight=0.5)  # must not raise

    def test_maybe_refit_respects_interval(self):
        sched = OpTangScheduler(rng=RandPool(seed=1), rank=1, refit_interval=100)
        sched.observe_new_edges("a", [1, 2])
        sched.observe_new_edges("b", [3])
        # First call always fires (same -(1<<60) sentinel convention as
        # TangRecommendationScheduler itself), consistent with seed_tang.py.
        assert sched.maybe_refit(exec_count=5) is True
        assert sched.maybe_refit(exec_count=10) is False  # under interval since
        assert sched.maybe_refit(exec_count=150) is True  # crosses interval
        assert sched.fitted

    def test_select_op_after_fit_uses_energies(self):
        sched = OpTangScheduler(rng=RandPool(seed=7), rank=2, refit_interval=1)
        ops = ["havoc", "splice", "arith", "dict_insert"]
        sched.observe_new_edges("havoc", [1, 2, 3])
        sched.observe_new_edges("splice", [1, 4])
        sched.observe_new_edges("arith", [5])
        sched.observe_new_edges("dict_insert", [1, 2, 3, 4, 5, 6])
        assert sched.maybe_refit(exec_count=1) is True
        assert sched.fitted
        picks = [sched.select_op(ops) for _ in range(20)]
        assert all(p in ops for p in picks)
