"""Tests for OpKruskalCountScheduler: Kruskal-count coupling over the
operator jump graph built from each op's own success rate.

Uniform rates are a useful degenerate case: identical jump lengths keep
every walker at its original relative spacing forever (mod n), so they
never couple. Coupling requires rates -- and therefore jump lengths -- to
differ across ops.
"""

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_kruskal_count import (
    OpKruskalCountScheduler,
    attractor_shares,
    jump_table,
    trace,
    walker_starts,
)


class TestWalkerStarts:
    def test_fewer_ops_than_walker_count(self):
        assert walker_starts(2) == [0, 1]

    def test_zero_ops(self):
        assert walker_starts(0) == []

    def test_one_op(self):
        assert walker_starts(1) == [0]

    def test_spaced_starts(self):
        assert walker_starts(8) == [0, 2, 4, 6]


class TestJumpTable:
    def test_zero_rate_jumps_by_one(self):
        assert jump_table([0.0, 0.0, 0.0]) == [1, 1, 1]

    def test_max_rate_jumps_to_domain_edge(self):
        # n=4: jump = 1 + floor(1.0 * 3) = 4
        assert jump_table([1.0, 0.0, 0.0, 0.0])[0] == 4

    def test_single_op_always_self_loops(self):
        assert jump_table([1.0]) == [1]

    def test_empty(self):
        assert jump_table([]) == []

    def test_clamped_out_of_range_rate(self):
        # Defensive: scores() never produces out-of-[0,1] rates, but the
        # table builder itself should not misbehave if it ever did.
        assert jump_table([5.0, -3.0]) == jump_table([1.0, 0.0])


class TestTrace:
    def test_no_coupling_possible_with_one_walker(self):
        t = trace([1])
        assert t.couple_steps == {}

    def test_uniform_jumps_couple_immediately(self):
        # All jump by 1 on a ring of 8: walkers stay in lockstep at their
        # original spacing forever and never land on the same index.
        t = trace([1] * 8)
        assert t.couple_steps == {}

    def test_differentiated_jumps_can_couple(self):
        # n=4, one high-rate op (index 0) jumps far while the rest crawl by
        # 1; walker starting at 0 should eventually land on another walker.
        jumps = jump_table([1.0, 0.0, 0.0, 0.0])
        t = trace(jumps)
        assert len(t.couple_steps) > 0

    def test_paths_length_matches_cap(self):
        jumps = jump_table([1.0, 0.0, 0.0, 0.0])
        t = trace(jumps)
        cap = max(2 * len(jumps), 1)
        assert all(len(p) == cap for p in t.paths)


class TestAttractorShares:
    def test_no_coupling_returns_all_zero(self):
        t = trace([1] * 8)
        assert attractor_shares(t, 8) == [0.0] * 8

    def test_shares_are_fractions_in_unit_interval(self):
        jumps = jump_table([1.0, 0.0, 0.0, 0.0])
        t = trace(jumps)
        shares = attractor_shares(t, 4)
        assert all(0.0 <= s <= 1.0 for s in shares)

    def test_empty_paths_returns_zero(self):
        from fuzzer_tool.core.schedulers.op_kruskal_count import WalkTrace

        assert attractor_shares(WalkTrace(), 3) == [0.0, 0.0, 0.0]


class TestOpKruskalCountScheduler:
    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpKruskalCountScheduler(rng=None)

    def test_select_op_single_arm_shortcut(self):
        sched = OpKruskalCountScheduler(rng=RandPool(seed=1))
        assert sched.select_op(["only"]) == "only"

    def test_select_op_empty(self):
        sched = OpKruskalCountScheduler(rng=RandPool(seed=1))
        assert sched.select_op([]) == ""

    def test_record_tracks_attempts_and_successes(self):
        sched = OpKruskalCountScheduler(rng=RandPool(seed=1))
        sched.record("a", success=False)
        sched.record("b", success=True)
        assert sched.attempts["a"] == 1
        assert sched.attempts["b"] == 1
        assert sched.successes.get("a", 0.0) == 0.0
        assert sched.successes.get("b", 0.0) == 1.0

    def test_record_weight_scales_success_only(self):
        sched = OpKruskalCountScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True, weight=0.5)
        assert sched.successes["a"] == pytest.approx(0.5)
        sched.record("a", success=False, weight=99.0)
        assert sched.successes["a"] == pytest.approx(0.5)
        assert sched.attempts["a"] == 2

    def test_rates_zero_for_unattempted_op(self):
        sched = OpKruskalCountScheduler(rng=RandPool(seed=1))
        assert sched.rates(["never_seen"]) == [0.0]

    def test_scores_never_below_raw_rate(self):
        """score = rate * (1 + share); share >= 0 always, so score >= rate."""
        sched = OpKruskalCountScheduler(rng=RandPool(seed=1))
        ops = ["a", "b", "c", "d"]
        sched.successes = {"a": 10.0, "b": 3.0, "c": 0.0, "d": 5.0}
        sched.attempts = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0}
        scores = sched.scores(ops)
        rates = dict(zip(ops, sched.rates(ops), strict=True))
        for op in ops:
            assert scores[op] >= rates[op] - 1e-9

    def test_scores_zero_op_scores_zero(self):
        sched = OpKruskalCountScheduler(rng=RandPool(seed=1))
        sched.successes = {"a": 10.0}
        sched.attempts = {"a": 10.0, "never_succeeded": 10.0}
        scores = sched.scores(["a", "never_succeeded"])
        # rate=0 => score = 0 * (1 + share) = 0 regardless of attractor share.
        assert scores["never_succeeded"] == 0.0

    def test_select_op_deterministic_with_fixed_seed(self):
        sched1 = OpKruskalCountScheduler(rng=RandPool(seed=42))
        sched2 = OpKruskalCountScheduler(rng=RandPool(seed=42))
        ops = ["a", "b", "c"]
        for _ in range(20):
            sched1.record("a", success=True)
            sched2.record("a", success=True)
        picks1 = [sched1.select_op(ops) for _ in range(10)]
        picks2 = [sched2.select_op(ops) for _ in range(10)]
        assert picks1 == picks2

    def test_select_op_all_zero_scores_is_uniform_fallback(self):
        # No records at all: every rate is 0, every score is 0, select_op
        # must not divide by zero and must still return a valid op.
        sched = OpKruskalCountScheduler(rng=RandPool(seed=7))
        ops = ["a", "b", "c"]
        picks = {sched.select_op(ops) for _ in range(50)}
        assert picks <= set(ops)
