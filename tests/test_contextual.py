"""Tests for ContextualLinUCBScheduler (LinUCB) and its cost feature.

Covers the scheduler in isolation (Sherman-Morrison correctness, cold-start
exploration, learning direction) plus the operators.py glue that appends
the _op_time_ema-derived cost feature to the shared seed context.
"""

import math

import numpy as np

from fuzzer_tool.core.schedulers.contextual import ContextualLinUCBScheduler
from fuzzer_tool.services.operators import CONTEXT_DIM, OperatorEngine


class TestContextualLinUCBScheduler:
    def test_init_arm_default_state(self):
        sched = ContextualLinUCBScheduler(dim=3, lambda_reg=2.0)
        sched.init_arm("bit_flip")
        inv_lambda = 1.0 / 2.0
        assert np.allclose(
            sched._A_inv["bit_flip"],
            [
                [inv_lambda, 0.0, 0.0],
                [0.0, inv_lambda, 0.0],
                [0.0, 0.0, inv_lambda],
            ],
        )
        assert np.allclose(sched._b["bit_flip"], [0.0, 0.0, 0.0])
        assert sched._pulls["bit_flip"] == 0

    def test_init_arm_idempotent(self):
        sched = ContextualLinUCBScheduler(dim=2)
        sched.init_arm("a")
        sched.record("a", [1.0, 0.0], 1.0)
        sched.init_arm("a")  # must not reset state
        assert sched._pulls["a"] == 1

    def test_score_cold_start_is_pure_exploration_bonus(self):
        # With b=0 the mean term is 0, so an unseen arm's score is
        # entirely the alpha * confidence term.
        sched = ContextualLinUCBScheduler(dim=2, alpha=1.5, lambda_reg=1.0)
        x = [1.0, 0.0]
        score = sched.score("fresh_op", x)
        # x^T A_inv x == 1.0 (A_inv = I here), so score == alpha * sqrt(1) == alpha
        assert math.isclose(score, 1.5, rel_tol=1e-9)

    def test_select_op_prefers_higher_scoring_arm(self):
        sched = ContextualLinUCBScheduler(dim=2, alpha=0.0, lambda_reg=1.0)
        x = [1.0, 0.0]
        # Manually bias "good" arm's regressor toward a high reward on x.
        sched.init_arm("good")
        sched.init_arm("bad")
        sched.record("good", x, reward=1.0)
        sched.record("bad", x, reward=0.0)
        assert sched.select_op(["good", "bad"], x) == "good"

    def test_select_op_single_op_shortcircuits(self):
        sched = ContextualLinUCBScheduler(dim=2)
        assert sched.select_op(["only"], [0.0, 0.0]) == "only"

    def test_select_op_empty_returns_empty_string(self):
        sched = ContextualLinUCBScheduler(dim=2)
        assert sched.select_op([], [0.0, 0.0]) == ""

    def test_select_op_accepts_per_arm_context_callable(self):
        # LinUCB is called with a callable in operators.py so each arm gets
        # its own cost feature appended to a shared seed-context prefix.
        sched = ContextualLinUCBScheduler(dim=2, alpha=0.0, lambda_reg=1.0)
        contexts = {"cheap": [1.0, 0.0], "expensive": [0.0, 1.0]}
        sched.init_arm("cheap")
        sched.init_arm("expensive")
        sched.record("cheap", contexts["cheap"], reward=1.0)
        sched.record("expensive", contexts["expensive"], reward=0.0)
        chosen = sched.select_op(["cheap", "expensive"], lambda op: contexts[op])
        assert chosen == "cheap"

    def test_record_learns_reward_direction(self):
        # After many rounds of reward=1 on x=[1,0] and reward=0 on x=[0,1]
        # for the same arm, theta should learn to separate the two
        # directions: predicted value on [1,0] > predicted value on [0,1].
        sched = ContextualLinUCBScheduler(dim=2, alpha=0.0, lambda_reg=0.1)
        for _ in range(50):
            sched.record("op", [1.0, 0.0], reward=1.0)
            sched.record("op", [0.0, 1.0], reward=0.0)
        assert sched.score("op", [1.0, 0.0]) > sched.score("op", [0.0, 1.0])

    def test_sherman_morrison_matches_direct_2x2_inverse(self):
        # Verify the incremental Sherman-Morrison update against a hand
        # computed direct inverse for a couple of rank-1 updates.
        sched = ContextualLinUCBScheduler(dim=2, lambda_reg=1.0)
        sched.init_arm("op")

        def direct_inverse(mat):
            (a, b), (c, d) = mat
            det = a * d - b * c
            return [[d / det, -b / det], [-c / det, a / det]]

        def matmul_add_outer(mat, x):
            return [[mat[i][j] + x[i] * x[j] for j in range(2)] for i in range(2)]

        # A starts at lambda*I = I.
        a_direct = [[1.0, 0.0], [0.0, 1.0]]
        for x, r in [([1.0, 2.0], 0.5), ([0.5, -1.0], 1.0), ([2.0, 0.1], 0.0)]:
            sched.record("op", x, r)
            a_direct = matmul_add_outer(a_direct, x)
            inv_direct = direct_inverse(a_direct)
            for i in range(2):
                for j in range(2):
                    assert math.isclose(
                        sched._A_inv["op"][i][j], inv_direct[i][j], rel_tol=1e-6, abs_tol=1e-9
                    )

    def test_bandit_stats(self):
        sched = ContextualLinUCBScheduler(dim=3)
        sched.record("a", [1.0, 0.0, 0.0], 1.0)
        sched.record("b", [0.0, 1.0, 0.0], 0.5)
        stats = sched.bandit_stats()
        assert stats["contextual_pulls"] == 2
        assert stats["operators_tracked"] == 2
        assert stats["dim"] == 3

    def test_select_op_batched_matches_per_arm_score(self):
        sched = ContextualLinUCBScheduler(dim=4, alpha=0.5, lambda_reg=1.0)
        ops = ["a", "b", "c"]
        for op in ops:
            sched.init_arm(op)
        # Give arms different histories so scores diverge.
        for _ in range(10):
            sched.record("a", [1.0, 0.0, 0.0, 0.0], reward=1.0)
            sched.record("b", [0.0, 1.0, 0.0, 0.0], reward=0.5)
            sched.record("c", [0.0, 0.0, 1.0, 0.0], reward=0.0)
        x = [0.5, 0.5, 0.5, 0.5]
        expected = max(ops, key=lambda op: sched.score(op, x))
        assert sched.select_op(ops, x) == expected

    def test_select_op_batched_callable_context_matches_per_arm(self):
        sched = ContextualLinUCBScheduler(dim=3, alpha=0.0, lambda_reg=1.0)
        ops = ["x", "y"]
        contexts = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0]}
        for op in ops:
            sched.init_arm(op)
        sched.record("x", contexts["x"], reward=1.0)
        sched.record("y", contexts["y"], reward=0.0)
        expected = max(ops, key=lambda op: sched.score(op, contexts[op]))
        assert sched.select_op(ops, lambda op: contexts[op]) == expected


def _minimal_fuzzer_stub():
    """A bare object with just enough attributes for the context-building
    helpers on OperatorEngine (max_len, seed_meta, _op_time_ema, etc.)."""

    class _Stub:
        max_len = 1024
        seed_meta = {}
        _edge_tracker = None
        _cmplog = None
        _corpus_size_stats = None
        _op_time_ema: dict = {}

    return _Stub()


class TestOpCostFeature:
    """_op_cost_feature squashes _op_time_ema (seconds) into ~[0, 1] via
    log10, so the LinUCB context sees relative operator cost per-seed."""

    def _engine(self):
        f = _minimal_fuzzer_stub()
        return OperatorEngine(f), f

    def test_default_is_neutral_when_untimed(self):
        engine, f = self._engine()
        assert engine._op_cost_feature("never_timed") == 0.5

    def test_cheap_operator_scores_low(self):
        engine, f = self._engine()
        f._op_time_ema = {"bit_flip": 1e-6}  # ~1us
        val = engine._op_cost_feature("bit_flip")
        assert 0.0 <= val < 0.3

    def test_expensive_operator_scores_high(self):
        engine, f = self._engine()
        f._op_time_ema = {"gradient_descent": 1.0}  # ~1s
        val = engine._op_cost_feature("gradient_descent")
        assert 0.7 < val <= 1.0

    def test_monotonic_in_cost(self):
        engine, f = self._engine()
        f._op_time_ema = {"fast": 1e-5, "mid": 1e-3, "slow": 1e-1}
        fast = engine._op_cost_feature("fast")
        mid = engine._op_cost_feature("mid")
        slow = engine._op_cost_feature("slow")
        assert fast < mid < slow

    def test_clamped_to_unit_interval(self):
        engine, f = self._engine()
        f._op_time_ema = {"absurdly_slow": 1e6, "absurdly_fast": 1e-30}
        assert engine._op_cost_feature("absurdly_slow") <= 1.0
        assert engine._op_cost_feature("absurdly_fast") >= 0.0


class TestContextVector:
    def _engine(self):
        f = _minimal_fuzzer_stub()
        return OperatorEngine(f), f

    def test_context_vector_length_matches_context_dim(self):
        engine, f = self._engine()
        f._current_context_shared = [0.0] * (CONTEXT_DIM - 1)
        f._op_time_ema = {"op": 1e-4}
        vec = engine._context_vector("op")
        assert len(vec) == CONTEXT_DIM

    def test_context_vector_appends_cost_as_last_element(self):
        engine, f = self._engine()
        shared = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        f._current_context_shared = shared
        f._op_time_ema = {"op": 1e-3}
        vec = engine._context_vector("op")
        assert vec[:-1] == shared
        assert math.isclose(vec[-1], engine._op_cost_feature("op"))

    def test_context_vector_falls_back_when_no_shared_context_cached(self):
        engine, f = self._engine()
        # _current_context_shared not set at all (e.g. contextual disabled)
        vec = engine._context_vector("op")
        assert len(vec) == CONTEXT_DIM
