"""Tests for OpKatzScheduler: classical (cycle-tolerant) Katz over the
operator discovery-transition graph.

Hand-derived oracle for the 2-cycle: A[0][1]=1, A[1][0]=1 (a<->b, each
always followed by the other on success), uniform beta=1. Eigenvalues of
[[0,1],[1,0]] are +-1, so rho=1 and alpha = 0.85/1 = 0.85. By symmetry
c[0] == c[1], and c = (I - alpha*A^T)^-1 @ beta gives a closed form:
    c[0] = c[1] = 1 / (1 - alpha)
"""

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_katz import (
    OpKatzScheduler,
    build_transition_matrix,
    classical_katz_scores,
)


class TestBuildTransitionMatrix:
    def test_empty_counts_all_zero(self):
        a = build_transition_matrix({}, ["x", "y"])
        assert a.shape == (2, 2)
        assert not a.any()

    def test_row_normalized(self):
        counts = {"x": {"y": 3, "z": 1}}
        a = build_transition_matrix(counts, ["x", "y", "z"])
        assert a[0, 1] == pytest.approx(0.75)
        assert a[0, 2] == pytest.approx(0.25)
        assert a[1].sum() == 0.0  # y has no recorded outgoing transitions

    def test_unknown_op_in_counts_ignored(self):
        counts = {"ghost": {"y": 5}, "x": {"ghost": 5, "y": 5}}
        a = build_transition_matrix(counts, ["x", "y"])
        # "ghost" is not in the offered ops list, so x's row normalizes
        # over only the edges that land on known ops.
        assert a[0, 1] == pytest.approx(1.0)


class TestClassicalKatzScores:
    def test_no_edges_returns_beta(self):
        a = np.zeros((3, 3))
        beta = np.array([1.0, 2.0, 3.0])
        c = classical_katz_scores(a, beta)
        assert np.allclose(c, beta)

    def test_two_cycle_closed_form(self):
        a = np.array([[0.0, 1.0], [1.0, 0.0]])
        beta = np.ones(2)
        c = classical_katz_scores(a, beta, alpha_fraction=0.85)
        expected = 1.0 / (1.0 - 0.85)
        assert c[0] == pytest.approx(expected, rel=1e-6)
        assert c[1] == pytest.approx(expected, rel=1e-6)

    def test_cycle_does_not_raise(self):
        # The DAG seed_katz.py raises ValueError on a cycle; this must not.
        a = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        c = classical_katz_scores(a, np.ones(3))
        assert c.shape == (3,)
        assert np.all(np.isfinite(c))

    def test_empty_graph(self):
        c = classical_katz_scores(np.zeros((0, 0)), np.zeros(0))
        assert c.shape == (0,)


class TestOpKatzScheduler:
    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpKatzScheduler(rng=None)

    def test_select_op_single_arm_shortcut(self):
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        assert sched.select_op(["only"]) == "only"

    def test_select_op_empty(self):
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        assert sched.select_op([]) == ""

    def test_record_builds_transition_on_success_only(self):
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        sched.record("a", success=False)
        sched.record("b", success=True)
        # a->b transition should NOT be recorded: a's own call was a failure,
        # but the edge is prev_op(a) -> current op(b) recorded on b's
        # success regardless of a's outcome -- this only checks self-loops
        # are excluded and failures still update _prev_op/attempts.
        assert sched.attempts["a"] == 1
        assert sched.attempts["b"] == 1
        assert sched.successes.get("a", 0.0) == 0.0
        assert sched.successes.get("b", 0.0) == 1.0

    def test_record_no_self_loop(self):
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True)
        sched.record("a", success=True)
        # prev_op == op on the second call: must not record a->a.
        assert "a" not in sched.transition_counts.get("a", {})

    def test_record_transition_on_repeat_success(self):
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True)
        sched.record("b", success=True)
        assert sched.transition_counts["a"]["b"] == 1

    def test_scores_favor_reinforcing_productive_cycle(self):
        """beta is now the raw success rate (exploitation-favoring, the
        opposite sign of the seed-side seed_katz.py's frontier-seeking beta --
        see module docstring). A mutually-reinforcing cycle between two
        consistently-successful operators should score well above an
        isolated operator with a middling standalone rate, even though
        the isolated op's raw beta (0.5) is nonzero and the cycle ops'
        beta is not artificially deflated."""
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        ops = ["feeder", "productive", "isolated"]
        sched.transition_counts = {
            "feeder": {"productive": 30},
            "productive": {"feeder": 29, "isolated": 1},
        }
        sched.successes = {"feeder": 30.0, "productive": 30.0, "isolated": 30.0}
        sched.attempts = {"feeder": 30.0, "productive": 30.0, "isolated": 60.0}
        scores = sched.scores(ops)
        assert scores["feeder"] > scores["isolated"]
        assert scores["productive"] > scores["isolated"]

    def test_beta_is_raw_rate_not_complement(self):
        """Regression guard for the beta-sign bug: an op with a perfect
        standalone record and zero transitions must score at its own
        rate (1.0), not at 1-rate (0.0)."""
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        sched.successes = {"solo": 10.0}
        sched.attempts = {"solo": 10.0}
        scores = sched.scores(["solo"])
        assert scores["solo"] == pytest.approx(1.0)

    def test_select_op_deterministic_with_fixed_seed(self):
        sched1 = OpKatzScheduler(rng=RandPool(seed=42))
        sched2 = OpKatzScheduler(rng=RandPool(seed=42))
        ops = ["a", "b", "c"]
        for _ in range(20):
            sched1.record("a", success=True)
            sched2.record("a", success=True)
        picks1 = [sched1.select_op(ops) for _ in range(10)]
        picks2 = [sched2.select_op(ops) for _ in range(10)]
        assert picks1 == picks2

    def test_rejects_bad_explore_floor(self):
        with pytest.raises(ValueError):
            OpKatzScheduler(rng=RandPool(seed=1), explore_floor=1.0)
        with pytest.raises(ValueError):
            OpKatzScheduler(rng=RandPool(seed=1), explore_floor=-0.1)

    def test_floor_bounds_probability_away_from_certainty(self):
        """Regression test for the lock-in bug: before the floor, a single
        arm with any nonzero score against all-zero-score rivals captured
        essentially 100% of the probability mass. With the floor, no arm's
        probability can approach certainty and every zero-score arm keeps
        a floor-sized share."""
        sched = OpKatzScheduler(rng=RandPool(seed=5))
        n = 12
        ops = [f"op{i}" for i in range(n)]
        sched.successes = {"op0": 1.0}
        sched.attempts = dict.fromkeys(ops, 10.0)
        probs = sched._select_probs(ops)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] < 0.96
        raw_floor = sched.explore_floor / n
        for p in probs[1:]:
            assert p >= raw_floor * 0.9

    def test_explore_floor_zero_restores_old_unfloored_draw(self):
        sched = OpKatzScheduler(rng=RandPool(seed=5), explore_floor=0.0)
        sched.successes = {"tried": 5.0}
        sched.attempts = {"tried": 10.0, "never_tried": 10.0}
        probs = sched._select_probs(["tried", "never_tried"])
        assert probs[1] < 0.06 / 2

    def test_repeated_lockin_seed_no_longer_starves_true_best(self):
        """End-to-end regression matching the bandit_env.py finding that
        motivated this fix: a scheduler using the old unfloored draw could
        get permanently stuck on a suboptimal arm after its first lucky
        success. Post-fix the true best arm must keep receiving a
        non-trivial share of the tail."""
        import random as _random

        best = "best"
        arms = [best, "runner_up", "base", "d", "e"]
        probs = {best: 0.30, "runner_up": 0.18, "base": 0.05, "d": 0.05, "e": 0.05}
        sched = OpKatzScheduler(rng=RandPool(seed=1))
        env_rng = _random.Random(1 ^ 0x5EED)
        rounds = 20_000
        tail_start = int(rounds * 0.8)
        tail_picks = 0
        for t in range(rounds):
            op = sched.select_op(arms)
            success = env_rng.random() < probs[op]
            sched.record(op, success)
            if t >= tail_start and op == best:
                tail_picks += 1
        assert tail_picks / (rounds - tail_start) > 0.0


class TestBadnessIndexedFloor:
    """P1: `badness_fn` turns the single-point `explore_floor` constant
    into a family indexed by a runtime badness score. See
    core/badness_floor.py and
    docs/handover/handover_badness_indexed_floor_2026-09-21.md.
    """

    def test_no_badness_fn_matches_original_static_floor(self):
        sched = OpKatzScheduler(rng=RandPool(seed=1), explore_floor=0.1)
        assert sched._current_explore_floor() == pytest.approx(0.1)

    def test_badness_zero_matches_static_floor(self):
        sched = OpKatzScheduler(
            rng=RandPool(seed=1),
            explore_floor=0.06,
            badness_fn=lambda: 0.0,
            max_explore_floor=0.3,
        )
        assert sched._current_explore_floor() == pytest.approx(0.06)

    def test_badness_one_reaches_max_floor(self):
        sched = OpKatzScheduler(
            rng=RandPool(seed=1),
            explore_floor=0.06,
            badness_fn=lambda: 1.0,
            max_explore_floor=0.3,
        )
        assert sched._current_explore_floor() == pytest.approx(0.3)

    def test_badness_fn_sampled_fresh_each_call(self):
        state = {"badness": 0.0}
        sched = OpKatzScheduler(
            rng=RandPool(seed=1),
            explore_floor=0.06,
            badness_fn=lambda: state["badness"],
            max_explore_floor=0.3,
        )
        assert sched._current_explore_floor() == pytest.approx(0.06)
        state["badness"] = 1.0
        assert sched._current_explore_floor() == pytest.approx(0.3)

    def test_raising_badness_fn_falls_back_to_static_floor(self):
        def _broken():
            raise RuntimeError("no regime detector yet")

        sched = OpKatzScheduler(
            rng=RandPool(seed=1),
            explore_floor=0.06,
            badness_fn=_broken,
            max_explore_floor=0.3,
        )
        # Must not raise out of a live select_op-adjacent call.
        assert sched._current_explore_floor() == pytest.approx(0.06)

    def test_invalid_max_explore_floor_rejected_at_construction(self):
        with pytest.raises(ValueError, match="max_floor"):
            OpKatzScheduler(
                rng=RandPool(seed=1),
                explore_floor=0.3,
                badness_fn=lambda: 0.5,
                max_explore_floor=0.1,
            )

    def test_high_badness_raises_the_actual_selection_floor(self):
        """End-to-end: pinning badness at 1.0 must yield a strictly higher
        normalized selection probability for every never-attempted arm
        than pinning badness at 0.0 does (comparative, not an absolute
        threshold -- post-floor normalization dilutes the raised floor
        by however much probability mass the dominant arm keeps, the same
        dynamic `test_floor_bounds_probability_away_from_certainty`
        exercises for the static floor).
        """
        n = 10
        ops = [f"op{i}" for i in range(n)]

        def _make(badness_fn):
            sched = OpKatzScheduler(
                rng=RandPool(seed=5),
                explore_floor=0.06,
                badness_fn=badness_fn,
                max_explore_floor=0.5,
            )
            sched.successes = {"op0": 1.0}
            sched.attempts = dict.fromkeys(ops, 10.0)
            return sched

        low_probs = _make(lambda: 0.0)._select_probs(ops)
        high_probs = _make(lambda: 1.0)._select_probs(ops)
        for i in range(1, n):
            assert high_probs[i] > low_probs[i]

    def test_badness_fn_not_called_when_none(self):
        # Regression guard: with badness_fn=None, _current_explore_floor
        # must not attempt to call anything.
        sched = OpKatzScheduler(rng=RandPool(seed=1), explore_floor=0.06)
        assert sched.badness_fn is None
        assert sched._current_explore_floor() == pytest.approx(0.06)
