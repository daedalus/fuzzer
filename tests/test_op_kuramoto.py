"""Tests for OpKuramotoScheduler: phase-coherence bandit over the operator
discovery-transition graph, wiring core/kuramoto.py's diagnostic machinery
into an actual scheduler (see that module's docstring for what is and isn't
established empirically)."""

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_kuramoto import OpKuramotoScheduler


class TestConstruction:
    def test_requires_rng(self):
        with pytest.raises(ValueError):
            OpKuramotoScheduler(rng=None)

    def test_rejects_bad_steps_per_batch(self):
        with pytest.raises(ValueError):
            OpKuramotoScheduler(rng=RandPool(seed=1), steps_per_batch=0)

    def test_rejects_bad_recompute_batch(self):
        with pytest.raises(ValueError):
            OpKuramotoScheduler(rng=RandPool(seed=1), recompute_batch=0)


class TestSelectOpShortcuts:
    def test_select_op_empty(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        assert sched.select_op([]) == ""

    def test_select_op_single_arm_shortcut(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        assert sched.select_op(["only"]) == "only"
        # single-arm shortcut still registers the arm
        assert "only" in sched._phase


class TestRecord:
    def test_record_transition_recorded_on_destination_success_only(self):
        """The edge prev_op -> op is recorded whenever op's own call
        succeeds, regardless of prev_op's own outcome -- same discovery-
        linked semantics as OpKatzScheduler.record. A failed op, by
        contrast, must never contribute an outgoing edge from itself."""
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.record("a", success=False)
        sched.record("b", success=True)
        assert sched.attempts["a"] == 1
        assert sched.attempts["b"] == 1
        assert sched.successes.get("a", 0.0) == 0.0
        assert sched.successes.get("b", 0.0) == 1.0
        assert sched.transition_counts["a"]["b"] == 1

        sched.record("c", success=False)
        # b's call succeeded above, but c's own call fails here -- no
        # outgoing edge from b should be recorded on this step.
        assert sched.transition_counts.get("b", {}) == {}

    def test_record_no_self_loop(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True)
        sched.record("a", success=True)
        assert "a" not in sched.transition_counts.get("a", {})

    def test_record_transition_on_repeat_success(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True)
        sched.record("b", success=True)
        assert sched.transition_counts["a"]["b"] == 1

    def test_record_initializes_phase(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        assert "a" not in sched._phase
        sched.record("a", success=True)
        assert "a" in sched._phase
        assert 0.0 <= sched._phase["a"] < 2.0 * math.pi


class TestPhaseAdvance:
    def test_phases_unchanged_before_batch_threshold(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1), recompute_batch=1000)
        sched.select_op(["a", "b"])
        before = dict(sched._phase)
        for _ in range(10):
            sched.record("a", success=True)
        assert sched._phase == before

    def test_phases_change_after_batch_threshold(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1), recompute_batch=5, k=2.0)
        sched.select_op(["a", "b"])
        before = dict(sched._phase)
        for _ in range(5):
            sched.record("a", success=True)
        sched.scores(["a", "b"])  # triggers _maybe_advance
        assert sched._phase != before

    def test_zero_coupling_zero_omega_is_free_rotation(self):
        """No transitions recorded (zero coupling) and no successes yet
        (omega=0 for both arms) must leave phases exactly fixed -- the same
        identity core/kuramoto.py's own test_kuramoto.py checks for
        kuramoto_step directly."""
        sched = OpKuramotoScheduler(rng=RandPool(seed=7), recompute_batch=1)
        sched.init_arm("a")
        sched.init_arm("b")
        before = dict(sched._phase)
        sched._maybe_advance(["a", "b"])
        # forcing pending high enough to trigger the batch
        sched._pending = sched.recompute_batch
        sched._maybe_advance(["a", "b"])
        assert sched._phase["a"] == pytest.approx(before["a"])
        assert sched._phase["b"] == pytest.approx(before["b"])


class TestScores:
    def test_incoherent_population_degrades_to_plain_rate(self):
        """With r pinned near 0 (phases spread evenly), the alignment term
        must vanish and score reduce to the raw rate -- the module
        docstring's explicit reason for multiplying by r rather than using
        cos(theta-psi) unconditionally."""
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.init_arm("a")
        sched.init_arm("b")
        sched.init_arm("c")
        # Evenly spaced on the circle -> r ~= 0 exactly (3-fold symmetry).
        sched._phase["a"] = 0.0
        sched._phase["b"] = 2.0 * math.pi / 3.0
        sched._phase["c"] = 4.0 * math.pi / 3.0
        sched.successes = {"a": 5.0, "b": 3.0, "c": 1.0}
        sched.attempts = {"a": 10.0, "b": 10.0, "c": 10.0}
        sched._pending = 0  # avoid triggering a phase advance this call
        scores = sched.scores(["a", "b", "c"])
        assert scores["a"] == pytest.approx(0.5, abs=1e-6)
        assert scores["b"] == pytest.approx(0.3, abs=1e-6)
        assert scores["c"] == pytest.approx(0.1, abs=1e-6)

    def test_zero_rate_arm_scores_zero_regardless_of_phase(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.init_arm("never_succeeded")
        sched.init_arm("other")
        sched.successes = {"other": 5.0}
        sched.attempts = {"never_succeeded": 10.0, "other": 10.0}
        sched._pending = 0
        scores = sched.scores(["never_succeeded", "other"])
        assert scores["never_succeeded"] == 0.0

    def test_phase_aligned_arm_scores_above_misaligned_at_equal_rate(self):
        """Two arms with the identical raw rate must be distinguished by
        phase alignment once the population is coherent (nonzero r)."""
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.init_arm("aligned")
        sched.init_arm("misaligned")
        sched.init_arm("anchor")
        # Two arms clustered at phase 0 (aligned + anchor), one at pi
        # (misaligned) -> psi ~= 0, r > 0, cos(0-0)=1 > cos(pi-0)=-1.
        sched._phase["aligned"] = 0.01
        sched._phase["anchor"] = -0.01
        sched._phase["misaligned"] = math.pi
        sched.successes = {"aligned": 5.0, "misaligned": 5.0, "anchor": 5.0}
        sched.attempts = {"aligned": 10.0, "misaligned": 10.0, "anchor": 10.0}
        sched._pending = 0
        scores = sched.scores(["aligned", "misaligned", "anchor"])
        assert scores["aligned"] > scores["misaligned"]


class TestSelectOp:
    def test_deterministic_with_fixed_seed(self):
        sched1 = OpKuramotoScheduler(rng=RandPool(seed=42))
        sched2 = OpKuramotoScheduler(rng=RandPool(seed=42))
        ops = ["a", "b", "c"]
        for _ in range(20):
            sched1.record("a", success=True)
            sched2.record("a", success=True)
        picks1 = [sched1.select_op(ops) for _ in range(10)]
        picks2 = [sched2.select_op(ops) for _ in range(10)]
        assert picks1 == picks2

    def test_cold_start_gives_every_op_nonzero_probability(self):
        """A never-attempted op scores 0 (rate=0), but the floored draw in
        select_op must still give it strictly positive selection
        probability rather than an outright-excluded, unreachable arm."""
        sched = OpKuramotoScheduler(rng=RandPool(seed=3))
        sched.init_arm("tried")
        sched.init_arm("never_tried")
        sched.successes = {"tried": 5.0}
        sched.attempts = {"tried": 10.0}
        sched._pending = 0
        probs = sched._select_probs(["tried", "never_tried"])
        assert probs[1] > 0.0

    def test_floor_bounds_probability_away_from_certainty(self):
        """Regression test for the lock-in bug: before the floor, a single
        arm with any nonzero score against all-zero-score rivals captured
        essentially 100% of the probability mass (ratio of orders of
        magnitude against the bare 1e-9 shift). With the floor, no arm's
        probability can exceed 1 minus the floor mass reserved for the rest
        of the pool."""
        sched = OpKuramotoScheduler(rng=RandPool(seed=5))
        n = 12
        ops = [f"op{i}" for i in range(n)]
        for op in ops:
            sched.init_arm(op)
        # One arm has the only recorded success; the rest are all at rate 0,
        # exactly the scenario that produced permanent lock-in pre-fix.
        sched.successes = {"op0": 1.0}
        sched.attempts = dict.fromkeys(ops, 10.0)
        sched._pending = 0
        probs = sched._select_probs(ops)
        assert probs.sum() == pytest.approx(1.0)
        # Pre-fix this arm's probability was ~1.0 (a ratio of orders of
        # magnitude against the bare 1e-9 shift on every other arm); the
        # floor must pull it well clear of that.
        assert probs[0] < 0.96
        # Every zero-score arm still gets a floor-sized share (slightly
        # below the raw floor/n after the whole vector renormalizes).
        raw_floor = sched.explore_floor / n
        for p in probs[1:]:
            assert p >= raw_floor * 0.9

    def test_explore_floor_zero_restores_old_unfloored_draw(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=5), explore_floor=0.0)
        sched.init_arm("tried")
        sched.init_arm("never_tried")
        sched.successes = {"tried": 5.0}
        sched.attempts = {"tried": 10.0}
        sched._pending = 0
        probs = sched._select_probs(["tried", "never_tried"])
        # never_tried's probability is the bare 1e-9-scale shift floor, not
        # the 0.06/2 mixed-in floor that explore_floor=0.06 would apply.
        assert probs[1] < 0.06 / 2

    def test_rejects_bad_explore_floor(self):
        with pytest.raises(ValueError):
            OpKuramotoScheduler(rng=RandPool(seed=1), explore_floor=1.0)
        with pytest.raises(ValueError):
            OpKuramotoScheduler(rng=RandPool(seed=1), explore_floor=-0.1)

    def test_repeated_lockin_seed_no_longer_starves_true_best(self):
        """End-to-end regression for the exact bandit_env.py finding: seed 1
        under StationaryBernoulli locked OpKuramotoScheduler onto a p=0.05
        base-rate arm for 19945/20000 picks pre-fix, starving the true
        p=0.30 best arm entirely after early exploration. Post-fix the best
        arm must keep receiving a non-trivial share of the tail."""
        import random as _random

        best, runner_up, base = "best", "runner_up", "base"
        arms = [best, runner_up, base, "d", "e"]
        probs = {best: 0.30, runner_up: 0.18, base: 0.05, "d": 0.05, "e": 0.05}
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
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
        # Old formula gave this seed a tail share of ~0.0 for the true best
        # arm; the floor should keep it from being starved out completely.
        assert tail_picks / (rounds - tail_start) > 0.0


class TestDiagnostics:
    def test_empty_scheduler(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        d = sched.diagnostics()
        assert d["n_arms"] == 0
        assert d["r"] == 0.0
        assert d["critical_coupling"] == float("inf")
        assert d["phases"] == {}

    def test_reports_all_tracked_arms(self):
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True)
        sched.record("b", success=False)
        d = sched.diagnostics()
        assert d["n_arms"] == 2
        assert set(d["phases"]) == {"a", "b"}
        assert d["rates"]["a"] == 1.0
        assert d["rates"]["b"] == 0.0

    def test_no_cycles_gives_infinite_critical_coupling(self):
        """A single one-shot a->b edge has spectral radius 0 (no cycle),
        which core/kuramoto.py's critical_coupling defines as inf -- see
        that function's docstring."""
        sched = OpKuramotoScheduler(rng=RandPool(seed=1))
        sched.record("a", success=True)
        sched.record("b", success=True)
        d = sched.diagnostics()
        assert d["spectral_radius"] == pytest.approx(0.0)
        assert d["critical_coupling"] == float("inf")


class TestSimulateIntegration:
    def test_all_to_all_identical_rate_synchronizes(self):
        """Sanity-checks this scheduler's own batched stepping against the
        same qualitative result core/kuramoto.py's test_kuramoto.py proves
        for the raw ODE: strong coupling + tight frequency spread should
        raise coherence over time, not leave it flat."""
        sched = OpKuramotoScheduler(
            rng=RandPool(seed=11), k=8.0, recompute_batch=1, steps_per_batch=10
        )
        ops = [f"op{i}" for i in range(6)]
        for op in ops:
            sched.init_arm(op)
        # Give every op the same success rate (tight omega spread) and a
        # fully-connected transition graph (strong coupling structure).
        sched.successes = {op: 8.0 for op in ops}
        sched.attempts = {op: 10.0 for op in ops}
        sched.transition_counts = {
            src: {dst: 1 for dst in ops if dst != src} for src in ops
        }
        r_before, _ = sched.diagnostics()["r"], None
        for _ in range(30):
            sched._pending = sched.recompute_batch
            sched._maybe_advance(ops)
        r_after = sched.diagnostics()["r"]
        assert r_after > r_before
