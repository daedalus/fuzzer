"""Tests for OpFireflyScheduler: Firefly Algorithm over operator
probability distributions (FA-Fuzz, IEEE 2023, document/10305545).
"""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_firefly import OpFireflyScheduler


def _make(n_fireflies=5, window_size=20, seed=1, **kw):
    sched = OpFireflyScheduler(
        n_fireflies=n_fireflies, window_size=window_size, rng=RandPool(seed), **kw
    )
    return sched


class TestInitArm:
    def test_registers_operators_in_order(self):
        s = _make()
        s.init_arm("a")
        s.init_arm("b")
        assert s.operators == ["a", "b"]
        assert s.op_index == {"a": 0, "b": 1}

    def test_duplicate_registration_is_noop(self):
        s = _make()
        s.init_arm("a")
        s.init_arm("a")
        assert s.operators == ["a"]

    def test_builds_n_fireflies(self):
        s = _make(n_fireflies=3)
        s.init_arm("a")
        s.init_arm("b")
        assert len(s.fireflies) == 3

    def test_new_operator_extends_existing_fireflies(self):
        s = _make(n_fireflies=2)
        s.init_arm("a")
        s.init_arm("b")
        assert all(len(fly.pos) == 2 for fly in s.fireflies)
        s.init_arm("c")
        assert all(len(fly.pos) == 3 for fly in s.fireflies)
        # Same fireflies (by name), not rebuilt from scratch.
        assert {fly.name for fly in s.fireflies} == {"f0", "f1"}


class TestInitialJitter:
    def test_fireflies_do_not_start_identical(self):
        """The exact bug op_mopt.py documents: identical starting positions
        give every pairwise (x_j - x_i) term zero regardless of brightness,
        so the swarm can never move. Fireflies must be jittered apart.
        """
        s = _make(n_fireflies=5)
        for op in ("a", "b", "c", "d"):
            s.init_arm(op)
        positions = [tuple(fly.pos) for fly in s.fireflies]
        assert len(set(positions)) > 1

    def test_each_firefly_position_is_a_distribution(self):
        s = _make(n_fireflies=4)
        for op in ("a", "b", "c"):
            s.init_arm(op)
        for fly in s.fireflies:
            assert abs(sum(fly.pos) - 1.0) < 1e-9
            assert all(x >= 0.0 for x in fly.pos)


class TestSelectOp:
    def test_empty_operators_returns_first_available(self):
        s = _make()
        op, fid = s.select_op(["x", "y"])
        assert op == "x"

    def test_returns_registered_operator(self):
        s = _make()
        for op in ("a", "b", "c"):
            s.init_arm(op)
        op, fid = s.select_op(["a", "b", "c"])
        assert op in ("a", "b", "c")
        assert 0 <= fid < len(s.fireflies)

    def test_only_offers_available_ops(self):
        s = _make(n_fireflies=3)
        for op in ("a", "b", "c", "d"):
            s.init_arm(op)
        seen = set()
        for _ in range(200):
            op, _ = s.select_op(["a", "c"])
            seen.add(op)
        assert seen <= {"a", "c"}


class TestRecordAndBrightness:
    def test_record_attributes_to_named_firefly_only(self):
        s = _make(n_fireflies=3, window_size=1000)
        for op in ("a", "b"):
            s.init_arm(op)
        s.record("a", True, firefly_id=1, weight=1.0)
        assert s.fireflies[1].execs_in_window == 1
        assert s.fireflies[0].execs_in_window == 0
        assert s.fireflies[2].execs_in_window == 0

    def test_record_without_firefly_id_broadcasts(self):
        s = _make(n_fireflies=3, window_size=1000)
        for op in ("a", "b"):
            s.init_arm(op)
        s.record("a", True, firefly_id=None)
        assert all(fly.execs_in_window == 1 for fly in s.fireflies)

    def test_window_boundary_triggers_update_and_resets_window(self):
        s = _make(n_fireflies=3, window_size=5)
        for op in ("a", "b", "c"):
            s.init_arm(op)
        for i in range(5):
            op, fid = s.select_op(["a", "b", "c"])
            s.record(op, success=(i % 2 == 0), firefly_id=fid)
        assert all(fly.execs_in_window == 0 for fly in s.fireflies)

    def test_fitness_held_not_reset_when_window_empty(self):
        """The held-fitness fix documented in the module: a firefly that
        gets zero executions in a window keeps its prior fitness instead
        of collapsing to zero, which would starve it permanently under
        fitness-proportional selection.
        """
        s = _make(n_fireflies=2, window_size=10)
        for op in ("a",):
            s.init_arm(op)
        fly = s.fireflies[0]
        fly.execs_in_window = 4
        fly.discoveries.extend([1.0, 1.0, 0.0, 0.0])
        s._update_fitness(fly)
        assert fly.fitness == 0.5
        # Now simulate an empty window for the same firefly.
        fly.execs_in_window = 0
        fly.discoveries.clear()
        s._update_fitness(fly)
        assert fly.fitness == 0.5  # unchanged, not reset to 0.0


class TestMovement:
    def test_brighter_firefly_attracts_dimmer_one(self):
        """Two fireflies, far apart on the simplex. Firefly 0 is made
        artificially bright and firefly 1 dark; after an update round,
        firefly 1 should have moved measurably closer to firefly 0's
        original position (beyond what pure alpha noise alone would
        explain across many trials).
        """
        s = _make(n_fireflies=2, window_size=1, alpha=0.0, gamma=0.1)
        for op in ("a", "b", "c", "d"):
            s.init_arm(op)
        s.fireflies[0].pos = [0.9, 0.05, 0.03, 0.02]
        s.fireflies[1].pos = [0.02, 0.03, 0.05, 0.9]
        s.fireflies[0].execs_in_window = 10
        s.fireflies[0].discoveries.extend([1.0] * 10)
        s.fireflies[1].execs_in_window = 10
        s.fireflies[1].discoveries.extend([0.0] * 10)

        before = list(s.fireflies[1].pos)
        s._firefly_update()
        after = s.fireflies[1].pos

        def dist(p, q):
            return sum((a - b) ** 2 for a, b in zip(p, q, strict=False)) ** 0.5

        target = [0.9, 0.05, 0.03, 0.02]
        assert dist(after, target) < dist(before, target)

    def test_brightest_firefly_only_gets_random_walk(self):
        """With alpha=0, the single brightest firefly (no brighter
        neighbor) should not move at all, since the pairwise loop is a
        no-op for it and the random term is zeroed.
        """
        s = _make(n_fireflies=2, window_size=1, alpha=0.0)
        for op in ("a", "b", "c"):
            s.init_arm(op)
        s.fireflies[0].pos = [0.7, 0.2, 0.1]
        s.fireflies[1].pos = [0.1, 0.2, 0.7]
        s.fireflies[0].execs_in_window = 5
        s.fireflies[0].discoveries.extend([1.0] * 5)
        s.fireflies[1].execs_in_window = 5
        s.fireflies[1].discoveries.extend([0.0] * 5)

        before = list(s.fireflies[0].pos)
        s._firefly_update()
        after = s.fireflies[0].pos
        assert all(abs(a - b) < 1e-9 for a, b in zip(before, after, strict=False))

    def test_alpha_decays_after_each_update(self):
        s = _make(n_fireflies=2, window_size=1, alpha=0.2, alpha_decay=0.5)
        for op in ("a", "b"):
            s.init_arm(op)
        s._firefly_update()
        assert abs(s.alpha - 0.1) < 1e-9
        s._firefly_update()
        assert abs(s.alpha - 0.05) < 1e-9

    def test_positions_stay_on_simplex_after_update(self):
        s = _make(n_fireflies=4, window_size=1)
        for op in ("a", "b", "c", "d", "e"):
            s.init_arm(op)
        for fly in s.fireflies:
            fly.execs_in_window = 3
            fly.discoveries.extend([1.0, 0.0, 1.0])
        s._firefly_update()
        for fly in s.fireflies:
            assert abs(sum(fly.pos) - 1.0) < 1e-9
            assert all(x >= 0.0 for x in fly.pos)


class TestFloorRelativeToBest:
    def test_absolute_floor_would_starve_but_relative_floor_does_not(self):
        """Mirrors MOptScheduler.select_op's own regression case: once one
        firefly's fitness clears ~0.1, an absolute 0.001 floor stops being
        a floor at all -- the leader takes over 99% of draws. The
        relative floor (0.1 * best) keeps the runner-up selectable.
        """
        s = _make(n_fireflies=2, window_size=1000)
        s.init_arm("a")
        s.init_arm("b")
        s.fireflies[0].fitness = 0.9
        s.fireflies[1].fitness = 0.0
        counts = {0: 0, 1: 0}
        for _ in range(2000):
            _, fid = s.select_op(["a"])
            counts[fid] += 1
        # With an absolute 0.001 floor this would be ~0.1% of draws;
        # the relative floor (0.1 * 0.9 = 0.09) should give the dark
        # firefly a solidly double-digit share.
        assert counts[1] / 2000 > 0.05


class TestStatsAndCompat:
    def test_firefly_stats_reports_all_fireflies(self):
        s = _make(n_fireflies=3)
        for op in ("a", "b"):
            s.init_arm(op)
        stats = s.firefly_stats()
        assert len(stats) == 3
        for row in stats:
            assert "name" in row and "fitness" in row and "top_op" in row

    def test_bandit_stats_tracks_global_totals(self):
        s = _make(n_fireflies=2, window_size=1000)
        s.init_arm("a")
        s.record("a", True, firefly_id=0)
        s.record("a", False, firefly_id=0)
        disc, fail = s.bandit_stats()["_firefly_global"]
        assert disc == 1
        assert fail == 1

    def test_supports_priors_is_false(self):
        assert OpFireflyScheduler.supports_priors is False
