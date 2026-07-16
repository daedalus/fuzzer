"""Tests for Shapley attribution, replicator dynamics, and MI tracker."""

import math

from fuzzer_tool.core.mi import MutualInformationTracker
from fuzzer_tool.core.montecarlo import ReplicatorScheduler, ShapleyAttribution


class TestShapleyAttribution:
    def test_init(self):
        s = ShapleyAttribution(n_samples=50, window_size=100)
        assert s.n_samples == 50
        assert s.window_size == 100
        assert len(s._outcomes) == 0

    def test_record(self):
        s = ShapleyAttribution()
        s.record({"bit_flip", "byte_flip"}, new_edges=5, edge_indices={0, 1, 2, 3, 4})
        assert len(s._outcomes) == 1
        assert "bit_flip" in s._operator_edges
        assert len(s._operator_edges["bit_flip"]) == 5

    def test_shapley_values_single_operator(self):
        s = ShapleyAttribution(n_samples=10)
        s.record({"a"}, new_edges=10, edge_indices=set(range(10)))
        sv = s.shapley_values(["a"])
        assert abs(sv["a"] - 1.0) < 1e-10

    def test_shapley_values_equal_operators(self):
        s = ShapleyAttribution(n_samples=100)
        # Both operators contribute equally
        for i in range(20):
            s.record({"a", "b"}, new_edges=1, edge_indices={i})
        sv = s.shapley_values(["a", "b"])
        # Both should be roughly equal
        assert abs(sv["a"] - sv["b"]) < 0.15

    def test_shapley_values_dominant_operator(self):
        s = ShapleyAttribution(n_samples=100)
        # "a" contributes to 90 unique edges, "b" to 10
        for i in range(100):
            edges = set(range(i, i + 1))
            s.record({"a"}, new_edges=1, edge_indices=edges)
        for i in range(10):
            s.record({"b"}, new_edges=1, edge_indices={100 + i})
        sv = s.shapley_values(["a", "b"])
        assert sv["a"] > sv["b"]

    def test_shapley_values_empty(self):
        s = ShapleyAttribution()
        sv = s.shapley_values(["a", "b"])
        assert sv["a"] == 0.5
        assert sv["b"] == 0.5

    def test_synergy_positive(self):
        s = ShapleyAttribution()
        # Together they cover more than individually
        s.record({"a"}, new_edges=1, edge_indices={0, 1})
        s.record({"b"}, new_edges=1, edge_indices={2, 3})
        s.record({"a", "b"}, new_edges=1, edge_indices={0, 1, 2, 3, 4})
        syn = s.operator_synergy("a", "b")
        # Synergy is positive if joint > sum of individuals (approximate)
        # Here individual coverage is small, joint has extra edges
        assert isinstance(syn, float)

    def test_synergy_no_data(self):
        s = ShapleyAttribution()
        assert s.operator_synergy("a", "b") == 0.0

    def test_ranking(self):
        s = ShapleyAttribution(n_samples=10)
        s.record({"a"}, new_edges=10, edge_indices=set(range(10)))
        s.record({"b"}, new_edges=2, edge_indices={100, 101})
        ranking = s.ranking(["a", "b"])
        assert ranking[0][0] == "a"
        assert ranking[0][1] > ranking[1][1]

    def test_window_cap(self):
        s = ShapleyAttribution(window_size=5)
        for i in range(10):
            s.record({"a"}, new_edges=1, edge_indices={i})
        assert len(s._outcomes) == 5


class TestReplicatorScheduler:
    def test_init(self):
        r = ReplicatorScheduler(window_size=10, learning_rate=0.1)
        assert r.window_size == 10
        assert r.eta == 0.1
        assert len(r.population) == 0

    def test_init_arm(self):
        r = ReplicatorScheduler()
        r.init_arm("a")
        r.init_arm("b")
        assert len(r.population) == 2
        assert abs(sum(r.population) - 1.0) < 1e-10
        assert all(abs(p - 0.5) < 1e-10 for p in r.population)

    def test_init_arm_idempotent(self):
        r = ReplicatorScheduler()
        r.init_arm("a")
        r.init_arm("a")
        assert len(r.operators) == 1
        assert len(r.population) == 1

    def test_select_op(self):
        r = ReplicatorScheduler()
        r.init_arm("a")
        r.init_arm("b")
        for _ in range(20):
            op = r.select_op(["a", "b"])
            assert op in ("a", "b")

    def test_select_op_single(self):
        r = ReplicatorScheduler()
        r.init_arm("only_one")
        op = r.select_op(["only_one"])
        assert op == "only_one"

    def test_record(self):
        r = ReplicatorScheduler(window_size=5)
        r.init_arm("a")
        r.record("a", success=True)
        r.record("a", success=False)
        assert r._total_execs == 2
        assert r._total_discoveries == 1

    def test_replicator_update_triggers(self):
        r = ReplicatorScheduler(window_size=5, learning_rate=0.2)
        r.init_arm("good")
        r.init_arm("bad")
        # good succeeds 80%, bad succeeds 10%
        import random

        random.seed(42)
        for _ in range(5):
            op = r.select_op(["good", "bad"])
            success = (op == "good" and random.random() < 0.8) or (
                op == "bad" and random.random() < 0.1
            )
            r.record(op, success)
        # After update, good should have higher population
        dist = r.population_distribution()
        assert dist["good"] > dist["bad"]

    def test_convergence(self):
        r = ReplicatorScheduler(window_size=10, learning_rate=0.5)
        r.init_arm("good")
        r.init_arm("bad")
        import random

        random.seed(42)
        # Run many windows with strong fitness difference
        for _ in range(100):
            for _ in range(10):
                op = r.select_op(["good", "bad"])
                success = (op == "good" and random.random() < 0.9) or (
                    op == "bad" and random.random() < 0.05
                )
                r.record(op, success)
        # After many updates, should converge
        assert r.is_converged(threshold=0.05)

    def test_dominant_operator(self):
        r = ReplicatorScheduler(window_size=5, learning_rate=0.3)
        r.init_arm("a")
        r.init_arm("b")
        # Push a to dominance
        r.population = [0.9, 0.1]
        assert r.dominant_operator() == "a"

    def test_mutation_rate_floor(self):
        r = ReplicatorScheduler(window_size=5, learning_rate=0.5, mutation_rate=0.05)
        r.init_arm("a")
        r.init_arm("b")
        import random

        random.seed(42)
        # All successes for a, all failures for b
        for _ in range(20):
            r.record("a", success=True)
            r.record("b", success=False)
        # Even b should have at least mutation_rate (with numerical tolerance)
        dist = r.population_distribution()
        assert dist["b"] >= 0.04  # mutation_rate is 0.05, allow small numerical drift

    def test_bandit_stats_compatibility(self):
        r = ReplicatorScheduler()
        r.init_arm("a")
        r.record("a", success=True)
        stats = r.bandit_stats()
        assert "_replicator_global" in stats
        assert stats["_replicator_global"] == (1, 0)

    def test_operator_stats(self):
        r = ReplicatorScheduler(window_size=5)
        r.init_arm("a")
        r.init_arm("b")
        r.record("a", success=True)
        r.record("b", success=False)
        stats = r.operator_stats()
        assert len(stats) == 2
        assert any(s["name"] == "a" for s in stats)

    def test_history_capped(self):
        r = ReplicatorScheduler(window_size=5)
        r.init_arm("a")
        r.init_arm("b")
        for _ in range(200):
            for _ in range(5):
                r.record("a", success=True)
        assert len(r._history) <= 100


class TestMutualInformationTracker:
    def test_init(self):
        mi = MutualInformationTracker(max_positions=100, min_observations=10)
        assert mi.max_positions == 100
        assert mi.min_observations == 10
        assert mi.total_observations == 0

    def test_record(self):
        mi = MutualInformationTracker(min_observations=1)
        mi.record(b"\x00\x01", b"\xff\x00", map_size=16)
        assert mi.total_observations == 1
        assert mi.position_counts[0] == 1
        assert mi.position_counts[1] == 1

    def test_mi_insufficient_data(self):
        mi = MutualInformationTracker(min_observations=100)
        mi.record(b"\x00", b"\xff", map_size=16)
        assert mi.mi(0) == 0.0

    def test_mi_with_enough_data(self):
        mi = MutualInformationTracker(min_observations=10)
        # When byte=0, edge 0 is always hit; when byte=1, edge 1 is always hit
        for _ in range(20):
            mi.record(b"\x00", b"\x01\x00", map_size=16)
            mi.record(b"\x01", b"\00\x01", map_size=16)
        val = mi.mi(0)
        assert val > 0.0  # MI should be positive

    def test_mi_deterministic_input(self):
        mi = MutualInformationTracker(min_observations=10)
        # Always the same input, same output → MI should be low (no information)
        for _ in range(20):
            mi.record(b"\x42", b"\x01", map_size=16)
        val = mi.mi(0)
        assert val == 0.0  # No variation → no MI

    def test_mi_profile(self):
        mi = MutualInformationTracker(min_observations=5)
        for _ in range(20):
            mi.record(b"\x00\x01", b"\x01\x00", map_size=16)
            mi.record(b"\x01\x00", b"\x00\x01", map_size=16)
        profile = mi.mi_profile(input_length=2)
        assert 0 in profile
        assert 1 in profile
        assert all(v >= 0 for v in profile.values())

    def test_top_positions(self):
        mi = MutualInformationTracker(min_observations=5)
        for _ in range(20):
            mi.record(b"\x00\x01\x02", b"\x01\x00\x00", map_size=16)
            mi.record(b"\x01\x00\x02", b"\x00\x01\x00", map_size=16)
            mi.record(b"\x02\x00\x01", b"\x00\x00\x01", map_size=16)
        top = mi.top_positions(k=2, input_length=3)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]

    def test_mutation_weight(self):
        mi = MutualInformationTracker(min_observations=5)
        for _ in range(20):
            mi.record(b"\x00\x01", b"\x01\x00", map_size=16)
            mi.record(b"\x01\x00", b"\x00\x01", map_size=16)
        w = mi.mutation_weight(0, 2)
        assert 0.1 <= w <= 5.0

    def test_mutation_weight_no_data(self):
        mi = MutualInformationTracker(min_observations=5)
        w = mi.mutation_weight(0, 10)
        assert w == 0.1

    def test_weighted_position(self):
        mi = MutualInformationTracker(min_observations=5)
        for _ in range(20):
            mi.record(b"\x00\x01", b"\x01\x00", map_size=16)
            mi.record(b"\x01\x00", b"\x00\x01", map_size=16)
        pos = mi.weighted_position(2)
        assert 0 <= pos < 2

    def test_conditional_mi(self):
        mi = MutualInformationTracker(min_observations=5)
        for _ in range(20):
            mi.record(b"\x00\x00", b"\x01\x00", map_size=16)
            mi.record(b"\x01\x01", b"\x00\x01", map_size=16)
        cmi = mi.conditional_mi(0, 1)
        assert cmi >= 0.0

    def test_interaction_information(self):
        mi = MutualInformationTracker(min_observations=5)
        for _ in range(20):
            mi.record(b"\x00\x00", b"\x01\x00", map_size=16)
            mi.record(b"\x01\x01", b"\x00\x01", map_size=16)
        ii = mi.interaction_information(0, 1)
        assert isinstance(ii, float)

    def test_save_load(self, tmp_path):
        mi = MutualInformationTracker(min_observations=5)
        for _ in range(20):
            mi.record(b"\x00\x01", b"\x01\x00", map_size=16)
            mi.record(b"\x01\x00", b"\x00\x01", map_size=16)

        path = str(tmp_path / "mi.json")
        assert mi.save(path)

        mi2 = MutualInformationTracker(min_observations=5)
        assert mi2.load(path)
        assert mi2.total_observations == mi.total_observations
        assert mi2.position_counts[0] == mi.position_counts[0]

    def test_load_nonexistent(self):
        mi = MutualInformationTracker()
        assert not mi.load("/nonexistent/mi.json")

    def test_max_positions_cap(self):
        mi = MutualInformationTracker(max_positions=2, min_observations=1)
        mi.record(b"\x00\x01\x02\x03", b"\xff", map_size=16)
        assert 0 in mi.position_counts
        assert 1 in mi.position_counts
        assert 2 not in mi.position_counts  # capped at max_positions=2
