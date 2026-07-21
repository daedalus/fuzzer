"""Tests for SHM coverage data structures and optimizations."""

import pytest
import tempfile
import os
import random

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.adapters.shm import ShmCoverage


class TestTemporalTracking:
    """Tests for temporal edge coverage tracking."""

    def test_edge_lifetimes(self):
        et = EdgeTracker(map_size=1024)
        et.record_edge_lifetimes({1, 2, 3}, exec_count=10)
        et.record_edge_lifetimes({1, 2, 3, 4}, exec_count=20)

        assert et._edge_first_seen[1] == 10
        assert et._edge_first_seen[4] == 20
        assert et._edge_last_seen[1] == 20
        assert et._edge_last_seen[4] == 20

    def test_edge_lifetime_stats(self):
        et = EdgeTracker(map_size=1024)
        et.record_edge_lifetimes({1, 2}, exec_count=10)
        et.record_edge_lifetimes({1, 2, 3}, exec_count=20)
        et.record_edge_lifetimes({1, 2, 3, 4}, exec_count=30)

        stats = et.edge_lifetime_stats()
        assert stats["median"] > 0
        assert stats["mean"] > 0
        assert stats["max"] > 0

    def test_coverage_timeline(self):
        et = EdgeTracker(map_size=1024)
        et.record_coverage_snapshot(100)
        et.record_coverage_snapshot(200)
        et.record_coverage_snapshot(300)

        assert len(et._coverage_timeline) == 3
        assert et._coverage_timeline[0] == (100, 0)
        assert et._coverage_timeline[1] == (200, 0)

    def test_coverage_growth_model(self):
        et = EdgeTracker(map_size=1024)
        # Simulate coverage growth
        for i in range(10):
            exec_count = (i + 1) * 100
            edges = min(50, int(30 * (1 - 0.7 ** (i + 1))))
            et._coverage_timeline.append((exec_count, edges))

        model = et.coverage_growth_model()
        assert model["confidence"] > 0
        assert model["projected_total"] > 0

    def test_edge_age_distribution(self):
        et = EdgeTracker(map_size=1024)
        for i in range(20):
            et._edge_first_seen[i] = i * 10

        dist = et.edge_age_distribution()
        assert dist["new"] + dist["mature"] + dist["old"] == 20


class TestBranchCorrelation:
    """Tests for branch correlation matrix."""

    def test_update_correlation(self):
        et = EdgeTracker(map_size=1024)
        et.update_correlation({1, 2, 3})
        et.update_correlation({1, 2, 4})
        et.update_correlation({1, 3, 5})

        assert et._correlation_total == 3
        assert len(et._correlation_matrix) > 0

    def test_branch_correlation(self):
        et = EdgeTracker(map_size=1024)
        et.update_correlation({1, 2, 3})
        et.update_correlation({1, 2, 4})

        corr_12 = et.branch_correlation(1, 2)
        corr_13 = et.branch_correlation(1, 3)
        assert corr_12 > corr_13  # 1,2 co-occur more

    def test_top_correlated_pairs(self):
        et = EdgeTracker(map_size=1024)
        et.update_correlation({1, 2, 3})
        et.update_correlation({1, 2, 4})
        et.update_correlation({1, 3, 5})

        top = et.top_correlated_pairs(k=3)
        assert len(top) <= 3
        assert top[0][2] >= top[-1][2]  # Sorted by correlation


class TestSeedClassification:
    """Tests for seed classification and dominance tree."""

    def test_classify_seeds(self):
        et = EdgeTracker(map_size=1024)
        et.seed_edges = {
            "seed_a": {1, 2, 3},
            "seed_b": {1, 2},
            "seed_c": {4, 5, 6},
        }
        et.cumulative_edges = {1, 2, 3, 4, 5, 6}

        # Set up MinHash
        for k, edges in et.seed_edges.items():
            sig = et._minhash.compute_signature(edges)
            et._minhash.add(k, sig)

        classifications = et.classify_seeds()
        assert "seed_a" in classifications
        assert classifications["seed_a"]["classification"] in ["keystone", "useful"]

    def test_coverage_dominance_tree(self):
        et = EdgeTracker(map_size=1024)
        et.seed_edges = {
            "seed_a": {1, 2},
            "seed_b": {1, 2, 3, 4},
        }

        # Set up MinHash
        for k, edges in et.seed_edges.items():
            sig = et._minhash.compute_signature(edges)
            et._minhash.add(k, sig)

        tree = et.coverage_dominance_tree()
        assert "seed_b" in tree
        assert "seed_a" in tree["seed_b"]

    def test_find_redundant_seeds(self):
        et = EdgeTracker(map_size=1024)
        et.seed_edges = {
            "seed_a": {1, 2},
            "seed_b": {1, 2, 3, 4},
        }

        # Set up MinHash
        for k, edges in et.seed_edges.items():
            sig = et._minhash.compute_signature(edges)
            et._minhash.add(k, sig)

        redundant = et.find_redundant_seeds()
        assert "seed_a" in redundant


class TestSHMCoverage:
    """Tests for SHM coverage optimizations."""

    def test_shm_cumulative_edges_correct(self):
        shm = ShmCoverage(size=4096)
        assert shm.cumulative_edges == 0

        # Set some edges
        for i in range(100):
            shm._map[i] = 1

        result = shm.is_new_coverage()
        assert result is True
        assert shm.cumulative_edges == 100

    def test_shm_same_edges_no_change(self):
        shm = ShmCoverage(size=4096)

        # First execution
        for i in range(100):
            shm._map[i] = 1
        shm.is_new_coverage()
        assert shm.cumulative_edges == 100

        # Same edges again
        shm.reset_edge_map()
        for i in range(100):
            shm._map[i] = 1
        result = shm.is_new_coverage()
        assert result is False
        assert shm.cumulative_edges == 100  # No change

    def test_shm_new_edges_increment(self):
        shm = ShmCoverage(size=4096)

        # First execution
        for i in range(100):
            shm._map[i] = 1
        shm.is_new_coverage()
        assert shm.cumulative_edges == 100

        # New edges
        shm.reset_edge_map()
        for i in range(100, 200):
            shm._map[i] = 1
        result = shm.is_new_coverage()
        assert result is True
        assert shm.cumulative_edges == 200

    def test_shm_record_edge(self):
        shm = ShmCoverage(size=4096)
        result = shm.record_edge(42)
        assert result is True
        assert shm.cumulative_edges == 1

        # Same edge again
        result = shm.record_edge(42)
        assert result is False
        assert shm.cumulative_edges == 1


class TestCrashETA:
    """Tests for crash_eta optimizations."""

    def test_record_crashes_only(self):
        from fuzzer_tool.core.crash_eta import CrashMITracker

        tracker = CrashMITracker()

        # Non-crash should not track positions
        tracker.total_execs = 0
        tracker.record(b"test input", is_crash=False)
        assert tracker.total_execs == 1
        assert len(tracker.position_counts) == 0  # Not tracked for non-crashes

        # Crash should track positions
        tracker.record(b"crash input", is_crash=True)
        assert tracker.total_execs == 2
        assert len(tracker.position_counts) > 0

    def test_weighted_position_caching(self):
        from fuzzer_tool.core.crash_eta import CrashMITracker

        tracker = CrashMITracker()
        tracker.load(
            {
                "position_counts": {str(i): 100 for i in range(100)},
                "byte_total": {},
                "joint_crash": {},
                "total_execs": 10000,
                "total_crashes": 100,
            }
        )

        # Warm up cache
        tracker.all_mi()

        # Should use cache
        pos = tracker.weighted_position(100)
        assert 0 <= pos < 100


class TestSeedPicker:
    """Tests for seed_picker optimizations."""

    def test_pareto_front_caching(self):
        from fuzzer_tool.services.seed_picker import SeedPicker

        # Create a minimal mock fuzzer
        class MockFuzzer:
            corpus = [b"seed1", b"seed2", b"seed3"]
            seed_meta = {
                b"seed1": {"fuzz_count": 10, "coverage_edges": 5, "added_at": 1000},
                b"seed2": {"fuzz_count": 5, "coverage_edges": 3, "added_at": 2000},
                b"seed3": {"fuzz_count": 20, "coverage_edges": 10, "added_at": 500},
            }
            _temperature = 1.0
            _cached_weights = {}
            exec_count = 100

            def _seed_key(self, data):
                return str(hash(data))

        f = MockFuzzer()
        picker = SeedPicker(f)

        # Test Pareto front computation
        scores = [(1.0, 2.0, 3.0), (2.0, 1.0, 2.0), (1.5, 1.5, 2.5)]
        front = picker._pareto_front(scores)
        assert len(front) > 0


class TestFastJSON:
    """Tests for fast_json module."""

    def test_loads_dumps(self):
        from fuzzer_tool.core.fast_json import loads, dumps

        data = {"key": "value", "num": 42, "nested": [1, 2, 3]}
        serialized = dumps(data)
        deserialized = loads(serialized)
        assert deserialized == data

    def test_json_decode_error(self):
        from fuzzer_tool.core.fast_json import loads, JSONDecodeError

        with pytest.raises(JSONDecodeError):
            loads("invalid json")
