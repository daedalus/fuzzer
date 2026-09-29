"""Unit tests for the German tank problem estimator."""

import pytest

from fuzzer_tool.core.rand_pool import RandPool


class TestGermanTankEstimator:
    """Tests for german_tank_estimate() method on EdgeTracker."""

    def _make_tracker_with_edges(self, edge_ids: set[int]):
        """Create an EdgeTracker with a specific set of edges."""
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=65536)
        for edge_id in edge_ids:
            tracker.cumulative_edges.add(edge_id)
        return tracker

    def test_empty_tracker(self):
        """Empty tracker should return zero estimate."""
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=65536)
        result = tracker.german_tank_estimate()
        assert result["estimate"] == 0
        assert result["sample_size"] == 0
        assert result["efficiency"] == 0.0
        assert result["confidence"] == "low"

    def test_single_edge(self):
        """Single edge should return that edge as max."""
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=65536)
        tracker.cumulative_edges.add(42)
        result = tracker.german_tank_estimate()
        assert result["observed_max"] == 42
        assert result["sample_size"] == 1
        # For single edge: M=42, m=42, k=1 -> N̂ = 42 + (42-42)/1 - 1 = 41
        assert result["estimate"] == 41.0
        # Efficiency = k/estimate = 1/41 ≈ 0.024, capped at 1.0
        assert abs(result["efficiency"] - (1 / 41)) < 0.001

    def test_two_edges(self):
        """Two edges should use the unbiased formula."""
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=65536)
        tracker.cumulative_edges.add(10)
        tracker.cumulative_edges.add(50)
        result = tracker.german_tank_estimate()
        assert result["observed_max"] == 50
        assert result["observed_min"] == 10
        assert result["sample_size"] == 2
        # N̂ = M + (M - m)/k - 1 = 50 + (50-10)/2 - 1 = 50 + 20 - 1 = 69
        assert result["estimate"] == 69.0
        # Efficiency = k/estimate = 2/69 ≈ 0.029
        assert abs(result["efficiency"] - (2 / 69)) < 0.001

    def test_uniform_distribution_unbiased(self):
        """Verify estimator is unbiased under uniform distribution."""

        rng = RandPool(seed=42)
        bounds = (0, 1000)

        # Generate uniform random edges
        edge_ids = set(rng.randint(bounds[0], bounds[1]) for _ in range(100))

        tracker = self._make_tracker_with_edges(edge_ids)
        result = tracker.german_tank_estimate()

        # Estimate should be close to upper bound for uniform distribution
        # For k=100 from N=1000, expected max is ~990, estimate ~990 * 1.01 ≈ 1000
        assert result["estimate"] > 900  # Should be in the ballpark
        assert result["estimate"] < 1100  # Not wildly off

    def test_efficiency_capping(self):
        """Test that efficiency is capped at 1.0 for edge cases."""
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=65536)
        # Add edges 0..49 (50 edges)
        for i in range(50):
            tracker.cumulative_edges.add(i)
        result = tracker.german_tank_estimate()
        # M=49, m=0, k=50 -> N̂ = 49 + (49-0)/50 - 1 = 49 + 0.98 - 1 = 48.98
        assert abs(result["estimate"] - 48.98) < 0.01
        # k/estimate = 50/48.98 ≈ 1.02, but should be capped at 1.0
        assert result["efficiency"] == 1.0

    def test_confidence_levels(self):
        """Confidence should be based on sample size."""

        # Low confidence: < 10 edges
        tracker = self._make_tracker_with_edges(set(range(5)))
        assert tracker.german_tank_estimate()["confidence"] == "low"

        # Medium confidence: 10-99 edges
        tracker = self._make_tracker_with_edges(set(range(50)))
        assert tracker.german_tank_estimate()["confidence"] == "medium"

        # High confidence: >= 100 edges
        tracker = self._make_tracker_with_edges(set(range(100)))
        assert tracker.german_tank_estimate()["confidence"] == "high"

    def test_coverage_growth_model_includes_gt(self):
        """coverage_growth_model should include German tank fields."""
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=65536)
        for i in range(50):
            tracker.cumulative_edges.add(i)
            tracker._coverage_execs.append(i * 100)
            tracker._coverage_edges.append(i)

        growth = tracker.coverage_growth_model()
        assert "german_tank" in growth
        assert "german_tank_efficiency" in growth
        assert growth["german_tank"] > 0
        assert 0 <= growth["german_tank_efficiency"] <= 1

    def test_bayesian_coverage_growth_model_fallback(self):
        """bayesian_coverage_growth_model should include GT when falling back."""
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=65536)
        # Add only 3 timeline points (insufficient for Bayesian fit)
        for i in range(3):
            tracker.cumulative_edges.add(i * 10)
            tracker._coverage_execs.append(i * 100)
            tracker._coverage_edges.append(i * 10)

        bayes = tracker.bayesian_coverage_growth_model()
        # Should fall back to frequentist model which includes GT
        assert "german_tank" in bayes
        assert "german_tank_efficiency" in bayes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
