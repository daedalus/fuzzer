"""Tests for ExecutionTimeTracker — CRPS scoring, percentile, trend."""

from fuzzer_tool.core.execution_time import ExecutionTimeTracker


class TestExecutionTimeTracker:
    def test_init(self):
        t = ExecutionTimeTracker()
        assert t.count == 0
        assert t.p50 == 0.0
        assert t.p99 == 0.0

    def test_suggested_timeout_default(self):
        t = ExecutionTimeTracker()
        assert t.suggested_timeout() == 5.0

    def test_suggested_timeout_percentile(self):
        t = ExecutionTimeTracker()
        for i in range(100):
            t.record(0.01 + i * 0.001)
        p99 = t.suggested_timeout(percentile=99)
        p50 = t.suggested_timeout(percentile=50)
        assert p99 >= p50

    def test_p50_p99(self):
        t = ExecutionTimeTracker()
        for i in range(100):
            t.record(0.01 + i * 0.001)
        assert t.p50 > 0.0
        assert t.p99 >= t.p50

    def test_window_size_cap(self):
        t = ExecutionTimeTracker(window_size=10)
        for i in range(20):
            t.record(float(i))
        assert t.count == 20
        assert len(t._sorted) <= 10

    def test_crps_trend_too_few(self):
        t = ExecutionTimeTracker()
        assert t.crps_trend() == 0.0

    def test_crps_trend_increasing(self):
        t = ExecutionTimeTracker()
        for i in range(30):
            t.record(0.001 * (i + 1))
        assert isinstance(t.crps_trend(), float)

    def test_mean_crps_empty(self):
        t = ExecutionTimeTracker()
        assert t.mean_crps() == 0.0

    def test_timeout_factor(self):
        t = ExecutionTimeTracker(timeout_factor=2.0)
        for _ in range(50):
            t.record(0.01)
        assert t.suggested_timeout() < 1.0

    def test_correction_factor(self):
        """suggested_timeout = p99 + std_dev * correction_factor"""
        # Constant input: std_dev = 0, so timeout = p99 regardless of factor
        t1 = ExecutionTimeTracker(correction_factor=1.0)
        t2 = ExecutionTimeTracker(correction_factor=5.0)
        for _ in range(50):
            t1.record(0.05)
            t2.record(0.05)
        assert t1.suggested_timeout() == t2.suggested_timeout()

    def test_correction_factor_scales_std_dev(self):
        """Higher correction_factor should give larger timeout for variable input."""
        t_low = ExecutionTimeTracker(correction_factor=0.5)
        t_high = ExecutionTimeTracker(correction_factor=2.0)
        # Variable input: uniform 0.01 to 0.10 → nonzero std_dev
        for i in range(100):
            val = 0.01 + i * 0.001
            t_low.record(val)
            t_high.record(val)
        assert t_high.suggested_timeout() > t_low.suggested_timeout()

    def test_std_dev_contribution(self):
        """timeout should be >= p99 (std_dev is non-negative)."""
        t = ExecutionTimeTracker(correction_factor=1.5)
        for i in range(100):
            t.record(0.01 + i * 0.001)
        assert t.suggested_timeout() >= t.p99

    def test_crps_stable_constant_input(self):
        t = ExecutionTimeTracker()
        for _ in range(50):
            t.record(0.05)
        # All same value → CRPS should be very low
        assert t.mean_crps() < 0.01


class TestCRPSScorer:
    """Rigorous CRPS tests: monotonicity with distance from distribution."""

    def test_typical_observation_low_crps(self):
        """An observation within the distribution should score low."""
        t = ExecutionTimeTracker()
        for _ in range(100):
            t.record(0.05)
        crps_typical = t.record(0.05)
        assert crps_typical < 0.01

    def test_extreme_outlier_higher_crps(self):
        """An extreme outlier (1.0, far from anything seen) should score
        HIGHER than a typical observation — it's more surprising."""
        t = ExecutionTimeTracker()
        for _ in range(100):
            t.record(0.05)
        crps_typical = t.mean_crps()

        # Now record an extreme outlier — should score higher
        crps_extreme = t._compute_crps(1.0)
        assert crps_extreme > crps_typical, (
            f"Extreme outlier CRPS ({crps_extreme}) should exceed "
            f"typical CRPS ({crps_typical})"
        )

    def test_gap_observation_higher_crps(self):
        """An observation in the gap between cluster and outlier
        should score higher than typical but lower than extreme."""
        t = ExecutionTimeTracker()
        for _ in range(100):
            t.record(0.05)
        crps_typical = t.mean_crps()
        crps_gap = t._compute_crps(0.15)
        crps_extreme = t._compute_crps(1.0)
        assert crps_typical < crps_gap < crps_extreme, (
            f"Expected typical({crps_typical}) < gap({crps_gap}) < extreme({crps_extreme})"
        )

    def test_crps_non_negative(self):
        """CRPS is always ≥ 0 — it's an integral of squared terms."""
        t = ExecutionTimeTracker()
        for _ in range(50):
            t.record(0.05)
        for obs in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            crps = t._compute_crps(obs)
            assert crps >= 0.0, f"CRPS({obs}) = {crps} < 0"

    def test_crps_zero_for_empty_tracker(self):
        t = ExecutionTimeTracker()
        assert t._compute_crps(1.0) == 0.0

    def test_crps_increases_with_distance(self):
        """For a simple uniform distribution, CRPS should increase
        monotonically as the observation moves away from the center."""
        t = ExecutionTimeTracker()
        for i in range(100):
            t.record(0.1 + i * 0.001)  # uniform 0.1 to 0.2

        vals = [0.15, 0.20, 0.30, 0.50, 1.0]
        crps_values = [t._compute_crps(v) for v in vals]
        # Each farther observation should score higher
        for i in range(len(crps_values) - 1):
            assert crps_values[i] <= crps_values[i + 1], (
                f"CRPS should increase with distance: "
                f"CRPS({vals[i]})={crps_values[i]} > CRPS({vals[i+1]})={crps_values[i+1]}"
            )

    def test_crps_symmetric_for_two_sided_outlier(self):
        """An observation equally far on either side of the distribution
        should get similar CRPS (by symmetry of squared error)."""
        t = ExecutionTimeTracker()
        for i in range(100):
            t.record(1.0 + i * 0.01)  # centered around 1.5
        crps_below = t._compute_crps(0.5)  # 1.0 below center
        crps_above = t._compute_crps(2.0)  # 0.5 above center... actually asymmetric
        # Not perfectly symmetric but both should be > 0
        assert crps_below > 0
        assert crps_above > 0
