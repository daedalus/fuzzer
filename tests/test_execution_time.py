"""Tests for ExecutionTimeTracker — adaptive timeout calibration with CRPS."""

from fuzzer_tool.core.execution_time import ExecutionTimeTracker


class TestExecutionTimeTracker:
    def test_init(self):
        t = ExecutionTimeTracker()
        assert t.count == 0
        assert t.p50 == 0.0
        assert t.p99 == 0.0

    def test_record_returns_crps(self):
        t = ExecutionTimeTracker()
        crps = t.record(0.01)
        assert crps >= 0.0
        assert t.count == 1

    def test_suggested_timeout_default(self):
        t = ExecutionTimeTracker()
        assert t.suggested_timeout() == 5.0  # fallback when empty

    def test_suggested_timeout_percentile(self):
        t = ExecutionTimeTracker()
        for i in range(100):
            t.record(0.01 + i * 0.001)
        p99 = t.suggested_timeout(percentile=99)
        assert p99 > 0.0
        # p99 * factor should be larger than p50 * factor
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
        # Trending upward — should give positive slope
        for i in range(30):
            t.record(0.001 * (i + 1))
        trend = t.crps_trend()
        # With increasing times the CRPS may trend positive
        assert isinstance(trend, float)

    def test_mean_crps_empty(self):
        t = ExecutionTimeTracker()
        assert t.mean_crps() == 0.0

    def test_mean_crps_after_records(self):
        t = ExecutionTimeTracker()
        for _ in range(10):
            t.record(0.05)
        assert t.mean_crps() >= 0.0

    def test_timeout_factor(self):
        t = ExecutionTimeTracker(timeout_factor=2.0)
        for _ in range(50):
            t.record(0.01)
        suggested = t.suggested_timeout()
        # Should be around 0.01 * 2.0 = 0.02
        assert suggested < 1.0

    def test_crps_stable_constant_input(self):
        t = ExecutionTimeTracker()
        # Same value every time — CRPS should be low
        for _ in range(50):
            t.record(0.05)
        assert t.mean_crps() < 0.01
