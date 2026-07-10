"""Tests for critical slowing down detector."""

from fuzzer_tool.core.critical_slowing import CriticalSlowingDown


class TestCriticalSlowingDown:
    def test_init(self):
        d = CriticalSlowingDown()
        assert d.window_size == 50
        assert d.rise_threshold == 1.5

    def test_no_detection_with_few_observations(self):
        d = CriticalSlowingDown(min_observations=10)
        for _ in range(5):
            d.observe(1.0)
        detected, reason = d.is_approaching_transition()
        assert not detected
        assert "need" in reason

    def test_baseline_established(self):
        d = CriticalSlowingDown(min_observations=5)
        for _ in range(10):
            d.observe(1.0)
        detected, reason = d.is_approaching_transition()
        assert not detected
        assert "baseline" in reason

    def test_detects_rising_variance_and_autocorrelation(self):
        d = CriticalSlowingDown(min_observations=5, rise_threshold=1.5)
        for _ in range(10):
            d.observe(1.0)
        d.is_approaching_transition()

        for i in range(20):
            d.observe(1.0 + i * 0.5)
        detected, reason = d.is_approaching_transition()
        assert detected
        assert "variance" in reason

    def test_no_detection_with_stable_series(self):
        d = CriticalSlowingDown(min_observations=5, rise_threshold=1.5)
        for _ in range(10):
            d.observe(1.0)
        d.is_approaching_transition()
        for _ in range(20):
            d.observe(1.0)
        detected, _ = d.is_approaching_transition()
        assert not detected

    def test_reset(self):
        d = CriticalSlowingDown(min_observations=5)
        for _ in range(10):
            d.observe(1.0)
        d.reset()
        assert len(d._history) == 0
        assert d._variance_baseline is None

    def test_save_load(self):
        d = CriticalSlowingDown(window_size=20)
        for i in range(10):
            d.observe(float(i))
        d.is_approaching_transition()
        data = d.save()
        d2 = CriticalSlowingDown()
        d2.load(data)
        assert len(d2._history) == 10
        assert d2._variance_baseline == d._variance_baseline
