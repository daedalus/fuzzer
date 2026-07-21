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

    def test_init_skew_rise_threshold(self):
        d = CriticalSlowingDown(skew_rise_threshold=2.0)
        assert d.skew_rise_threshold == 2.0

    def test_productive_tier_with_high_skewness(self):
        """When variance + autocorrelation + skewness all rise, the
        verdict should say 'productive'."""
        d = CriticalSlowingDown(min_observations=5, rise_threshold=1.5, skew_rise_threshold=1.5)
        # Establish baseline with some skewness: mostly low values, occasional spike
        for i in range(10):
            d.observe(1.0 if i < 9 else 5.0)  # skewed baseline
        d.is_approaching_transition()

        # Now: much higher variance + autocorrelation + even more skewness
        for i in range(20):
            val = 1.0 + i * 0.5 if i < 19 else 50.0  # heavy right tail
            d.observe(val)
        detected, reason = d.is_approaching_transition()
        assert detected
        assert "productive" in reason

    def test_fallback_tier_without_skewness(self):
        """When variance + autocorrelation rise but skewness stays flat,
        the verdict should NOT say 'productive'."""
        d = CriticalSlowingDown(
            min_observations=5,
            rise_threshold=1.5,
            skew_rise_threshold=10.0,  # very high threshold — won't trigger
        )
        for _ in range(10):
            d.observe(1.0)
        d.is_approaching_transition()

        for i in range(20):
            d.observe(1.0 + i * 0.5)
        detected, reason = d.is_approaching_transition()
        assert detected
        assert "productive" not in reason
        assert "approaching transition" in reason

    def test_save_load_skewness(self):
        d = CriticalSlowingDown(window_size=20)
        for i in range(10):
            d.observe(float(i))
        d.is_approaching_transition()
        data = d.save()
        assert "skewness_baseline" in data
        assert "skew_rise_threshold" in data
        d2 = CriticalSlowingDown()
        d2.load(data)
        assert d2._skewness_baseline == d._skewness_baseline
        assert d2.skew_rise_threshold == d.skew_rise_threshold

    def test_reset_clears_skewness(self):
        d = CriticalSlowingDown(min_observations=5)
        for _ in range(10):
            d.observe(1.0)
        d.is_approaching_transition()
        d.reset()
        assert d._skewness_baseline is None
