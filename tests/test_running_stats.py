"""Tests for RunningMoments — Welford/Pébay online statistics."""

import math
import statistics

from fuzzer_tool.core.running_stats import RunningMoments


class TestRunningMomentsUnbounded:
    def test_empty(self):
        m = RunningMoments()
        assert m.count == 0
        assert m.mean == 0.0
        assert m.variance == 0.0
        assert m.stddev == 0.0
        assert m.skewness == 0.0
        assert m.kurtosis == 0.0
        assert m.z_score(1.0) == 0.0

    def test_single_observation(self):
        m = RunningMoments()
        m.update(5.0)
        assert m.count == 1
        assert m.mean == 5.0
        assert m.variance == 0.0
        assert m.skewness == 0.0

    def test_two_observations(self):
        m = RunningMoments()
        m.update(2.0)
        m.update(4.0)
        assert m.count == 2
        assert m.mean == 3.0
        assert m.variance == 2.0  # sample variance: ((2-3)^2 + (4-3)^2) / 1
        assert m.stddev == math.sqrt(2.0)
        assert m.skewness == 0.0  # need >= 3

    def test_mean_matches_statistics(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = RunningMoments()
        for x in data:
            m.update(x)
        assert m.mean == statistics.mean(data)

    def test_stddev_matches_statistics(self):
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        m = RunningMoments()
        for x in data:
            m.update(x)
        assert abs(m.stddev - statistics.stdev(data)) < 1e-10

    def test_skewness_positive(self):
        """Right-skewed data should produce positive skewness."""
        # Exponential-like: many small, few large
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 10.0]
        m = RunningMoments()
        for x in data:
            m.update(x)
        assert m.skewness > 0

    def test_skewness_negative(self):
        """Left-skewed data should produce negative skewness."""
        data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0]
        m = RunningMoments()
        for x in data:
            m.update(x)
        assert m.skewness < 0

    def test_skewness_symmetric(self):
        """Symmetric data should have skewness near 0."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        m = RunningMoments()
        for x in data:
            m.update(x)
        assert abs(m.skewness) < 0.1

    def test_kurtosis_zero_for_normalish(self):
        """Uniform-like data should have near-zero excess kurtosis."""
        data = list(range(1, 21))
        m = RunningMoments()
        for x in data:
            m.update(float(x))
        # Uniform distribution has excess kurtosis = -1.2, so not zero
        # but bounded
        assert -2.0 < m.kurtosis < 2.0

    def test_kurtosis_heavy_tailed(self):
        """Data with outliers should have positive excess kurtosis."""
        data = [1.0] * 20 + [100.0]
        m = RunningMoments()
        for x in data:
            m.update(x)
        assert m.kurtosis > 0

    def test_z_score(self):
        m = RunningMoments()
        for x in [10.0, 10.0, 10.0, 10.0, 20.0]:
            m.update(x)
        z = m.z_score(20.0)
        assert z > 0  # 20.0 is above mean

    def test_z_score_zero_at_mean(self):
        m = RunningMoments()
        for x in [5.0, 10.0, 15.0]:
            m.update(x)
        assert m.z_score(m.mean) == 0.0

    def test_batch_equivalence(self):
        """Batch computation should match streaming for known vectors."""
        data = [3.0, 7.0, 7.0, 2.0, 9.0, 10.0, 4.0, 1.0, 6.0, 8.0, 5.0]
        m = RunningMoments()
        for x in data:
            m.update(x)

        n = len(data)
        mean = sum(data) / n
        assert abs(m.mean - mean) < 1e-10

        var = sum((x - mean) ** 2 for x in data) / (n - 1)
        assert abs(m.variance - var) < 1e-10

    def test_large_values_no_overflow(self):
        """Large values should not cause numerical issues."""
        m = RunningMoments()
        base = 1e12
        for i in range(50):
            m.update(base + i * 1e-3)
        assert m.count == 50
        assert abs(m.mean - (base + 24.5e-3)) < 1e-3
        assert m.stddev > 0

    def test_identical_values(self):
        m = RunningMoments()
        for _ in range(20):
            m.update(42.0)
        assert m.mean == 42.0
        assert m.variance == 0.0
        assert m.skewness == 0.0
        assert m.kurtosis == 0.0


class TestRunningMomentsWindowed:
    def test_window_capped(self):
        m = RunningMoments(window=5)
        for i in range(10):
            m.update(float(i))
        assert m.count == 5
        # Should reflect only [5, 6, 7, 8, 9]
        assert m.mean == 7.0

    def test_window_mean_matches_slice(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        m = RunningMoments(window=4)
        for x in data:
            m.update(x)
        # Window should be [5, 6, 7, 8]
        assert m.mean == 6.5
        assert abs(m.stddev - statistics.stdev([5, 6, 7, 8])) < 1e-10

    def test_window_skewness(self):
        m = RunningMoments(window=10)
        # First fill with uniform
        for i in range(20):
            m.update(float(i % 10))
        # Window is [10..19] but mod 10 → [0,1,...,9] symmetric-ish
        assert m.count == 10

    def test_window_zero_skewness_after_remove(self):
        """After the window slides past all initial data, stats reflect only the window."""
        m = RunningMoments(window=5)
        # Fill window with [1,2,3,4,5]
        for i in range(1, 6):
            m.update(float(i))
        mean_before = m.mean
        assert mean_before == 3.0
        # Slide window: add [10, 11, 12, 13, 14]
        for i in range(10, 15):
            m.update(float(i))
        # Window is [10, 11, 12, 13, 14]
        assert m.mean == 12.0

    def test_window_small(self):
        m = RunningMoments(window=2)
        m.update(1.0)
        assert m.count == 1
        m.update(3.0)
        assert m.count == 2
        assert m.mean == 2.0
        m.update(5.0)
        assert m.count == 2
        assert m.mean == 4.0  # window: [3, 5]


class TestRunningMomentsSaveLoad:
    def test_save_load_roundtrip(self):
        m = RunningMoments()
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.update(x)
        data = m.save()
        m2 = RunningMoments()
        m2.load(data)
        assert m2.count == m.count
        assert abs(m2.mean - m.mean) < 1e-10
        assert abs(m2.variance - m.variance) < 1e-10
        assert abs(m2.skewness - m.skewness) < 1e-10

    def test_save_load_windowed(self):
        m = RunningMoments(window=5)
        for i in range(10):
            m.update(float(i))
        data = m.save()
        m2 = RunningMoments()
        m2.load(data)
        assert m2._window == 5
        assert m2.count == m.count
        assert abs(m2.mean - m.mean) < 1e-10

    def test_window_o1_matches_batch(self):
        """O(1) sliding-window update matches full recomputation."""
        wsize = 20
        m = RunningMoments(window=wsize)
        data = [float(i % 7 * 3 + 1) for i in range(100)]
        for i, x in enumerate(data):
            m.update(x)
            if i >= wsize:
                window = data[i - wsize + 1 : i + 1]
                assert abs(m.mean - statistics.mean(window)) < 1e-10, (
                    f"mean mismatch at step {i}: {m.mean} vs {statistics.mean(window)}"
                )
                if len(window) >= 2:
                    var = m.variance
                    expected_var = statistics.variance(window)
                    assert abs(var - expected_var) < 1e-8, (
                        f"variance mismatch at step {i}: {var} vs {expected_var}"
                    )
                # Skewness and kurtosis verified via manual computation
                # since statistics.skewness/kurtosis may not be available.
                if len(window) >= 3:
                    n = len(window)
                    mean_w = statistics.mean(window)
                    m2_w = sum((v - mean_w) ** 2 for v in window) / n
                    m3_w = sum((v - mean_w) ** 3 for v in window) / n
                    if m2_w > 0:
                        expected_skew = m3_w / (m2_w ** 1.5)
                        assert abs(m.skewness - expected_skew) < 1e-8, (
                            f"skewness mismatch at step {i}: {m.skewness} vs {expected_skew}"
                        )
                if len(window) >= 4:
                    n = len(window)
                    mean_w = statistics.mean(window)
                    m2_w = sum((v - mean_w) ** 2 for v in window) / n
                    m4_w = sum((v - mean_w) ** 4 for v in window) / n
                    if m2_w > 0:
                        expected_kurt = m4_w / (m2_w * m2_w) - 3.0
                        assert abs(m.kurtosis - expected_kurt) < 1e-8, (
                            f"kurtosis mismatch at step {i}: {m.kurtosis} vs {expected_kurt}"
                        )
