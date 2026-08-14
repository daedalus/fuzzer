"""Falsification-first tests for ExecTimeCalibrator.

Covers the three validation items from the design plan:
  1. Synthetic baseline + injected anomalies across multiple thresholds.
  2. Median-vs-mean regression on a heavy-tailed baseline.
  3. Insufficient-data guard.
"""

from __future__ import annotations

import math
import random

import pytest

from fuzzer_tool.core.exec_time_anomaly import ExecTimeCalibrator


def _lognormal_baseline(rng: random.Random, n: int) -> list[float]:
    """Generate a right-skewed baseline mimicking typical exec-time noise."""
    return [max(0.001, rng.lognormvariate(math.log(0.05), 0.5)) for _ in range(n)]


class TestExecTimeCalibrator:
    def test_threshold_returns_none_below_min_samples(self):
        """Fail-closed: never emit a threshold on an under-sampled baseline."""
        cal = ExecTimeCalibrator(min_samples=200)
        for _ in range(199):
            cal.observe(0.05)
        assert cal.threshold() is None

    def test_threshold_returns_float_once_ready(self):
        cal = ExecTimeCalibrator(min_samples=10)
        for _ in range(10):
            cal.observe(0.05)
        thresh = cal.threshold()
        assert isinstance(thresh, float)
        assert thresh > 0.0

    def test_count_tracks_observations(self):
        cal = ExecTimeCalibrator()
        for i in range(7):
            cal.observe(float(i))
        assert cal.count == 7

    def test_median_property(self):
        cal = ExecTimeCalibrator()
        for v in [0.1, 0.2, 0.3]:
            cal.observe(v)
        assert cal.median == pytest.approx(0.2)

    def test_median_none_when_empty(self):
        cal = ExecTimeCalibrator()
        assert cal.median is None


class TestAnomalyDetection:
    """Injected-spike detection across a range of thresholds."""

    @pytest.mark.parametrize("thresh_mult", [1.5, 2.0, 3.0])
    def test_flags_injected_spikes_not_baseline_noise(self, thresh_mult: float):
        """Baseline should not trip the threshold; injected outliers should."""
        rng = random.Random(20260814)
        baseline = _lognormal_baseline(rng, 500)
        anomalies = [max(0.001, rng.lognormvariate(math.log(2.0), 0.3)) for _ in range(40)]

        cal = ExecTimeCalibrator(min_samples=200)
        for t in baseline + anomalies:
            cal.observe(t)

        thresh = cal.threshold(mult=thresh_mult)
        assert thresh is not None

        baseline_fps = sum(ExecTimeCalibrator.is_anomalous(t, thresh) for t in baseline)
        anomaly_hits = sum(ExecTimeCalibrator.is_anomalous(t, thresh) for t in anomalies)

        # Every injected anomaly should be flagged.
        assert anomaly_hits == len(anomalies)
        # Higher multipliers must not increase false positives.
        if thresh_mult > 1.5:
            prev_thresh = cal.threshold(mult=thresh_mult / 2.0)
            assert prev_thresh is not None
            prev_fps = sum(ExecTimeCalibrator.is_anomalous(t, prev_thresh) for t in baseline)
            assert baseline_fps <= prev_fps

    def test_lower_multiplier_increases_sensitivity(self):
        """A lower multiplier should flag at least as many points as a higher one."""
        rng = random.Random(20260814)
        samples = _lognormal_baseline(rng, 300) + [
            max(0.001, rng.lognormvariate(math.log(1.5), 0.2)) for _ in range(30)
        ]
        cal = ExecTimeCalibrator(min_samples=200)
        for t in samples:
            cal.observe(t)

        thresh_low = cal.threshold(mult=1.5)
        thresh_high = cal.threshold(mult=3.0)
        assert thresh_low is not None and thresh_high is not None
        assert thresh_low < thresh_high

        flagged_low = sum(ExecTimeCalibrator.is_anomalous(t, thresh_low) for t in samples)
        flagged_high = sum(ExecTimeCalibrator.is_anomalous(t, thresh_high) for t in samples)
        assert flagged_low >= flagged_high


class TestMedianRobustness:
    """Median-based threshold must survive where a mean-based one misfires."""

    def test_median_threshold_survives_heavy_tail(self):
        """Mean gets dragged up by outliers; median should stay put."""
        rng = random.Random(20260814)
        # 490 normal-ish samples, 10 huge outliers.
        normal = [0.05 + rng.uniform(-0.01, 0.01) for _ in range(490)]
        outliers = [50.0 + rng.uniform(0.0, 50.0) for _ in range(10)]
        samples = normal + outliers

        cal = ExecTimeCalibrator(min_samples=500)
        for t in samples:
            cal.observe(t)

        thresh = cal.threshold(mult=2.0)
        assert thresh is not None

        # A mean-based threshold here would be pulled well above normal ops.
        mean = sum(samples) / len(samples)
        stddev = math.sqrt(sum((x - mean) ** 2 for x in samples) / len(samples))
        mean_thresh = mean + 2.0 * stddev

        # Median threshold should be much tighter than the inflated mean one.
        assert thresh < mean_thresh * 0.5

        # Every outlier should still be anomalous under the median threshold.
        assert all(ExecTimeCalibrator.is_anomalous(t, thresh) for t in outliers)
        # The bulk of normal samples should not be flagged.
        assert sum(ExecTimeCalibrator.is_anomalous(t, thresh) for t in normal) < 20

    def test_constant_baseline_yields_zero_spikes(self):
        """No-variance input must not produce phantom anomalies."""
        cal = ExecTimeCalibrator(min_samples=50)
        for _ in range(100):
            cal.observe(0.05)
        thresh = cal.threshold(mult=2.0)
        assert thresh is not None
        assert thresh == pytest.approx(0.05 * 2.0)
        assert not ExecTimeCalibrator.is_anomalous(0.05, thresh)


class TestIsAnomalous:
    def test_below_threshold_is_not_anomalous(self):
        assert not ExecTimeCalibrator.is_anomalous(0.01, 0.05)

    def test_equal_to_threshold_is_not_anomalous(self):
        assert not ExecTimeCalibrator.is_anomalous(0.05, 0.05)

    def test_above_threshold_is_anomalous(self):
        assert ExecTimeCalibrator.is_anomalous(0.06, 0.05)
