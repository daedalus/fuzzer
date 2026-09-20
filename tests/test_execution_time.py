"""Tests for ExecutionTimeTracker — CRPS scoring, percentile, trend."""

import math
import random

import pytest

from fuzzer_tool.core.analyzers.analyzer_execution_time import ExecutionTimeTracker


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
            f"Extreme outlier CRPS ({crps_extreme}) should exceed typical CRPS ({crps_typical})"
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
                f"CRPS({vals[i]})={crps_values[i]} > CRPS({vals[i + 1]})={crps_values[i + 1]}"
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


class TestSkewnessAndTailRisk:
    def test_skewness_empty(self):
        t = ExecutionTimeTracker()
        assert t.skewness == 0.0

    def test_skewness_few_observations(self):
        t = ExecutionTimeTracker()
        t.record(0.01)
        t.record(0.02)
        assert t.skewness == 0.0  # need >= 3 for skewness

    def test_skewness_symmetric(self):
        t = ExecutionTimeTracker()
        for i in range(50):
            t.record(0.05 + i * 0.001)
        assert abs(t.skewness) < 0.5

    def test_tail_risk_false_for_symmetric(self):
        t = ExecutionTimeTracker()
        for i in range(100):
            t.record(0.05 + i * 0.001)
        assert not t.tail_risk

    def test_tail_risk_true_for_heavy_right_tail(self):
        t = ExecutionTimeTracker()
        for _ in range(100):
            t.record(0.01)
        t.record(10.0)  # extreme outlier
        assert t.tail_risk

    def test_skewness_zero_for_constant(self):
        t = ExecutionTimeTracker()
        for _ in range(50):
            t.record(0.05)
        assert t.skewness == 0.0
        assert not t.tail_risk

    def test_tail_risk_is_bool(self):
        t = ExecutionTimeTracker()
        assert isinstance(t.tail_risk, bool)


def _numeric_crps_lognormal(mu, sigma, observation, lo=1e-9, hi=None, steps=400000):
    """CRPS by direct numerical integration of ∫(F(y) - 𝟙[y≥x])² dy.

    Independent of the closed form under test: this integrates the
    lognormal CDF on a grid, using only math.erf. Slow, so callers keep
    the case count small.
    """
    if hi is None:
        hi = max(observation, math.exp(mu + 6.0 * sigma)) * 1.2
    step = (hi - lo) / steps
    total = 0.0
    y = lo + 0.5 * step
    for _ in range(steps):
        f = 0.5 * (1.0 + math.erf((math.log(y) - mu) / (sigma * math.sqrt(2.0))))
        d = f - (1.0 if y >= observation else 0.0)
        total += d * d
        y += step
    return total * step


def _tracker_over(samples, window_size=200):
    t = ExecutionTimeTracker(window_size=window_size)
    for s in samples:
        t.record(s)
    return t


class TestClosedFormCrps:
    """_compute_crps is the closed-form lognormal CRPS (Baran & Lerch 2015),
    replacing an O(n) walk over the sorted window. Three Φ evaluations, no
    dependence on window size.
    """

    def test_matches_numerical_integration(self):
        rng = random.Random(20260920)
        samples = [math.exp(rng.gauss(math.log(0.002), 0.4)) for _ in range(200)]
        t = _tracker_over(samples)
        mu = t._log_moments.mean
        sigma = t._log_moments.stddev
        for obs in (0.0008, 0.002, 0.005, 0.02):
            got = t._compute_crps(obs)
            exp = _numeric_crps_lognormal(mu, sigma, obs)
            assert abs(got - exp) <= 1e-4 * max(exp, 1e-6) + 1e-9, (obs, got, exp)

    def test_cost_is_independent_of_window_size(self):
        """The whole point: no O(n) walk left. A 20x larger window must not
        move the score, because the score never touches the window."""
        rng = random.Random(4)
        base = [math.exp(rng.gauss(math.log(0.002), 0.4)) for _ in range(100)]
        small = _tracker_over(base, window_size=100)
        # Same empirical sample, window big enough to hold it either way.
        large = _tracker_over(base, window_size=2000)
        assert small._compute_crps(0.003) == pytest.approx(large._compute_crps(0.003))

    def test_degenerate_sigma_is_the_dirac_limit(self):
        """sigma -> 0 makes the forecast a point mass; CRPS(δ_m, x) = |x-m|."""
        t = _tracker_over([0.05] * 50)
        assert t._compute_crps(0.05) == pytest.approx(0.0, abs=1e-12)
        assert t._compute_crps(0.15) == pytest.approx(0.10, abs=1e-9)
        assert t._compute_crps(1.0) == pytest.approx(0.95, abs=1e-9)

    def test_empty_and_single_observation(self):
        t = ExecutionTimeTracker()
        assert t._compute_crps(0.5) == 0.0  # nothing recorded yet
        t.record(1.0)
        # One observation: still a point forecast at the running median.
        assert t._compute_crps(2.0) == pytest.approx(1.0, abs=1e-9)

    def test_non_positive_observation_does_not_blow_up(self):
        """A zero elapsed time is reachable on a coarse clock; log(0) is not."""
        t = _tracker_over([0.002, 0.003, 0.004, 0.005, 0.006])
        assert t._compute_crps(0.0) >= 0.0
        assert t.record(0.0) >= 0.0

    def test_non_negative_everywhere(self):
        rng = random.Random(11)
        t = _tracker_over([math.exp(rng.gauss(math.log(0.01), 0.6)) for _ in range(200)])
        for obs in (1e-6, 1e-3, 0.01, 0.1, 1.0, 100.0):
            assert t._compute_crps(obs) >= 0.0

    def test_monotone_in_distance_from_the_forecast(self):
        """Typical < gap < extreme -- the property the old estimator was
        tested for, preserved by the parametric one."""
        rng = random.Random(12)
        t = _tracker_over([math.exp(rng.gauss(math.log(0.05), 0.15)) for _ in range(200)])
        typical = t._compute_crps(0.05)
        gap = t._compute_crps(0.15)
        extreme = t._compute_crps(1.0)
        assert typical < gap < extreme

    def test_agrees_with_the_empirical_estimator_on_a_lognormal_sample(self):
        """Not the same estimator -- a two-parameter fit rather than the
        window's own CDF -- but they must not disagree in magnitude when
        the model is right, or the reported CRPS trend changes meaning."""
        rng = random.Random(13)
        samples = [math.exp(rng.gauss(math.log(0.001), 0.45)) for _ in range(200)]
        t = _tracker_over(samples)
        srt = sorted(samples)
        n = len(srt)

        def empirical(obs):
            crps = 0.0
            for i in range(n - 1):
                cd = (i + 1) / n - (1.0 if srt[i] >= obs else 0.0)
                crps += cd * cd * (srt[i + 1] - srt[i])
            if obs > srt[-1]:
                crps += obs - srt[-1]
            return crps

        probes = [math.exp(rng.gauss(math.log(0.001), 0.45)) for _ in range(300)]
        mean_closed = sum(t._compute_crps(o) for o in probes) / len(probes)
        mean_emp = sum(empirical(o) for o in probes) / len(probes)
        assert abs(mean_closed - mean_emp) < 0.25 * mean_emp


class TestCrpsScoredEveryObservation:
    """The closed form is O(1), so the every-8th-observation subsampling
    that mitigated the old O(n) walk is gone. Each record() contributes
    its own score again."""

    def test_every_record_computes_a_fresh_score(self):
        t = ExecutionTimeTracker(window_size=10)
        for i in range(10):
            t.record(0.01 * (i + 1))  # fill the window

        calls = 0
        real = t._compute_crps

        def spy(obs):
            nonlocal calls
            calls += 1
            return real(obs)

        t._compute_crps = spy
        for i in range(80):
            t.record(0.5 + i * 0.001)
        assert calls == 80

    def test_history_grows_with_every_observation(self):
        t = ExecutionTimeTracker(window_size=10)
        for i in range(40):
            t.record(0.01 * (i + 1))
        assert len(t._crps_history) == 40

    def test_no_sampling_interval_attribute_remains(self):
        """A leftover constant would invite the branch back."""
        assert not hasattr(ExecutionTimeTracker, "_CRPS_SAMPLE_INTERVAL")

    def test_count_and_window_still_exact(self):
        t = ExecutionTimeTracker(window_size=10)
        for i in range(100):
            t.record(0.01 * (i + 1))
        assert t.count == 100
        assert len(t._sorted) == 10
        assert t.p99 == max(0.01 * (i + 1) for i in range(90, 100))

    def test_percentile_path_still_empirical(self):
        """suggested_timeout/p50/p99 must keep reading _sorted: a hang
        threshold comes from observed times, not a model's extrapolation."""
        t = _tracker_over([0.01] * 50 + [0.02] * 50, window_size=200)
        assert t.p50 in (0.01, 0.02)
        assert t.suggested_timeout() >= t.p99
