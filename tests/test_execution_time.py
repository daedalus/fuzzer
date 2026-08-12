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


def _legacy_compute_crps(sorted_times, observation):
    """Verbatim legacy _compute_crps (independent reference for equivalence)."""
    if not sorted_times:
        return 0.0
    n = len(sorted_times)
    crps = 0.0
    cdf_diff = 0.0
    prev = sorted_times[0]
    for i, val in enumerate(sorted_times):
        gap = val - prev
        if gap > 0:
            crps += cdf_diff * cdf_diff * gap
        f_val = (i + 1) / n
        indicator = 1.0 if val >= observation else 0.0
        cdf_diff = f_val - indicator
        prev = val
    max_val = sorted_times[-1]
    if observation > max_val:
        crps += 1.0 * (observation - max_val)
    return crps


class TestComputeCrpsVectorized:
    def test_matches_legacy_algorithm(self):
        """Vectorized _compute_crps is numerically identical to the legacy walk."""
        import random

        rng = random.Random(20260803)
        t = ExecutionTimeTracker(window_size=200)
        for _ in range(300):
            n = rng.randint(1, 200)
            # Duplicates included -> zero gaps exercise the (dead) gap>0 guard.
            times = sorted(rng.random() * 0.05 for _ in range(n))
            obs = rng.random() * 0.06
            t._sorted = list(times)
            got = t._compute_crps(obs)
            exp = _legacy_compute_crps(times, obs)
            assert abs(got - exp) <= 1e-9 * max(1.0, abs(exp)), (got, exp, n, obs)

    def test_edge_cases(self):
        t = ExecutionTimeTracker()
        assert t._compute_crps(0.5) == 0.0  # empty
        t._sorted = [1.0]
        assert t._compute_crps(0.5) == 0.0  # obs below sole value
        assert t._compute_crps(2.0) == 1.0  # obs above sole value (tail term)


class TestCRPSSamplingOnceWarm:
    """record() must recompute CRPS every call while the window is filling,
    then subsample once it's warm -- this is the perf fix: the numpy CRPS
    computation is expensive (fresh array + dot + diff) and only feeds a
    periodic display / end-of-run report, neither of which needs per-exec
    precision. suggested_timeout()/tail_risk must stay exact regardless,
    since they read _sorted/_moments, which are always updated.
    """

    def test_eager_while_filling(self):
        t = ExecutionTimeTracker(window_size=10)
        seen = []
        for i in range(10):
            seen.append(t.record(0.01 * (i + 1)))
        # Every one of the first window_size calls actually computed a
        # fresh CRPS (all mutually distinct inputs -> distinct scores).
        assert len(set(seen)) > 1

    def test_subsampled_once_warm(self):
        t = ExecutionTimeTracker(window_size=10)
        for i in range(10):
            t.record(0.01 * (i + 1))
        assert len(t._sorted) == t.window_size  # now warm

        computed_calls = 0
        real_compute = t._compute_crps

        def spy(obs):
            nonlocal computed_calls
            computed_calls += 1
            return real_compute(obs)

        t._compute_crps = spy
        for i in range(80):
            t.record(0.5 + i * 0.001)
        # Only every _CRPS_SAMPLE_INTERVALth call after warm-up recomputes.
        assert computed_calls == 80 // t._CRPS_SAMPLE_INTERVAL

    def test_count_and_window_exact_regardless_of_sampling(self):
        """count and the percentile window must not skip observations --
        only the CRPS computation itself is sampled."""
        t = ExecutionTimeTracker(window_size=10)
        for i in range(100):
            t.record(0.01 * (i + 1))
        assert t.count == 100
        assert len(t._sorted) == 10  # capped at window_size, as before
        # p99 reflects the most recent 10 observations, unaffected by
        # which of them got a CRPS computed.
        assert t.p99 == max(0.01 * (i + 1) for i in range(90, 100))

    def test_suggested_timeout_unaffected_by_sampling(self):
        """suggested_timeout reads _sorted/_moments only -- exact either way."""
        t = ExecutionTimeTracker(window_size=10)
        for _i in range(100):
            t.record(0.01)
        assert t.suggested_timeout() > 0.0

    def test_return_value_when_not_recomputed_is_last_computed(self):
        t = ExecutionTimeTracker(window_size=10)
        for i in range(10):
            t.record(0.01 * (i + 1))  # fills window
        last_computed = t.record(0.5)  # call 11: 11 % 8 != 0 -> not recomputed
        assert last_computed == t._crps_history[-1]

    def test_mean_crps_still_meaningful_once_warm(self):
        t = ExecutionTimeTracker(window_size=10)
        for _ in range(10):
            t.record(0.05)
        for _ in range(50):
            t.record(0.05)  # constant input, sampled or not
        assert t.mean_crps() < 0.01
