"""Regression tests: hot-path optimization invariants.

Two optimizations trade a second representation (or a cache) for speed.
Both are only safe while the fast path stays equivalent to the slow one,
and neither equivalence was previously pinned:

  * RandPool keeps Python-list mirrors of its numpy pools because scalar
    element access on a numpy array is ~2x slower than on a list. Bulk
    methods still slice the numpy arrays. If the mirror ever drifts out of
    sync with the array — a refill that updates one but not the other —
    the scalar and bulk draws silently diverge, which would corrupt
    reproducibility from a fixed seed without raising anything.

  * ExecutionTimeTracker caches the arange(1,n+1)/n divisor used by the
    CRPS computation. A stale or mis-keyed cache would silently return a
    wrong score rather than fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.analyzers.analyzer_execution_time import ExecutionTimeTracker
from fuzzer_tool.core.rand_pool import RandPool


class TestRandPoolMirrorConsistency:
    def test_list_mirrors_match_numpy_pools_after_refill(self):
        pool = RandPool()
        pool._refill()
        assert pool._pool_l == pool._pool.tolist()
        assert pool._m256_l == pool._m256.tolist()

    def test_mirrors_stay_consistent_across_many_refills(self):
        pool = RandPool()
        for _ in range(5):
            pool._refill()
            assert pool._pool_l == pool._pool.tolist(), "uint32 mirror drifted"
            assert pool._m256_l == pool._m256.tolist(), "uint8 mirror drifted"

    def test_m256_mirror_is_pool_mod_256(self):
        """The %256 fast path is only valid if the mirror really is the
        modulus of the pool."""
        pool = RandPool()
        pool._refill()
        expected = [v % 256 for v in pool._pool_l]
        assert pool._m256_l == expected

    def test_scalar_draws_come_from_the_mirror_in_order(self):
        """randint must consume the same values, in the same order, that a
        direct read of the mirror would give."""
        pool = RandPool()
        pool._refill()
        start = pool._idx
        expected = [pool._pool_l[start + i] % 100 for i in range(20)]
        got = [pool.randint(0, 99) for _ in range(20)]
        assert got == expected

    def test_width_256_fast_path_matches_mirror(self):
        pool = RandPool()
        pool._refill()
        start = pool._idx
        expected = [pool._m256_l[start + i] for i in range(20)]
        got = [pool.randint(0, 255) for _ in range(20)]
        assert got == expected

    def test_choice_and_randint_share_the_pool_cursor(self):
        """Mixing scalar methods must not double-read or skip entries."""
        pool = RandPool()
        pool._refill()
        start = pool._idx
        pool.randint(0, 9)
        pool.choice([1, 2, 3])
        pool.randrange(7)
        assert pool._idx == start + 3

    def test_refill_happens_before_reading_past_the_end(self):
        """Exhausting the pool must refill rather than IndexError."""
        pool = RandPool()
        pool._refill()
        pool._idx = len(pool._pool_l) - 1
        pool.randint(0, 255)  # consumes the last entry
        pool.randint(0, 255)  # must trigger a refill
        assert pool._idx <= len(pool._pool_l)

    def test_bulk_and_scalar_paths_both_stay_in_range(self):
        pool = RandPool()
        for _ in range(200):
            assert 5 <= pool.randint(5, 9) <= 9
        assert all(5 <= v <= 9 for v in pool.randint_list(5, 9, 500))


class TestCrpsClosedForm:
    """The lognormal closed form must equal the integral it claims to solve.

    These replace three tests that pinned ``_crps_ramp_cache`` and an
    empirical-CDF recomputation. That estimator was deliberately retired
    for a two-parameter lognormal fit (see ``_compute_crps``'s docstring),
    so the old tests were asserting against an implementation that no
    longer exists -- two raised AttributeError and the third compared the
    parametric answer to the empirical one and called the difference a bug.

    Checking the formula against a numeric quadrature of its own defining
    integral is what the old "matches uncached reference" test was for, and
    it survives the estimator change instead of being invalidated by it.
    """

    @staticmethod
    def _numeric_crps(mu: float, sigma: float, obs: float) -> float:
        """Direct quadrature of integral (F(y) - 1[y >= obs])^2 dy."""
        ys = np.linspace(1e-9, 5.0, 2_000_001)
        phi = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in (np.log(ys) - mu) / sigma])
        return float(np.trapezoid((phi - (ys >= obs)) ** 2, ys))

    def test_closed_form_matches_numeric_integration(self):
        t = ExecutionTimeTracker(window_size=64)
        rng = np.random.default_rng(7)
        for v in rng.lognormal(math.log(0.02), 0.4, size=40):
            t.record(float(v))

        expected = self._numeric_crps(t._log_moments.mean, t._log_moments.stddev, 0.02)
        # Tolerance is set by the quadrature grid, not by the formula: the
        # residual shrinks with the step size (1.2e-3 at 4e5 points, 2.4e-4
        # at 2e6), which is what a correct closed form looks like against a
        # discretised integral.
        assert t._compute_crps(0.02) == pytest.approx(expected, rel=1e-3)

    def test_degenerate_window_is_the_point_forecast(self):
        """sigma -> 0 must give |x - m|, not a division by zero."""
        t = ExecutionTimeTracker(window_size=50)
        for _ in range(8):
            t.record(0.01)
        assert t._compute_crps(0.03) == pytest.approx(0.02, abs=1e-12)

    def test_single_observation_is_the_point_forecast(self):
        t = ExecutionTimeTracker(window_size=50)
        t.record(0.01)
        assert t._compute_crps(0.04) == pytest.approx(0.03, abs=1e-12)

    def test_crps_empty_history_is_zero(self):
        t = ExecutionTimeTracker(window_size=10)
        assert t._compute_crps(0.5) == 0.0
