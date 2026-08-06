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

import numpy as np
import pytest

from fuzzer_tool.core.execution_time import ExecutionTimeTracker
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


class TestCrpsRampCache:
    def test_cached_ramp_matches_freshly_computed(self):
        t = ExecutionTimeTracker(window_size=50)
        for v in (0.01, 0.02, 0.03, 0.04):
            t.record(v)
        # record() scores the observation against the CDF *before* appending
        # it, so the cache is keyed on the pre-append lengths (n-1 max).
        assert t._crps_ramp_cache, "ramp cache was never populated"
        for n, ramp in t._crps_ramp_cache.items():
            np.testing.assert_allclose(ramp, np.arange(1, n + 1) / n)

    def test_cache_is_keyed_by_length(self):
        """Different window fill levels must not share one ramp."""
        t = ExecutionTimeTracker(window_size=50)
        for i in range(10):
            t.record(0.001 * (i + 1))
        for n, ramp in t._crps_ramp_cache.items():
            assert len(ramp) == n
            np.testing.assert_allclose(ramp, np.arange(1, n + 1) / n)

    def test_crps_matches_uncached_reference(self):
        """The cached implementation must equal a direct recomputation."""
        t = ExecutionTimeTracker(window_size=64)
        rng = np.random.default_rng(7)
        for v in rng.uniform(0.001, 0.05, size=40):
            t.record(float(v))

        obs = 0.02
        arr = np.asarray(t._sorted, dtype=np.float64)
        n = len(arr)
        cd = np.arange(1, n + 1) / n - (arr >= obs)
        expected = float(np.sum(cd[:-1] * cd[:-1] * np.diff(arr)))
        if obs > arr[-1]:
            expected += obs - arr[-1]

        assert t._compute_crps(obs) == pytest.approx(expected, abs=1e-12)

    def test_crps_empty_history_is_zero(self):
        t = ExecutionTimeTracker(window_size=10)
        assert t._compute_crps(0.5) == 0.0
