"""CrashMITracker.all_mi ran the per-position Python mi() loop on every rebuild.

record() invalidates the cache each exec, so the position arena's crash_mi
arm rebuilt it about once per round (6.4 s per 3k --hail-mary execs). The
whole profile is now one numpy pass; values match the scalar mi() to float
summation order.
"""

import math

from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker
from fuzzer_tool.core.rand_pool import RandPool

TOL = 1e-12


def _trained(seed: int, execs: int) -> CrashMITracker:
    rng = RandPool(seed)
    t = CrashMITracker(min_observations=3)
    for _ in range(execs):
        data = rng.randbytes(rng.randint(4, 60))
        t.record(data, is_crash=rng.random() < 0.2 and data[0] < 128)
    return t


def test_regression_crash_mi_vectorized():
    for seed in range(8):
        t = _trained(seed, 300)
        got = t.all_mi()
        assert got, "need observed positions"
        assert set(got) == {p for p, c in t.position_counts.items() if c >= t.min_observations}
        for pos, val in got.items():
            assert math.isclose(val, t.mi(pos), rel_tol=TOL, abs_tol=TOL)


def test_no_crash_is_zero_everywhere():
    """Falsification: no crash, no information."""
    t = CrashMITracker(min_observations=1)
    for i in range(20):
        t.record(bytes([i, i + 1]), is_crash=False)
    assert set(t.all_mi().values()) == {0.0}


def test_cache_still_invalidates_on_record():
    """Adversarial: a crash recorded after a build changes the profile."""
    t = _trained(1, 100)
    before = dict(t.all_mi())
    t.record(b"\x00" * 30, is_crash=True)
    assert t.all_mi() != before
