"""Crash MI from running sums: no per-rebuild pass over positions x 256.

With f(k) = k*log2(k) and, per position, byte counts t (all execs), c
(crashing) and d = t - c:

    MI_p = (sum f(c) + sum f(d) - sum f(t) + C_p log2(n/C) + D_p log2(n/(n-C))) / n

record() keeps the three sums of f and C_p up to date (one byte per
position per exec), so all_mi() is O(positions). Rebuilding the dense form
cost 18 ms per call, 3.6 s per 10k default-mode execs.
"""

import math

from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker
from fuzzer_tool.core.rand_pool import RandPool

TOL = 1e-9


def _trained(seed: int, execs: int, max_len: int) -> CrashMITracker:
    rng = RandPool(seed)
    t = CrashMITracker(min_observations=3)
    for _ in range(execs):
        data = rng.randbytes(rng.randint(1, max_len))
        t.record(data, is_crash=rng.random() < 0.15 and data[0] < 160)
    return t


def _check(t: CrashMITracker) -> None:
    got = t.all_mi()
    assert got
    for pos, val in got.items():
        assert math.isclose(val, t.mi(pos), rel_tol=TOL, abs_tol=TOL), pos


def test_regression_crash_mi_closed_form():
    for seed in range(6):
        _check(_trained(seed, 400, 30 + seed * 40))


def test_rows_grow_mid_campaign():
    """Adversarial: inputs that lengthen later add rows to the sums."""
    t = _trained(1, 100, 8)
    rng = RandPool(5)
    for _ in range(100):
        t.record(rng.randbytes(300), is_crash=rng.random() < 0.3)
    _check(t)


def test_save_load_rebuilds_sums():
    t = _trained(2, 300, 50)
    u = CrashMITracker()
    u.load(t.save())
    assert u.all_mi().keys() == t.all_mi().keys()
    for pos, val in u.all_mi().items():
        assert math.isclose(val, t.mi(pos), rel_tol=TOL, abs_tol=TOL)


def test_no_crash_or_all_crash_is_zero():
    """Falsification: one outcome class carries no information."""
    for crash in (False, True):
        t = CrashMITracker(min_observations=1)
        for i in range(30):
            t.record(bytes([i, 255 - i]), is_crash=crash)
        assert set(t.all_mi().values()) == {0.0}
