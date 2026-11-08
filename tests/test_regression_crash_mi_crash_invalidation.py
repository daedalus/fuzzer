"""Regression: every crash rebuilt CrashMITracker's MI sampler.

``record`` cleared ``_cache_valid`` on each crash, so the next
``weighted_position`` recomputed MI over every observed position: 7.7% of a
fuzzgoat run at a 7% crash rate. Now a crash invalidates only at power-of-two
crash counts; the 50-exec cadence bounds staleness after that.
"""

from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker
from fuzzer_tool.core.cadence import phase_of

_PERIOD = 50
_SITE = "crash_eta.invalidate"


def _cadence_free_execs(n: int) -> bool:
    # Exec counts 1..n never hit the 50-exec invalidation for this site.
    return all((k + phase_of(_SITE, _PERIOD)) % _PERIOD for k in range(1, n + 1))


def _invalidated_on(crashes: int) -> list[int]:
    """Crash counts at which a crash cleared the sampler cache."""
    assert _cadence_free_execs(crashes)
    t = CrashMITracker(min_observations=1)
    hits = []
    for k in range(1, crashes + 1):
        t._cache_valid = True
        t.record(bytes([k & 0xFF]) * 4, is_crash=True)
        if not t._cache_valid:
            hits.append(k)
    return hits


def test_regression_crash_invalidates_at_powers_of_two():
    n = 16
    expected = [k for k in range(1, n + 1) if k & (k - 1) == 0]
    assert _invalidated_on(n) == expected


def test_first_crash_invalidates():
    """Falsification: the first crash turns MI on; the sampler must see it."""
    assert _invalidated_on(1) == [1]


def test_clean_execs_never_invalidate_off_cadence():
    """Adversarial: non-crash records only refresh on the 50-exec cadence."""
    n = 10
    assert _cadence_free_execs(n)
    t = CrashMITracker(min_observations=1)
    for k in range(n):
        t._cache_valid = True
        t.record(bytes([k]) * 4, is_crash=False)
        assert t._cache_valid
