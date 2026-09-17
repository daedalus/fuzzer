"""Regression: ExecutionTimeTracker's sorted window mirrors its deque.

``_times`` is a ``deque(maxlen=window_size)``, which drops its oldest value
inside ``append()``. ``record()`` read ``_times[0]`` after appending, so it
removed the second-oldest value from ``_sorted`` and kept the evicted one.
With window 5 and inputs ``100, 1..9`` the deque held ``[5..9]`` while
``_sorted`` still held ``100`` and ``suggested_timeout()`` returned ~102s.
"""

import random

import pytest

from fuzzer_tool.core.execution_time import ExecutionTimeTracker


def test_early_outlier_leaves_the_window():
    t = ExecutionTimeTracker(window_size=5)
    for v in [100, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        t.record(float(v))
    assert t._sorted == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert t.suggested_timeout() < 100.0


@pytest.mark.parametrize("window", [1, 2, 7, 50])
def test_sorted_matches_deque_with_duplicates(window):
    rng = random.Random(window)
    t = ExecutionTimeTracker(window_size=window)
    for i in range(1000):
        # Coarse values force duplicates, where bisect_left must still hit
        # an equal element rather than a neighbour.
        t.record(float(rng.randrange(10)) * 0.01)
        assert t._sorted == sorted(t._times), f"diverged at record {i}"


def test_control_matches_itself():
    """Hard Rule 46: the oracle (sorted deque) against a second tracker."""
    a, b = ExecutionTimeTracker(window_size=13), ExecutionTimeTracker(window_size=13)
    rng = random.Random(0)
    for _ in range(300):
        v = rng.random()
        a.record(v)
        b.record(v)
    assert sorted(a._times) == sorted(b._times)
