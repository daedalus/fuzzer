"""Regression tests for CrashMITracker.weighted_position sampling.

The draw was scaled by the weight of every tracked position while the
scan only accumulated the positions below the input length, so any input
shorter than the tracked range degenerated to "always return the highest
eligible position".
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from fuzzer_tool.core.crash_eta import CrashMITracker


def _tracker_with_positions(n_positions: int, input_len_for_mi: int = 200):
    """A tracker with MI spread over *n_positions* byte positions."""
    tracker = CrashMITracker(min_observations=1)
    rnd = random.Random(0)
    for _ in range(400):
        data = bytes(rnd.randrange(256) for _ in range(n_positions))
        tracker.record(data, is_crash=(data[0] & 1) == 0)
    tracker.all_mi()
    return tracker


def test_draw_is_normalised_by_the_eligible_prefix_not_the_total():
    """No single position may absorb the mass that fell off the old loop.

    Before the fix, with 200 tracked positions and input_length=50, the
    last eligible position came back on 75.0% of draws against a correct
    share of 2.1%.
    """
    tracker = _tracker_with_positions(200)
    eligible = [p for p in tracker._cached_positions if p < 50]
    assert len(eligible) > 10, "test degenerate: too few eligible positions"

    weights = {p: max(tracker._mi_cache[p], 0.01) for p in eligible}
    total = sum(weights.values())
    assert total < tracker._cached_total * 0.9, (
        "test degenerate: the eligible prefix is nearly the whole mass, "
        "so the old bug would not show"
    )

    draws = 40000
    counts = Counter(tracker.weighted_position(50) for _ in range(draws))

    assert set(counts) <= set(eligible), "sampled an ineligible position"

    last = eligible[-1]
    observed = counts[last] / draws
    expected = weights[last] / total
    assert observed == pytest.approx(expected, abs=0.03), (
        f"position {last} drawn {observed:.3f} of the time, expected {expected:.3f}"
    )


def test_full_distribution_matches_the_prefix_weights():
    tracker = _tracker_with_positions(200)
    eligible = [p for p in tracker._cached_positions if p < 60]
    weights = {p: max(tracker._mi_cache[p], 0.01) for p in eligible}
    total = sum(weights.values())

    draws = 60000
    counts = Counter(tracker.weighted_position(60) for _ in range(draws))
    for p in eligible:
        assert counts[p] / draws == pytest.approx(weights[p] / total, abs=0.02), (
            f"position {p} off distribution"
        )


def test_unrestricted_draw_covers_every_position():
    """With input_length past the end, every tracked position is eligible."""
    tracker = _tracker_with_positions(60)
    counts = Counter(tracker.weighted_position(10_000) for _ in range(20000))
    assert set(counts) == set(tracker._cached_positions)


def test_returns_none_when_no_position_is_eligible():
    tracker = _tracker_with_positions(60)
    assert min(tracker._cached_positions) == 0
    assert tracker.weighted_position(0) is None


def test_returns_none_with_no_data():
    assert CrashMITracker().weighted_position(100) is None


def test_never_returns_a_position_at_or_past_the_input_length():
    tracker = _tracker_with_positions(200)
    for length in (1, 5, 33, 128, 199, 200, 500):
        for _ in range(200):
            p = tracker.weighted_position(length)
            if p is not None:
                assert p < length
