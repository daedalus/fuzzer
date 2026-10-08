"""Regression: ``join_streams`` must terminate on degenerate gaps/timestamps.

Pre-fix, the mismatch branch only advanced pointers with
``ts < min_ts + max_gap``: with ``max_gap <= 0``, NaN, or +inf on every
stream nothing advanced and the loop spun forever. Timestamps can come
from a corrupted state file, so the join must not trust them.
"""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.temporal_join import join_streams


@pytest.mark.timeout(5)
def test_regression_zero_gap_matches_exact_only() -> None:
    """max_gap=0 joins only identical timestamps and still terminates."""
    streams = [
        [(0.0, "a0"), (1.0, "a1"), (3.0, "a3")],
        [(0.5, "b0"), (1.0, "b1"), (2.0, "b2"), (3.0, "b3")],
    ]

    assert join_streams(streams, 0.0) == [("a1", "b1"), ("a3", "b3")]


@pytest.mark.timeout(5)
def test_regression_non_finite_ts_skipped() -> None:
    """NaN and +/-inf entries are skipped; finite neighbours still pair."""
    nan, inf = math.nan, math.inf
    streams = [
        [(nan, "an"), (1.0, "a1"), (inf, "ai")],
        [(-inf, "bn"), (1.2, "b1"), (inf, "bi")],
    ]

    assert join_streams(streams, 0.5) == [("a1", "b1")]


@pytest.mark.timeout(5)
def test_regression_negative_gap_rejected() -> None:
    """A negative or NaN tolerance is a caller bug, not an empty join."""
    streams = [[(0.0, "a")], [(0.0, "b")]]

    with pytest.raises(ValueError):
        join_streams(streams, -1.0)
    with pytest.raises(ValueError):
        join_streams(streams, math.nan)
